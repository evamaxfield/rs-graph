#!/usr/bin/env python

import logging

import pandas as pd
import typer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import (
    DATA_FILES_DIR,
    MATCHED_DEV_AUTHOR_IDS_PATH,
    load_author_contributions_dataset,
    load_basic_repos_dataset,
    load_repo_contributors_dataset,
)
from rs_graph.data.enrichment import github, semantic_scholar
from rs_graph.modeling import (
    DEFAULT_DEV_AUTHOR_EMBEDDING_MODEL_NAME,
    load_base_dev_author_em_model,
    predict_dev_author_matches,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


@app.command()
def get_upstreams(debug: bool = False) -> None:
    """Get upstream dependencies from repos."""
    # Setup logger
    setup_logger(debug=debug)

    # Read repos dataset
    rs_graph_repos = load_basic_repos_dataset()

    # Get upstream deps
    upstream_deps, failed = github.get_upstream_dependencies_for_repos(
        repos=rs_graph_repos.repo.tolist(),
    )

    # Convert to dataframe
    upstream_deps_df = pd.DataFrame([d.to_dict() for d in upstream_deps])
    failed_df = pd.DataFrame([f.to_dict() for f in failed])

    # Store upstream deps
    output_filepath_for_successful_repos = DATA_FILES_DIR / "upstream-deps.parquet"
    upstream_deps_df.to_parquet(output_filepath_for_successful_repos)
    log.info(f"Stored upstream deps to: '{output_filepath_for_successful_repos}'")

    # Store failed repos
    output_filepath_for_failed_repos = DATA_FILES_DIR / "upstream-deps-failed.parquet"
    failed_df.to_parquet(output_filepath_for_failed_repos)
    log.info(f"Stored failed repos to: '{output_filepath_for_failed_repos}'")


@app.command()
def get_extended_paper_details(debug: bool = False) -> None:
    """
    Get the extended paper details for each paper in the dataset.

    Be sure to set the `SEMANTIC_SCHOLAR_API_KEY` environment variable.
    """
    # Setup logger
    setup_logger(debug=debug)

    # Read repos dataset
    rs_graph_repos = load_basic_repos_dataset()

    # Get extended paper details
    extended_paper_details = semantic_scholar.get_extended_paper_details(
        paper_dois=rs_graph_repos.doi.tolist(),
    )

    # Convert to dataframe
    extended_paper_details_df = pd.DataFrame(
        [d.to_dict() for d in extended_paper_details]
    )

    # Store extended paper details
    output_filepath_for_author_details = (
        DATA_FILES_DIR / "extended-paper-details.parquet"
    )
    extended_paper_details_df.to_parquet(output_filepath_for_author_details)
    log.info(
        f"Stored extended paper details to: '{output_filepath_for_author_details}'"
    )


@app.command()
def get_repo_contributors(debug: bool = False) -> None:
    """
    Get the contributors for each repo in the dataset.

    Be sure to set the `GITHUB_TOKEN` environment variable.
    """
    # Setup logger
    setup_logger(debug=debug)

    # Read repos dataset
    rs_graph_repos = load_basic_repos_dataset()

    # Store repo contributors
    output_filepath_for_repo_contributors = DATA_FILES_DIR / "repo-contributors.parquet"

    # Get repo contributors
    repo_contributors = github.get_repo_contributors_for_repos(
        repo_urls=rs_graph_repos.repo.tolist(),
        cache_file=output_filepath_for_repo_contributors,
    )

    # Convert to dataframe
    repo_contributors_df = pd.DataFrame([d.to_dict() for d in repo_contributors])

    # Store repo contributors
    repo_contributors_df.to_parquet(output_filepath_for_repo_contributors)
    log.info(f"Stored repo contributors to: '{output_filepath_for_repo_contributors}'")


def _finalize_matches(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    # Only store basic info
    basic_info = []
    for _, row in matches.iterrows():
        basic_info.append(
            {
                "gh_username": row.dev_details["username"],
                "author_id": row.author_details["author_id"],
                "match_proba": row.match_proba,
            }
        )

    # Convert to dataframe
    basic_info_df = pd.DataFrame(basic_info)

    # In the case where the same gh_username is tied to multiple authors,
    # keep the one with the highest match_proba
    basic_info_df = basic_info_df.sort_values(
        by="match_proba",
        ascending=False,
    ).drop_duplicates(subset=["gh_username"])

    # In the case where the same author_id is tied to multiple gh_usernames,
    # keep the one with the highest match_proba
    basic_info_df = basic_info_df.sort_values(
        by="match_proba",
        ascending=False,
    ).drop_duplicates(subset=["author_id"])

    return basic_info_df


@app.command()
def match_devs_and_authors(  # noqa: C901
    batch_size: int = 1000,
    debug: bool = False,
) -> None:
    # Setup logger
    setup_logger(debug=debug)

    # Load datasets
    log.info("Loading datasets...")
    devs = load_repo_contributors_dataset().drop_duplicates(subset=["username"])
    authors = load_author_contributions_dataset()

    # Construct repo_to_devs and repo_to_authors luts
    log.info("Constructing repo_to_devs and repo_to_authors luts...")
    repo_to_devs_lut: dict[str, list[str]] = {}
    for _, row in devs.iterrows():
        if row.repo not in repo_to_devs_lut:
            repo_to_devs_lut[row.repo] = []
        repo_to_devs_lut[row.repo].append(row.username)

    repo_to_authors_lut: dict[str, list[str]] = {}
    simple_author_details = []
    for _, row in authors.iterrows():
        if row.author_id is None:
            continue

        # For each repo in contributions, add the author to the lut
        for contrib in row.contributions:
            if contrib["repo"] not in repo_to_authors_lut:
                repo_to_authors_lut[contrib["repo"]] = []

            repo_to_authors_lut[contrib["repo"]].append(row.author_id)

        # Add author to authors_with_names_and_ids
        simple_author_details.append(
            {
                "author_id": row.author_id,
                "name": row["name"],
            }
        )

    # Convert to dataframe
    simple_author_details_df = pd.DataFrame(simple_author_details)

    # For each repo, create pairwise comparisons between devs and authors
    comparisons = []
    for repo, dev_usernames in tqdm(
        repo_to_devs_lut.items(),
        desc="Creating pairwise comparisons between devs and authors",
        total=len(repo_to_devs_lut),
    ):
        if repo in repo_to_authors_lut:
            author_ids = repo_to_authors_lut[repo]
            for dev_username in dev_usernames:
                for author_id in author_ids:
                    comparisons.append(
                        {
                            "repo": repo,
                            "dev_details": devs.loc[devs.username == dev_username]
                            .iloc[0]
                            .to_dict(),
                            "author_details": simple_author_details_df.loc[
                                simple_author_details_df.author_id == author_id
                            ]
                            .iloc[0]
                            .to_dict(),
                        }
                    )

    # Convert to dataframe
    comparisons_df = pd.DataFrame(comparisons)

    # Load models once
    log.info("Loading models...")
    embed_model = SentenceTransformer(DEFAULT_DEV_AUTHOR_EMBEDDING_MODEL_NAME)
    clf_model = load_base_dev_author_em_model()

    # Take batches of batch_size and predict matches
    predicted_rows: list[dict[str, str | dict]] = []
    for i in tqdm(
        range(0, len(comparisons_df), batch_size),
        desc="Cached-batched prediction",
        total=len(comparisons_df) // batch_size,
    ):
        batch = comparisons_df.iloc[i : i + batch_size]
        batch_predictions, batch_proba_matches = predict_dev_author_matches(
            devs=batch.dev_details.tolist(),
            authors=batch.author_details.tolist(),
            embedding_model=embed_model,
            clf_model=clf_model,
            show_progress_bar=False,
            additional_return_proba=True,
        )

        # Type checking, we must check that batch_proba_matches aren't None
        assert batch_proba_matches is not None

        # Add each prediction to predicted_rows
        for i, pred in enumerate(batch_predictions):
            if pred == "match":
                predicted_rows.append(
                    {
                        **batch.iloc[i].to_dict(),
                        "match_prediction": pred,
                        "match_proba": batch_proba_matches[i],
                    }
                )

        # Convert to dataframe
        predicted_df = pd.DataFrame(predicted_rows)

        # Finalize matches
        predicted_df = _finalize_matches(predicted_df).reset_index(drop=True)

        # Cache
        predicted_df.to_parquet(MATCHED_DEV_AUTHOR_IDS_PATH)

    # Log stored path
    log.info(f"Stored matched dev and author ids to: '{MATCHED_DEV_AUTHOR_IDS_PATH}'")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
