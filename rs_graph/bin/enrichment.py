#!/usr/bin/env python

import logging

import pandas as pd
import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import DATA_FILES_DIR, load_rs_graph_repos_dataset
from rs_graph.data.enrichment import github, semantic_scholar

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
    rs_graph_repos = load_rs_graph_repos_dataset()

    # Get upstream deps
    upstream_deps, failed = github.get_upstream_dependencies_for_repos(
        repos=rs_graph_repos.repo.tolist(),
    )

    # Convert to dataframe
    upstream_deps_df = pd.DataFrame([d.to_dict() for d in upstream_deps])
    failed_df = pd.DataFrame([f.to_dict() for f in failed])

    # Store upstream deps
    output_filepath_for_successful_repos = (
        DATA_FILES_DIR / "rs-graph-upstream-deps.parquet"
    )
    upstream_deps_df.to_parquet(output_filepath_for_successful_repos)
    log.info(f"Stored upstream deps to: '{output_filepath_for_successful_repos}'")

    # Store failed repos
    output_filepath_for_failed_repos = (
        DATA_FILES_DIR / "rs-graph-upstream-deps-failed.parquet"
    )
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
    rs_graph_repos = load_rs_graph_repos_dataset()

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
        DATA_FILES_DIR / "rs-graph-extended-paper-details.parquet"
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
    rs_graph_repos = load_rs_graph_repos_dataset().sample(300)

    # Get repo contributors
    repo_contributors = github.get_repo_contributors_for_repos(
        repo_urls=rs_graph_repos.repo.tolist(),
    )

    # Convert to dataframe
    repo_contributors_df = pd.DataFrame([d.to_dict() for d in repo_contributors])

    # Store repo contributors
    output_filepath_for_repo_contributors = (
        DATA_FILES_DIR / "rs-graph-repo-contributors.parquet"
    )
    repo_contributors_df.to_parquet(output_filepath_for_repo_contributors)
    log.info(
        f"Stored repo contributors to: '{output_filepath_for_repo_contributors}'"
    )

###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
