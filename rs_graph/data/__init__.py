"""Stored dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dataclasses_json import DataClassJsonMixin

###############################################################################
# Remote storage paths

GCS_PROJECT_ID = "sci-software-graph"
REMOTE_STORAGE_BUCKET = "gs://sci-software-graph-data-store"

###############################################################################
# Local storage paths

DATA_FILES_DIR = Path(__file__).parent / "files"

# Dataset sources are found via path globbing
DATASET_SOURCE_FILE_PATTERN = "-short-paper-details.parquet"

# Other datasets are formed from enrichment and have hardcoded paths
UPSTREAM_DEPS_PATH = DATA_FILES_DIR / "upstream-deps.parquet"
EXTENDED_PAPER_DETAILS_PATH = DATA_FILES_DIR / "extended-paper-details.parquet"
REPO_CONTRIBUTORS_PATH = DATA_FILES_DIR / "repo-contributors.parquet"
MATCHED_DEV_AUTHOR_IDS_PATH = DATA_FILES_DIR / "matched-dev-author-ids.parquet"

# Annotated datasets
ANNOTATED_DEV_AUTHOR_EM_IRR_PATH = DATA_FILES_DIR / "annotated-dev-author-em-irr.csv"
ANNOTATED_DEV_AUTHOR_EM_PATH = DATA_FILES_DIR / "annotated-dev-author-em.csv"

###############################################################################


def load_basic_repos_dataset() -> pd.DataFrame:
    """Load the base dataset (all dataset sources)."""
    # Find all dataset files
    dataset_files = list(DATA_FILES_DIR.glob(f"*{DATASET_SOURCE_FILE_PATTERN}"))

    # Load all datasets
    datasets = []
    for dataset_file in dataset_files:
        datasets.append(pd.read_parquet(dataset_file))

    # Concatenate
    rs_graph = pd.concat(datasets)

    # Drop duplicates and keep first
    rs_graph = rs_graph.drop_duplicates(subset=["repo"], keep="first")

    return rs_graph


def load_upstream_dependencies_dataset() -> pd.DataFrame:
    """Load the upstream dependencies dataset."""
    return pd.read_parquet(UPSTREAM_DEPS_PATH)


def load_extended_paper_details_dataset() -> pd.DataFrame:
    """Load the extended paper details dataset."""
    return pd.read_parquet(EXTENDED_PAPER_DETAILS_PATH)


def load_embeddings_dataset() -> pd.DataFrame:
    """Load the extended papers details dataset then format to embeddings focus."""
    # Load extended paper details
    df = load_extended_paper_details_dataset()

    # Load and process data
    embedding_rows = []
    for _, row in df.iterrows():
        if row.embedding is not None:
            embedding_rows.append(
                {
                    "url": row.url,
                    "doi": row.doi,
                    "title": row.title,
                    "embedding": np.array(row.embedding["vector"]),
                    "citation_count": row.citation_count,
                }
            )

    # Convert to frame and store log citation count
    embeddings = pd.DataFrame(embedding_rows).reset_index(drop=True)
    embeddings["log_citation_count"] = np.log(embeddings.citation_count)

    return embeddings


@dataclass
class Contribution(DataClassJsonMixin):
    repo: str
    doi: str
    author_position: int
    co_authors: list[str]


@dataclass
class AuthorshipDetails(DataClassJsonMixin):
    name: str
    h_index: int
    contributions: list[Contribution]


def load_author_contributions_dataset() -> pd.DataFrame:
    # Load extended paper details dataset
    paper_details_df = load_extended_paper_details_dataset()
    repos_df = load_basic_repos_dataset()

    # Create a look up table for each author
    all_author_contributions: dict[str, AuthorshipDetails] = {}
    for _, paper_details in paper_details_df.iterrows():
        # Get DOI so we don't have to do a lot of getitems
        doi = paper_details["doi"]

        # Get matching row in the repos dataset
        repo_row = repos_df.loc[repos_df.doi == doi]

        # Skip if no matching row
        if len(repo_row) == 0:
            continue
        else:
            repo_row = repo_row.iloc[0]

        # Iter each author
        for author_details in paper_details["authors"]:
            a_id = author_details["author_id"]

            # Compute co-authors
            co_authors = []
            for co_author_details in paper_details["authors"]:
                if co_author_details["author_id"] != a_id:
                    co_authors.append(co_author_details["name"])

            # Add new author
            if a_id not in all_author_contributions:
                # Add new author
                all_author_contributions[a_id] = AuthorshipDetails(
                    name=author_details["name"],
                    h_index=author_details["h_index"],
                    contributions=[
                        Contribution(
                            repo=repo_row["repo"],
                            doi=doi,
                            author_position=author_details["author_position"],
                            co_authors=co_authors,
                        )
                    ],
                )

            # Update existing author
            else:
                # Get existing author
                existing_author_details = all_author_contributions[a_id]

                # Add new contribution
                existing_author_details.contributions.append(
                    Contribution(
                        repo=repo_row["repo"],
                        doi=doi,
                        author_position=author_details["author_position"],
                        co_authors=co_authors,
                    )
                )

    # Convert to dataframe
    all_author_details_df = pd.DataFrame(
        [
            {
                "author_id": author_id,
                **author_details.to_dict(),
            }
            for author_id, author_details in all_author_contributions.items()
        ]
    )
    return all_author_details_df


def load_repo_contributors_dataset() -> pd.DataFrame:
    """Load the repo contributors dataset."""
    return pd.read_parquet(REPO_CONTRIBUTORS_PATH)


def load_annotated_dev_author_em_irr_dataset() -> pd.DataFrame:
    """Load the annotated author dev em irr dataset."""
    return pd.read_csv(ANNOTATED_DEV_AUTHOR_EM_IRR_PATH)


def load_multi_annotator_dev_author_em_irr_dataset() -> pd.DataFrame:
    """Load the multi-annotator author dev em irr dataset."""
    # Load akhil and terra separately
    akhil = pd.read_csv(DATA_FILES_DIR / "annotated-dev-author-em-dataset-akhil.csv")
    terra = pd.read_csv(DATA_FILES_DIR / "annotated-dev-author-em-dataset-terra.csv")

    # Keep only the first 2999 rows
    akhil = akhil.head(2999)
    terra = terra.head(2999)

    # Drop columns from terra for merge
    terra = terra.drop(columns=["dev_details", "author_details"])

    # Merge on index
    multi_annotator = akhil.merge(
        terra,
        on=["github_id", "semantic_scholar_id"],
        suffixes=("_akhil", "_terra"),
    )

    return multi_annotator


def load_matched_dev_author_ids_dataset() -> pd.DataFrame:
    """Load the matched dev author ids dataset."""
    authors = load_author_contributions_dataset()
    matched_dev_authors = pd.read_parquet(MATCHED_DEV_AUTHOR_IDS_PATH)

    # Merge on author_id
    matched_dev_authors = matched_dev_authors.merge(
        authors,
        on="author_id",
    )

    return matched_dev_authors
