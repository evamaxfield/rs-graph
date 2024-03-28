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
EXTENDED_PAPER_DETAILS_PATH = DATA_FILES_DIR / "extended-paper-details.parquet"
REPO_CONTRIBUTORS_PATH = DATA_FILES_DIR / "repo-contributors.parquet"

# Annotated datasets

# Dev Author EM datasets
ANNOTATED_DEV_AUTHOR_EM_PRACTICE_AKHIL_PATH = (
    DATA_FILES_DIR / "annotated-dev-author-em-practice-akhil.csv"
)
ANNOTATED_DEV_AUTHOR_EM_PRACTICE_TERRA_PATH = (
    DATA_FILES_DIR / "annotated-dev-author-em-practice-terra.csv"
)
ANNOTATED_DEV_AUTHOR_EM_UNRESOLVED_AKHIL_PATH = (
    DATA_FILES_DIR / "annotated-dev-author-em-unresolved-akhil.csv"
)
ANNOTATED_DEV_AUTHOR_EM_UNRESOLVED_TERRA_PATH = (
    DATA_FILES_DIR / "annotated-dev-author-em-unresolved-terra.csv"
)
ANNOTATED_DEV_AUTHOR_EM_PATH = DATA_FILES_DIR / "annotated-dev-author-em-resolved.csv"

# Repo Paper EM datasets
ANNOTATED_REPO_PAPER_EM_PRACTICE_AKHIL_PATH = (
    DATA_FILES_DIR / "annotated-repo-paper-em-practice-akhil.csv"
)
ANNOTATED_REPO_PAPER_EM_PRACTICE_TERRA_PATH = (
    DATA_FILES_DIR / "annotated-repo-paper-em-practice-terra.csv"
)

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


def load_extended_paper_details_dataset() -> pd.DataFrame:
    """Load the extended paper details dataset."""
    return pd.read_parquet(EXTENDED_PAPER_DETAILS_PATH)


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


def load_multi_annotator_dev_author_em_irr_dataset(
    use_full_dataset: bool = False,
) -> pd.DataFrame:
    """Load the multi-annotator author dev em irr dataset."""
    if use_full_dataset:
        # Load akhil and terra separately
        akhil = pd.read_csv(ANNOTATED_DEV_AUTHOR_EM_UNRESOLVED_AKHIL_PATH)
        terra = pd.read_csv(ANNOTATED_DEV_AUTHOR_EM_UNRESOLVED_TERRA_PATH)

        # Drop columns from terra for merge
        terra = terra.drop(columns=["dev_details", "author_details"])

        # Merge on index
        return akhil.merge(
            terra,
            on=["github_id", "semantic_scholar_id"],
            suffixes=("_akhil", "_terra"),
        )

    # Load akhil and terra separately
    akhil = pd.read_csv(ANNOTATED_DEV_AUTHOR_EM_PRACTICE_AKHIL_PATH)[:50]
    terra = pd.read_csv(ANNOTATED_DEV_AUTHOR_EM_PRACTICE_TERRA_PATH)[:50]

    # Merge on shared columns
    return akhil.merge(
        terra,
        on=["dev_details", "author_details"],
        suffixes=("_akhil", "_terra"),
    )


def load_annotated_dev_author_em_dataset() -> pd.DataFrame:
    """Load the annotated dev author em dataset."""
    return pd.read_csv(ANNOTATED_DEV_AUTHOR_EM_PATH)


def load_multi_annotator_repo_paper_em_irr_dataset() -> pd.DataFrame:
    """Load the multi-annotator repo paper em irr dataset."""
    # Load akhil and terra separately
    akhil = pd.read_csv(ANNOTATED_REPO_PAPER_EM_PRACTICE_AKHIL_PATH)
    terra = pd.read_csv(ANNOTATED_REPO_PAPER_EM_PRACTICE_TERRA_PATH)

    # Sort by "id_"
    akhil = akhil.sort_values(by="id_")
    terra = terra.sort_values(by="id_")

    # If a row has a "remove" label in the "remove" column,
    # then make the value in the "match" column, "remove"
    akhil.loc[akhil["remove"] == "remove", "match"] = "remove"
    terra.loc[terra["remove"] == "remove", "match"] = "remove"

    # Drop the remove column
    akhil = akhil.drop(columns=["remove"])
    terra = terra.drop(columns=["remove"])

    # Replace True and False values in "match" column with
    # "match" and "no-match"
    akhil.loc[akhil["match"] is True, "match"] = "match"
    akhil.loc[akhil["match"] is False, "match"] = "no-match"
    terra.loc[terra["match"] is True, "match"] = "match"
    terra.loc[terra["match"] is False, "match"] = "no-match"

    # Merge on shared columns
    return akhil.merge(
        terra,
        on=["source", "id_", "paper_url", "repo_url"],
        suffixes=("_akhil", "_terra"),
    )
