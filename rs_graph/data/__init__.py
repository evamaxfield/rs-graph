"""Stored dataset loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

###############################################################################
# Remote storage paths

GCS_PROJECT_ID = "sci-software-graph"
REMOTE_STORAGE_BUCKET = "gs://sci-software-graph-data-store"

###############################################################################
# Local storage paths

DATA_FILES_DIR = Path(__file__).parent / "files"

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
