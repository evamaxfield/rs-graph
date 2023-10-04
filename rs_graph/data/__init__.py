"""Preprocessing and availability of different datasets."""

from pathlib import Path

import pandas as pd

###############################################################################

DATA_FILES_DIR = Path(__file__).parent / "files"
JOSS_DATASET_PATH = DATA_FILES_DIR / "joss-2023-10-02.parquet"
SOFTWAREX_DATASET_PATH = DATA_FILES_DIR / "softwarex-2023-10-03.parquet"

###############################################################################


def load_joss_repos() -> pd.DataFrame:
    """Load the JOSS dataset."""
    return pd.read_parquet(JOSS_DATASET_PATH)


def load_softwarex_repos() -> pd.DataFrame:
    """Load the SoftwareX dataset."""
    return pd.read_parquet(SOFTWAREX_DATASET_PATH)


def load_rs_graph_repos_dataset() -> pd.DataFrame:
    """Load the base dataset."""
    # Load softwarex and use _parent_repo_url where possible otherwise use repo
    softwarex_df = load_softwarex_repos()
    softwarex_df["repo"] = softwarex_df["_parent_repo_url"].fillna(softwarex_df["repo"])
    softwarex_df = softwarex_df.drop(columns=["_parent_repo_url"])

    # Concat
    rs_graph = pd.concat(
        [
            load_joss_repos(),
            softwarex_df,
        ],
        ignore_index=True,
    )

    # Drop duplicates and keep first
    rs_graph = rs_graph.drop_duplicates(subset=["repo"], keep="first")

    return rs_graph
