"""Stored dataset loaders."""

from pathlib import Path

import numpy as np
import pandas as pd

###############################################################################

DATA_FILES_DIR = Path(__file__).parent / "files"

# Dataset sources are found via path globbing
DATASET_SOURCE_FILE_PATTERN = "-short-paper-details.parquet"

# Other datasets are formed from enrichment and have hardcoded paths
RS_GRAPH_UPSTREAM_DEPS_PATH = DATA_FILES_DIR / "rs-graph-upstream-deps.parquet"
RS_GRAPH_EXTENDED_PAPER_DETAILS_PATH = (
    DATA_FILES_DIR / "rs-graph-extended-paper-details.parquet"
)

###############################################################################


def load_rs_graph_repos_dataset() -> pd.DataFrame:
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


def load_rs_graph_upstream_dependencies_dataset() -> pd.DataFrame:
    """Load the upstream dependencies dataset."""
    return pd.read_parquet(RS_GRAPH_UPSTREAM_DEPS_PATH)


def load_rs_graph_extended_paper_details_dataset() -> pd.DataFrame:
    """Load the extended paper details dataset."""
    return pd.read_parquet(RS_GRAPH_EXTENDED_PAPER_DETAILS_PATH)


def load_rs_graph_embeddings_dataset() -> pd.DataFrame:
    """Load the extended papers details dataset then format to embeddings focus."""
    # Load extended paper details
    df = load_rs_graph_extended_paper_details_dataset()

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

    return df
