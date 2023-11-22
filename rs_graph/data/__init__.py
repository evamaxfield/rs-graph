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
RS_GRAPH_UPSTREAM_DEPS_PATH = DATA_FILES_DIR / "rs-graph-upstream-deps.parquet"
RS_GRAPH_EXTENDED_PAPER_DETAILS_PATH = (
    DATA_FILES_DIR / "rs-graph-extended-paper-details.parquet"
)
RS_GRAPH_REPO_CONTRIBUTORS_PATH = DATA_FILES_DIR / "rs-graph-repo-contributors.parquet"
RS_GRAPH_DEDUPED_REPO_CONTRIBUTORS_PATH = (
    DATA_FILES_DIR / "rs-graph-deduped-repo-contributors.parquet"
)
RS_GRAPH_LINKED_AUTHORS_DEVS_PATH = (
    DATA_FILES_DIR / "rs-graph-linked-authors-devs.parquet"
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

    return embeddings


@dataclass
class Contribution(DataClassJsonMixin):
    repo: str
    doi: str
    author_position: int
    co_authors: list[str]
    embedding: list[float] | None


@dataclass
class AuthorshipDetails(DataClassJsonMixin):
    name: str
    aliases: set[str]
    h_index: int
    contributions: list[Contribution]


def _get_canonical_name(author_details: dict) -> str:
    """Get the canonical name for an author."""
    # Get longest name in aliases to use as "canonical name"
    canonical_name = author_details["name"]
    if author_details["aliases"] is not None:
        for alias in author_details["aliases"]:
            if len(alias) > len(canonical_name):
                canonical_name = alias

    return canonical_name


def load_rs_graph_author_contributions_dataset() -> pd.DataFrame:
    # Load extended paper details dataset
    paper_details_df = load_rs_graph_extended_paper_details_dataset()
    repos_df = load_rs_graph_repos_dataset()

    # Create a look up table for each author
    all_author_contributions: dict[str, AuthorshipDetails] = {}
    for _, paper_details in paper_details_df.iterrows():
        # Get DOI so we don't have to do a lot of getitems
        doi = paper_details["doi"]

        # Embedding or none
        embedding_details = paper_details["embedding"]
        embedding = None if embedding_details is None else embedding_details["vector"]

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
                    co_author_canonical_name = _get_canonical_name(co_author_details)
                    co_authors.append(co_author_canonical_name)

            # Add new author
            if a_id not in all_author_contributions:
                # Get longest name in aliases to use as "cannonical name"
                canonical_name = _get_canonical_name(author_details)

                # Convert aliases to set and "original author name"
                aliases = (
                    set(author_details["aliases"])
                    if author_details["aliases"] is not None
                    else set()
                )
                aliases.add(author_details["name"])

                # Add new author
                all_author_contributions[a_id] = AuthorshipDetails(
                    name=canonical_name,
                    aliases=aliases,
                    h_index=author_details["h_index"],
                    contributions=[
                        Contribution(
                            repo=repo_row["repo"],
                            doi=doi,
                            author_position=author_details["author_position"],
                            co_authors=co_authors,
                            embedding=embedding,
                        )
                    ],
                )

            # Update existing author
            else:
                # Get existing author
                existing_author_details = all_author_contributions[a_id]

                # Always add new aliases
                if author_details["aliases"] is not None:
                    existing_author_details.aliases.update(author_details["aliases"])

                # Always add possibly new name to aliases
                existing_author_details.aliases.add(author_details["name"])

                # Update cannonical name if need be
                for alias in existing_author_details.aliases:
                    if len(alias) > len(existing_author_details.name):
                        existing_author_details.name = alias

                # Add new contribution
                existing_author_details.contributions.append(
                    Contribution(
                        repo=repo_row["repo"],
                        doi=doi,
                        author_position=author_details["author_position"],
                        co_authors=co_authors,
                        embedding=embedding,
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


def load_rs_graph_repo_contributors_dataset() -> pd.DataFrame:
    """Load the repo contributors dataset."""
    return pd.read_parquet(RS_GRAPH_REPO_CONTRIBUTORS_PATH)


def load_rs_graph_deduped_repo_contributors_dataset() -> pd.DataFrame:
    """Load the deduped repo contributors dataset."""
    return pd.read_parquet(RS_GRAPH_DEDUPED_REPO_CONTRIBUTORS_PATH)


def load_rs_graph_linked_authors_devs_dataset() -> pd.DataFrame:
    """Load the linked authors devs dataset."""
    return pd.read_parquet(RS_GRAPH_LINKED_AUTHORS_DEVS_PATH)
