#!/usr/bin/env python

import logging

import dedupe
import numpy as np
import pandas as pd
import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import (
    DATA_FILES_DIR,
    load_rs_graph_author_contributions_dataset,
    load_rs_graph_repo_contributors_dataset,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


def _cosine_diff_or_none_comparator(
    a: np.ndarray | None,
    b: np.ndarray | None,
) -> float:
    if a is None or b is None:
        return 0.5  # "random"

    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.command()
def train_author_deduper(debug: bool = False) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the author contributions dataset
    author_contributions = load_rs_graph_author_contributions_dataset()

    # Format as records dict
    author_records = {}
    for _, row in author_contributions.sample(500).iterrows():
        # Get all co authors from all contributions
        co_authors = set()
        for contribution in row.contributions:
            co_authors.update(set(contribution["co_authors"]))

        # Get all embeddings of contributed papers as np arrays
        contribution_embeddings = []
        for contribution in row.contributions:
            if contribution["embedding"] is not None:
                contribution_embeddings.append(np.array(contribution["embedding"]))

        # Take mean
        if len(contribution_embeddings) > 0:
            mean_embedding = np.mean(np.stack(contribution_embeddings), axis=0)
        else:
            mean_embedding = None

        # Create new record
        author_records[row.author_id] = {
            "name": row["name"],
            "aliases": tuple(row.aliases),
            "co_authors": tuple(co_authors),
            "mean_embedding": mean_embedding,
        }

    # Variables for dedupe
    variables = [
        {"field": "name", "type": "Name"},
        {"field": "aliases", "type": "Set"},
        {"field": "co_authors", "type": "Set"},
        {
            "field": "mean_embedding",
            "type": "Custom",
            "comparator": _cosine_diff_or_none_comparator,
        },
    ]

    # Create deduper
    deduper = dedupe.Dedupe(variables)

    # Create training pairs
    deduper.prepare_training(author_records, sample_size=400)

    # Start training
    dedupe.console_label(deduper)


@app.command()
def train_developer_deduper(debug: bool = False) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the repo contributors dataset
    log.info("Loading developer contributions dataset...")
    developer_contributions = load_rs_graph_repo_contributors_dataset()

    # Each row should be a unique person
    log.info("Getting unique developers frame...")
    unique_devs = []
    for username, group in developer_contributions.groupby("username"):
        flat_co_contribs = []
        for co_contribs in group.co_contributors:
            flat_co_contribs.extend(co_contribs)

        unique_devs.append(
            {
                "username": username,
                "repos": tuple(group.repo),
                "name": group["name"].iloc[0],
                "company": group.company.iloc[0],
                "email": group.email.iloc[0],
                "location": group.location.iloc[0],
                "bio": group.bio.iloc[0],
                "co_contributors": tuple(flat_co_contribs),
            }
        )

    # Reform as df
    unique_devs_df = pd.DataFrame(unique_devs)

    # Format as records dict
    developer_records = {i: row.to_dict() for i, row in unique_devs_df.iterrows()}

    # Variables for dedupe
    variables = [
        {"field": "username", "type": "String"},
        {"field": "repos", "type": "Set"},
        {"field": "name", "type": "Name", "has missing": True},
        {"field": "company", "type": "String", "has missing": True},
        {"field": "email", "type": "String", "has missing": True},
        {"field": "location", "type": "String", "has missing": True},
        {"field": "bio", "type": "String", "has missing": True},
        {"field": "co_contributors", "type": "Set"},
    ]

    # Init deduper
    log.info("Initializing deduper...")
    deduper = dedupe.Dedupe(variables)

    # Prepare training samples
    log.info("Preparing training samples...")
    deduper.prepare_training(
        developer_records,
        sample_size=int(len(developer_records) * 0.8),
    )

    # Start annotating
    log.info("Starting annotation...")
    dedupe.console_label(deduper)

    # Train the model
    log.info("Training model...")
    deduper.train()

    # Save the model training settings
    train_settings_filepath = (
        DATA_FILES_DIR / "developer-deduper-training-settings.json"
    )
    with open("train_settings_filepath", "w") as f:
        deduper.write_training(f)
    log.info(
        f"Stored developer deduper training settings to: '{train_settings_filepath}'"
    )

    # Save the model clustering settings / weights
    cluster_settings_filepath = (
        DATA_FILES_DIR / "developer-deduper-cluster-settings.pkl"
    )
    with open(cluster_settings_filepath, "wb") as f:
        deduper.write_settings(f)
    log.info(
        f"Stored developer deduper cluster settings to: '{cluster_settings_filepath}'"
    )

    # Partition the records
    log.info("Clustering all records...")
    clustered_records = deduper.partition(developer_records, threshold=0.5)
    cluster_membership = {}
    for cluster_id, (records, scores) in enumerate(clustered_records):
        for record_id, score in zip(records, scores, strict=True):
            cluster_membership[record_id] = {
                "cluster_id": cluster_id,
                "confidence_score": score,
            }

    # Add cluster membership to unique devs dataframe
    clustered_unique_devs = []
    for dev_id, cluster_details in cluster_membership.items():
        dev_details = developer_records[dev_id]
        dev_details["cluster_id"] = cluster_details["cluster_id"]
        dev_details["confidence"] = cluster_details["confidence_score"]
        clustered_unique_devs.append(dev_details)

    # Convert to dataframe
    clustered_unique_devs_df = pd.DataFrame(clustered_unique_devs)

    # Store clustered unique devs
    output_filepath_for_clustered_unique_devs = (
        DATA_FILES_DIR / "rs-graph-clustered-unique-devs.parquet"
    )
    clustered_unique_devs_df.to_parquet(output_filepath_for_clustered_unique_devs)
    log.info(
        f"Stored clustered unique devs to: "
        f"'{output_filepath_for_clustered_unique_devs}'"
    )

    # Log n deduped
    n_clusters = clustered_unique_devs_df.cluster_id.nunique()
    log.info(f"Number of original 'unique devs': {len(unique_devs_df)}")
    log.info(f"Number of deduped 'unique devs': {n_clusters}")
    diff = len(unique_devs_df) - n_clusters
    log.info(f"Difference: {diff} ({diff / len(unique_devs_df) * 100:.2f}%)")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
