#!/usr/bin/env python

import logging

import dedupe
import pandas as pd
import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import (
    DATA_FILES_DIR,
    RS_GRAPH_DEDUPED_REPO_CONTRIBUTORS_PATH,
    load_rs_graph_repo_contributors_dataset,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


def _clustered_devs_dataframe_to_storage_ready(
    clustered_unique_devs_df: pd.DataFrame,
) -> None:
    # Prepare final dataset that is ready for linkage
    log.info("Preparing final dataset for linkage...")
    processed_rows = []
    for cluster_id, group in clustered_unique_devs_df.groupby("cluster_id"):
        # Get canonical username (longest username)
        canonical_username = ""
        for username in group.username:
            if len(username) > len(canonical_username):
                canonical_username = username

        # Get all usernames
        all_usernames = group.username.to_list()

        # Get all repos
        all_repos = set()
        for repos in group.repos:
            all_repos.update(repos)

        # Get all names
        all_names = {name for name in group.name if name is not None}

        # Get all companies
        all_companies = {company for company in group.company if company is not None}

        # Get all emails
        all_emails = {email for email in group.email if email is not None}

        # Get all locations
        all_locations = {
            location for location in group.location if location is not None
        }

        # Get all bios
        all_bios = {bio for bio in group.bio if bio is not None}

        # Get all co-contributors
        all_co_contributors = set()
        for co_contributors in group.co_contributors:
            all_co_contributors.update(co_contributors)

        # Add to processed rows
        processed_rows.append(
            {
                "canonical_username": canonical_username,
                "usernames": all_usernames,
                "repos": all_repos,
                "names": all_names,
                "companies": all_companies,
                "emails": all_emails,
                "locations": all_locations,
                "bios": all_bios,
                "co_contributors": all_co_contributors,
                "cluster_id": cluster_id,
            }
        )

    # Convert to dataframe
    processed_rows_df = pd.DataFrame(processed_rows)
    processed_rows_df.to_parquet(RS_GRAPH_DEDUPED_REPO_CONTRIBUTORS_PATH)
    log.info(
        f"Stored deduped repo contributors to: "
        f"'{RS_GRAPH_DEDUPED_REPO_CONTRIBUTORS_PATH}'"
    )


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

    # Prepare final dataset that is ready for linkage
    _clustered_devs_dataframe_to_storage_ready(clustered_unique_devs_df)


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
