#!/usr/bin/env python

import logging

import dedupe
import pandas as pd
import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import (
    DATA_FILES_DIR,
    RS_GRAPH_DEDUPED_REPO_CONTRIBUTORS_PATH,
    RS_GRAPH_LINKED_AUTHORS_DEVS_PATH,
    load_rs_graph_author_contributions_dataset,
    load_rs_graph_deduped_repo_contributors_dataset,
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


@app.command()
def train_author_developer_linker(debug: bool = False) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the repo contributors dataset
    log.info("Loading developer contributions dataset...")
    devs = load_rs_graph_deduped_repo_contributors_dataset()

    # Load the author contributions dataset
    log.info("Loading author contributions dataset...")
    authors = load_rs_graph_author_contributions_dataset()

    # Construct dataframe of author details ready for processing
    log.info("Constructing author details dataframe...")
    authors_ready_rows = []
    for _, author in authors.iterrows():
        repos = {contribution["repo"] for contribution in author.contributions}

        authors_ready_rows.append(
            {
                "name": author["name"],
                "secondary_name": author["name"],  # username comparison
                "repos": tuple(repos),
            }
        )

    # Make frame
    authors_ready = pd.DataFrame(authors_ready_rows)

    # Construct dataframe of developer details ready for processing
    log.info("Constructing developer details dataframe...")
    devs_ready_rows = []
    for _, dev in devs.iterrows():
        canonical_name = ""
        for name in dev.names:
            if len(name) > len(canonical_name):
                canonical_name = name

        # Catch no names for person
        # use username
        if len(canonical_name) == 0:
            canonical_name = dev.canonical_username

        devs_ready_rows.append(
            {
                "name": canonical_name,
                "secondary_name": dev.canonical_username,
                "repos": tuple(dev.repos),
            }
        )

    # Make frame
    devs_ready = pd.DataFrame(devs_ready_rows)

    # Convert both to records orientation
    author_records = {i: row.to_dict() for i, row in authors_ready.iterrows()}
    dev_records = {i: row.to_dict() for i, row in devs_ready.iterrows()}

    # Linker fields
    fields = [
        # author given name <--> github given name (or fallback github username)
        # both string diff and name diff
        {"field": "name", "type": "String"},
        {"field": "name", "type": "Name"},
        # author given name <--> github username
        {"field": "secondary_name", "type": "String"},
        # set repos authored (via JOSS / softwareX) <--> set repos contributed
        {"field": "repos", "type": "Set"},
    ]

    # Create linker
    linker = dedupe.RecordLink(fields)

    # Find which dataset had less records
    # and create sample size using 80% of dataset size
    min_dataset_size = min(len(author_records), len(dev_records))
    sample_size = int(min_dataset_size * 0.8)

    # Prepare training samples
    log.info("Preparing training samples...")
    linker.prepare_training(author_records, dev_records, sample_size=sample_size)

    # Start annotating
    dedupe.console_label(linker)

    # Train the model
    linker.train()

    # Save the model training settings
    train_settings_filepath = (
        DATA_FILES_DIR / "author-dev-linker-training-settings.json"
    )
    with open(train_settings_filepath, "w") as f:
        linker.write_training(f)

    # Save the model clustering settings / weights
    cluster_settings_filepath = (
        DATA_FILES_DIR / "author-dev-linker-cluster-settings.pkl"
    )
    with open(cluster_settings_filepath, "wb") as f:
        linker.write_settings(f)

    # Join the records
    linked_records = linker.join(author_records, dev_records, threshold=0.0)

    # Store linked records
    linked_details_rows = []
    for (author_id, dev_id), confidence in linked_records:
        # Get matching author and dev
        author = authors.loc[author_id]
        dev = devs.loc[dev_id]

        # Create linked record
        linked_details_rows.append(
            {
                "author_id": author.author_id,
                "name": author["name"],
                "h_index": author.h_index,
                "usernames": dev.usernames,
                "repos": dev.repos,
                "confidence": confidence,
            }
        )

    # Make frame
    linked_details = pd.DataFrame(linked_details_rows)

    # Store linked details
    linked_details.to_parquet(RS_GRAPH_LINKED_AUTHORS_DEVS_PATH)
    log.info(f"Stored linked author-devs to: " f"'{RS_GRAPH_LINKED_AUTHORS_DEVS_PATH}'")

    # Log n linked
    n_linked = linked_details.author_id.nunique()
    log.info(f"Number of linked authors and devs: {n_linked}")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
