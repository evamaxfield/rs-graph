#!/usr/bin/env python

import logging
from uuid import uuid4

import dedupe
import numpy as np
import pandas as pd
import typer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm

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


def _get_unique_devs_frame_from_dev_contributions(
    dev_contributions_df: pd.DataFrame,
) -> pd.DataFrame:
    # Each row should be a unique person
    log.info("Getting unique developers frame...")
    unique_devs = []
    for username, group in dev_contributions_df.groupby("username"):
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
    return pd.DataFrame(unique_devs)


def _dataframe_to_joined_str_items(
    df: pd.DataFrame,
    unique_key: str,
    tqdm_desc: str,
    ignore_columns: list[str] | None = None,
) -> dict[str, str]:
    # Init ignore columns
    if ignore_columns is None:
        ignore_columns = []

    joined_str_dict = {}
    for _, row in tqdm(df.iterrows(), desc=tqdm_desc):
        # Remove none values and store as dict
        details = row.to_dict()
        details = {key: value for key, value in details.items() if value is not None}

        # Construct string
        dev_values_list = []
        for key, value in details.items():
            if key not in ignore_columns:
                if isinstance(value, tuple):
                    multi_values = [v for v in value if v is not None]
                    value = ", ".join(multi_values)

                dev_values_list.append(f"{key}: {value};")

        # Get unique key value
        try:
            unique_key_value = details[unique_key]
        except KeyError:
            unique_key_value = None

        # If no unique key value, create a guid
        if unique_key_value is None:
            # Create a guid
            unique_key_value = str(uuid4())

        # Construct string
        joined_str_dict[unique_key_value] = "\n".join(dev_values_list)

    return joined_str_dict


# @app.command()
# def create_developer_deduper_dataset_for_annotation(
#     top_n_similar: int = 3,
#     model_name: str = "BAAI/bge-base-en-v1.5",
#     debug: bool = False,
# ) -> None:
#     # Setup logging
#     setup_logger(debug=debug)

#     # Load the repo contributors dataset
#     log.info("Loading developer contributions dataset...")
#     developer_contributions = load_rs_graph_repo_contributors_dataset()

#     # Get unique devs frame
#     unique_devs_df = _get_unique_devs_frame_from_dev_contributions(
#         developer_contributions,
#     )

#     # Start the dev comparisons details
#     prepped_devs_list = _dataframe_to_joined_str_items(
#         unique_devs_df,
#         tqdm_desc="Converting dev details to strings...",
#     )

#     # Create embeddings for each dev
#     log.info("Creating embeddings for each dev...")
#     model = SentenceTransformer(model_name)
#     dev_embeddings = model.encode(
#         prepped_devs_list,
#         show_progress_bar=True,
#     )

#     # Construct pairwise similarity matrix
#     pairwise_similarity_matrix: list[list[float]] = []
#     for dev_embedding in tqdm(
#         dev_embeddings,
#         desc="Calculating pairwise similarity matrix",
#     ):
#         this_dev_similarity_row = []
#         for other_dev_embedding in dev_embeddings:
#             # Cosine similarity
#             similarity = cos_sim(dev_embedding, other_dev_embedding)
#             this_dev_similarity_row.append(similarity.item())

#         pairwise_similarity_matrix.append(this_dev_similarity_row)

#     # Using the constructed pairwise similarity matrix
#     # Get top n most similar devs using cosine similarity
#     # for every dev (not including themselves)
#     # And finally construct the dev comparison dataframe
#     dev_comparison_rows = []
#     for i in tqdm(
#         range(len(prepped_devs_list)),
#         desc="Constructing dev comparison rows",
#     ):
#         # Get top n most similar devs
#         top_n_similar_devs = sorted(
#             [
#                 (j, similarity)
#                 for j, similarity in enumerate(pairwise_similarity_matrix[i])
#                 if i != j
#             ],
#             key=lambda x: x[1],
#             reverse=True,
#         )[:top_n_similar]

#         # Get the dev details
#         dev_details = prepped_devs_list[i]

#         # Get the top n most similar dev details
#         top_n_similar_dev_details = []
#         for j, similarity in top_n_similar_devs:
#             top_n_similar_dev_details.append(
#                 {
#                     "similarity": similarity,
#                     "details": prepped_devs_list[j],
#                 }
#             )

#         # Add to dev comparison rows
#         for other_dev_details in top_n_similar_dev_details:
#             dev_comparison_rows.append(
#                 {
#                     "dev_1_details": dev_details,
#                     "dev_2_details": other_dev_details["details"],
#                     "similarity": other_dev_details["similarity"],
#                 }
#             )

#     # Convert to dataframe
#     dev_comparison_df = pd.DataFrame(dev_comparison_rows)

#     # Store to disk
#     output_filepath = DATA_FILES_DIR / "developer-deduper-annotation-dataset.csv"
#     dev_comparison_df.to_csv(output_filepath, index=False)


@app.command()
def create_author_developer_linker_dataset_for_annotation(  # noqa: C901
    top_n_similar: int = 3,
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    debug: bool = False,
) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the repo contributors dataset
    log.info("Loading developer contributions dataset...")
    devs = load_rs_graph_repo_contributors_dataset()

    # Get unique devs frame
    unique_devs_df = _get_unique_devs_frame_from_dev_contributions(
        devs,
    )

    # Load the author contributions dataset
    log.info("Loading author contributions dataset...")
    authors = load_rs_graph_author_contributions_dataset()

    # Create lookup for repo to authors
    repo_to_authors: dict[str, set[str]] = {}
    author_id_to_name_dict: dict[str, str] = {}
    remade_authors = []
    for _, author in authors.iterrows():
        # Get uuid4 if author has no author id
        if author.author_id is None:
            author.author_id = str(uuid4())

        for contribution in author.contributions:
            if contribution["repo"] not in repo_to_authors:
                repo_to_authors[contribution["repo"]] = set()

            repo_to_authors[contribution["repo"]].add(author.author_id)

        remade_authors.append(author)

        # Add to name lookup
        author_id_to_name_dict[author.author_id] = author["name"]

    # Remake authors
    authors = pd.DataFrame(remade_authors)

    # Construct dataframe of author details ready for processing
    log.info("Constructing author details dataframe...")
    authors_ready_rows = []
    for _, author in authors.iterrows():
        repos = {contribution["repo"] for contribution in author.contributions}

        # Get all co-authors
        all_co_authors = set()
        for repo in repos:
            # Get author ids from repo_to_authors
            co_author_ids = repo_to_authors[repo]

            # Get names from author_id_to_name_dict
            co_author_names = {
                author_id_to_name_dict[author_id] for author_id in co_author_ids
            }

            # Add to all co-authors
            all_co_authors.update(co_author_names)

        # Add new author
        authors_ready_rows.append(
            {
                "author_id": author.author_id,
                "name": author["name"],
                "repos": tuple(repos),
                "co_authors": tuple(all_co_authors),
            }
        )

    # Make frame
    authors_ready = pd.DataFrame(authors_ready_rows)

    # Prep devs and authors details for embedding and then comparison
    prepped_devs_dict = _dataframe_to_joined_str_items(
        unique_devs_df,
        unique_key="username",
        tqdm_desc="Converting dev details to strings...",
    )
    prepped_authors_dict = _dataframe_to_joined_str_items(
        authors_ready,
        unique_key="author_id",
        tqdm_desc="Converting author details to strings...",
        ignore_columns=["author_id"],
    )

    # Create embeddings for each dev
    log.info("Creating embeddings for each dev...")
    model = SentenceTransformer(model_name)
    dev_embeddings = model.encode(
        list(prepped_devs_dict.values()),
        show_progress_bar=True,
    )
    dev_embeddings_dict = {  # noqa: C416
        username: embedding
        for username, embedding in zip(
            prepped_devs_dict.keys(),
            dev_embeddings,
            strict=True,
        )
    }

    # Create embeddings for each author
    log.info("Creating embeddings for each author...")
    author_embeddings = model.encode(
        list(prepped_authors_dict.values()),
        show_progress_bar=True,
    )
    author_embeddings_dict = {  # noqa: C416
        author_id: embedding
        for author_id, embedding in zip(
            prepped_authors_dict.keys(),
            author_embeddings,
            strict=True,
        )
    }

    # For dev, get the authors with shared repos,
    # then take the top n most similar authors
    author_dev_comparison_rows = []
    for username, dev_details in tqdm(
        prepped_devs_dict.items(),
        desc="Constructing author-dev comparison rows",
    ):
        # Get the dev embedding
        dev_embedding = dev_embeddings_dict[username]

        # Get the full dev details from the original dataframe
        full_dev_details = unique_devs_df.loc[unique_devs_df.username == username].iloc[
            0
        ]

        # Get list of repos from full dev details
        repos = full_dev_details.repos

        # Get the authors with matching repos
        shared_authors = set()
        for repo in repos:
            if repo in repo_to_authors:
                shared_authors.update(repo_to_authors[repo])

        if len(shared_authors) == 0:
            continue

        # Get the author embeddings
        author_embeddings = [
            author_embeddings_dict[author_id] for author_id in shared_authors
        ]

        # Get the similarity scores
        similarity_scores = cos_sim(
            dev_embedding,
            author_embeddings,
        )[0].tolist()

        # Get the top n most similar authors
        top_n_similar_authors = sorted(
            (
                (author_id, similarity)
                for author_id, similarity in zip(
                    shared_authors, similarity_scores, strict=True
                )
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n_similar]

        # Get the top n most similar author details
        top_n_similar_author_details = []
        for author_id, similarity in top_n_similar_authors:
            top_n_similar_author_details.append(
                {
                    "similarity": similarity,
                    "details": prepped_authors_dict[author_id],
                }
            )

        # Add to author-dev comparison rows
        for repo_related_author_details in top_n_similar_author_details:
            author_dev_comparison_rows.append(
                {
                    "dev_details": dev_details,
                    "author_details": repo_related_author_details["details"],
                    "similarity": repo_related_author_details["similarity"],
                }
            )

    # Convert to dataframe
    author_dev_comparison_df = pd.DataFrame(author_dev_comparison_rows)

    # Store to disk
    output_filepath = DATA_FILES_DIR / "author-dev-linker-annotation-dataset.csv"
    author_dev_comparison_df.to_csv(output_filepath, index=False)


@app.command()
def create_irr_subset_for_author_dev_linker_annotation(
    n: int = 100,
    debug: bool = False,
) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the author-dev linker annotation dataset
    log.info("Loading author-dev linker annotation dataset...")
    author_dev_linker_annotation_df = pd.read_csv(
        DATA_FILES_DIR / "author-dev-linker-annotation-dataset.csv",
    )

    # Set seed
    np.random.seed(12)

    # Get n random rows
    log.info(f"Getting {n} random rows...")
    random_rows = author_dev_linker_annotation_df.sample(n=n)

    # Store to disk
    output_filepath = DATA_FILES_DIR / "author-dev-linker-annotation-dataset-irr.csv"
    random_rows.to_csv(output_filepath, index=False)


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
        for record_id, score in zip(records, scores, strict=True):  # type: ignore
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

    # TODO: use None for no name and attach "has missing"

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
