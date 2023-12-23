#!/usr/bin/env python

import logging
from uuid import uuid4

import numpy as np
import pandas as pd
import typer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from skops import io as sk_io
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from tqdm import tqdm

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import (
    load_annotated_author_dev_em_dataset,
    load_annotated_author_dev_em_irr_dataset,
    load_author_contributions_dataset,
    load_repo_contributors_dataset,
)
from rs_graph.modeling import (
    AUTHOR_DEV_EM_MODEL_PATH,
    DEFAULT_AUTHOR_DEV_EMBEDDING_MODEL_NAME,
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
        # Convert to dict and convert None values to "None" string
        details = row.to_dict()
        for key, value in details.items():
            if value is None:
                details[key] = str(value)

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


@app.command()
def create_author_developer_em_dataset_for_annotation(  # noqa: C901
    top_n_similar: int = 3,
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    debug: bool = False,
) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the repo contributors dataset
    log.info("Loading developer contributions dataset...")
    devs = load_repo_contributors_dataset()

    # Get unique devs frame
    unique_devs_df = _get_unique_devs_frame_from_dev_contributions(
        devs,
    )

    # Load the author contributions dataset
    log.info("Loading author contributions dataset...")
    authors = load_author_contributions_dataset()

    # Create lookup for repo to authors
    repo_to_authors: dict[str, set[str]] = {}
    author_id_to_name_dict: dict[str, str] = {}
    remade_authors = []
    for _, author in authors.iterrows():
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

        # Remove self from co-authors
        all_co_authors.discard(author["name"])

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

    # Only include the username, name, repos, and co_contributors columns
    # in the unique devs dataframe
    unique_devs_df = unique_devs_df[
        ["username", "name", "email", "repos", "co_contributors"]
    ].copy()

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
    output_filepath = "author-dev-em-annotation-dataset.csv"
    author_dev_comparison_df.to_csv(output_filepath, index=False)
    log.info(f"Stored to {output_filepath}")


@app.command()
def create_irr_subset_for_author_dev_em_annotation(
    full_annotation_dataset_path: str = "author-dev-em-annotation-dataset.csv",
    n: int = 100,
    debug: bool = False,
) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the author-dev EM annotation dataset
    log.info("Loading author-dev EM annotation dataset...")
    author_dev_em_annotation_df = pd.read_csv(full_annotation_dataset_path)

    # Set seed
    np.random.seed(12)

    # Get n random rows
    log.info(f"Getting {n} random rows...")
    random_rows = author_dev_em_annotation_df.sample(n=n)

    # Store to disk
    output_filepath = "author-dev-em-annotation-dataset-irr.csv"
    random_rows.to_csv(output_filepath, index=False)
    log.info(f"Stored to {output_filepath}")


@app.command()
def calculate_irr_for_author_dev_em_annotation(
    debug: bool = False,
) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the author-dev EM annotation dataset
    log.info("Loading author-dev EM annotation dataset...")
    irr_df = load_annotated_author_dev_em_irr_dataset()

    # Make a frame of _just_ the annotations
    annotations = irr_df[["eva-match", "lindsey-match", "nic-match"]]

    # Aggregate
    agg_raters, _ = aggregate_raters(annotations)

    # Calc Kappa's and return
    kappa = fleiss_kappa(agg_raters)

    def _interpret_score(v: float) -> str:
        if v < 0:
            return "No agreement"
        if v < 0.2:
            return "Poor agreement"
        if v >= 0.2 and v < 0.4:
            return "Fair agreement"
        if v >= 0.4 and v < 0.6:
            return "Moderate agreement"
        if v >= 0.6 and v < 0.8:
            return "Substantial agreement"
        return "Almost perfect agreement"

    # Run print
    print(
        f"Inter-rater Reliability (Fleiss Kappa): "
        f"{kappa} ({_interpret_score(kappa)})"
    )
    print()

    print("Differing Labels:")
    print("-" * 80)
    diff_rows = irr_df.loc[
        (
            (irr_df["eva-match"] != irr_df["lindsey-match"])
            | (irr_df["eva-match"] != irr_df["nic-match"])
        )
    ]
    for _, row in diff_rows.iterrows():
        print("dev_details:")
        for line in row.dev_details.split("\n"):
            print(f"\t{line}")
        print()
        print("author_details:")
        for line in row.author_details.split("\n"):
            print(f"\t{line}")
        print()
        print("labels:")
        print(row[["eva-match", "lindsey-match", "nic-match"]])
        print()
        print()
        print("-" * 40)
        print()


def _from_details_to_dicts(
    dev_details: str,
    author_details: str,
    model: SentenceTransformer,
) -> dict[str, float]:
    # Split into items then into dict
    dev_items = dev_details.replace("\n", "").split(";")
    author_items = author_details.replace("\n", "").split(";")
    dev_dict = {
        pair[0].strip(): pair[1].strip()
        for pair in [item.split(":", 1) for item in dev_items]
        if len(pair) == 2
    }
    author_dict = {
        pair[0].strip(): pair[1].strip()
        for pair in [item.split(":", 1) for item in author_items]
        if len(pair) == 2
    }

    # Remove "repos" keys
    dev_dict.pop("repos", None)
    author_dict.pop("repos", None)

    # Create embedding for each part
    dev_values_embeddings = model.encode(
        list(dev_dict.values()),
        show_progress_bar=False,
    ).tolist()
    author_values_embeddings = model.encode(
        list(author_dict.values()),
        show_progress_bar=False,
    ).tolist()
    dev_embedding_dict = dict(
        zip(
            dev_dict.keys(),
            dev_values_embeddings,
            strict=True,
        )
    )
    author_embedding_dict = dict(
        zip(
            author_dict.keys(),
            author_values_embeddings,
            strict=True,
        )
    )

    # Create interactions by creating pairwise combinations
    this_x = {}
    for dev_key, dev_value in dev_embedding_dict.items():
        for author_key, author_value in author_embedding_dict.items():
            pairwise_key = f"dev-{dev_key}---author-{author_key}"
            for i, dev_i_v in enumerate(dev_value):
                this_x[f"{pairwise_key}---dim-{i}"] = dev_i_v * author_value[i]

    return this_x


@app.command()
def train_author_dev_em_classifier(
    embedding_model_name: str = DEFAULT_AUTHOR_DEV_EMBEDDING_MODEL_NAME,
    debug: bool = False,
) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the author-dev EM annotation dataset
    log.info("Loading author-dev EM annotation dataset...")
    annotated_data = load_annotated_author_dev_em_dataset()

    # Print value counts of "match"
    print("Match value counts: ")
    print(annotated_data.match.value_counts())
    print()

    # Init embedding model
    embedding_model = SentenceTransformer(embedding_model_name)

    # Process each row
    proccessed_features = []
    for _, row in tqdm(
        annotated_data.iterrows(),
        total=len(annotated_data),
        desc="Getting embedding features for each annotated pair",
    ):
        # Add to list
        proccessed_features.append(
            _from_details_to_dicts(
                row.dev_details,
                row.author_details,
                embedding_model,
            )
        )

    # Create dataframe and copy over labels
    prepped_df = pd.DataFrame(proccessed_features)
    prepped_df["match"] = annotated_data.match

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(
        prepped_df.drop(columns=["match"]),
        prepped_df.match,
        test_size=0.2,
        stratify=prepped_df.match,
        shuffle=True,
        random_state=12,
    )

    # Train model
    log.info("Training classifier...")
    classifier = LogisticRegressionCV(
        cv=10,
        max_iter=1000,
        random_state=12,
        class_weight="balanced",
    ).fit(x_train, y_train)

    # Save model
    log.info("Saving classifier...")
    sk_io.dump(classifier, AUTHOR_DEV_EM_MODEL_PATH)

    # Evaluate model
    log.info("Evaluating classifier...")
    y_pred = classifier.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    log.info(f"pre: {precision}, " f"rec: {recall}, " f"f1: {f1}")

    # Get confusion matrix plot
    confusion = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    # Save confusion matrix plot
    matrix_save_path = "author-dev-em-confusion-matrix.png"
    confusion.figure_.savefig(matrix_save_path)
    log.info(f"Confusion matrix saved to {matrix_save_path}")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
