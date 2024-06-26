#!/usr/bin/env python

import logging
import random
from uuid import uuid4

import numpy as np
import pandas as pd
import typer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sqlmodel import Session, select
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from tqdm import tqdm

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import (
    DATA_FILES_DIR,
    load_annotated_dev_author_em_dataset,
    load_author_contributions_dataset,
    load_multi_annotator_dev_author_em_irr_dataset,
    load_repo_contributors_dataset,
)
from rs_graph.db import models as db_models
from rs_graph.db import utils as db_utils

# load_repo_contributors_dataset,
from rs_graph.ml.dev_author_em_clf import (
    DEV_AUTHOR_EM_EMBEDDING_MODEL,
    DEV_AUTHOR_EM_TEMPLATE,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


def _get_unique_devs_frame_from_dev_contributions(
    dev_contributions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    # Each row should be a unique person
    log.info("Getting unique developers frame...")

    # Make repo to dev lut
    repo_to_devs_lut: dict[str, set[str]] = {}
    for _, dev in dev_contributions_df.iterrows():
        for repo in dev.repo:
            if repo not in repo_to_devs_lut:
                repo_to_devs_lut[repo] = set()

            repo_to_devs_lut[repo].add(dev.username)

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
    return pd.DataFrame(unique_devs), repo_to_devs_lut


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
def create_developer_author_em_dataset_for_annotation(  # noqa: C901
    top_n_similar: int = 3,
    seed: int = 12,
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    debug: bool = False,
) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Load the repo contributors dataset
    log.info("Loading developer contributions dataset...")
    # devs = load_repo_contributors_dataset()
    devs = pd.DataFrame()

    # Get unique devs frame
    unique_devs_df, _ = _get_unique_devs_frame_from_dev_contributions(
        devs,
    )

    # Load the author contributions dataset
    log.info("Loading author contributions dataset...")
    # authors = load_author_contributions_dataset()
    authors = pd.DataFrame()

    # Drop authors without an author_id
    authors = authors.dropna(subset=["author_id"])

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
    dev_author_comparison_rows = []
    for username, dev_details in tqdm(
        prepped_devs_dict.items(),
        desc="Constructing dev-author comparison rows",
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
                    "author_id": author_id,
                    "similarity": similarity,
                    "details": prepped_authors_dict[author_id],
                }
            )

        # Add to dev-author comparison rows
        for repo_related_author_details in top_n_similar_author_details:
            dev_author_comparison_rows.append(
                {
                    "github_id": full_dev_details.username,
                    "semantic_scholar_id": repo_related_author_details["author_id"],
                    "dev_details": dev_details,
                    "author_details": repo_related_author_details["details"],
                    "similarity": repo_related_author_details["similarity"],
                }
            )

    # Convert to dataframe
    dev_author_comparison_df = pd.DataFrame(dev_author_comparison_rows)

    # Remove rows with no semantic scholar id
    dev_author_comparison_df = dev_author_comparison_df.dropna(
        subset=["semantic_scholar_id"]
    )

    # Shuffle the dataframe
    dev_author_comparison_df = dev_author_comparison_df.sample(
        frac=1,
        random_state=seed,
    )

    # Store to disk
    output_filepath = "dev-author-em-annotation-dataset.csv"
    dev_author_comparison_df.to_csv(output_filepath, index=False)
    log.info(f"Stored to {output_filepath}")


@app.command()
def create_practice_subset_for_dev_author_em_annotation(
    full_annotation_dataset_path: str = "dev-author-em-annotation-dataset.csv",
    n: int = 100,
    debug: bool = False,
) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the dev-author EM annotation dataset
    log.info("Loading dev-author EM annotation dataset...")
    dev_author_em_annotation_df = pd.read_csv(full_annotation_dataset_path)

    # Set seed
    np.random.seed(12)

    # Get n random rows
    log.info(f"Getting {n} random rows...")
    random_rows = dev_author_em_annotation_df.sample(n=n)

    # Store to disk
    output_filepath = "dev-author-em-annotation-dataset-irr.csv"
    random_rows.to_csv(output_filepath, index=False)
    log.info(f"Stored to {output_filepath}")


def _interpret_fliess_kappa(v: float) -> str:
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


def _print_irr_stats(
    irr_df: pd.DataFrame,
    compare_col_one: str,
    compare_col_two: str,
    include_diff_rows: bool = False,
) -> None:
    # Make a frame of _just_ the annotations
    annotations = irr_df[["match_terra", "match_akhil"]]

    # Aggregate
    agg_raters, _ = aggregate_raters(annotations)

    # Calc Kappa's and return
    kappa = fleiss_kappa(agg_raters)

    # Run print
    print(
        f"Inter-rater Reliability (Fleiss Kappa): "
        f"{kappa} ({_interpret_fliess_kappa(kappa)})"
    )
    print()

    if include_diff_rows:
        print("Differing Labels:")
        print("-" * 80)
        diff_rows = irr_df.loc[(irr_df["match_terra"] != irr_df["match_akhil"])]
        for _, row in diff_rows.iterrows():
            print(f"{compare_col_one}:")
            for line in row[compare_col_one].split("\n"):
                print(f"\t{line}")
            print()
            print(f"{compare_col_two}:")
            for line in row[compare_col_two].split("\n"):
                print(f"\t{line}")
            print()
            print("labels:")
            print(row[["match_terra", "match_akhil"]])
            print()
            print()
            print("-" * 40)
            print()


@app.command()
def calculate_irr_for_dev_author_em_annotation(
    use_full_dataset: bool = False,
    include_diff_rows: bool = False,
    debug: bool = False,
) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the dev-author EM annotation dataset
    log.info("Loading dev-author EM annotation dataset...")
    irr_df = load_multi_annotator_dev_author_em_irr_dataset(use_full_dataset)

    _print_irr_stats(
        irr_df=irr_df,
        compare_col_one="dev_details",
        compare_col_two="author_details",
        include_diff_rows=include_diff_rows,
    )


@app.command()
def train_and_eval_all_dev_author_em_classifiers(debug: bool = False) -> None:
    from rs_graph.ml.train_and_eval_all_dev_author_em_clf import (
        train_and_eval_all_dev_author_em_classifiers,
    )

    # Setup logging
    setup_logger(debug=debug)

    # Train and evaluate all dev-author EM classifiers
    train_and_eval_all_dev_author_em_classifiers()


@app.command()
def train_dev_author_em_classifier(
    base_model_name: str = DEV_AUTHOR_EM_EMBEDDING_MODEL,
    test_size: float = 0.15,
    seed: int = 12,
    confusion_matrix_save_name: str = "dev-author-em-confusion-matrix.png",
    misclassifications_save_name: str = "dev-author-em-misclassifications.csv",
    debug: bool = False,
) -> None:
    import datasets
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        pipeline,
    )

    # Setup logging
    setup_logger(debug=debug)

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Load the datasets
    dev_author_full_details = load_annotated_dev_author_em_dataset()

    # Cast semantic_scholar_id to string
    dev_author_full_details["semantic_scholar_id"] = dev_author_full_details[
        "semantic_scholar_id"
    ].astype(str)

    # Load the authors dataset
    authors = load_author_contributions_dataset()

    # Drop everything but the author_id and the name
    authors = authors[["author_id", "name"]].dropna()

    # Load the repos dataset to get to devs
    repos = load_repo_contributors_dataset()

    # Get unique devs by grouping by username
    # and then taking the first email and first "name"
    devs = repos.groupby("username").first().reset_index()

    # For each row in terra, get the matching author and dev
    log.info("Getting expanded details for each author and dev...")
    constructed_rows = []
    for _, row in dev_author_full_details.iterrows():
        try:
            # Get the author details
            author_details = authors[
                authors["author_id"] == row["semantic_scholar_id"]
            ].iloc[0]

            # Get the dev details
            dev_details = devs[devs["username"] == row["github_id"]].iloc[0]

            # Add to the dataset
            constructed_rows.append(
                {
                    "dev_username": row["github_id"],
                    "dev_name": dev_details["name"],
                    "dev_email": dev_details["email"],
                    "author_id": row["semantic_scholar_id"],
                    "author_name": author_details["name"],
                    "match": "match" if row["match"] else "no-match",
                }
            )
        except Exception:
            pass

    # Convert to dataframe
    expanded_details = pd.DataFrame(constructed_rows)

    # Create embedding ready strings for each dev-author pair
    log.info("Constructing dev-author comparison strings...")
    prepped_rows = []
    for _, row in expanded_details.iterrows():
        prepped_rows.append(
            {
                "text": DEV_AUTHOR_EM_TEMPLATE.format(
                    dev_username=row["dev_username"],
                    dev_name=row["dev_name"],
                    dev_email=row["dev_email"],
                    author_name=row["author_name"],
                ),
                "label": row["match"],
            }
        )

    # Convert to dataframe
    prepped_data = pd.DataFrame(prepped_rows)

    # Get n classes and labels
    num_classes = prepped_data["label"].nunique()
    class_labels = prepped_data["label"].unique().tolist()

    # Create splits
    log.info("Creating train and test splits...")
    train_df, test_df = train_test_split(
        prepped_data,
        stratify=prepped_data["label"].tolist(),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    # Log split counts
    split_counts = []
    for split_name, split_df in [
        ("train", train_df),
        ("test", test_df),
    ]:
        split_counts.append(
            {
                "split": split_name,
                **split_df["label"].value_counts().to_dict(),
            }
        )
    split_counts_df = pd.DataFrame(split_counts)
    print("Split counts:")
    print(split_counts_df)
    print()

    # Construct features for the dataset
    features = datasets.Features(
        text=datasets.Value("string"),
        label=datasets.ClassLabel(
            num_classes=num_classes,
            names=class_labels,
        ),
    )

    # Create dataset dict
    ds_dict = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_pandas(
                train_df,
                features=features,
                preserve_index=False,
            ),
            "test": datasets.Dataset.from_pandas(
                test_df,
                features=features,
                preserve_index=False,
            ),
        }
    )

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_ds_dict = ds_dict.map(tokenize_function, batched=True)

    # Construct label to id and vice-versa LUTs
    label2id, id2label = {}, {}
    for i, label in enumerate(class_labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Create Training Args and Trainer
    training_args = TrainingArguments(
        output_dir="dev-author-em-clf",
        overwrite_output_dir=True,
        num_train_epochs=2,
        learning_rate=5e-5,
        logging_steps=20,
        auto_find_batch_size=True,
        seed=12,
    )
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_ds_dict["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    trainer.train()
    trainer.save_model("dev-author-em-clf")  # TODO: Update to use an official name

    # Load model from dir as pipeline
    trained_clf = pipeline(
        "text-classification",
        model="dev-author-em-clf/",
    )

    # Evaluate
    log.info("Evaluating the classifier...")
    y_pred = [pred["label"] for pred in trained_clf(test_df["text"].tolist())]
    accuracy = accuracy_score(test_df["label"].tolist(), y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["label"].tolist(),
        y_pred,
        average="binary",
        pos_label="match",
    )

    # Print results
    print(
        f"Evaluation results -- "
        f"Accuracy: {accuracy:.3f}, "
        f"Precision: {precision:.3f}, "
        f"Recall: {recall:.3f}, "
        f"F1: {f1:.3f}"
    )

    # Create confusion matrix and ROC curve
    confusion_matrix = ConfusionMatrixDisplay.from_predictions(
        test_df["label"].tolist(),
        y_pred,
    )
    confusion_matrix_save_path = DATA_FILES_DIR / confusion_matrix_save_name
    confusion_matrix.figure_.savefig(confusion_matrix_save_path)

    # Add predicted values to test_df to find misclassifications
    test_df["predicted"] = y_pred
    misclassifications = test_df.loc[test_df["label"] != test_df["predicted"]]

    # Store misclassifications
    # Drop the embedding column
    misclassifications_save_path = DATA_FILES_DIR / misclassifications_save_name
    misclassifications.to_csv(misclassifications_save_path, index=False)


def _create_dataset_for_document_repository_training(
    prod: bool = False,
) -> pd.DataFrame:
    # Get DB engine
    engine = db_utils.get_engine(prod=prod)

    # Start session
    rows = []
    with Session(engine) as session:
        # Create a dataframe of:
        # "document_id", "document_doi", "document_title", "document_abstract"
        # "document_publication_date", "document_contributor_names",
        # "repository_id", "repository_name",
        # "repository_description", "repository_creation_date",
        # "repository_contributor_usernames", "repository_contributor_names",
        # "repository_contributor_emails"
        # "match"
        stmt = (
            select(
                db_models.Document,
                db_models.DocumentAbstract,
                db_models.DocumentRepositoryLink,
                db_models.Repository,
                db_models.RepositoryReadme,
            )
            .join(
                db_models.DocumentAbstract,
                db_models.DocumentAbstract.document_id == db_models.Document.id,
            )
            .join(
                db_models.DocumentRepositoryLink,
                db_models.DocumentRepositoryLink.document_id == db_models.Document.id,
            )
            .join(
                db_models.Repository,
                db_models.DocumentRepositoryLink.repository_id
                == db_models.Repository.id,
            )
            .join(
                db_models.RepositoryReadme,
                db_models.RepositoryReadme.repository_id == db_models.Repository.id,
            )
        )

        # Iter data
        for doc, doc_abstract, link, repo, repo_readme in session.exec(stmt):
            # Get doc contributors for this doc
            doc_contribs_statement = (
                select(
                    db_models.Researcher,
                )
                .join(
                    db_models.DocumentContributor,
                )
                .where(
                    db_models.DocumentContributor.document_id == doc.id,
                )
            )
            doc_contrib_names = []
            for researcher in session.exec(doc_contribs_statement):
                doc_contrib_names.append(researcher.name.strip())

            # Get repo contributors for this repo
            repo_contribs_statement = (
                select(
                    db_models.DeveloperAccount,
                )
                .join(
                    db_models.RepositoryContributor,
                )
                .where(
                    db_models.RepositoryContributor.repository_id == repo.id,
                )
            )
            repo_contrib_usernames = []
            repo_contrib_names = []
            repo_contrib_emails = []
            for dev_account in session.exec(repo_contribs_statement):
                repo_contrib_usernames.append(dev_account.username.strip())
                if dev_account.name is not None:
                    repo_contrib_names.append(dev_account.name.strip())
                if dev_account.email is not None:
                    repo_contrib_emails.append(dev_account.email.strip())

            # Remove Nones
            repo_contrib_names = [v for v in repo_contrib_names if v is not None]
            repo_contrib_emails = [v for v in repo_contrib_emails if v is not None]

            # Add to rows
            rows.append(
                {
                    "dataset_source": link.dataset_source_id,
                    "document_id": doc.id,
                    "document_doi": doc.doi,
                    "document_title": doc.title,
                    "document_abstract": doc_abstract.content,
                    "document_publication_date": doc.publication_date.isoformat(),
                    "document_contributor_names": ", ".join(doc_contrib_names),
                    "repository_id": repo.id,
                    "repository_name": repo.name,
                    "repository_readme": repo_readme.content,
                    "repository_description": repo.description,
                    "repository_creation_date": (
                        repo.creation_datetime.date().isoformat()
                    ),
                    "repository_contributor_usernames": ", ".join(
                        repo_contrib_usernames
                    ),
                    "repository_contributor_names": ", ".join(repo_contrib_names),
                    "repository_contributor_emails": ", ".join(repo_contrib_emails),
                    "match": "match",
                }
            )

    # Convert to dataframe
    df = pd.DataFrame(rows)

    # For every row, create a random negative sample
    # using the repository details of another row
    negative_samples = []
    for _, row in df.iterrows():
        # Get a random row
        random_row = df.sample(n=1).iloc[0]

        # Add to negative samples
        negative_samples.append(
            {
                "dataset_source": -1,
                "document_id": row["document_id"],
                "document_doi": row["document_doi"],
                "document_title": row["document_title"],
                "document_abstract": row["document_abstract"],
                "document_publication_date": row["document_publication_date"],
                "document_contributor_names": row["document_contributor_names"],
                "repository_id": random_row["repository_id"],
                "repository_name": random_row["repository_name"],
                "repository_readme": random_row["repository_readme"],
                "repository_description": random_row["repository_description"],
                "repository_creation_date": random_row["repository_creation_date"],
                "repository_contributor_usernames": random_row[
                    "repository_contributor_usernames"
                ],
                "repository_contributor_names": random_row[
                    "repository_contributor_names"
                ],
                "repository_contributor_emails": random_row[
                    "repository_contributor_emails"
                ],
                "match": "no-match",
            }
        )

    # Convert to dataframe
    negative_df = pd.DataFrame(negative_samples)

    # Concatenate the two dataframes
    full_df = pd.concat([df, negative_df])

    # Shuffle the dataframe
    full_df = full_df.sample(frac=1).reset_index(drop=True)

    return full_df


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
