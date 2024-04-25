#!/usr/bin/env python

import os
import random
import shutil
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from autotrain.trainers.text_classification.__main__ import train as ft_train
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import tqdm
from transformers import Pipeline, pipeline

from ..data import (
    load_annotated_dev_author_em_dataset,
)

# Models used for testing, both fine-tune and semantic logit
BASE_MODELS = {
    "gte": "thenlper/gte-base",
    "bge": "BAAI/bge-base-en-v1.5",
    "deberta": "microsoft/deberta-v3-base",
    "bert-multilingual": "google-bert/bert-base-multilingual-cased",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bert-uncased": "google-bert/bert-base-uncased",
    "distilbert": "distilbert/distilbert-base-uncased",
}

# Optional fields to create combinations
OPTIONAL_DATA_FIELDS = [
    "dev_name",
    "dev_email",
    "dev_bio",
    "dev_co_contributors",
    "author_co_authors",
]

# Create all combinations
OPTIONAL_DATA_FIELDSETS: list[tuple[str, ...]] = [
    (),  # include no optional data
]
for i in range(1, len(OPTIONAL_DATA_FIELDS) + 1):
    OPTIONAL_DATA_FIELDSETS.extend(list(combinations(OPTIONAL_DATA_FIELDS, i)))

# Fine-tune default settings
DEFAULT_HF_DATASET_PATH = "evamxb/dev-author-em-dataset"
DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH = Path("autotrain-text-classification-temp/")
DEFAULT_MODEL_MAX_SEQ_LENGTH = 256
FINE_TUNE_COMMAND_DICT = {
    "data_path": DEFAULT_HF_DATASET_PATH,
    "project_name": str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
    "text_column": "text",
    "target_column": "label",
    "train_split": "train",
    "valid_split": "valid",
    "epochs": 3,
    "lr": 5e-5,
    "auto_find_batch_size": True,
    "seed": 12,
    "max_seq_length": DEFAULT_MODEL_MAX_SEQ_LENGTH,
}

# Evaluation storage path
EVAL_STORAGE_PATH = Path("model-eval-results/")

###############################################################################


def train_and_eval_all_dev_author_em_classifiers() -> None:  # noqa: C901
    # Load environment variables
    load_dotenv()
    FINE_TUNE_COMMAND_DICT["token"] = os.environ["HF_AUTH_TOKEN"]

    # Basic template for model input
    # We choose username and name because they are always available
    # All other optional fields are added for experimentation because
    # Users may not always have them
    model_str_input_template = """
    # Developer Details

    username: {dev_username}
    {dev_extras}

    ---

    # Author Details

    name: {author_name}
    {author_extras}
    """.strip()

    # Delete prior results and then remake
    shutil.rmtree(EVAL_STORAGE_PATH, ignore_errors=True)
    EVAL_STORAGE_PATH.mkdir(exist_ok=True)

    # Set seed
    np.random.seed(12)
    random.seed(12)

    ###############################################################################

    # Load data
    dev_author_full_details = load_annotated_dev_author_em_dataset()

    # Drop any rows with na
    dev_author_full_details = dev_author_full_details.dropna()

    # Cast semantic_scholar_id to string
    dev_author_full_details["semantic_scholar_id"] = dev_author_full_details[
        "semantic_scholar_id"
    ].astype(str)

    # Load the authors dataset
    # authors = load_author_contributions_dataset()
    authors = pd.DataFrame()

    # Drop everything but the author_id and the name
    authors = authors[["author_id", "name"]].dropna()

    # Load the repos dataset to get to devs
    # repos = load_repo_contributors_dataset()
    repos = pd.DataFrame()

    # Get unique devs by grouping by username
    # and then taking the first email and first "name"
    devs = repos.groupby("username").first().reset_index()

    # For each row in terra, get the matching author and dev
    full_dataset = []
    for _, row in dev_author_full_details.iterrows():
        try:
            # Get the author details
            author_details = authors[
                authors["author_id"] == row["semantic_scholar_id"]
            ].iloc[0]

            # Get the dev details
            dev_details = devs[devs["username"] == row["github_id"]].iloc[0]

            # Add to the dataset
            full_dataset.append(
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
    dev_author_full_details = pd.DataFrame(full_dataset)

    # Store class details required for feature construction
    num_classes = dev_author_full_details["match"].nunique()
    class_labels = list(dev_author_full_details["match"].unique())

    def convert_split_details_to_text_input_dataset(
        df: pd.DataFrame,
        fieldset: tuple[str, ...],
    ) -> tuple[pd.DataFrame, datasets.Dataset]:
        # Construct the model input strings
        rows = []
        for _, row in df.iterrows():
            dev_extras = "\n".join(
                [
                    f"{field.replace('dev_', '')}: {row[field]}"
                    for field in fieldset
                    if field.startswith("dev_")
                ]
            )
            author_extras = "\n".join(
                [
                    f"{field.replace('author_', '')}: {row[field]}"
                    for field in fieldset
                    if field.startswith("author_")
                ]
            )
            model_str_input = model_str_input_template.format(
                dev_username=row["dev_username"],
                dev_extras=dev_extras,
                author_name=row["author_name"],
                author_extras=author_extras,
            )
            rows.append(
                {
                    "text": model_str_input.strip(),
                    "label": row["match"],
                }
            )

        # Construct features for the dataset
        features = datasets.Features(
            text=datasets.Value("string"),
            label=datasets.ClassLabel(
                num_classes=num_classes,
                names=class_labels,
            ),
        )

        # Construct the dataset
        return (
            pd.DataFrame(rows),
            datasets.Dataset.from_pandas(
                pd.DataFrame(rows),
                features=features,
                preserve_index=False,
            ),
        )

    @dataclass
    class EvaluationResults(DataClassJsonMixin):
        random_sample_frac: float
        fieldset: str
        model: str
        accuracy: float
        precision: float
        recall: float
        f1: float
        time_pred: float

    def evaluate(
        model: LogisticRegressionCV | Pipeline,
        x_test: list[np.ndarray] | list[str],
        y_test: list[str],
        model_name: str,
        df: pd.DataFrame,
        fieldset: str,
        random_sample_frac: float,
    ) -> EvaluationResults:
        # Evaluate the model
        print("Evaluating model")

        # Recore perf time
        start_time = time.time()
        y_pred = model.predict(x_test)
        perf_time = (time.time() - start_time) / len(y_test)

        # Get the actual predictions from Pipeline
        if isinstance(model, Pipeline):
            y_pred = [pred["label"] for pred in y_pred]

        # Metrics
        accuracy = accuracy_score(
            y_test,
            y_pred,
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="binary",
            pos_label="match",
        )

        # Print results
        print(
            f"Accuracy: {accuracy}, "
            f"Precision: {precision}, "
            f"Recall: {recall}, "
            f"F1: {f1}, "
            f"Time/Pred: {perf_time}"
        )

        # Create storage dir for model evals
        this_model_eval_storage = EVAL_STORAGE_PATH / (
            "neg-sample-frac-" + str(random_sample_frac * 100)
        )
        this_model_eval_storage.mkdir(exist_ok=True)

        # Create sub-dir for fieldset
        if len(fieldset) == 0:
            fieldset_dir_name = "no-optional-data"
        else:
            fieldset_dir_name = fieldset
        this_model_eval_storage = this_model_eval_storage / fieldset_dir_name
        this_model_eval_storage.mkdir(exist_ok=True)

        this_model_eval_storage = this_model_eval_storage / model_name
        this_model_eval_storage.mkdir(exist_ok=True)

        # Create confusion matrix display
        cm = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
        )

        # Save confusion matrix
        cm.figure_.savefig(this_model_eval_storage / "confusion.png")

        # Add a "predicted" column
        df["predicted"] = y_pred

        # Find rows of misclassifications
        misclassifications = df[df["label"] != df["predicted"]]

        # If "embedding" is in the columns, drop it
        if "embedding" in misclassifications.columns:
            misclassifications = misclassifications.drop(columns=["embedding"])

        # Save misclassifications
        misclassifications.to_csv(
            this_model_eval_storage / "misclassifications.csv",
            index=False,
        )

        return EvaluationResults(
            random_sample_frac=random_sample_frac,
            fieldset=fieldset,
            model=this_iter_model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            time_pred=perf_time,
        )

    # Iter through different dataset sizes
    for random_neg_sample_frac in tqdm(
        [0.1, 0.25, 0.5, 1],
        desc="Random neg sample fracs",
    ):
        print("Randomly sampling negative examples with frac:", random_neg_sample_frac)

        # Get all positive examples
        positive_examples = dev_author_full_details[
            dev_author_full_details["match"] == "match"
        ]

        # Get all negative examples
        negative_examples = dev_author_full_details[
            dev_author_full_details["match"] == "no-match"
        ]

        # Sample negatives with using the random_neg_sample_frac
        sampled_negative_examples = negative_examples.sample(
            frac=random_neg_sample_frac,
            random_state=12,
        )

        # Combine the two
        random_neg_sampled_df = pd.concat(
            [positive_examples, sampled_negative_examples], ignore_index=True
        ).reset_index(drop=True)

        # Create splits
        train_df, test_and_valid_df = train_test_split(
            random_neg_sampled_df,
            test_size=0.4,
            shuffle=True,
            stratify=random_neg_sampled_df["match"],
        )
        valid_df, test_df = train_test_split(
            test_and_valid_df,
            test_size=0.5,
            shuffle=True,
            stratify=test_and_valid_df["match"],
        )

        # Add the negative examples NOT included in the random sample
        # to the test set
        negatives_not_in_sample = negative_examples[
            ~negative_examples.index.isin(sampled_negative_examples.index)
        ]
        test_df = pd.concat(
            [test_df, negatives_not_in_sample], ignore_index=True
        ).reset_index(drop=True)

        # Create a dataframe where the rows are the different splits
        # and there are three columns one column is the split name,
        # the other columns are the counts of match
        split_counts = []
        for split_name, split_df in [
            ("train", train_df),
            ("valid", valid_df),
            ("test", test_df),
        ]:
            split_counts.append(
                {
                    "split": split_name,
                    **split_df["match"].value_counts().to_dict(),
                }
            )
        split_counts_df = pd.DataFrame(split_counts)
        print("Split counts:")
        print(split_counts_df)
        print()

        # Iter through fieldsets
        results = []
        for fieldset in tqdm(
            OPTIONAL_DATA_FIELDSETS,
            desc="Fieldsets",
        ):
            print()
            print(f"Working on fieldset: '{fieldset}'")
            try:
                # Create the datasets
                (
                    fieldset_train_df,
                    fieldset_train_ds,
                ) = convert_split_details_to_text_input_dataset(
                    train_df,
                    fieldset,
                )
                (
                    fieldset_valid_df,
                    fieldset_valid_ds,
                ) = convert_split_details_to_text_input_dataset(
                    valid_df,
                    fieldset,
                )
                (
                    fieldset_test_df,
                    fieldset_test_ds,
                ) = convert_split_details_to_text_input_dataset(
                    test_df,
                    fieldset,
                )

                # Store as dataset dict
                fieldset_ds_dict = datasets.DatasetDict(
                    {
                        "train": fieldset_train_ds,
                        "valid": fieldset_valid_ds,
                        "test": fieldset_test_ds,
                    }
                )

                # Print example input
                print("Example input:")
                print("-" * 20)
                print()
                print(fieldset_train_df.sample(1).iloc[0].text)
                print()
                print("-" * 20)
                print()

                # Push to hub
                print("Pushing this fieldset to hub")
                fieldset_ds_dict.push_to_hub(
                    DEFAULT_HF_DATASET_PATH,
                    private=True,
                    token=os.environ["HF_AUTH_TOKEN"],
                )
                print()
                print()

                # Train each semantic logit model
                for model_short_name, hf_model_path in tqdm(
                    BASE_MODELS.items(),
                    desc="Semantic logit models",
                    leave=False,
                ):
                    # Set seed
                    np.random.seed(12)
                    random.seed(12)

                    this_iter_model_name = f"semantic-logit-{model_short_name}"
                    print()
                    print(f"Working on: {this_iter_model_name}")
                    try:
                        # Init sentence transformer
                        sentence_transformer = SentenceTransformer(hf_model_path)

                        # Preprocess all of the text to embeddings
                        print("Preprocessing text to embeddings")
                        fieldset_train_df["embedding"] = [
                            np.array(embed)
                            for embed in sentence_transformer.encode(
                                fieldset_train_df["text"].tolist(),
                            ).tolist()
                        ]
                        fieldset_test_df["embedding"] = [
                            np.array(embed)
                            for embed in sentence_transformer.encode(
                                fieldset_test_df["text"].tolist(),
                            ).tolist()
                        ]

                        # Train model
                        print("Training model")
                        clf = LogisticRegressionCV(
                            cv=10,
                            max_iter=3000,
                            random_state=12,
                            class_weight="balanced",
                        ).fit(
                            fieldset_train_df["embedding"].tolist(),
                            fieldset_train_df["label"].tolist(),
                        )

                        # Evaluate model
                        results.append(
                            evaluate(
                                clf,
                                fieldset_test_df["embedding"].tolist(),
                                fieldset_test_df["label"].tolist(),
                                this_iter_model_name,
                                fieldset_test_df,
                                "-".join(fieldset),
                                random_neg_sample_frac,
                            ).to_dict(),
                        )

                        print()

                    except Exception as e:
                        print(f"Error during: {this_iter_model_name}, Error: {e}")
                        results.append(
                            {
                                "fieldset": fieldset,
                                "model": this_iter_model_name,
                                "error_level": "semantic model training",
                                "error": str(e),
                            }
                        )

                # Train each fine-tuned model
                for model_short_name, hf_model_path in tqdm(
                    BASE_MODELS.items(),
                    desc="Fine-tune models",
                    leave=False,
                ):
                    # Set seed
                    np.random.seed(12)
                    random.seed(12)

                    this_iter_model_name = f"fine-tune-{model_short_name}"
                    print()
                    print(f"Working on: {this_iter_model_name}")
                    try:
                        # Delete existing temp storage if exists
                        if DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH.exists():
                            shutil.rmtree(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH)

                        # Update the fine-tune command dict
                        this_iter_command_dict = FINE_TUNE_COMMAND_DICT.copy()
                        this_iter_command_dict["model"] = hf_model_path

                        # Train the model
                        ft_train(
                            this_iter_command_dict,
                        )

                        # Evaluate the model
                        ft_transformer_pipe = pipeline(
                            task="text-classification",
                            model=str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
                            tokenizer=str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
                            padding=True,
                            truncation=True,
                            max_length=DEFAULT_MODEL_MAX_SEQ_LENGTH,
                        )
                        results.append(
                            evaluate(
                                ft_transformer_pipe,
                                fieldset_test_df["text"].tolist(),
                                fieldset_test_df["label"].tolist(),
                                this_iter_model_name,
                                fieldset_test_df,
                                "-".join(fieldset),
                                random_neg_sample_frac,
                            ).to_dict(),
                        )

                    except Exception as e:
                        print(f"Error during: {this_iter_model_name}, Error: {e}")
                        results.append(
                            {
                                "fieldset": fieldset,
                                "model": this_iter_model_name,
                                "error_level": "fine-tune model training",
                                "error": str(e),
                            }
                        )

            except Exception as e:
                print(f"Error during: {fieldset}, Error: {e}")
                results.append(
                    {
                        "fieldset": fieldset,
                        "error_level": "dataset creation",
                        "error": str(e),
                    }
                )

            print()

            # Print results
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by="f1", ascending=False).reset_index(
                drop=True
            )
            results_df.to_csv("all-model-results.csv", index=False)
            print("Current standings")
            print(
                tabulate(
                    results_df.head(10),
                    headers="keys",
                    tablefmt="psql",
                    showindex=False,
                )
            )

            print()

    print()
    print("-" * 80)
    print()

    # Print results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False).reset_index(drop=True)
    results_df.to_csv("all-model-results.csv", index=False)
    print("Final standings")
    print(
        tabulate(
            results_df.head(10),
            headers="keys",
            tablefmt="psql",
            showindex=False,
        )
    )
