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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from tabulate import tabulate
from tqdm import tqdm
from transformers import Pipeline, pipeline

from ..data import (
    load_annotated_dev_author_em_dataset,
    load_author_contributions_dataset,
    load_repo_contributors_dataset,
)

# Models used for testing, both fine-tune and semantic logit
BASE_MODELS = {
    "deberta": "microsoft/deberta-v3-base",
    "bert-multilingual": "google-bert/bert-base-multilingual-cased",
    "distilbert": "distilbert/distilbert-base-uncased",
}

# Optional fields to create combinations
OPTIONAL_DATA_FIELDS = [
    "dev_name",
    "dev_email",
]

# Holdout authors and devs sample size
HOLDOUT_SAMPLE_SIZE = 0.2

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
    # "valid_split": "valid",
    "epochs": 2,
    "lr": 5e-5,
    "auto_find_batch_size": True,
    "seed": 12,
    "max_seq_length": DEFAULT_MODEL_MAX_SEQ_LENGTH,
    "logging_steps": 10,
}

# Evaluation storage path
EVAL_STORAGE_PATH = Path("model-eval-results/")

MODEL_STR_INPUT_TEMPLATE = """
<comparison-set>
\t<developer-details>
\t\t<username>{dev_username}</username>{dev_extras}
\t</developer-details>
\t<author-details>
\t\t<name>{author_name}</name>
\t</author-details>
</comparison-set>
""".strip()


###############################################################################


def train_and_eval_all_dev_author_em_classifiers() -> None:  # noqa: C901
    # Load environment variables
    load_dotenv()
    FINE_TUNE_COMMAND_DICT["token"] = os.environ["HF_AUTH_TOKEN"]

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
    authors = load_author_contributions_dataset()

    # Drop everything but the author_id and the name
    authors = authors[["author_id", "name"]].dropna()

    # Load the repos dataset to get to devs
    repos = load_repo_contributors_dataset()

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
                    "dev_bio": dev_details["bio"],
                    "author_id": row["semantic_scholar_id"],
                    "author_name": author_details["name"],
                    "match": "match" if row["match"] else "no-match",
                }
            )
        except Exception:
            pass

    # Convert to dataframe
    dev_author_full_details = pd.DataFrame(full_dataset)

    # Create test set holdout by selecting 10% random unique devs and authors
    # and then adding all of their comparisons to the test set
    unique_devs = pd.Series(dev_author_full_details["dev_username"].unique())
    unique_authors = pd.Series(dev_author_full_details["author_id"].unique())
    test_devs = unique_devs.sample(
        frac=HOLDOUT_SAMPLE_SIZE,
        random_state=12,
        replace=False,
    )
    test_authors = unique_authors.sample(
        frac=HOLDOUT_SAMPLE_SIZE,
        random_state=12,
        replace=False,
    )
    train_rows = []
    test_rows = []
    for _, row in dev_author_full_details.iterrows():
        if (
            row["dev_username"] in test_devs.values
            or row["author_id"] in test_authors.values
        ):
            test_rows.append(row)
        else:
            train_rows.append(row)

    # Create train and test sets
    train_set = pd.DataFrame(train_rows)
    test_set = pd.DataFrame(test_rows)

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
            dev_extras = "\n\t\t".join(
                [
                    f"<{field.replace('dev_', '')}>"
                    f"{row[field]}"
                    f"</{field.replace('dev_', '')}>"
                    for field in fieldset
                    if field.startswith("dev_")
                ]
            )
            dev_extras = f"\n\t\t{dev_extras}" if len(dev_extras) > 0 else ""
            author_extras = "\n\t\t".join(
                [
                    f"<{field.replace('author_', '')}>"
                    f"{row[field]}"
                    f"</{field.replace('author_', '')}>"
                    for field in fieldset
                    if field.startswith("author_")
                ]
            )
            author_extras = f"\n\t\t{author_extras}" if len(author_extras) > 0 else ""
            model_str_input = MODEL_STR_INPUT_TEMPLATE.format(
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

        # Create sub-dir for fieldset
        if len(fieldset) == 0:
            fieldset_dir_name = "no-optional-data"
        else:
            fieldset_dir_name = fieldset

        # Create storage dir for model evals
        this_model_eval_storage = EVAL_STORAGE_PATH / fieldset_dir_name
        this_model_eval_storage.mkdir(exist_ok=True)

        # Model short name
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
            fieldset=fieldset,
            model=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            time_pred=perf_time,
        )

    # Create a dataframe where the rows are the different splits
    # and there are three columns one column is the split name,
    # the other columns are the counts of match
    split_counts = []
    for split_name, split_df in [
        ("train", train_set),
        # ("valid", valid_df),
        ("test", test_set),
    ]:
        split_counts.append(
            {
                "split": split_name,
                **split_df["match"].value_counts().to_dict(),
                **{
                    f"{k}%": v
                    for k, v in split_df["match"]
                    .value_counts(normalize=True)
                    .to_dict()
                    .items()
                },
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
                train_set,
                fieldset,
            )
            # (
            #     fieldset_valid_df,
            #     fieldset_valid_ds,
            # ) = convert_split_details_to_text_input_dataset(
            #     valid_df,
            #     fieldset,
            # )
            (
                fieldset_test_df,
                fieldset_test_ds,
            ) = convert_split_details_to_text_input_dataset(
                test_set,
                fieldset,
            )

            # Store as dataset dict
            fieldset_ds_dict = datasets.DatasetDict(
                {
                    "train": fieldset_train_ds,
                    # "valid": fieldset_valid_ds,
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

            # Train each fine-tuned model
            for model_short_name, hf_model_path in tqdm(
                BASE_MODELS.items(),
                desc="Fine-tune models",
                leave=False,
            ):
                # Set seed
                np.random.seed(12)
                random.seed(12)

                print()
                print(f"Working on: {model_short_name}")
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
                            model_short_name,
                            fieldset_test_df,
                            "-".join(fieldset),
                        ).to_dict(),
                    )

                except Exception as e:
                    print(f"Error during: {model_short_name}, Error: {e}")
                    results.append(
                        {
                            "fieldset": fieldset,
                            "model": model_short_name,
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
