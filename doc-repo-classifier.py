#!/usr/bin/env python

import os
import random
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from autotrain.trainers.text_classification.__main__ import train as ft_train
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import Pipeline, pipeline

from rs_graph.bin.modeling import _create_dataset_for_document_repository_training

# Fine-tune default settings
BASE_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
DEFAULT_HF_DATASET_PATH = "evamxb/doc-repo-em-dataset"
DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH = Path("autotrain-text-classification-temp/")
FINE_TUNE_COMMAND_DICT = {
    "data_path": DEFAULT_HF_DATASET_PATH,
    "project_name": str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
    "text_column": "text",
    "target_column": "label",
    "train_split": "train",
    "valid_split": "valid",
    "epochs": 7,
    "lr": 2e-5,
    "auto_find_batch_size": True,
    "seed": 42,
}

# Evaluation storage path
EVAL_STORAGE_PATH = Path("model-eval-results/")

###############################################################################

# Load environment variables
load_dotenv()
FINE_TUNE_COMMAND_DICT["token"] = os.environ["HF_AUTH_TOKEN"]

fill_template = """
### Article Details

Title: {article_title}
Abstract: {article_abstract}
Publication Date: {article_publication_date}
Contributor Names: {article_contributor_names}

---

### Repository Details

Name: {repository_name}
Description: {repository_description}
Creation Date: {repository_creation_date}
Contributor Usernames: {repository_contributor_usernames}
Contributor Names: {repository_contributor_names}
Contributor Emails: {repository_contributor_emails}
""".strip()


def _train() -> None:
    df = _create_dataset_for_document_repository_training()
    df["text"] = df.apply(
        lambda row: fill_template.format(
            article_title=str(row["document_title"]),
            article_abstract=str(row["document_abstract"]),
            article_publication_date=str(row["document_publication_date"]),
            article_contributor_names=str(row["document_contributor_names"]),
            repository_name=str(row["repository_name"]),
            repository_description=str(row["repository_description"]),
            repository_creation_date=str(row["repository_creation_date"]),
            repository_contributor_usernames=str(
                row["repository_contributor_usernames"]
            ),
            repository_contributor_names=str(row["repository_contributor_names"]),
            repository_contributor_emails=str(row["repository_contributor_emails"]),
        ),
        axis=1,
    )

    # Drop everything but dataset_source_id, document_id, repository_id, text, and match
    # rename match to label
    prepped_df = df[
        ["dataset_source", "document_id", "repository_id", "text", "match"]
    ].copy()
    prepped_df = prepped_df.rename(columns={"match": "label"})

    # Save out dataset
    prepped_df.to_csv("prepped-dataset.csv", index=False)

    train_df, valid_and_test_df = train_test_split(
        prepped_df,
        test_size=0.2,
        random_state=42,
        stratify=prepped_df["label"],
    )
    valid_df, test_df = train_test_split(
        valid_and_test_df,
        test_size=0.75,
        random_state=42,
        stratify=valid_and_test_df["label"],
    )

    # Set seed
    np.random.seed(42)
    random.seed(42)

    def convert_split_details_to_text_input_dataset(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, datasets.Dataset]:
        # Construct features for the dataset
        features = datasets.Features(
            text=datasets.Value("string"),
            label=datasets.ClassLabel(
                num_classes=df.label.nunique(),
                names=list(df.label.unique()),
            ),
        )

        # Construct the dataset
        return (
            df,
            datasets.Dataset.from_pandas(
                df,
                features=features,
                preserve_index=False,
            ),
        )

    @dataclass
    class EvaluationResults(DataClassJsonMixin):
        accuracy: float
        precision: float
        recall: float
        f1: float

    def evaluate(
        model: Pipeline,
        x_test: list[np.ndarray] | list[str],
        y_test: list[str],
        df: pd.DataFrame,
    ) -> tuple[EvaluationResults, ConfusionMatrixDisplay, pd.DataFrame]:
        # Evaluate the model
        print("Evaluating model")
        y_pred = model.predict(x_test)

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
        )

        # Create confusion matrix display
        cm = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
        )

        # Add a "predicted" column
        df["predicted"] = y_pred

        # Find rows of misclassifications
        misclassifications = df[df["label"] != df["predicted"]]

        return (
            EvaluationResults(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
            ),
            cm,
            misclassifications,
        )

    # Create the datasets
    (
        fieldset_train_df,
        fieldset_train_ds,
    ) = convert_split_details_to_text_input_dataset(
        train_df[["text", "label"]],
    )
    (_, fieldset_valid_ds) = convert_split_details_to_text_input_dataset(
        valid_df[["text", "label"]],
    )
    (fieldset_test_df, fieldset_test_ds) = convert_split_details_to_text_input_dataset(
        test_df[["text", "label"]],
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
    example = fieldset_train_df.sample(1).iloc[0]
    print(example.text)
    print()
    print()
    print("Label:", example.label)
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

    # Set seed
    np.random.seed(42)
    random.seed(42)

    # Update the fine-tune command dict
    this_iter_command_dict = FINE_TUNE_COMMAND_DICT.copy()
    this_iter_command_dict["model"] = BASE_MODEL

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
    )

    results, cm, misclassifications = evaluate(
        ft_transformer_pipe,
        fieldset_test_df["text"].tolist(),
        fieldset_test_df["label"].tolist(),
        fieldset_test_df,
    )

    print(results)
    cm.figure_.savefig("confusion_matrix.png")
    misclassifications.to_csv("misclassifications.csv", index=False)


if __name__ == "__main__":
    _train()
