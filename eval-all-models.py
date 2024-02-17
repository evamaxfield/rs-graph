import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import random
from tabulate import tabulate
from sklearn.linear_model import LogisticRegressionCV
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import datasets
from itertools import combinations
from transformers import pipeline, Pipeline
from autotrain.trainers.text_classification.__main__ import train as ft_train
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

###############################################################################

# Load environment variables
load_dotenv()

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
    "dev_username",
    "dev_email",
    "dev_bio",
    "dev_co_contributors",
    "author_co_authors",
]

# Create all combinations
OPTIONAL_DATA_FIELDSETS = [
    (),  # include no optional data
]
for i in range(1, len(OPTIONAL_DATA_FIELDS) + 1):
    OPTIONAL_DATA_FIELDSETS.extend(list(combinations(OPTIONAL_DATA_FIELDS, i)))

# Basic template for model input
model_str_input_template = """
# Developer Details

name: {dev_name}
{dev_extras}

---

# Author Details

name: {author_name}
{author_extras}
""".strip()

# Fine-tune default settings
DEFAULT_HF_DATASET_PATH = "evamxb/dev-author-em-dataset"
DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH = Path("autotrain-text-classification-temp/")
FINE_TUNE_COMMAND_DICT = {
    "data_path": DEFAULT_HF_DATASET_PATH,
    "token": os.environ["HF_AUTH_TOKEN"],
    "project_name": str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
    "text_column": "text",
    "target_column": "label",
    "train_split": "test",
    "valid_split": "valid",
    "epochs": 1,
    "lr": 5e-5,
    "auto_find_batch_size": True,
    "seed": 12,
}

# Set seed
np.random.seed(12)
random.seed(12)

###############################################################################

# Load data
dev_author_full_details = pd.read_parquet("dev-author-em-full-details.parquet")

# Create splits
train_df, test_and_valid_df = train_test_split(
    dev_author_full_details,
    test_size=0.4,
    shuffle=True,
    stratify=dev_author_full_details["match"],
)
valid_df, test_df = train_test_split(
    test_and_valid_df,
    test_size=0.5,
    shuffle=True,
    stratify=test_and_valid_df["match"],
)

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
        dev_extras = "\n".join([
            f"{field.strip('dev_')}: {row[field]}"
            for field in fieldset if field.startswith("dev_")
        ])
        author_extras = "\n".join([
            f"{field.strip('author_')}: {row[field]}"
            for field in fieldset if field.startswith("author_")
        ])
        model_str_input = model_str_input_template.format(
            dev_name=row["dev_name"],
            dev_extras=dev_extras,
            author_name=row["author_name"],
            author_extras=author_extras,
        )
        rows.append({
            "text": model_str_input.strip(),
            "label": row["match"],
        })    
    
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

    return EvaluationResults(
        fieldset=fieldset,
        model=this_iter_model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        time_pred=perf_time,
    )

# Train semantic logit models
results = []
for fieldset in tqdm(
    OPTIONAL_DATA_FIELDSETS,
    desc="Fieldsets",
):
    print()
    print(f"Working on fieldset: '{fieldset}'")
    try:
        # Create the datasets
        fieldset_train_df, fieldset_train_ds = convert_split_details_to_text_input_dataset(
            train_df,
            fieldset,
        )
        fieldset_valid_df, fieldset_valid_ds = convert_split_details_to_text_input_dataset(
            valid_df,
            fieldset,
        )
        fieldset_test_df, fieldset_test_ds = convert_split_details_to_text_input_dataset(
            test_df,
            fieldset,
        )

        # Store as dataset dict
        fieldset_ds_dict = datasets.DatasetDict({
            "train": fieldset_train_ds,
            "valid": fieldset_valid_ds,
            "test": fieldset_test_ds,
        })

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
                    max_iter=5000,
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
                    ).to_dict(),
                )

                print()

            except Exception as e:
                print(f"Error during: {this_iter_model_name}, Error: {e}")
                results.append({
                    "fieldset": fieldset,
                    "model": this_iter_model_name,
                    "error_level": "semantic model training",
                    "error": str(e),
                })

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
                )
                results.append(
                    evaluate(
                        ft_transformer_pipe,
                        fieldset_test_df["text"].tolist(),
                        fieldset_test_df["label"].tolist(),
                    ).to_dict(),
                )

            except Exception as e:
                print(f"Error during: {this_iter_model_name}, Error: {e}")
                results.append({
                    "fieldset": fieldset,
                    "model": this_iter_model_name,
                    "error_level": "fine-tune model training",
                    "error": str(e),
                })


    except Exception as e:
        print(f"Error during: {fieldset}, Error: {e}")
        results.append({
            "fieldset": fieldset,
            "error_level": "dataset creation",
            "error": str(e),
        })

    print()

    # Print results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False).reset_index(drop=True)
    results_df.to_csv("all-model-results.csv", index=False)
    print("Current standings")
    print(tabulate(
        results_df.head(10),
        headers="keys",
        tablefmt="psql",
        showindex=False,
    ))

    print()

print()
print("-" * 80)
print()

# Print results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="f1", ascending=False).reset_index(drop=True)
results_df.to_csv("all-model-results.csv", index=False)
print("Final standings")
print(tabulate(
    results_df.head(10),
    headers="keys",
    tablefmt="psql",
    showindex=False,
))