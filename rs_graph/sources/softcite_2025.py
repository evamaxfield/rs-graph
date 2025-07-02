#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import polars as pl

from .. import types

###############################################################################

# SOFTCITE DATA
THIS_DIR = Path(__file__).parent.resolve()
SOFTCITE_2025_FILES_DIR = THIS_DIR / "softcite-2025-files"
SOFTCITE_PREPPED_DATA_STORAGE_PATH = SOFTCITE_2025_FILES_DIR / "softcite-2025-prepped.parquet"
SOFTCITE_MENTIONS_NAME = "mentions.pdf.parquet"
SOFTCITE_PAPERS_NAME = "papers.parquet"
SOFTCITE_PURPOSE_ASSESSMENTS_NAME = "purpose_assessments.pdf.parquet"
SOFTCITE_ANNOTATION_READY_PATH = SOFTCITE_2025_FILES_DIR / "softcite-2025-annotation-ready.csv"
SOFTCITE_ANNOTATED_PATH = SOFTCITE_2025_FILES_DIR / "softcite-2025-annotated.csv"
SOFTCITE_MINI_CLS_PATH = SOFTCITE_2025_FILES_DIR / "softcite-2025-mini-classifier.skops"

# Make sure the directory exists
SOFTCITE_2025_FILES_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################


def _prep_softcite_2025_data_for_annotation(data_dir: str | Path) -> None:
    # Resolve and convert to Path object
    data_dir_path = Path(data_dir).resolve()
    if not all(
        path.exists()
        for path in (
            data_dir_path / SOFTCITE_MENTIONS_NAME,
            data_dir_path / SOFTCITE_PAPERS_NAME,
            data_dir_path / SOFTCITE_PURPOSE_ASSESSMENTS_NAME,
        )
    ):
        raise FileNotFoundError(
            "SoftCite 2025 raw data files not found."
            "Please ensure the data is downloaded and placed in the correct directory. "
            "SoftCite 2025 data can be downloaded from: "
            "https://doi.org/10.5281/zenodo.15149379 -- "
            "provide the path to the data directory you want "
            "to process (p01, p05, or full)."
        )

    # Read the data
    mentions = pl.scan_parquet(data_dir_path / SOFTCITE_MENTIONS_NAME)
    papers = pl.scan_parquet(data_dir_path / SOFTCITE_PAPERS_NAME)
    purpose_assessments = pl.read_parquet(data_dir_path / SOFTCITE_PURPOSE_ASSESSMENTS_NAME)

    # Filter for high confidence "created" and "shared" mentions
    created_and_shared = (
        purpose_assessments.filter(
            pl.col("scope") == "document",
            pl.col("purpose").is_in(["created", "shared"]),
            pl.col("certainty_score") >= 0.5,
        )
        .select(
            "software_mention_id",
            "purpose",
            "certainty_score",
        )
        .sort("certainty_score", descending=True)
        .unique(subset=["software_mention_id"])
    )

    # Get the unique GitHub URLs from the mentions
    created_github_urls = (
        mentions.select(
            "software_mention_id",
            "paper_id",
            pl.col("url_raw")
            .str.to_lowercase()
            .str.replace_all(r"\s+", "")
            .str.strip_chars_end(")")
            .alias("url_raw"),
        )
        .join(
            created_and_shared,
            on="software_mention_id",
            how="inner",
        )
        .unique(
            subset=["software_mention_id", "url_raw"],
        )
        .filter(pl.col("url_raw").str.contains("github"))
        .join(
            papers.select(
                "paper_id",
                "doi",
            ),
            on="paper_id",
            how="left",
        )
        .select(
            pl.col("paper_id"),
            pl.col("doi").str.strip_chars().str.to_lowercase().alias("doi"),
            (pl.lit("https://doi.org/") + pl.col("doi").str.strip_chars()).alias("doi_url"),
            pl.col("software_mention_id").str.strip_chars().alias("software_mention_id"),
            pl.col("url_raw"),
            pl.col("certainty_score"),
            pl.col("purpose"),
        )
        .unique(
            subset=["doi", "url_raw"],
        )
        .collect()
    )

    # Take sample of 200 and store to CSV for annotation
    created_github_urls.sample(n=200).write_csv(SOFTCITE_ANNOTATION_READY_PATH)


def _train_softcite_mini_classifier_from_annotation_data(data_dir: str | Path) -> None:
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from skops.io import dump

    # Resolve and convert to Path object
    data_dir_path = Path(data_dir).resolve()
    if not all(
        path.exists()
        for path in (
            data_dir_path / SOFTCITE_MENTIONS_NAME,
            data_dir_path / SOFTCITE_PAPERS_NAME,
            data_dir_path / SOFTCITE_PURPOSE_ASSESSMENTS_NAME,
        )
    ):
        raise FileNotFoundError(
            "SoftCite 2025 raw data files not found."
            "Please ensure the data is downloaded and placed in the correct directory. "
            "SoftCite 2025 data can be downloaded from: "
            "https://doi.org/10.5281/zenodo.15149379 -- "
            "provide the path to the data directory you want "
            "to process (p01, p05, or full)."
        )

    # Read the data
    annotated = pl.read_csv(SOFTCITE_ANNOTATED_PATH)
    annotated = annotated.drop_nulls(subset=["label"])
    purpose_assessments = pl.read_parquet(data_dir_path / SOFTCITE_PURPOSE_ASSESSMENTS_NAME)

    # Join the two datasets together
    purpose_with_label = (
        purpose_assessments.with_columns(
            pl.col("software_mention_id").str.strip_chars().alias("software_mention_id"),
        )
        .filter(
            pl.col("scope") == "document",
            pl.col("software_mention_id").is_in(annotated["software_mention_id"].to_list()),
        )
        .join(
            annotated.select(
                "software_mention_id",
                "label",
            ),
            on="software_mention_id",
            how="left",
        )
        .select(
            "software_mention_id",
            "label",
            "purpose",
            "certainty_score",
        )
        .filter(pl.col("label").is_in(["yes", "no"]))
        .pivot(
            on="purpose",
            values="certainty_score",
            index=["software_mention_id", "label"],
        )
    )

    # Train logit from sklearn for best
    modeling_data = purpose_with_label.with_columns(
        (pl.col("used") * pl.col("shared")).alias("used_shared"),
        (pl.col("created") * pl.col("shared")).alias("created_shared"),
        (pl.col("used") * pl.col("created")).alias("used_created"),
    )

    # Create splits
    train_set, test_set = train_test_split(
        modeling_data.to_pandas(),
        test_size=0.2,
        random_state=12,
        stratify=modeling_data["label"].to_list(),
    )

    cols_for_modeling = [
        "shared",
        "created",
        "used",
        "used_shared",
        "created_shared",
        "used_created",
    ]

    # Fit model
    logit = LogisticRegressionCV(
        max_iter=1000,
        random_state=12,
        class_weight="balanced",
    ).fit(
        train_set[[*cols_for_modeling]],
        train_set["label"].to_list(),
    )

    # Predict
    preds = logit.predict(test_set[[*cols_for_modeling]])

    # Evaluate
    print(
        classification_report(
            y_true=test_set["label"],
            y_pred=preds,
            labels=["yes", "no"],
        )
    )

    # Store to mini classifier path
    dump(logit, SOFTCITE_MINI_CLS_PATH)


def _prep_softcite_2025_data_for_use(data_dir: str | Path) -> None:
    """Get the prepped SoftCite 2025 data for annotation."""
    from sklearn.linear_model import LogisticRegressionCV
    from skops.io import load

    # Resolve and convert to Path object
    data_dir_path = Path(data_dir).resolve()
    if not all(
        path.exists()
        for path in (
            data_dir_path / SOFTCITE_MENTIONS_NAME,
            data_dir_path / SOFTCITE_PAPERS_NAME,
            data_dir_path / SOFTCITE_PURPOSE_ASSESSMENTS_NAME,
        )
    ):
        raise FileNotFoundError(
            "SoftCite 2025 raw data files not found."
            "Please ensure the data is downloaded and placed in the correct directory. "
            "SoftCite 2025 data can be downloaded from: "
            "https://doi.org/10.5281/zenodo.15149379 -- "
            "provide the path to the data directory you want "
            "to process (p01, p05, or full)."
        )

    # Read the data
    mentions = pl.scan_parquet(data_dir_path / SOFTCITE_MENTIONS_NAME)
    papers = pl.scan_parquet(data_dir_path / SOFTCITE_PAPERS_NAME)
    purpose_assessments = pl.read_parquet(data_dir_path / SOFTCITE_PURPOSE_ASSESSMENTS_NAME)

    # Load skops model
    logit: LogisticRegressionCV = load(SOFTCITE_MINI_CLS_PATH)

    # Pivot the purpose assessments to get the purpose columns
    purpose_assessments = purpose_assessments.filter(
        pl.col("scope") == "document",
    ).pivot(
        on="purpose",
        values="certainty_score",
        index="software_mention_id",
    )

    # Create computed columns across purpose assessments
    purpose_assessments = purpose_assessments.with_columns(
        (pl.col("used") * pl.col("shared")).alias("used_shared"),
        (pl.col("created") * pl.col("shared")).alias("created_shared"),
        (pl.col("used") * pl.col("created")).alias("used_created"),
    )

    # Apply the model to the purpose assessments
    cols_for_modeling = [
        "shared",
        "created",
        "used",
        "used_shared",
        "created_shared",
        "used_created",
    ]
    purpose_assessments = purpose_assessments.with_columns(
        pl.Series(
            "predicted_label",
            logit.predict(purpose_assessments.select(*cols_for_modeling).to_pandas()),
        ),
    )

    # Log counts
    total_counts = purpose_assessments["predicted_label"].value_counts()
    total_yes = total_counts.filter(pl.col("predicted_label") == "yes")["count"].item()
    total_no = total_counts.filter(pl.col("predicted_label") == "no")["count"].item()
    print(
        f"Total processed: {total_yes + total_no} "
        f"(hopefully-matches: {total_yes}, likely-not-matches: {total_no})"
    )

    # Filter to only those with a "yes" prediction
    hopefully_matches = purpose_assessments.filter(
        pl.col("predicted_label") == "yes",
    )["software_mention_id"].to_list()

    # Join with mentions and papers to get the final prepped dataset
    prepped_data = (
        mentions.filter(
            pl.col("software_mention_id").is_in(hopefully_matches),
            pl.col("url_raw").str.contains(r"(?i)github"),
        )
        .unique(
            subset="software_mention_id",
        )
        .select(
            pl.col("software_mention_id"),
            pl.col("paper_id"),
            pl.col("url_raw"),
        )
        .join(
            papers.select(
                "paper_id",
                "doi",
            ),
            on="paper_id",
            how="left",
        )
        .select(
            pl.col("software_mention_id"),
            pl.col("paper_id"),
            pl.col("doi").str.strip_chars().str.to_lowercase().alias("doi"),
            pl.col("url_raw")
            .str.to_lowercase()
            .str.replace_all(r"\s+", "")
            .str.strip_chars()
            .str.strip_chars_end(")")
            # Remove whitespace
            .alias("repo_url"),
        )
        .unique(
            subset=["doi", "repo_url"],
        )
        .collect()
    )

    # Print total found pairs
    total_found = len(prepped_data)
    print(f"Total (hopefully matched) pairs found: {total_found}")

    # Print head to look at the prepped data
    print(prepped_data.head())

    # Store the prepped data
    prepped_data.write_parquet(SOFTCITE_PREPPED_DATA_STORAGE_PATH)


def get_dataset(
    **kwargs: dict[str, str],
) -> types.SuccessAndErroredResultsLists:
    """Load the SoftCite 2025 dataset."""
    # Load the dataset
    df = pl.read_parquet(SOFTCITE_PREPPED_DATA_STORAGE_PATH)

    # Iter rows and convert
    results = []
    for row in df.iter_rows(named=True):
        results.append(
            types.BasicRepositoryDocumentPair(
                source="softcite_2025",
                repo_url=row["repo_url"],
                paper_doi=row["doi"],
            )
        )

    # Return the results
    return types.SuccessAndErroredResultsLists(
        successful_results=results,
        errored_results=[],
    )
