#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from sci_soft_models.software_mentions_repo_clf import (
    SoftwareMentionDetails,
    load_software_mentions_repo_clf_model,
    predict_repo_match_from_software_mentions,
)
from tqdm import tqdm

from .. import types

###############################################################################

# SOFTCITE DATA
THIS_DIR = Path(__file__).parent.resolve()
SOFTCITE_2025_FILES_DIR = THIS_DIR / "softcite-2025-files"
SOFTCITE_PREPPED_DATA_STORAGE_PATH = SOFTCITE_2025_FILES_DIR / "softcite-2025-prepped.parquet"
SOFTCITE_MENTIONS_NAME = "mentions.pdf.parquet"
SOFTCITE_PAPERS_NAME = "papers.parquet"
SOFTCITE_PURPOSE_ASSESSMENTS_NAME = "purpose_assessments.pdf.parquet"
SOFTCITE_INITIAL_SAMPLE_FOR_AGREEMENT_PATH = (
    SOFTCITE_2025_FILES_DIR / "softcite-2025-initial-sample-for-agreement.csv"
)
SOFTCITE_ANNOTATION_READY_PATH = SOFTCITE_2025_FILES_DIR / "softcite-2025-annotation-ready.csv"
SOFTCITE_ANNOTATED_PATH = SOFTCITE_2025_FILES_DIR / "softcite-2025-annotated.csv"

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
    purpose_assessments = pl.scan_parquet(data_dir_path / SOFTCITE_PURPOSE_ASSESSMENTS_NAME)

    # We want to get 400 of each high confidence "created", "shared", and "used"
    # Additionally create a subset of 10 from each for initial annotation
    full_subsets = []
    initial_sample_subsets = []
    for purpose in ["created", "shared", "used"]:
        # Filter for high confidence "created" and "shared" mentions
        purpose_selection = (
            purpose_assessments.filter(
                pl.col("scope") == "document",
                pl.col("purpose") == purpose,
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
        purpose_selection_with_github_urls = (
            mentions.filter(
                pl.col("url_raw").str.len_chars() > 0,
                pl.col("context_full_text").str.contains(pl.col("url_raw"), literal=True),
            )
            .select(
                "software_mention_id",
                "paper_id",
                "url_raw",
                pl.col("url_raw")
                .str.to_lowercase()
                .str.replace_all(r"\s+", "")
                .str.strip_chars_end(")")
                .alias("url_cleaned"),
                "context_full_text",
                pl.col("context_full_text")
                .str.to_lowercase()
                .alias("context_full_text_cleaned"),
            )
            .filter(
                pl.col("url_cleaned").str.contains("github"),
                pl.col("url_cleaned") != pl.lit("github"),
            )
            .join(
                purpose_selection,
                on="software_mention_id",
                how="inner",
            )
            .unique(
                subset=["software_mention_id", "url_cleaned"],
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
                pl.col("paper_id"),
                pl.col("doi").str.strip_chars().str.to_lowercase().alias("article_doi"),
                (pl.lit("https://doi.org/") + pl.col("doi").str.strip_chars()).alias(
                    "article_url"
                ),
                pl.col("software_mention_id").str.strip_chars().alias("software_mention_id"),
                pl.col("url_cleaned").alias("repository_url"),
                pl.col("purpose").alias("mention_purpose"),
                pl.col("certainty_score").alias("mention_purpose_certainty"),
                pl.col("context_full_text").alias("mention_context"),
            )
            .unique(
                subset=["article_doi", "repository_url"],
            )
            .collect()
        )

        # Take sample of 410 and store to CSV for annotation
        sample_n_or_max = min(410, len(purpose_selection_with_github_urls))
        sample_selection = purpose_selection_with_github_urls.sample(
            n=sample_n_or_max,
            seed=12,
        )
        initial_sample = sample_selection.head(10)
        remaining_sample_n = max(0, sample_n_or_max - 10)
        remaining_sample = sample_selection.tail(remaining_sample_n)
        initial_sample_subsets.append(initial_sample)
        full_subsets.append(remaining_sample)

    # Combine and store the initial sample for annotation
    initial_sample_annotation_ready: pl.DataFrame = pl.concat(initial_sample_subsets)
    initial_sample_annotation_ready = initial_sample_annotation_ready.sample(
        fraction=1.0, shuffle=True
    )
    initial_sample_annotation_ready.write_csv(SOFTCITE_INITIAL_SAMPLE_FOR_AGREEMENT_PATH)

    # Combine and store the full annotation ready data
    full_annotation_ready: pl.DataFrame = pl.concat(full_subsets)
    full_annotation_ready = full_annotation_ready.sample(fraction=1.0, shuffle=True)
    full_annotation_ready.write_csv(SOFTCITE_ANNOTATION_READY_PATH)


def _prep_softcite_2025_data_for_use(data_dir: str | Path) -> None:
    """Prepare the SoftCite 2025 data for use in the pipeline."""
    # Load env
    load_dotenv()

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
    print("Combining and filtering SoftCite 2025 data...")
    mentions = pl.scan_parquet(data_dir_path / SOFTCITE_MENTIONS_NAME)
    papers = pl.scan_parquet(data_dir_path / SOFTCITE_PAPERS_NAME)
    purpose_assessments = pl.scan_parquet(data_dir_path / SOFTCITE_PURPOSE_ASSESSMENTS_NAME)

    # Filter purpose assessments: document scope, certainty >= 0.5, all purposes
    filtered_purpose_assessments = (
        purpose_assessments.filter(
            pl.col("scope") == "document",
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

    # Filter mentions for GitHub URLs, join with purpose assessments and papers
    filtered_data = (
        mentions.filter(
            pl.col("url_raw").str.len_chars() > 0,
            pl.col("context_full_text").str.contains(pl.col("url_raw"), literal=True),
        )
        .select(
            "software_mention_id",
            "paper_id",
            "url_raw",
            pl.col("url_raw")
            .str.to_lowercase()
            .str.replace_all(r"\s+", "")
            .str.strip_chars_end(")")
            .alias("url_cleaned"),
            "context_full_text",
        )
        .filter(
            pl.col("url_cleaned").str.contains("github"),
            pl.col("url_cleaned") != pl.lit("github"),
        )
        .join(
            filtered_purpose_assessments,
            on="software_mention_id",
            how="inner",
        )
        .unique(
            subset=["software_mention_id", "url_cleaned"],
        )
        .join(
            papers.select("paper_id", "doi", "title"),
            on="paper_id",
            how="left",
        )
        .select(
            pl.col("software_mention_id"),
            pl.col("paper_id"),
            pl.col("doi").str.strip_chars().str.to_lowercase().alias("doi"),
            pl.col("title").alias("article_title"),
            pl.col("url_cleaned").alias("repo_url"),
            pl.col("context_full_text").alias("mention_context"),
        )
        .unique(
            subset=["doi", "repo_url"],
        )
        .collect()
    )

    print(f"Total candidate pairs before model: {len(filtered_data)}")

    # Load the transformer model
    model = load_software_mentions_repo_clf_model()

    # Construct SoftwareMentionDetails for each row
    software_mentions = [
        SoftwareMentionDetails(
            article_title=row["article_title"] or "",
            mention_context=row["mention_context"] or "",
            repository_url=row["repo_url"],
        )
        for row in filtered_data.iter_rows(named=True)
    ]

    # Run predictions in batches
    batch_size = 128
    all_predictions = []
    for i in tqdm(
        range(0, len(software_mentions), batch_size),
        total=(len(software_mentions) + batch_size - 1) // batch_size,
        desc="Running repo match predictions",
    ):
        batch = software_mentions[i : i + batch_size]
        batch_preds = predict_repo_match_from_software_mentions(
            software_mentions=batch,
            loaded_software_mentions_repo_clf_model=model,
        )
        all_predictions.extend(batch_preds)

    # Add prediction results back to the dataframe
    filtered_data = filtered_data.with_columns(
        pl.Series(
            "prediction",
            [p.repo_match_prediction for p in all_predictions],
        ),
        pl.Series(
            "confidence",
            [p.confidence for p in all_predictions],
        ),
    )

    # Log counts
    total_counts = filtered_data["prediction"].value_counts()
    match_count = total_counts.filter(pl.col("prediction") == "match")["count"]
    no_match_count = total_counts.filter(pl.col("prediction") == "no-match")["count"]
    match_n = match_count.item() if len(match_count) > 0 else 0
    no_match_n = no_match_count.item() if len(no_match_count) > 0 else 0
    print(
        f"Total processed: {match_n + no_match_n} "
        f"(matches: {match_n}, no-matches: {no_match_n})"
    )

    # Filter to only matches
    prepped_data = filtered_data.filter(
        pl.col("prediction") == "match",
    ).select(
        "software_mention_id",
        "paper_id",
        "doi",
        "repo_url",
        "confidence",
    )

    # Print total found pairs
    total_found = len(prepped_data)
    print(f"Total matched pairs found: {total_found}")

    # Print head to look at the prepped data
    print(prepped_data.head())

    # Store the prepped data
    prepped_data.write_parquet(SOFTCITE_PREPPED_DATA_STORAGE_PATH)


def get_dataset(
    **kwargs: dict[str, str],
) -> types.SuccessAndErroredResultsLists[types.BasicRepositoryDocumentPair]:
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
