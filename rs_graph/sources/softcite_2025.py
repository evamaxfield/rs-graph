#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import polars as pl

###############################################################################

# SOFTCITE DATA
SOFTCITE_PREPPED_DATA_STORAGE_PATH = Path(__file__).parent / "softcite-2025-prepped.parquet"
SOFTCITE_MENTIONS_NAME = "mentions.pdf.parquet"
SOFTCITE_PAPERS_NAME = "papers.parquet"
SOFTCITE_PURPOSE_ASSESSMENTS_NAME = "purpose_assessments.pdf.parquet"

###############################################################################


def _prep_softcite_2025_data(data_dir: Path) -> None:
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

    # Filter for high confidence "created" and "shared" mentions
    created_and_shared = (
        purpose_assessments.filter(
            pl.col("scope") == "document",
            pl.col("purpose").is_in(["created", "shared"]),
            pl.col("certainty_score") >= 0.5,
        )
        .select(
            "software_mention_id",
            "certainty_score",
        )
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
            pl.col("software_mention_id").str.strip_chars().alias("software_mention_id"),
            pl.col("url_raw"),
            pl.col("certainty_score"),
        )
        .unique(
            subset=["doi", "url_raw"],
        )
        .collect()
    )

    # Store the prepped data
    created_github_urls.write_parquet(SOFTCITE_PREPPED_DATA_STORAGE_PATH)
