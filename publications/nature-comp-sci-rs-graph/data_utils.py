#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import polars as pl
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

###############################################################################

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET = "sci-soft-collections/rs-graph-v2-full"

###############################################################################


def load_table(table: str) -> pl.DataFrame:
    """Load a single table from the HuggingFace dataset as a Polars DataFrame."""
    ds = load_dataset(HF_DATASET, table, split="train")
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


def load_base_dataset(
    one_to_one_only: bool = False,
    min_year: int = 2008,
    confidence_threshold: float = 0.9994,
    top_n_fields: int = 5,
) -> pl.DataFrame:
    # Load environment variables from .env file (if it exists)
    load_dotenv()

    documents = load_table("document")
    article_repo_links = load_table("document_repository_link")
    repositories = load_table("repository")
    document_topics = load_table("document_topic")
    topics = load_table("topic")
    document_contributors = load_table("document_contributor")
    researchers = load_table("researcher")
    dataset_sources = load_table("dataset_source")

    # Per-document author statistics (count, mean citations).
    document_author_stats = (
        document_contributors.join(
            researchers.select(
                pl.col("id").alias("researcher_id"),
                pl.col("cited_by_count").alias("researcher_cited_by_count"),
            ),
            on="researcher_id",
            how="left",
        )
        .group_by("document_id")
        .agg(
            pl.len().alias("document_author_count"),
            pl.mean("researcher_cited_by_count").alias("document_author_mean_citations"),
        )
    )

    # Top topic per document (highest score) → field name + domain name.
    document_top_topics = (
        document_topics.sort("score", descending=True)
        .unique(subset="document_id", keep="first")
        .join(
            topics.select(
                pl.col("id").alias("topic_id"),
                pl.col("field_name").alias("document_field_name"),
                pl.col("domain_name").alias("document_domain_name"),
            ),
            on="topic_id",
        )
        .select("document_id", "document_field_name", "document_domain_name")
    )

    # Build the merged base dataframe.
    merged = (
        article_repo_links.select(
            pl.col("id").alias("document_repository_link_id"),
            "document_id",
            "repository_id",
            "dataset_source_id",
            "predictive_model_confidence",
            pl.col("iteration").alias("link_processing_iteration"),
        )
        .join(
            documents.select(*[pl.col(c).alias(f"document_{c}") for c in documents.columns]),
            on="document_id",
        )
        .join(
            repositories.select(
                *[pl.col(c).alias(f"repository_{c}") for c in repositories.columns]
            ),
            on="repository_id",
        )
        .join(document_top_topics, on="document_id")
        .join(document_author_stats, on="document_id", how="left")
    )

    # Parse publication date and extract year.
    merged = merged.with_columns(
        pl.col("document_publication_date")
        .str.to_date("%Y-%m-%d")
        .alias("document_publication_date_parsed"),
    ).with_columns(
        pl.col("document_publication_date_parsed").dt.year().alias("document_publication_year"),
    )

    # Filter: published after GitHub's founding (2008).
    merged = merged.filter(pl.col("document_publication_year") >= min_year)

    # Filter: high-confidence or author-provided pairs (confidence > threshold or null).
    merged = merged.filter(
        (pl.col("predictive_model_confidence") > confidence_threshold)
        | pl.col("predictive_model_confidence").is_null()
    )

    # Optional: restrict to strict 1:1 pairs (each document and repository appears once).
    if one_to_one_only:
        merged = merged.unique(subset="document_id", keep="none").unique(
            subset="repository_id", keep="none"
        )

    # Prune field names: top N kept as-is, everything else → "Other".
    top_field_names = (
        merged.get_column("document_field_name")
        .value_counts(sort=True)
        .head(top_n_fields)
        .get_column("document_field_name")
        .to_list()
    )
    merged = merged.with_columns(
        pl.when(pl.col("document_field_name").is_in(top_field_names))
        .then(pl.col("document_field_name"))
        .otherwise(pl.lit("Other"))
        .alias("document_field_name_pruned")
    )

    # Add dataset source name from dataset_sources table.
    merged = merged.join(
        dataset_sources.select(
            pl.col("id").alias("dataset_source_id"),
            pl.col("name").alias("dataset_source_name"),
        ),
        on="dataset_source_id",
        how="left",
    )

    # Convert dataset source name to canonical from shorthand
    # pwc -> "Papers with Code"
    merged = merged.with_columns(
        pl.when(pl.col("dataset_source_name") == "pwc")
        .then(pl.lit("Papers with Code"))
        .when(pl.col("dataset_source_name") == "plos")
        .then(pl.lit("PLOS"))
        .when(pl.col("dataset_source_name") == "joss")
        .then(pl.lit("JOSS"))
        .when(pl.col("dataset_source_name") == "softwarex")
        .then(pl.lit("SoftwareX"))
        .when(pl.col("dataset_source_name") == "softcite_2025")
        .then(pl.lit("SoftCite 2025"))
        .when(pl.col("dataset_source_name") == "snowball-sampling-discovery")
        .then(pl.lit("Mined"))
        .otherwise(pl.col("dataset_source_name"))
        .alias("dataset_source_name_canonical")
    )

    return merged
