"""Load RS-Graph article-repository pairs with the recommended filters.

Requires: pip install datasets polars
"""

from functools import partial

import polars as pl
from datasets import Dataset, load_dataset

REPO_ID = "sci-soft-collections/rs-graph-v2"
PAIR_CONFIDENCE_THRESHOLD = 0.9994  # NULL = curated link, always kept
MIN_PUBLICATION_YEAR = 2008  # GitHub launched in 2008
IDENTITY_CONFIDENCE_THRESHOLD = 0.97
SOURCE_DISPLAY_NAMES = {
    "pwc": "Papers with Code",
    "plos": "PLOS",
    "joss": "JOSS",
    "softwarex": "SoftwareX",
    "softcite_2025": "SoftCite 2025",
    "snowball-sampling-discovery": "Mined",
}


def load_table(
    table: str, repo_id: str | None = None, revision: str | None = None
) -> pl.DataFrame:
    """Load one RS-Graph table as a polars DataFrame."""
    ds = load_dataset(repo_id or REPO_ID, table, split="train", revision=revision)
    assert isinstance(ds, Dataset)
    return pl.DataFrame(ds.data.table)


def load_filtered_pairs(
    require_identity_link: bool = False,
    repo_id: str | None = None,
    revision: str | None = None,
) -> pl.DataFrame:
    """Return filtered article-repository pairs joined with article and repository metadata."""
    load = partial(load_table, repo_id=repo_id, revision=revision)
    # 1. pair confidence >= threshold, or NULL
    pairs = (
        load("document_repository_link")
        .filter(
            pl.col("predictive_model_confidence").is_null()
            | (pl.col("predictive_model_confidence") >= PAIR_CONFIDENCE_THRESHOLD)
        )
        .select(
            pl.col("id").alias("document_repository_link_id"),
            "document_id",
            "repository_id",
            "dataset_source_id",
            pl.col("predictive_model_confidence").alias("pair_confidence"),
        )
    )
    top_topic = (
        load("document_topic")
        .sort("score", descending=True)
        .unique(subset="document_id", keep="first")
        .join(
            load("topic").select(
                pl.col("id").alias("topic_id"),
                pl.col("field_name").alias("document_field_name"),
                pl.col("domain_name").alias("document_domain_name"),
            ),
            on="topic_id",
        )
        .select("document_id", "document_field_name", "document_domain_name")
    )
    sources = load("dataset_source").select(
        pl.col("id").alias("dataset_source_id"),
        pl.col("name").alias("dataset_source_name"),
    )
    pairs = (
        pairs.join(
            load("document").select(pl.all().name.prefix("document_")),
            on="document_id",
        )
        .join(
            load("repository").select(pl.all().name.prefix("repository_")),
            on="repository_id",
        )
        .join(top_topic, on="document_id", how="left")
        .join(sources, on="dataset_source_id", how="left")
        .with_columns(
            pl.col("document_publication_date")
            .str.to_date("%Y-%m-%d", strict=False)
            .dt.year()
            .alias("document_publication_year"),
            pl.col("repository_creation_datetime")
            .str.to_datetime(strict=False)
            .dt.year()
            .alias("repository_creation_year"),
            pl.col("dataset_source_name")
            .replace(SOURCE_DISPLAY_NAMES)
            .alias("dataset_source_display_name"),
            ("https://doi.org/" + pl.col("document_doi")).alias("document_url"),
            (
                "https://github.com/"
                + pl.col("repository_owner")
                + "/"
                + pl.col("repository_name")
            ).alias("repository_url"),
        )
        # 2. publication-year floor
        .filter(pl.col("document_publication_year") >= MIN_PUBLICATION_YEAR)
    )
    # 3. optional: keep pairs whose article has a confidently linked author
    # and whose repository has a confidently linked contributor
    if require_identity_link:
        links = load("researcher_developer_account_link").filter(
            pl.col("predictive_model_confidence") >= IDENTITY_CONFIDENCE_THRESHOLD
        )
        docs = (
            load("document_contributor")
            .join(links.select("researcher_id"), on="researcher_id", how="semi")
            .select("document_id")
            .unique()
        )
        repos = (
            load("repository_contributor")
            .join(links.select("developer_account_id"), on="developer_account_id", how="semi")
            .select("repository_id")
            .unique()
        )
        pairs = pairs.join(docs, on="document_id", how="semi").join(
            repos, on="repository_id", how="semi"
        )
    return pairs
