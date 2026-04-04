import os
from pathlib import Path

import polars as pl
import typer
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv

###############################################################################

app = typer.Typer()

THIS_FILE_PATH = Path(__file__).resolve()
THIS_DIR = THIS_FILE_PATH.parent
RESULTS_DIR = THIS_DIR / "results" / "dataset-size-comparison"

###############################################################################


# Helper to load a table as a polars DataFrame (zero-copy via Arrow)
def load_table(table: str) -> pl.DataFrame:
    ds = load_dataset("evamxb/rs-graph-v2", table, split="train")
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


def _load_our_dataset() -> pl.DataFrame:
    # Load all pair info
    documents = load_table("document")
    article_repo_links = load_table("document_repository_link")
    repositories = load_table("repository")
    dataset_sources = load_table("dataset_source")

    merged = (
        article_repo_links.select(
            pl.col("id").alias("document_repository_link_id"),
            pl.col("document_id"),
            pl.col("repository_id"),
            pl.col("dataset_source_id"),
            pl.col("predictive_model_confidence"),
        )
        .join(
            documents.select(
                *[pl.col(col).alias(f"document_{col}") for col in documents.columns]
            ),
            on="document_id",
        )
        .join(
            repositories.select(
                *[pl.col(col).alias(f"repository_{col}") for col in repositories.columns]
            ),
            on="repository_id",
        )
        .join(
            dataset_sources.select(
                pl.col("id").alias("dataset_source_id"),
                pl.col("name").alias("dataset_source_name"),
            ),
            on="dataset_source_id",
        )
    )

    # Create document publication year column as integer (extract year from date)
    merged = merged.with_columns(
        pl.col("document_publication_date")
        .str.to_date("%Y-%m-%d")
        .alias("document_publication_date_parsed"),
    ).with_columns(
        pl.col("document_publication_date_parsed").dt.year().alias("document_publication_year"),
    )

    # Filter to only pairs published after 2008 (the year GitHub was founded)
    merged = merged.filter(pl.col("document_publication_year") >= 2008)

    return merged


def _create_many_to_many_csvs(our_dataset: pl.DataFrame) -> None:
    # Get the set of article-repo pairs that has multiple
    # repositories for the same article
    article_multi_repo_doc_ids = (
        our_dataset.group_by("document_id")
        .agg(pl.col("repository_id").n_unique().alias("unique_repo_count"))
        .filter(pl.col("unique_repo_count") > 1)
        .get_column("document_id")
        .to_list()
    )

    our_dataset.filter(pl.col("document_id").is_in(article_multi_repo_doc_ids)).select(
        "document_repository_link_id",
        "document_id",
        "repository_id",
        "predictive_model_confidence",
        (pl.lit("https://doi.org/") + pl.col("document_doi")).alias("document_url"),
        (
            pl.lit("https://github.com/")
            + pl.col("repository_owner")
            + pl.lit("/")
            + pl.col("repository_name")
        ).alias("repository_url"),
    ).sort("document_id").write_csv(RESULTS_DIR / "high_confidence_multi_repo_pairs.csv")

    # Same thing but for the set of article-repo pairs that has multiple
    # articles for the same repository
    repo_multi_article_repo_ids = (
        our_dataset.group_by("repository_id")
        .agg(pl.col("document_id").n_unique().alias("unique_doc_count"))
        .filter(pl.col("unique_doc_count") > 1)
        .get_column("repository_id")
        .to_list()
    )
    our_dataset.filter(pl.col("repository_id").is_in(repo_multi_article_repo_ids)).select(
        "document_repository_link_id",
        "document_id",
        "repository_id",
        "predictive_model_confidence",
        (pl.lit("https://doi.org/") + pl.col("document_doi")).alias("document_url"),
        (
            pl.lit("https://github.com/")
            + pl.col("repository_owner")
            + pl.lit("/")
            + pl.col("repository_name")
        ).alias("repository_url"),
    ).sort("repository_id").write_csv(RESULTS_DIR / "high_confidence_multi_article_pairs.csv")


def _load_pwc_dataset() -> pl.DataFrame:
    # Load papers-with-code dataset and filter to only "is_official" repos
    pwc_ds_dict = load_dataset("pwc-archive/links-between-paper-and-code")
    assert isinstance(pwc_ds_dict, DatasetDict)
    pwc_df = pwc_ds_dict["train"].to_polars()
    assert isinstance(pwc_df, pl.DataFrame)

    return pwc_df


def _load_softcite_dataset() -> pl.DataFrame:
    # Load softcite 2025 dataset
    softcite_papers = pl.scan_parquet(
        "~/Downloads/softcite-extractions-oa-data/full_dataset/papers.parquet"
    )
    softcite_papers = softcite_papers.filter(pl.col("has_mentions")).collect().unique("doi")

    return softcite_papers


def _get_counts() -> pl.DataFrame:
    our_dataset = _load_our_dataset()
    pwc_df = _load_pwc_dataset()
    softcite_papers = _load_softcite_dataset()

    # Filter to high confidence pairs (confidence null or >= 0.9994)
    our_dataset_high_conf_filtered = our_dataset.filter(
        (pl.col("predictive_model_confidence") > 0.9994)
        | (pl.col("predictive_model_confidence").is_null())
    )

    counts = [
        {"dataset": "ours", "subset": "full", "count": len(our_dataset)},
        {
            "dataset": "ours",
            "subset": "high_conf_filtered",
            "count": len(our_dataset_high_conf_filtered),
        },
        {"dataset": "pwc", "subset": "full", "count": len(pwc_df)},
        {
            "dataset": "pwc",
            "subset": "is_official_filtered",
            "count": len(pwc_df.filter(pl.col("is_official"))),
        },
        {
            "dataset": "pwc",
            "subset": "accessible_and_verified",
            "count": len(
                our_dataset_high_conf_filtered.filter(pl.col("dataset_source_name") == "pwc")
            ),
        },
        {
            "dataset": "softcite",
            "subset": "papers_with_mentions",
            "count": len(softcite_papers),
        },
    ]

    return pl.DataFrame(counts)


@app.command()
def main() -> None:
    load_dotenv()
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Create results dir
    RESULTS_DIR.mkdir(exist_ok=True)

    # Get counts for each dataset and store in a CSV for easy reference in the paper
    counts = _get_counts()

    # Always create the many-to-many CSVs for extra inspection
    our_dataset = _load_our_dataset()
    high_conf_filtered = our_dataset.filter(
        (pl.col("predictive_model_confidence") > 0.9994)
        | (pl.col("predictive_model_confidence").is_null())
    )
    _create_many_to_many_csvs(high_conf_filtered)

    # Write counts to CSV for easy reference in the paper
    counts.write_csv(RESULTS_DIR / "dataset-size-comparison.csv")
    print(counts)


###############################################################################

if __name__ == "__main__":
    app()
