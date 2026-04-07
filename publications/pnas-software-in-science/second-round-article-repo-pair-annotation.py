import os
from pathlib import Path

import polars as pl
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

###############################################################################

load_dotenv()
os.environ["HF_DATASETS_OFFLINE"] = "1"

THIS_FILE_PATH = Path(__file__).resolve()
THIS_DIR = THIS_FILE_PATH.parent
RESULTS_DIR = THIS_DIR / "results" / "annotation"

app = typer.Typer()

###############################################################################


# Helper to load a table as a polars DataFrame (zero-copy via Arrow)
def load_table(table: str) -> pl.DataFrame:
    ds = load_dataset("evamxb/rs-graph-v2-full", table, split="train")
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


@app.command()
def main() -> None:
    # Make results dir if it doesn't exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all pair info
    documents = load_table("document")
    article_repo_links = load_table("document_repository_link")
    repositories = load_table("repository")

    merged = (
        article_repo_links.select(
            pl.col("id").alias("document_repository_link_id"),
            pl.col("document_id"),
            pl.col("repository_id"),
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
    print(f"Count of all document-repository pairs: {len(merged)}")

    # Drop to unique documents, then drop to unique repositories
    merged = merged.unique(
        subset="document_id",
        keep="none",
    ).unique(
        subset="repository_id",
        keep="none",
    )

    # Take a sample of 100 pairs for manual inspection
    # Where model confidence is above 0.99
    final_sample = (
        merged.filter(pl.col("predictive_model_confidence") >= 0.99)
        .sample(
            n=100,
            seed=42,
        )
        .select(
            pl.col("document_repository_link_id"),
            pl.col("document_id"),
            pl.col("repository_id"),
            (pl.lit("https://doi.org/") + pl.col("document_doi")).alias("document_url"),
            (
                pl.lit("https://github.com/")
                + pl.col("repository_owner")
                + pl.lit("/")
                + pl.col("repository_name")
            ).alias("repository_url"),
            pl.col("predictive_model_confidence"),
            pl.lit(None).alias("label"),
        )
        .sort(pl.col("predictive_model_confidence"), descending=False)
    )
    final_sample.write_csv(
        RESULTS_DIR / "annotation-second-round-sample-article-repo-pairs.csv"
    )


if __name__ == "__main__":
    app()
