#!/usr/bin/env python

"""Upload rs-graph-v2 database tables to HuggingFace Hub as a dataset."""

import os

import polars as pl
import typer
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from sqlalchemy import inspect as sa_inspect
from tqdm import tqdm

from rs_graph.db.utils import get_engine

###############################################################################

app = typer.Typer()

###############################################################################

REDACTED_TABLES: set[str] = {
    "researcher_developer_account_link",
    "developer_account",
    "repository_contributor",
    "researcher",
    "document_contributor",
    "document_contributor_institution",
}

SKIPPED_TABLES: set[str] = {
    "alembic_version",
}

###############################################################################


@app.command()
def upload_to_huggingface(
    repo_id: str = "evamaxfield/rs-graph-v2",
    use_prod: bool = False,
    redact: bool = True,
) -> None:
    """
    Read all tables from the rs-graph-v2 SQLite database and upload
    them to HuggingFace Hub as a DatasetDict.

    Requires authentication via `huggingface-cli login` or the HF_TOKEN
    environment variable.
    """
    # Load environment variables from .env file, if it exists
    load_dotenv()

    # Check for HF_TOKEN environment variable
    if not os.getenv("HF_TOKEN"):
        raise ValueError("Environment variable 'HF_TOKEN' is not set")

    engine = get_engine(use_prod=use_prod)
    table_names = sa_inspect(engine).get_table_names()
    to_process_table_names = [
        table_name
        for table_name in table_names
        if table_name not in SKIPPED_TABLES
        and (not redact or table_name not in REDACTED_TABLES)
    ]

    hf_datasets: dict[str, Dataset] = {}
    for table_name in tqdm(
        to_process_table_names,
        desc="Reading tables",
    ):
        print(f"Reading table: {table_name}")
        df = pl.read_database(
            f"SELECT * FROM {table_name}",
            connection=engine,
            infer_schema_length=None,
        )
        hf_datasets[table_name] = Dataset.from_polars(df)

    dataset_dict = DatasetDict(hf_datasets)

    print(f"Uploading {len(hf_datasets)} tables to {repo_id}")
    dataset_dict.push_to_hub(repo_id, private=True)
    print("Upload complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    app()
