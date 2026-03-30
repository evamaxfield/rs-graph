#!/usr/bin/env python

"""Upload rs-graph-v2 database tables to HuggingFace Hub as a dataset."""

import os
import tempfile
from pathlib import Path

import polars as pl
import typer
from datasets import DatasetDict, load_dataset
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
    repo_id: str = "evamxb/rs-graph-v2",
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write each table to parquet one at a time to avoid OOM
        for table_name in tqdm(
            to_process_table_names,
            desc="Writing tables to parquet",
        ):
            print(f"Reading table: {table_name}")
            df = pl.read_database(
                f"SELECT * FROM {table_name}",
                connection=engine,
                infer_schema_length=None,
            )
            df.write_parquet(tmpdir_path / f"{table_name}.parquet")
            del df

        # Build DatasetDict from parquet files (memory-mapped, not loaded into RAM)
        data_files = {
            table_name: str(tmpdir_path / f"{table_name}.parquet")
            for table_name in to_process_table_names
        }
        dataset_dict = load_dataset("parquet", data_files=data_files)
        assert isinstance(dataset_dict, DatasetDict)

        print(f"Uploading {len(to_process_table_names)} tables to {repo_id}")
        dataset_dict.push_to_hub(repo_id, private=True)

    print("Upload complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    app()
