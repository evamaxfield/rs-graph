#!/usr/bin/env python

"""Upload rs-graph-v2 database tables to HuggingFace Hub as a dataset."""

import math
import os
import shutil
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import typer
from datasets import Dataset
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
    "repository_file",  # This table is massive, skip it for now.
}

BATCH_SIZE: int = 2**20  # 1,048,576 rows per batch

###############################################################################


@app.command()
def upload_to_huggingface(
    database_path: str = typer.Argument(
        help="Path to the SQLite database file to use.",
    ),
    redact: bool = True,
    org: str = "evamxb",
) -> None:
    """
    Read all tables from the rs-graph-v2 SQLite database and upload
    them to HuggingFace Hub as a DatasetDict.

    Requires authentication via `huggingface-cli login` or the HF_TOKEN
    environment variable.
    """
    # Decide repo id based on redact flag
    if redact:
        repo_id = f"{org}/rs-graph-v2-redacted"
    else:
        repo_id = f"{org}/rs-graph-v2-full"

    # Load environment variables from .env file, if it exists
    load_dotenv()

    # Check for HF_TOKEN environment variable
    if not os.getenv("HF_TOKEN"):
        raise ValueError("Environment variable 'HF_TOKEN' is not set")

    print(f"Will upload dataset to HuggingFace Hub repo: {repo_id}")

    # Output directory
    redact_flag = "redacted" if redact else "full"
    output_dir = Path(f"rs-graph-v2-hf-upload-{redact_flag}/")

    # Remove existing output directory if it exists to ensure a clean slate
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Remake
    output_dir.mkdir()

    engine = get_engine(database_path)
    table_names = sa_inspect(engine).get_table_names()
    to_process_table_names = [
        table_name
        for table_name in table_names
        if table_name not in SKIPPED_TABLES
        and (not redact or table_name not in REDACTED_TABLES)
    ]

    # Write each table to parquet in batches to avoid OOM
    for table_name in tqdm(
        to_process_table_names,
        desc="Writing tables to parquet",
    ):
        row_count: int = pl.read_database(
            f"SELECT COUNT(*) AS cnt FROM {table_name}",
            connection=engine,
        ).item(0, 0)
        print(f"Reading table: {table_name} ({row_count} rows)")

        if row_count == 0:
            # Empty table — write a schema-only parquet
            df = pl.read_database(
                f"SELECT * FROM {table_name} LIMIT 0",
                connection=engine,
                infer_schema_length=None,
            )
            df.write_parquet(output_dir / f"{table_name}.parquet")
            continue

        # Keyset pagination on `id` (every table's primary key -- see
        # db/models.py) instead of LIMIT/OFFSET: OFFSET makes SQLite re-scan
        # through every already-read row on each batch, so a large table split
        # into many batches degrades toward quadratic instead of linear.
        # `WHERE id > last_seen_id ORDER BY id LIMIT batch_size` reads each row
        # exactly once regardless of how many batches the table is split into.
        #
        # Each batch is appended straight into the table's final parquet file
        # via a single ParquetWriter, instead of writing N separate batch files
        # and then reading them all back to recombine into one -- that combine
        # step was a full extra read+write pass over data that had just been
        # written a moment before.
        last_id = -1
        writer: pq.ParquetWriter | None = None
        try:
            with tqdm(
                desc=f"Processing {table_name}",
                total=math.ceil(row_count / BATCH_SIZE),
            ) as pbar:
                while True:
                    batch_df = pl.read_database(
                        f"SELECT * FROM {table_name} WHERE id > {last_id} "
                        f"ORDER BY id LIMIT {BATCH_SIZE}",
                        connection=engine,
                        infer_schema_length=None,
                    )
                    if batch_df.is_empty():
                        break
                    batch_table = batch_df.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_dir / f"{table_name}.parquet", batch_table.schema
                        )
                    writer.write_table(batch_table)
                    last_id = batch_df["id"].max()
                    del batch_df, batch_table
                    pbar.update(1)
        finally:
            if writer is not None:
                writer.close()

    # Upload each table as a separate config
    print(f"Uploading {len(to_process_table_names)} tables to {repo_id}")
    for table_name in tqdm(
        to_process_table_names,
        desc="Uploading tables",
    ):
        ds = Dataset.from_parquet(str(output_dir / f"{table_name}.parquet"))
        assert isinstance(ds, Dataset)
        ds.push_to_hub(repo_id, config_name=table_name, private=True)
        del ds

    print("Upload complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    app()
