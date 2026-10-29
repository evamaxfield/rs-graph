#!/usr/bin/env python

"""Upload rs-graph-v2 database tables to HuggingFace Hub as full and public datasets."""

import math
import os
import shutil
from enum import StrEnum
from pathlib import Path
from string import Template

import polars as pl
import pyarrow.parquet as pq
import typer
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from sqlalchemy import inspect as sa_inspect
from tqdm import tqdm

from rs_graph.db.utils import get_engine

###############################################################################

app = typer.Typer()

###############################################################################

PUBLIC_DROPPED_TABLES: set[str] = {"document_abstract"}

PUBLIC_DROPPED_COLUMNS: dict[str, list[str]] = {"developer_account": ["name", "email"]}

SKIPPED_TABLES: set[str] = {
    "alembic_version",
}

BATCH_SIZE: int = 2**20  # 1,048,576 rows per batch

REPO_NAMES: dict[str, str] = {"full": "rs-graph-v2-full", "public": "rs-graph-v2"}

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "release" / "templates"

GATED_YAML = """extra_gated_prompt: >-
  This variant includes article abstracts and developer display names and email
  addresses. Access is granted by manual approval for research use.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
  I will use this for research only and not contact or re-identify individuals: checkbox"""

SUMMARY = (
    "RS-Graph v2 links scientific articles to the code repositories they describe, together "
    "with article, researcher, repository and developer-account metadata and the confidence "
    "scores of the models that produced each link."
)

CARD_TEXT: dict[str, dict[str, str]] = {
    "full": {
        "title": "RS-Graph v2 (full, gated)",
        "summary": f"{SUMMARY} This is the **full** variant, gated with manual approval.",
        "access": "Request access with the form on this page. Requests are approved manually.",
        "contents": (
            "Everything in the public variant, plus the `document_abstract` table (article "
            "abstracts) and `developer_account.name` and `developer_account.email`."
        ),
        "texts": "article titles, article abstracts",
        "other": "**Public Variant:** https://huggingface.co/datasets/{org}/rs-graph-v2",
    },
    "public": {
        "title": "RS-Graph v2 (public)",
        "summary": f"{SUMMARY} This is the **public** variant, without abstracts and developer "
        "names or emails.",
        "access": "Open to everyone, no request needed.",
        "contents": (
            "Withheld from this variant: the `document_abstract` table (fetch abstracts from "
            "OpenAlex by DOI) and `developer_account.name` and `developer_account.email` "
            "(fetch the public profile from the GitHub API by `developer_account.username`). "
            "Every other table is identical to the full variant. Links between researchers "
            "and developer usernames are model inferences, each with its "
            "`predictive_model_confidence` in `researcher_developer_account_link`."
        ),
        "texts": "article titles",
        "other": "**Gated Access Variant:** "
        "https://huggingface.co/datasets/{org}/rs-graph-v2-full",
    },
}


class Variant(StrEnum):
    full = "full"
    public = "public"
    both = "both"


###############################################################################


def render_card(variant: str, org: str, tables: list[str]) -> str:
    """Render the dataset card for a variant from the template and the load snippet."""
    text = CARD_TEXT[variant]
    repo_id = f"{org}/{REPO_NAMES[variant]}"
    snippet = (TEMPLATES_DIR / "load_rs_graph.py").read_text().rstrip("\n")
    # The snippet's REPO_ID points at the public repo
    snippet = snippet.replace('sci-soft-collections/rs-graph-v2"', f'{repo_id}"')
    configs = "\n".join(
        f"- config_name: {t}\n  data_files:\n  - split: train\n    path: {t}/train-*"
        for t in tables
    )
    return Template((TEMPLATES_DIR / "dataset_card.md").read_text()).substitute(
        title=text["title"],
        configs=configs,
        gated=GATED_YAML if variant == "full" else "",
        summary=text["summary"],
        access=text["access"],
        contents=text["contents"],
        tables=", ".join(f"`{t}`" for t in tables),
        load_snippet=snippet,
        texts=text["texts"],
        other=text["other"].format(org=org),
    )


@app.command()
def upload_to_huggingface(  # noqa: C901
    database_path: str = typer.Argument(
        help="Path to the SQLite database file to use.",
    ),
    variant: Variant = Variant.both,
    org: str = "sci-soft-collections",
    include_repository_file: bool = True,
) -> None:
    """
    Read all tables from the rs-graph-v2 SQLite database and upload
    them to HuggingFace Hub as a DatasetDict, as a full variant, a public
    variant (no abstracts, developer names or emails), or both.

    Requires authentication via `huggingface-cli login` or the HF_TOKEN
    environment variable.
    """
    variants = ["full", "public"] if variant == Variant.both else [variant.value]

    # Load environment variables from .env file, if it exists
    load_dotenv()

    # Check for HF_TOKEN environment variable
    if not os.getenv("HF_TOKEN"):
        raise ValueError("Environment variable 'HF_TOKEN' is not set")

    print(f"Will upload variants {variants} to HuggingFace Hub org: {org}")

    # Output directories, every table is exported once into full/
    root_dir = Path("rs-graph-v2-hf-upload/")
    output_dir = root_dir / "full"
    public_dir = root_dir / "public"

    # Remove existing output directory if it exists to ensure a clean slate
    if root_dir.exists():
        shutil.rmtree(root_dir)

    # Remake
    output_dir.mkdir(parents=True)

    engine = get_engine(database_path)
    table_names = sa_inspect(engine).get_table_names()
    to_process_table_names = [
        table_name for table_name in table_names if table_name not in SKIPPED_TABLES
    ]

    # Handle repository file
    if not include_repository_file:
        if "repository_file" in to_process_table_names:
            to_process_table_names.remove("repository_file")

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

    # Build the public variant from the full export
    public_table_names = [
        table_name
        for table_name in to_process_table_names
        if table_name not in PUBLIC_DROPPED_TABLES
    ]
    if "public" in variants:
        public_dir.mkdir()
        for table_name in public_table_names:
            src = output_dir / f"{table_name}.parquet"
            dst = public_dir / f"{table_name}.parquet"
            if table_name in PUBLIC_DROPPED_COLUMNS:
                pl.scan_parquet(src).drop(PUBLIC_DROPPED_COLUMNS[table_name]).sink_parquet(dst)
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        # Schema-only checks, no row data is read
        if (public_dir / "document_abstract.parquet").exists():
            raise ValueError("document_abstract.parquet found in the public variant")
        dev_path = public_dir / "developer_account.parquet"
        if dev_path.exists():
            leaked = {"name", "email"} & set(pq.read_schema(dev_path).names)
            if leaked:
                raise ValueError(
                    f"developer_account has columns {leaked} in the public variant"
                )

    # Upload each table as a separate config, then the dataset card
    for name in variants:
        repo_id = f"{org}/{REPO_NAMES[name]}"
        variant_dir = output_dir if name == "full" else public_dir
        variant_tables = to_process_table_names if name == "full" else public_table_names
        print(f"Uploading {len(variant_tables)} tables to {repo_id}")
        for table_name in tqdm(
            variant_tables,
            desc=f"Uploading {name} tables",
        ):
            ds = Dataset.from_parquet(str(variant_dir / f"{table_name}.parquet"))
            assert isinstance(ds, Dataset)
            ds.push_to_hub(repo_id, config_name=table_name, private=True)
            del ds

        # Upload the card last, push_to_hub writes its own README for the configs
        card_path = variant_dir / "README.md"
        card_path.write_text(render_card(name, org, variant_tables))
        HfApi().upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

    print("Upload complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    app()
