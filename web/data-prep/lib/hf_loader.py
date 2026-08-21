"""Shared HuggingFace table loader for rs-graph data-prep scripts."""

import os

import polars as pl
from datasets import load_dataset
from dotenv import load_dotenv

DATASET_REPO = "evamxb/rs-graph-v2-full"

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
load_dotenv(_ENV_PATH)


def load_table(table: str) -> pl.DataFrame:
    """Load a single rs-graph table as a polars DataFrame."""
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(DATASET_REPO, table, split="train", token=token)
    return pl.from_arrow(ds.data.table)
