"""Shared HuggingFace table loader for rs-graph data-prep scripts."""

import datetime
import os
from typing import Any

import polars as pl
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

DATASET_REPO = "sci-soft-collections/rs-graph-v2-full"

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
load_dotenv(_ENV_PATH)


def load_table(table: str) -> pl.DataFrame:
    """Load a single rs-graph table as a polars DataFrame."""
    token = os.environ.get("HF_TOKEN")
    revision = os.environ.get("RS_GRAPH_REVISION")
    ds = load_dataset(DATASET_REPO, table, split="train", token=token, revision=revision)
    return pl.from_arrow(ds.data.table)


def use_release_loader(loader: Any) -> None:
    """Point a load_rs_graph module at the repo and revision set in the environment."""
    loader.REPO_ID = os.environ.get("RS_GRAPH_REPO_ID", DATASET_REPO)
    revision = os.environ.get("RS_GRAPH_REVISION")
    load = loader.load_dataset
    loader.load_dataset = lambda *args, **kwargs: load(
        *args, **{**kwargs, "revision": revision}
    )


def run_meta() -> dict:
    """Provenance for an output JSON: repo, resolved commit sha and run date."""
    revision = os.environ.get("RS_GRAPH_REVISION")
    sha = (
        HfApi(token=os.environ.get("HF_TOKEN"))
        .dataset_info(DATASET_REPO, revision=revision)
        .sha
    )
    return {
        "repo_id": DATASET_REPO,
        "revision_sha": sha,
        "run_date": datetime.date.today().isoformat(),
    }
