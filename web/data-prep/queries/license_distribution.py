"""License-mix note for the Limitations section: what can you actually reuse.

Restricted to the same high-precision-linked repo population used everywhere
else on the site -- see lib/confidence.py.
"""

import json
import os

import polars as pl
from lib.confidence import filter_high_precision_document_repository_links
from lib.hf_loader import load_table

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "output", "license_distribution.json"
)

TOP_N = 8


def run() -> dict:
    # --- site-snippet:start ---
    hp_repo_ids = (
        filter_high_precision_document_repository_links(load_table("document_repository_link"))
        .select("repository_id")
        .unique()
    )

    repos = (
        load_table("repository")
        .rename({"id": "repository_id"})
        .join(hp_repo_ids, on="repository_id", how="inner")
    )
    # --- site-snippet:end ---

    n_repos = repos.height
    n_no_license = repos.filter(pl.col("license").is_null()).height
    licensed = repos.filter(pl.col("license").is_not_null())

    license_counts = (
        licensed.group_by("license")
        .agg(n=pl.len())
        .with_columns(pct=(pl.col("n") / n_repos * 100).round(1))
        .sort("n", descending=True)
        .head(TOP_N)
    )

    result = {
        "methodology": "document_repository_link filtered to NULL OR confidence >= 0.9994",
        "n_repos": n_repos,
        "n_no_license": n_no_license,
        "pct_no_license": round(n_no_license / n_repos * 100, 1),
        "top_licenses": [
            {"license": row["license"], "n": int(row["n"]), "pct": float(row["pct"])}
            for row in license_counts.iter_rows(named=True)
        ],
    }
    return result


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
