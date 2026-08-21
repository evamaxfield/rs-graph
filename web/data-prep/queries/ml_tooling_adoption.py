"""AI4Science: how has ML/AI-library adoption spread through scientific code?

For repositories in a high-precision article-repository link, flag any repo
that imports a canonical ML/AI library (RepositoryImport.software_name_normalized
matching torch/pytorch, tensorflow, keras, jax, or sklearn/scikitlearn), then
break the adoption rate out two ways: by repo creation year (2008-2024, the
2025 partial year excluded) and by scientific domain (via each document's
top-scoring topic -> Topic.domain_name). Same precompute pattern as Q1/Q3 --
RepositoryImport + confidence-filtered join + group-by, no new machinery.
"""

import json
import os

import polars as pl

from lib.confidence import (
    PUBLICATION_YEAR_FLOOR,
    filter_high_precision_document_repository_links,
)
from lib.hf_loader import load_table

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "ml_tooling_adoption.json")

# Canonical ML/AI library set. A judgment call -- excludes e.g. xgboost,
# lightgbm, huggingface/transformers, onnx -- documented here rather than ad
# hoc, doesn't change the qualitative story (torch alone is already the
# dominant driver, see Q3's top-imported-libraries ranking).
ML_LIBRARY_NAMES = {"torch", "pytorch", "tensorflow", "keras", "jax", "sklearn", "scikitlearn"}


def run() -> dict:
    # --- site-snippet:start ---
    hp_links = filter_high_precision_document_repository_links(
        load_table("document_repository_link")
    ).select("document_id", "repository_id")
    hp_repo_ids = hp_links.select("repository_id").unique()

    imports = load_table("repository_import").join(hp_repo_ids, on="repository_id", how="inner")
    ml_repo_ids = (
        imports.filter(pl.col("software_name_normalized").is_in(ML_LIBRARY_NAMES))
        .select("repository_id")
        .unique()
        .with_columns(has_ml_import=True)
    )

    repositories = load_table("repository").join(
        hp_repo_ids, left_on="id", right_on="repository_id", how="inner"
    )
    repos = repositories.join(
        ml_repo_ids, left_on="id", right_on="repository_id", how="left"
    ).with_columns(pl.col("has_ml_import").fill_null(False))
    repos = repos.with_columns(
        pl.col("creation_datetime").str.to_datetime().dt.year().alias("creation_year")
    ).filter(
        (pl.col("creation_year") >= PUBLICATION_YEAR_FLOOR) & (pl.col("creation_year") <= 2024)
    )

    by_year = (
        repos.group_by("creation_year")
        .agg(total=pl.len(), with_ml=pl.col("has_ml_import").sum())
        .with_columns(pct=(pl.col("with_ml") / pl.col("total") * 100).round(1))
        .sort("creation_year")
    )

    # domain breakdown: each document's best-scoring topic -> domain_name
    document_topic = load_table("document_topic")
    topic = load_table("topic").select("id", "domain_name").rename({"id": "topic_id"})
    best_topic = (
        document_topic.sort("score", descending=True)
        .group_by("document_id")
        .head(1)
        .join(topic, on="topic_id")
        .select("document_id", "domain_name")
    )
    repo_domain = (
        hp_links.join(best_topic, on="document_id")
        .select("repository_id", "domain_name")
        .unique(subset=["repository_id"], keep="first")
    )
    domain_repos = repo_domain.join(
        ml_repo_ids, on="repository_id", how="left"
    ).with_columns(pl.col("has_ml_import").fill_null(False))
    by_domain = (
        domain_repos.group_by("domain_name")
        .agg(total=pl.len(), with_ml=pl.col("has_ml_import").sum())
        .with_columns(pct=(pl.col("with_ml") / pl.col("total") * 100).round(1))
        .sort("total", descending=True)
    )
    # --- site-snippet:end ---

    series = [
        {
            "year": int(row["creation_year"]),
            "total": int(row["total"]),
            "with_ml": int(row["with_ml"]),
            "pct": float(row["pct"]),
        }
        for row in by_year.iter_rows(named=True)
    ]
    domains = [
        {
            "domain_name": row["domain_name"],
            "total": int(row["total"]),
            "with_ml": int(row["with_ml"]),
            "pct": float(row["pct"]),
        }
        for row in by_domain.iter_rows(named=True)
    ]

    return {
        "question": (
            "How has AI/ML-tooling adoption spread through scientific code, "
            "and does it vary by domain?"
        ),
        "methodology": (
            "document_repository_link filtered to NULL OR confidence >= 0.9994; "
            f"repo creation_year >= {PUBLICATION_YEAR_FLOOR} and <= 2024 (2025 excluded, partial "
            "year); ML/AI import set: torch/pytorch, tensorflow, keras, jax, sklearn/scikitlearn"
        ),
        "ml_library_names": sorted(ML_LIBRARY_NAMES),
        "series": series,
        "domains": domains,
    }


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
