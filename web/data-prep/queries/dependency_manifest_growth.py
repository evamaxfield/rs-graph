"""Q1: How has dependency-manifest adoption grown over time?

For repositories in a high-precision article-repository link, what fraction
declare at least one dependency (RepositoryDependency), by repo creation year.
Restricted to creation_year >= 2008 (GitHub's launch) -- earlier years are a
data-encoding artifact, not real linked code. See lib/confidence.py.

Also breaks the same series out by scientific domain (via each document's
top-scoring topic -> Topic.domain_name, same join pattern as
ml_tooling_adoption.py), for the site's domain-filter dropdown on this chart.
"""

import json
import os

import polars as pl

from lib.confidence import (
    PUBLICATION_YEAR_FLOOR,
    filter_high_precision_document_repository_links,
)
from lib.hf_loader import load_table

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "output", "dependency_manifest_growth.json"
)


def run() -> dict:
    # --- site-snippet:start ---
    doc_repo_links = filter_high_precision_document_repository_links(
        load_table("document_repository_link")
    )
    repositories = load_table("repository")
    dependencies = load_table("repository_dependency")

    hp_repo_ids = doc_repo_links.select("repository_id").unique()

    repos = repositories.join(hp_repo_ids, left_on="id", right_on="repository_id", how="inner")
    repos = repos.with_columns(
        pl.col("creation_datetime").str.to_datetime().dt.year().alias("creation_year")
    )
    repos = repos.filter(pl.col("creation_year") >= PUBLICATION_YEAR_FLOOR)

    repo_ids_with_deps = dependencies.select("repository_id").unique().with_columns(has_deps=True)
    repos = repos.join(repo_ids_with_deps, left_on="id", right_on="repository_id", how="left")
    repos = repos.with_columns(pl.col("has_deps").fill_null(False))

    by_year = (
        repos.group_by("creation_year")
        .agg(total=pl.len(), with_deps=pl.col("has_deps").sum())
        .with_columns(pct=(pl.col("with_deps") / pl.col("total") * 100).round(1))
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
        doc_repo_links.select("document_id", "repository_id")
        .join(best_topic, on="document_id")
        .select("repository_id", "domain_name")
        .unique(subset=["repository_id"], keep="first")
    )
    domain_repos = repos.join(repo_domain, left_on="id", right_on="repository_id", how="inner")
    top_domains = (
        domain_repos.group_by("domain_name").agg(n=pl.len()).sort("n", descending=True).head(4)
    )["domain_name"].to_list()
    by_domain_year = (
        domain_repos.filter(pl.col("domain_name").is_in(top_domains))
        .group_by(["domain_name", "creation_year"])
        .agg(total=pl.len(), with_deps=pl.col("has_deps").sum())
        .with_columns(pct=(pl.col("with_deps") / pl.col("total") * 100).round(1))
        .sort(["domain_name", "creation_year"])
    )
    # --- site-snippet:end ---

    by_year = by_year.filter(pl.col("creation_year") <= 2026)
    by_domain_year = by_domain_year.filter(pl.col("creation_year") <= 2026)

    series = [
        {
            "year": int(row["creation_year"]),
            "total": int(row["total"]),
            "with_deps": int(row["with_deps"]),
            "pct": float(row["pct"]),
        }
        for row in by_year.iter_rows(named=True)
    ]

    domains = {}
    for domain_name in top_domains:
        rows = by_domain_year.filter(pl.col("domain_name") == domain_name)
        domains[domain_name] = [
            {
                "year": int(row["creation_year"]),
                "total": int(row["total"]),
                "with_deps": int(row["with_deps"]),
                "pct": float(row["pct"]),
            }
            for row in rows.iter_rows(named=True)
        ]

    return {
        "question": "How has dependency-manifest adoption grown over time?",
        "methodology": (
            "document_repository_link filtered to NULL OR confidence >= 0.9994; "
            f"repo creation_year >= {PUBLICATION_YEAR_FLOOR}"
        ),
        "series": series,
        "domains": domains,
    }


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
