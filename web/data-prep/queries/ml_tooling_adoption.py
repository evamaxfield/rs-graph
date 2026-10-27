"""How fast ML libraries and dependency manifests spread among repositories, 2010-2024."""

import json
import os

import load_rs_graph
from lib.hf_loader import run_meta, use_release_loader

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "output", "ml_tooling_adoption.json"
)

use_release_loader(load_rs_graph)


def run() -> dict:
    # --- site-snippet:start ---
    import polars as pl
    from load_rs_graph import load_filtered_pairs, load_table

    ml_library_names = (
        "torch pytorch tensorflow keras jax flax sklearn scikitlearn transformers "
        "sentencetransformers xgboost lightgbm catboost onnx onnxruntime diffusers accelerate "
        "peft timm"
    ).split()

    all_pairs = load_filtered_pairs()
    imports = load_table("repository_import")
    ml_repos = imports.filter(pl.col("software_name_normalized").is_in(ml_library_names))
    ml_repos = ml_repos["repository_id"].unique().implode()
    manifest_repos = load_table("repository_dependency")["repository_id"].unique().implode()

    # A repository counts once per domain of its linked papers, and once in "All domains"
    repos = all_pairs.select(
        "repository_id",
        pl.col("repository_creation_year").alias("year"),
        pl.col("document_domain_name").alias("domain"),
    ).unique()
    repos = repos.filter(pl.col("year").is_between(2010, 2024))
    repos = pl.concat(
        [
            repos.drop_nulls("domain"),
            repos.unique(subset="repository_id").with_columns(domain=pl.lit("All domains")),
        ]
    )

    adoption = (
        repos.with_columns(
            **{
                "ML library import": pl.col("repository_id").is_in(ml_repos),
                "Dependency manifest": pl.col("repository_id").is_in(manifest_repos),
            }
        )
        .unpivot(
            on=["ML library import", "Dependency manifest"],
            index=["year", "domain"],
            variable_name="series",
            value_name="flagged",
        )
        .group_by("year", "domain", "series")
        .agg(n_repos=pl.len(), n_flagged=pl.col("flagged").sum())
        .with_columns(pct=(pl.col("n_flagged") / pl.col("n_repos") * 100).round(1))
        .sort("domain", "series", "year")
    )

    overall = adoption.filter(pl.col("domain") == "All domains")
    headline = {}
    for series, key in [("ML library import", "ml"), ("Dependency manifest", "manifest")]:
        rows = overall.filter(pl.col("series") == series).sort("year")
        headline[f"{key}_pct_2010"] = rows["pct"][0]
        headline[f"{key}_pct_2024"] = rows["pct"][-1]
        headline[f"{key}_n_repos_2024"] = rows["n_repos"][-1]
    print(headline)
    print(overall.sort("series", "year"))
    # --- site-snippet:end ---

    return {
        "meta": {
            **run_meta(),
            "thresholds": {
                "pair_confidence": load_rs_graph.PAIR_CONFIDENCE_THRESHOLD,
                "identity_confidence": load_rs_graph.IDENTITY_CONFIDENCE_THRESHOLD,
                "min_publication_year": load_rs_graph.MIN_PUBLICATION_YEAR,
            },
            "n_pairs": all_pairs.height,
            "n_rows": adoption.height,
        },
        "headline": headline,
        "rows": adoption.to_dicts(),
    }


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
