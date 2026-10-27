"""How often a paper's code has a contributor who is not one of its authors."""

import json
import os

import load_rs_graph
from lib.hf_loader import run_meta, use_release_loader

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "output", "non_author_contributors.json"
)

use_release_loader(load_rs_graph)


def run() -> dict:
    # --- site-snippet:start ---
    import polars as pl
    from load_rs_graph import load_filtered_pairs, load_table

    all_pairs = load_filtered_pairs()
    authors = load_table("document_contributor").select("document_id", "researcher_id").unique()
    devs = load_table("repository_contributor").select("repository_id", "developer_account_id")
    links = load_table("researcher_developer_account_link").select(
        "researcher_id",
        "developer_account_id",
        pl.col("predictive_model_confidence").alias("conf"),
    )

    # Cited papers with one repository each, last pushed within 90 days of publication
    pairs = all_pairs.filter(pl.col("document_cited_by_count") >= 1)
    pairs = pairs.filter(pl.col("document_id").is_unique())
    pairs = pairs.filter(pl.col("repository_id").is_unique())
    pushed = pl.col("repository_last_pushed_datetime").str.to_datetime(strict=False)
    published = pl.col("document_publication_date").str.to_date(strict=False).cast(pl.Datetime)
    pairs = pairs.filter((pushed - published).dt.total_days() <= 90)

    # At least 3 authors (up to the 97th percentile) and at least 1 code contributor
    n_authors = authors.group_by("document_id").agg(n_authors=pl.len())
    n_devs = devs.group_by("repository_id").agg(n_devs=pl.len())
    pairs = pairs.join(n_authors, on="document_id").join(n_devs, on="repository_id")
    pairs = pairs.filter(pl.col("n_authors") >= 3)
    pairs = pairs.filter(pl.col("n_authors") <= pl.col("n_authors").quantile(0.97, "linear"))

    # Each researcher keeps their best-scoring link among the repository's contributors
    pair_devs = pairs.select("document_id", "repository_id").join(devs, on="repository_id")
    pair_links = (
        pair_devs.join(links, on="developer_account_id")
        .sort("conf", descending=True)
        .unique(subset=["document_id", "researcher_id"], keep="first")
        .filter(pl.col("conf") >= 0.97)
        .join(
            authors.with_columns(is_author=True),
            on=["document_id", "researcher_id"],
            how="left",
        )
        .group_by("document_id", "developer_account_id")
        .agg(n_linked=pl.len(), is_author=pl.col("is_author").is_not_null().any())
    )

    # A contributor is a non-author unless linked to one of the paper's authors (those linked to
    # more than 3 researchers are skipped)
    non_author_docs = (
        pair_devs.join(pair_links, on=["document_id", "developer_account_id"], how="left")
        .filter(pl.col("n_linked").fill_null(0) <= 3)
        .filter(~pl.col("is_author").fill_null(False))
        .select("document_id")
        .unique()
    )
    pairs = pairs.with_columns(
        has_non_author=pl.col("document_id").is_in(non_author_docs["document_id"].implode())
    )

    # Papers without an assigned topic count in the headline only
    by_field = (
        pairs.drop_nulls("document_field_name")
        .group_by("document_domain_name", "document_field_name")
        .agg(n_papers=pl.len(), n_with_non_author_contributor=pl.col("has_non_author").sum())
        .with_columns(
            pct_non_author=(
                pl.col("n_with_non_author_contributor") / pl.col("n_papers") * 100
            ).round(1)
        )
        .rename({"document_domain_name": "domain", "document_field_name": "field"})
        .sort("n_papers", descending=True)
    )
    n_with_non_author = pairs.filter("has_non_author").height
    headline = {
        "n_papers": pairs.height,
        "n_with_non_author_contributor": n_with_non_author,
        "pct_non_author": round(n_with_non_author / pairs.height * 100, 1),
    }
    print(headline)
    print(by_field)
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
            "n_rows": by_field.height,
        },
        "headline": headline,
        "rows": by_field.to_dicts(),
    }


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
