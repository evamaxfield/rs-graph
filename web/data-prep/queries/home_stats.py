"""Home page's dataset-at-a-glance stats.

Restricted to the same high-precision, year-floored population used
everywhere else on the site -- see /methodology for the confidence filter
and lib/confidence.py's PUBLICATION_YEAR_FLOOR for the 2008 (GitHub launch)
floor rationale.
"""

import json
import os

import polars as pl

from lib.confidence import (
    PUBLICATION_YEAR_FLOOR,
    filter_high_precision_document_repository_links,
)
from lib.hf_loader import load_table

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "home_stats.json")


def run() -> dict:
    # --- site-snippet:start ---
    hp_links = filter_high_precision_document_repository_links(
        load_table("document_repository_link")
    ).select("document_id", "repository_id")

    documents = load_table("document").with_columns(
        pl.col("publication_date").str.to_date().dt.year().alias("publication_year")
    )
    documents = documents.filter(pl.col("publication_year") >= PUBLICATION_YEAR_FLOOR)

    filtered_links = hp_links.join(
        documents.select("id", "publication_year").rename({"id": "document_id"}),
        on="document_id",
        how="inner",
    )

    topic = load_table("topic")

    # distinct publication venues (journals, conferences, etc.) behind the
    # filtered document population -- via Document -> Location -> Source
    locations = load_table("location").select("id", "source_id").rename({"id": "location_id"})
    sources = load_table("source").select("id", "name", "source_type").rename({"id": "source_id"})
    venue_docs = documents.join(
        filtered_links.select("document_id").unique(), left_on="id", right_on="document_id", how="semi"
    )
    venues = (
        venue_docs.select(pl.col("primary_location_id").alias("location_id"))
        .drop_nulls()
        .join(locations, on="location_id", how="inner")
        .join(sources, on="source_id", how="inner")
    )
    n_distinct_venues = venues["name"].n_unique()
    venue_type_counts = (
        venues.select("name", "source_type")
        .unique()
        .group_by("source_type")
        .agg(n=pl.len())
        .sort("n", descending=True)
    )
    # --- site-snippet:end ---

    return {
        "methodology": (
            "document_repository_link filtered to NULL OR confidence >= 0.9994; "
            f"publication_year >= {PUBLICATION_YEAR_FLOOR}"
        ),
        "n_documents": filtered_links["document_id"].n_unique(),
        "n_repositories": filtered_links["repository_id"].n_unique(),
        "n_links": filtered_links.height,
        "year_min": int(filtered_links["publication_year"].min()),
        "year_max": int(filtered_links["publication_year"].max()),
        "n_domains": topic["domain_name"].n_unique(),
        "n_fields": topic["field_name"].n_unique(),
        "n_distinct_venues": n_distinct_venues,
        "venue_type_counts": [
            {"source_type": row["source_type"], "n": int(row["n"])}
            for row in venue_type_counts.iter_rows(named=True)
        ],
    }


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
