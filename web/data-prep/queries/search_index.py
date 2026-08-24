import json
import os

import polars as pl
from lib.confidence import filter_high_precision_document_repository_links
from lib.hf_loader import load_table

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "search_articles_repos.json")


def build_pairs() -> dict:
    links = filter_high_precision_document_repository_links(
        load_table("document_repository_link")
    ).select("document_id", "repository_id")

    documents = load_table("document").select(
        "id", "title", "doi", "cited_by_count", "publication_date"
    )
    repositories = load_table("repository").select("id", "owner", "name", "stargazers_count")

    # Each document's best-scoring topic -> field_name, same join pattern
    # used by ml_tooling_adoption.py / dependency_manifest_growth.py (there
    # it's domain_name; here we want the finer-grained field_name -- 26
    # values vs. 4 -- since it's far more informative per-item on a browse
    # card than the broad domain would be).
    document_topic = load_table("document_topic")
    topic = load_table("topic").select("id", "field_name").rename({"id": "topic_id"})
    best_field = (
        document_topic.sort("score", descending=True)
        .group_by("document_id")
        .head(1)
        .join(topic, on="topic_id")
        .select("document_id", "field_name")
    )

    fields = sorted(best_field["field_name"].unique().to_list())
    field_to_idx = {name: i for i, name in enumerate(fields)}

    pairs = (
        links.join(documents, left_on="document_id", right_on="id", how="inner")
        .join(repositories, left_on="repository_id", right_on="id", how="inner")
        .join(best_field, on="document_id", how="left")
        .with_columns(
            (pl.col("owner") + "/" + pl.col("name")).alias("repo_full_name"),
        )
        .sort("publication_date", descending=True)
    )

    records: list[list] = []
    for row in pairs.iter_rows(named=True):
        field_name = row["field_name"]
        records.append(
            [
                row["title"],
                row["doi"],
                row["cited_by_count"],
                row["publication_date"],
                field_to_idx.get(field_name) if field_name is not None else None,
                row["repo_full_name"],
                row["stargazers_count"],
            ]
        )

    return {"fields": fields, "records": records}


def run() -> dict:
    payload = build_pairs()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    return {
        "articles_repos_path": OUTPUT_PATH,
        "n_pairs": len(payload["records"]),
        "n_fields": len(payload["fields"]),
        "articles_repos_bytes": os.path.getsize(OUTPUT_PATH),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
