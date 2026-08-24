"""Explore-the-Data page: one precomputed pairs index for browse + search.

Rewritten 2026-08-23 (Papers-with-Code-style redesign) to be pair-oriented
rather than entity-oriented. The previous version emitted two separate
record kinds -- one row per Document, one row per Repository -- mixed into a
single flat array (544,128 rows for the two entity tables combined). That
matched a plain "search everything" box, but it isn't the right shape for a
browse UI: a user looking at the page wants article+repo pairs (paper here,
code there), not documents and repositories as independent, unlinked things.

This version emits one record per `document_repository_link` row (still
restricted to the high-precision subset -- NULL or >=0.9994 confidence, same
threshold as everywhere else on the site), with the paired document and
repository's fields joined onto it directly. 282,057 pairs as of this run --
noticeably *fewer* rows than the old 544,128, because it's one row per pair
rather than one row per entity (a document/repository with multiple links
only appeared once as an entity before; now it appears once per pair, which
is usually fewer since most links are 1:1).

Size discipline, since pairs carry more fields per row than the old entity
records did:
- `doi` is stored bare (e.g. "10.1234/xyz"), not as a full
  "https://doi.org/..." URL -- the site prepends that prefix client-side.
  Same for the repo: stored as "owner/name", with "https://github.com/"
  prepended client-side. Both prefixes are constant per-kind, so storing them
  per-row would just be dead weight repeated 282k times.
- Field (specific research area, e.g. "Computer Science" -- there are 26 of
  these, see Topic.field_name) is stored as a small integer index into a
  `fields` lookup array shipped once at the top of the file, not as a repeated
  string. domain_name (4 broad domains) was considered instead since it's
  even smaller, but field_name is what the home page's "26 research fields"
  figure refers to and is far more informative on a per-item badge than a
  4-value domain would be -- and coding it as an int makes the string-
  repetition cost a non-issue either way.
- No abstract/README snippets (the old `--with-snippets` mode never shipped
  to the site -- build-search-index.mjs only ever copied the no-snippets
  file). Not reintroduced here.

Every document/repository in a rs-graph has at least one non-null title,
doi, cited_by_count, publication_date (document side) and owner, name,
stargazers_count (repository side) -- verified null-free this pass, so no
null-handling branches are needed for those fields. A document's field
lookup (via document_topic -> topic) can still be missing if a document has
no topic scores at all, so that one stays nullable.

Output shape (a JSON object, not a bare array, unlike the old format --
build-search-index.mjs just copies the file byte-for-byte either way, so this
is purely a site-side (search.astro) parsing concern):

    {
      "fields": ["Agricultural and Biological Sciences", ...],  // index -> field_name
      "records": [
        [title, doi, cited_by_count, publication_date, field_idx_or_null, repo_full_name, stars],
        ...
      ]
    }

`records` is pre-sorted by publication_date descending -- the site's default
"latest" view is then just "take the first N", no client-side sort needed.
"""

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
    repositories = load_table("repository").select(
        "id", "owner", "name", "stargazers_count"
    )

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
