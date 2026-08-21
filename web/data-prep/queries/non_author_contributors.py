"""Q2: whether code contributions are recognized as authorship (qss-code-authors replication).

Matches repository contributors to researchers via ResearcherDeveloperAccountLink
(confidence >= 0.9), restricted throughout to high-precision article-repository links.
See doi.org/10.1162/QSS.a.465 for the original v1 finding this replicates.
"""

import json
import os

import polars as pl
from lib.confidence import (
    filter_high_confidence_researcher_developer_links,
    filter_high_precision_document_repository_links,
)
from lib.hf_loader import load_table

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "output", "non_author_contributors.json"
)


def run() -> dict:
    # --- site-snippet:start ---
    hp_links = filter_high_precision_document_repository_links(
        load_table("document_repository_link")
    ).select("document_id", "repository_id")
    repo_contributors = load_table("repository_contributor").select(
        "repository_id", "developer_account_id"
    )
    rdal = filter_high_confidence_researcher_developer_links(
        load_table("researcher_developer_account_link")
    ).select("researcher_id", "developer_account_id")
    authors = load_table("document_contributor").select("document_id", "researcher_id").unique()

    # every confidently-matched code contributor's researcher_id, per document
    doc_contributor_researchers = (
        hp_links.join(repo_contributors, on="repository_id")
        .join(rdal, on="developer_account_id")
        .select("document_id", "researcher_id")
        .unique()
    )

    # contributors who are NOT listed authors on that document
    non_author_rows = doc_contributor_researchers.join(
        authors, on=["document_id", "researcher_id"], how="anti"
    )

    unrestricted_docs = hp_links.select("document_id").unique()
    matched_docs = doc_contributor_researchers.select("document_id").unique()
    non_author_docs = non_author_rows.select("document_id").unique()

    # paper-equivalent filter: single linked repository, 3-11 listed authors
    repo_count_per_doc = hp_links.group_by("document_id").agg(n_repos=pl.len())
    author_count_per_doc = authors.group_by("document_id").agg(n_authors=pl.len())
    paper_docs = (
        repo_count_per_doc.filter(pl.col("n_repos") == 1)
        .join(author_count_per_doc, on="document_id")
        .filter(pl.col("n_authors").is_between(3, 11))
        .select("document_id")
    )
    paper_non_author_docs = non_author_docs.join(paper_docs, on="document_id")
    # --- site-snippet:end ---

    n_unrestricted = unrestricted_docs.height
    n_matched = matched_docs.height
    n_non_author = non_author_docs.height
    n_paper = paper_docs.height
    n_paper_non_author = paper_non_author_docs.height

    result = {
        "question": "Are code contributions recognized as authorship?",
        "methodology": (
            "document_repository_link filtered to NULL OR confidence >= 0.9994; "
            "researcher_developer_account_link filtered to confidence >= 0.9"
        ),
        "unrestricted": {
            "n_hp_linked_documents": n_unrestricted,
            "n_with_confidently_matched_contributor": n_matched,
            "n_with_non_author_contributor": n_non_author,
            "pct_of_all_hp_linked": round(n_non_author / n_unrestricted * 100, 2),
            "pct_of_matched": round(n_non_author / n_matched * 100, 2),
        },
        "paper_equivalent": {
            "description": "single linked repository, 3-11 listed authors",
            "n_qualifying_documents": n_paper,
            "n_with_non_author_contributor": n_paper_non_author,
            "pct": round(n_paper_non_author / n_paper * 100, 2),
            "published_v1_finding_pct": 28.6,
        },
    }
    return result


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
