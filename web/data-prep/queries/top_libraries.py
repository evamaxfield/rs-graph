"""Q3: what libraries this science actually runs on, and whether the paper text says so.

Three parallel rankings (imported, manifest-declared, mentioned-in-text), a
per-scientific-field normalized breakdown, and an import-vs-mention gap metric.
All restricted to high-precision article-repository links. Manifest-declared
dependencies are split per-ecosystem (pypi/npm/cran) rather than pooled, since
pooled top-10 is entirely pypi and gives nothing to color-encode on site.
"""

import json
import os

import polars as pl
from lib.confidence import filter_high_precision_document_repository_links
from lib.hf_loader import load_table

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "top_libraries.json")

TOP_N = 10
MIN_IMPORTING_REPOS_FOR_GAP = 50


def _top_n_by_distinct(df: pl.DataFrame, id_col: str, name_col: str, n: int) -> pl.DataFrame:
    return (
        df.group_by(name_col)
        .agg(count=pl.col(id_col).n_unique())
        .sort("count", descending=True)
        .head(n)
    )


def run() -> dict:
    # --- site-snippet:start ---
    hp_links = filter_high_precision_document_repository_links(
        load_table("document_repository_link")
    ).select("document_id", "repository_id")
    hp_repo_ids = hp_links.select("repository_id").unique()
    hp_doc_ids = hp_links.select("document_id").unique()

    imports = load_table("repository_import").join(hp_repo_ids, on="repository_id", how="inner")
    dependencies = load_table("repository_dependency").join(
        hp_repo_ids, on="repository_id", how="inner"
    )
    mentions = load_table("document_software_mention").join(
        hp_doc_ids, on="document_id", how="inner"
    )

    top_imports = _top_n_by_distinct(
        imports, "repository_id", "software_name_normalized", TOP_N
    )
    top_mentions = _top_n_by_distinct(
        mentions, "document_id", "software_name_normalized", TOP_N
    )

    # Top dependencies split by package ecosystem rather than pooled overall --
    # pooled top-10 is entirely pypi (scientific software skews Python), which
    # gives nothing to color-encode. Top N per ecosystem across the three
    # largest language ecosystems is the more informative, genuinely
    # multi-colored version of this chart.
    top_ecosystems = ["pypi", "npm", "cran"]
    ecosystem_top_n = 5
    dependencies_eco = dependencies.with_columns(pl.col("ecosystem").fill_null("unknown"))
    top_dependencies_rows = []
    for ecosystem in top_ecosystems:
        eco_top = _top_n_by_distinct(
            dependencies_eco.filter(pl.col("ecosystem") == ecosystem),
            "repository_id",
            "software_name_normalized",
            ecosystem_top_n,
        )
        for row in eco_top.iter_rows(named=True):
            top_dependencies_rows.append(
                {
                    "name": row["software_name_normalized"],
                    "count": int(row["count"]),
                    "ecosystem": ecosystem,
                }
            )
    top_dependencies_rows.sort(key=lambda r: r["count"], reverse=True)

    # field normalization: each document's best-scoring topic -> field_name
    document_topic = load_table("document_topic")
    topic = load_table("topic").select("id", "field_name").rename({"id": "topic_id"})
    best_topic = (
        document_topic.sort("score", descending=True)
        .group_by("document_id")
        .head(1)
        .join(topic, on="topic_id")
        .select("document_id", "field_name")
    )
    repo_field = (
        hp_links.join(best_topic, on="document_id")
        .select("repository_id", "field_name")
        .unique(subset=["repository_id"], keep="first")
    )
    field_sizes = (
        repo_field.group_by("field_name").agg(n_repos=pl.len()).sort("n_repos", descending=True)
    )
    top_fields = field_sizes.head(8)["field_name"].to_list()

    field_slices = []
    for field_name in top_fields:
        field_repo_ids = repo_field.filter(pl.col("field_name") == field_name).select(
            "repository_id"
        )
        n_field_repos = field_repo_ids.height
        field_imports = imports.join(field_repo_ids, on="repository_id", how="inner")
        lib_pcts = (
            field_imports.group_by("software_name_normalized")
            .agg(n=pl.col("repository_id").n_unique())
            .with_columns(pct=(pl.col("n") / n_field_repos * 100).round(1))
            .sort("n", descending=True)
            .head(TOP_N)
        )
        field_slices.append(
            {
                "field_name": field_name,
                "n_repos": n_field_repos,
                "top_libraries": [
                    {"name": row["software_name_normalized"], "pct": float(row["pct"])}
                    for row in lib_pcts.iter_rows(named=True)
                ],
            }
        )

    # import-vs-mention gap: libraries with >= MIN_IMPORTING_REPOS_FOR_GAP importing repos
    n_hp_repos = hp_repo_ids.height
    n_hp_docs = hp_doc_ids.height
    import_pcts = (
        imports.group_by("software_name_normalized")
        .agg(n=pl.col("repository_id").n_unique())
        .filter(pl.col("n") >= MIN_IMPORTING_REPOS_FOR_GAP)
        .with_columns(import_pct=(pl.col("n") / n_hp_repos * 100))
    )
    mention_pcts = (
        mentions.group_by("software_name_normalized")
        .agg(n_mentions=pl.col("document_id").n_unique())
        .with_columns(mention_pct=(pl.col("n_mentions") / n_hp_docs * 100))
    )
    gap = (
        import_pcts.join(mention_pcts, on="software_name_normalized", how="left")
        .with_columns(pl.col("mention_pct").fill_null(0.0))
        .with_columns(gap_points=(pl.col("import_pct") - pl.col("mention_pct")))
        .sort("gap_points", descending=True)
        .head(TOP_N)
    )
    # --- site-snippet:end ---

    result = {
        "question": (
            "What libraries does this science actually run on -- "
            "and does the paper text say so?"
        ),
        "methodology": "document_repository_link filtered to NULL OR confidence >= 0.9994",
        "n_hp_repos": n_hp_repos,
        "n_hp_docs": n_hp_docs,
        "top_imported": [
            {"name": row["software_name_normalized"], "count": int(row["count"])}
            for row in top_imports.iter_rows(named=True)
        ],
        "top_dependencies": top_dependencies_rows,
        "top_mentions": [
            {"name": row["software_name_normalized"], "count": int(row["count"])}
            for row in top_mentions.iter_rows(named=True)
        ],
        "field_slices": field_slices,
        "import_vs_mention_gap": [
            {
                "name": row["software_name_normalized"],
                "import_pct": round(float(row["import_pct"]), 2),
                "mention_pct": round(float(row["mention_pct"]), 2),
                "gap_points": round(float(row["gap_points"]), 1),
            }
            for row in gap.iter_rows(named=True)
        ],
    }
    return result


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
