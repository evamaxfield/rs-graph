"""Q3 supplement: a per-library lookup table joining all three views.

Each of the three Q3 rankings (imported, manifest-declared dependencies,
mentioned-in-text) uses a different naming convention for the same underlying
library -- torch/pytorch being the canonical example -- and top_libraries.py
only ever surfaces the top 10 of each in isolation. This builds one row per
library, for the union of the top 200 (by distinct count) in each view, so a
user can look up any one library and see where it lands (or doesn't) across
all three. Dependencies are pooled across ecosystems into one count/rank
(this is a lookup table, not a chart needing ecosystem color), with the
top-contributing ecosystem retained as a side signal.
"""

import json
import os

import polars as pl
from lib.confidence import filter_high_precision_document_repository_links
from lib.hf_loader import load_table

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "library_cross_view.json")

TOP_N_PER_VIEW = 200


def _ranked_counts(df: pl.DataFrame, id_col: str, name_col: str) -> pl.DataFrame:
    return (
        df.group_by(name_col)
        .agg(count=pl.col(id_col).n_unique())
        .sort("count", descending=True)
        .with_columns(rank=pl.int_range(1, pl.len() + 1))
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

    import_ranked = _ranked_counts(imports, "repository_id", "software_name_normalized")
    mention_ranked = _ranked_counts(mentions, "document_id", "software_name_normalized")

    # Dependencies pooled across ecosystems -- one row per (name, repository_id)
    # regardless of which ecosystem(s) declared it, so a repo declaring the same
    # normalized name in two ecosystems only counts once.
    dependencies_pooled = dependencies.select(
        "repository_id", "software_name_normalized", "ecosystem"
    ).unique(subset=["repository_id", "software_name_normalized"], keep="first")
    dependency_ranked = _ranked_counts(
        dependencies_pooled, "repository_id", "software_name_normalized"
    )
    # Top-contributing ecosystem per library: whichever ecosystem accounts for
    # the most distinct declaring repos for that name.
    dependency_top_ecosystem = (
        dependencies_pooled.group_by(["software_name_normalized", "ecosystem"])
        .agg(n=pl.col("repository_id").n_unique())
        .sort("n", descending=True)
        .group_by("software_name_normalized")
        .head(1)
        .select(
            "software_name_normalized",
            pl.col("ecosystem").alias("dependency_top_ecosystem"),
        )
    )

    union_names = (
        set(import_ranked.head(TOP_N_PER_VIEW)["software_name_normalized"].to_list())
        | set(dependency_ranked.head(TOP_N_PER_VIEW)["software_name_normalized"].to_list())
        | set(mention_ranked.head(TOP_N_PER_VIEW)["software_name_normalized"].to_list())
    )
    union_df = pl.DataFrame({"software_name_normalized": sorted(union_names)})

    merged = (
        union_df.join(
            import_ranked.rename({"count": "import_count", "rank": "import_rank"}),
            on="software_name_normalized",
            how="left",
        )
        .join(
            dependency_ranked.rename({"count": "dependency_count", "rank": "dependency_rank"}),
            on="software_name_normalized",
            how="left",
        )
        .join(dependency_top_ecosystem, on="software_name_normalized", how="left")
        .join(
            mention_ranked.rename({"count": "mention_count", "rank": "mention_rank"}),
            on="software_name_normalized",
            how="left",
        )
    )
    # Only keep a view's count/rank if the name is actually within that view's
    # own top N -- the union can include a name that is top-200 in one view but
    # ranked, say, #350 in another; that tail rank isn't meaningful to expose.
    merged = merged.with_columns(
        [
            pl.when(pl.col("import_rank") <= TOP_N_PER_VIEW)
            .then(pl.col("import_count"))
            .otherwise(None)
            .alias("import_count"),
            pl.when(pl.col("import_rank") <= TOP_N_PER_VIEW)
            .then(pl.col("import_rank"))
            .otherwise(None)
            .alias("import_rank"),
            pl.when(pl.col("dependency_rank") <= TOP_N_PER_VIEW)
            .then(pl.col("dependency_count"))
            .otherwise(None)
            .alias("dependency_count"),
            pl.when(pl.col("dependency_rank") <= TOP_N_PER_VIEW)
            .then(pl.col("dependency_rank"))
            .otherwise(None)
            .alias("dependency_rank"),
            pl.when(pl.col("dependency_rank") <= TOP_N_PER_VIEW)
            .then(pl.col("dependency_top_ecosystem"))
            .otherwise(None)
            .alias("dependency_top_ecosystem"),
            pl.when(pl.col("mention_rank") <= TOP_N_PER_VIEW)
            .then(pl.col("mention_count"))
            .otherwise(None)
            .alias("mention_count"),
            pl.when(pl.col("mention_rank") <= TOP_N_PER_VIEW)
            .then(pl.col("mention_rank"))
            .otherwise(None)
            .alias("mention_rank"),
        ]
    ).sort("import_count", descending=True, nulls_last=True)

    n_hp_repos = hp_repo_ids.height
    n_hp_docs = hp_doc_ids.height
    # --- site-snippet:end ---

    rows = []
    for row in merged.iter_rows(named=True):
        rows.append(
            {
                "name": row["software_name_normalized"],
                "import_count": (
                    int(row["import_count"]) if row["import_count"] is not None else None
                ),
                "import_rank": (
                    int(row["import_rank"]) if row["import_rank"] is not None else None
                ),
                "dependency_count": (
                    int(row["dependency_count"]) if row["dependency_count"] is not None else None
                ),
                "dependency_rank": (
                    int(row["dependency_rank"]) if row["dependency_rank"] is not None else None
                ),
                "dependency_top_ecosystem": row["dependency_top_ecosystem"],
                "mention_count": (
                    int(row["mention_count"]) if row["mention_count"] is not None else None
                ),
                "mention_rank": (
                    int(row["mention_rank"]) if row["mention_rank"] is not None else None
                ),
            }
        )

    result = {
        "question": (
            "For a given library, where does it land across imports, declared "
            "dependencies, and paper-text mentions?"
        ),
        "methodology": (
            f"Union of top-{TOP_N_PER_VIEW}-by-distinct-count in each of the three "
            "views (imports, dependencies pooled across ecosystems, mentions), "
            "restricted to high-precision article-repository links. A null "
            f"count/rank means the library fell outside that view's top {TOP_N_PER_VIEW} "
            "-- not that it was measured and found at zero."
        ),
        "top_n_per_view": TOP_N_PER_VIEW,
        "n_hp_repos": n_hp_repos,
        "n_hp_docs": n_hp_docs,
        "rows": rows,
    }
    return result


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {len(result['rows'])} rows to {OUTPUT_PATH}")
