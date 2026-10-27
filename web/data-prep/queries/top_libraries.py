"""Which libraries a paper's code imports that the paper never mentions."""

import json
import os

import load_rs_graph
from lib.hf_loader import run_meta, use_release_loader

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "top_libraries.json")

use_release_loader(load_rs_graph)


def run() -> dict:
    # --- site-snippet:start ---
    # pip install rapidfuzz scipy numpy pyyaml
    # pip install "rs-graph @ git+https://github.com/evamaxfield/rs-graph"
    import polars as pl
    from load_rs_graph import load_filtered_pairs, load_table

    from rs_graph.utils.software_alignment import align_software_names

    all_pairs = load_filtered_pairs()
    # Software-mention extraction is complete only through 2022
    pairs = all_pairs.filter(pl.col("document_publication_year") <= 2022)
    pairs = pairs.unique(subset=["document_id", "repository_id"])

    imports = load_table("repository_import")
    mentions = load_table("document_software_mention")
    # Cut mention names at the first version or junk character, then drop non-names
    mentions = mentions.with_columns(
        pl.col("software_name_normalized")
        .str.replace(r"[<>=!~;@,()\[\]{}#%*:/\\|$?\"'`].*$", "")
        .str.strip_chars(".")
    ).filter(pl.col("software_name_normalized").str.contains(r"^[a-z0-9]"))

    def names_by(table: pl.DataFrame, key: str) -> dict[int, list[str]]:
        grouped = table.group_by(key).agg(pl.col("software_name_normalized").drop_nulls())
        return dict(
            zip(grouped[key], grouped["software_name_normalized"].to_list(), strict=True)
        )

    imports_by_repo = names_by(imports, "repository_id")
    mentions_by_doc = names_by(mentions, "document_id")

    # Per pair, align import names to mention names one-to-one; unmatched imports go unmentioned
    rows, n_eligible_pairs = [], 0
    for doc_id, repo_id, field in pairs.select(
        "document_id", "repository_id", "document_field_name"
    ).iter_rows():
        if repo_id not in imports_by_repo or doc_id not in mentions_by_doc:
            continue
        n_eligible_pairs += 1
        matches = align_software_names(
            items_a=imports_by_repo[repo_id],
            items_b=mentions_by_doc[doc_id],
            source_a="import",
            source_b="mention",
            cutoff=85.0,
            method="global_min_diff",
        )
        matched = {m.normalized_item_one for m in matches}
        rows.extend(
            (field, name, name not in matched) for name in set(imports_by_repo[repo_id])
        )
    per_import = pl.DataFrame(rows, schema=["field", "library", "not_mentioned"], orient="row")

    # Pairs without an assigned field count under "All fields" only
    in_field = per_import.drop_nulls("field")
    libraries = (
        pl.concat([in_field, per_import.with_columns(field=pl.lit("All fields"))])
        .group_by("field", "library")
        .agg(n_pairs_importing=pl.len(), n_pairs_not_mentioned=pl.col("not_mentioned").sum())
        .with_columns(
            pct_not_mentioned=(
                pl.col("n_pairs_not_mentioned") / pl.col("n_pairs_importing") * 100
            ).round(1)
        )
        .sort(["n_pairs_not_mentioned", "library"], descending=[True, False])
        .group_by("field", maintain_order=True)
        .head(50)
    )

    top = libraries.filter(pl.col("field") == "All fields")
    headline = {"n_eligible_pairs": n_eligible_pairs, **top.row(0, named=True)}
    print(headline)
    print(top.head(5))
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
            "n_rows": libraries.height,
        },
        "headline": headline,
        "rows": libraries.to_dicts(),
    }


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
