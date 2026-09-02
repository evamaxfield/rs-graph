"""
THROWAWAY diagnostic script — cutoff (75 vs 85 vs 90) and name-column
(software_name vs software_name_normalized) sensitivity comparison for
rs_graph.utils.software_alignment.align_software_names(method="global_min_diff").

Not part of the paper pipeline. Uncommitted per standing instruction: never commit
without an explicit ask.

Pulls real article-repository pair data from HF `sci-soft-collections/rs-graph-v2-full`,
applies the standard paper filters (confidence >=0.9994 or NULL, published after 2008),
takes a random sample of pairs that have both >=1 import and >=1 mention, and for each
pair runs the import-vs-mention alignment at cutoffs 75/85/90, using both the raw
`software_name` column and the `software_name_normalized` column (populated via
`normalize_name()`, see `rs_graph/utils/identifier_normalization.py`), to see:
  1. how much the matched-pair set differs across cutoffs (75 vs 85 vs 90), and
  2. how much the matched-pair set differs between raw and normalized names.

Note on (2): `align_software_names` already runs every name it's given through
`normalize_name()` internally before scoring (see `software_alignment.py`), and that
function is idempotent (its transforms -- lowercasing, stripping hyphens/underscores/
spaces/newlines -- are all no-ops on a string that's already had them applied). So in
principle raw and normalized inputs should produce byte-identical similarity matrices
and match sets. This script verifies that empirically against real data rather than
assuming it.
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parents[3]))
from rs_graph.utils.identifier_normalization import normalize_name  # noqa: E402
from rs_graph.utils.software_alignment import _solve_global_min_diff  # noqa: E402
from rs_graph.utils.software_alternates import are_alternates  # noqa: E402

DATASET_REPO = "sci-soft-collections/rs-graph-v2-full"
RANDOM_SEED = 42
SAMPLE_SIZE = 10_000  # pairs with >=1 import and >=1 mention
CUTOFFS = [75.0, 85.0, 90.0]
NAME_COLUMNS = ["software_name", "software_name_normalized"]

SUPPLEMENTAL_DIR = Path(__file__).parent
RS_GRAPH_REPO_ROOT = Path(__file__).parents[3]

load_dotenv(str(RS_GRAPH_REPO_ROOT / ".env"))


def load_table(table: str) -> pl.DataFrame:
    ds = load_dataset(DATASET_REPO, table, split="train", token=os.environ.get("HF_TOKEN"))
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


def align_at_all_cutoffs(
    items_a: list[str],
    items_b: list[str],
    source_a: str,
    source_b: str,
    cutoffs: list[float],
    use_alternates: bool = True,
) -> dict[float, list[dict]]:
    """
    Same math as align_software_names, but builds the fuzzy similarity matrix once
    and thresholds it at every cutoff in `cutoffs`, instead of recomputing the matrix
    once per cutoff. With 3 cutoffs x 2 name columns x 10k pairs, recomputing the
    (rapidfuzz) matrix per cutoff would waste a real amount of time for no benefit,
    since the matrix itself doesn't depend on the cutoff at all.
    """
    if not items_a or not items_b:
        return {c: [] for c in cutoffs}

    lut_a = {orig: normalize_name(orig) for orig in items_a}
    lut_b = {orig: normalize_name(orig) for orig in items_b}
    norm_a = [lut_a[x] for x in items_a]
    norm_b = [lut_b[x] for x in items_b]

    sim_matrix = np.zeros((len(norm_b), len(norm_a)))
    for i, nb in enumerate(norm_b):
        for j, na in enumerate(norm_a):
            if use_alternates and are_alternates(na, nb):
                sim_matrix[i, j] = 100.0
            else:
                sim_matrix[i, j] = fuzz.ratio(nb, na)

    out: dict[float, list[dict]] = {}
    for cutoff in cutoffs:
        pairs = _solve_global_min_diff(sim_matrix, cutoff)
        out[cutoff] = [
            {
                "item_one_source": source_a,
                "item_one": items_a[j],
                "normalized_item_one": lut_a[items_a[j]],
                "item_two_source": source_b,
                "item_two": items_b[i],
                "normalized_item_two": lut_b[items_b[i]],
                "score": score,
            }
            for i, j, score in pairs
        ]
    return out


def main() -> None:
    print("Loading base tables from HuggingFace...")
    documents = load_table("document")
    article_repo_links = load_table("document_repository_link")
    repository_imports = load_table("repository_import")
    document_software_mentions = load_table("document_software_mention")
    print(f"  document: {len(documents):,} rows")
    print(f"  document_repository_link: {len(article_repo_links):,} rows")
    print(f"  repository_import: {len(repository_imports):,} rows")
    print(f"  document_software_mention: {len(document_software_mentions):,} rows")

    # Confirm software_name_normalized actually exists on both source tables before
    # assuming we can use it (per task instruction -- don't just assume the column
    # is present on the HF dataset the way it's present on the SQLModel schema).
    for col in NAME_COLUMNS:
        assert col in repository_imports.columns, f"{col!r} missing from repository_import"
        assert col in document_software_mentions.columns, (
            f"{col!r} missing from document_software_mention"
        )
    print(f"Confirmed both name columns present on both tables: {NAME_COLUMNS}")

    # ---- Standard paper filters ----
    merged = article_repo_links.select(
        pl.col("id").alias("document_repository_link_id"),
        "document_id",
        "repository_id",
        "predictive_model_confidence",
    ).join(
        documents.select(
            pl.col("id").alias("document_id"),
            pl.col("publication_date"),
        ),
        on="document_id",
    )
    print(f"After joining documents: {len(merged):,} pairs")

    merged = merged.with_columns(
        pl.col("publication_date").str.to_date("%Y-%m-%d", strict=False).alias("pub_date_parsed")
    ).with_columns(pl.col("pub_date_parsed").dt.year().alias("pub_year"))

    merged = merged.filter(pl.col("pub_year") >= 2008)
    print(f"After filtering to published after 2008: {len(merged):,} pairs")

    merged = merged.filter(
        (pl.col("predictive_model_confidence") >= 0.9994)
        | pl.col("predictive_model_confidence").is_null()
    )
    print(f"After filtering to confidence >=0.9994 or NULL: {len(merged):,} pairs")

    # ---- Restrict to pairs with >=1 import and >=1 mention ----
    repo_ids_with_imports = set(repository_imports.get_column("repository_id").unique().to_list())
    doc_ids_with_mentions = set(
        document_software_mentions.get_column("document_id").unique().to_list()
    )

    merged = merged.filter(
        pl.col("repository_id").is_in(repo_ids_with_imports)
        & pl.col("document_id").is_in(doc_ids_with_mentions)
    )
    print(f"After restricting to pairs with >=1 import and >=1 mention: {len(merged):,} pairs")

    # ---- Sample ----
    all_pairs = merged.select("document_repository_link_id", "document_id", "repository_id")
    n_available = len(all_pairs)
    sample_n = min(SAMPLE_SIZE, n_available)
    rng = random.Random(RANDOM_SEED)
    sample_idx = rng.sample(range(n_available), sample_n)
    sample = all_pairs[sample_idx]
    print(f"\nSampling {sample_n:,} of {n_available:,} eligible pairs (seed={RANDOM_SEED}).")

    # Pre-index imports/mentions for fast per-pair lookup, once per name column
    imports_by_repo_by_col = {
        col: {
            repo_id[0]: grp.get_column(col).drop_nulls().to_list()
            for repo_id, grp in repository_imports.group_by("repository_id")
        }
        for col in NAME_COLUMNS
    }
    mentions_by_doc_by_col = {
        col: {
            doc_id[0]: grp.get_column(col).drop_nulls().to_list()
            for doc_id, grp in document_software_mentions.group_by("document_id")
        }
        for col in NAME_COLUMNS
    }

    # ---- Run alignment for every (name_column, cutoff) combination, per pair ----
    # results[name_col][cutoff] -> list of match rows (dicts, with link/doc/repo ids)
    results: dict[str, dict[float, list[dict]]] = {
        col: {c: [] for c in CUTOFFS} for col in NAME_COLUMNS
    }

    n_pairs_tested = 0
    for row in sample.iter_rows(named=True):
        doc_id = row["document_id"]
        repo_id = row["repository_id"]
        link_id = row["document_repository_link_id"]

        # Only count a pair as "tested" once (against the raw column's availability;
        # the normalized column has the same row-presence by construction since it's
        # derived from the same source rows).
        raw_imports = imports_by_repo_by_col["software_name"].get(repo_id, [])
        raw_mentions = mentions_by_doc_by_col["software_name"].get(doc_id, [])
        if not raw_imports or not raw_mentions:
            continue
        n_pairs_tested += 1

        for col in NAME_COLUMNS:
            imports = imports_by_repo_by_col[col].get(repo_id, [])
            mentions = mentions_by_doc_by_col[col].get(doc_id, [])
            if not imports or not mentions:
                continue
            per_cutoff = align_at_all_cutoffs(
                items_a=imports,
                items_b=mentions,
                source_a="import",
                source_b="mention",
                cutoffs=CUTOFFS,
            )
            for cutoff, matches in per_cutoff.items():
                for m in matches:
                    results[col][cutoff].append(
                        {"link_id": link_id, "doc_id": doc_id, "repo_id": repo_id, **m}
                    )

    print(f"Pairs actually tested (both imports and mentions present): {n_pairs_tested:,}")

    dfs = {
        col: {
            c: (pl.DataFrame(results[col][c]) if results[col][c] else pl.DataFrame())
            for c in CUTOFFS
        }
        for col in NAME_COLUMNS
    }

    # ---- Per-column, per-cutoff match counts ----
    print("\n" + "=" * 78)
    print("MATCH COUNTS BY NAME COLUMN AND CUTOFF")
    print("=" * 78)
    for col in NAME_COLUMNS:
        counts = {c: len(dfs[col][c]) for c in CUTOFFS}
        print(f"{col}: " + ", ".join(f"cutoff={c:g} -> {n:,}" for c, n in counts.items()))

    # ---- Sanity check: stricter cutoff should never add matches absent at a looser one ----
    print("\n" + "=" * 78)
    print("MONOTONICITY SANITY CHECK (stricter cutoff must be a subset of looser one)")
    print("=" * 78)
    for col in NAME_COLUMNS:
        for looser, stricter in [(75.0, 85.0), (85.0, 90.0), (75.0, 90.0)]:
            df_loose = dfs[col][looser]
            df_strict = dfs[col][stricter]
            if len(df_loose) == 0 and len(df_strict) == 0:
                print(f"{col}: {looser:g}->{stricter:g}: both empty")
                continue
            keys_loose = (
                set(zip(df_loose["link_id"], df_loose["normalized_item_one"], df_loose["normalized_item_two"], strict=False))
                if len(df_loose) > 0
                else set()
            )
            keys_strict = (
                set(zip(df_strict["link_id"], df_strict["normalized_item_one"], df_strict["normalized_item_two"], strict=False))
                if len(df_strict) > 0
                else set()
            )
            n_new_at_strict = len(keys_strict - keys_loose)
            print(
                f"{col}: matches at {stricter:g} not present at {looser:g} "
                f"(should be 0): {n_new_at_strict}"
            )

    # ---- Cutoff sensitivity, per column: matches at 75 that don't survive to 85 / 90 ----
    print("\n" + "=" * 78)
    print("CUTOFF SENSITIVITY (matches present at 75, by whether they clear 85 / 90)")
    print("=" * 78)

    cutoff_sensitivity_tables: dict[str, pl.DataFrame] = {}
    for col in NAME_COLUMNS:
        df75 = dfs[col][75.0]
        if len(df75) == 0:
            cutoff_sensitivity_tables[col] = pl.DataFrame()
            continue
        keys_85 = (
            set(
                zip(
                    dfs[col][85.0]["link_id"],
                    dfs[col][85.0]["normalized_item_one"],
                    dfs[col][85.0]["normalized_item_two"],
                    strict=False,
                )
            )
            if len(dfs[col][85.0]) > 0
            else set()
        )
        keys_90 = (
            set(
                zip(
                    dfs[col][90.0]["link_id"],
                    dfs[col][90.0]["normalized_item_one"],
                    dfs[col][90.0]["normalized_item_two"],
                    strict=False,
                )
            )
            if len(dfs[col][90.0]) > 0
            else set()
        )
        annotated = df75.with_columns(
            pl.struct(["link_id", "normalized_item_one", "normalized_item_two"])
            .map_elements(
                lambda s: (s["link_id"], s["normalized_item_one"], s["normalized_item_two"]) in keys_85,
                return_dtype=pl.Boolean,
            )
            .alias("clears_85"),
            pl.struct(["link_id", "normalized_item_one", "normalized_item_two"])
            .map_elements(
                lambda s: (s["link_id"], s["normalized_item_one"], s["normalized_item_two"]) in keys_90,
                return_dtype=pl.Boolean,
            )
            .alias("clears_90"),
        ).sort("score", descending=True)
        cutoff_sensitivity_tables[col] = annotated

        n75 = len(df75)
        n_dropped_by_85 = n75 - int(annotated["clears_85"].sum())
        n_dropped_by_90 = n75 - int(annotated["clears_90"].sum())
        n_dropped_75_to_85_only = n_dropped_by_85
        n_dropped_85_to_90 = int(annotated["clears_85"].sum()) - int(annotated["clears_90"].sum())
        print(f"\n[{col}]")
        print(f"  matches @75: {n75:,}")
        print(f"  dropped between 75->85: {n_dropped_75_to_85_only:,} ({100 * n_dropped_75_to_85_only / n75:.1f}% of @75)")
        print(f"  dropped between 85->90: {n_dropped_85_to_90:,}")
        print(f"  total dropped 75->90: {n_dropped_by_90:,} ({100 * n_dropped_by_90 / n75:.1f}% of @75)")

    # ---- Raw vs normalized comparison, at each cutoff ----
    print("\n" + "=" * 78)
    print("RAW (software_name) vs NORMALIZED (software_name_normalized) COMPARISON")
    print("=" * 78)
    divergence_rows: list[dict] = []
    for cutoff in CUTOFFS:
        df_raw = dfs["software_name"][cutoff]
        df_norm = dfs["software_name_normalized"][cutoff]
        keys_raw = (
            set(zip(df_raw["link_id"], df_raw["normalized_item_one"], df_raw["normalized_item_two"], strict=False))
            if len(df_raw) > 0
            else set()
        )
        keys_norm = (
            set(zip(df_norm["link_id"], df_norm["normalized_item_one"], df_norm["normalized_item_two"], strict=False))
            if len(df_norm) > 0
            else set()
        )
        only_raw = keys_raw - keys_norm
        only_norm = keys_norm - keys_raw
        print(
            f"cutoff={cutoff:g}: raw matches={len(df_raw):,}, normalized matches={len(df_norm):,}, "
            f"present in raw only={len(only_raw)}, present in normalized only={len(only_norm)}"
        )
        for k in only_raw:
            divergence_rows.append(
                {"cutoff": cutoff, "present_in": "raw_only", "link_id": k[0], "norm_import": k[1], "norm_mention": k[2]}
            )
        for k in only_norm:
            divergence_rows.append(
                {"cutoff": cutoff, "present_in": "normalized_only", "link_id": k[0], "norm_import": k[1], "norm_mention": k[2]}
            )

    # ---- Known generic-word false-positive collisions: does raising the cutoff help? ----
    print("\n" + "=" * 78)
    print("KNOWN GENERIC-WORD COLLISIONS (from the prior 3,000-pair / cutoff=75-vs-90 round)")
    print("=" * 78)
    watch_pairs = {("coda", "code"), ("core", "code"), ("packaging", "package")}
    for col in NAME_COLUMNS:
        df75 = dfs[col][75.0]
        if len(df75) == 0:
            continue
        hits = df75.filter(
            pl.struct(["normalized_item_one", "normalized_item_two"]).map_elements(
                lambda s: (s["normalized_item_one"], s["normalized_item_two"]) in watch_pairs
                or (s["normalized_item_two"], s["normalized_item_one"]) in watch_pairs,
                return_dtype=pl.Boolean,
            )
        )
        print(f"[{col}] occurrences of watched generic-word collisions @75: {len(hits):,}")
        for c in [85.0, 90.0]:
            dfc = dfs[col][c]
            if len(dfc) == 0:
                still = 0
            else:
                still = len(
                    dfc.filter(
                        pl.struct(["normalized_item_one", "normalized_item_two"]).map_elements(
                            lambda s: (s["normalized_item_one"], s["normalized_item_two"]) in watch_pairs
                            or (s["normalized_item_two"], s["normalized_item_one"]) in watch_pairs,
                            return_dtype=pl.Boolean,
                        )
                    )
                )
            print(f"  still present @{c:g}: {still:,}")

    # ---- Save outputs ----
    SUPPLEMENTAL_DIR.mkdir(parents=True, exist_ok=True)

    main_table = cutoff_sensitivity_tables.get("software_name", pl.DataFrame())
    if len(main_table) > 0:
        out_path = SUPPLEMENTAL_DIR / "cutoff-sensitive-pairs.csv"
        main_table.write_csv(out_path)
        print(f"\nCutoff-sensitivity table (raw software_name, {len(main_table):,} rows @75, "
              f"annotated with clears_85/clears_90) written to: {out_path}")

    norm_table = cutoff_sensitivity_tables.get("software_name_normalized", pl.DataFrame())
    if len(norm_table) > 0:
        out_path_norm = SUPPLEMENTAL_DIR / "cutoff-sensitive-pairs-normalized.csv"
        norm_table.write_csv(out_path_norm)
        print(f"Cutoff-sensitivity table (software_name_normalized, {len(norm_table):,} rows @75) "
              f"written to: {out_path_norm}")

    div_df = pl.DataFrame(divergence_rows) if divergence_rows else pl.DataFrame()
    out_path_div = SUPPLEMENTAL_DIR / "raw-vs-normalized-divergence.csv"
    div_df.write_csv(out_path_div)
    print(
        f"Raw-vs-normalized divergence table ({len(div_df):,} rows -- expected to be 0, "
        f"since align_software_names normalizes any input it's given) written to: {out_path_div}"
    )

    print("\n" + "=" * 78)
    print(f"TOP 30 CUTOFF-SENSITIVE PAIRS (raw software_name, present@75, sorted by score desc)")
    print("=" * 78)
    if len(main_table) > 0:
        top = main_table.head(30)
        print(f"{'import':<28} {'mention':<28} {'score':>7} {'@85':>5} {'@90':>5}")
        print("-" * 78)
        for r in top.iter_rows(named=True):
            print(
                f"{r['item_one'][:27]:<28} {r['item_two'][:27]:<28} {r['score']:>7.2f} "
                f"{'Y' if r['clears_85'] else 'N':>5} {'Y' if r['clears_90'] else 'N':>5}"
            )


if __name__ == "__main__":
    main()
