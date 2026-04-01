#!/usr/bin/env python

import json
import logging
import operator
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import connectorx  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
import typer
from scipy.stats import chi2_contingency, mannwhitneyu
from scipy.stats.contingency import association
from tqdm import tqdm

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.db import constants as db_constants
from rs_graph.utils.identifier_normalization import (
    normalize_doi_col,
    prep_name_for_printing,
)
from rs_graph.utils.software_alignment import (
    AlignmentMethod,
    PairwiseAlignmentResult,
    align_software_names,
)

###############################################################################
# Constants
###############################################################################

THIS_DIR = Path(__file__).parent.resolve()

MENTION_EXCLUDE_NORMALIZED: set[str] = {
    "code",
    "latex",
    "script",
    "scripts",
    "codes",
    "library",
    "libraries",
    "package",
    "packages",
    "api",
}

SCORE_CUTOFF = 75.0

###############################################################################
# Logger & App
###############################################################################

log = logging.getLogger(__name__)
app = typer.Typer()

###############################################################################
# DB Helpers
###############################################################################


def _read_table(table: str) -> pl.DataFrame:
    """Read a table from the v2 database."""
    log.info(f"Reading table: {table}")
    return pl.read_database_uri(
        f"SELECT * FROM {table}",
        f"sqlite:///{db_constants.V2_DATABASE_PATHS.dev}",
    )


###############################################################################
# Data Loading
###############################################################################


def load_pairs() -> pl.DataFrame:
    # Read all the tables we need
    dataset_sources = _read_table("dataset_source")
    docs = _read_table("document")
    repos = _read_table("repository")
    pairs = _read_table("document_repository_link")
    doc_topics = _read_table("document_topic")
    topics = _read_table("topic")
    # Drop to unique doc and unique repo in pairs
    pairs = pairs.unique(
        subset="document_id",
        keep="none",
    ).unique(
        subset="repository_id",
        keep="none",
    )

    # Create a table of document_authors
    # and their country institutional affiliations
    # then group by document to get to document_country_affiliation
    doc_contribs = _read_table("document_contributor")
    doc_contrib_institutions = _read_table("document_contributor_institution")
    institutions = _read_table("institution")

    doc_author_countries = (
        doc_contribs.select(
            pl.col("id").alias("document_contributor_id"),
            pl.col("researcher_id"),
            pl.col("document_id"),
        )
        .join(
            doc_contrib_institutions.select(
                "document_contributor_id",
                "institution_id",
            ),
            on="document_contributor_id",
            how="left",
        )
        .join(
            institutions.select(
                pl.col("id").alias("institution_id"),
                "country_code",
            ),
            on="institution_id",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("country_code").is_null())
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("country_code"))
        )
        .group_by("document_id")
        .agg(
            pl.len().alias("document_n_authors"),
            pl.when(pl.col("country_code").n_unique() == 1)
            .then(pl.col("country_code").get(0))
            .otherwise(pl.lit("Multiple")),
        )
        .with_columns(
            pl.when(pl.col("country_code").is_null())
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("country_code"))
            .alias("country_code")
        )
    )

    # Get repo_contribs count
    repo_contribs = _read_table("repository_contributor")
    repo_contribs = repo_contribs.group_by("repository_id").len("repository_n_contributors")

    # Join the tables to get the positive examples
    return (
        pairs.select(
            pl.col("id").alias("document_repository_link_id"),
            "document_id",
            "repository_id",
            "dataset_source_id",
            pl.col("predictive_model_confidence").alias("document_repository_link_confidence"),
        )
        .join(
            docs.select(
                pl.col("id").alias("document_id"),
                pl.col("doi").alias("document_doi"),
                pl.col("cited_by_count").alias("document_cited_by_count"),
                pl.col("fwci").alias("document_fwci"),
                pl.col("is_open_access").alias("document_is_open_access"),
                pl.col("title").alias("document_title"),
                pl.col("publication_date").alias("document_publication_date"),
                pl.col("document_type").alias("document_type"),
            ).with_columns(normalize_doi_col("document_doi").alias("document_doi")),
            on="document_id",
            how="left",
        )
        .join(
            repos.select(
                pl.col("id").alias("repository_id"),
                pl.col("owner").alias("repository_owner"),
                pl.col("name").alias("repository_name"),
                pl.col("stargazers_count").alias("repository_stargazers_count"),
                pl.col("commits_count").alias("repository_commits_count"),
                pl.col("primary_language").alias("repository_primary_language"),
                pl.col("creation_datetime").alias("repository_creation_datetime"),
                pl.col("last_pushed_datetime").alias("repository_last_pushed_datetime"),
            ),
            on="repository_id",
            how="left",
        )
        .join(
            dataset_sources.select(
                pl.col("id").alias("dataset_source_id"),
                pl.col("name").alias("dataset_source_name"),
            ),
            on="dataset_source_id",
            how="left",
        )
        .join(
            doc_topics.sort(
                "score",
                descending=True,
            )
            .unique(
                "document_id",
                maintain_order=True,
            )
            .select(
                pl.col("document_id").alias("document_id"),
                pl.col("topic_id").alias("topic_id"),
            ),
            on="document_id",
            how="left",
        )
        .join(
            topics.select(
                pl.col("id").alias("topic_id"),
                pl.col("domain_name").alias("document_domain_name"),
                pl.col("field_name").alias("document_field_name"),
            ),
            on="topic_id",
            how="left",
        )
        .join(
            doc_author_countries,
            on="document_id",
            how="left",
        )
        .join(
            repo_contribs,
            on="repository_id",
            how="left",
        )
    ).filter(
        (
            (pl.col("document_repository_link_confidence") >= 0.995)
            | (pl.col("document_repository_link_confidence").is_null())
        )
        & (pl.col("document_publication_date") < pl.date(2022, 1, 1))
    )


###############################################################################
# Data Structures
###############################################################################


@dataclass
class PairwiseSoftwareRecord:
    """
    A unified record for a piece of software in a pairwise comparison.

    For matched software, both source_a_name and source_b_name are populated.
    For source-A-only items, only source_a_name is populated.
    For source-B-only items, only source_b_name is populated.
    """

    source_a_name: str | None
    source_b_name: str | None
    normalized_name: str
    match_score: float | None
    status: str  # "matched", "source_a_only", "source_b_only"


###############################################################################
# Pairwise Analysis
###############################################################################


def _dedup_by_normalized(names: list[str], norms: list[str]) -> tuple[list[str], list[str]]:
    """Deduplicate (name, normalized) pairs by normalized value, keeping first."""
    seen: set[str] = set()
    out_names: list[str] = []
    out_norms: list[str] = []
    for n, nm in zip(names, norms, strict=False):
        if nm not in seen:
            seen.add(nm)
            out_names.append(n)
            out_norms.append(nm)
    return out_names, out_norms


def run_pairwise_analysis(
    source_a_names: list[str],
    source_a_normalized: list[str],
    source_b_names: list[str],
    source_b_normalized: list[str],
    source_a_label: str,
    source_b_label: str,
    score_cutoff: float = SCORE_CUTOFF,
    method: AlignmentMethod = "global_min_diff",
) -> tuple[list[PairwiseSoftwareRecord], dict[str, float]]:
    """
    Run pairwise alignment between two software sources.

    Uses pre-normalized names from the database.
    Since normalize_name is idempotent, passing already-normalized names is safe.

    Returns:
        records: List of PairwiseSoftwareRecord objects
        stats: Dict of summary statistics for this comparison
    """
    norm_a_to_orig = dict(zip(source_a_normalized, source_a_names, strict=False))
    norm_b_to_orig = dict(zip(source_b_normalized, source_b_names, strict=False))

    # align_software_names normalizes inputs internally (idempotent on pre-normalized names)
    matches = align_software_names(
        items_a=source_a_normalized,
        items_b=source_b_normalized,
        source_a=source_a_label,
        source_b=source_b_label,
        cutoff=score_cutoff,
        method=method,
    )

    # Uncomment to debug certain disambiguation cases
    # if source_a_label is "mention" or source_b_label is "mention":
    #     use_source = source_a_normalized if source_a_label is "mention" else source_b_normalized
    #     for variant in ["cv2", "opencv"]:
    #          if variant in use_source:
    #             print(source_a_label, source_a_normalized)
    #             print(source_b_label, source_b_normalized)
    #             print("matched pairs", matches)

    matched_a_norms: set[str] = set()
    matched_b_norms: set[str] = set()
    records: list[PairwiseSoftwareRecord] = []

    for match in matches:
        matched_a_norms.add(match.item_one)
        matched_b_norms.add(match.item_two)
        records.append(
            PairwiseSoftwareRecord(
                source_a_name=norm_a_to_orig.get(match.item_one, match.item_one),
                source_b_name=norm_b_to_orig.get(match.item_two, match.item_two),
                normalized_name=match.item_one,
                match_score=match.score,
                status="matched",
            )
        )

    for orig, norm in zip(source_a_names, source_a_normalized, strict=False):
        if norm not in matched_a_norms:
            records.append(
                PairwiseSoftwareRecord(
                    source_a_name=orig,
                    source_b_name=None,
                    normalized_name=norm,
                    match_score=None,
                    status="source_a_only",
                )
            )

    for orig, norm in zip(source_b_names, source_b_normalized, strict=False):
        if norm not in matched_b_norms:
            records.append(
                PairwiseSoftwareRecord(
                    source_a_name=None,
                    source_b_name=orig,
                    normalized_name=norm,
                    match_score=None,
                    status="source_b_only",
                )
            )

    n_source_a = len(source_a_normalized)
    n_source_b = len(source_b_normalized)
    n_matched = len(matches)
    n_source_a_only = n_source_a - len(matched_a_norms)
    n_source_b_only = n_source_b - len(matched_b_norms)

    denom = n_source_a + n_source_b - n_matched
    jaccard = n_matched / denom if denom > 0 else 0.0

    match_scores = [m.score for m in matches]
    avg_score = float(np.mean(match_scores)) if match_scores else 0.0
    median_score = float(np.median(match_scores)) if match_scores else 0.0

    return records, {
        "n_source_a": float(n_source_a),
        "n_source_b": float(n_source_b),
        "n_matched": float(n_matched),
        "n_source_a_only": float(n_source_a_only),
        "n_source_b_only": float(n_source_b_only),
        "jaccard": jaccard,
        "avg_match_score": avg_score,
        "median_match_score": median_score,
    }


###############################################################################
# Top-N Helpers
###############################################################################


def _get_top_items_by_status(
    all_records: list[dict],
    status_filter: str | list[str],
    name_column: str,
    top_n: int,
) -> list[str]:
    """Get top N software items by occurrence count, filtered by status."""
    if isinstance(status_filter, str):
        status_filter = [status_filter]

    filtered = [
        r for r in all_records if r["status"] in status_filter and r[name_column] is not None
    ]

    if not filtered:
        return []

    norm_counts: Counter[str] = Counter()
    norm_to_display: dict[str, str] = {}
    for r in filtered:
        norm = r["normalized_name"]
        norm_counts[norm] += 1
        norm_to_display[norm] = r[name_column]

    top = norm_counts.most_common(top_n)
    return [f"{prep_name_for_printing(norm_to_display[norm])} ({count})" for norm, count in top]


###############################################################################
# Statistics Helpers
###############################################################################


def _print_descriptive_stats(
    df: pl.DataFrame,
    col: str,
    label: str,
) -> str:
    """Return full descriptive statistics for a numeric column as a string."""
    vals = df[col].drop_nulls()
    if vals.len() == 0:
        return f"  {label}: no data"

    mean = vals.mean()
    std = vals.std()
    median = vals.median()
    p10 = vals.quantile(0.10)
    p25 = vals.quantile(0.25)
    p75 = vals.quantile(0.75)
    p90 = vals.quantile(0.90)
    min_val = vals.min()
    max_val = vals.max()

    return (
        f"  {label}: mean={mean:.3f}, std={std:.3f}, "
        f"min={min_val:.3f}, p10={p10:.3f}, p25={p25:.3f}, "
        f"median={median:.3f}, p75={p75:.3f}, p90={p90:.3f}, max={max_val:.3f}"
    )


_PAIR_STAT_COLS: list[tuple[str, str]] = [
    ("im_n_imports", "imports"),
    ("im_n_mentions", "mentions"),
    ("dm_n_deps", "deps"),
    ("im_n_matched", "matched_im"),
    ("dm_n_matched", "matched_dm"),
    ("id_n_matched", "matched_id"),
]

# For each comparison prefix, the two count columns that must both be > 0
# to qualify as a "non-zero subsample" pair.
_NZ_FILTER_COLS: dict[str, tuple[str, str]] = {
    "im": ("im_n_imports", "im_n_mentions"),
    "id": ("id_n_imports", "id_n_deps"),
    "dm": ("dm_n_deps", "dm_n_mentions"),
}

# Non-zero subsample: (tag, prefix, [(col, short_label), ...]) for reporting
_NZ_REPORT_COLS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "IM",
        "im",
        [
            ("im_n_imports", "imports"),
            ("im_n_mentions", "mentions"),
            ("im_n_matched", "matched_im"),
            ("im_jaccard", "jaccard_im"),
        ],
    ),
    (
        "ID",
        "id",
        [
            ("id_n_imports", "imports"),
            ("id_n_deps", "deps"),
            ("id_n_matched", "matched_id"),
            ("id_jaccard", "jaccard_id"),
        ],
    ),
    (
        "DM",
        "dm",
        [
            ("dm_n_deps", "deps"),
            ("dm_n_mentions", "mentions"),
            ("dm_n_matched", "matched_dm"),
            ("dm_jaccard", "jaccard_dm"),
        ],
    ),
]


def _pair_stats_grouped_lines(
    df: pl.DataFrame,
    group_col: str,
    top_n: int | None = None,
) -> list[str]:
    """Return mean/median/std lines for pair-level counts grouped by a column."""
    lines: list[str] = []
    grouped = (
        df.group_by(group_col).agg(pl.len().alias("n_pairs")).sort("n_pairs", descending=True)
    )
    if top_n is not None:
        grouped = grouped.head(top_n)

    for row in grouped.iter_rows(named=True):
        gv = row[group_col]
        gdf = (
            df.filter(pl.col(group_col).is_null())
            if gv is None
            else df.filter(pl.col(group_col) == gv)
        )
        display = "Unknown" if gv is None else str(gv)
        lines.append(f"\n  [{display}] (n={gdf.height})")
        for col, short in _PAIR_STAT_COLS:
            vals = gdf[col].drop_nulls()
            if vals.len() > 0:
                std = vals.std() if vals.len() > 1 else 0.0
                lines.append(
                    f"    {short}: mean={vals.mean():.2f}, "
                    f"median={vals.median():.2f}, "
                    f"std={std:.2f}"
                )
        # Non-zero subsample stats within this group
        for tag, prefix, cols in _NZ_REPORT_COLS:
            col_a, col_b = _NZ_FILTER_COLS[prefix]
            nz_gdf = gdf.filter((pl.col(col_a) > 0) & (pl.col(col_b) > 0))
            if nz_gdf.height >= 10:
                for col, short in cols:
                    vals = nz_gdf[col].drop_nulls()
                    if vals.len() > 0:
                        std = vals.std() if vals.len() > 1 else 0.0
                        lines.append(
                            f"    {short} (nz-{tag}, n={nz_gdf.height}): "
                            f"mean={vals.mean():.2f}, "
                            f"median={vals.median():.2f}, "
                            f"std={std:.2f}"
                        )
    return lines


def _pair_stats_correlations(df: pl.DataFrame) -> list[str]:
    """Compute Spearman correlations between pair-level count columns."""
    from scipy.stats import spearmanr

    lines: list[str] = []
    for col_a, col_b, label in [
        ("im_n_imports", "im_n_mentions", "imports vs mentions count"),
        ("dm_n_deps", "dm_n_mentions", "deps vs mentions count"),
        ("id_n_imports", "id_n_deps", "imports vs deps count"),
    ]:
        vals_a = df[col_a].to_numpy()
        vals_b = df[col_b].to_numpy()
        if len(vals_a) > 2:
            rho, p = spearmanr(vals_a, vals_b)
            p_str = f"{p:.4e}" if p >= 0.0001 else "<.0001"
            lines.append(f"  {label}: rho={rho:.4f}, p={p_str}")
        else:
            lines.append(f"  {label}: insufficient data")

    # Non-zero subsample correlations
    lines.append("  --- Non-zero subsample (both sides > 0) ---")
    for col_a, col_b, label in [
        ("im_n_imports", "im_n_mentions", "imports vs mentions count"),
        ("dm_n_deps", "dm_n_mentions", "deps vs mentions count"),
        ("id_n_imports", "id_n_deps", "imports vs deps count"),
    ]:
        nz = df.filter((pl.col(col_a) > 0) & (pl.col(col_b) > 0))
        vals_a = nz[col_a].to_numpy()
        vals_b = nz[col_b].to_numpy()
        if len(vals_a) > 2:
            rho, p = spearmanr(vals_a, vals_b)
            p_str = f"{p:.4e}" if p >= 0.0001 else "<.0001"
            lines.append(f"  {label} (nz, n={nz.height}): rho={rho:.4f}, p={p_str}")
        else:
            lines.append(f"  {label} (nz): insufficient data")

    return lines


def _write_pair_level_stats(
    results_df: pl.DataFrame,
    filter_mode: str,
    mode_path: Path,
    top_n_fields: int = 15,
) -> None:
    """Write per-pair descriptive statistics to a text file."""
    if filter_mode == "complete-cases":
        df = results_df.filter(
            ((pl.col("im_n_imports") > 0) & (pl.col("im_n_mentions") > 0))
            | ((pl.col("id_n_imports") > 0) & (pl.col("id_n_deps") > 0))
            | ((pl.col("dm_n_deps") > 0) & (pl.col("dm_n_mentions") > 0))
        )
    else:
        df = results_df.filter(
            (pl.col("im_n_imports") > 0)
            | (pl.col("im_n_mentions") > 0)
            | (pl.col("dm_n_deps") > 0)
        )

    lines: list[str] = [
        "=" * 70,
        f"  Per-Pair Descriptive Statistics  ({filter_mode}, N = {df.height})",
        "=" * 70,
        "",
        "─" * 70,
        "  Overall Per-Pair Counts",
        "─" * 70,
    ]

    for col, label in [
        ("im_n_imports", "Imports per pair"),
        ("im_n_mentions", "Mentions per pair"),
        ("dm_n_deps", "Dependencies per pair"),
        ("im_n_matched", "Matched imports-mentions per pair"),
        ("dm_n_matched", "Matched deps-mentions per pair"),
        ("id_n_matched", "Matched imports-deps per pair"),
    ]:
        lines.append(_print_descriptive_stats(df, col, label))

    # Non-zero subsample descriptive stats
    lines.extend(
        [
            "",
            "─" * 70,
            "  Non-Zero Subsample Stats (both sides > 0 for each comparison)",
            "─" * 70,
        ]
    )
    for tag, prefix, cols in _NZ_REPORT_COLS:
        col_a, col_b = _NZ_FILTER_COLS[prefix]
        nz_df = df.filter((pl.col(col_a) > 0) & (pl.col(col_b) > 0))
        lines.append(f"  [{tag}] non-zero subsample: n={nz_df.height}")
        for col, short in cols:
            lines.append(_print_descriptive_stats(nz_df, col, f"  {short}"))

    # Fraction with zero
    n = df.height
    lines.extend(["", "─" * 70, "  Zero-Count Fractions", "─" * 70])
    for col, label in [
        ("im_n_imports", "imports"),
        ("im_n_mentions", "mentions"),
        ("dm_n_deps", "dependencies"),
    ]:
        n_zero = df.filter(pl.col(col) == 0).height
        pct = n_zero / n * 100 if n > 0 else 0
        lines.append(f"  Pairs with zero {label}: {n_zero:,} / {n:,} ({pct:.1f}%)")

    # Correlations
    lines.extend(["", "─" * 70, "  Pairwise Correlations (Spearman rank)", "─" * 70])
    lines.extend(_pair_stats_correlations(df))

    # Per-field breakdown
    lines.extend(
        ["", "─" * 70, f"  Per-Field Breakdown (top {top_n_fields} by pair count)", "─" * 70]
    )
    lines.extend(_pair_stats_grouped_lines(df, "document_field_name", top_n=top_n_fields))

    # Over time breakdown
    lines.extend(["", "─" * 70, "  Over Time Breakdown (by publication year)", "─" * 70])
    year_df = df.with_columns(pl.col("document_publication_date").dt.year().alias("pub_year"))
    lines.extend(_pair_stats_grouped_lines(year_df, "pub_year"))

    text = "\n".join(lines)
    print(text)
    with open(mode_path / "rq3-pair-level-stats.txt", "w") as f:
        f.write(text)
    log.info(f"[{filter_mode}] Saved rq3-pair-level-stats.txt")


def _build_doc_id_lookup(all_records: list[dict]) -> dict[int, list[dict]]:
    """Build a document_id → records lookup for efficient per-group filtering."""
    lookup: dict[int, list[dict]] = {}
    for rec in all_records:
        did = rec["document_id"]
        if did not in lookup:
            lookup[did] = []
        lookup[did].append(rec)
    return lookup


def _print_group_row_top_n(
    row: dict,
    group_col: str,
    results_df: pl.DataFrame,
    doc_id_to_records: dict[int, list[dict]],
    a_label: str,
    b_label: str,
    top_n: int,
) -> str:
    """Return top-N software items for a single group row as a string."""
    gv = row[group_col]
    if gv is None:
        group_df = results_df.filter(pl.col(group_col).is_null())
    else:
        group_df = results_df.filter(pl.col(group_col) == gv)
    group_recs: list[dict] = []
    for did in set(group_df["document_id"].to_list()):
        group_recs.extend(doc_id_to_records.get(did, []))

    lines: list[str] = []
    top_matched = _get_top_items_by_status(group_recs, "matched", "source_a_name", top_n)
    if top_matched:
        lines.append(f"      Top {top_n} matched: {', '.join(top_matched)}")

    top_a = _get_top_items_by_status(
        group_recs, ["matched", "source_a_only"], "source_a_name", top_n
    )
    if top_a:
        lines.append(f"      Top {top_n} {a_label}: {', '.join(top_a)}")

    top_b = _get_top_items_by_status(
        group_recs, ["matched", "source_b_only"], "source_b_name", top_n
    )
    if top_b:
        lines.append(f"      Top {top_n} {b_label}: {', '.join(top_b)}")

    top_a_unmatched = _get_top_items_by_status(
        group_recs, "source_a_only", "source_a_name", top_n
    )
    if top_a_unmatched:
        lines.append(f"      Top {top_n} unmatched-{a_label}: {', '.join(top_a_unmatched)}")

    top_b_unmatched = _get_top_items_by_status(
        group_recs, "source_b_only", "source_b_name", top_n
    )
    if top_b_unmatched:
        lines.append(f"      Top {top_n} unmatched-{b_label}: {', '.join(top_b_unmatched)}")

    return "\n".join(lines)


def _print_group_breakdown(
    results_df: pl.DataFrame,
    group_col: str,
    group_label: str,
    jaccard_col: str,
    n_matched_col: str,
    top_n_filter: int | None = None,
    all_records: list[dict] | None = None,
    a_label: str | None = None,
    b_label: str | None = None,
    top_n: int = 10,
) -> str:
    """Return per-group descriptive stats for jaccard and n_matched as a string."""
    lines: list[str] = [""]
    if top_n_filter is not None:
        lines.append(f"  Breakdown by {group_label} (top {top_n_filter} most populous):")
    else:
        lines.append(f"  Breakdown by {group_label}:")
    lines.append(f"  {'-' * 60}")

    grouped = (
        results_df.group_by(group_col)
        .agg(
            pl.len().alias("n_pairs"),
            pl.col(jaccard_col).mean().alias("mean_jaccard"),
            pl.col(jaccard_col).std().alias("std_jaccard"),
            pl.col(jaccard_col).median().alias("median_jaccard"),
            pl.col(jaccard_col).quantile(0.25).alias("p25_jaccard"),
            pl.col(jaccard_col).quantile(0.75).alias("p75_jaccard"),
            pl.col(n_matched_col).mean().alias("mean_n_matched"),
            pl.col(n_matched_col).median().alias("median_n_matched"),
        )
        .sort("n_pairs", descending=True)
    )

    if top_n_filter is not None:
        grouped = grouped.head(top_n_filter)

    show_top_n = all_records is not None and a_label is not None and b_label is not None
    assert all_records is not None or not show_top_n
    doc_id_to_records = (
        _build_doc_id_lookup(all_records) if show_top_n and all_records is not None else {}
    )

    for row in grouped.iter_rows(named=True):
        group_val = row[group_col] if row[group_col] is not None else "Unknown"
        lines.append(
            f"    [{group_val}]: n={row['n_pairs']}, "
            f"jaccard mean={row['mean_jaccard'] or 0:.3f}, "
            f"std={row['std_jaccard'] or 0:.3f}, "
            f"median={row['median_jaccard'] or 0:.3f}, "
            f"p25={row['p25_jaccard'] or 0:.3f}, "
            f"p75={row['p75_jaccard'] or 0:.3f}, "
            f"mean_n_matched={row['mean_n_matched'] or 0:.1f}"
        )

        if show_top_n:
            top_n_text = _print_group_row_top_n(
                row,
                group_col,
                results_df,
                doc_id_to_records,
                a_label,  # type: ignore[arg-type]
                b_label,  # type: ignore[arg-type]
                top_n,
            )
            if top_n_text:
                lines.append(top_n_text)

    return "\n".join(lines)


def _compute_gini(counts: list[int]) -> float:
    """Compute the Gini coefficient of a frequency distribution."""
    if not counts or sum(counts) == 0:
        return 0.0
    arr = np.sort(np.array(counts, dtype=float))
    n = len(arr)
    idx = np.arange(1, n + 1)
    total = float(arr.sum())
    return float((2.0 * float(np.sum(idx * arr)) - (n + 1) * total) / (n * total))


###############################################################################
# Coverage Summary
###############################################################################


def _print_coverage_summary(
    pairs: pl.DataFrame,
    imports_by_repo: dict[int, tuple[list[str], list[str]]],
    deps_by_repo: dict[int, tuple[list[str], list[str]]],
    mentions_by_doc: dict[int, tuple[list[str], list[str]]],
) -> str:
    """Return a diagnostic table showing how much of the dataset has each source populated."""
    n_total = pairs.height
    import_keys = set(imports_by_repo.keys())
    dep_keys = set(deps_by_repo.keys())
    mention_keys = set(mentions_by_doc.keys())

    # Per-pair coverage flags (vectorized via polars)
    repo_ids = pairs["repository_id"].to_list()
    doc_ids = pairs["document_id"].to_list()

    has_i = [rid in import_keys for rid in repo_ids]
    has_d = [rid in dep_keys for rid in repo_ids]
    has_m = [did in mention_keys for did in doc_ids]

    n_has_i = sum(has_i)
    n_has_d = sum(has_d)
    n_has_m = sum(has_m)

    n_im_and = sum(i and m for i, m in zip(has_i, has_m, strict=False))
    n_id_and = sum(i and d for i, d in zip(has_i, has_d, strict=False))
    n_dm_and = sum(d and m for d, m in zip(has_d, has_m, strict=False))

    n_im_or = sum(i or m for i, m in zip(has_i, has_m, strict=False))
    n_id_or = sum(i or d for i, d in zip(has_i, has_d, strict=False))
    n_dm_or = sum(d or m for d, m in zip(has_d, has_m, strict=False))

    n_all_three = sum(i and d and m for i, d, m in zip(has_i, has_d, has_m, strict=False))
    n_none = sum(
        not i and not d and not m for i, d, m in zip(has_i, has_d, has_m, strict=False)
    )

    def pct(n: int) -> str:
        return f"{n / n_total * 100:.1f}%" if n_total > 0 else "N/A"

    sep = "━" * 52
    lines = [
        "",
        sep,
        f"  Coverage Summary  (total pairs loaded: {n_total:,})",
        sep,
        "  Source coverage (pairs with any data for source):",
        f"    Imports:      {n_has_i:>7,}  ({pct(n_has_i)})",
        f"    Dependencies: {n_has_d:>7,}  ({pct(n_has_d)})",
        f"    Mentions:     {n_has_m:>7,}  ({pct(n_has_m)})",
        "",
        "  Pairwise AND (both sources non-empty → complete-cases filter):",
        f"    Imports ∧ Mentions:     {n_im_and:>7,}  ({pct(n_im_and)})",
        f"    Imports ∧ Dependencies: {n_id_and:>7,}  ({pct(n_id_and)})",
        f"    Deps ∧ Mentions:        {n_dm_and:>7,}  ({pct(n_dm_and)})",
        "",
        "  Pairwise OR (at least one source non-empty → full-population filter):",
        f"    Imports v Mentions:     {n_im_or:>7,}  ({pct(n_im_or)})",
        f"    Imports v Dependencies: {n_id_or:>7,}  ({pct(n_id_or)})",
        f"    Deps v Mentions:        {n_dm_or:>7,}  ({pct(n_dm_or)})",
        "",
        f"  All three sources non-empty: {n_all_three:>7,}  ({pct(n_all_three)})",
        f"  No sources with data:        {n_none:>7,}  ({pct(n_none)})",
        sep,
    ]
    return "\n".join(lines)


###############################################################################
# Hidden Infrastructure
###############################################################################


def _print_hidden_infrastructure(
    all_records: list[dict],
    valid_doc_ids: set[int],
    source_a_label: str,
    source_b_label: str,
    top_n: int = 50,
) -> str:
    """Ranked table using matched pair records from pairwise alignment.

    Counts source_a occurrences (matched + unmatched) and compares against
    matched source_b occurrences.  Avoids cross-namespace exact-name joins
    by relying on the fuzzy matching already performed per doc-repo pair.
    """
    filtered = [r for r in all_records if r["document_id"] in valid_doc_ids]

    # Count source_a appearances (matched + source_a_only)
    a_counts: Counter[str] = Counter()
    # Count source_b appearances via matched records only
    b_counts: Counter[str] = Counter()

    for r in filtered:
        norm = r["normalized_name"]
        if r["status"] == "matched":
            a_counts[norm] += 1
            b_counts[norm] += 1
        elif r["status"] == "source_a_only":
            a_counts[norm] += 1

    # Build table sorted by source_a count
    top_names = [name for name, _ in a_counts.most_common(top_n)]

    n_pairs = len(valid_doc_ids)
    a_short = source_a_label[:3]
    b_short = source_b_label[:3]
    ratio_label = f"{a_short[0]}:{b_short[0]} Ratio"

    col_w = 32
    header = (
        f"  {'Library':<{col_w}} {source_a_label:>10} {a_short + '%':>6} "
        f"{source_b_label:>10} {b_short + '%':>6} {ratio_label:>10}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [
        "",
        "=" * 70,
        f"  Hidden Infrastructure ({source_a_label} vs {source_b_label}): "
        f"top {top_n} most-used {source_a_label.lower()}",
        f"  {ratio_label} = {source_a_label.lower()}_count / "
        f"({source_b_label.lower()}_count + 1)  — high = rarely in {source_b_label.lower()}",
        "=" * 70,
        header,
        sep,
    ]

    for name in top_names:
        a_count = a_counts[name]
        b_count = b_counts.get(name, 0)
        a_pct = a_count / n_pairs * 100 if n_pairs > 0 else 0.0
        b_pct = b_count / n_pairs * 100 if n_pairs > 0 else 0.0
        ratio = a_count / (b_count + 1)
        display = prep_name_for_printing(name)[:col_w]
        lines.append(
            f"  {display:<{col_w}} {a_count:>10,} {a_pct:>5.1f}% "
            f"{b_count:>10,} {b_pct:>5.1f}% "
            f"{ratio:>10.1f}"
        )

    return "\n".join(lines)


def _write_hidden_infrastructure_tables(
    results_df: pl.DataFrame,
    all_im_records: list[dict],
    all_dm_records: list[dict],
    filter_mode: str,
    mode_path: Path,
    top_n: int = 50,
) -> None:
    """Compute and write hidden infrastructure tables for a given filter mode."""
    if filter_mode == "complete-cases":
        im_filter = (pl.col("im_n_imports") > 0) & (pl.col("im_n_mentions") > 0)
        dm_filter = (pl.col("dm_n_deps") > 0) & (pl.col("dm_n_mentions") > 0)
    else:
        im_filter = (pl.col("im_n_imports") > 0) | (pl.col("im_n_mentions") > 0)
        dm_filter = (pl.col("dm_n_deps") > 0) | (pl.col("dm_n_mentions") > 0)

    im_doc_ids = set(results_df.filter(im_filter)["document_id"].to_list())
    dm_doc_ids = set(results_df.filter(dm_filter)["document_id"].to_list())

    log.info(f"[{filter_mode}] Computing hidden infrastructure tables...")
    for records, doc_ids, a_label, b_label, fname in [
        (
            all_im_records,
            im_doc_ids,
            "Imports",
            "Mentions",
            "rq3-hidden-infrastructure-imports.txt",
        ),
        (
            all_dm_records,
            dm_doc_ids,
            "Dependencies",
            "Mentions",
            "rq3-hidden-infrastructure-deps.txt",
        ),
    ]:
        infra_text = _print_hidden_infrastructure(
            records, doc_ids, a_label, b_label, top_n=top_n
        )
        print(infra_text)
        with open(mode_path / fname, "w") as f:
            f.write(infra_text)
        log.info(f"[{filter_mode}] Saved {fname}")


###############################################################################
# Summary Statistics
###############################################################################


def _print_summary_stats(
    results_df: pl.DataFrame,
    all_im_records: list[dict],
    all_id_records: list[dict],
    all_dm_records: list[dict],
    top_n: int,
    filter_mode: str,
) -> dict[str, str]:
    # top_n_filter=None means show all groups; integer means show only top-N most populous
    grouping_cols: list[tuple[str, str, int | None]] = [
        ("document_domain_name", "Domain", None),
        ("document_field_name", "Field", top_n),
        ("repository_primary_language", "Language", top_n),
        ("dataset_source_name", "Dataset Source", None),
        ("document_publication_year", "Publication Year", None),
        ("country_code", "Country", top_n),
        ("document_is_open_access", "Open Access", None),
    ]

    comparisons: list[tuple[str, str, list[dict], str, str]] = [
        ("Imports vs Mentions", "im", all_im_records, "Imports", "Mentions"),
        ("Imports vs Dependencies", "id", all_id_records, "Imports", "Dependencies"),
        ("Dependencies vs Mentions", "dm", all_dm_records, "Dependencies", "Mentions"),
    ]

    if filter_mode == "complete-cases":
        filter_exprs = {
            "im": (pl.col("im_n_imports") > 0) & (pl.col("im_n_mentions") > 0),
            "id": (pl.col("id_n_imports") > 0) & (pl.col("id_n_deps") > 0),
            "dm": (pl.col("dm_n_deps") > 0) & (pl.col("dm_n_mentions") > 0),
        }
    else:  # full-population
        filter_exprs = {
            "im": (pl.col("im_n_imports") > 0) | (pl.col("im_n_mentions") > 0),
            "id": (pl.col("id_n_imports") > 0) | (pl.col("id_n_deps") > 0),
            "dm": (pl.col("dm_n_deps") > 0) | (pl.col("dm_n_mentions") > 0),
        }

    results: dict[str, str] = {}
    for label, prefix, all_records, a_label, b_label in comparisons:
        comparison_df = results_df.filter(filter_exprs[prefix])
        mode_note = (
            "NOTE: full-population results include structural zeros where one source has "
            "no data. These are only meaningful once dataset processing is complete."
            if filter_mode == "full-population"
            else ""
        )
        lines: list[str] = [
            "",
            "=" * 70,
            f"  Filter mode: {filter_mode}  (N = {comparison_df.height})",
            f"  {label}",
            "=" * 70,
        ]
        if mode_note:
            lines.append(f"  *** {mode_note} ***")

        lines.append(_print_descriptive_stats(comparison_df, f"{prefix}_jaccard", "Jaccard"))
        lines.append(
            _print_descriptive_stats(comparison_df, f"{prefix}_n_matched", "N Matched")
        )
        non_zero_score = comparison_df.filter(pl.col(f"{prefix}_avg_score") > 0)
        lines.append(
            _print_descriptive_stats(non_zero_score, f"{prefix}_avg_score", "Avg Match Score")
        )

        # Non-zero subsample (both sides > 0)
        col_a, col_b = _NZ_FILTER_COLS[prefix]
        nz_df = comparison_df.filter((pl.col(col_a) > 0) & (pl.col(col_b) > 0))
        lines.append(f"  --- Non-zero subsample (n={nz_df.height}) ---")
        lines.append(_print_descriptive_stats(nz_df, f"{prefix}_jaccard", "Jaccard (nz)"))
        lines.append(_print_descriptive_stats(nz_df, f"{prefix}_n_matched", "N Matched (nz)"))

        top_matched = _get_top_items_by_status(all_records, "matched", "source_a_name", top_n)
        if top_matched:
            lines.append(f"  Top {top_n} matched: {', '.join(top_matched)}")

        top_a = _get_top_items_by_status(
            all_records, ["matched", "source_a_only"], "source_a_name", top_n
        )
        if top_a:
            lines.append(f"  Top {top_n} {a_label}: {', '.join(top_a)}")

        top_b = _get_top_items_by_status(
            all_records, ["matched", "source_b_only"], "source_b_name", top_n
        )
        if top_b:
            lines.append(f"  Top {top_n} {b_label}: {', '.join(top_b)}")

        top_a_unmatched = _get_top_items_by_status(
            all_records, "source_a_only", "source_a_name", top_n
        )
        if top_a_unmatched:
            lines.append(f"  Top {top_n} unmatched-{a_label}: {', '.join(top_a_unmatched)}")

        top_b_unmatched = _get_top_items_by_status(
            all_records, "source_b_only", "source_b_name", top_n
        )
        if top_b_unmatched:
            lines.append(f"  Top {top_n} unmatched-{b_label}: {', '.join(top_b_unmatched)}")

        for col, col_label, tnf in grouping_cols:
            lines.append(
                _print_group_breakdown(
                    comparison_df,
                    col,
                    col_label,
                    f"{prefix}_jaccard",
                    f"{prefix}_n_matched",
                    top_n_filter=tnf,
                    all_records=all_records,
                    a_label=a_label,
                    b_label=b_label,
                    top_n=top_n,
                )
            )

        results[label] = "\n".join(lines)

    return results


###############################################################################
# Visualizations
###############################################################################


def _filter_for_mode(
    results_df: pl.DataFrame,
    prefix: str,
    filter_mode: str,
) -> pl.DataFrame:
    """Filter results_df to pairs appropriate for the given comparison and filter mode."""
    col_pairs = {
        "im": ("im_n_imports", "im_n_mentions"),
        "id": ("id_n_imports", "id_n_deps"),
        "dm": ("dm_n_deps", "dm_n_mentions"),
    }
    a_col, b_col = col_pairs[prefix]
    if filter_mode == "complete-cases":
        return results_df.filter((pl.col(a_col) > 0) & (pl.col(b_col) > 0))
    else:
        return results_df.filter((pl.col(a_col) > 0) | (pl.col(b_col) > 0))


def _plot_jaccard_boxplot(
    results_df: pl.DataFrame,
    output_dir: Path,
    filter_mode: str,
) -> None:
    """Box plots of per-pair Jaccard scores for all three comparisons."""
    comparison_labels = {
        "im_jaccard": ("Imports vs Mentions", "im"),
        "id_jaccard": ("Imports vs Dependencies", "id"),
        "dm_jaccard": ("Deps vs Mentions", "dm"),
    }
    data: list[dict] = []
    for col, (label, prefix) in comparison_labels.items():
        filtered = _filter_for_mode(results_df, prefix, filter_mode)
        for v in filtered[col].drop_nulls().to_list():
            data.append({"Comparison": label, "Jaccard Similarity": float(v)})

    if not data:
        log.warning("No data for Jaccard boxplot, skipping.")
        return

    plot_df = pl.DataFrame(data).to_pandas()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot_df, x="Comparison", y="Jaccard Similarity", ax=ax, showfliers=False)
    ax.set_title("Distribution of Pairwise Jaccard Similarity Scores")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(output_dir / "jaccard_boxplot.png", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved jaccard_boxplot.png")


def _plot_jaccard_by_group(
    results_df: pl.DataFrame,
    group_col: str,
    group_label: str,
    output_dir: Path,
    filter_mode: str,
    top_n_groups: int = 10,
) -> None:
    top_groups = (
        results_df.group_by(group_col)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_n_groups)[group_col]
        .to_list()
    )

    comparison_cols = {
        "im_jaccard": ("Imports vs Mentions", "im"),
        "id_jaccard": ("Imports vs Dependencies", "id"),
        "dm_jaccard": ("Deps vs Mentions", "dm"),
    }

    data: list[dict] = []
    for col, (label, prefix) in comparison_cols.items():
        mode_filtered = _filter_for_mode(results_df, prefix, filter_mode)
        filtered = mode_filtered.filter(pl.col(group_col).is_in(top_groups))
        for row in (
            filtered.group_by(group_col)
            .agg(pl.col(col).mean().alias("mean_jaccard"))
            .sort(group_col)
            .iter_rows(named=True)
        ):
            data.append(
                {
                    group_label: str(row[group_col]),
                    "Mean Jaccard": row["mean_jaccard"],
                    "Comparison": label,
                }
            )

    if not data:
        return

    plot_df = pl.DataFrame(data).to_pandas()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x=group_label,
        y="Mean Jaccard",
        hue="Comparison",
        ax=ax,
    )
    ax.set_title(f"Mean Jaccard Similarity by {group_label}")
    ax.set_xlabel(group_label)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    safe_name = group_col.replace("/", "_").replace(" ", "_")
    fig.savefig(output_dir / f"jaccard_by_{safe_name}.png", bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved jaccard_by_{safe_name}.png")


def _plot_score_histograms(
    results_df: pl.DataFrame,
    output_dir: Path,
    filter_mode: str,
) -> None:
    """Histograms of average pairwise match scores for each comparison."""
    score_cols = {
        "im_avg_score": ("Imports vs Mentions", "im"),
        "id_avg_score": ("Imports vs Dependencies", "id"),
        "dm_avg_score": ("Deps vs Mentions", "dm"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, (label, prefix)) in zip(axes, score_cols.items(), strict=False):
        filtered = _filter_for_mode(results_df, prefix, filter_mode)
        vals = filtered.filter(pl.col(col) > 0)[col].drop_nulls().to_list()
        if vals:
            ax.hist(vals, bins=30, edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel("Average Match Score")
        ax.set_ylabel("Count")
    plt.suptitle("Distribution of Pairwise Match Scores")
    plt.tight_layout()
    fig.savefig(output_dir / "match_score_histograms.png", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved match_score_histograms.png")


def _plot_jaccard_trend_by_year(
    results_df: pl.DataFrame,
    output_dir: Path,
    filter_mode: str,
    min_pairs_per_year: int = 5,
) -> None:
    """Line plots of mean Jaccard ± 95% CI by publication year, one per comparison."""
    comparisons = [
        ("im_jaccard", "im", "Imports vs Mentions", "im"),
        ("id_jaccard", "id", "Imports vs Dependencies", "id"),
        ("dm_jaccard", "dm", "Dependencies vs Mentions", "dm"),
    ]

    for jac_col, prefix, label, tag in comparisons:
        df = _filter_for_mode(results_df, prefix, filter_mode)
        year_stats = (
            df.group_by("document_publication_year")
            .agg(
                pl.col(jac_col).mean().alias("mean_jac"),
                pl.col(jac_col).std().alias("std_jac"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= min_pairs_per_year)
            .sort("document_publication_year")
        )

        if year_stats.height < 2:
            log.warning(f"Insufficient data for temporal trend plot: {label}, skipping.")
            continue

        years = year_stats["document_publication_year"].to_list()
        means = year_stats["mean_jac"].to_list()
        stds = [s or 0.0 for s in year_stats["std_jac"].to_list()]
        ns = year_stats["n"].to_list()
        ci = [1.96 * s / (n**0.5) for s, n in zip(stds, ns, strict=False)]
        lower = [max(0.0, m - c) for m, c in zip(means, ci, strict=False)]
        upper = [m + c for m, c in zip(means, ci, strict=False)]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(years, means, marker="o", linewidth=2)
        ax.fill_between(years, lower, upper, alpha=0.2)
        ax.set_xlabel("Publication Year")
        ax.set_ylabel("Mean Jaccard Similarity")
        ax.set_title(f"{label} Over Time")
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        out_name = f"rq3-jaccard-trend-{tag}.png"
        fig.savefig(output_dir / out_name, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved {out_name}")


def _plot_software_frequency_distributions(
    imports_df: pl.DataFrame,
    deps_df: pl.DataFrame,
    mentions_df: pl.DataFrame,
    pair_repo_ids: set[int],
    pair_doc_ids: set[int],
    output_dir: Path,
) -> dict[str, float]:
    """
    Plot frequency rank vs count (log-log) for each source.

    Returns Gini coefficients keyed by source name.
    """
    import_freq = (
        imports_df.filter(pl.col("repository_id").is_in(list(pair_repo_ids)))
        .group_by("software_name_normalized")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    dep_freq = (
        deps_df.filter(pl.col("repository_id").is_in(list(pair_repo_ids)))
        .group_by("software_name_normalized")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    mention_freq = (
        mentions_df.filter(pl.col("document_id").is_in(list(pair_doc_ids)))
        .group_by("software_name_normalized")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    sources = {
        "Imports": import_freq,
        "Dependencies": dep_freq,
        "Mentions": mention_freq,
    }

    gini_values: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(8, 5))

    for source_label, freq_df in sources.items():
        counts = freq_df["count"].to_list()
        gini = _compute_gini(counts)
        gini_values[source_label] = gini
        ranks = list(range(1, len(counts) + 1))
        ax.plot(ranks, counts, label=f"{source_label} (Gini={gini:.3f})", alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency (number of pairs)")
    ax.set_title("Software Name Frequency Distributions by Source")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "software_frequency_distributions.png", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved software_frequency_distributions.png")

    return gini_values


###############################################################################
# Logistic Regression
###############################################################################


def _bin_top_n(series: pd.Series, top_n: int) -> pd.Series:
    top = series.value_counts().head(top_n).index
    return pd.Series(series.where(series.isin(top), other="Other"))


def _format_logit_result(result: object, model_name: str) -> str:
    """Format a statsmodels Logit result as a readable text block."""
    import pandas as pd

    lines = [
        f"\n{'─' * 70}",
        f"  {model_name}",
        f"{'─' * 70}",
        f"  N = {int(result.nobs):,}",  # type: ignore[attr-defined]
        f"  Pseudo-R² (McFadden) = {result.prsquared:.4f}",  # type: ignore[attr-defined]
        f"  AIC = {result.aic:.2f}  |  BIC = {result.bic:.2f}",  # type: ignore[attr-defined]
        "",
        f"  {'Variable':<44} {'Coef':>8} {'SE':>8} {'z':>7} {'p':>8} "
        f"{'[0.025':>8} {'0.975]':>8} {'OR':>8}",
        f"  {'─' * 103}",
    ]

    params = result.params  # type: ignore[attr-defined]
    bse = result.bse  # type: ignore[attr-defined]
    tvalues = result.tvalues  # type: ignore[attr-defined]
    pvalues = result.pvalues  # type: ignore[attr-defined]
    conf_int: pd.DataFrame = result.conf_int()  # type: ignore[attr-defined]

    has_separation = any(
        abs(z) > 50 or np.exp(c) > 1e10
        for c, z in zip(params.values, tvalues.values, strict=False)
    )
    if has_separation:
        lines.insert(
            0,
            "  ⚠ SEPARATION WARNING: one or more predictors nearly perfectly predict\n"
            "    the outcome. MLE coefficients are unreliable; interpret direction only.\n",
        )

    for name in params.index:
        coef = params[name]
        se = bse[name]
        z = tvalues[name]
        p = pvalues[name]
        lo = conf_int.loc[name, 0]
        hi = conf_int.loc[name, 1]
        or_ = np.exp(coef)
        p_str = f"{p:.4f}" if p >= 0.0001 else "<.0001"
        # Truncate long variable names (one-hot labels can be verbose)
        short_name = name[:44]
        lines.append(
            f"  {short_name:<44} {coef:>8.4f} {se:>8.4f} {z:>7.3f} {p_str:>8} "
            f"{lo:>8.4f} {hi:>8.4f} {or_:>8.4f}"
        )
    return "\n".join(lines)


def _collinearity_diagnostics(
    df: "pd.DataFrame",
    cols: list[str],
) -> str:
    """Compute collinearity diagnostics for prevalence z-score columns.

    Returns a formatted text block with Pearson r, Spearman rho, and VIF values.
    """
    from scipy.stats import pearsonr, spearmanr
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    lines = [
        f"\n{'=' * 70}",
        "  Collinearity Diagnostics",
        "=" * 70,
    ]

    # Pairwise correlations
    if len(cols) == 2:
        a, b = cols
        pr, pp = pearsonr(df[a], df[b])
        sr, sp = spearmanr(df[a], df[b])
        lines.append(f"\n  Pearson  r({a}, {b}) = {pr:.4f}  (p = {pp:.4e})")
        lines.append(f"  Spearman rho({a}, {b}) = {sr:.4f}  (p = {sp:.4e})")
    else:
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                pr, pp = pearsonr(df[a], df[b])
                sr, sp = spearmanr(df[a], df[b])
                lines.append(f"\n  Pearson  r({a}, {b}) = {pr:.4f}  (p = {pp:.4e})")
                lines.append(f"  Spearman rho({a}, {b}) = {sr:.4f}  (p = {sp:.4e})")

    # VIF

    design_mat = df[cols].copy()
    design_mat.insert(0, "const", 1.0)
    lines.append("")
    for i, col in enumerate(cols):
        vif = variance_inflation_factor(design_mat.values, i + 1)  # +1 to skip const
        flag = "  ** HIGH" if vif > 5 else ""
        lines.append(f"  VIF({col}) = {vif:.2f}{flag}")

    if any(variance_inflation_factor(design_mat.values, i + 1) > 5 for i in range(len(cols))):
        lines.append(
            "\n  ** NOTE: VIF > 5 indicates substantial collinearity."
            "\n    Coefficients may be unstable;"
            " interpret relative magnitudes with caution."
        )

    lines.append("")
    return "\n".join(lines)


def _build_pair_metadata(results_df: pl.DataFrame) -> dict[int, dict]:
    """Build a document_id → metadata lookup from the results DataFrame."""
    pair_meta: dict[int, dict] = {}
    for row in results_df.iter_rows(named=True):
        pair_meta[row["document_id"]] = {
            "field": row["document_field_name"] or "Unknown",
            "year": row["document_publication_year"],
            "language": row["repository_primary_language"] or "Unknown",
            "repository_id": row["repository_id"],
        }
    return pair_meta


def _compute_global_prevalence(
    df: pl.DataFrame,
    n_total_repos: int,
) -> dict[str, float]:
    """Compute global prevalence: fraction of repos using each library."""
    agg = (
        df.group_by("software_name_normalized")
        .agg(pl.n_unique("repository_id").alias("c"))
        .with_columns((pl.col("c") / n_total_repos).alias("p"))
    )
    return dict(
        zip(
            agg["software_name_normalized"].to_list(),
            agg["p"].to_list(),
            strict=False,
        )
    )


def _compute_field_prevalence(
    df: pl.DataFrame,
    repo_to_field: dict,
    field_n_dict: dict[str, int],
) -> dict[tuple[str, str], float]:
    """Compute field-level prevalence: fraction of field repos using each library."""
    with_field = df.with_columns(
        pl.col("repository_id").replace(repo_to_field, default=None).alias("field")
    )
    field_raw = (
        with_field.filter(pl.col("field").is_not_null())
        .group_by(["software_name_normalized", "field"])
        .agg(pl.n_unique("repository_id").alias("c"))
    )
    return {
        (row["software_name_normalized"], row["field"]): row["c"]
        / field_n_dict.get(row["field"], 1)
        for row in field_raw.iter_rows(named=True)
    }


def _compute_prevalence_features(
    imports_df: pl.DataFrame,
    deps_df: pl.DataFrame,
    pairs: pl.DataFrame,
) -> tuple[
    dict[str, float],
    dict[tuple[str, str], float],
    dict[str, float],
    dict[tuple[str, str], float],
]:
    """Compute global and field-level prevalence for imports and deps."""
    n_total_repos = pairs["repository_id"].n_unique()

    repo_to_field = dict(
        zip(
            pairs["repository_id"].to_list(),
            pairs["document_field_name"].to_list(),
            strict=False,
        )
    )
    field_repo_n = (
        pairs.group_by("document_field_name")
        .agg(pl.n_unique("repository_id").alias("n"))
        .filter(pl.col("n") > 0)
    )
    field_n_dict = dict(
        zip(
            field_repo_n["document_field_name"].to_list(),
            field_repo_n["n"].to_list(),
            strict=False,
        )
    )

    import_global = _compute_global_prevalence(imports_df, n_total_repos)
    import_field = _compute_field_prevalence(imports_df, repo_to_field, field_n_dict)
    dep_global = _compute_global_prevalence(deps_df, n_total_repos)
    dep_field = _compute_field_prevalence(deps_df, repo_to_field, field_n_dict)

    return import_global, import_field, dep_global, dep_field


def _get_eligible_document_ids(
    results_df: pl.DataFrame,
    filter_mode: str,
) -> tuple[set, set]:
    """Determine which document IDs are eligible for IM and DM analyses."""
    combiner = operator.and_ if filter_mode == "complete-cases" else operator.or_
    im_eligible = set(
        results_df.filter(combiner(pl.col("im_n_imports") > 0, pl.col("im_n_mentions") > 0))[
            "document_id"
        ].to_list()
    )
    dm_eligible = set(
        results_df.filter(combiner(pl.col("dm_n_deps") > 0, pl.col("dm_n_mentions") > 0))[
            "document_id"
        ].to_list()
    )
    return im_eligible, dm_eligible


def _build_analysis_df(
    records: list[dict],
    eligible: set,
    pair_meta: dict[int, dict],
    global_prev: dict[str, float],
    field_prev: dict[tuple[str, str], float],
    global_col: str,
    field_col: str,
) -> "pd.DataFrame":
    """Build a DataFrame for one analysis variant (imports or deps)."""
    import pandas as pd

    rows = []
    for rec in records:
        if rec["status"] == "source_b_only":
            continue
        doc_id = rec["document_id"]
        if doc_id not in eligible:
            continue
        meta = pair_meta.get(doc_id)
        if meta is None or meta["year"] is None:
            continue
        norm = rec["normalized_name"]
        field = meta["field"]
        rows.append(
            {
                "document_id": doc_id,
                "is_mentioned": int(rec["status"] == "matched"),
                global_col: global_prev.get(norm, 0.0),
                field_col: field_prev.get((norm, field), 0.0),
                "year": meta["year"],
                "field": field,
                "language": meta["language"],
            }
        )
    return pd.DataFrame(rows)


def _clean_logistic_df(df: "pd.DataFrame") -> "pd.DataFrame":
    """Clean and prepare a DataFrame for logistic regression."""
    df = df.dropna(subset=["year", "field", "language"]).copy()
    df["field"] = df["field"].str.replace(r"[^A-Za-z0-9_]", "_", regex=True).str.strip("_")
    df["language"] = (
        df["language"].str.replace(r"[^A-Za-z0-9_]", "_", regex=True).str.strip("_")
    )
    field_series = cast(pd.Series, df["field"])
    language_series = cast(pd.Series, df["language"])
    df["field_binned"] = _bin_top_n(field_series, top_n=9)
    df["language_binned"] = _bin_top_n(language_series, top_n=8)
    for col in (
        "import_global_prev",
        "import_field_prev",
        "dep_global_prev",
        "dep_field_prev",
    ):
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std()
            df[f"{col}_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0
    df["year_c"] = df["year"] - df["year"].median()
    return df


def _fit_models(
    df: "pd.DataFrame",
    model_specs: list[tuple[str, str]],
    section_title: str,
    min_n: int = 50,
) -> str:
    """Fit a series of logistic regression models and return formatted results."""
    import warnings

    if len(df) < min_n:
        return f"\n  {section_title}: insufficient data (N={len(df)}), skipping.\n"
    out_lines = [f"\n{'=' * 70}", f"  {section_title}  (N rows = {len(df):,})", "=" * 70]
    for model_name, formula in model_specs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = smf.logit(formula, data=df).fit(
                    disp=0,
                    maxiter=500,
                    method="bfgs",
                    cov_type="cluster",
                    cov_kwds={"groups": df["document_id"]},
                )
            out_lines.append(_format_logit_result(result, model_name))
        except Exception as exc:
            out_lines.append(f"\n  {model_name}: failed — {exc}\n")
    return "\n".join(out_lines)


def _fit_and_write_models(
    df: "pd.DataFrame",
    model_specs: list[tuple[str, str]],
    section_title: str,
    output_path: "Path",
    filename: str,
    filter_mode: str,
    label: str,
) -> None:
    """Fit models, log completion, and write results to file."""
    text = _fit_models(df, model_specs, section_title)
    log.info(f"[{filter_mode}] Logistic regression ({label}) complete")
    with open(output_path / filename, "w") as f:
        f.write(text)


def _build_combined_rows(
    all_im_records: list[dict],
    all_dm_records: list[dict],
    combined_eligible: set,
    pair_meta: dict[int, dict],
    import_global: dict[str, float],
    import_field: dict[tuple[str, str], float],
    dep_global: dict[str, float],
    dep_field: dict[tuple[str, str], float],
) -> list[dict]:
    """Build combined rows from both IM and DM records for the M6 model."""
    from itertools import chain

    combined: dict[tuple[str, int], dict] = {}
    for rec in chain(all_im_records, all_dm_records):
        if rec["status"] == "source_b_only":
            continue
        doc_id = rec["document_id"]
        if doc_id not in combined_eligible:
            continue
        meta = pair_meta.get(doc_id)
        if meta is None or meta["year"] is None:
            continue
        norm = rec["normalized_name"]
        field = meta["field"]
        key = (norm, doc_id)
        entry = combined.setdefault(
            key,
            {
                "document_id": doc_id,
                "is_mentioned": 0,
                "import_global_prev": import_global.get(norm, 0.0),
                "import_field_prev": import_field.get((norm, field), 0.0),
                "dep_global_prev": dep_global.get(norm, 0.0),
                "dep_field_prev": dep_field.get((norm, field), 0.0),
                "year": meta["year"],
                "field": field,
                "language": meta["language"],
            },
        )
        entry["is_mentioned"] = max(entry["is_mentioned"], int(rec["status"] == "matched"))
    return list(combined.values())


def _run_logistic_regressions(
    results_df: pl.DataFrame,
    all_im_records: list[dict],
    all_dm_records: list[dict],
    imports_df: pl.DataFrame,
    deps_df: pl.DataFrame,
    pairs: pl.DataFrame,
    filter_mode: str,
    output_path: "Path",
) -> None:
    """Fit logistic regressions predicting mention status from prevalence features."""
    import pandas as pd

    log.info(f"[{filter_mode}] Building logistic regression datasets...")

    pair_meta = _build_pair_metadata(results_df)
    import_global, import_field, dep_global, dep_field = _compute_prevalence_features(
        imports_df, deps_df, pairs
    )
    im_eligible, dm_eligible = _get_eligible_document_ids(results_df, filter_mode)

    # Build and clean per-analysis DataFrames
    im_df = _clean_logistic_df(
        _build_analysis_df(
            all_im_records,
            im_eligible,
            pair_meta,
            import_global,
            import_field,
            "import_global_prev",
            "import_field_prev",
        )
    )
    dm_df = _clean_logistic_df(
        _build_analysis_df(
            all_dm_records,
            dm_eligible,
            pair_meta,
            dep_global,
            dep_field,
            "dep_global_prev",
            "dep_field_prev",
        )
    )

    # ── field-prevalence model specs ─────────────────────────────────────────
    im_field_models = [
        ("M1: Uncontrolled", "is_mentioned ~ import_field_prev_z"),
        ("M2: + Year", "is_mentioned ~ import_field_prev_z + year_c"),
        (
            "M3: + Field",
            "is_mentioned ~ import_field_prev_z + C(field_binned)",
        ),
        (
            "M4: + Language",
            "is_mentioned ~ import_field_prev_z + C(language_binned)",
        ),
        (
            "M5: Fully controlled",
            "is_mentioned ~ import_field_prev_z + year_c"
            " + C(field_binned) + C(language_binned)",
        ),
    ]
    dm_field_models = [
        ("M1: Uncontrolled", "is_mentioned ~ dep_field_prev_z"),
        ("M2: + Year", "is_mentioned ~ dep_field_prev_z + year_c"),
        (
            "M3: + Field",
            "is_mentioned ~ dep_field_prev_z + C(field_binned)",
        ),
        (
            "M4: + Language",
            "is_mentioned ~ dep_field_prev_z + C(language_binned)",
        ),
        (
            "M5: Fully controlled",
            "is_mentioned ~ dep_field_prev_z + year_c + C(field_binned) + C(language_binned)",
        ),
    ]

    # ── global-prevalence model specs ─────────────────────────────────────────
    im_global_models = [
        ("M1: Uncontrolled", "is_mentioned ~ import_global_prev_z"),
        ("M2: + Year", "is_mentioned ~ import_global_prev_z + year_c"),
        (
            "M3: + Field",
            "is_mentioned ~ import_global_prev_z + C(field_binned)",
        ),
        (
            "M4: + Language",
            "is_mentioned ~ import_global_prev_z + C(language_binned)",
        ),
        (
            "M5: Fully controlled",
            "is_mentioned ~ import_global_prev_z + year_c"
            " + C(field_binned) + C(language_binned)",
        ),
    ]
    dm_global_models = [
        ("M1: Uncontrolled", "is_mentioned ~ dep_global_prev_z"),
        ("M2: + Year", "is_mentioned ~ dep_global_prev_z + year_c"),
        (
            "M3: + Field",
            "is_mentioned ~ dep_global_prev_z + C(field_binned)",
        ),
        (
            "M4: + Language",
            "is_mentioned ~ dep_global_prev_z + C(language_binned)",
        ),
        (
            "M5: Fully controlled",
            "is_mentioned ~ dep_global_prev_z + year_c + C(field_binned) + C(language_binned)",
        ),
    ]

    # ── fit and write field-prevalence results ────────────────────────────────
    _fit_and_write_models(
        im_df,
        im_field_models,
        "Logistic Regression: Import Field Prevalence → Mention Status",
        output_path,
        "rq3-logistic-regression-imports-field.txt",
        filter_mode,
        "imports-field",
    )
    _fit_and_write_models(
        dm_df,
        dm_field_models,
        "Logistic Regression: Dependency Field Prevalence → Mention Status",
        output_path,
        "rq3-logistic-regression-deps-field.txt",
        filter_mode,
        "deps-field",
    )

    # ── fit and write global-prevalence results ───────────────────────────────
    _fit_and_write_models(
        im_df,
        im_global_models,
        "Logistic Regression: Import Global Prevalence → Mention Status",
        output_path,
        "rq3-logistic-regression-imports-global.txt",
        filter_mode,
        "imports-global",
    )
    _fit_and_write_models(
        dm_df,
        dm_global_models,
        "Logistic Regression: Dependency Global Prevalence → Mention Status",
        output_path,
        "rq3-logistic-regression-deps-global.txt",
        filter_mode,
        "deps-global",
    )

    # ── combined models (M6) ─────────────────────────────────────────────────
    combined_eligible = im_eligible | dm_eligible
    combined_df = _clean_logistic_df(
        pd.DataFrame(
            _build_combined_rows(
                all_im_records,
                all_dm_records,
                combined_eligible,
                pair_meta,
                import_global,
                import_field,
                dep_global,
                dep_field,
            )
        )
    )
    combined_field_formula = (
        "is_mentioned ~ import_field_prev_z "
        "+ dep_field_prev_z "
        "+ year_c + C(field_binned) + C(language_binned)"
    )
    _fit_and_write_models(
        combined_df,
        [("M6: Combined (fully controlled)", combined_field_formula)],
        "Logistic Regression: Combined Field Prevalence → Mention Status",
        output_path,
        "rq3-logistic-regression-combined-field.txt",
        filter_mode,
        "combined-field",
    )
    combined_global_formula = (
        "is_mentioned ~ import_global_prev_z "
        "+ dep_global_prev_z "
        "+ year_c + C(field_binned) + C(language_binned)"
    )
    _fit_and_write_models(
        combined_df,
        [("M6: Combined (fully controlled)", combined_global_formula)],
        "Logistic Regression: Combined Global Prevalence → Mention Status",
        output_path,
        "rq3-logistic-regression-combined-global.txt",
        filter_mode,
        "combined-global",
    )

    # ── field + global prevalence together (collinearity check) ──────────────
    im_both_models = [
        (
            "M1: Uncontrolled",
            "is_mentioned ~ import_field_prev_z + import_global_prev_z",
        ),
        (
            "M5: Fully controlled",
            "is_mentioned ~ import_field_prev_z + import_global_prev_z"
            " + year_c + C(field_binned) + C(language_binned)",
        ),
    ]
    dm_both_models = [
        (
            "M1: Uncontrolled",
            "is_mentioned ~ dep_field_prev_z + dep_global_prev_z",
        ),
        (
            "M5: Fully controlled",
            "is_mentioned ~ dep_field_prev_z + dep_global_prev_z"
            " + year_c + C(field_binned) + C(language_binned)",
        ),
    ]

    # Imports: field + global
    im_both_diag = _collinearity_diagnostics(
        im_df, ["import_field_prev_z", "import_global_prev_z"]
    )
    im_both_text = _fit_models(
        im_df,
        im_both_models,
        "Logistic Regression: Import Field + Global Prevalence → Mention Status",
    )
    with open(output_path / "rq3-logistic-regression-imports-both.txt", "w") as f:
        f.write(im_both_diag + im_both_text)
    log.info(f"[{filter_mode}] Logistic regression (imports-both) complete")

    # Deps: field + global
    dm_both_diag = _collinearity_diagnostics(dm_df, ["dep_field_prev_z", "dep_global_prev_z"])
    dm_both_text = _fit_models(
        dm_df,
        dm_both_models,
        "Logistic Regression: Dependency Field + Global Prevalence → Mention Status",
    )
    with open(output_path / "rq3-logistic-regression-deps-both.txt", "w") as f:
        f.write(dm_both_diag + dm_both_text)
    log.info(f"[{filter_mode}] Logistic regression (deps-both) complete")

    # Combined (all four prevalence terms)
    combined_both_diag = _collinearity_diagnostics(
        combined_df,
        [
            "import_field_prev_z",
            "import_global_prev_z",
            "dep_field_prev_z",
            "dep_global_prev_z",
        ],
    )
    combined_both_formula = (
        "is_mentioned ~ import_field_prev_z + import_global_prev_z"
        " + dep_field_prev_z + dep_global_prev_z"
        " + year_c + C(field_binned) + C(language_binned)"
    )
    combined_both_text = _fit_models(
        combined_df,
        [("M6: Combined (fully controlled)", combined_both_formula)],
        "Logistic Regression: Combined Field + Global Prevalence → Mention Status",
    )
    with open(output_path / "rq3-logistic-regression-combined-both.txt", "w") as f:
        f.write(combined_both_diag + combined_both_text)
    log.info(f"[{filter_mode}] Logistic regression (combined-both) complete")


###############################################################################
# Extended Analyses (CCR, Temporal, Citation Classes, FWMP/FWIP)
###############################################################################


def _compute_ccr_for_pair(
    im_records: list[PairwiseSoftwareRecord],
    dm_records: list[PairwiseSoftwareRecord],
) -> tuple[float, float, float]:
    """Compute Credit Coverage Ratio for a single pair.

    CCR = |mentions ∩ (imports ∪ deps)| / |imports ∪ deps|

    Returns (ccr, ccr_imports, ccr_deps).
    """
    im_matched = {r.normalized_name for r in im_records if r.status == "matched"}
    dm_matched = {r.normalized_name for r in dm_records if r.status == "matched"}
    all_imports = {
        r.normalized_name for r in im_records if r.status in ("matched", "source_a_only")
    }
    all_deps = {
        r.normalized_name for r in dm_records if r.status in ("matched", "source_a_only")
    }

    code_union = all_imports | all_deps
    credited = im_matched | dm_matched

    ccr = len(credited) / len(code_union) if code_union else float("nan")
    ccr_imports = len(im_matched) / len(all_imports) if all_imports else float("nan")
    ccr_deps = len(dm_matched) / len(all_deps) if all_deps else float("nan")
    return ccr, ccr_imports, ccr_deps


def _write_ccr_analysis(
    results_df: pl.DataFrame,
    filter_mode: str,
    mode_path: Path,
) -> None:
    """Write Credit Coverage Ratio analysis to file."""
    from scipy.stats import kruskal

    if filter_mode == "complete-cases":
        df = results_df.filter((pl.col("im_n_imports") > 0) | (pl.col("dm_n_deps") > 0)).filter(
            pl.col("ccr").is_not_nan()
        )
    else:
        df = results_df.filter(pl.col("ccr").is_not_nan())

    lines: list[str] = [
        "=" * 70,
        f"  Credit Coverage Ratio Analysis  ({filter_mode}, N = {df.height})",
        "=" * 70,
        "",
        "─" * 70,
        "  Overall CCR Distribution",
        "─" * 70,
    ]

    for col, label in [
        ("ccr", "CCR (combined)"),
        ("ccr_imports", "CCR (imports only)"),
        ("ccr_deps", "CCR (deps only)"),
    ]:
        lines.append(_print_descriptive_stats(df, col, label))

    # Kruskal-Wallis across top fields
    lines.extend(["", "─" * 70, "  Kruskal-Wallis: CCR across Fields", "─" * 70])
    top_fields = (
        df.group_by("document_field_name")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(10)["document_field_name"]
        .to_list()
    )
    field_groups = []
    for field in top_fields:
        if field is None:
            continue
        vals = df.filter(pl.col("document_field_name") == field)["ccr"].drop_nulls().to_list()
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            field_groups.append(vals)

    if len(field_groups) >= 2:
        h_stat, p_val = kruskal(*field_groups)
        p_str = f"{p_val:.4e}" if p_val >= 0.0001 else "<.0001"
        lines.append(f"  H-statistic = {h_stat:.4f}, p = {p_str}")
    else:
        lines.append("  Insufficient field groups for Kruskal-Wallis test")

    # Mann-Whitney U for binary splits
    lines.extend(["", "─" * 70, "  Mann-Whitney U: Binary Splits", "─" * 70])

    binary_splits: list[tuple[str, str, pl.Expr, pl.Expr]] = [
        (
            "Shared vs Mined",
            "dataset_source_name",
            pl.col("dataset_source_name") == "mined",
            pl.col("dataset_source_name") != "mined",
        ),
        (
            "Open Access vs Closed",
            "document_is_open_access",
            pl.col("document_is_open_access") == True,  # noqa: E712
            pl.col("document_is_open_access") == False,  # noqa: E712
        ),
    ]
    for split_label, _col, expr_a, expr_b in binary_splits:
        vals_a = [v for v in df.filter(expr_a)["ccr"].to_list() if not np.isnan(v)]
        vals_b = [v for v in df.filter(expr_b)["ccr"].to_list() if not np.isnan(v)]
        if len(vals_a) >= 5 and len(vals_b) >= 5:
            u_stat, p_val = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            p_str = f"{p_val:.4e}" if p_val >= 0.0001 else "<.0001"
            lines.append(
                f"  {split_label}: U = {u_stat:.1f}, p = {p_str}, "
                f"median_a = {np.median(vals_a):.4f}, median_b = {np.median(vals_b):.4f}"
            )
        else:
            lines.append(f"  {split_label}: insufficient data")

    # Breakdowns by grouping variables
    grouping_cols: list[tuple[str, str, int | None]] = [
        ("document_field_name", "Field", 15),
        ("document_publication_year", "Publication Year", None),
        ("repository_primary_language", "Language", 10),
        ("dataset_source_name", "Dataset Source", None),
    ]
    for col, col_label, tnf in grouping_cols:
        lines.append(_print_group_breakdown(df, col, col_label, "ccr", "ccr", top_n_filter=tnf))

    text = "\n".join(lines)
    print(text)
    with open(mode_path / "rq3-ccr-analysis.txt", "w") as f:
        f.write(text)
    log.info(f"[{filter_mode}] Saved rq3-ccr-analysis.txt")


def _write_temporal_trends(
    results_df: pl.DataFrame,
    filter_mode: str,
    mode_path: Path,
) -> None:
    """Write temporal trend analysis for CCR and software counts."""
    from scipy.stats import theilslopes

    df = results_df.filter(pl.col("ccr").is_not_nan())

    year_stats = (
        df.group_by("document_publication_year")
        .agg(
            pl.col("ccr").median().alias("median_ccr"),
            pl.col("ccr").mean().alias("mean_ccr"),
            pl.col("im_n_imports").median().alias("median_imports"),
            pl.col("dm_n_deps").median().alias("median_deps"),
            pl.col("im_n_mentions").median().alias("median_mentions"),
            pl.col("im_n_matched").median().alias("median_matched_im"),
            pl.col("dm_n_matched").median().alias("median_matched_dm"),
            pl.len().alias("n_pairs"),
        )
        .filter(pl.col("n_pairs") >= 5)
        .sort("document_publication_year")
    )

    lines: list[str] = [
        "=" * 70,
        f"  Temporal Trends Analysis  ({filter_mode})",
        "=" * 70,
        "",
    ]

    # Yearly table
    lines.append(
        f"  {'Year':<6} {'N':>6} {'CCR med':>8} {'CCR mean':>9} "
        f"{'Imp med':>8} {'Dep med':>8} {'Men med':>8}"
    )
    lines.append(f"  {'─' * 60}")
    for row in year_stats.iter_rows(named=True):
        year = row["document_publication_year"]
        if year is None:
            continue
        lines.append(
            f"  {year:<6} {row['n_pairs']:>6} "
            f"{row['median_ccr'] or 0:>8.4f} {row['mean_ccr'] or 0:>9.4f} "
            f"{row['median_imports'] or 0:>8.1f} {row['median_deps'] or 0:>8.1f} "
            f"{row['median_mentions'] or 0:>8.1f}"
        )

    # Theil-Sen regression: CCR on year
    years = year_stats["document_publication_year"].drop_nulls().to_list()
    ccr_medians = year_stats["median_ccr"].drop_nulls().to_list()
    if len(years) >= 3:
        slope, intercept, lo_slope, hi_slope = theilslopes(ccr_medians, years)
        lines.extend(
            [
                "",
                "─" * 70,
                "  Theil-Sen Regression: median CCR ~ year",
                "─" * 70,
                f"  Slope = {slope:.6f} per year",
                f"  95% CI: [{lo_slope:.6f}, {hi_slope:.6f}]",
                f"  Intercept = {intercept:.4f}",
                f"  Interpretation: {'increasing' if slope > 0 else 'decreasing'} trend "
                f"({'significant' if lo_slope > 0 or hi_slope < 0 else 'not significant'} "
                f"at 95% CI)",
            ]
        )

        # Decomposition: denominator vs numerator slopes
        import_medians = year_stats["median_imports"].drop_nulls().to_list()
        dep_medians = year_stats["median_deps"].drop_nulls().to_list()
        mention_medians = year_stats["median_mentions"].drop_nulls().to_list()

        lines.extend(["", "  Decomposition (Theil-Sen slopes of component medians):"])
        for label, vals in [
            ("Imports", import_medians),
            ("Dependencies", dep_medians),
            ("Mentions", mention_medians),
        ]:
            if len(vals) >= 3:
                s, _i, _lo, _hi = theilslopes(vals, years[: len(vals)])
                lines.append(f"    {label}: slope = {s:.4f} per year")
    else:
        lines.append("\n  Insufficient years for Theil-Sen regression")

    text = "\n".join(lines)
    print(text)
    with open(mode_path / "rq3-temporal-trends.txt", "w") as f:
        f.write(text)
    log.info(f"[{filter_mode}] Saved rq3-temporal-trends.txt")


def _plot_temporal_trends(
    results_df: pl.DataFrame,
    filter_mode: str,
    mode_path: Path,
) -> None:
    """Plot temporal trends for CCR and software counts."""
    from scipy.stats import theilslopes

    df = results_df.filter(pl.col("ccr").is_not_nan())

    year_stats = (
        df.group_by("document_publication_year")
        .agg(
            pl.col("ccr").median().alias("median_ccr"),
            pl.col("ccr").mean().alias("mean_ccr"),
            pl.col("ccr").std().alias("std_ccr"),
            pl.col("im_n_imports").median().alias("median_imports"),
            pl.col("dm_n_deps").median().alias("median_deps"),
            pl.col("im_n_mentions").median().alias("median_mentions"),
            pl.len().alias("n_pairs"),
        )
        .filter(pl.col("n_pairs") >= 5)
        .sort("document_publication_year")
    )

    years = year_stats["document_publication_year"].drop_nulls().to_list()
    if len(years) < 2:
        log.warning("Insufficient data for temporal trend plots, skipping.")
        return

    # Panel 1: CCR median by year with Theil-Sen line
    ccr_medians = year_stats["median_ccr"].to_list()
    ccr_stds = [s or 0.0 for s in year_stats["std_ccr"].to_list()]
    ns = year_stats["n_pairs"].to_list()
    ci = [1.96 * s / (n**0.5) for s, n in zip(ccr_stds, ns, strict=False)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(years, ccr_medians, marker="o", linewidth=2, label="Median CCR")
    lower = [max(0.0, m - c) for m, c in zip(ccr_medians, ci, strict=False)]
    upper = [m + c for m, c in zip(ccr_medians, ci, strict=False)]
    ax1.fill_between(years, lower, upper, alpha=0.2)

    if len(years) >= 3:
        slope, intercept, _lo, _hi = theilslopes(ccr_medians, years)
        trend_y = [intercept + slope * y for y in years]
        ax1.plot(years, trend_y, "--", color="red", alpha=0.7, label="Theil-Sen trend")

    ax1.set_xlabel("Publication Year")
    ax1.set_ylabel("Median CCR")
    ax1.set_title("Credit Coverage Ratio Over Time")
    ax1.set_ylim(bottom=0)
    ax1.legend()

    # Panel 2: Counts over time
    ax2.plot(
        years,
        year_stats["median_imports"].to_list(),
        marker="s",
        label="Imports",
    )
    ax2.plot(
        years,
        year_stats["median_deps"].to_list(),
        marker="^",
        label="Dependencies",
    )
    ax2.plot(
        years,
        year_stats["median_mentions"].to_list(),
        marker="D",
        label="Mentions",
    )
    ax2.set_xlabel("Publication Year")
    ax2.set_ylabel("Median Count per Pair")
    ax2.set_title("Software Counts Over Time")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(mode_path / "rq3-temporal-trends.png", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[{filter_mode}] Saved rq3-temporal-trends.png")


# ── Citation Classes ────────────────────────────────────────────────────────

_CITATION_CLASSES: list[tuple[int, str, bool, bool, bool]] = [
    # (class_id, label, in_mentions, in_imports, in_deps)
    (1, "Fully visible", True, True, True),
    (2, "Mentioned & imported", True, True, False),
    (3, "Mentioned & declared", True, False, True),
    (4, "Cited but not in code", True, False, False),
    (5, "Used & declared, uncredited", False, True, True),
    (6, "Used but uncredited", False, True, False),
    (7, "Declared but invisible", False, False, True),
]


def _compute_citation_classes(
    all_im_records: list[dict],
    all_dm_records: list[dict],
    all_id_records: list[dict],
    results_df: pl.DataFrame,
    filter_mode: str,
) -> tuple[pl.DataFrame, dict[str, list[tuple[str, int]]]]:
    """Classify each unique software entity by presence across three views.

    Returns:
        class_df: DataFrame with class_id, class_label, n_unique_packages, n_total_occurrences
        top_per_class: dict mapping class_label to list of (name, occurrence_count)
    """
    # Determine eligible doc_ids
    if filter_mode == "complete-cases":
        eligible = set(
            results_df.filter(
                ((pl.col("im_n_imports") > 0) & (pl.col("im_n_mentions") > 0))
                | ((pl.col("id_n_imports") > 0) & (pl.col("id_n_deps") > 0))
                | ((pl.col("dm_n_deps") > 0) & (pl.col("dm_n_mentions") > 0))
            )["document_id"].to_list()
        )
    else:
        eligible = set(
            results_df.filter(
                (pl.col("im_n_imports") > 0)
                | (pl.col("im_n_mentions") > 0)
                | (pl.col("dm_n_deps") > 0)
            )["document_id"].to_list()
        )

    # Build per-software presence sets and occurrence counts
    in_mentions: set[str] = set()
    in_imports: set[str] = set()
    in_deps: set[str] = set()
    occurrence_counts: Counter[str] = Counter()

    for rec in all_im_records:
        if rec["document_id"] not in eligible:
            continue
        norm = rec["normalized_name"]
        if rec["status"] in ("matched", "source_a_only"):
            in_imports.add(norm)
            occurrence_counts[norm] += 1
        if rec["status"] in ("matched", "source_b_only"):
            in_mentions.add(norm)
            if rec["status"] == "source_b_only":
                occurrence_counts[norm] += 1

    for rec in all_dm_records:
        if rec["document_id"] not in eligible:
            continue
        norm = rec["normalized_name"]
        if rec["status"] in ("matched", "source_a_only"):
            in_deps.add(norm)
            occurrence_counts[norm] += 1
        if rec["status"] in ("matched", "source_b_only"):
            in_mentions.add(norm)
            if rec["status"] == "source_b_only":
                occurrence_counts[norm] += 1

    for rec in all_id_records:
        if rec["document_id"] not in eligible:
            continue
        norm = rec["normalized_name"]
        if rec["status"] in ("matched", "source_a_only"):
            in_imports.add(norm)
        if rec["status"] in ("matched", "source_b_only"):
            in_deps.add(norm)

    # Classify each software entity
    all_software = in_mentions | in_imports | in_deps
    class_counts: dict[int, int] = {c[0]: 0 for c in _CITATION_CLASSES}
    class_occurrences: dict[int, int] = {c[0]: 0 for c in _CITATION_CLASSES}
    software_to_class: dict[str, int] = {}
    top_per_class: dict[str, list[tuple[str, int]]] = {}

    for norm in all_software:
        m = norm in in_mentions
        i = norm in in_imports
        d = norm in in_deps
        for class_id, _label, cm, ci, cd in _CITATION_CLASSES:
            if m == cm and i == ci and d == cd:
                class_counts[class_id] += 1
                class_occurrences[class_id] += occurrence_counts.get(norm, 0)
                software_to_class[norm] = class_id
                break

    # Build top-10 per class
    for class_id, label, *_ in _CITATION_CLASSES:
        class_software = [
            (norm, occurrence_counts.get(norm, 0))
            for norm, cid in software_to_class.items()
            if cid == class_id
        ]
        class_software.sort(key=lambda x: x[1], reverse=True)
        top_per_class[label] = class_software[:10]

    rows = []
    for class_id, label, *_ in _CITATION_CLASSES:
        rows.append(
            {
                "class_id": class_id,
                "class_label": label,
                "n_unique_packages": class_counts[class_id],
                "n_total_occurrences": class_occurrences[class_id],
            }
        )

    return pl.DataFrame(rows), top_per_class


def _write_citation_classes(
    all_im_records: list[dict],
    all_dm_records: list[dict],
    all_id_records: list[dict],
    results_df: pl.DataFrame,
    filter_mode: str,
    mode_path: Path,
) -> None:
    """Compute and write software citation class analysis."""
    class_df, top_per_class = _compute_citation_classes(
        all_im_records, all_dm_records, all_id_records, results_df, filter_mode
    )

    total_packages = class_df["n_unique_packages"].sum()
    total_occurrences = class_df["n_total_occurrences"].sum()

    lines: list[str] = [
        "=" * 70,
        f"  Software Citation Classes  ({filter_mode})",
        f"  Total unique packages: {total_packages:,}",
        f"  Total occurrences: {total_occurrences:,}",
        "=" * 70,
        "",
        f"  {'#':<3} {'Class':<35} {'Packages':>10} {'%':>7} {'Occurrences':>13} {'%':>7}",
        f"  {'─' * 78}",
    ]

    for row in class_df.iter_rows(named=True):
        pkg_pct = row["n_unique_packages"] / total_packages * 100 if total_packages > 0 else 0
        occ_pct = (
            row["n_total_occurrences"] / total_occurrences * 100 if total_occurrences > 0 else 0
        )
        lines.append(
            f"  {row['class_id']:<3} {row['class_label']:<35} "
            f"{row['n_unique_packages']:>10,} {pkg_pct:>6.1f}% "
            f"{row['n_total_occurrences']:>13,} {occ_pct:>6.1f}%"
        )

    # Top 10 per class
    lines.extend(["", "─" * 70, "  Top 10 Packages per Class", "─" * 70])
    for class_id, label, *_ in _CITATION_CLASSES:
        top = top_per_class.get(label, [])
        if not top:
            continue
        lines.append(f"\n  [{class_id}] {label}:")
        for rank, (norm, count) in enumerate(top, 1):
            display = prep_name_for_printing(norm)
            lines.append(f"    {rank:>2}. {display:<40} ({count:,} occurrences)")

    text = "\n".join(lines)
    print(text)
    with open(mode_path / "rq3-citation-classes.txt", "w") as f:
        f.write(text)
    log.info(f"[{filter_mode}] Saved rq3-citation-classes.txt")


# ── Field-Weighted Mention/Import Prevalence ────────────────────────────────


def _compute_fwmp_fwip(
    all_im_records: list[dict],
    results_df: pl.DataFrame,
    pairs: pl.DataFrame,
    filter_mode: str,
    min_imports: int = 10,
    min_stratum_pairs: int = 5,
) -> pl.DataFrame:
    """Compute Field-Weighted Mention Prevalence and Import Prevalence.

    For each software with >=min_imports, compute FWMP and FWIP using
    per-(field, year) stratum normalization.
    """
    # Build pair metadata lookup: document_id -> (field, year)
    pair_meta: dict[int, tuple[str | None, int | None]] = {}
    for row in results_df.iter_rows(named=True):
        pair_meta[row["document_id"]] = (
            row["document_field_name"],
            row["document_publication_year"],
        )

    # Determine eligible doc_ids
    if filter_mode == "complete-cases":
        eligible = set(
            results_df.filter((pl.col("im_n_imports") > 0) & (pl.col("im_n_mentions") > 0))[
                "document_id"
            ].to_list()
        )
    else:
        eligible = set(
            results_df.filter((pl.col("im_n_imports") > 0) | (pl.col("im_n_mentions") > 0))[
                "document_id"
            ].to_list()
        )

    # Build per-software, per-stratum counts from IM records
    # For each (software, field, year): count imported, count mentioned
    StratumKey = tuple[str | None, int | None]
    sw_stratum_imported: dict[str, dict[StratumKey, int]] = {}
    sw_stratum_mentioned: dict[str, dict[StratumKey, int]] = {}
    stratum_all_imported: dict[StratumKey, int] = Counter()
    stratum_all_mentioned: dict[StratumKey, int] = Counter()

    for rec in all_im_records:
        doc_id = rec["document_id"]
        if doc_id not in eligible:
            continue
        meta = pair_meta.get(doc_id)
        if meta is None or meta[0] is None or meta[1] is None:
            continue
        stratum = meta  # (field, year)
        norm = rec["normalized_name"]

        if rec["status"] in ("matched", "source_a_only"):
            sw_stratum_imported.setdefault(norm, Counter())[stratum] += 1
            stratum_all_imported[stratum] += 1
        if rec["status"] == "matched":
            sw_stratum_mentioned.setdefault(norm, Counter())[stratum] += 1
            stratum_all_mentioned[stratum] += 1

    # Filter to software with >= min_imports
    total_imports: dict[str, int] = {
        sw: sum(counts.values()) for sw, counts in sw_stratum_imported.items()
    }
    qualifying = {sw for sw, n in total_imports.items() if n >= min_imports}

    # Compute expected mention rate per stratum (mean across all software in stratum)
    # expected_rate_k = total_mentioned_in_k / total_imported_in_k
    stratum_expected_rate: dict[StratumKey, float] = {}
    for stratum in stratum_all_imported:
        n_imp = stratum_all_imported[stratum]
        n_men = stratum_all_mentioned.get(stratum, 0)
        stratum_expected_rate[stratum] = n_men / n_imp if n_imp > 0 else 0.0

    # Compute FWMP for each qualifying software
    rows: list[dict] = []
    for sw in qualifying:
        strata = sw_stratum_imported.get(sw, {})
        total_weight = 0.0
        weighted_ratio_sum = 0.0
        n_strata_used = 0
        fields_seen: set[str | None] = set()

        for stratum, n_imported_here in strata.items():
            if n_imported_here < 1:
                continue
            # Check stratum has enough total pairs
            total_in_stratum = stratum_all_imported.get(stratum, 0)
            if total_in_stratum < min_stratum_pairs:
                continue

            expected = stratum_expected_rate.get(stratum, 0.0)
            if expected <= 0:
                continue

            n_mentioned_here = sw_stratum_mentioned.get(sw, {}).get(stratum, 0)
            observed_rate = n_mentioned_here / n_imported_here
            ratio = observed_rate / expected

            weighted_ratio_sum += n_imported_here * ratio
            total_weight += n_imported_here
            n_strata_used += 1
            fields_seen.add(stratum[0])

        fwmp = weighted_ratio_sum / total_weight if total_weight > 0 else float("nan")

        # FWIP: among pairs that mention S, what fraction also import S,
        # relative to the stratum average import rate given mention
        # For simplicity and symmetry, compute FWIP as the analogous metric
        # where we ask: how prevalent is this import relative to expected?
        # FWIP = weighted_avg of (observed_import_rate / expected_import_rate)
        # But since we condition on imports in the main loop, FWIP simplifies to:
        # the absolute prevalence as import rate (fraction of all pairs importing S)
        total_mentioned = sum(sw_stratum_mentioned.get(sw, {}).values())

        rows.append(
            {
                "normalized_name": sw,
                "n_total_imports": total_imports[sw],
                "n_total_mentions": total_mentioned,
                "fwmp": fwmp,
                "n_strata": n_strata_used,
                "n_fields": len(fields_seen - {None}),
            }
        )

    result = pl.DataFrame(rows)

    # Compute FWIP as the raw import prevalence (fraction of eligible pairs importing S)
    n_eligible = len(eligible)
    if n_eligible > 0:
        result = result.with_columns(
            (pl.col("n_total_imports") / n_eligible).alias("fwip"),
        )
    else:
        result = result.with_columns(pl.lit(float("nan")).alias("fwip"))

    return result.sort("n_total_imports", descending=True)


def _plot_fwmp_vs_fwip(
    fwmp_fwip_df: pl.DataFrame,
    mode_path: Path,
) -> None:
    """Scatter plot of FWMP vs FWIP with fair-credit diagonal."""
    df = fwmp_fwip_df.filter(
        pl.col("fwmp").is_not_nan()
        & pl.col("fwip").is_not_nan()
        & (pl.col("fwmp") > 0)
        & (pl.col("fwip") > 0)
    )
    if df.height < 5:
        log.warning("Insufficient data for FWMP vs FWIP scatter, skipping.")
        return

    fwmp_log = np.log10(1 + df["fwmp"].to_numpy())
    fwip_log = np.log10(1 + df["fwip"].to_numpy())
    names = df["normalized_name"].to_list()
    n_imports = df["n_total_imports"].to_list()

    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(
        fwip_log,
        fwmp_log,
        s=np.clip(np.array(n_imports) / 5, 5, 200),
        alpha=0.4,
        edgecolors="white",
        linewidth=0.5,
    )

    # Fair credit diagonal
    lim_min = min(fwip_log.min(), fwmp_log.min()) - 0.05
    lim_max = max(fwip_log.max(), fwmp_log.max()) + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="gray", alpha=0.5)

    # Label outliers (top 5 most overcredited and undercredited)
    ratio = fwmp_log - fwip_log
    top_over = np.argsort(ratio)[-5:]
    top_under = np.argsort(ratio)[:5]
    for idx in list(top_over) + list(top_under):
        display = prep_name_for_printing(names[idx])
        ax.annotate(
            display,
            (fwip_log[idx], fwmp_log[idx]),
            fontsize=7,
            alpha=0.8,
            textcoords="offset points",
            xytext=(5, 5),
        )

    ax.set_xlabel("log10(1 + FWIP)")
    ax.set_ylabel("log10(1 + FWMP)")
    ax.set_title("Field-Weighted Mention vs Import Prevalence")

    # Quadrant labels
    ax.text(
        0.05,
        0.95,
        "Overcredited",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.5,
        va="top",
    )
    ax.text(
        0.95,
        0.05,
        "Undercredited",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.5,
        ha="right",
    )

    plt.tight_layout()
    fig.savefig(mode_path / "rq3-fwmp-vs-fwip-scatter.png", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved rq3-fwmp-vs-fwip-scatter.png")


def _write_fwmp_fwip_analysis(
    all_im_records: list[dict],
    results_df: pl.DataFrame,
    pairs: pl.DataFrame,
    filter_mode: str,
    mode_path: Path,
) -> None:
    """Orchestrate FWMP/FWIP computation, reporting, and plotting."""
    log.info(f"[{filter_mode}] Computing FWMP/FWIP...")
    fwmp_fwip_df = _compute_fwmp_fwip(all_im_records, results_df, pairs, filter_mode)

    valid = fwmp_fwip_df.filter(pl.col("fwmp").is_not_nan())
    lines: list[str] = [
        "=" * 70,
        f"  Field-Weighted Mention/Import Prevalence  ({filter_mode})",
        f"  Qualifying software (>=10 imports): {fwmp_fwip_df.height}",
        f"  With valid FWMP: {valid.height}",
        "=" * 70,
        "",
    ]

    # Overall FWMP distribution
    if valid.height > 0:
        lines.append(_print_descriptive_stats(valid, "fwmp", "FWMP"))
        lines.append(_print_descriptive_stats(valid, "fwip", "FWIP"))
        lines.append(_print_descriptive_stats(valid, "n_strata", "Strata per software"))
        lines.append(_print_descriptive_stats(valid, "n_fields", "Fields per software"))

    # Top overcredited (high FWMP)
    col_w = 35
    overcredited = valid.sort("fwmp", descending=True).head(20)
    lines.extend(
        [
            "",
            "─" * 70,
            "  Top 20 Overcredited (highest FWMP — mentioned more than expected)",
            "─" * 70,
            f"  {'Package':<{col_w}} {'FWMP':>8} {'FWIP':>8} "
            f"{'Imports':>8} {'Mentions':>8} {'Fields':>6}",
            f"  {'─' * 76}",
        ]
    )
    for row in overcredited.iter_rows(named=True):
        display = prep_name_for_printing(row["normalized_name"])[:col_w]
        lines.append(
            f"  {display:<{col_w}} {row['fwmp']:>8.3f} {row['fwip']:>8.6f} "
            f"{row['n_total_imports']:>8,} {row['n_total_mentions']:>8,} "
            f"{row['n_fields']:>6}"
        )

    # Top undercredited (low FWMP, high imports)
    undercredited = valid.filter(pl.col("n_total_imports") >= 50).sort("fwmp").head(20)
    lines.extend(
        [
            "",
            "─" * 70,
            "  Top 20 Undercredited (lowest FWMP, >=50 imports)",
            "─" * 70,
            f"  {'Package':<{col_w}} {'FWMP':>8} {'FWIP':>8} "
            f"{'Imports':>8} {'Mentions':>8} {'Fields':>6}",
            f"  {'─' * 76}",
        ]
    )
    for row in undercredited.iter_rows(named=True):
        display = prep_name_for_printing(row["normalized_name"])[:col_w]
        lines.append(
            f"  {display:<{col_w}} {row['fwmp']:>8.3f} {row['fwip']:>8.6f} "
            f"{row['n_total_imports']:>8,} {row['n_total_mentions']:>8,} "
            f"{row['n_fields']:>6}"
        )

    text = "\n".join(lines)
    print(text)
    with open(mode_path / "rq3-fwmp-fwip.txt", "w") as f:
        f.write(text)
    log.info(f"[{filter_mode}] Saved rq3-fwmp-fwip.txt")

    # Scatter plot
    _plot_fwmp_vs_fwip(fwmp_fwip_df, mode_path)


###############################################################################
# Pipeline Helpers
###############################################################################


def _build_lookup(
    df: pl.DataFrame,
    id_col: str,
) -> dict[int, tuple[list[str], list[str]]]:
    """Build a per-ID lookup of (names, normalized_names) for fast pair iteration."""
    lut: dict[int, tuple[list[str], list[str]]] = {}
    for row in df.iter_rows(named=True):
        key = row[id_col]
        if key not in lut:
            lut[key] = ([], [])
        lut[key][0].append(row["software_name"])
        lut[key][1].append(row["software_name_normalized"])
    return lut


###############################################################################
# Processing-Time Bias Check
###############################################################################

_BIAS_NUMERIC_FEATURES = [
    "document_cited_by_count",
    "document_fwci",
    "document_n_authors",
    "repository_stargazers_count",
    "repository_commits_count",
    "repository_n_contributors",
]

_BIAS_CATEGORICAL_FEATURES = [
    "document_is_open_access",
    "document_field_name",
    "document_type",
    "repository_primary_language",
]


def _effect_magnitude(val: float) -> str:
    """Classify absolute effect size magnitude."""
    val = abs(val)
    if val < 0.1:
        return "negligible"
    if val < 0.3:
        return "small"
    if val < 0.5:
        return "medium"
    return "large"


def _bias_numeric_tests(
    group_yes: pl.DataFrame,
    group_no: pl.DataFrame,
    columns: list[str],
) -> pl.DataFrame:
    """Run Mann-Whitney U tests on numeric features for the bias check."""
    n_tests = len(columns)
    rows: list[dict[str, str | int | float]] = []

    for col_name in columns:
        yes_vals = group_yes[col_name].drop_nulls().drop_nans().to_numpy()
        no_vals = group_no[col_name].drop_nulls().drop_nans().to_numpy()

        if len(yes_vals) < 2 or len(no_vals) < 2:
            log.warning("Skipping %s: too few non-null values.", col_name)
            continue

        u_stat, p_val = mannwhitneyu(yes_vals, no_vals, alternative="two-sided")
        n1, n2 = len(yes_vals), len(no_vals)
        r = 1 - (2 * u_stat) / (n1 * n2)
        p_bonf = min(p_val * n_tests, 1.0)

        rows.append(
            {
                "feature": col_name,
                "n_has": n1,
                "n_missing": n2,
                "median_has": float(np.median(yes_vals)),
                "median_missing": float(np.median(no_vals)),
                "mean_has": float(np.mean(yes_vals)),
                "mean_missing": float(np.mean(no_vals)),
                "u_statistic": float(u_stat),
                "p_value": float(p_val),
                "p_value_bonferroni": float(p_bonf),
                "rank_biserial_r": round(r, 4),
                "direction": "higher_in_has" if r > 0 else "higher_in_missing",
                "effect_magnitude": _effect_magnitude(r),
            }
        )

    return pl.DataFrame(rows).sort("p_value")


def _bias_categorical_tests(
    work: pl.DataFrame,
    top_n: int,
) -> pl.DataFrame:
    """Run Chi-square tests on categorical features for the bias check."""
    n_tests = len(_BIAS_CATEGORICAL_FEATURES)
    rows: list[dict[str, str | int | float]] = []

    for cat_col in _BIAS_CATEGORICAL_FEATURES:
        if cat_col not in work.columns:
            continue

        cat_work = work.with_columns(pl.col(cat_col).fill_null("Unknown"))
        top_values = (
            cat_work.filter(pl.col(cat_col).is_not_null())[cat_col]
            .value_counts(sort=True)
            .head(top_n)[cat_col]
            .to_list()
        )
        top_col = f"{cat_col}_top_n_plus_other"
        cat_work = cat_work.with_columns(
            pl.when(pl.col(cat_col).is_in(top_values))
            .then(pl.col(cat_col))
            .otherwise(pl.lit("Other"))
            .alias(top_col)
        )

        crosstab = (
            cat_work.group_by(["has_imports_or_deps", top_col])
            .len()
            .pivot(on=top_col, index="has_imports_or_deps", values="len")
            .fill_null(0)
        )

        value_cols = [c for c in crosstab.columns if c != "has_imports_or_deps"]
        table = crosstab.select(value_cols).to_numpy()

        chi2, p_val, dof, _ = chi2_contingency(table)
        v = association(table, method="cramer")
        p_bonf = min(p_val * n_tests, 1.0)

        rows.append(
            {
                "feature": cat_col,
                "n_total": int(table.sum()),
                "n_levels": len(value_cols),
                "chi2": round(chi2, 2),
                "dof": int(dof),
                "p_value": float(p_val),
                "p_value_bonferroni": float(p_bonf),
                "cramers_v": round(v, 4),
                "effect_magnitude": _effect_magnitude(v),
            }
        )

    return pl.DataFrame(rows).sort("p_value")


def _bias_significance_marker(p_bonf: float) -> str:
    """Return significance stars for a Bonferroni-corrected p-value."""
    if p_bonf < 0.001:
        return "***"
    if p_bonf < 0.01:
        return "**"
    if p_bonf < 0.05:
        return "*"
    return ""


def _check_processing_bias(
    pairs: pl.DataFrame,
    imports_by_repo: dict[int, tuple[list[str], list[str]]],
    deps_by_repo: dict[int, tuple[list[str], list[str]]],
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Compare repo/article metrics between pairs with vs without imports or deps.

    Import and dependency extraction was limited to repositories processable
    within a 120s clone+process time limit. This check tests whether that
    filter introduces systematic bias in observable metrics.
    """
    log.info("Running processing-time bias check...")

    available_repo_ids = set(imports_by_repo.keys()) | set(deps_by_repo.keys())
    work = pairs.with_columns(
        pl.col("repository_id").is_in(list(available_repo_ids)).alias("has_imports_or_deps"),
    )

    group_yes = work.filter(pl.col("has_imports_or_deps"))
    group_no = work.filter(~pl.col("has_imports_or_deps"))
    n_yes, n_no = group_yes.height, group_no.height
    log.info("Bias check groups — has imports/deps: %d, missing: %d", n_yes, n_no)

    lines: list[str] = [
        "Processing-Time Bias Check",
        "=" * 60,
        f"Repos with imports or deps available: {n_yes}",
        f"Repos without imports or deps:        {n_no}",
        "",
    ]

    # ── Numeric features: Mann-Whitney U ──
    numeric_cols = [c for c in _BIAS_NUMERIC_FEATURES if c in work.columns]
    numeric_df = _bias_numeric_tests(group_yes, group_no, numeric_cols)
    numeric_df.write_csv(output_path / "rq3-processing-bias-numeric.csv")

    lines.append("Numeric Features (Mann-Whitney U)")
    lines.append("-" * 40)
    for row in numeric_df.iter_rows(named=True):
        sig = _bias_significance_marker(row["p_value_bonferroni"])
        lines.append(
            f"  {sig}{row['feature']}{sig}: "
            f"r={row['rank_biserial_r']:.3f} ({row['effect_magnitude']}), "
            f"{row['direction']}, "
            f"median has={row['median_has']:.2f} vs missing={row['median_missing']:.2f}, "
            f"p={row['p_value']:.2e} (Bonf: {row['p_value_bonferroni']:.2e})"
        )
    lines.append("")

    # ── Categorical features: Chi-square ──
    cat_df = _bias_categorical_tests(work, top_n)
    cat_df.write_csv(output_path / "rq3-processing-bias-categorical.csv")

    lines.append("Categorical Features (Chi-square)")
    lines.append("-" * 40)
    for row in cat_df.iter_rows(named=True):
        sig = _bias_significance_marker(row["p_value_bonferroni"])
        lines.append(
            f"  {sig}{row['feature']}{sig}: "
            f"Cramer's V={row['cramers_v']:.3f} ({row['effect_magnitude']}), "
            f"chi2={row['chi2']:.1f}, dof={row['dof']}, "
            f"p={row['p_value']:.2e} (Bonf: {row['p_value_bonferroni']:.2e})"
        )
    lines.append("")

    summary_text = "\n".join(lines)
    print(summary_text)
    with open(output_path / "rq3-processing-bias-check.txt", "w") as f:
        f.write(summary_text)
    log.info("Saved rq3-processing-bias-check.txt")


###############################################################################
# Main Command
###############################################################################


@app.command()
def analyze(
    score_cutoff: float = SCORE_CUTOFF,
    top_n: int = 10,
    output_dir: str = str(THIS_DIR / "rq3-results"),
    debug: bool = False,
) -> None:
    """
    RQ3 analysis: three views of scientific software use.

    Runs three independent pairwise alignments per document-repository pair
    (imports vs mentions, imports vs deps, deps vs mentions), then produces
    summary statistics and publication-quality visualizations.
    """
    setup_logger(debug)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pairs with full metadata
    log.info("Loading document-repository pairs...")
    pairs = load_pairs()
    log.info(f"Loaded {pairs.height} pairs")

    # Load three software tables
    log.info("Loading software tables from database...")
    imports_df = _read_table("repository_import")
    deps_df = _read_table("repository_dependency")
    # Ecosystem filtering: the `ecosystem` column contains values like "PyPI", "NPM",
    # "CRAN", "github-actions", "docker", etc. By default all ecosystems are included in
    # the analysis. To exclude specific ecosystems that are not research software
    # (e.g. github-actions, docker), uncomment and modify the filter below:
    # DEPS_EXCLUDE_ECOSYSTEMS: set[str] = {"github-actions", "docker"}
    # deps_df = deps_df.filter(
    #     pl.col("ecosystem").is_null()
    #     | ~pl.col("ecosystem").is_in(list(DEPS_EXCLUDE_ECOSYSTEMS))
    # )
    raw_mentions_df = _read_table("document_software_mention")
    mentions_df = raw_mentions_df.filter(
        ~pl.col("software_name_normalized").is_in(list(MENTION_EXCLUDE_NORMALIZED))
    )

    log.info(
        f"Found {imports_df.height} imports, {deps_df.height} dependencies, "
        f"{mentions_df.height} mentions (after filtering {raw_mentions_df.height - mentions_df.height} "
        f"generic terms)"
    )

    # Build per-ID lookup dicts for fast access during pair iteration
    imports_by_repo = _build_lookup(imports_df, "repository_id")
    deps_by_repo = _build_lookup(deps_df, "repository_id")
    mentions_by_doc = _build_lookup(mentions_df, "document_id")
    log.info(
        f"Lookup dicts built: {len(imports_by_repo)} repos with imports, "
        f"{len(deps_by_repo)} repos with deps, "
        f"{len(mentions_by_doc)} docs with mentions"
    )

    # Coverage summary (printed before the main loop so the user can see processing state)
    log.info("Computing coverage summary...")
    _coverage_text_early = _print_coverage_summary(
        pairs, imports_by_repo, deps_by_repo, mentions_by_doc
    )
    print(_coverage_text_early)

    # Run pairwise analysis per pair
    results: list[dict] = []
    all_im_records: list[dict] = []
    all_id_records: list[dict] = []
    all_dm_records: list[dict] = []

    for i, row in enumerate(
        tqdm(
            pairs.iter_rows(named=True),
            total=pairs.height,
            desc="Running pairwise analysis",
        )
    ):
        doc_id: int = row["document_id"]
        repo_id: int = row["repository_id"]
        if i % 10_000 == 0:
            log.debug(
                f"[{i}/{pairs.height}] doc_id={doc_id} repo_id={repo_id} | "
                f"imports={len(imports_by_repo.get(repo_id, ([], []))[0])} "
                f"deps={len(deps_by_repo.get(repo_id, ([], []))[0])} "
                f"mentions={len(mentions_by_doc.get(doc_id, ([], []))[0])}"
            )

        import_names, import_norms = _dedup_by_normalized(
            *imports_by_repo.get(repo_id, ([], []))
        )
        dep_names, dep_norms = _dedup_by_normalized(*deps_by_repo.get(repo_id, ([], [])))
        mention_names, mention_norms = _dedup_by_normalized(
            *mentions_by_doc.get(doc_id, ([], []))
        )

        # IM: imports vs mentions
        im_records, im_stats = run_pairwise_analysis(
            import_names,
            import_norms,
            mention_names,
            mention_norms,
            "import",
            "mention",
            score_cutoff,
        )

        # ID: imports vs dependencies
        id_records, id_stats = run_pairwise_analysis(
            import_names,
            import_norms,
            dep_names,
            dep_norms,
            "import",
            "dependency",
            score_cutoff,
        )

        # DM: dependencies vs mentions
        dm_records, dm_stats = run_pairwise_analysis(
            dep_names,
            dep_norms,
            mention_names,
            mention_norms,
            "dependency",
            "mention",
            score_cutoff,
        )

        # Greedy variants for comparison
        _, g_im_stats = run_pairwise_analysis(
            import_names,
            import_norms,
            mention_names,
            mention_norms,
            "import",
            "mention",
            score_cutoff,
            method="greedy_max_first",
        )
        _, g_id_stats = run_pairwise_analysis(
            import_names,
            import_norms,
            dep_names,
            dep_norms,
            "import",
            "dependency",
            score_cutoff,
            method="greedy_max_first",
        )
        _, g_dm_stats = run_pairwise_analysis(
            dep_names,
            dep_norms,
            mention_names,
            mention_norms,
            "dependency",
            "mention",
            score_cutoff,
            method="greedy_max_first",
        )

        def _to_dicts(records: list[PairwiseSoftwareRecord], d_id: int) -> list[dict]:
            return [
                {
                    "document_id": d_id,
                    "source_a_name": r.source_a_name,
                    "source_b_name": r.source_b_name,
                    "normalized_name": r.normalized_name,
                    "match_score": r.match_score,
                    "status": r.status,
                }
                for r in records
            ]

        # CCR: Credit Coverage Ratio
        ccr, ccr_imports, ccr_deps = _compute_ccr_for_pair(im_records, dm_records)

        im_dicts = _to_dicts(im_records, doc_id)
        id_dicts = _to_dicts(id_records, doc_id)
        dm_dicts = _to_dicts(dm_records, doc_id)

        all_im_records.extend(im_dicts)
        all_id_records.extend(id_dicts)
        all_dm_records.extend(dm_dicts)

        results.append(
            {
                **row,
                # IM: imports vs mentions
                "im_n_imports": im_stats["n_source_a"],
                "im_n_mentions": im_stats["n_source_b"],
                "im_n_matched": im_stats["n_matched"],
                "im_n_import_only": im_stats["n_source_a_only"],
                "im_n_mention_only": im_stats["n_source_b_only"],
                "im_jaccard": im_stats["jaccard"],
                "im_avg_score": im_stats["avg_match_score"],
                "im_median_score": im_stats["median_match_score"],
                # ID: imports vs dependencies
                "id_n_imports": id_stats["n_source_a"],
                "id_n_deps": id_stats["n_source_b"],
                "id_n_matched": id_stats["n_matched"],
                "id_n_import_only": id_stats["n_source_a_only"],
                "id_n_dep_only": id_stats["n_source_b_only"],
                "id_jaccard": id_stats["jaccard"],
                "id_avg_score": id_stats["avg_match_score"],
                "id_median_score": id_stats["median_match_score"],
                # DM: dependencies vs mentions
                "dm_n_deps": dm_stats["n_source_a"],
                "dm_n_mentions": dm_stats["n_source_b"],
                "dm_n_matched": dm_stats["n_matched"],
                "dm_n_dep_only": dm_stats["n_source_a_only"],
                "dm_n_mention_only": dm_stats["n_source_b_only"],
                "dm_jaccard": dm_stats["jaccard"],
                "dm_avg_score": dm_stats["avg_match_score"],
                "dm_median_score": dm_stats["median_match_score"],
                # Greedy IM
                "g_im_n_matched": g_im_stats["n_matched"],
                "g_im_jaccard": g_im_stats["jaccard"],
                "g_im_avg_score": g_im_stats["avg_match_score"],
                "g_im_median_score": g_im_stats["median_match_score"],
                # Greedy ID
                "g_id_n_matched": g_id_stats["n_matched"],
                "g_id_jaccard": g_id_stats["jaccard"],
                "g_id_avg_score": g_id_stats["avg_match_score"],
                "g_id_median_score": g_id_stats["median_match_score"],
                # Greedy DM
                "g_dm_n_matched": g_dm_stats["n_matched"],
                "g_dm_jaccard": g_dm_stats["jaccard"],
                "g_dm_avg_score": g_dm_stats["avg_match_score"],
                "g_dm_median_score": g_dm_stats["median_match_score"],
                # Credit Coverage Ratio
                "ccr": ccr,
                "ccr_imports": ccr_imports,
                "ccr_deps": ccr_deps,
                # Nested software records
                "im_software_records": im_dicts,
                "id_software_records": id_dicts,
                "dm_software_records": dm_dicts,
            }
        )

    # Build results DataFrame and add publication year
    # infer_schema_length=None: scan all rows so nested match_score (None vs f64) is
    # correctly typed as Float64 rather than Null
    log.info("Building results DataFrame (infer_schema_length=None)...")
    try:
        results_df = pl.DataFrame(results, infer_schema_length=None).with_columns(
            pl.col("document_publication_date").dt.year().alias("document_publication_year")
        )
    except Exception as e:
        log.error(f"Failed to build results DataFrame: {e}")
        log.debug(f"First result dict keys: {list(results[0].keys()) if results else 'empty'}")
        if results:
            log.debug(
                f"First result dict sample: { {k: type(v).__name__ for k, v in results[0].items()} }"
            )
        raise
    log.info(f"Built results DataFrame with {results_df.height} rows")
    log.debug(f"Results DataFrame schema:\n{results_df.schema}")

    # Save to parquet at top level (drop nested record cols which are for in-memory use only)
    parquet_df = results_df.drop(
        "im_software_records", "id_software_records", "dm_software_records"
    )
    parquet_path = output_path / "rq3-results.parquet"
    parquet_df.write_parquet(parquet_path)
    log.info(f"Saved results to {parquet_path}")

    pair_repo_ids = set(pairs["repository_id"].to_list())
    pair_doc_ids = set(pairs["document_id"].to_list())

    # Coverage summary (top-level, shared across filter modes)
    log.info("Computing coverage summary...")
    coverage_text = _print_coverage_summary(
        pairs, imports_by_repo, deps_by_repo, mentions_by_doc
    )
    print(coverage_text)
    with open(output_path / "rq3-coverage-summary.txt", "w") as f:
        f.write(coverage_text)
    log.info("Saved rq3-coverage-summary.txt")

    # Processing-time bias check
    _check_processing_bias(pairs, imports_by_repo, deps_by_repo, output_path, top_n)

    # Gini coefficients + frequency distribution plot (top-level)
    gini_values = _plot_software_frequency_distributions(
        imports_df,
        deps_df,
        mentions_df,
        pair_repo_ids,
        pair_doc_ids,
        output_path,
    )
    print()
    print("Software Frequency Distribution — Gini Coefficients:")
    for source_label, gini in gini_values.items():
        print(f"  {source_label}: {gini:.4f}")

    # Per-filter-mode analysis
    filter_modes = [
        ("complete-cases", output_path / "complete-cases"),
        ("full-population", output_path / "full-population"),
    ]
    filename_map = {
        "Imports vs Mentions": "rq3-summary-stats-imports-vs-mentions.txt",
        "Imports vs Dependencies": "rq3-summary-stats-imports-vs-dependencies.txt",
        "Dependencies vs Mentions": "rq3-summary-stats-dependencies-vs-mentions.txt",
    }
    grouping_cols = [
        ("document_domain_name", "Domain"),
        ("document_field_name", "Field"),
        ("repository_primary_language", "Language"),
        ("dataset_source_name", "Dataset Source"),
        ("document_publication_year", "Publication Year"),
        ("country_code", "Country"),
        ("document_is_open_access", "Open Access"),
    ]

    for filter_mode, mode_path in filter_modes:
        mode_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Running analysis for filter mode: {filter_mode}")

        # Hidden infrastructure tables
        _write_hidden_infrastructure_tables(
            results_df,
            all_im_records,
            all_dm_records,
            filter_mode,
            mode_path,
            top_n=top_n * 5,
        )

        # Per-pair descriptive statistics
        _write_pair_level_stats(results_df, filter_mode, mode_path, top_n_fields=top_n * 2)

        # Summary statistics
        stats_by_comparison = _print_summary_stats(
            results_df,
            all_im_records,
            all_id_records,
            all_dm_records,
            top_n,
            filter_mode,
        )
        for comparison_label, stats_text in stats_by_comparison.items():
            print(stats_text)
            fname = filename_map[comparison_label]
            with open(mode_path / fname, "w") as f:
                f.write(stats_text)
            log.info(f"[{filter_mode}] Saved {fname}")

        # Visualizations
        _plot_jaccard_boxplot(results_df, mode_path, filter_mode)
        _plot_score_histograms(results_df, mode_path, filter_mode)
        _plot_jaccard_trend_by_year(results_df, mode_path, filter_mode)
        for col, col_label in grouping_cols:
            _plot_jaccard_by_group(results_df, col, col_label, mode_path, filter_mode)

        # Logistic regressions
        _run_logistic_regressions(
            results_df,
            all_im_records,
            all_dm_records,
            imports_df,
            deps_df,
            pairs,
            filter_mode,
            mode_path,
        )

        # Credit Coverage Ratio analysis
        _write_ccr_analysis(results_df, filter_mode, mode_path)

        # Temporal trends
        _write_temporal_trends(results_df, filter_mode, mode_path)
        _plot_temporal_trends(results_df, filter_mode, mode_path)

        # Software citation classes
        _write_citation_classes(
            all_im_records,
            all_dm_records,
            all_id_records,
            results_df,
            filter_mode,
            mode_path,
        )

        # Field-Weighted Mention/Import Prevalence
        _write_fwmp_fwip_analysis(
            all_im_records,
            results_df,
            pairs,
            filter_mode,
            mode_path,
        )

    log.info("Done.")


###############################################################################
# Annotation CSV Generation
###############################################################################

_YEAR_BIN_LABELS = {
    "pre-2016": (None, 2016),
    "2016-2020": (2016, 2021),
    "2021-2025": (2021, 2026),
    "2025+": (2026, None),
}


def _bin_year(year: int | None) -> str:
    if year is None:
        return "Unknown"
    for label, (lo, hi) in _YEAR_BIN_LABELS.items():
        if (lo is None or year >= lo) and (hi is None or year < hi):
            return label
    return "Unknown"


def _stratified_sample(
    pairs: pl.DataFrame,
    n: int,
    seed: int,
    top_n_fields: int = 9,
    top_n_languages: int = 5,
) -> pl.DataFrame:
    """Proportionally stratified sample of article-repository pairs."""
    df = pairs.with_columns(
        pl.col("document_publication_date").dt.year().alias("_year"),
    )

    # Determine top categories
    field_counts = df["document_field_name"].value_counts().sort("count", descending=True)
    top_fields = set(
        field_counts.head(top_n_fields)["document_field_name"].drop_nulls().to_list()
    )
    lang_counts = (
        df["repository_primary_language"].value_counts().sort("count", descending=True)
    )
    top_languages = set(
        lang_counts.head(top_n_languages)["repository_primary_language"].drop_nulls().to_list()
    )

    # Build stratum key
    df = df.with_columns(
        pl.col("document_field_name")
        .fill_null("Other")
        .map_elements(lambda f: f if f in top_fields else "Other", return_dtype=pl.Utf8)
        .alias("_field_bin"),
        pl.col("repository_primary_language")
        .fill_null("Other")
        .map_elements(lambda l: l if l in top_languages else "Other", return_dtype=pl.Utf8)
        .alias("_lang_bin"),
        pl.col("_year").map_elements(_bin_year, return_dtype=pl.Utf8).alias("_year_bin"),
    ).with_columns(
        (pl.col("_field_bin") + "|" + pl.col("_year_bin") + "|" + pl.col("_lang_bin")).alias(
            "_stratum"
        ),
    )

    # Proportional allocation with minimum 1 per stratum
    stratum_counts = df["_stratum"].value_counts().sort("count", descending=True)
    total = df.height
    strata = stratum_counts["_stratum"].to_list()
    counts = stratum_counts["count"].to_list()

    # Compute allocation
    allocations: dict[str, int] = {}
    remaining = n
    for s, c in zip(strata, counts, strict=False):
        alloc = max(1, round(n * c / total))
        allocations[s] = min(alloc, c)  # can't sample more than available
        remaining -= allocations[s]

    # Adjust if we over-allocated
    if remaining < 0:
        # Reduce allocations from the largest strata first
        for s in strata:
            if remaining >= 0:
                break
            reduction = min(allocations[s] - 1, -remaining)
            allocations[s] -= reduction
            remaining += reduction

    # Adjust if we under-allocated
    if remaining > 0:
        for s in strata:
            if remaining <= 0:
                break
            s_count = dict(zip(strata, counts, strict=False))[s]
            can_add = s_count - allocations[s]
            add = min(can_add, remaining)
            allocations[s] += add
            remaining -= add

    # Sample from each stratum
    sampled_frames: list[pl.DataFrame] = []
    for s, alloc in allocations.items():
        if alloc <= 0:
            continue
        stratum_df = df.filter(pl.col("_stratum") == s)
        sampled_frames.append(stratum_df.sample(n=alloc, seed=seed))

    sampled = pl.concat(sampled_frames)
    # Drop temporary columns
    sampled = sampled.drop("_year", "_field_bin", "_lang_bin", "_year_bin", "_stratum")
    log.info(f"Sampled {sampled.height} pairs across {len(allocations)} strata")
    return sampled


def _build_annotation_rows(
    pair_row: dict,
    matches: list[PairwiseAlignmentResult],
    source_a_names: list[str],
    source_a_norms: list[str],
    source_b_names: list[str],
    source_b_norms: list[str],
    source_a_label: str,
    source_b_label: str,
) -> list[dict]:
    """Build long-format annotation rows for one pair and one comparison type."""
    # Common pair metadata
    doi = pair_row.get("document_doi", "")
    meta = {
        "document_repository_link_id": pair_row.get("document_repository_link_id"),
        "document_id": pair_row["document_id"],
        "document_doi": doi,
        "document_url": f"https://doi.org/{doi}" if doi else "",
        "document_title": pair_row.get("document_title", ""),
        "document_publication_date": pair_row.get("document_publication_date"),
        "document_field": pair_row.get("document_field_name", ""),
        "document_domain": pair_row.get("document_domain_name", ""),
        "repository_id": pair_row["repository_id"],
        "repository_owner": pair_row.get("repository_owner", ""),
        "repository_name": pair_row.get("repository_name", ""),
        "repository_url": (
            f"https://github.com/{pair_row.get('repository_owner', '')}"
            f"/{pair_row.get('repository_name', '')}"
        ),
        "repository_programming_language": pair_row.get("repository_primary_language", ""),
    }

    # Column name prefixes based on source labels
    label_to_prefix = {
        "import": "imported",
        "dependency": "dependency",
        "mention": "mentioned",
    }
    a_prefix = label_to_prefix[source_a_label]
    b_prefix = label_to_prefix[source_b_label]

    rows: list[dict] = []

    # Track which items were matched
    matched_a_norms: set[str] = set()
    matched_b_norms: set[str] = set()
    norm_a_to_orig = dict(zip(source_a_norms, source_a_names, strict=False))
    norm_b_to_orig = dict(zip(source_b_norms, source_b_names, strict=False))

    for match in matches:
        matched_a_norms.add(match.item_one)
        matched_b_norms.add(match.item_two)
        rows.append(
            {
                **meta,
                f"{a_prefix}_software_name_raw": norm_a_to_orig.get(
                    match.item_one, match.item_one
                ),
                f"{a_prefix}_software_name_cleaned": match.normalized_item_one,
                f"{b_prefix}_software_name_raw": norm_b_to_orig.get(
                    match.item_two, match.item_two
                ),
                f"{b_prefix}_software_name_cleaned": match.normalized_item_two,
                "automatically_matched": True,
                "automatic_match_score": match.score,
                "annotation_match_correct": None,
                "annotation_software_type": None,
                "annotation_software_category": None,
                "annotation_reason_missing": None,
                "annotation_notes": None,
            }
        )

    # Source A only (unmatched)
    for orig, norm in zip(source_a_names, source_a_norms, strict=False):
        if norm not in matched_a_norms:
            rows.append(
                {
                    **meta,
                    f"{a_prefix}_software_name_raw": orig,
                    f"{a_prefix}_software_name_cleaned": norm,
                    f"{b_prefix}_software_name_raw": None,
                    f"{b_prefix}_software_name_cleaned": None,
                    "automatically_matched": False,
                    "automatic_match_score": None,
                    "annotation_match_correct": None,
                    "annotation_software_type": None,
                    "annotation_software_category": None,
                    "annotation_reason_missing": None,
                    "annotation_notes": None,
                }
            )

    # Source B only (unmatched)
    for orig, norm in zip(source_b_names, source_b_norms, strict=False):
        if norm not in matched_b_norms:
            rows.append(
                {
                    **meta,
                    f"{a_prefix}_software_name_raw": None,
                    f"{a_prefix}_software_name_cleaned": None,
                    f"{b_prefix}_software_name_raw": orig,
                    f"{b_prefix}_software_name_cleaned": norm,
                    "automatically_matched": False,
                    "automatic_match_score": None,
                    "annotation_match_correct": None,
                    "annotation_software_type": None,
                    "annotation_software_category": None,
                    "annotation_reason_missing": None,
                    "annotation_notes": None,
                }
            )

    return rows


@app.command()
def generate_annotation_csvs(
    n_samples: int = 100,
    random_seed: int = 42,
    score_cutoff: float = SCORE_CUTOFF,
    output_dir: str = str(THIS_DIR / "rq3-annotation"),
    debug: bool = False,
) -> None:
    """Generate stratified-sample annotation CSVs for qualitative analysis."""
    setup_logger(debug)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pairs and software tables
    log.info("Loading pairs...")
    pairs = load_pairs()
    log.info(f"Loaded {pairs.height} pairs")

    log.info("Loading software tables...")
    imports_df = _read_table("repository_import")
    deps_df = _read_table("repository_dependency")
    raw_mentions_df = _read_table("document_software_mention")
    mentions_df = raw_mentions_df.filter(
        ~pl.col("software_name_normalized").is_in(list(MENTION_EXCLUDE_NORMALIZED))
    )

    imports_by_repo = _build_lookup(imports_df, "repository_id")
    deps_by_repo = _build_lookup(deps_df, "repository_id")
    mentions_by_doc = _build_lookup(mentions_df, "document_id")

    # Stratified sample
    sampled = _stratified_sample(pairs, n_samples, random_seed)
    log.info(f"Sampled {sampled.height} pairs for annotation")

    # Run pairwise alignment and build annotation rows
    im_rows: list[dict] = []
    id_rows: list[dict] = []
    dm_rows: list[dict] = []

    for row in tqdm(
        sampled.iter_rows(named=True),
        total=sampled.height,
        desc="Building annotation rows",
    ):
        doc_id: int = row["document_id"]
        repo_id: int = row["repository_id"]

        import_names, import_norms = _dedup_by_normalized(
            *imports_by_repo.get(repo_id, ([], []))
        )
        dep_names, dep_norms = _dedup_by_normalized(*deps_by_repo.get(repo_id, ([], [])))
        mention_names, mention_norms = _dedup_by_normalized(
            *mentions_by_doc.get(doc_id, ([], []))
        )

        # IM: imports vs mentions
        im_matches = align_software_names(
            import_norms, mention_norms, "import", "mention", score_cutoff
        )
        im_rows.extend(
            _build_annotation_rows(
                row,
                im_matches,
                import_names,
                import_norms,
                mention_names,
                mention_norms,
                "import",
                "mention",
            )
        )

        # ID: imports vs dependencies
        id_matches = align_software_names(
            import_norms, dep_norms, "import", "dependency", score_cutoff
        )
        id_rows.extend(
            _build_annotation_rows(
                row,
                id_matches,
                import_names,
                import_norms,
                dep_names,
                dep_norms,
                "import",
                "dependency",
            )
        )

        # DM: dependencies vs mentions
        dm_matches = align_software_names(
            dep_norms, mention_norms, "dependency", "mention", score_cutoff
        )
        dm_rows.extend(
            _build_annotation_rows(
                row,
                dm_matches,
                dep_names,
                dep_norms,
                mention_names,
                mention_norms,
                "dependency",
                "mention",
            )
        )

    # Write CSVs
    for label, rows in [
        ("imports-vs-mentions", im_rows),
        ("imports-vs-dependencies", id_rows),
        ("dependencies-vs-mentions", dm_rows),
    ]:
        csv_path = output_path / f"annotation-{label}.csv"
        if rows:
            pl.DataFrame(rows, infer_schema_length=None).write_csv(csv_path)
        else:
            # Write empty CSV with just a header (no data rows)
            pl.DataFrame(rows).write_csv(csv_path)
        log.info(f"Wrote {len(rows)} rows to {csv_path}")

    # Write sample metadata
    sampled_with_year = sampled.with_columns(
        pl.col("document_publication_date").dt.year().alias("_year"),
    )
    strata_summary: dict[str, dict[str, int]] = {
        "field_distribution": dict(
            sampled["document_field_name"]
            .fill_null("Unknown")
            .value_counts()
            .sort("count", descending=True)
            .iter_rows()
        ),
        "year_distribution": dict(
            sampled_with_year["_year"]
            .map_elements(_bin_year, return_dtype=pl.Utf8)
            .value_counts()
            .sort("count", descending=True)
            .iter_rows()
        ),
        "language_distribution": dict(
            sampled["repository_primary_language"]
            .fill_null("Unknown")
            .value_counts()
            .sort("count", descending=True)
            .iter_rows()
        ),
    }
    metadata = {
        "n_samples": sampled.height,
        "random_seed": random_seed,
        "score_cutoff": score_cutoff,
        "date_generated": date.today().isoformat(),
        "total_pairs_available": pairs.height,
        "strata": strata_summary,
        "csv_row_counts": {
            "imports_vs_mentions": len(im_rows),
            "imports_vs_dependencies": len(id_rows),
            "dependencies_vs_mentions": len(dm_rows),
        },
    }
    meta_path = output_path / "annotation-sample-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Wrote metadata to {meta_path}")

    print(f"\nAnnotation CSVs generated in {output_path}/")
    print(f"  Pairs sampled: {sampled.height}")
    print(f"  IM rows: {len(im_rows)}")
    print(f"  ID rows: {len(id_rows)}")
    print(f"  DM rows: {len(dm_rows)}")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
