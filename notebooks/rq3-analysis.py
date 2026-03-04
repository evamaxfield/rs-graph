#!/usr/bin/env python

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import connectorx  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import typer
from tqdm import tqdm

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.db import constants as db_constants
from rs_graph.utils.identifier_normalization import (
    normalize_doi_col,
    prep_name_for_printing,
)
from rs_graph.utils.software_alignment import align_software_names

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
                pl.col("publication_date").alias("document_publication_date"),
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
        (pl.col("document_repository_link_confidence") >= 0.995)
        | (pl.col("document_repository_link_confidence").is_null())
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
    )

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
) -> None:
    """Print full descriptive statistics for a numeric column."""
    vals = df[col].drop_nulls()
    if vals.len() == 0:
        print(f"  {label}: no data")
        return

    mean = vals.mean()
    std = vals.std()
    median = vals.median()
    p10 = vals.quantile(0.10)
    p25 = vals.quantile(0.25)
    p75 = vals.quantile(0.75)
    p90 = vals.quantile(0.90)
    min_val = vals.min()
    max_val = vals.max()

    print(
        f"  {label}: mean={mean:.3f}, std={std:.3f}, "
        f"min={min_val:.3f}, p10={p10:.3f}, p25={p25:.3f}, "
        f"median={median:.3f}, p75={p75:.3f}, p90={p90:.3f}, max={max_val:.3f}"
    )


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
) -> None:
    """Print per-group descriptive stats for jaccard and n_matched."""
    print()
    if top_n_filter is not None:
        print(f"  Breakdown by {group_label} (top {top_n_filter} most populous):")
    else:
        print(f"  Breakdown by {group_label}:")
    print(f"  {'-' * 60}")

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

    # Pre-build doc_id → records lookup for efficient per-group filtering
    doc_id_to_records: dict[int, list[dict]] = {}
    if show_top_n:
        for rec in all_records:  # type: ignore[union-attr]
            did = rec["document_id"]
            if did not in doc_id_to_records:
                doc_id_to_records[did] = []
            doc_id_to_records[did].append(rec)

    for row in grouped.iter_rows(named=True):
        group_val = row[group_col] if row[group_col] is not None else "Unknown"
        print(
            f"    [{group_val}]: n={row['n_pairs']}, "
            f"jaccard mean={row['mean_jaccard'] or 0:.3f}, "
            f"std={row['std_jaccard'] or 0:.3f}, "
            f"median={row['median_jaccard'] or 0:.3f}, "
            f"p25={row['p25_jaccard'] or 0:.3f}, "
            f"p75={row['p75_jaccard'] or 0:.3f}, "
            f"mean_n_matched={row['mean_n_matched'] or 0:.1f}"
        )

        if show_top_n:
            gv = row[group_col]
            if gv is None:
                group_df = results_df.filter(pl.col(group_col).is_null())
            else:
                group_df = results_df.filter(pl.col(group_col) == gv)
            group_doc_ids = set(group_df["document_id"].to_list())
            group_recs: list[dict] = []
            for did in group_doc_ids:
                group_recs.extend(doc_id_to_records.get(did, []))

            top_matched = _get_top_items_by_status(group_recs, "matched", "source_a_name", top_n)
            if top_matched:
                print(f"      Top {top_n} matched: {', '.join(top_matched)}")

            top_a = _get_top_items_by_status(
                group_recs, ["matched", "source_a_only"], "source_a_name", top_n
            )
            if top_a:
                print(f"      Top {top_n} {a_label}: {', '.join(top_a)}")

            top_b = _get_top_items_by_status(
                group_recs, ["matched", "source_b_only"], "source_b_name", top_n
            )
            if top_b:
                print(f"      Top {top_n} {b_label}: {', '.join(top_b)}")

            top_a_unmatched = _get_top_items_by_status(
                group_recs, "source_a_only", "source_a_name", top_n
            )
            if top_a_unmatched:
                print(f"      Top {top_n} unmatched-{a_label}: {', '.join(top_a_unmatched)}")

            top_b_unmatched = _get_top_items_by_status(
                group_recs, "source_b_only", "source_b_name", top_n
            )
            if top_b_unmatched:
                print(f"      Top {top_n} unmatched-{b_label}: {', '.join(top_b_unmatched)}")


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
# Summary Statistics
###############################################################################


def _print_summary_stats(
    results_df: pl.DataFrame,
    all_im_records: list[dict],
    all_id_records: list[dict],
    all_dm_records: list[dict],
    top_n: int,
) -> None:
    """Print full descriptive statistics for all three pairwise comparisons."""
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

    # TODO: Currently using AND logic (both sources must have data) for testing purposes
    # while imports/dependencies processing is still incomplete. Once the full dataset
    # has been processed for all source types, switch to OR logic so that one-sided pairs
    # (e.g. a repo with imports but a paper with no mentions) are included for a complete
    # picture. To switch: change each & to | in the filter expressions below.
    filter_exprs = {
        "im": (pl.col("im_n_imports") > 0) & (pl.col("im_n_mentions") > 0),
        "id": (pl.col("id_n_imports") > 0) & (pl.col("id_n_deps") > 0),
        "dm": (pl.col("dm_n_deps") > 0) & (pl.col("dm_n_mentions") > 0),
    }

    for label, prefix, all_records, a_label, b_label in comparisons:
        comparison_df = results_df.filter(filter_exprs[prefix])
        print()
        print("=" * 70)
        print(f"  {label}  (N pairs with data for both sources: {comparison_df.height})")
        print("=" * 70)

        _print_descriptive_stats(comparison_df, f"{prefix}_jaccard", "Jaccard")
        _print_descriptive_stats(comparison_df, f"{prefix}_n_matched", "N Matched")
        non_zero_score = comparison_df.filter(pl.col(f"{prefix}_avg_score") > 0)
        _print_descriptive_stats(non_zero_score, f"{prefix}_avg_score", "Avg Match Score")

        top_matched = _get_top_items_by_status(all_records, "matched", "source_a_name", top_n)
        if top_matched:
            print(f"  Top {top_n} matched: {', '.join(top_matched)}")

        top_a = _get_top_items_by_status(
            all_records, ["matched", "source_a_only"], "source_a_name", top_n
        )
        if top_a:
            print(f"  Top {top_n} {a_label}: {', '.join(top_a)}")

        top_b = _get_top_items_by_status(
            all_records, ["matched", "source_b_only"], "source_b_name", top_n
        )
        if top_b:
            print(f"  Top {top_n} {b_label}: {', '.join(top_b)}")

        top_a_unmatched = _get_top_items_by_status(
            all_records, "source_a_only", "source_a_name", top_n
        )
        if top_a_unmatched:
            print(f"  Top {top_n} unmatched-{a_label}: {', '.join(top_a_unmatched)}")

        top_b_unmatched = _get_top_items_by_status(
            all_records, "source_b_only", "source_b_name", top_n
        )
        if top_b_unmatched:
            print(f"  Top {top_n} unmatched-{b_label}: {', '.join(top_b_unmatched)}")

        for col, col_label, tnf in grouping_cols:
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


###############################################################################
# Visualizations
###############################################################################


def _plot_jaccard_boxplot(
    results_df: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Box plots of per-pair Jaccard scores for all three comparisons."""
    comparison_labels = {
        "im_jaccard": "Imports vs Mentions",
        "id_jaccard": "Imports vs Dependencies",
        "dm_jaccard": "Deps vs Mentions",
    }
    data: list[dict] = []
    for col, label in comparison_labels.items():
        for v in results_df[col].drop_nulls().to_list():
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
    top_n_groups: int = 10,
) -> None:
    top_groups = (
        results_df.group_by(group_col)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_n_groups)[group_col]
        .to_list()
    )
    filtered = results_df.filter(pl.col(group_col).is_in(top_groups))

    comparison_cols = {
        "im_jaccard": "Imports vs Mentions",
        "id_jaccard": "Imports vs Dependencies",
        "dm_jaccard": "Deps vs Mentions",
    }

    data: list[dict] = []
    for col, label in comparison_cols.items():
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
) -> None:
    """Histograms of average pairwise match scores for each comparison."""
    score_cols = {
        "im_avg_score": "Imports vs Mentions",
        "id_avg_score": "Imports vs Dependencies",
        "dm_avg_score": "Deps vs Mentions",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, label) in zip(axes, score_cols.items(), strict=False):
        vals = results_df.filter(pl.col(col) > 0)[col].drop_nulls().to_list()
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
    def _build_lookup(
        df: pl.DataFrame,
        id_col: str,
    ) -> dict[int, tuple[list[str], list[str]]]:
        lut: dict[int, tuple[list[str], list[str]]] = {}
        for row in df.iter_rows(named=True):
            key = row[id_col]
            if key not in lut:
                lut[key] = ([], [])
            lut[key][0].append(row["software_name"])
            lut[key][1].append(row["software_name_normalized"])
        return lut

    imports_by_repo = _build_lookup(imports_df, "repository_id")
    deps_by_repo = _build_lookup(deps_df, "repository_id")
    mentions_by_doc = _build_lookup(mentions_df, "document_id")
    log.info(
        f"Lookup dicts built: {len(imports_by_repo)} repos with imports, "
        f"{len(deps_by_repo)} repos with deps, "
        f"{len(mentions_by_doc)} docs with mentions"
    )

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

    # Save to parquet (drop nested record cols which are for in-memory use only)
    parquet_df = results_df.drop(
        "im_software_records", "id_software_records", "dm_software_records"
    )
    parquet_path = output_path / "rq3-results.parquet"
    parquet_df.write_parquet(parquet_path)
    log.info(f"Saved results to {parquet_path}")

    # Summary statistics
    log.info("Printing summary stats...")
    _print_summary_stats(
        results_df,
        all_im_records,
        all_id_records,
        all_dm_records,
        top_n,
    )
    log.info("Summary stats complete.")

    # Gini coefficients + frequency distribution plot
    pair_repo_ids = set(pairs["repository_id"].to_list())
    pair_doc_ids = set(pairs["document_id"].to_list())
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

    # Remaining visualizations
    _plot_jaccard_boxplot(results_df, output_path)
    _plot_score_histograms(results_df, output_path)

    grouping_cols = [
        ("document_domain_name", "Domain"),
        ("document_field_name", "Field"),
        ("repository_primary_language", "Language"),
        ("dataset_source_name", "Dataset Source"),
        ("document_publication_year", "Publication Year"),
        ("country_code", "Country"),
        ("document_is_open_access", "Open Access"),
    ]
    for col, col_label in grouping_cols:
        _plot_jaccard_by_group(results_df, col, col_label, output_path)

    log.info("Done.")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
