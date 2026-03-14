import json
import logging
import random
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import colormaps as cmaps
import connectorx  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rustworkx as rx
import seaborn as sns
import typer
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from scipy.stats.contingency import association
from tqdm import tqdm

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.db import constants as db_constants

###############################################################################
# Constants
###############################################################################

PALETTE = cmaps.bold._colors.tolist()

SHARED_SOURCES = frozenset({"pwc", "plos", "joss", "softwarex"})
MINED_SOURCES = frozenset({"snowball-sampling-discovery"})

NUMERIC_FEATURES = [
    "document_fwci_log1p",
    "repository_fwsi_log1p",
    "document_cited_by_count_log1p",
    "repository_stargazers_count_log1p",
    "repository_commits_count_log1p",
    "repository_size_kb_log1p",
    "repository_n_contributors",
    "repository_n_files",
    "days_from_repo_creation_to_publication",
    "days_from_publication_to_last_push",
    "document_publication_year",
    "document_n_authors",
    "repository_commit_duration_days",
    "document_n_unique_author_countries",
    "document_author_country_entropy",
]

CATEGORICAL_FEATURES = [
    "document_is_open_access",
    "document_field_name",
    "document_type",
    "repository_primary_language",
]

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "document_fwci_log1p": "Article FWCI (log)",
    "repository_fwsi_log1p": "Repo FWSI (log)",
    "document_cited_by_count_log1p": "Citations (log)",
    "repository_stargazers_count_log1p": "Repo Stars (log)",
    "repository_commits_count_log1p": "Repo Commits (log)",
    "repository_size_kb_log1p": "Repo Size KB (log)",
    "repository_n_contributors": "Repo Contributors",
    "repository_n_files": "Repo Files",
    "days_from_repo_creation_to_publication": "Days Repo Created to Publication",
    "days_from_publication_to_last_push": "Days Publication to Last Push",
    "document_publication_year": "Publication Year",
    "document_n_authors": "Number of Authors",
    "repository_commit_duration_days": "Repo Commit Duration (days)",
    "document_is_open_access": "Open Access",
    "document_field_name": "Research Field",
    "document_type": "Document Type",
    "repository_primary_language": "Primary Language",
    "document_n_unique_author_countries": "Unique Author Countries",
    "document_author_country_entropy": "Author Country Entropy",
}

###############################################################################
# Logger & App
###############################################################################

log = logging.getLogger(__name__)
app = typer.Typer()

###############################################################################
# Shared helpers (duplicated from rq1-analysis.py)
###############################################################################


def _read_table(table: str) -> pl.DataFrame:
    """Read a table from the v2 database."""
    log.info(f"Reading table: {table}")
    return pl.read_database_uri(
        f"SELECT * FROM {table}",
        f"sqlite:///{db_constants.V2_DATABASE_PATHS.dev}",
    )


def _read_sql(query: str) -> pl.DataFrame:
    """Execute a raw SQL query against the v2 database."""
    log.info(f"Running query: {query[:80].strip()}...")
    return pl.read_database_uri(
        query,
        f"sqlite:///{db_constants.V2_DATABASE_PATHS.dev}",
    )


def _add_top_n_other_column(
    df: pl.DataFrame,
    source_col: str,
    n: int,
) -> tuple[pl.DataFrame, list[str], str]:
    """Get top-N values of *source_col*, add a new column mapping the rest to 'Other'."""
    top_values = (
        df.filter(pl.col(source_col).is_not_null())[source_col]
        .value_counts(sort=True)
        .head(n)[source_col]
        .to_list()
    )
    new_col = f"{source_col}_top_n_plus_other"
    df = df.with_columns(
        pl.when(pl.col(source_col).is_in(top_values))
        .then(pl.col(source_col))
        .otherwise(pl.lit("Other"))
        .alias(new_col)
    )
    return df, top_values, new_col


def _filter_finite(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Keep rows where every column in *cols* is not-null, not-NaN, and finite."""
    exprs = []
    for c in cols:
        exprs.extend(
            [
                pl.col(c).is_not_null(),
                pl.col(c).is_not_nan(),
                pl.col(c).is_finite(),
            ]
        )
    return df.filter(*exprs)


def _build_coauthorship_graph(
    doc_contribs: pl.DataFrame,
) -> tuple[rx.PyGraph, dict[int, int], dict[int, int]]:
    """Build a simple (deduplicated) co-authorship graph from document contributors."""
    graph: rx.PyGraph = rx.PyGraph()
    node_to_idx: dict[int, int] = {}
    idx_to_node: dict[int, int] = {}
    edge_seen: set[tuple[int, int]] = set()

    if doc_contribs.height == 0:
        return graph, node_to_idx, idx_to_node

    for _, group in tqdm(
        doc_contribs.group_by("document_id"),
        total=doc_contribs["document_id"].n_unique(),
        desc="Building co-authorship network",
    ):
        researcher_ids = sorted(set(group["researcher_id"].to_list()))

        for rid in researcher_ids:
            if rid not in node_to_idx:
                idx = graph.add_node(rid)
                node_to_idx[rid] = idx
                idx_to_node[idx] = rid

        for rid_a, rid_b in combinations(researcher_ids, 2):
            idx_a = node_to_idx[rid_a]
            idx_b = node_to_idx[rid_b]
            edge = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)

            if edge not in edge_seen:
                graph.add_edge(edge[0], edge[1], 1)
                edge_seen.add(edge)

    return graph, node_to_idx, idx_to_node


def _get_author_developer_pairs_connected_to_pairs(
    pair_links: pl.DataFrame,
) -> pl.DataFrame:
    """Return matched author-developer pairs connected to provided doc-repo pairs."""
    required_cols = {"document_id", "repository_id"}
    if not required_cols.issubset(pair_links.columns):
        raise ValueError("pair_links must include document_id and repository_id")

    pair_cols = [
        c for c in ["document_id", "repository_id", "iteration"] if c in pair_links.columns
    ]

    pair_links = pair_links.select(pair_cols).unique()

    doc_contribs = (
        _read_table("document_contributor").select("document_id", "researcher_id").unique()
    )
    repo_contribs = (
        _read_table("repository_contributor")
        .select("repository_id", "developer_account_id")
        .unique()
    )
    researcher_dev_links = (
        _read_table("researcher_developer_account_link")
        .filter(
            pl.col("predictive_model_confidence").is_null()
            | (pl.col("predictive_model_confidence") >= 0.97)
        )
        .select("researcher_id", "developer_account_id")
        .unique()
    )
    log.debug(
        "Researcher-developer links after confidence filter (>= 0.97): %d",
        researcher_dev_links.height,
    )

    connected_pairs = (
        pair_links.join(doc_contribs, on="document_id", how="inner")
        .join(researcher_dev_links, on="researcher_id", how="inner")
        .join(repo_contribs, on=["repository_id", "developer_account_id"], how="inner")
        .select([*pair_cols, "researcher_id", "developer_account_id"])
        .unique()
    )

    return connected_pairs


###############################################################################
# Data loading
###############################################################################


def load_rq2_pairs(
    sample_size: int | None = None,
    doc_repo_confidence_threshold: float | None = None,
) -> pl.DataFrame:
    """Load document-repository pairs with provenance labels for RQ2."""
    log.debug("Reading database tables...")

    dataset_sources = _read_table("dataset_source")
    docs = _read_table("document")
    repos = _read_table("repository")
    pairs = _read_table("document_repository_link")
    doc_topics = _read_sql("""
        SELECT document_id, topic_id
        FROM document_topic
        WHERE (document_id, score) IN (
            SELECT document_id, MAX(score)
            FROM document_topic
            GROUP BY document_id
        )
    """)
    topics = _read_table("topic")

    log.info("Raw document_repository_link rows: %d", pairs.height)

    # Drop predicted doc-repo pairs below confidence threshold
    if doc_repo_confidence_threshold is not None:
        pairs = pairs.filter(
            pl.col("predictive_model_confidence").is_null()
            | (pl.col("predictive_model_confidence") >= doc_repo_confidence_threshold)
        )
        log.info(
            "Pairs after confidence filter (>= %s): %d",
            doc_repo_confidence_threshold,
            pairs.height,
        )
    else:
        log.info("Pairs (no confidence filter): %d", pairs.height)

    # Join dataset source name for provenance labeling
    pairs = pairs.join(
        dataset_sources.select(
            pl.col("id").alias("dataset_source_id"),
            pl.col("name").alias("dataset_source_name"),
        ),
        on="dataset_source_id",
        how="left",
    )

    # Exclude softcite_2025 rows entirely (neither truly shared nor mined)
    pairs = pairs.filter(~pl.col("dataset_source_name").str.contains("softcite_2025"))
    log.info("Pairs after excluding softcite_2025: %d", pairs.height)

    # Assign pair_source_label per link row
    pairs = pairs.with_columns(
        pl.when(pl.col("dataset_source_name").is_in(list(SHARED_SOURCES)))
        .then(pl.lit("shared"))
        .when(pl.col("dataset_source_name").is_in(list(MINED_SOURCES)))
        .then(pl.lit("mined"))
        .otherwise(pl.lit("unknown"))
        .alias("pair_source_label_raw")
    )

    log.info(
        "Link-level source counts:\n%s",
        pairs["pair_source_label_raw"].value_counts(sort=True),
    )

    # For pairs that exist in multiple sources, choose "shared" if any shared row exists.
    # First, determine the canonical label per (document_id, repository_id).
    pair_labels = (
        pairs.group_by(["document_id", "repository_id"])
        .agg(
            pl.col("pair_source_label_raw").unique().alias("source_labels"),
            pl.col("dataset_source_name").unique().alias("dataset_source_names"),
            pl.col("iteration").min().alias("iteration"),
        )
        .with_columns(
            pl.when(pl.col("source_labels").list.contains("shared"))
            .then(pl.lit("shared"))
            .otherwise(pl.lit("mined"))
            .alias("pair_source_label"),
            pl.col("dataset_source_names").list.join(", ").alias("dataset_source_names_str"),
        )
        .drop("source_labels", "dataset_source_names")
    )

    log.info(
        "Unique pair-level source counts:\n%s",
        pair_labels["pair_source_label"].value_counts(sort=True),
    )

    # Deduplicate to unique (document_id, repository_id) pairs
    pairs_deduped = pair_labels.select(
        "document_id",
        "repository_id",
        "pair_source_label",
        "dataset_source_names_str",
        "iteration",
    )

    # Apply canonical-pair filter: keep only doc/repo IDs that appear exactly once
    pairs_deduped = pairs_deduped.unique(
        subset="document_id",
        keep="none",
    ).unique(
        subset="repository_id",
        keep="none",
    )
    log.info("Unique canonical pairs: %d", pairs_deduped.height)
    log.info(
        "Canonical pair source counts:\n%s",
        pairs_deduped["pair_source_label"].value_counts(sort=True),
    )

    if sample_size is not None:
        log.debug("Sampling %d pairs...", sample_size)
        sampled_doc_ids = (
            pairs_deduped.select("document_id")
            .unique()
            .sample(n=min(sample_size, pairs_deduped.height), seed=42)
        )
        pairs_deduped = pairs_deduped.filter(
            pl.col("document_id").is_in(sampled_doc_ids["document_id"])
        )

    # --- Join article and repository features (same as RQ1) ---

    doc_author_country_rows = _read_sql("""
        SELECT
            dc.document_id,
            dc.researcher_id,
            COALESCE(i.country_code, 'Unknown') AS country_code
        FROM document_contributor dc
        LEFT JOIN document_contributor_institution dci
            ON dc.id = dci.document_contributor_id
        LEFT JOIN institution i
            ON dci.institution_id = i.id
    """)

    doc_author_country_entropy = (
        doc_author_country_rows.group_by(["document_id", "country_code"])
        .agg(pl.len().alias("author_count"))
        .with_columns(
            (pl.col("author_count") / pl.col("author_count").sum().over("document_id")).alias(
                "country_share"
            )
        )
        .with_columns(
            (-(pl.col("country_share") * pl.col("country_share").log())).alias(
                "entropy_component"
            )
        )
        .group_by("document_id")
        .agg(
            pl.col("entropy_component").sum().alias("document_author_country_entropy"),
            pl.col("country_code").n_unique().alias("document_n_unique_author_countries"),
        )
    )

    doc_author_countries = (
        doc_author_country_rows.group_by("document_id")
        .agg(pl.len().alias("document_n_authors"))
        .join(doc_author_country_entropy, on="document_id", how="left")
        .with_columns(
            pl.col("document_author_country_entropy").fill_null(0.0),
            pl.col("document_n_unique_author_countries").fill_null(0),
        )
    )

    repo_contribs = _read_sql("""
        SELECT
            repository_id,
            COUNT(*) AS repository_n_contributors
        FROM repository_contributor
        GROUP BY repository_id
    """)

    repo_file_counts = _read_sql("""
        SELECT
            repository_id,
            COUNT(*) AS repository_n_files
        FROM repository_file
        WHERE tree_type = 'blob'
        GROUP BY repository_id
    """)

    result = (
        pairs_deduped.join(
            docs.select(
                pl.col("id").alias("document_id"),
                pl.col("title").alias("document_title"),
                pl.col("doi").alias("document_doi"),
                pl.col("cited_by_count").alias("document_cited_by_count"),
                pl.col("fwci").alias("document_fwci"),
                pl.col("is_open_access").cast(pl.Utf8).alias("document_is_open_access"),
                pl.col("publication_date").alias("document_publication_date"),
                pl.col("document_type").alias("document_type"),
            ),
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
                pl.col("size_kb").alias("repository_size_kb"),
                pl.col("forks_count").alias("repository_forks_count"),
                pl.col("creation_datetime").alias("repository_creation_datetime"),
                pl.col("last_pushed_datetime").alias("repository_last_pushed_datetime"),
            ),
            on="repository_id",
            how="left",
        )
        .join(
            doc_topics,
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
        .join(doc_author_countries, on="document_id", how="left")
        .join(repo_contribs, on="repository_id", how="left")
        .join(repo_file_counts, on="repository_id", how="left")
        .with_columns(
            pl.col("document_publication_date").dt.year().alias("document_publication_year"),
            (
                (
                    pl.col("repository_last_pushed_datetime")
                    - pl.col("repository_creation_datetime")
                ).dt.total_days()
            ).alias("repository_commit_duration_days"),
            (
                (
                    pl.col("document_publication_date") - pl.col("repository_creation_datetime")
                ).dt.total_days()
            ).alias("days_from_repo_creation_to_publication"),
            (
                (
                    pl.col("repository_last_pushed_datetime")
                    - pl.col("document_publication_date")
                ).dt.total_days()
            ).alias("days_from_publication_to_last_push"),
        )
    )

    # Compute FWSI (same as RQ1)
    field_year_type_expected_stars = result.group_by(
        [
            "document_field_name",
            "document_publication_year",
            "document_type",
        ]
    ).agg(
        pl.col("repository_stargazers_count").mean().alias("expected_stars"),
    )

    result = result.join(
        field_year_type_expected_stars,
        on=["document_field_name", "document_publication_year", "document_type"],
        how="left",
    )

    result = result.with_columns(
        pl.when((pl.col("expected_stars").is_not_null()) & (pl.col("expected_stars") > 0))
        .then(pl.col("repository_stargazers_count") / pl.col("expected_stars"))
        .otherwise(pl.lit(None))
        .alias("repository_fwsi")
    ).drop("expected_stars")

    # Add log transforms
    log_cols = {
        "document_fwci": "document_fwci_log1p",
        "repository_fwsi": "repository_fwsi_log1p",
        "document_cited_by_count": "document_cited_by_count_log1p",
        "repository_stargazers_count": "repository_stargazers_count_log1p",
        "repository_commits_count": "repository_commits_count_log1p",
        "repository_size_kb": "repository_size_kb_log1p",
    }
    for src, dst in log_cols.items():
        result = result.with_columns(
            (pl.lit(1) + pl.col(src).cast(pl.Float64)).log().alias(dst)
        )

    log.info("Final RQ2 pairs: %d", result.height)

    return result


###############################################################################
# Workstream 2: Univariate Feature Comparison
###############################################################################


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


def run_feature_comparison(  # noqa: C901
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Univariate statistical tests comparing shared vs mined pairs."""
    log.info("Starting univariate feature comparison...")

    shared = pairs.filter(pl.col("pair_source_label") == "shared")
    mined = pairs.filter(pl.col("pair_source_label") == "mined")
    n_shared = shared.height
    n_mined = mined.height
    log.info("Class counts — shared: %d, mined: %d", n_shared, n_mined)

    # ── A. Numeric features: Mann-Whitney U tests ──
    numeric_cols = [c for c in NUMERIC_FEATURES if c in pairs.columns]
    n_numeric_tests = len(numeric_cols)
    numeric_rows: list[dict[str, str | int | float]] = []

    for col_name in tqdm(
        numeric_cols,
        desc="Comparing numeric features",
    ):
        shared_vals = shared[col_name].drop_nulls().drop_nans().to_numpy()
        mined_vals = mined[col_name].drop_nulls().drop_nans().to_numpy()

        if len(shared_vals) < 2 or len(mined_vals) < 2:
            log.warning("Skipping %s: too few non-null values.", col_name)
            continue

        u_stat, p_val = mannwhitneyu(shared_vals, mined_vals, alternative="two-sided")
        n1, n2 = len(shared_vals), len(mined_vals)
        r = 1 - (2 * u_stat) / (n1 * n2)
        p_bonf = min(p_val * n_numeric_tests, 1.0)

        numeric_rows.append(
            {
                "feature": col_name,
                "n_shared": len(shared_vals),
                "n_mined": len(mined_vals),
                "median_shared": float(np.median(shared_vals)),
                "median_mined": float(np.median(mined_vals)),
                "mean_shared": float(np.mean(shared_vals)),
                "mean_mined": float(np.mean(mined_vals)),
                "u_statistic": float(u_stat),
                "p_value": float(p_val),
                "p_value_bonferroni": float(p_bonf),
                "rank_biserial_r": round(r, 4),
                "direction": "higher_in_shared" if r > 0 else "higher_in_mined",
                "effect_magnitude": _effect_magnitude(r),
            }
        )

    numeric_df = pl.DataFrame(numeric_rows).sort("p_value")
    numeric_df.write_csv(results_dir / "numeric-feature-tests.csv")
    log.info("Saved numeric-feature-tests.csv (%d features)", numeric_df.height)

    # ── B. Categorical features: Chi-square tests ──
    n_cat_tests = len(CATEGORICAL_FEATURES)
    cat_rows: list[dict[str, str | int | float]] = []

    for cat_col in CATEGORICAL_FEATURES:
        if cat_col not in pairs.columns:
            continue

        work_df = pairs.with_columns(pl.col(cat_col).fill_null("Unknown"))
        work_df, _, top_col = _add_top_n_other_column(work_df, cat_col, top_n)

        # Build contingency table
        crosstab = (
            work_df.group_by(["pair_source_label", top_col])
            .len()
            .pivot(on=top_col, index="pair_source_label", values="len")
            .fill_null(0)
        )

        # Save crosstab
        crosstab.write_csv(results_dir / f"categorical-{cat_col}-crosstab.csv")

        # Extract matrix (rows = shared/mined, cols = category levels)
        value_cols = [c for c in crosstab.columns if c != "pair_source_label"]
        table = crosstab.select(value_cols).to_numpy()

        chi2, p_val, dof, _ = chi2_contingency(table)
        v = association(table, method="cramer")
        p_bonf = min(p_val * n_cat_tests, 1.0)

        cat_rows.append(
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

    cat_df = pl.DataFrame(cat_rows).sort("p_value")
    cat_df.write_csv(results_dir / "categorical-feature-tests.csv")
    log.info("Saved categorical-feature-tests.csv (%d features)", cat_df.height)

    # ── C. Effect size dot plot ──
    # Combine numeric (rank-biserial r) and categorical (Cramer's V) into one plot
    effect_rows: list[dict[str, str | float]] = []
    for row in numeric_df.iter_rows(named=True):
        sig = "*" if row["p_value_bonferroni"] < 0.05 else ""
        if row["p_value_bonferroni"] < 0.001:
            sig = "***"
        elif row["p_value_bonferroni"] < 0.01:
            sig = "**"
        effect_rows.append(
            {
                "feature": row["feature"],
                "effect_size": row["rank_biserial_r"],
                "abs_effect_size": abs(row["rank_biserial_r"]),
                "test_type": "Mann-Whitney U",
                "significance": sig,
            }
        )
    for row in cat_df.iter_rows(named=True):
        sig = "*" if row["p_value_bonferroni"] < 0.05 else ""
        if row["p_value_bonferroni"] < 0.001:
            sig = "***"
        elif row["p_value_bonferroni"] < 0.01:
            sig = "**"
        effect_rows.append(
            {
                "feature": row["feature"],
                "effect_size": row["cramers_v"],
                "abs_effect_size": row["cramers_v"],
                "test_type": "Chi-square",
                "significance": sig,
            }
        )

    effect_df = pl.DataFrame(effect_rows).sort("abs_effect_size")

    fig, ax = plt.subplots(figsize=(10, max(6, effect_df.height * 0.4)))
    y_positions = list(range(effect_df.height))
    sizes = effect_df["effect_size"].to_list()
    labels = [
        f"{FEATURE_DISPLAY_NAMES.get(row['feature'], row['feature'])} {row['significance']}"
        for row in effect_df.iter_rows(named=True)
    ]
    colors = [PALETTE[0] if s >= 0 else PALETTE[1] for s in sizes]

    ax.barh(y_positions, sizes, color=colors, height=0.6)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Effect Size (rank-biserial r / Cramer's V)")
    ax.set_title(
        "Feature Comparison: Shared vs Mined Pairs\n"
        f"(N={n_shared + n_mined:,}; "
        f"{n_shared:,} shared, {n_mined:,} mined)",
        fontsize=13,
    )
    fig.savefig(results_dir / "effect-size-dotplot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("Saved effect-size-dotplot.png")

    # ── D. Distributional comparison plots for significant numeric features ──
    sig_numeric = numeric_df.filter(pl.col("p_value_bonferroni") < 0.05)

    for row in sig_numeric.iter_rows(named=True):
        feat = row["feature"]
        display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
        shared_vals = shared[feat].drop_nulls().drop_nans().to_numpy()
        mined_vals = mined[feat].drop_nulls().drop_nans().to_numpy()

        # Compute y-axis limits from 1st-99th percentile to avoid outlier distortion
        all_vals = np.concatenate([shared_vals, mined_vals])
        y_lo = float(np.percentile(all_vals, 1))
        y_hi = float(np.percentile(all_vals, 99))
        y_pad = (y_hi - y_lo) * 0.05
        y_lo -= y_pad
        y_hi += y_pad

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Violin plot
        plot_data = pl.DataFrame(
            {
                feat: np.concatenate([shared_vals, mined_vals]),
                "pair_source_label": (
                    ["shared"] * len(shared_vals) + ["mined"] * len(mined_vals)
                ),
            }
        ).to_pandas()
        sns.violinplot(
            data=plot_data,
            x="pair_source_label",
            y=feat,
            hue="pair_source_label",
            palette=[PALETTE[0], PALETTE[1]],
            inner="quartile",
            ax=ax1,
        )
        ax1.set_ylim(y_lo, y_hi)
        ax1.set_xlabel("")
        ax1.set_ylabel(display_name)

        # KDE overlay
        sns.kdeplot(shared_vals, label="Shared", color=PALETTE[0], ax=ax2)
        sns.kdeplot(mined_vals, label="Mined", color=PALETTE[1], ax=ax2)
        ax2.legend()
        ax2.set_xlabel(display_name)

        fig.suptitle(
            f"{display_name}\n"
            f"Mann-Whitney U p={row['p_value']:.2e}, "
            f"rank-biserial r={row['rank_biserial_r']:.3f} "
            f"({row['effect_magnitude']})",
            fontsize=13,
        )
        fig.subplots_adjust(top=0.85)
        fig.savefig(results_dir / f"dist-numeric-{feat}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    # Overview grid of all significant numeric features
    if sig_numeric.height > 0:
        n_feats = sig_numeric.height
        ncols = min(3, n_feats)
        nrows = (n_feats + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True
        )
        axes_flat = np.array(axes).flatten() if n_feats > 1 else [axes]

        for i, row in enumerate(sig_numeric.iter_rows(named=True)):
            feat = row["feature"]
            display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
            ax = axes_flat[i]
            shared_vals = shared[feat].drop_nulls().drop_nans().to_numpy()
            mined_vals = mined[feat].drop_nulls().drop_nans().to_numpy()

            all_vals = np.concatenate([shared_vals, mined_vals])
            y_lo = float(np.percentile(all_vals, 1))
            y_hi = float(np.percentile(all_vals, 99))
            y_pad = (y_hi - y_lo) * 0.05
            y_lo -= y_pad
            y_hi += y_pad

            plot_data = pl.DataFrame(
                {
                    feat: np.concatenate([shared_vals, mined_vals]),
                    "pair_source_label": (
                        ["shared"] * len(shared_vals) + ["mined"] * len(mined_vals)
                    ),
                }
            ).to_pandas()
            sns.violinplot(
                data=plot_data,
                x="pair_source_label",
                y=feat,
                hue="pair_source_label",
                palette=[PALETTE[0], PALETTE[1]],
                inner="quartile",
                ax=ax,
            )
            ax.set_ylim(y_lo, y_hi)
            ax.set_ylabel(display_name)
            ax.set_title(f"{display_name}\nr={row['rank_biserial_r']:.3f}", fontsize=9)
            ax.set_xlabel("")

        # Hide unused axes
        for j in range(n_feats, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Significant Numeric Features: Shared vs Mined", fontsize=14)
        fig.savefig(results_dir / "dist-numeric-overview.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        log.info("Saved dist-numeric-overview.png")

    # ── Distributional comparison plots for significant categorical features ──
    sig_cat = cat_df.filter(pl.col("p_value_bonferroni") < 0.05)

    for row in sig_cat.iter_rows(named=True):
        cat_col = row["feature"]
        work_df = pairs.with_columns(pl.col(cat_col).fill_null("Unknown"))
        work_df, _, top_col = _add_top_n_other_column(work_df, cat_col, top_n)

        # Compute proportions within each group
        props = (
            work_df.group_by(["pair_source_label", top_col])
            .len()
            .with_columns(
                (pl.col("len") / pl.col("len").sum().over("pair_source_label")).alias(
                    "proportion"
                )
            )
            .sort(top_col)
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        props_pd = props.to_pandas()
        categories = sorted(props_pd[top_col].unique())
        x = np.arange(len(categories))
        width = 0.35

        shared_props = []
        mined_props = []
        for cat in categories:
            s = props_pd[
                (props_pd["pair_source_label"] == "shared") & (props_pd[top_col] == cat)
            ]["proportion"]
            shared_props.append(float(s.iloc[0]) if len(s) > 0 else 0)
            m = props_pd[
                (props_pd["pair_source_label"] == "mined") & (props_pd[top_col] == cat)
            ]["proportion"]
            mined_props.append(float(m.iloc[0]) if len(m) > 0 else 0)

        ax.bar(x - width / 2, shared_props, width, label="Shared", color=PALETTE[0])
        ax.bar(x + width / 2, mined_props, width, label="Mined", color=PALETTE[1])
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Proportion within group")
        ax.legend()
        cat_display = FEATURE_DISPLAY_NAMES.get(cat_col, cat_col)
        ax.set_title(
            f"{cat_display}\n"
            f"Chi-square p={row['p_value']:.2e}, "
            f"Cramer's V={row['cramers_v']:.3f} ({row['effect_magnitude']})",
            fontsize=13,
        )
        fig.savefig(
            results_dir / f"dist-categorical-{cat_col}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

    # ── Combined 2 by 2 categorical overview figure ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    axes_flat = axes.flatten()

    _cat_plot_order = [
        "document_field_name",
        "document_is_open_access",
        "document_type",
        "repository_primary_language",
    ]

    for i, cat_col in enumerate(_cat_plot_order):
        ax = axes_flat[i]
        if cat_col not in pairs.columns:
            ax.set_visible(False)
            continue

        work_df = pairs.with_columns(pl.col(cat_col).fill_null("Unknown"))
        work_df, _, top_col = _add_top_n_other_column(work_df, cat_col, top_n)

        props = (
            work_df.group_by(["pair_source_label", top_col])
            .len()
            .with_columns(
                (pl.col("len") / pl.col("len").sum().over("pair_source_label")).alias(
                    "proportion"
                )
            )
            .sort(top_col)
        )

        props_pd = props.to_pandas()
        categories = sorted(props_pd[top_col].unique())
        x = np.arange(len(categories))
        width = 0.35

        shared_props = []
        mined_props = []
        for cat in categories:
            s = props_pd[
                (props_pd["pair_source_label"] == "shared") & (props_pd[top_col] == cat)
            ]["proportion"]
            shared_props.append(float(s.iloc[0]) if len(s) > 0 else 0)
            m = props_pd[
                (props_pd["pair_source_label"] == "mined") & (props_pd[top_col] == cat)
            ]["proportion"]
            mined_props.append(float(m.iloc[0]) if len(m) > 0 else 0)

        ax.bar(x - width / 2, shared_props, width, label="Shared", color=PALETTE[0])
        ax.bar(x + width / 2, mined_props, width, label="Mined", color=PALETTE[1])
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Proportion within group")
        ax.legend()

        cat_display = FEATURE_DISPLAY_NAMES.get(cat_col, cat_col)
        stat_row = cat_df.filter(pl.col("feature") == cat_col)
        if stat_row.height > 0:
            r = stat_row.row(0, named=True)
            ax.set_title(
                f"{cat_display}\n"
                f"Chi-square p={r['p_value']:.2e}, "
                f"Cramer's V={r['cramers_v']:.3f} ({r['effect_magnitude']})",
                fontsize=11,
            )
        else:
            ax.set_title(cat_display, fontsize=11)

    fig.suptitle("Categorical Features: Shared vs Mined Pairs", fontsize=16)
    fig.savefig(results_dir / "dist-categorical-overview.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("Saved dist-categorical-overview.png")

    # ── E. Summary markdown ──
    lines = [
        "# Feature Comparison: Shared vs Mined Pairs\n",
        f"- **N**: {n_shared + n_mined:,} pairs ({n_shared:,} shared, {n_mined:,} mined)",
        "",
        "## Numeric Features (Mann-Whitney U)\n",
        "Positive rank-biserial r means higher in **shared**; "
        "negative means higher in **mined**.\n",
    ]
    for row in numeric_df.sort("p_value").iter_rows(named=True):
        sig = "**" if row["p_value_bonferroni"] < 0.05 else ""
        lines.append(
            f"- {sig}{row['feature']}{sig}: r={row['rank_biserial_r']:.3f} "
            f"({row['effect_magnitude']}), "
            f"p={row['p_value']:.2e} "
            f"(Bonf: {row['p_value_bonferroni']:.2e}), "
            f"median shared={row['median_shared']:.3f}, "
            f"median mined={row['median_mined']:.3f}"
        )

    lines.extend(
        [
            "",
            "## Categorical Features (Chi-square)\n",
        ]
    )
    for row in cat_df.sort("p_value").iter_rows(named=True):
        sig = "**" if row["p_value_bonferroni"] < 0.05 else ""
        lines.append(
            f"- {sig}{row['feature']}{sig}: Cramer's V={row['cramers_v']:.3f} "
            f"({row['effect_magnitude']}), "
            f"chi2={row['chi2']:.1f}, dof={row['dof']}, "
            f"p={row['p_value']:.2e} (Bonf: {row['p_value_bonferroni']:.2e})"
        )

    with open(results_dir / "feature-comparison-summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    log.info("Feature comparison complete.")


###############################################################################
# Workstream 3: Proportional Coverage
###############################################################################


def compute_coverage(
    pairs: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Compute proportional coverage metrics (pair-level and author-developer-level)."""
    log.info("Computing coverage metrics...")

    # --- Pair-level coverage (overall) ---
    n_shared = int(pairs.filter(pl.col("pair_source_label") == "shared").height)
    n_mined = int(pairs.filter(pl.col("pair_source_label") == "mined").height)
    n_total = n_shared + n_mined

    overall_coverage = {
        "n_shared_pairs": n_shared,
        "n_mined_pairs": n_mined,
        "n_total_pairs": n_total,
        "shared_pair_coverage": round(n_shared / n_total, 4) if n_total > 0 else 0,
        "mined_pair_coverage": round(n_mined / n_total, 4) if n_total > 0 else 0,
    }

    # --- Author-developer coverage ---
    shared_links = pairs.filter(pl.col("pair_source_label") == "shared").select(
        "document_id", "repository_id"
    )
    full_links = pairs.select("document_id", "repository_id")

    shared_ad = _get_author_developer_pairs_connected_to_pairs(shared_links)
    n_shared_ad = shared_ad.select("researcher_id", "developer_account_id").unique().height

    full_ad = _get_author_developer_pairs_connected_to_pairs(full_links)
    n_full_ad = full_ad.select("researcher_id", "developer_account_id").unique().height

    overall_coverage["n_shared_author_dev_pairs"] = n_shared_ad
    overall_coverage["n_full_author_dev_pairs"] = n_full_ad
    overall_coverage["shared_author_dev_coverage"] = (
        round(n_shared_ad / n_full_ad, 4) if n_full_ad > 0 else 0
    )

    with open(results_dir / "coverage-overall.json", "w") as f:
        json.dump(overall_coverage, f, indent=2)
    log.info("Overall coverage: %s", overall_coverage)

    # --- Coverage by field ---
    field_coverage_rows: list[dict[str, str | int | float]] = []
    for (field_name,), group in pairs.group_by("document_field_name"):
        if field_name is None:
            field_name = "Unknown"
        n_s = int(group.filter(pl.col("pair_source_label") == "shared").height)
        n_m = int(group.filter(pl.col("pair_source_label") == "mined").height)
        n_t = n_s + n_m
        field_coverage_rows.append(
            {
                "field": str(field_name),
                "n_shared": n_s,
                "n_mined": n_m,
                "n_total": n_t,
                "shared_coverage": round(n_s / n_t, 4) if n_t > 0 else 0,
            }
        )
    pl.DataFrame(field_coverage_rows).sort("n_total", descending=True).write_csv(
        results_dir / "coverage-by-field.csv"
    )

    # --- Coverage by year ---
    year_coverage_rows: list[dict[str, int | float | None]] = []
    for (year,), group in pairs.group_by("document_publication_year"):
        n_s = int(group.filter(pl.col("pair_source_label") == "shared").height)
        n_m = int(group.filter(pl.col("pair_source_label") == "mined").height)
        n_t = n_s + n_m
        year_coverage_rows.append(
            {
                "year": year,
                "n_shared": n_s,
                "n_mined": n_m,
                "n_total": n_t,
                "shared_coverage": round(n_s / n_t, 4) if n_t > 0 else 0,
            }
        )
    pl.DataFrame(year_coverage_rows).sort("year").write_csv(
        results_dir / "coverage-by-year.csv"
    )

    # --- Coverage by iteration ---
    mined_pairs = pairs.filter(
        pl.col("pair_source_label") == "mined",
        pl.col("iteration").is_not_null(),
    )
    iter_coverage_rows: list[dict[str, int | float | None]] = []
    for (iteration,), group in mined_pairs.group_by("iteration"):
        iter_coverage_rows.append(
            {
                "iteration": iteration,
                "n_mined_pairs": int(group.height),
            }
        )
    if iter_coverage_rows:
        pl.DataFrame(iter_coverage_rows).sort("iteration").write_csv(
            results_dir / "coverage-by-iteration.csv"
        )

    # --- Coverage bar chart ---
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), constrained_layout=True)

    # Pair coverage bar
    bars_left = axes[0].bar(
        ["Shared", "Mined"],
        [n_shared, n_mined],
        color=[PALETTE[0], PALETTE[1]],
    )
    for bar, val, pct in zip(
        bars_left,
        [n_shared, n_mined],
        [overall_coverage["shared_pair_coverage"], overall_coverage["mined_pair_coverage"]],
        strict=False,
    ):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}\n({pct:.1%})",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    axes[0].set_title("Article-Repository Pair Coverage", fontsize=12)
    axes[0].set_ylabel("Number of Pairs")

    # Author-developer coverage bar
    n_mined_only_ad = n_full_ad - n_shared_ad
    bars_right = axes[1].bar(
        ["Shared-only", "Mined-added"],
        [n_shared_ad, n_mined_only_ad],
        color=[PALETTE[0], PALETTE[1]],
    )
    ad_total = n_full_ad
    for bar, val in zip(bars_right, [n_shared_ad, n_mined_only_ad], strict=False):
        pct = val / ad_total if ad_total > 0 else 0
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}\n({pct:.1%})",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    axes[1].set_title(
        "Author-Developer Pair Coverage\n"
        f"(Shared covers {overall_coverage['shared_author_dev_coverage']:.1%} of full)",
        fontsize=12,
    )
    axes[1].set_ylabel("Number of Author-Developer Pairs")

    fig.savefig(results_dir / "coverage-bars.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    log.info("Coverage analysis complete.")


###############################################################################
# Workstream 4: Component Analysis
###############################################################################


def _compute_graph_stats(
    graph: rx.PyGraph,
    label: str,
) -> dict[str, int | float | list[int]]:
    """Compute standard network statistics for a graph."""
    components = rx.connected_components(graph) if graph.num_nodes() > 0 else []
    component_sizes = sorted([len(c) for c in components], reverse=True)

    total_nodes = graph.num_nodes()
    largest_cc_size = component_sizes[0] if component_sizes else 0
    coverage = largest_cc_size / total_nodes if total_nodes > 0 else 0

    return {
        "label": label,
        "total_nodes": total_nodes,
        "total_edges": graph.num_edges(),
        "total_components": len(components),
        "largest_component_size": largest_cc_size,
        "coverage_pct": round(coverage * 100, 2),
        "largest_5": component_sizes[:5],
        "isolates_size_1": component_sizes.count(1),
        "size_2_to_10": sum(1 for s in component_sizes if 2 <= s <= 10),
        "size_11_to_100": sum(1 for s in component_sizes if 11 <= s <= 100),
        "size_gt_100": sum(1 for s in component_sizes if s > 100),
    }


def analyze_network_components(  # noqa: C901
    pairs: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Build co-authorship graphs and compare components across subsets."""
    log.info("Starting network component analysis...")

    all_doc_contribs = _read_table("document_contributor")

    # Shared-only graph
    shared_doc_ids = (
        pairs.filter(pl.col("pair_source_label") == "shared")["document_id"].unique().to_list()
    )
    shared_contribs = all_doc_contribs.filter(pl.col("document_id").is_in(shared_doc_ids))
    graph_shared, node_to_idx_shared, _ = _build_coauthorship_graph(shared_contribs)
    stats_shared = _compute_graph_stats(graph_shared, "shared")
    with open(results_dir / "network-stats-shared.json", "w") as f:
        json.dump(stats_shared, f, indent=2)

    # Full graph (shared + mined)
    full_doc_ids = pairs["document_id"].unique().to_list()
    full_contribs = all_doc_contribs.filter(pl.col("document_id").is_in(full_doc_ids))
    graph_full, node_to_idx_full, _ = _build_coauthorship_graph(full_contribs)
    stats_full = _compute_graph_stats(graph_full, "full")
    with open(results_dir / "network-stats-full.json", "w") as f:
        json.dump(stats_full, f, indent=2)

    # Mined-only graph
    mined_doc_ids = (
        pairs.filter(pl.col("pair_source_label") == "mined")["document_id"].unique().to_list()
    )
    mined_contribs = all_doc_contribs.filter(pl.col("document_id").is_in(mined_doc_ids))
    graph_mined, _, _ = _build_coauthorship_graph(mined_contribs)
    stats_mined = _compute_graph_stats(graph_mined, "mined")
    with open(results_dir / "network-stats-mined.json", "w") as f:
        json.dump(stats_mined, f, indent=2)

    # --- Bridging analysis ---
    # For each researcher in the shared graph, find their shared-only component.
    # Then check which full-graph component they belong to.
    # Count how many shared components merge in the full graph.

    shared_components = (
        rx.connected_components(graph_shared) if graph_shared.num_nodes() > 0 else []
    )
    full_components = rx.connected_components(graph_full) if graph_full.num_nodes() > 0 else []

    # Map researcher_id -> shared component index
    shared_rid_to_comp: dict[int, int] = {}
    for comp_idx, comp_nodes in enumerate(shared_components):
        for node_idx in comp_nodes:
            rid = graph_shared[node_idx]
            shared_rid_to_comp[rid] = comp_idx

    # Map researcher_id -> full component index
    full_rid_to_comp: dict[int, int] = {}
    for comp_idx, comp_nodes in enumerate(full_components):
        for node_idx in comp_nodes:
            rid = graph_full[node_idx]
            full_rid_to_comp[rid] = comp_idx

    # For each full component, find which shared components its members belonged to
    full_comp_to_shared_comps: defaultdict[int, set[int]] = defaultdict(set)
    for rid, shared_comp in shared_rid_to_comp.items():
        if rid in full_rid_to_comp:
            full_comp = full_rid_to_comp[rid]
            full_comp_to_shared_comps[full_comp].add(shared_comp)

    # Count merges: full components that contain members from >1 shared component
    n_merging_full_components = sum(
        1 for shared_comps in full_comp_to_shared_comps.values() if len(shared_comps) > 1
    )
    n_shared_comps_merged = sum(
        len(shared_comps)
        for shared_comps in full_comp_to_shared_comps.values()
        if len(shared_comps) > 1
    )

    bridging_summary = {
        "n_shared_components": stats_shared["total_components"],
        "n_full_components": stats_full["total_components"],
        "delta_components": stats_full["total_components"] - stats_shared["total_components"],
        "delta_coverage_pct": round(
            stats_full["coverage_pct"] - stats_shared["coverage_pct"], 2
        ),
        "n_merging_full_components": n_merging_full_components,
        "n_shared_components_that_merged": n_shared_comps_merged,
    }
    with open(results_dir / "network-bridging-summary.json", "w") as f:
        json.dump(bridging_summary, f, indent=2)

    log.info("Bridging summary: %s", bridging_summary)

    # --- Save comparison table CSV ---
    table_rows = []
    for stats in [stats_shared, stats_full, stats_mined]:
        table_rows.append(
            {
                "subset": stats["label"],
                "total_nodes": stats["total_nodes"],
                "total_edges": stats["total_edges"],
                "total_components": stats["total_components"],
                "largest_component_size": stats["largest_component_size"],
                "coverage_pct": stats["coverage_pct"],
            }
        )
    pl.DataFrame(table_rows).write_csv(results_dir / "network-comparison-table.csv")
    log.info("Saved network-comparison-table.csv")

    # --- Comparison bar chart ---
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), constrained_layout=True)

    subset_labels = ["Shared", "Full", "Mined"]
    all_stats = [stats_shared, stats_full, stats_mined]

    # Nodes
    node_raw = [s["total_nodes"] for s in all_stats]
    bars = axes[0].bar(subset_labels, node_raw, color=PALETTE[:3])
    for bar, raw in zip(bars, node_raw, strict=False):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{raw:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].set_title("Total Nodes")
    axes[0].set_ylabel("Count")

    # Components
    comp_raw = [s["total_components"] for s in all_stats]
    bars = axes[1].bar(subset_labels, comp_raw, color=PALETTE[:3])
    for bar, raw in zip(bars, comp_raw, strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{raw:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[1].set_title("Connected Components")
    axes[1].set_ylabel("Count")

    # Largest CC coverage (already a percentage)
    cov_vals = [s["coverage_pct"] for s in all_stats]
    bars = axes[2].bar(subset_labels, cov_vals, color=PALETTE[:3])
    for bar, val in zip(bars, cov_vals, strict=False):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[2].set_title("Largest Component Coverage (%)")
    axes[2].set_ylabel("Coverage (%)")

    fig.suptitle(
        "Co-Authorship Network Comparison: Shared vs Full vs Mined",
        fontsize=14,
    )
    fig.savefig(results_dir / "network-component-comparison.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    log.info("Network component analysis complete.")


###############################################################################
# Workstream 5: Shortest Path Analysis
###############################################################################


def _sample_shortest_paths(
    graph: rx.PyGraph,
    n_iterations: int,
    seed: int,
    desc: str = "Sampling shortest paths",
) -> tuple[dict[str, float | int], list[int]]:
    """Sample random shortest paths from the largest connected component.

    Returns
    -------
    tuple
        (stats_dict, raw_lengths) where raw_lengths can be used for plotting.
    """
    empty_stats: dict[str, float | int] = {
        "valid_paths": 0,
        "mean": float("nan"),
        "std": float("nan"),
        "median": float("nan"),
        "p10": float("nan"),
        "p90": float("nan"),
    }

    components = rx.connected_components(graph) if graph.num_nodes() > 0 else []
    if not components:
        return empty_stats, []

    largest_cc = max(components, key=len)
    subgraph = graph.subgraph(list(largest_cc))

    if subgraph.num_nodes() < 2:
        return empty_stats, []

    rng = random.Random(seed)
    subgraph_indices = list(range(subgraph.num_nodes()))
    dijkstra_lengths: list[int] = []

    for _ in tqdm(range(n_iterations), desc=desc):
        source, target = rng.sample(subgraph_indices, 2)
        dijkstra_res = rx.dijkstra_shortest_path_lengths(
            subgraph,
            source,
            lambda _: 1,
            goal=target,
        )
        dijkstra_lengths.append(dijkstra_res[target])

    vec = np.array(dijkstra_lengths)
    stats: dict[str, float | int] = {
        "largest_cc_nodes": int(subgraph.num_nodes()),
        "largest_cc_edges": int(subgraph.num_edges()),
        "valid_paths": len(vec),
        "mean": round(float(np.mean(vec)), 4),
        "std": round(float(np.std(vec)), 4),
        "median": round(float(np.median(vec)), 4),
        "p10": round(float(np.quantile(vec, 0.10)), 4),
        "p90": round(float(np.quantile(vec, 0.90)), 4),
    }
    return stats, dijkstra_lengths


def analyze_shortest_paths(
    pairs: pl.DataFrame,
    results_dir: Path,
    n_iterations: int,
) -> None:
    """Shortest-path analysis across shared-only, mined-only, and full subsets."""
    log.info("Starting shortest path analysis...")

    all_doc_contribs = _read_table("document_contributor")
    all_path_lengths: dict[str, list[int]] = {}

    subsets = {
        "shared": pairs.filter(pl.col("pair_source_label") == "shared"),
        "mined": pairs.filter(pl.col("pair_source_label") == "mined"),
        "full": pairs,
    }

    for subset_name, subset_df in subsets.items():
        log.info("Building graph for subset: %s (%d pairs)", subset_name, subset_df.height)
        doc_ids = subset_df["document_id"].unique().to_list()
        contribs = all_doc_contribs.filter(pl.col("document_id").is_in(doc_ids))
        graph, _, _ = _build_coauthorship_graph(contribs)

        path_stats, raw_lengths = _sample_shortest_paths(
            graph,
            n_iterations,
            seed=42,
            desc=f"Shortest paths ({subset_name})",
        )

        with open(results_dir / f"shortest-path-{subset_name}.json", "w") as f:
            json.dump(path_stats, f, indent=2)

        log.info("Shortest path stats (%s): %s", subset_name, path_stats)

        if raw_lengths:
            all_path_lengths[subset_name] = raw_lengths

    # --- Distribution plot ---
    subset_order = ["shared", "full", "mined"]
    if all_path_lengths:
        plot_rows: list[dict[str, str | int]] = []
        for subset_name in subset_order:
            for length in all_path_lengths.get(subset_name, []):
                plot_rows.append({"subset": subset_name, "shortest_path_length": length})

        plot_df = pl.DataFrame(plot_rows)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=plot_df,
            x="subset",
            y="shortest_path_length",
            hue="subset",
            order=subset_order,
            hue_order=subset_order,
            palette=PALETTE[:3],
            showfliers=False,
            ax=ax,
        )
        ax.set_title(
            "Shortest Path Length Distribution by Subset\n"
            "(Sampled from largest connected component of each co-authorship graph)",
            fontsize=13,
        )
        ax.set_xlabel("Subset")
        ax.set_ylabel("Shortest Path Length")
        fig.savefig(
            results_dir / "shortest-path-distributions.png", bbox_inches="tight", dpi=300
        )
        plt.close(fig)

    log.info("Shortest path analysis complete.")


###############################################################################
# Summary Writer
###############################################################################


def write_summary(base_dir: Path, results_dir: Path) -> None:  # noqa: C901
    """Write a summary markdown file from saved result artifacts.

    Parameters
    ----------
    base_dir:
        Root results directory containing step-{i}/ subdirectories.
    results_dir:
        Directory to write the summary file into (step-6/).
    """
    lines = ["# RQ2 Analysis Summary\n"]

    # Coverage (step-3)
    coverage_path = base_dir / "step-3" / "coverage-overall.json"
    if coverage_path.exists():
        with open(coverage_path) as f:
            coverage = json.load(f)
        lines.extend(
            [
                "## Coverage\n",
                f"- **Total pairs**: {coverage['n_total_pairs']:,}",
                f"- **Shared pairs**: {coverage['n_shared_pairs']:,} ({coverage['shared_pair_coverage']:.1%})",
                f"- **Mined pairs**: {coverage['n_mined_pairs']:,} ({coverage['mined_pair_coverage']:.1%})",
                f"- **Shared author-dev pairs**: {coverage['n_shared_author_dev_pairs']:,}",
                f"- **Full author-dev pairs**: {coverage['n_full_author_dev_pairs']:,}",
                f"- **Shared author-dev coverage**: {coverage['shared_author_dev_coverage']:.1%}",
                "",
            ]
        )

    # Feature comparison (step-2)
    numeric_path = base_dir / "step-2" / "numeric-feature-tests.csv"
    if numeric_path.exists():
        numeric_df = pl.read_csv(numeric_path)
        sig_numeric = numeric_df.filter(pl.col("p_value_bonferroni") < 0.05)
        lines.extend(
            [
                "## Feature Comparison\n",
                f"- **Numeric features tested**: {numeric_df.height}",
                f"- **Significant (Bonferroni p < 0.05)**: {sig_numeric.height}",
                "",
                "### Significant Numeric Features (by effect size)\n",
            ]
        )
        for row in sig_numeric.sort("rank_biserial_r", descending=True).iter_rows(named=True):
            lines.append(
                f"- **{row['feature']}**: r={row['rank_biserial_r']:.3f} "
                f"({row['effect_magnitude']}), {row['direction']}, "
                f"median shared={row['median_shared']:.3f} vs "
                f"mined={row['median_mined']:.3f}"
            )
        lines.append("")

    cat_path = base_dir / "step-2" / "categorical-feature-tests.csv"
    if cat_path.exists():
        cat_df = pl.read_csv(cat_path)
        sig_cat = cat_df.filter(pl.col("p_value_bonferroni") < 0.05)
        lines.extend(
            [
                "### Significant Categorical Features\n",
            ]
        )
        for row in sig_cat.sort("cramers_v", descending=True).iter_rows(named=True):
            lines.append(
                f"- **{row['feature']}**: Cramer's V={row['cramers_v']:.3f} "
                f"({row['effect_magnitude']}), "
                f"chi2={row['chi2']:.1f}"
            )
        lines.append("")

    # Network stats (step-4)
    for label in ["shared", "full", "mined"]:
        stats_path = base_dir / "step-4" / f"network-stats-{label}.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            lines.extend(
                [
                    f"## Network: {label.capitalize()}\n",
                    f"- **Nodes**: {stats['total_nodes']:,}",
                    f"- **Edges**: {stats['total_edges']:,}",
                    f"- **Components**: {stats['total_components']:,}",
                    f"- **Largest CC coverage**: {stats['coverage_pct']:.1f}%",
                    "",
                ]
            )

    # Bridging (step-4)
    bridging_path = base_dir / "step-4" / "network-bridging-summary.json"
    if bridging_path.exists():
        with open(bridging_path) as f:
            bridging = json.load(f)
        lines.extend(
            [
                "## Bridging Analysis\n",
                f"- **Delta components**: {bridging['delta_components']:,}",
                f"- **Delta coverage**: {bridging['delta_coverage_pct']:.1f}%",
                f"- **Merging full components**: {bridging['n_merging_full_components']:,}",
                f"- **Shared components merged**: {bridging['n_shared_components_that_merged']:,}",
                "",
            ]
        )

    # Shortest paths (step-5)
    for label in ["shared", "mined", "full"]:
        sp_path = base_dir / "step-5" / f"shortest-path-{label}.json"
        if sp_path.exists():
            with open(sp_path) as f:
                sp = json.load(f)
            lines.extend(
                [
                    f"## Shortest Paths: {label.capitalize()}\n",
                    f"- **Mean**: {sp['mean']:.2f} (std: {sp['std']:.2f})",
                    f"- **Median**: {sp['median']:.2f}",
                    f"- **p10-p90**: {sp['p10']:.2f} - {sp['p90']:.2f}",
                    f"- **Sampled paths**: {sp['valid_paths']:,}",
                    "",
                ]
            )

    with open(results_dir / "rq2-summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    log.info("Saved rq2-summary.md")


###############################################################################
# Additional analyses
###############################################################################

_FINE_SOURCE_LABEL_EXPR = (
    pl.when(pl.col("dataset_source_names_str").str.to_lowercase().str.contains("joss"))
    .then(pl.lit("JOSS"))
    .when(pl.col("dataset_source_names_str").str.to_lowercase().str.contains("plos"))
    .then(pl.lit("PLOS"))
    .when(pl.col("dataset_source_names_str").str.to_lowercase().str.contains("pwc"))
    .then(pl.lit("PwC"))
    .when(pl.col("dataset_source_names_str").str.to_lowercase().str.contains("softwarex"))
    .then(pl.lit("SoftwareX"))
    .when(pl.col("dataset_source_names_str").str.to_lowercase().str.contains("snowball"))
    .then(pl.lit("Mined"))
    .otherwise(pl.lit("Other"))
    .alias("fine_source_label")
)

_FINE_SOURCE_ORDER = ["JOSS", "PLOS", "PwC", "SoftwareX", "Mined"]


def run_source_level_comparison(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Supplementary: disaggregate shared sources into JOSS/PLOS/PwC/SoftwareX vs Mined."""
    log.info("Starting source-level comparison...")

    source_df = pairs.with_columns(_FINE_SOURCE_LABEL_EXPR)
    present_sources = [
        s
        for s in _FINE_SOURCE_ORDER
        if source_df.filter(pl.col("fine_source_label") == s).height >= 5
    ]

    source_df["fine_source_label"].value_counts(sort=True).write_csv(
        results_dir / "source-level-counts.csv"
    )

    # Kruskal-Wallis across sources for key numeric features
    numeric_cols = [c for c in NUMERIC_FEATURES if c in source_df.columns]
    kw_rows: list[dict] = []
    for col_name in numeric_cols:
        groups = [
            source_df.filter(pl.col("fine_source_label") == s)[col_name]
            .drop_nulls()
            .drop_nans()
            .to_numpy()
            for s in present_sources
        ]
        valid_groups = [g for g in groups if len(g) >= 2]
        if len(valid_groups) < 2:
            continue
        kw_stat, kw_p = kruskal(*valid_groups)
        kw_rows.append(
            {
                "feature": col_name,
                "kruskal_h": round(float(kw_stat), 4),
                "p_value": float(kw_p),
            }
        )
    if kw_rows:
        pl.DataFrame(kw_rows).sort("p_value").write_csv(
            results_dir / "source-level-numeric-kruskal.csv"
        )

    # Field composition by source
    field_source_df = source_df.filter(
        pl.col("document_field_name").is_not_null(),
        pl.col("document_field_name") != "",
    )
    field_source_df, _, field_col = _add_top_n_other_column(
        field_source_df, "document_field_name", top_n
    )
    field_source_counts = field_source_df.group_by(["fine_source_label", field_col]).agg(
        pl.len().alias("n_pairs")
    )
    source_totals = source_df.group_by("fine_source_label").agg(pl.len().alias("total_pairs"))
    field_source_pct = field_source_counts.join(
        source_totals, on="fine_source_label", how="left"
    ).with_columns((pl.col("n_pairs") / pl.col("total_pairs") * 100).alias("pct_pairs"))
    field_source_pct.write_csv(results_dir / "source-level-field-composition.csv")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7), constrained_layout=True)

    # Panel 1: Field composition stacked bar
    try:
        pivot_pandas = (
            field_source_pct.filter(pl.col("fine_source_label").is_in(present_sources))
            .pivot(on=field_col, index="fine_source_label", values="pct_pairs")
            .fill_null(0)
            .to_pandas()
            .set_index("fine_source_label")
            .reindex(present_sources)
        )
        pivot_pandas.plot(
            kind="barh",
            stacked=True,
            ax=axes[0],
            colormap="tab10",
        )
        axes[0].set_title(
            f"Field Composition by Data Source (Top {top_n} + Other)\n(% of pairs per source)",
            fontsize=12,
        )
        axes[0].set_xlabel("Percentage of Pairs (%)")
        axes[0].set_ylabel("")
        axes[0].legend(title="Field", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    except Exception as exc:
        log.warning("Failed to render source field composition chart: %s", exc)
        axes[0].set_axis_off()

    # Panel 2: FWCI violin by source
    fwci_source_df = source_df.filter(
        pl.col("fine_source_label").is_in(present_sources),
        pl.col("document_fwci_log1p").is_not_null(),
        pl.col("document_fwci_log1p").is_not_nan(),
        pl.col("document_fwci_log1p").is_finite(),
    )
    if fwci_source_df.height > 0:
        sns.violinplot(
            data=fwci_source_df.to_pandas(),
            y="fine_source_label",
            x="document_fwci_log1p",
            order=present_sources,
            ax=axes[1],
            inner="quartile",
            density_norm="width",
        )
        axes[1].set_title(
            "Article FWCI Distribution by Data Source\n(log(1+FWCI))",
            fontsize=12,
        )
        axes[1].set_xlabel("Article FWCI (log)")
        axes[1].set_ylabel("")
    else:
        axes[1].set_axis_off()

    fig.savefig(results_dir / "source-level-comparison.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("Source-level comparison complete.")


def analyze_field_coverage_gap(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Quantify per-field mining dependence and characterize most mining-dependent fields."""
    log.info("Starting field coverage gap analysis...")

    field_df = pairs.filter(
        pl.col("document_field_name").is_not_null(),
        pl.col("document_field_name") != "",
    )
    field_df, top_fields, field_col = _add_top_n_other_column(
        field_df, "document_field_name", top_n
    )

    field_source_counts = field_df.group_by([field_col, "pair_source_label"]).agg(
        pl.len().alias("n_pairs")
    )
    field_totals = field_df.group_by(field_col).agg(pl.len().alias("total_pairs"))

    field_gap = (
        field_source_counts.pivot(on="pair_source_label", index=field_col, values="n_pairs")
        .fill_null(0)
        .join(field_totals, on=field_col, how="left")
    )

    if "mined" not in field_gap.columns or "shared" not in field_gap.columns:
        log.warning("Missing mined or shared column in field_gap; skipping.")
        return

    field_gap = field_gap.with_columns(
        (pl.col("mined") / pl.col("total_pairs") * 100).alias("pct_mined")
    ).sort("pct_mined", descending=True)
    field_gap.write_csv(results_dir / "field-coverage-gap.csv")

    # For top-5 most mining-dependent fields: compare FWCI within field
    top_mined_fields = field_gap.head(5)[field_col].to_list()
    within_field_rows: list[dict] = []
    for field_name in top_mined_fields:
        sub = field_df.filter(pl.col(field_col) == field_name)
        shared_fwci = sub.filter(
            pl.col("pair_source_label") == "shared",
            pl.col("document_fwci_log1p").is_not_null(),
            pl.col("document_fwci_log1p").is_not_nan(),
            pl.col("document_fwci_log1p").is_finite(),
        )["document_fwci_log1p"].to_numpy()
        mined_fwci = sub.filter(
            pl.col("pair_source_label") == "mined",
            pl.col("document_fwci_log1p").is_not_null(),
            pl.col("document_fwci_log1p").is_not_nan(),
            pl.col("document_fwci_log1p").is_finite(),
        )["document_fwci_log1p"].to_numpy()
        within_field_rows.append(
            {
                "field": field_name,
                "n_shared": len(shared_fwci),
                "n_mined": len(mined_fwci),
                "median_fwci_log1p_shared": float(np.median(shared_fwci))
                if len(shared_fwci) > 0
                else float("nan"),
                "median_fwci_log1p_mined": float(np.median(mined_fwci))
                if len(mined_fwci) > 0
                else float("nan"),
            }
        )
    pl.DataFrame(within_field_rows).write_csv(
        results_dir / "top-mined-fields-fwci-comparison.csv"
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7), constrained_layout=True)

    # Panel 1: Lollipop — % mined by field (ascending order)
    plot_data = field_gap.sort("pct_mined", descending=False)
    field_plot_order = plot_data[field_col].to_list()
    pct_mined_vals = plot_data["pct_mined"].to_list()
    y_pos = list(range(len(field_plot_order)))

    axes[0].hlines(y=y_pos, xmin=0, xmax=pct_mined_vals, color="gray", linewidth=1.5, alpha=0.7)
    axes[0].scatter(pct_mined_vals, y_pos, color=PALETTE[0], s=80, zorder=3)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(field_plot_order, fontsize=10)
    axes[0].set_xlabel("% of Field's Pairs from Mining")
    axes[0].set_ylabel("")
    axes[0].set_title(
        f"Field Dependence on Mining (Top {top_n} + Other)\n"
        "(% of pairs per field that came from snowball sampling)",
        fontsize=12,
    )
    axes[0].axvline(x=50, color="red", linestyle="--", alpha=0.5, label="50%")
    axes[0].legend()

    # Panel 2: Stacked bar shared vs mined by field (sorted by % mined descending)
    try:
        field_source_pct = field_source_counts.join(
            field_totals, on=field_col, how="left"
        ).with_columns((pl.col("n_pairs") / pl.col("total_pairs") * 100).alias("pct_pairs"))
        pivot_pandas = (
            field_source_pct.pivot(on="pair_source_label", index=field_col, values="pct_pairs")
            .fill_null(0)
            .to_pandas()
            .set_index(field_col)
            .reindex(field_gap.sort("pct_mined", descending=True)[field_col].to_list())
        )
        source_cols = [c for c in ["shared", "mined"] if c in pivot_pandas.columns]
        pivot_pandas[source_cols].plot(
            kind="barh",
            stacked=True,
            ax=axes[1],
            color=[PALETTE[0], PALETTE[1]],
        )
        axes[1].set_title(
            "Shared vs Mined Composition by Field\n(% of pairs per field)",
            fontsize=12,
        )
        axes[1].set_xlabel("Percentage of Pairs (%)")
        axes[1].set_ylabel("")
        axes[1].legend(title="Source", loc="lower right")
    except Exception as exc:
        log.warning("Failed to render field coverage gap stacked bar: %s", exc)
        axes[1].set_axis_off()

    fig.savefig(results_dir / "field-coverage-gap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("Field coverage gap analysis complete.")


def plot_publication_year_by_source(
    pairs: pl.DataFrame,
    results_dir: Path,
) -> None:
    """KDE and era-bin comparison of publication year distributions per data source."""
    log.info("Starting publication year by source analysis...")

    source_df = pairs.with_columns(_FINE_SOURCE_LABEL_EXPR).filter(
        pl.col("document_publication_year").is_not_null(),
        pl.col("document_publication_year") > 2010,
        pl.col("document_publication_year") < 2025,
    )

    present_sources = [
        s
        for s in _FINE_SOURCE_ORDER
        if source_df.filter(pl.col("fine_source_label") == s).height >= 10
    ]

    median_years = (
        source_df.filter(pl.col("fine_source_label").is_in(present_sources))
        .group_by("fine_source_label")
        .agg(pl.col("document_publication_year").median().alias("median_year"))
        .sort("median_year")
    )
    median_years.write_csv(results_dir / "publication-year-median-by-source.csv")

    binned_df = source_df.filter(
        pl.col("fine_source_label").is_in(present_sources)
    ).with_columns(
        pl.when(pl.col("document_publication_year") < 2015)
        .then(pl.lit("pre-2015"))
        .when(pl.col("document_publication_year") < 2020)
        .then(pl.lit("2015-2019"))
        .otherwise(pl.lit("2020+"))
        .alias("year_bin")
    )
    year_bin_order = ["pre-2015", "2015-2019", "2020+"]

    binned_counts = binned_df.group_by(["fine_source_label", "year_bin"]).agg(
        pl.len().alias("n_pairs")
    )
    source_totals = binned_df.group_by("fine_source_label").agg(pl.len().alias("total_pairs"))
    binned_pct = binned_counts.join(
        source_totals, on="fine_source_label", how="left"
    ).with_columns((pl.col("n_pairs") / pl.col("total_pairs") * 100).alias("pct_pairs"))
    binned_pct.write_csv(results_dir / "publication-year-bins-by-source.csv")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), constrained_layout=True)

    # Panel 1: Histogram + KDE per source
    median_year_lut = {
        row["fine_source_label"]: int(row["median_year"])
        for row in median_years.iter_rows(named=True)
    }
    for i, src in enumerate(present_sources):
        src_years = (
            source_df.filter(pl.col("fine_source_label") == src)["document_publication_year"]
            .cast(pl.Float64)
            .to_numpy()
        )
        if len(src_years) < 10:
            continue
        color = PALETTE[i % len(PALETTE)]
        med = median_year_lut.get(src, "?")
        axes[0].hist(
            src_years,
            bins=range(2010, 2026),
            density=True,
            alpha=0.25,
            color=color,
            label=f"{src} (median={med})",
        )
        sns.kdeplot(
            data=src_years, ax=axes[0], color=color, linewidth=2.0, bw_adjust=1.2, label=""
        )
    axes[0].set_title(
        "Publication Year Distribution by Data Source\n(histogram + KDE; median in legend)",
        fontsize=12,
    )
    axes[0].set_xlabel("Publication Year")
    axes[0].set_ylabel("Density")
    axes[0].legend(title="Source", fontsize=8)

    # Panel 2: Stacked bar of era bins by source
    try:
        pivot_pandas = (
            binned_pct.filter(pl.col("fine_source_label").is_in(present_sources))
            .pivot(on="year_bin", index="fine_source_label", values="pct_pairs")
            .fill_null(0)
            .to_pandas()
            .set_index("fine_source_label")
            .reindex(present_sources)
        )
        bin_cols = [c for c in year_bin_order if c in pivot_pandas.columns]
        pivot_pandas[bin_cols].plot(
            kind="barh",
            stacked=True,
            ax=axes[1],
            colormap="viridis",
        )
        axes[1].set_title(
            "Era Composition by Data Source\n(% of pairs per era)",
            fontsize=12,
        )
        axes[1].set_xlabel("Percentage of Pairs (%)")
        axes[1].set_ylabel("")
        axes[1].legend(title="Era", loc="lower right")
    except Exception as exc:
        log.warning("Failed to render year bins stacked bar: %s", exc)
        axes[1].set_axis_off()

    fig.savefig(results_dir / "publication-year-by-source.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("Publication year by source analysis complete.")


###############################################################################
# CLI
###############################################################################


def _run_rq2_analyses(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
    n_shortest_path_iterations: int,
    label: str,
    run_network_analyses: bool = True,
) -> None:
    """Run all RQ2 per-subset analyses and save results under results_dir."""
    total_steps = 9 if run_network_analyses else 7

    def _step_dir(n: int) -> Path:
        d = results_dir / f"step-{n}"
        d.mkdir(exist_ok=True)
        return d

    step = 0

    # Step 1: Save pairs dataset
    step += 1
    log.info("[%s] Step %d/%d: Saving pairs dataset...", label, step, total_steps)
    step_dir = _step_dir(step)
    pairs.write_parquet(step_dir / "rq2-pairs.parquet")
    log.info("[%s] Saved rq2-pairs.parquet", label)

    missingness_rows: list[dict[str, str | int | float]] = []
    for col_name in pairs.columns:
        n_total = pairs.height
        n_null = pairs[col_name].null_count()
        missingness_rows.append(
            {
                "column": col_name,
                "n_total": n_total,
                "n_null": n_null,
                "pct_null": round(100 * n_null / n_total, 2) if n_total > 0 else 0,
            }
        )
    pl.DataFrame(missingness_rows).write_csv(step_dir / "rq2-missingness.csv")

    schema_dict = {col: str(dtype) for col, dtype in pairs.schema.items()}
    with open(step_dir / "rq2-pairs-schema.json", "w") as f:
        json.dump(schema_dict, f, indent=2)

    readme_lines = [
        "# rq2-pairs.parquet",
        "",
        "## What is this file?",
        "",
        "This is the assembled analysis dataset for RQ2. It joins",
        "`document_repository_link` with document metadata (from `document`),",
        "repository metadata (from `repository`), topic/field classifications,",
        "author counts, contributor counts, file counts, provenance labels",
        "(`pair_source_label`: shared vs mined), and derived features",
        "(log transforms, FWSI, temporal durations).",
        "",
        "## Why persist it?",
        "",
        "1. **Downstream steps** (feature comparison, coverage, network analysis)",
        "   read from this DataFrame rather than re-querying the database.",
        "2. **Reproducibility**: The exact dataset used for analysis is preserved.",
        "3. **Auditability**: The schema JSON and missingness CSV document the",
        "   data shape and completeness at analysis time.",
        "",
        "## Schema",
        "",
        "See `rq2-pairs-schema.json` for the Polars dtype of each column.",
        "See `rq2-missingness.csv` for null/NaN counts per column.",
        "",
    ]
    with open(step_dir / "README.md", "w") as f:
        f.write("\n".join(readme_lines))

    # Step 2: Feature comparison
    step += 1
    log.info("[%s] Step %d/%d: Univariate feature comparison...", label, step, total_steps)
    run_feature_comparison(pairs, _step_dir(step), top_n)

    # Step 3: Coverage
    step += 1
    log.info("[%s] Step %d/%d: Coverage analysis...", label, step, total_steps)
    compute_coverage(pairs, _step_dir(step))

    if run_network_analyses:
        # Step 4: Network components
        step += 1
        log.info("[%s] Step %d/%d: Network component analysis...", label, step, total_steps)
        analyze_network_components(pairs, _step_dir(step))

        # Step 5: Shortest paths
        step += 1
        log.info("[%s] Step %d/%d: Shortest path analysis...", label, step, total_steps)
        analyze_shortest_paths(pairs, _step_dir(step), n_shortest_path_iterations)

    # Step: Source-level disaggregation (supplementary)
    step += 1
    log.info("[%s] Step %d/%d: Source-level comparison...", label, step, total_steps)
    run_source_level_comparison(pairs, _step_dir(step), top_n)

    # Step: Field coverage gap
    step += 1
    log.info("[%s] Step %d/%d: Field coverage gap analysis...", label, step, total_steps)
    analyze_field_coverage_gap(pairs, _step_dir(step), top_n)

    # Step: Publication year by source
    step += 1
    log.info("[%s] Step %d/%d: Publication year by source...", label, step, total_steps)
    plot_publication_year_by_source(pairs, _step_dir(step))

    # Final step: Summary
    step += 1
    log.info("[%s] Step %d/%d: Writing summary...", label, step, total_steps)
    write_summary(base_dir=results_dir, results_dir=_step_dir(step))


@app.command()
def analyze(
    top_n: int = typer.Option(9, help="Number of top categories (rest grouped as 'Other')."),
    n_shortest_path_iterations: int = typer.Option(
        5000, help="Random shortest path iterations for network analysis."
    ),
    sample_size: int | None = typer.Option(
        None, help="Sample this many pairs for faster analysis."
    ),
    run_network_analyses: bool = typer.Option(
        True, help="Whether to run the (potentially time-consuming) network analyses."
    ),
    debug: bool = typer.Option(False, help="Enable debug logging."),
) -> None:
    """Run the full RQ2 analysis pipeline.

    Each pair-based analysis is run twice — once on **all** pairs and once on
    a **high-confidence** subset (doc-repo confidence null or >= 0.995).
    Results are written to ``all/`` and ``high-conf/`` subdirectories.
    """
    setup_logger(debug=debug)

    results_dir = Path(__file__).parent / "rq2-results"

    # Delete existing results if present, to ensure clean slate for each run
    if results_dir.exists():
        log.info("Deleting existing results directory: %s", results_dir)
        shutil.rmtree(results_dir)

    results_dir.mkdir(exist_ok=True)

    sns.set_palette(PALETTE)

    log.info("Loading pairs (all)...")
    # pairs_all = load_rq2_pairs(sample_size=sample_size)
    log.info("Loading pairs (high-conf, >= 0.995)...")
    pairs_high_conf = load_rq2_pairs(
        sample_size=sample_size,
        doc_repo_confidence_threshold=0.995,
    )

    # for label, pairs in [("all", pairs_all), ("high-conf", pairs_high_conf)]:
    for label, pairs in [("high-conf", pairs_high_conf)]:
        subset_dir = results_dir / label
        subset_dir.mkdir(exist_ok=True)
        log.info("Running RQ2 analyses for subset: %s", label)
        _run_rq2_analyses(
            pairs,
            subset_dir,
            top_n,
            n_shortest_path_iterations,
            label,
            run_network_analyses=run_network_analyses,
        )

    log.info("RQ2 analysis complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    app()
