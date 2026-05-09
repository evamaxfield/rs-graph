import datetime
import json
import logging
import random
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import cast

import colormaps as cmaps
import connectorx  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rustworkx as rx
import seaborn as sns
import typer
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from tqdm import tqdm

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.db import constants as db_constants

###############################################################################
# Constants
###############################################################################

PALETTE = cmaps.bold._colors.tolist()  # type: ignore[attr-defined]

FEATURE_NAME_TO_VIZ_NAME_LUT = {
    "document_fwci": "Document FWCI",
    "repository_fwsi": "Repository FWSI",
    "document_cited_by_count": "Document Cited By Count",
    "repository_stargazers_count": "Repository Stargazers Count",
    "document_n_authors": "Document Number of Authors",
    "repository_n_contributors": "Repository Number of Contributors",
    "repository_n_files": "Repository Number of Files",
    "repository_commits_count": "Repository Commits Count",
}

OVER_TIME_METRICS_TO_VIZ_NAME_LUT = {
    "repository_commit_duration_days": "Repository Commit Duration (Days)",
    "repository_size_kb": "Repository Size (KB)",
    "repository_commits_count": "Repository Commits Count",
    "repository_n_files": "Repository Number of Files",
}

FIELD_LANGUAGE_TITLE_LUT = {
    "document_field_name_top_n_plus_other": "Field Counts",
    "field_count": "Field Count over Time",
    "repository_primary_language_top_n_plus_other": "Primary Language Counts",
    "language_count": "Primary Language Count over Time",
}

DATE_FEATURES_TO_VIZ_NAME_LUT = {
    "repository_creation_datetime": "Repository Creation Date",
    "document_publication_date": "Document Publication Date",
    "repository_last_pushed_datetime": "Repository Last Pushed Date",
}

LIFECYCLE_RELEASE_WINDOW_DAYS = 30
LIFECYCLE_LONG_MAINTENANCE_DAYS = 365

###############################################################################
# Logger & App
###############################################################################

log = logging.getLogger(__name__)
app = typer.Typer()

###############################################################################
# Private helpers
###############################################################################


def _add_top_n_other_column(
    df: pl.DataFrame,
    source_col: str,
    n: int,
) -> tuple[pl.DataFrame, list[str], str]:
    """Get top-N values of *source_col*, add a new column mapping the rest to 'Other'.

    Returns (modified_df, top_values_list, new_col_name).
    """
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


def _add_45_degree_line(
    ax: plt.Axes,
    x_series: pl.Series,
    y_series: pl.Series,
) -> None:
    """Draw a dashed red 45-degree reference line spanning the range of two date series."""
    x_dates = x_series.dt.date()
    y_dates = y_series.dt.date()
    x_min = cast(datetime.date, x_dates.min())
    y_min = cast(datetime.date, y_dates.min())
    x_max = cast(datetime.date, x_dates.max())
    y_max = cast(datetime.date, y_dates.max())
    min_date = min(x_min, y_min)
    max_date = max(x_max, y_max)
    date_range = [min_date, max_date]
    ax.plot(
        date_range,  # type: ignore[arg-type]
        date_range,  # type: ignore[arg-type]
        color="red",
        linestyle="--",
    )


def _plot_fwci_fwsi_point_data(row: dict, color: str, ax: plt.Axes) -> None:
    """Annotate a single point on the FWCI-vs-FWSI scatter plot."""
    doc_title_parts = row["document_title"].split()
    doc_title_short = (
        " ".join(doc_title_parts[:4]) + "..."
        if len(doc_title_parts) > 4
        else row["document_title"]
    )
    point_label = f"{doc_title_short} -- {row['repository_owner']}/{row['repository_name']}"
    ax.text(
        row["document_fwci_log10"] + 0.1,
        row["repository_fwsi_log10"] - 0.02,
        point_label,
        fontsize=8,
        color=color,
    )
    ax.plot(
        row["document_fwci_log10"],
        row["repository_fwsi_log10"],
        "o",
        color=color,
    )


def _compute_spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rho using rank correlation (without SciPy dependency)."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")

    x_rank = pl.Series("x", x).rank("average").to_numpy()
    y_rank = pl.Series("y", y).rank("average").to_numpy()

    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")

    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _fit_simple_linear_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Fit y = intercept + slope * x and return coefficients and R^2."""
    if len(x) < 2 or len(y) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
        }

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_var = float(np.sum((x - x_mean) ** 2))

    if x_var == 0:
        return {
            "slope": float("nan"),
            "intercept": y_mean,
            "r_squared": float("nan"),
        }

    slope = float(np.sum((x - x_mean) * (y - y_mean)) / x_var)
    intercept = y_mean - slope * x_mean

    y_hat = intercept + slope * x
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))
    r_squared = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
    }


def _kaplan_meier_curve(
    durations: np.ndarray,
    event_observed: np.ndarray,
) -> pl.DataFrame:
    """Compute a Kaplan-Meier survival curve from durations and event indicators.

    event_observed: 1 means event observed (maintenance ended), 0 means right-censored.
    """
    if len(durations) == 0:
        return pl.DataFrame(
            {
                "duration_days": pl.Series([], dtype=pl.Int64),
                "survival_probability": pl.Series([], dtype=pl.Float64),
                "at_risk": pl.Series([], dtype=pl.Int64),
                "events": pl.Series([], dtype=pl.Int64),
                "censored": pl.Series([], dtype=pl.Int64),
            }
        )

    order = np.argsort(durations)
    durations_sorted = durations[order]
    events_sorted = event_observed[order]

    unique_times = np.unique(durations_sorted)
    n_at_risk = len(durations_sorted)
    survival = 1.0

    rows: list[dict[str, float | int]] = []

    for t in unique_times:
        mask = durations_sorted == t
        d_i = int(np.sum(events_sorted[mask]))
        n_i = int(np.sum(mask))
        c_i = n_i - d_i

        if n_at_risk > 0:
            survival *= 1.0 - (d_i / n_at_risk)

        rows.append(
            {
                "duration_days": int(t),
                "survival_probability": float(survival),
                "at_risk": int(n_at_risk),
                "events": d_i,
                "censored": c_i,
            }
        )

        n_at_risk -= n_i

    return pl.DataFrame(rows)


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
    confidence_threshold: float | None = None,
) -> pl.DataFrame:
    """Return matched author-developer pairs connected to provided doc-repo pairs.

    Parameters
    ----------
    pair_links
        Must include ``document_id``, ``repository_id``, and optionally ``iteration``.
    confidence_threshold
        When set, keep only researcher-developer links whose
        ``predictive_model_confidence`` is null or >= this value.
        When ``None`` (default), all links are returned.
    """
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
    researcher_dev_links_raw = _read_table("researcher_developer_account_link")
    if confidence_threshold is not None:
        researcher_dev_links = (
            researcher_dev_links_raw.filter(
                pl.col("predictive_model_confidence").is_null()
                | (pl.col("predictive_model_confidence") >= confidence_threshold)
            )
            .select("researcher_id", "developer_account_id")
            .unique()
        )
        log.debug(
            "Researcher-developer links after confidence filter (>= %s): %d",
            confidence_threshold,
            researcher_dev_links.height,
        )
    else:
        researcher_dev_links = researcher_dev_links_raw.select(
            "researcher_id", "developer_account_id"
        ).unique()
        log.debug(
            "Researcher-developer links (no confidence filter): %d",
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


def load_pairs(
    sample_size: int | None = None,
    doc_repo_confidence_threshold: float | None = None,
) -> pl.DataFrame:
    """Load document-repository pairs with all relevant metadata.

    Parameters
    ----------
    sample_size
        Optional number of pairs to sample for faster analysis.
    doc_repo_confidence_threshold
        When set, keep only pairs whose ``predictive_model_confidence`` is
        null (seed/shared) or >= this value.  When ``None`` (default), all
        pairs are returned regardless of confidence.
    """
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

    # Optionally drop predicted doc-repo pairs below confidence threshold
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

    # Keep one canonical pair per document and repository for RQ1 analyses.
    log.debug("Deduplicating pairs to get one canonical pair per document and repository...")
    pairs = pairs.unique(
        subset="document_id",
        keep="none",
    ).unique(
        subset="repository_id",
        keep="none",
    )
    log.info("Unique canonical pairs: %d", pairs.height)

    if sample_size is not None:
        log.debug("Sampling %d pairs...", sample_size)
        sampled_doc_ids = (
            pairs.select("document_id")
            .unique()
            .sample(n=min(sample_size, pairs.height), seed=42)
        )
        pairs = pairs.filter(pl.col("document_id").is_in(sampled_doc_ids["document_id"]))

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
            pl.when(pl.col("country_code").n_unique() == 1)
            .then(pl.col("country_code").first())
            .otherwise(pl.lit("Multiple"))
            .alias("country_code"),
        )
    )

    doc_author_countries = (
        doc_author_country_rows.group_by("document_id")
        .agg(pl.len().alias("document_n_authors"))
        .join(doc_author_country_entropy, on="document_id", how="left")
        .with_columns(
            pl.col("document_author_country_entropy").fill_null(0.0),
            pl.col("document_n_unique_author_countries").fill_null(0),
            pl.when(pl.col("country_code").is_null())
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("country_code"))
            .alias("country_code"),
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

    repo_language_counts = _read_sql("""
        SELECT
            repository_id,
            COUNT(*) AS repository_n_languages
        FROM repository_language
        GROUP BY repository_id
    """)

    result = (
        pairs.select(
            "document_id",
            "repository_id",
            "dataset_source_id",
            "iteration",
            pl.col("predictive_model_confidence").alias("document_repository_link_confidence"),
        )
        .join(
            docs.select(
                pl.col("id").alias("document_id"),
                pl.col("title").alias("document_title"),
                pl.col("doi").alias("document_doi"),
                pl.col("cited_by_count").alias("document_cited_by_count"),
                pl.col("fwci").alias("document_fwci"),
                pl.col("is_open_access").alias("document_is_open_access"),
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
                pl.col("open_issues_count").alias("repository_open_issues_count"),
                pl.col("watchers_count").alias("repository_watchers_count"),
                pl.col("is_fork").alias("repository_is_fork"),
                pl.col("license").alias("repository_license"),
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
        .join(repo_language_counts, on="repository_id", how="left")
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

    log.debug("Loaded %d pairs", result.height)
    return result


###############################################################################
# Analysis / plotting functions
###############################################################################


def print_descriptive_stats(pairs: pl.DataFrame, results_dir: Path) -> None:
    """Print field value counts and save a descriptive-statistics CSV."""
    field_data = pairs.filter(
        pl.col("document_field_name").is_not_null(),
        pl.col("document_field_name") != "",
    )
    log.debug(
        "Field value counts:\n%s",
        field_data["document_field_name"].value_counts(sort=True),
    )

    metrics = [
        "document_cited_by_count",
        "document_fwci",
        "document_n_authors",
        "document_n_unique_author_countries",
        "document_author_country_entropy",
        "repository_stargazers_count",
        "repository_fwsi",
        "repository_commits_count",
        "repository_n_contributors",
        "repository_n_files",
        "repository_n_languages",
        "repository_size_kb",
        "repository_commit_duration_days",
        "days_from_repo_creation_to_publication",
        "days_from_publication_to_last_push",
    ]
    metrics = [m for m in metrics if m in pairs.columns]

    stats_df = (
        pairs[metrics]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .filter(pl.col("statistic").is_in(["mean", "std", "25%", "50%", "75%", "min", "max"]))
        .transpose(include_header=True, header_name="metric", column_names="statistic")
    )

    stats_df.write_csv(results_dir / "descriptive_stats.csv")
    log.debug("Saved descriptive_stats.csv")


def plot_field_countplot(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Horizontal countplot of document fields (top-N + Other)."""
    field_data = pairs.filter(
        pl.col("document_field_name").is_not_null(),
        pl.col("document_field_name") != "",
    )
    field_data, top_fields, field_col = _add_top_n_other_column(
        field_data, "document_field_name", top_n
    )
    field_order = [*top_fields, "Other"]

    (
        field_data[field_col]
        .value_counts(sort=True)
        .rename({field_col: "field", "count": "pair_count"})
        .write_csv(results_dir / "field_countplot_data.csv")
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.countplot(
        data=field_data,
        y=field_col,
        order=field_order,
        hue=field_col,
        hue_order=field_order,
        legend=False,
        ax=ax,
    )
    ax.set_title(
        f"Document Count by Academic Field (Top {top_n} + Other)\n"
        "(Each pair assigned its top-scoring OpenAlex topic field)",
        fontsize=13,
    )
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    fig.savefig(results_dir / "field_countplot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_features_by_field(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """4x2 boxplots of key features by top-N fields + Other."""
    pairs, top_fields, field_col = _add_top_n_other_column(pairs, "document_field_name", top_n)
    field_hue_order = [*top_fields, "Other"]

    features_to_plot = list(FEATURE_NAME_TO_VIZ_NAME_LUT.keys())

    features_melted = pairs.select(
        "document_id",
        "repository_id",
        field_col,
        *features_to_plot,
    ).unpivot(
        on=features_to_plot,
        variable_name="feature",
        value_name="value",
        index=["document_id", "repository_id", field_col],
    )

    sort_map = {name: i for i, name in enumerate(FEATURE_NAME_TO_VIZ_NAME_LUT)}
    features_melted = features_melted.with_columns(
        pl.col("feature")
        .replace_strict(sort_map, default=None, return_dtype=pl.Int32)
        .alias("feature_sort_order")
    )
    features_melted = features_melted.sort("feature_sort_order").drop("feature_sort_order")

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(20, 12),
        constrained_layout=True,
    )
    fig.suptitle(
        f"Key Document and Repository Features by Academic Field (Top {top_n} + Other)\n"
        "Boxplots show median and IQR; outliers removed for clarity",
        fontsize=16,
    )
    for ax, ((feature_name,), group_df) in zip(
        axes.flat, features_melted.group_by("feature", maintain_order=True), strict=True
    ):
        y_order = (
            group_df.group_by(field_col)
            .agg(
                pl.col("value").median().alias("median_value"),
                pl.col("value").mean().alias("mean_value"),
            )
            .sort(
                ["median_value", "mean_value", field_col],
                descending=True,
            )[field_col]
            .to_list()
        )

        sns.boxplot(
            data=group_df,
            y=field_col,
            x="value",
            hue=field_col,
            hue_order=field_hue_order,
            order=y_order,
            ax=ax,
            showfliers=False,
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(FEATURE_NAME_TO_VIZ_NAME_LUT.get(feature_name, feature_name), fontsize=16)

    fig.savefig(results_dir / "features_by_field_boxplots.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_pairs_over_time(pairs: pl.DataFrame, results_dir: Path) -> None:
    """Countplot of pairs by publication year."""
    plot_data = pairs.filter(
        pl.col("document_publication_year") > 2010,
        pl.col("document_publication_year") < 2025,
    ).with_columns(pl.col("document_publication_year").cast(pl.Int32))

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.countplot(
        data=plot_data,
        x="document_publication_year",
        hue="document_publication_year",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title(
        "Document-Repository Pairs by Publication Year\n"
        "(Number of unique article-repository pairs published each year)",
        fontsize=14,
    )
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Number of Pairs")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(results_dir / "pairs_over_time.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_pairs_over_time_by_field(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Line plot of pairs by publication year, split by academic field."""
    plot_data = pairs.filter(
        pl.col("document_publication_year") > 2010,
        pl.col("document_publication_year") < 2025,
    ).with_columns(pl.col("document_publication_year").cast(pl.Int32))

    plot_data, top_fields, field_col = _add_top_n_other_column(
        plot_data, "document_field_name", top_n
    )
    field_hue_order = [*top_fields, "Other"]

    field_year_counts = (
        plot_data.group_by([field_col, "document_publication_year"])
        .agg(pl.len().alias("count"))
        .sort("document_publication_year")
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=field_year_counts,
        x="document_publication_year",
        y="count",
        hue=field_col,
        hue_order=field_hue_order,
        marker="o",
        ax=ax,
    )
    ax.set_title(
        f"Document-Repository Pairs Over Time by Academic Field (Top {top_n} + Other)\n"
        "(Each line shows the number of pairs published per year for a given field)",
        fontsize=13,
    )
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Number of Pairs")
    ax.legend(title="Field", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.savefig(results_dir / "pairs_over_time_by_field.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_iteration_expansion(results_dir: Path) -> None:
    """Plot growth by mining iteration for pairs and author-developer pairs.

    Tracks both **all** pairs and **high-confidence** subsets:
    - Article-repo high-conf: ``predictive_model_confidence`` is null or >= 0.9994
    - Author-developer high-conf: ``predictive_model_confidence`` is null or >= 0.97
    """
    links = _read_table("document_repository_link")
    dataset_sources = _read_table("dataset_source")

    links = links.join(
        dataset_sources.select(
            pl.col("id").alias("dataset_source_id"),
            pl.col("name").alias("dataset_source_name"),
        ),
        on="dataset_source_id",
        how="left",
    )

    _is_high_conf_doc_repo = pl.col("predictive_model_confidence").is_null() | (
        pl.col("predictive_model_confidence") >= 0.9994
    )

    shared_links = links.filter(pl.col("dataset_source_name") != "snowball-sampling-discovery")
    mined_links = links.filter(
        pl.col("dataset_source_name") == "snowball-sampling-discovery",
        pl.col("iteration").is_not_null(),
    )

    # --- Article-repository pair counts (all + high-conf) -------------------
    shared_pair_count = shared_links.select("document_id", "repository_id").unique().height
    shared_pair_count_high_conf = (
        shared_links.filter(_is_high_conf_doc_repo)
        .select("document_id", "repository_id")
        .unique()
        .height
    )

    mined_pairs_by_iteration = (
        mined_links.select("document_id", "repository_id", "iteration")
        .unique()
        .group_by("iteration")
        .agg(pl.len().alias("new_article_repository_pairs"))
        .sort("iteration")
    )
    mined_pairs_high_conf_by_iteration = (
        mined_links.filter(_is_high_conf_doc_repo)
        .select("document_id", "repository_id", "iteration")
        .unique()
        .group_by("iteration")
        .agg(pl.len().alias("new_article_repository_pairs_high_conf"))
        .sort("iteration")
    )

    # --- Author-developer pair counts (all + high-conf) ---------------------
    shared_unique_pairs = shared_links.select("document_id", "repository_id").unique()
    shared_author_dev_count = (
        _get_author_developer_pairs_connected_to_pairs(shared_unique_pairs)
        .select("researcher_id", "developer_account_id")
        .unique()
        .height
    )
    shared_author_dev_count_high_conf = (
        _get_author_developer_pairs_connected_to_pairs(
            shared_unique_pairs,
            confidence_threshold=0.97,
        )
        .select("researcher_id", "developer_account_id")
        .unique()
        .height
    )

    mined_unique_pairs = mined_links.select(
        "document_id", "repository_id", "iteration"
    ).unique()
    mined_author_dev_links = _get_author_developer_pairs_connected_to_pairs(mined_unique_pairs)
    mined_author_dev_links_high_conf = _get_author_developer_pairs_connected_to_pairs(
        mined_unique_pairs,
        confidence_threshold=0.97,
    )

    new_author_dev_by_iteration = (
        mined_author_dev_links.group_by("researcher_id", "developer_account_id")
        .agg(pl.col("iteration").min().alias("iteration"))
        .group_by("iteration")
        .agg(pl.len().alias("new_author_developer_pairs"))
        .sort("iteration")
    )
    new_author_dev_high_conf_by_iteration = (
        mined_author_dev_links_high_conf.group_by("researcher_id", "developer_account_id")
        .agg(pl.col("iteration").min().alias("iteration"))
        .group_by("iteration")
        .agg(pl.len().alias("new_author_developer_pairs_high_conf"))
        .sort("iteration")
    )

    # --- Early return when no mined data ------------------------------------
    if mined_pairs_by_iteration.height == 0 and new_author_dev_by_iteration.height == 0:
        log.warning("No mined iterations found. Skipping iteration expansion plots.")
        summary = {
            "shared_seed_article_repository_pairs": int(shared_pair_count),
            "shared_seed_article_repository_pairs_high_conf": int(shared_pair_count_high_conf),
            "shared_seed_author_developer_pairs": int(shared_author_dev_count),
            "shared_seed_author_developer_pairs_high_conf": int(
                shared_author_dev_count_high_conf
            ),
            "iterations_found": 0,
        }
        with open(results_dir / "iteration_expansion_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return

    # --- Build growth DataFrame ---------------------------------------------
    iterations = (
        pl.concat(
            [
                mined_pairs_by_iteration.select("iteration"),
                mined_pairs_high_conf_by_iteration.select("iteration"),
                new_author_dev_by_iteration.select("iteration"),
                new_author_dev_high_conf_by_iteration.select("iteration"),
            ]
        )
        .unique()
        .sort("iteration")
    )

    growth_df = (
        iterations.join(mined_pairs_by_iteration, on="iteration", how="left")
        .join(mined_pairs_high_conf_by_iteration, on="iteration", how="left")
        .join(new_author_dev_by_iteration, on="iteration", how="left")
        .join(new_author_dev_high_conf_by_iteration, on="iteration", how="left")
        .with_columns(
            pl.col("new_article_repository_pairs").fill_null(0).cast(pl.Int64),
            pl.col("new_article_repository_pairs_high_conf").fill_null(0).cast(pl.Int64),
            pl.col("new_author_developer_pairs").fill_null(0).cast(pl.Int64),
            pl.col("new_author_developer_pairs_high_conf").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            (
                pl.col("new_article_repository_pairs").cum_sum() + pl.lit(shared_pair_count)
            ).alias("cumulative_article_repository_pairs"),
            (
                pl.col("new_article_repository_pairs_high_conf").cum_sum()
                + pl.lit(shared_pair_count_high_conf)
            ).alias("cumulative_article_repository_pairs_high_conf"),
            (
                pl.col("new_author_developer_pairs").cum_sum() + pl.lit(shared_author_dev_count)
            ).alias("cumulative_author_developer_pairs"),
            (
                pl.col("new_author_developer_pairs_high_conf").cum_sum()
                + pl.lit(shared_author_dev_count_high_conf)
            ).alias("cumulative_author_developer_pairs_high_conf"),
        )
        .sort("iteration")
    )

    seed_row = pl.DataFrame(
        {
            "iteration": [0],
            "new_article_repository_pairs": [0],
            "new_article_repository_pairs_high_conf": [0],
            "new_author_developer_pairs": [0],
            "new_author_developer_pairs_high_conf": [0],
            "cumulative_article_repository_pairs": [shared_pair_count],
            "cumulative_article_repository_pairs_high_conf": [shared_pair_count_high_conf],
            "cumulative_author_developer_pairs": [shared_author_dev_count],
            "cumulative_author_developer_pairs_high_conf": [shared_author_dev_count_high_conf],
        }
    )
    growth_with_seed_df = pl.concat([seed_row, growth_df], how="vertical").sort("iteration")

    growth_df.write_csv(results_dir / "iteration_expansion_growth.csv")
    growth_with_seed_df.write_csv(results_dir / "iteration_expansion_growth_with_seed.csv")

    summary = {
        "shared_seed_article_repository_pairs": int(shared_pair_count),
        "shared_seed_article_repository_pairs_high_conf": int(shared_pair_count_high_conf),
        "shared_seed_author_developer_pairs": int(shared_author_dev_count),
        "shared_seed_author_developer_pairs_high_conf": int(shared_author_dev_count_high_conf),
        "iterations_found": int(growth_df.height),
        "final_cumulative_article_repository_pairs": int(
            growth_with_seed_df["cumulative_article_repository_pairs"].max()
        ),
        "final_cumulative_article_repository_pairs_high_conf": int(
            growth_with_seed_df["cumulative_article_repository_pairs_high_conf"].max()
        ),
        "final_cumulative_author_developer_pairs": int(
            growth_with_seed_df["cumulative_author_developer_pairs"].max()
        ),
        "final_cumulative_author_developer_pairs_high_conf": int(
            growth_with_seed_df["cumulative_author_developer_pairs_high_conf"].max()
        ),
    }
    with open(results_dir / "iteration_expansion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Cast iteration to int to avoid categorical-units warnings
    growth_df = growth_df.with_columns(pl.col("iteration").cast(pl.Int32))
    growth_with_seed_df = growth_with_seed_df.with_columns(pl.col("iteration").cast(pl.Int32))

    # --- Plotting -----------------------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), constrained_layout=True)

    # Left panel: article-repository pairs
    sns.barplot(
        data=growth_df,
        x="iteration",
        y="new_article_repository_pairs",
        ax=axes[0],
        color=PALETTE[0],
        alpha=0.4,
        label="All (new)",
    )
    sns.barplot(
        data=growth_df,
        x="iteration",
        y="new_article_repository_pairs_high_conf",
        ax=axes[0],
        color=PALETTE[0],
        alpha=1.0,
        label="High-conf (new)",
    )
    ax0_twin = axes[0].twinx()
    sns.lineplot(
        data=growth_with_seed_df,
        x="iteration",
        y="cumulative_article_repository_pairs",
        marker="o",
        ax=ax0_twin,
        color=PALETTE[1],
        linestyle="-",
        label="All (cumulative)",
    )
    sns.lineplot(
        data=growth_with_seed_df,
        x="iteration",
        y="cumulative_article_repository_pairs_high_conf",
        marker="s",
        ax=ax0_twin,
        color=PALETTE[1],
        linestyle="--",
        label="High-conf (cumulative)",
    )
    axes[0].set_xlabel("Mining Iteration")
    axes[0].set_ylabel("New Article-Repository Pairs")
    ax0_twin.set_ylabel("Cumulative Article-Repository Pairs")
    axes[0].set_title(
        "Article-Repository Pair Expansion\n"
        "(Bars = new pairs; lines = cumulative; high-conf = null or >= 0.9994)",
        fontsize=12,
    )
    h0a, l0a = axes[0].get_legend_handles_labels()
    h0b, l0b = ax0_twin.get_legend_handles_labels()
    axes[0].legend(h0a + h0b, l0a + l0b, loc="upper left", fontsize=8)
    ax0_twin.get_legend().remove()

    # Right panel: author-developer pairs
    sns.barplot(
        data=growth_df,
        x="iteration",
        y="new_author_developer_pairs",
        ax=axes[1],
        color=PALETTE[2],
        alpha=0.4,
        label="All (new)",
    )
    sns.barplot(
        data=growth_df,
        x="iteration",
        y="new_author_developer_pairs_high_conf",
        ax=axes[1],
        color=PALETTE[2],
        alpha=1.0,
        label="High-conf (new)",
    )
    ax1_twin = axes[1].twinx()
    sns.lineplot(
        data=growth_with_seed_df,
        x="iteration",
        y="cumulative_author_developer_pairs",
        marker="o",
        ax=ax1_twin,
        color=PALETTE[3],
        linestyle="-",
        label="All (cumulative)",
    )
    sns.lineplot(
        data=growth_with_seed_df,
        x="iteration",
        y="cumulative_author_developer_pairs_high_conf",
        marker="s",
        ax=ax1_twin,
        color=PALETTE[3],
        linestyle="--",
        label="High-conf (cumulative)",
    )
    axes[1].set_xlabel("Mining Iteration")
    axes[1].set_ylabel("New Author-Developer Pairs")
    ax1_twin.set_ylabel("Cumulative Author-Developer Pairs")
    axes[1].set_title(
        "Author-Developer Pair Expansion\n"
        "(Bars = new pairs; lines = cumulative; high-conf = null or >= 0.97)",
        fontsize=12,
    )
    h1a, l1a = axes[1].get_legend_handles_labels()
    h1b, l1b = ax1_twin.get_legend_handles_labels()
    axes[1].legend(h1a + h1b, l1a + l1b, loc="upper left", fontsize=8)
    ax1_twin.get_legend().remove()

    fig.savefig(results_dir / "iteration_expansion.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_field_and_language_counts(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """2x2 grid: field/language counts and counts over time."""
    pairs, top_fields, field_col = _add_top_n_other_column(pairs, "document_field_name", top_n)
    field_hue_order = [*top_fields, "Other"]

    pairs, top_langs, lang_col = _add_top_n_other_column(
        pairs, "repository_primary_language", top_n
    )
    lang_hue_order = [*top_langs, "Other"]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(20, 10),
        constrained_layout=True,
    )

    sns.countplot(
        data=pairs,
        x=field_col,
        order=pairs[field_col].value_counts(sort=True)[field_col],
        hue=field_col,
        hue_order=field_hue_order,
        ax=axes[0, 0],
    )

    field_over_time = (
        pairs.filter(
            pl.col("document_publication_year") > 2010,
            pl.col("document_publication_year") < 2025,
        )
        .with_columns(pl.col("document_publication_year").cast(pl.Int32))
        .group_by([field_col, "document_publication_year"])
        .agg(pl.len().alias("field_count"))
    )
    sns.lineplot(
        data=field_over_time,
        x="document_publication_year",
        y="field_count",
        hue=field_col,
        hue_order=field_hue_order,
        ax=axes[0, 1],
        legend=False,
    )

    sns.countplot(
        data=pairs,
        x=lang_col,
        order=pairs[lang_col].value_counts(sort=True)[lang_col],
        hue=lang_col,
        hue_order=lang_hue_order,
        ax=axes[1, 0],
    )

    lang_over_time = (
        pairs.filter(
            pl.col("document_publication_year") > 2010,
            pl.col("document_publication_year") < 2025,
        )
        .with_columns(pl.col("document_publication_year").cast(pl.Int32))
        .group_by([lang_col, "document_publication_year"])
        .agg(pl.len().alias("language_count"))
    )
    sns.lineplot(
        data=lang_over_time,
        x="document_publication_year",
        y="language_count",
        hue=lang_col,
        hue_order=lang_hue_order,
        ax=axes[1, 1],
        legend=False,
    )

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45)
        feature_name = ax.get_xlabel()
        feature_name = (
            ax.get_ylabel() if feature_name == "document_publication_year" else feature_name
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(FIELD_LANGUAGE_TITLE_LUT.get(feature_name, feature_name), fontsize=16)

    (
        pairs[lang_col]
        .value_counts(sort=True)
        .rename({lang_col: "language", "count": "pair_count"})
        .write_csv(results_dir / "language_counts.csv")
    )

    (
        pairs.group_by([field_col, lang_col])
        .agg(pl.len().alias("pair_count"))
        .sort([field_col, "pair_count"], descending=[False, True])
        .rename({field_col: "field", lang_col: "language"})
        .write_csv(results_dir / "language_counts_by_field.csv")
    )

    fig.savefig(results_dir / "field_and_language_counts.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_repo_metrics_over_time(pairs: pl.DataFrame, results_dir: Path) -> None:
    """Bar charts of repository metrics over time using seaborn's default mean estimator."""
    over_time_metrics = list(OVER_TIME_METRICS_TO_VIZ_NAME_LUT.keys())

    over_time_df = (
        pairs.filter(
            pl.col("document_publication_year") < 2025,
            pl.col("document_publication_year") > 2010,
        )
        .with_columns(pl.col("document_publication_year").cast(pl.Int32))
        .unpivot(
            on=over_time_metrics,
            variable_name="metric",
            value_name="mean_value",
            index=["document_publication_year"],
        )
    )

    g = sns.catplot(
        data=over_time_df,
        x="document_publication_year",
        y="mean_value",
        hue="metric",
        col="metric",
        col_wrap=2,
        kind="bar",
        sharey=False,
        legend=False,
    )

    g.set_titles("{col_name}")
    for i, ax in enumerate(g.axes):
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("")
        ax.set_ylabel("Mean Value (with 95% CI)")
        ax.set_title(OVER_TIME_METRICS_TO_VIZ_NAME_LUT.get(over_time_metrics[i]), fontsize=16)
    g.figure.savefig(results_dir / "repo_metrics_over_time.png", bbox_inches="tight", dpi=300)
    plt.close(g.figure)


def plot_geographic_diversity_depth(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Analyze geographic diversity depth using unique-country counts and entropy."""
    geo_df = (
        pairs.select(
            "document_id",
            "document_field_name",
            "document_publication_year",
            "document_n_unique_author_countries",
            "document_author_country_entropy",
        )
        .filter(
            pl.col("document_n_unique_author_countries").is_not_null(),
            pl.col("document_author_country_entropy").is_not_null(),
            pl.col("document_publication_year") > 2010,
            pl.col("document_publication_year") < 2025,
        )
        .with_columns(pl.col("document_publication_year").cast(pl.Int32))
    )

    if geo_df.height == 0:
        log.warning(
            "No geographic diversity data available. Skipping geographic diversity plots."
        )
        return

    geo_df, top_fields, field_col = _add_top_n_other_column(
        geo_df, "document_field_name", top_n
    )

    geo_df = geo_df.with_columns(
        pl.when(pl.col("document_n_unique_author_countries") >= 10)
        .then(pl.lit("10+"))
        .otherwise(pl.col("document_n_unique_author_countries").cast(pl.Int64).cast(pl.Utf8))
        .alias("author_country_count_bin")
    )

    bin_order = [
        b
        for b in [*map(str, range(1, 10)), "10+"]
        if b in geo_df["author_country_count_bin"].unique().to_list()
    ]

    summary_df = (
        geo_df.group_by(field_col)
        .agg(
            pl.len().alias("n_pairs"),
            pl.col("document_n_unique_author_countries").mean().alias("mean_unique_countries"),
            pl.col("document_n_unique_author_countries")
            .median()
            .alias("median_unique_countries"),
            pl.col("document_author_country_entropy").mean().alias("mean_country_entropy"),
            pl.col("document_author_country_entropy").median().alias("median_country_entropy"),
        )
        .sort("n_pairs", descending=True)
    )
    summary_df.write_csv(results_dir / "geographic_diversity_summary.csv")

    yearly_geo_df = (
        geo_df.group_by("document_publication_year")
        .agg(
            pl.col("document_n_unique_author_countries").mean().alias("mean_unique_countries"),
            pl.col("document_author_country_entropy").mean().alias("mean_country_entropy"),
        )
        .sort("document_publication_year")
    )

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10), constrained_layout=True)

    sns.countplot(
        data=geo_df,
        x="author_country_count_bin",
        order=bin_order,
        hue="author_country_count_bin",
        hue_order=bin_order,
        palette="viridis",
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Unique Author-Team Countries per Pair")
    axes[0, 0].set_xlabel("Unique Countries in Author Team")
    axes[0, 0].set_ylabel("Pair Count")

    field_order = (
        geo_df.group_by(field_col)
        .agg(pl.col("document_author_country_entropy").median().alias("med_entropy"))
        .sort("med_entropy", descending=True)[field_col]
        .to_list()
    )
    sns.boxplot(
        data=geo_df,
        y=field_col,
        x="document_author_country_entropy",
        order=field_order,
        showfliers=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(
        "Author-Team Country Entropy by Field\n"
        "(0 = single country, higher = more geographically diverse)",
        fontsize=12,
    )
    axes[0, 1].set_xlabel("Country Entropy")
    axes[0, 1].set_ylabel("")

    sns.lineplot(
        data=yearly_geo_df,
        x="document_publication_year",
        y="mean_unique_countries",
        marker="o",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Mean Unique Author-Team Countries Over Time")
    axes[1, 0].set_xlabel("Publication Year")
    axes[1, 0].set_ylabel("Mean Unique Countries")

    sns.lineplot(
        data=yearly_geo_df,
        x="document_publication_year",
        y="mean_country_entropy",
        marker="o",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(
        "Mean Author-Team Country Entropy Over Time\n"
        "(0 = all authors from one country, higher = more diverse)",
        fontsize=12,
    )
    axes[1, 1].set_xlabel("Publication Year")
    axes[1, 1].set_ylabel("Mean Country Entropy")

    for ax in axes[1, :]:
        ax.tick_params(axis="x", rotation=45)

    fig.savefig(results_dir / "geographic_diversity_depth.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_fwci_vs_fwsi(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """FWCI-vs-FWSI scatter, labeling, distributions, and relationship statistics."""
    fwci_fwsi_plot_data = _filter_finite(
        pairs,
        ["document_fwci", "repository_fwsi"],
    ).with_columns(
        (pl.lit(1) + pl.col("document_fwci")).log10().alias("document_fwci_log10"),
        (pl.lit(1) + pl.col("repository_fwsi")).log10().alias("repository_fwsi_log10"),
    )
    fwci_fwsi_plot_data = _filter_finite(
        fwci_fwsi_plot_data,
        ["document_fwci_log10", "repository_fwsi_log10"],
    )

    if fwci_fwsi_plot_data.height == 0:
        log.warning("No finite FWCI/FWSI rows available. Skipping FWCI/FWSI plots.")
        return

    x = fwci_fwsi_plot_data["document_fwci_log10"].to_numpy()
    y = fwci_fwsi_plot_data["repository_fwsi_log10"].to_numpy()

    spearman_rho = _compute_spearman_rho(x, y)
    pearson_r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
    regression = _fit_simple_linear_regression(x, y)

    relationship_stats = {
        "n_pairs": int(fwci_fwsi_plot_data.height),
        "spearman_rho": float(spearman_rho),
        "pearson_r": float(pearson_r),
        "slope": float(regression["slope"]),
        "intercept": float(regression["intercept"]),
        "r_squared": float(regression["r_squared"]),
    }
    with open(results_dir / "fwci_fwsi_relationship_stats.json", "w") as f:
        json.dump(relationship_stats, f, indent=2)

    fwci_fwsi_with_fields, top_fields, field_col = _add_top_n_other_column(
        fwci_fwsi_plot_data,
        "document_field_name",
        top_n,
    )

    field_relationship_rows: list[dict[str, float | int | str]] = []
    for field_name in top_fields:
        field_df = fwci_fwsi_with_fields.filter(pl.col(field_col) == field_name)
        if field_df.height < 25:
            continue

        field_x = field_df["document_fwci_log10"].to_numpy()
        field_y = field_df["repository_fwsi_log10"].to_numpy()
        field_reg = _fit_simple_linear_regression(field_x, field_y)

        field_relationship_rows.append(
            {
                "field": field_name,
                "n_pairs": int(field_df.height),
                "spearman_rho": float(_compute_spearman_rho(field_x, field_y)),
                "pearson_r": float(np.corrcoef(field_x, field_y)[0, 1])
                if len(field_x) > 1
                else float("nan"),
                "slope": float(field_reg["slope"]),
                "intercept": float(field_reg["intercept"]),
                "r_squared": float(field_reg["r_squared"]),
            }
        )

    if field_relationship_rows:
        pl.DataFrame(field_relationship_rows).sort("n_pairs", descending=True).write_csv(
            results_dir / "fwci_fwsi_field_relationship_stats.csv"
        )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=fwci_fwsi_plot_data,
        x="document_fwci_log10",
        y="repository_fwsi_log10",
        alpha=0.1,
        ax=ax,
    )

    if np.isfinite(regression["slope"]):
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        y_line = regression["intercept"] + regression["slope"] * x_line
        ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=2)

    fwci_fwsi_plot_data = fwci_fwsi_plot_data.with_columns(
        (pl.col("document_fwci_log10") * pl.col("repository_fwsi_log10")).alias(
            "fwci_fwsi_product_log10"
        )
    )

    labeled_frames: list[pl.DataFrame] = [
        fwci_fwsi_plot_data.sort("document_fwci_log10", descending=True)[0:4:2],
        fwci_fwsi_plot_data.sort("repository_fwsi_log10", descending=True)[9],
        fwci_fwsi_plot_data.sort("fwci_fwsi_product_log10", descending=True)[6],
        fwci_fwsi_plot_data.with_columns(
            (
                (pl.col("document_fwci_log10") - 1.5).abs()
                + (pl.col("repository_fwsi_log10") - 1.5).abs()
            ).alias("distance_to_1_5_1_5")
        )
        .sort("distance_to_1_5_1_5")
        .head(1),
        fwci_fwsi_plot_data.with_columns(
            (
                (pl.col("document_fwci_log10") - 2.0).abs()
                + (pl.col("repository_fwsi_log10") - 1.0).abs()
            ).alias("distance_to_2_0_1_0")
        )
        .sort("distance_to_2_0_1_0")
        .head(1),
        fwci_fwsi_plot_data.with_columns(
            (
                (pl.col("document_fwci_log10") - 0.95).abs()
                + (pl.col("repository_fwsi_log10") - 2.0).abs()
            ).alias("distance_to_1_0_2_0")
        )
        .sort("distance_to_1_0_2_0")
        .head(1),
    ]
    for rows_df in labeled_frames:
        for row in rows_df.iter_rows(named=True):
            _plot_fwci_fwsi_point_data(row, "black", ax)

    ax.set_title(
        "Field-Weighted Citation Impact (FWCI) vs Field-Weighted Star Impact (FWSI)\n"
        f"log10(1+value); Spearman rho={spearman_rho:.3f}, R^2={regression['r_squared']:.3f}",
        fontsize=13,
    )
    ax.set_xlabel("Document FWCI (log10)")
    ax.set_ylabel("Repository FWSI (log10)")
    ax.set_xlim(right=ax.get_xlim()[1] + 1.5)
    sns.despine(ax=ax)

    fig.savefig(results_dir / "fwci_vs_fwsi_scatter.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fwci_fwsi_plot_data.write_parquet(results_dir / "fwci_fwsi_plot_data.parquet")

    labeled_cols = [
        "document_title",
        "document_doi",
        "repository_owner",
        "repository_name",
        "document_fwci_log10",
        "repository_fwsi_log10",
    ]
    all_labeled = pl.concat(
        [f.select([c for c in labeled_cols if c in f.columns]) for f in labeled_frames]
    ).unique()
    all_labeled.write_csv(results_dir / "fwci_fwsi_labeled_points.csv")
    log.debug("Saved fwci_fwsi_labeled_points.csv")

    fwci_fwsi_melted = fwci_fwsi_plot_data.select(
        "document_id",
        "repository_id",
        "document_fwci_log10",
        "repository_fwsi_log10",
    ).unpivot(
        on=["document_fwci_log10", "repository_fwsi_log10"],
        index=["document_id", "repository_id"],
        variable_name="metric",
        value_name="log10_value",
    )

    g = sns.displot(
        data=fwci_fwsi_melted,
        x="log10_value",
        hue="metric",
        col="metric",
        bins=20,
        stat="proportion",
        legend=False,
    )
    g.set_titles("{col_name}")
    g.figure.savefig(results_dir / "fwci_fwsi_distributions.png", bbox_inches="tight", dpi=300)
    plt.close(g.figure)


def plot_date_relationships(pairs: pl.DataFrame, results_dir: Path) -> None:
    """Scatter plots of date relationships and day-difference distributions."""
    date_df = pairs.select(
        "document_publication_date",
        "repository_creation_datetime",
        "repository_last_pushed_datetime",
    ).filter(
        pl.col("document_publication_date").dt.year() < 2025,
        pl.col("document_publication_date").dt.year() > 2010,
        pl.col("repository_creation_datetime").dt.year() < 2025,
        pl.col("repository_creation_datetime").dt.year() > 2010,
        pl.col("repository_last_pushed_datetime").dt.year() > 2010,
    )

    if date_df.height == 0:
        log.warning("No date-relationship rows available. Skipping date relationship plots.")
        return

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), constrained_layout=True)

    sns.scatterplot(
        data=date_df,
        x="document_publication_date",
        y="repository_creation_datetime",
        alpha=0.1,
        ax=axes[0],
    )
    _add_45_degree_line(
        axes[0],
        date_df["document_publication_date"],
        date_df["repository_creation_datetime"],
    )

    sns.scatterplot(
        data=date_df,
        x="repository_last_pushed_datetime",
        y="document_publication_date",
        alpha=0.1,
        ax=axes[1],
    )
    axes[1].invert_yaxis()
    _add_45_degree_line(
        axes[1],
        date_df["repository_last_pushed_datetime"],
        date_df["document_publication_date"],
    )

    for ax in axes.flat:
        x_label = ax.get_xlabel()
        y_label = ax.get_ylabel()
        ax.set_xlabel(DATE_FEATURES_TO_VIZ_NAME_LUT.get(x_label, x_label))
        ax.set_ylabel(DATE_FEATURES_TO_VIZ_NAME_LUT.get(y_label, y_label))

    axes[0].set_title(
        "Publication Date vs Repository Creation Date\n"
        "(Below red line = repo created before publication;\n"
        "above = repo created after publication)",
        fontsize=11,
    )
    axes[1].set_title(
        "Last Push vs Publication Date\n"
        "(Above red line = last push after publication;\n"
        "below = last push before publication)",
        fontsize=11,
    )

    fig.savefig(results_dir / "date_relationships_scatter.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    date_days_df = (
        date_df.with_columns(
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
        .select(
            "days_from_repo_creation_to_publication",
            "days_from_publication_to_last_push",
        )
        .unpivot(
            on=[
                "days_from_repo_creation_to_publication",
                "days_from_publication_to_last_push",
            ],
            index=[],
            variable_name="date_difference_type",
            value_name="days_difference",
        )
    )

    if date_days_df.height == 0:
        return

    g = sns.displot(
        data=date_days_df.filter(
            pl.col("days_difference") > date_days_df["days_difference"].quantile(0.01),
            pl.col("days_difference") < date_days_df["days_difference"].quantile(0.99),
        ),
        x="days_difference",
        hue="date_difference_type",
        col="date_difference_type",
        bins=20,
        stat="proportion",
        legend=False,
    )
    g.set_titles("{col_name}")
    g.figure.savefig(
        results_dir / "date_difference_distributions.png", bbox_inches="tight", dpi=300
    )
    plt.close(g.figure)


def plot_lifecycle_and_survival(pairs: pl.DataFrame, results_dir: Path) -> None:
    """Analyze lifecycle archetypes and maintenance survival after publication."""
    lifecycle_df = pairs.select(
        "document_id",
        "repository_id",
        "document_publication_date",
        "repository_creation_datetime",
        "repository_last_pushed_datetime",
        "days_from_repo_creation_to_publication",
        "days_from_publication_to_last_push",
    ).filter(
        pl.col("document_publication_date").is_not_null(),
        pl.col("repository_creation_datetime").is_not_null(),
        pl.col("repository_last_pushed_datetime").is_not_null(),
        pl.col("days_from_repo_creation_to_publication").is_not_null(),
        pl.col("days_from_publication_to_last_push").is_not_null(),
    )

    if lifecycle_df.height == 0:
        log.warning("No lifecycle rows available. Skipping lifecycle/survival analyses.")
        return

    lifecycle_df = lifecycle_df.with_columns(
        pl.when(
            pl.col("days_from_repo_creation_to_publication") > LIFECYCLE_RELEASE_WINDOW_DAYS
        )
        .then(pl.lit(f"Created >{LIFECYCLE_RELEASE_WINDOW_DAYS}d before pub"))
        .when(pl.col("days_from_repo_creation_to_publication") < -LIFECYCLE_RELEASE_WINDOW_DAYS)
        .then(pl.lit(f"Created >{LIFECYCLE_RELEASE_WINDOW_DAYS}d after pub"))
        .otherwise(pl.lit(f"Created within {LIFECYCLE_RELEASE_WINDOW_DAYS}d of pub"))
        .alias("creation_timing"),
        pl.when(pl.col("days_from_publication_to_last_push") <= 0)
        .then(pl.lit("No post-pub maintenance (0d)"))
        .when(pl.col("days_from_publication_to_last_push") <= LIFECYCLE_LONG_MAINTENANCE_DAYS)
        .then(pl.lit(f"Short-term maintenance (1-{LIFECYCLE_LONG_MAINTENANCE_DAYS}d)"))
        .otherwise(pl.lit(f"Long-term maintenance (>{LIFECYCLE_LONG_MAINTENANCE_DAYS}d)"))
        .alias("maintenance_timing"),
    ).with_columns(
        pl.concat_str(["creation_timing", pl.lit(" | "), "maintenance_timing"]).alias(
            "lifecycle_archetype"
        )
    )

    lifecycle_counts = (
        lifecycle_df.group_by("lifecycle_archetype")
        .agg(pl.len().alias("n_pairs"))
        .sort("n_pairs", descending=True)
    )
    lifecycle_counts.write_csv(results_dir / "lifecycle_archetype_counts.csv")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=lifecycle_counts,
        y="lifecycle_archetype",
        x="n_pairs",
        hue="lifecycle_archetype",
        legend=False,
        ax=ax,
    )
    ax.set_title(
        "Repository Lifecycle and Maintenance Archetypes\n"
        f"(Creation timing: within/beyond {LIFECYCLE_RELEASE_WINDOW_DAYS}d of publication; "
        f"maintenance: short-term <= {LIFECYCLE_LONG_MAINTENANCE_DAYS}d, "
        f"long-term > {LIFECYCLE_LONG_MAINTENANCE_DAYS}d)",
        fontsize=12,
    )
    ax.set_xlabel("Pair Count")
    ax.set_ylabel("")
    fig.savefig(results_dir / "lifecycle_archetypes.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    survival_df = lifecycle_df.filter(
        pl.col("days_from_publication_to_last_push") >= 0
    ).with_columns(
        pl.col("days_from_publication_to_last_push").cast(pl.Int64).alias("duration_days")
    )

    if survival_df.height < 2:
        log.warning("Insufficient rows for survival analysis after publication.")
        return

    data_cutoff_date = survival_df["repository_last_pushed_datetime"].max()
    censor_window_days = 30

    survival_df = survival_df.with_columns(
        (pl.lit(data_cutoff_date) - pl.col("repository_last_pushed_datetime"))
        .dt.total_days()
        .alias("days_from_last_push_to_cutoff")
    ).with_columns(
        pl.when(pl.col("days_from_last_push_to_cutoff") <= censor_window_days)
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("event_observed")
    )

    km_frames: list[pl.DataFrame] = []
    km_summary: list[dict[str, float | int | str | None]] = []

    def _append_km(group_name: str, group_df: pl.DataFrame) -> None:
        durations = group_df["duration_days"].to_numpy()
        events = group_df["event_observed"].to_numpy()
        curve = _kaplan_meier_curve(durations, events)
        if curve.height == 0:
            return

        curve = curve.with_columns(pl.lit(group_name).alias("group"))
        km_frames.append(curve)

        median_candidates = curve.filter(pl.col("survival_probability") <= 0.5)
        median_days = (
            int(median_candidates["duration_days"].min())
            if median_candidates.height > 0
            else None
        )
        km_summary.append(
            {
                "group": group_name,
                "n_pairs": int(group_df.height),
                "event_rate": cast(float, group_df["event_observed"].mean()),
                "median_survival_days": median_days,
            }
        )

    _append_km("Overall", survival_df)

    for (creation_timing,), group_df in survival_df.group_by(
        "creation_timing", maintain_order=True
    ):
        if group_df.height >= 50:
            _append_km(str(creation_timing), group_df)

    if not km_frames:
        log.warning("No KM curves produced.")
        return

    km_curve_df = pl.concat(km_frames, how="vertical")
    km_curve_df.write_csv(results_dir / "publication_maintenance_survival_curve.csv")

    with open(results_dir / "publication_maintenance_survival_summary.json", "w") as f:
        json.dump(km_summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    for (group_name,), group_curve in km_curve_df.group_by("group", maintain_order=True):
        x_vals = group_curve["duration_days"].to_list()
        y_vals = group_curve["survival_probability"].to_list()
        ax.step(x_vals, y_vals, where="post", label=str(group_name))

    ax.set_title(
        "Post-Publication Maintenance Survival (Kaplan-Meier)\n"
        "(Probability that a repository is still being maintained N days after publication;\n"
        "curves dropping faster indicate fields/archetypes where maintenance ends sooner)",
        fontsize=12,
    )
    ax.set_xlabel("Days from Publication to Last Push")
    ax.set_ylabel("Survival Probability (Still Maintained)")
    ax.set_ylim(0, 1.01)
    # Cap x-axis to a readable range; most meaningful differences are within a few years
    max_x = min(int(survival_df["duration_days"].quantile(0.95)), 3650)
    ax.set_xlim(0, max_x)
    ax.legend()
    fig.savefig(
        results_dir / "publication_maintenance_survival.png", bbox_inches="tight", dpi=300
    )
    plt.close(fig)


def analyze_network_coverage(
    pairs: pl.DataFrame,
    results_dir: Path,
    n_iterations: int,
) -> None:
    """Build co-authorship graph, compute component stats and shortest paths; save JSON."""
    doc_contribs = _read_table("document_contributor")
    doc_ids_in_dataset = pairs["document_id"].unique().to_list()
    doc_contribs = doc_contribs.filter(pl.col("document_id").is_in(doc_ids_in_dataset))

    graph, _, _ = _build_coauthorship_graph(doc_contribs)

    log.info(
        "Network built: %s nodes, %s edges",
        f"{graph.num_nodes():,}",
        f"{graph.num_edges():,}",
    )

    components = rx.connected_components(graph) if graph.num_nodes() > 0 else []
    component_sizes = sorted([len(c) for c in components], reverse=True)

    total_nodes = graph.num_nodes()
    largest_component_size = component_sizes[0] if component_sizes else 0
    coverage = largest_component_size / total_nodes if total_nodes > 0 else 0

    network_stats = {
        "total_components": len(components),
        "total_nodes": total_nodes,
        "largest_component_size": largest_component_size,
        "coverage_pct": round(coverage * 100, 2),
        "largest_5": component_sizes[:5],
        "isolates_size_1": component_sizes.count(1),
        "size_2_to_10": sum(1 for s in component_sizes if 2 <= s <= 10),
        "size_11_to_100": sum(1 for s in component_sizes if 11 <= s <= 100),
        "size_gt_100": sum(1 for s in component_sizes if s > 100),
    }
    log.info("Network coverage: %.2f%%", coverage * 100)

    with open(results_dir / "network_coverage.json", "w") as f:
        json.dump(network_stats, f, indent=2)
    log.debug("Saved network_coverage.json")

    if not components:
        path_stats = {
            "valid_paths": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
        }
        with open(results_dir / "shortest_path_stats.json", "w") as f:
            json.dump(path_stats, f, indent=2)
        log.warning("No graph components available for shortest-path analysis.")
        return

    largest_cc = max(components, key=len)
    subgraph = graph.subgraph(list(largest_cc))

    log.info(
        "Largest component: %s nodes, %s edges",
        f"{subgraph.num_nodes():,}",
        f"{subgraph.num_edges():,}",
    )

    if subgraph.num_nodes() < 2:
        path_stats = {
            "valid_paths": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
        }
        with open(results_dir / "shortest_path_stats.json", "w") as f:
            json.dump(path_stats, f, indent=2)
        log.warning("Largest component has <2 nodes; skipping shortest-path sampling.")
        return

    subgraph_indices = list(range(subgraph.num_nodes()))
    dijkstra_lengths: list[int] = []
    for _ in tqdm(range(n_iterations), desc="Getting random shortest paths"):
        source, target = random.sample(subgraph_indices, 2)
        dijkstra_res = rx.dijkstra_shortest_path_lengths(
            subgraph,
            source,
            lambda _: 1,
            goal=target,
        )
        dijkstra_lengths.append(int(dijkstra_res[target]))

    dijkstra_vec = np.array(dijkstra_lengths)
    path_stats = {
        "valid_paths": len(dijkstra_vec),
        "mean": round(float(np.mean(dijkstra_vec)), 4),
        "std": round(float(np.std(dijkstra_vec)), 4),
        "median": round(float(np.median(dijkstra_vec)), 4),
        "p10": round(float(np.quantile(dijkstra_vec, 0.10)), 4),
        "p90": round(float(np.quantile(dijkstra_vec, 0.90)), 4),
    }
    log.info(
        "Shortest paths — mean: %.4f, std: %.4f, median: %.4f",
        path_stats["mean"],
        path_stats["std"],
        path_stats["median"],
    )

    with open(results_dir / "shortest_path_stats.json", "w") as f:
        json.dump(path_stats, f, indent=2)
    log.debug("Saved shortest_path_stats.json")


def analyze_network_role_by_code_contribution_status(  # noqa: C901
    pairs: pl.DataFrame,
    results_dir: Path,
    n_iterations: int,
) -> None:
    """Analyze network role of authors split by code-contribution status."""
    doc_ids_in_dataset = pairs["document_id"].unique().to_list()
    doc_contribs = _read_table("document_contributor").filter(
        pl.col("document_id").is_in(doc_ids_in_dataset)
    )

    graph, _, idx_to_node = _build_coauthorship_graph(doc_contribs)

    if graph.num_nodes() == 0:
        log.warning("No network nodes available for code-contribution status analysis.")
        return

    code_contrib_researcher_ids = set(
        _get_author_developer_pairs_connected_to_pairs(
            pairs.select("document_id", "repository_id").unique(),
            confidence_threshold=0.97,
        )["researcher_id"]
        .unique()
        .to_list()
    )

    degree_rows: list[dict[str, int | bool]] = []
    for idx in range(graph.num_nodes()):
        rid = idx_to_node[idx]
        degree_rows.append(
            {
                "researcher_id": int(rid),
                "degree": int(graph.degree(idx)),
                "is_code_contributor": bool(rid in code_contrib_researcher_ids),
            }
        )

    degree_df = pl.DataFrame(degree_rows)
    degree_df.write_csv(results_dir / "network_degree_by_code_contribution_status.csv")

    degree_summary = degree_df.group_by("is_code_contributor").agg(
        pl.len().alias("n_authors"),
        pl.col("degree").mean().alias("mean_degree"),
        pl.col("degree").median().alias("median_degree"),
        pl.col("degree").quantile(0.25).alias("degree_q25"),
        pl.col("degree").quantile(0.75).alias("degree_q75"),
    )
    degree_summary.write_csv(results_dir / "network_degree_status_summary.csv")

    edge_type_counts = {
        "contributor_contributor": 0,
        "contributor_non_contributor": 0,
        "non_contributor_non_contributor": 0,
    }

    edge_list = list(graph.edge_list()) if hasattr(graph, "edge_list") else []
    for idx_u, idx_v in edge_list:
        u_is_contrib = idx_to_node[idx_u] in code_contrib_researcher_ids
        v_is_contrib = idx_to_node[idx_v] in code_contrib_researcher_ids

        if u_is_contrib and v_is_contrib:
            edge_type_counts["contributor_contributor"] += 1
        elif u_is_contrib or v_is_contrib:
            edge_type_counts["contributor_non_contributor"] += 1
        else:
            edge_type_counts["non_contributor_non_contributor"] += 1

    components = rx.connected_components(graph)
    largest_cc: set[int] = max(components, key=len) if components else set()
    largest_cc_researchers = {idx_to_node[idx] for idx in largest_cc}

    path_lengths_by_status: defaultdict[str, list[int]] = defaultdict(list)
    if len(largest_cc) >= 2:
        subgraph = graph.subgraph(list(largest_cc))
        subgraph_indices = list(range(subgraph.num_nodes()))

        for _ in tqdm(
            range(n_iterations),
            desc="Shortest paths by code-contribution status",
        ):
            source, target = random.sample(subgraph_indices, 2)
            dijkstra_res = rx.dijkstra_shortest_path_lengths(
                subgraph,
                source,
                lambda _: 1,
                goal=target,
            )
            path_len = int(dijkstra_res[target])

            source_rid = subgraph[source]
            target_rid = subgraph[target]
            source_is_contrib = source_rid in code_contrib_researcher_ids
            target_is_contrib = target_rid in code_contrib_researcher_ids

            if source_is_contrib and target_is_contrib:
                key = "contributor-contributor"
            elif source_is_contrib or target_is_contrib:
                key = "contributor-non_contributor"
            else:
                key = "non_contributor-non_contributor"

            path_lengths_by_status[key].append(path_len)

    path_summary_rows: list[dict[str, float | int | str]] = []
    path_plot_rows: list[dict[str, float | str]] = []
    for key, values in path_lengths_by_status.items():
        if not values:
            continue
        vec = np.array(values)
        path_summary_rows.append(
            {
                "status_pair": key,
                "n_paths": len(vec),
                "mean": float(np.mean(vec)),
                "median": float(np.median(vec)),
                "std": float(np.std(vec)),
            }
        )
        path_plot_rows.extend(
            [{"status_pair": key, "shortest_path_length": float(v)} for v in values]
        )

    if path_summary_rows:
        pl.DataFrame(path_summary_rows).write_csv(
            results_dir / "network_shortest_path_by_code_contribution_status.csv"
        )

    role_summary = {
        "total_authors": int(graph.num_nodes()),
        "code_contributor_authors": len(code_contrib_researcher_ids),
        "pct_code_contributor_authors": round(
            100 * len(code_contrib_researcher_ids) / graph.num_nodes(), 2
        ),
        "largest_component_size": len(largest_cc_researchers),
        "largest_component_code_contributor_pct": round(
            100
            * len(largest_cc_researchers.intersection(code_contrib_researcher_ids))
            / len(largest_cc_researchers),
            2,
        )
        if largest_cc_researchers
        else 0,
        "edge_type_counts": edge_type_counts,
    }

    with open(results_dir / "network_code_contribution_role_summary.json", "w") as f:
        json.dump(role_summary, f, indent=2)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10), constrained_layout=True)

    degree_plot_df = degree_df.with_columns(
        pl.when(pl.col("is_code_contributor"))
        .then(pl.lit("Code Contributor"))
        .otherwise(pl.lit("Non Contributor"))
        .alias("status_label")
    )
    sns.boxplot(
        data=degree_plot_df,
        x="status_label",
        y="degree",
        hue="status_label",
        palette={"Code Contributor": "#1f77b4", "Non Contributor": "#ff7f0e"},
        showfliers=False,
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(
        "Co-Authorship Degree by Code-Contribution Status\n"
        "(Degree = number of unique co-authors; higher = more collaborative)",
        fontsize=11,
    )
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Node Degree (# unique co-authors)")

    edge_mix_df = pl.DataFrame(
        {
            "edge_type": list(edge_type_counts.keys()),
            "count": list(edge_type_counts.values()),
        }
    )
    sns.barplot(
        data=edge_mix_df,
        x="edge_type",
        y="count",
        hue="edge_type",
        legend=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(
        "Edge Mix by Code-Contribution Status\n"
        "(Co-authorship edges classified by contributor status of both endpoints)",
        fontsize=11,
    )
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Edge Count")
    axes[0, 1].tick_params(axis="x", rotation=30)

    if path_plot_rows:
        path_plot_df = pl.DataFrame(path_plot_rows)
        sns.boxplot(
            data=path_plot_df,
            x="status_pair",
            y="shortest_path_length",
            showfliers=False,
            ax=axes[1, 0],
        )
        axes[1, 0].set_title(
            "Shortest Path Length by Status Pair\n"
            "(Fewer hops = closer in co-authorship network)",
            fontsize=11,
        )
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_ylabel("Shortest Path Length")
        axes[1, 0].tick_params(axis="x", rotation=30)
    else:
        axes[1, 0].text(0.5, 0.5, "Insufficient path data", ha="center", va="center")
        axes[1, 0].set_axis_off()

    try:
        if hasattr(rx, "spring_layout") and len(largest_cc) >= 2:
            sample_nodes = list(largest_cc)
            max_nodes_for_plot = 250
            if len(sample_nodes) > max_nodes_for_plot:
                sample_nodes = random.sample(sample_nodes, max_nodes_for_plot)

            sample_subgraph = graph.subgraph(sample_nodes)
            layout = rx.spring_layout(sample_subgraph, seed=42)
            coords = {i: layout[i] for i in range(sample_subgraph.num_nodes())}

            for idx_u, idx_v in (
                sample_subgraph.edge_list() if hasattr(sample_subgraph, "edge_list") else []
            ):
                x_vals = [coords[idx_u][0], coords[idx_v][0]]
                y_vals = [coords[idx_u][1], coords[idx_v][1]]
                axes[1, 1].plot(x_vals, y_vals, color="lightgray", linewidth=0.25, alpha=0.5)

            xs: list[float] = []
            ys: list[float] = []
            colors: list[str] = []
            for idx in range(sample_subgraph.num_nodes()):
                rid = sample_subgraph[idx]
                xs.append(coords[idx][0])
                ys.append(coords[idx][1])
                colors.append("#1f77b4" if rid in code_contrib_researcher_ids else "#ff7f0e")

            axes[1, 1].scatter(xs, ys, c=colors, s=14, alpha=0.85)
            axes[1, 1].set_title("Sampled Co-Authorship Network Colored by Contribution Status")
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
            axes[1, 1].set_xlabel("")
            axes[1, 1].set_ylabel("")

            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#1f77b4",
                    label="Code Contributor",
                    markersize=6,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#ff7f0e",
                    label="Non Contributor",
                    markersize=6,
                ),
            ]
            axes[1, 1].legend(handles=legend_handles, loc="best")
        else:
            axes[1, 1].text(0.5, 0.5, "spring_layout unavailable", ha="center", va="center")
            axes[1, 1].set_axis_off()
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to render sampled network layout: %s", exc)
        axes[1, 1].text(0.5, 0.5, "Failed to render network layout", ha="center", va="center")
        axes[1, 1].set_axis_off()

    fig.savefig(
        results_dir / "network_code_contribution_role.png", bbox_inches="tight", dpi=300
    )
    plt.close(fig)


###############################################################################
# Additional analyses
###############################################################################

# Regex patterns matched against the full license name string (case-insensitive).
# Permissive: MIT, Apache, BSD family, Unlicense, ISC, Boost, zlib, CC0, etc.
_PERMISSIVE_LICENSE_RE = (
    r"(?i)^mit\b|apache|bsd|unlicense|isc license|boost software"
    r"|zlib|ncsa|creative commons zero|cc0|blue oak|academic free"
    r"|universal permissive|mulan permissive|cern.*permissive"
    r"|mit no attribution"
)
# Copyleft: GPL, LGPL, AGPL, MPL, EUPL, Eclipse, etc.
_COPYLEFT_LICENSE_RE = (
    r"(?i)gnu general public|gnu lesser|gnu affero|mozilla public"
    r"|european union public|eclipse public|cecill|open software license"
    r"|strongly reciprocal|weakly reciprocal"
)
_LICENSE_ORDER = ["Permissive", "Copyleft", "Other", "No License"]


def analyze_license_distribution(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Analyze repository license distribution by field and relationship with FWCI."""
    license_df = pairs.with_columns(
        pl.when(pl.col("repository_license").is_null())
        .then(pl.lit("No License"))
        .when(pl.col("repository_license").str.contains(_PERMISSIVE_LICENSE_RE))
        .then(pl.lit("Permissive"))
        .when(pl.col("repository_license").str.contains(_COPYLEFT_LICENSE_RE))
        .then(pl.lit("Copyleft"))
        .otherwise(pl.lit("Other"))
        .alias("license_category")
    )

    license_counts = (
        license_df["license_category"].value_counts(sort=True).rename({"count": "n_pairs"})
    )
    license_counts.write_csv(results_dir / "license_category_counts.csv")

    field_license_df = license_df.filter(
        pl.col("document_field_name").is_not_null(),
        pl.col("document_field_name") != "",
    )
    field_license_df, top_fields, field_col = _add_top_n_other_column(
        field_license_df, "document_field_name", top_n
    )

    field_license_counts = field_license_df.group_by([field_col, "license_category"]).agg(
        pl.len().alias("n_pairs")
    )
    field_totals = field_license_df.group_by(field_col).agg(pl.len().alias("total_pairs"))
    field_license_pct = field_license_counts.join(
        field_totals, on=field_col, how="left"
    ).with_columns((pl.col("n_pairs") / pl.col("total_pairs") * 100).alias("pct_pairs"))
    field_license_pct.write_csv(results_dir / "license_by_field_pct.csv")

    # Chi-square: field x license category
    contingency = field_license_counts.pivot(
        on="license_category", index=field_col, values="n_pairs"
    ).fill_null(0)
    ct_matrix = contingency.select(
        [c for c in contingency.columns if c != field_col]
    ).to_numpy()
    if ct_matrix.shape[0] >= 2 and ct_matrix.shape[1] >= 2:
        chi2_stat, chi2_p, _, _ = chi2_contingency(ct_matrix)
        with open(results_dir / "license_field_chisq.json", "w") as f:
            json.dump({"chi2": float(chi2_stat), "p_value": float(chi2_p)}, f, indent=2)

    # Kruskal-Wallis: FWCI by license category
    fwci_license_df = _filter_finite(license_df, ["document_fwci"])
    kw_groups = [
        fwci_license_df.filter(pl.col("license_category") == cat)["document_fwci"].to_numpy()
        for cat in _LICENSE_ORDER
        if fwci_license_df.filter(pl.col("license_category") == cat).height >= 5
    ]
    if len(kw_groups) >= 2:
        kw_stat, kw_p = kruskal(*kw_groups)
        with open(results_dir / "license_fwci_kruskal.json", "w") as f:
            json.dump({"kruskal_h": float(kw_stat), "p_value": float(kw_p)}, f, indent=2)

    license_fwci_order = (
        fwci_license_df.group_by("license_category")
        .agg(pl.col("document_fwci").median().alias("median_fwci"))
        .sort("median_fwci", descending=True)["license_category"]
        .to_list()
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7), constrained_layout=True)

    # Panel 1: Stacked bar of license categories by field
    field_order = (
        field_license_df.group_by(field_col)
        .agg(pl.len().alias("total"))
        .sort("total", descending=True)[field_col]
        .to_list()
    )
    pivot_pandas = (
        field_license_pct.pivot(on="license_category", index=field_col, values="pct_pairs")
        .fill_null(0)
        .to_pandas()
        .set_index(field_col)
        .reindex(field_order)
    )
    present_license_cols = [c for c in _LICENSE_ORDER if c in pivot_pandas.columns]
    pivot_pandas[present_license_cols].plot(
        kind="barh",
        stacked=True,
        ax=axes[0],
        colormap="Set2",
    )
    axes[0].set_title(
        f"License Category Composition by Field (Top {top_n} + Other)\n(% of pairs per field)",
        fontsize=12,
    )
    axes[0].set_xlabel("Percentage of Pairs (%)")
    axes[0].set_ylabel("")
    axes[0].legend(title="License Category", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Panel 2: FWCI by license category
    kw_label = ""
    if len(kw_groups) >= 2:
        kw_label = f"\nKruskal-Wallis H={kw_stat:.1f}, p={kw_p:.3e}"
    sns.boxplot(
        data=fwci_license_df,
        y="license_category",
        x="document_fwci",
        order=license_fwci_order,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_title(
        f"Document FWCI by License Category{kw_label}",
        fontsize=12,
    )
    axes[1].set_xlabel("Document FWCI")
    axes[1].set_ylabel("")

    fig.savefig(results_dir / "license_distribution.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def analyze_lifecycle_vs_fwci(
    pairs: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Test whether lifecycle phases predict article citation impact (FWCI)."""
    lc_df = _filter_finite(
        pairs,
        [
            "document_fwci",
            "days_from_repo_creation_to_publication",
            "days_from_publication_to_last_push",
        ],
    ).with_columns(
        pl.when(
            pl.col("days_from_repo_creation_to_publication") > LIFECYCLE_RELEASE_WINDOW_DAYS
        )
        .then(pl.lit(f"Repo before pub (>{LIFECYCLE_RELEASE_WINDOW_DAYS}d)"))
        .when(pl.col("days_from_repo_creation_to_publication") < -LIFECYCLE_RELEASE_WINDOW_DAYS)
        .then(pl.lit(f"Repo after pub (>{LIFECYCLE_RELEASE_WINDOW_DAYS}d)"))
        .otherwise(pl.lit(f"Concurrent (within {LIFECYCLE_RELEASE_WINDOW_DAYS}d)"))
        .alias("creation_timing"),
        pl.when(pl.col("days_from_publication_to_last_push") <= 0)
        .then(pl.lit("No post-pub maintenance"))
        .when(pl.col("days_from_publication_to_last_push") <= LIFECYCLE_LONG_MAINTENANCE_DAYS)
        .then(pl.lit(f"Short-term (<={LIFECYCLE_LONG_MAINTENANCE_DAYS}d)"))
        .otherwise(pl.lit(f"Long-term (>{LIFECYCLE_LONG_MAINTENANCE_DAYS}d)"))
        .alias("maintenance_timing"),
    )

    if lc_df.height == 0:
        log.warning("No lifecycle-FWCI rows available. Skipping.")
        return

    creation_cats = [
        f"Repo before pub (>{LIFECYCLE_RELEASE_WINDOW_DAYS}d)",
        f"Concurrent (within {LIFECYCLE_RELEASE_WINDOW_DAYS}d)",
        f"Repo after pub (>{LIFECYCLE_RELEASE_WINDOW_DAYS}d)",
    ]
    maintenance_cats = [
        "No post-pub maintenance",
        f"Short-term (<={LIFECYCLE_LONG_MAINTENANCE_DAYS}d)",
        f"Long-term (>{LIFECYCLE_LONG_MAINTENANCE_DAYS}d)",
    ]

    def _kw_label(cats: list[str], col: str) -> tuple[list[str], str]:
        present = [c for c in cats if lc_df.filter(pl.col(col) == c).height >= 5]
        groups = [lc_df.filter(pl.col(col) == c)["document_fwci"].to_numpy() for c in present]
        label = ""
        if len(groups) >= 2:
            h, p = kruskal(*groups)
            label = f"\nKruskal-Wallis H={h:.1f}, p={p:.3e}"
        return present, label

    creation_order, creation_label = _kw_label(creation_cats, "creation_timing")
    maintenance_order, maintenance_label = _kw_label(maintenance_cats, "maintenance_timing")

    summary_rows: list[dict] = []
    for dim, cats_order, col in [
        ("creation_timing", creation_order, "creation_timing"),
        ("maintenance_timing", maintenance_order, "maintenance_timing"),
    ]:
        for cat in cats_order:
            vals = lc_df.filter(pl.col(col) == cat)["document_fwci"].to_numpy()
            summary_rows.append(
                {
                    "dimension": dim,
                    "category": cat,
                    "n": len(vals),
                    "median_fwci": float(np.median(vals)),
                    "mean_fwci": float(np.mean(vals)),
                }
            )
    pl.DataFrame(summary_rows).write_csv(results_dir / "lifecycle_fwci_summary.csv")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), constrained_layout=True)

    sns.boxplot(
        data=lc_df,
        y="creation_timing",
        x="document_fwci",
        order=creation_order,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set_title(
        f"FWCI by Repository Creation Timing{creation_label}",
        fontsize=12,
    )
    axes[0].set_xlabel("Document FWCI")
    axes[0].set_ylabel("")

    sns.boxplot(
        data=lc_df,
        y="maintenance_timing",
        x="document_fwci",
        order=maintenance_order,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_title(
        f"FWCI by Post-Publication Maintenance Duration{maintenance_label}",
        fontsize=12,
    )
    axes[1].set_xlabel("Document FWCI")
    axes[1].set_ylabel("")

    fig.savefig(results_dir / "lifecycle_vs_fwci.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def analyze_geographic_diversity_vs_fwci(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Correlate author-team geographic diversity with article citation impact (FWCI)."""
    geo_fwci_df = _filter_finite(pairs, ["document_fwci", "document_author_country_entropy"])

    if geo_fwci_df.height < 10:
        log.warning("Insufficient rows for geographic diversity vs FWCI. Skipping.")
        return

    x_all = geo_fwci_df["document_author_country_entropy"].to_numpy()
    y_all = geo_fwci_df["document_fwci"].to_numpy()
    overall_rho = _compute_spearman_rho(x_all, y_all)

    geo_field_df = geo_fwci_df.filter(
        pl.col("document_field_name").is_not_null(),
        pl.col("document_field_name") != "",
    )
    geo_field_df, top_fields, field_col = _add_top_n_other_column(
        geo_field_df, "document_field_name", top_n
    )
    field_rho_rows: list[dict] = []
    for field_name in top_fields:
        fd = geo_field_df.filter(pl.col(field_col) == field_name)
        if fd.height < 25:
            continue
        field_rho_rows.append(
            {
                "field": field_name,
                "n": fd.height,
                "spearman_rho": _compute_spearman_rho(
                    fd["document_author_country_entropy"].to_numpy(),
                    fd["document_fwci"].to_numpy(),
                ),
            }
        )

    pl.DataFrame(
        [
            {"field": "Overall", "n": geo_fwci_df.height, "spearman_rho": overall_rho},
            *field_rho_rows,
        ]
    ).write_csv(results_dir / "geographic_diversity_vs_fwci_correlations.csv")

    # Mann-Whitney: single-country vs multi-country FWCI
    single_fwci = geo_fwci_df.filter(pl.col("document_n_unique_author_countries") == 1)[
        "document_fwci"
    ].to_numpy()
    multi_fwci = geo_fwci_df.filter(pl.col("document_n_unique_author_countries") > 1)[
        "document_fwci"
    ].to_numpy()

    mw_label = ""
    if len(single_fwci) >= 2 and len(multi_fwci) >= 2:
        u_stat, p_val = mannwhitneyu(single_fwci, multi_fwci, alternative="two-sided")
        n1, n2 = len(single_fwci), len(multi_fwci)
        r = 1 - (2 * u_stat) / (n1 * n2)
        mw_label = f"U={u_stat:.0f}, p={p_val:.3e}, r={r:.3f}"
        with open(results_dir / "single_vs_multi_country_fwci.json", "w") as f:
            json.dump(
                {
                    "n_single_country": int(n1),
                    "n_multi_country": int(n2),
                    "median_fwci_single": float(np.median(single_fwci)),
                    "median_fwci_multi": float(np.median(multi_fwci)),
                    "u_statistic": float(u_stat),
                    "p_value": float(p_val),
                    "rank_biserial_r": float(r),
                },
                f,
                indent=2,
            )

    reg = _fit_simple_linear_regression(x_all, y_all)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7), constrained_layout=True)

    # Panel 1: Scatter entropy vs FWCI with trend line
    p99_fwci = cast(float, geo_fwci_df["document_fwci"].quantile(0.99))
    sns.scatterplot(
        data=geo_fwci_df.filter(pl.col("document_fwci") < p99_fwci),
        x="document_author_country_entropy",
        y="document_fwci",
        alpha=0.15,
        ax=axes[0],
    )
    if np.isfinite(reg["slope"]):
        x_line = np.linspace(float(x_all.min()), float(x_all.max()), 100)
        axes[0].plot(
            x_line,
            reg["intercept"] + reg["slope"] * x_line,
            color="black",
            linestyle="--",
            linewidth=2,
        )
    axes[0].set_title(
        f"Country Diversity Entropy vs Document FWCI\nSpearman rho={overall_rho:.3f}",
        fontsize=13,
    )
    axes[0].set_xlabel("Author Country Entropy (0=single country)")
    axes[0].set_ylabel("Document FWCI")

    # Panel 2: Single vs multi-country FWCI box plot
    country_group_df = geo_fwci_df.with_columns(
        pl.when(pl.col("document_n_unique_author_countries") == 1)
        .then(pl.lit("Single Country"))
        .otherwise(pl.lit("Multi-Country"))
        .alias("country_group")
    ).filter(pl.col("document_fwci") < p99_fwci)
    sns.boxplot(
        data=country_group_df,
        x="country_group",
        y="document_fwci",
        order=["Single Country", "Multi-Country"],
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_title(
        f"FWCI: Single Country vs Multi-Country Author Teams\n{mw_label}",
        fontsize=12,
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Document FWCI")

    fig.savefig(results_dir / "geographic_diversity_vs_fwci.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def analyze_code_contributor_ratio(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Analyze confirmed author-coder overlap ratio vs total contributor ratio by field."""
    pair_links = pairs.select("document_id", "repository_id").unique()

    confirmed_overlaps = (
        _get_author_developer_pairs_connected_to_pairs(
            pair_links,
            confidence_threshold=0.97,
        )
        .group_by(["document_id", "repository_id"])
        .agg(pl.n_unique("developer_account_id").alias("confirmed_overlap_count"))
    )

    ratio_df = (
        pairs.select(
            "document_id",
            "repository_id",
            "document_n_authors",
            "repository_n_contributors",
            "document_fwci",
            "document_field_name",
        )
        .join(confirmed_overlaps, on=["document_id", "repository_id"], how="left")
        .with_columns(pl.col("confirmed_overlap_count").fill_null(0))
        .filter(
            pl.col("document_n_authors").is_not_null(),
            pl.col("document_n_authors") > 0,
            pl.col("repository_n_contributors").is_not_null(),
            pl.col("repository_n_contributors") > 0,
        )
        .with_columns(
            (pl.col("repository_n_contributors") / pl.col("document_n_authors"))
            .clip(upper_bound=1.0)
            .alias("total_code_author_ratio"),
            (pl.col("confirmed_overlap_count") / pl.col("document_n_authors"))
            .clip(upper_bound=1.0)
            .alias("confirmed_overlap_ratio"),
        )
    )

    if ratio_df.height == 0:
        log.warning("No rows available for code contributor ratio analysis. Skipping.")
        return

    with open(results_dir / "code_contributor_ratio_summary.json", "w") as f:
        json.dump(
            {
                "n_pairs": int(ratio_df.height),
                "pairs_with_zero_confirmed_overlap": int(
                    ratio_df.filter(pl.col("confirmed_overlap_count") == 0).height
                ),
                "pct_zero_confirmed_overlap": round(
                    100
                    * ratio_df.filter(pl.col("confirmed_overlap_count") == 0).height
                    / ratio_df.height,
                    2,
                ),
                "median_total_ratio": cast(float, ratio_df["total_code_author_ratio"].median()),
                "median_confirmed_ratio": cast(
                    float, ratio_df["confirmed_overlap_ratio"].median()
                ),
            },
            f,
            indent=2,
        )

    ratio_field_df = ratio_df.filter(
        pl.col("document_field_name").is_not_null(),
        pl.col("document_field_name") != "",
    )
    ratio_field_df, top_fields, field_col = _add_top_n_other_column(
        ratio_field_df, "document_field_name", top_n
    )
    field_ratio_stats = (
        ratio_field_df.group_by(field_col)
        .agg(
            pl.len().alias("n_pairs"),
            pl.col("total_code_author_ratio").median().alias("median_total_ratio"),
            pl.col("confirmed_overlap_ratio").median().alias("median_confirmed_ratio"),
        )
        .sort("median_confirmed_ratio", descending=True)
    )
    field_ratio_stats.write_csv(results_dir / "code_contributor_ratio_by_field.csv")

    # Spearman correlations with FWCI
    fwci_ratio_df = _filter_finite(ratio_df, ["document_fwci"])
    if fwci_ratio_df.height >= 10:
        total_rho = _compute_spearman_rho(
            fwci_ratio_df["total_code_author_ratio"].to_numpy(),
            fwci_ratio_df["document_fwci"].to_numpy(),
        )
        conf_rho = _compute_spearman_rho(
            fwci_ratio_df["confirmed_overlap_ratio"].to_numpy(),
            fwci_ratio_df["document_fwci"].to_numpy(),
        )
        with open(results_dir / "code_contributor_ratio_fwci_correlations.json", "w") as f:
            json.dump(
                {
                    "total_ratio_spearman_rho_vs_fwci": float(total_rho),
                    "confirmed_ratio_spearman_rho_vs_fwci": float(conf_rho),
                    "n_pairs_with_fwci": int(fwci_ratio_df.height),
                },
                f,
                indent=2,
            )

    field_order = field_ratio_stats[field_col].to_list()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7), constrained_layout=True)
    fig.suptitle(
        "Code Contributor Overlap Ratios by Academic Field\n"
        "(Total = all GitHub contributors / paper authors; "
        "Confirmed = matched author-devs who contributed to the linked repo / paper authors)",
        fontsize=11,
    )

    sns.boxplot(
        data=ratio_field_df,
        y=field_col,
        x="total_code_author_ratio",
        order=field_order,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set_title("Total Contributor / Author Ratio by Field", fontsize=12)
    axes[0].set_xlabel("Ratio (clipped to 1.0)")
    axes[0].set_ylabel("")

    sns.boxplot(
        data=ratio_field_df,
        y=field_col,
        x="confirmed_overlap_ratio",
        order=field_order,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_title("Confirmed Author-Coder Overlap Ratio by Field", fontsize=12)
    axes[1].set_xlabel("Ratio (clipped to 1.0)")
    axes[1].set_ylabel("")

    fig.savefig(results_dir / "code_contributor_ratio.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_correlation_matrix(
    pairs: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Compute Spearman correlation matrix for all numeric features; rank vs FWCI and FWSI."""
    numeric_cols = [
        "document_fwci",
        "document_cited_by_count",
        "document_n_authors",
        "document_n_unique_author_countries",
        "document_author_country_entropy",
        "document_publication_year",
        "repository_fwsi",
        "repository_stargazers_count",
        "repository_forks_count",
        "repository_commits_count",
        "repository_n_contributors",
        "repository_n_files",
        "repository_n_languages",
        "repository_size_kb",
        "repository_commit_duration_days",
        "days_from_repo_creation_to_publication",
        "days_from_publication_to_last_push",
    ]
    log1p_cols = frozenset(
        {
            "document_fwci",
            "document_cited_by_count",
            "repository_fwsi",
            "repository_stargazers_count",
            "repository_forks_count",
            "repository_commits_count",
            "repository_n_contributors",
            "repository_n_files",
            "repository_size_kb",
            "repository_commit_duration_days",
        }
    )

    available_cols = [c for c in numeric_cols if c in pairs.columns]
    if len(available_cols) < 2:
        log.warning("Fewer than 2 numeric columns available. Skipping correlation matrix.")
        return

    transformed_col_names: list[str] = []
    transform_exprs = []
    for c in available_cols:
        if c in log1p_cols:
            new_name = f"{c}_log1p"
            transform_exprs.append(
                (pl.lit(1) + pl.col(c).cast(pl.Float64)).log().alias(new_name)
            )
            transformed_col_names.append(new_name)
        else:
            transform_exprs.append(pl.col(c).cast(pl.Float64))
            transformed_col_names.append(c)

    pairs_transformed = pairs.select(transform_exprs)

    n = len(transformed_col_names)
    corr_matrix = np.full((n, n), float("nan"))
    for i in range(n):
        corr_matrix[i, i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = transformed_col_names[i], transformed_col_names[j]
            pair_df = _filter_finite(pairs_transformed.select(ci, cj), [ci, cj])
            if pair_df.height >= 5:
                rho = _compute_spearman_rho(
                    pair_df[ci].to_numpy(),
                    pair_df[cj].to_numpy(),
                )
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

    corr_out = pl.DataFrame(
        {
            "feature": transformed_col_names,
            **{transformed_col_names[j]: corr_matrix[:, j].tolist() for j in range(n)},
        }
    )
    corr_out.write_csv(results_dir / "spearman_correlation_matrix.csv")

    # Ranked correlations vs FWCI and FWSI
    for raw_col, _target_name, out_filename in [
        ("document_fwci", "Document FWCI", "correlations_vs_fwci.csv"),
        ("repository_fwsi", "Repository FWSI", "correlations_vs_fwsi.csv"),
    ]:
        t_col = f"{raw_col}_log1p" if raw_col in log1p_cols else raw_col
        if t_col not in transformed_col_names:
            continue
        idx = transformed_col_names.index(t_col)
        rho_vals = corr_matrix[idx, :]
        ranked = sorted(
            zip(transformed_col_names, rho_vals, strict=False),
            key=lambda x: abs(x[1]) if np.isfinite(x[1]) else 0,
            reverse=True,
        )
        pl.DataFrame(
            {
                "feature": [r[0] for r in ranked],
                "spearman_rho_vs_" + raw_col: [float(r[1]) for r in ranked],
            }
        ).filter(pl.col("feature") != t_col).write_csv(results_dir / out_filename)

    # Heatmap
    short_labels = [
        c.replace("_log1p", "")
        .replace("document_", "doc.")
        .replace("repository_", "repo.")
        .replace("_", " ")
        for c in transformed_col_names
    ]

    fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
    mask = np.isnan(corr_matrix)
    sns.heatmap(
        corr_matrix,
        xticklabels=short_labels,
        yticklabels=short_labels,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="RdBu_r",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        mask=mask,
        ax=ax,
    )
    ax.set_title(
        "Spearman Correlation Matrix — All Numeric Features\n"
        "(right-skewed features log(1+x) transformed before ranking)",
        fontsize=13,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.savefig(results_dir / "correlation_matrix.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def analyze_multilanguage_repositories(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Analyze multi-language repositories: field patterns and impact on FWCI."""
    ml_df = pairs.filter(
        pl.col("repository_n_languages").is_not_null(),
        pl.col("repository_n_languages") > 0,
    ).with_columns(
        pl.when(pl.col("repository_n_languages") == 1)
        .then(pl.lit("1 language"))
        .when(pl.col("repository_n_languages") <= 3)
        .then(pl.lit("2-3 languages"))
        .otherwise(pl.lit("4+ languages"))
        .alias("language_count_bin")
    )

    lang_bin_order = ["1 language", "2-3 languages", "4+ languages"]

    ml_df["language_count_bin"].value_counts(sort=True).write_csv(
        results_dir / "multilanguage_distribution.csv"
    )

    ml_field_df = ml_df.filter(
        pl.col("document_field_name").is_not_null(),
        pl.col("document_field_name") != "",
    )
    ml_field_df, top_fields, field_col = _add_top_n_other_column(
        ml_field_df, "document_field_name", top_n
    )

    # Kruskal-Wallis: n_languages by field
    kw_groups = [
        ml_field_df.filter(pl.col(field_col) == f)["repository_n_languages"].to_numpy()
        for f in [*top_fields, "Other"]
        if ml_field_df.filter(pl.col(field_col) == f).height >= 5
    ]
    kw_h, kw_p = float("nan"), float("nan")
    if len(kw_groups) >= 2:
        kw_h, kw_p = kruskal(*kw_groups)
        with open(results_dir / "multilanguage_field_kruskal.json", "w") as f:
            json.dump({"kruskal_h": float(kw_h), "p_value": float(kw_p)}, f, indent=2)

    # Mann-Whitney: single vs multi-language FWCI
    fwci_ml_df = _filter_finite(ml_df, ["document_fwci"])
    single_fwci = fwci_ml_df.filter(pl.col("repository_n_languages") == 1)[
        "document_fwci"
    ].to_numpy()
    multi_fwci = fwci_ml_df.filter(pl.col("repository_n_languages") > 1)[
        "document_fwci"
    ].to_numpy()
    mw_label = ""
    if len(single_fwci) >= 2 and len(multi_fwci) >= 2:
        u_stat, p_val = mannwhitneyu(single_fwci, multi_fwci, alternative="two-sided")
        n1, n2 = len(single_fwci), len(multi_fwci)
        r = 1 - (2 * u_stat) / (n1 * n2)
        mw_label = f"Mann-Whitney U={u_stat:.0f}, p={p_val:.3e}, r={r:.3f}"
        with open(results_dir / "multilanguage_fwci_mw.json", "w") as f:
            json.dump(
                {
                    "n_single_lang": int(n1),
                    "n_multi_lang": int(n2),
                    "median_fwci_single": float(np.median(single_fwci)),
                    "median_fwci_multi": float(np.median(multi_fwci)),
                    "u_statistic": float(u_stat),
                    "p_value": float(p_val),
                    "rank_biserial_r": float(r),
                },
                f,
                indent=2,
            )

    # Spearman rho: n_languages vs n_contributors
    ml_contrib_df = _filter_finite(
        ml_df, ["repository_n_languages", "repository_n_contributors"]
    )
    contrib_rho = _compute_spearman_rho(
        ml_contrib_df["repository_n_languages"].to_numpy().astype(float),
        ml_contrib_df["repository_n_contributors"].to_numpy().astype(float),
    )
    with open(results_dir / "multilanguage_contributors_correlation.json", "w") as f:
        json.dump(
            {
                "n_languages_vs_n_contributors_spearman_rho": float(contrib_rho),
                "n_pairs": int(ml_contrib_df.height),
            },
            f,
            indent=2,
        )

    mean_lang_by_field = (
        ml_field_df.group_by(field_col)
        .agg(
            pl.col("repository_n_languages").mean().alias("mean_n_languages"),
            pl.len().alias("n_pairs"),
        )
        .sort("mean_n_languages", descending=True)
    )
    mean_lang_by_field.write_csv(results_dir / "multilanguage_mean_by_field.csv")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7), constrained_layout=True)

    sns.barplot(
        data=mean_lang_by_field,
        y=field_col,
        x="mean_n_languages",
        hue=field_col,
        order=mean_lang_by_field[field_col].to_list(),
        legend=False,
        ax=axes[0],
    )
    kw_subtitle = f"\nKruskal-Wallis H={kw_h:.1f}, p={kw_p:.3e}" if np.isfinite(kw_h) else ""
    axes[0].set_title(
        f"Mean Number of Repository Languages by Field{kw_subtitle}",
        fontsize=12,
    )
    axes[0].set_xlabel("Mean Number of Languages")
    axes[0].set_ylabel("")

    sns.boxplot(
        data=fwci_ml_df,
        y="language_count_bin",
        x="document_fwci",
        order=lang_bin_order,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_title(
        f"Document FWCI by Repository Language Count\n{mw_label}",
        fontsize=12,
    )
    axes[1].set_xlabel("Document FWCI")
    axes[1].set_ylabel("")

    fig.savefig(results_dir / "multilanguage_analysis.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


###############################################################################
# CLI
###############################################################################


def _run_pair_analyses(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
    n_shortest_path_iterations: int,
    label: str,
    run_network_analyses: bool = True,
) -> None:
    """Run all pair-based analyses, writing results into *results_dir*.

    Parameters
    ----------
    pairs
        Document-repository pair DataFrame (output of :func:`load_pairs`).
    results_dir
        Directory to write step subdirectories and result files into.
    top_n
        Number of top categories for field/language breakdowns.
    n_shortest_path_iterations
        Random shortest-path iterations for network analysis.
    label
        Human-readable label for log messages (e.g. ``"all"`` or ``"high-conf"``).
    run_network_analyses
        Whether to run the (potentially time-consuming) network analyses. Set to
        False to skip those steps and produce results for the other analyses only.
    """
    total_steps = 18 if run_network_analyses else 16
    step = 0

    def _step_dir(step_num: int) -> Path:
        d = results_dir / f"step-{step_num}"
        d.mkdir(exist_ok=True)
        return d

    step += 1
    log.info("[%s] Step %d/%d: Descriptive statistics...", label, step, total_steps)
    print_descriptive_stats(pairs, _step_dir(step))

    step += 1
    log.info("[%s] Step %d/%d: Field countplot...", label, step, total_steps)
    plot_field_countplot(pairs, _step_dir(step), top_n)

    step += 1
    log.info("[%s] Step %d/%d: Features by field boxplots...", label, step, total_steps)
    plot_features_by_field(pairs, _step_dir(step), top_n)

    step += 1
    log.info("[%s] Step %d/%d: Pairs over time...", label, step, total_steps)
    sd = _step_dir(step)
    plot_pairs_over_time(pairs, sd)
    plot_pairs_over_time_by_field(pairs, sd, top_n)

    step += 1
    log.info("[%s] Step %d/%d: Field and language counts...", label, step, total_steps)
    plot_field_and_language_counts(pairs, _step_dir(step), top_n)

    step += 1
    log.info("[%s] Step %d/%d: Repository metrics over time...", label, step, total_steps)
    plot_repo_metrics_over_time(pairs, _step_dir(step))

    step += 1
    log.info("[%s] Step %d/%d: Geographic diversity depth...", label, step, total_steps)
    plot_geographic_diversity_depth(pairs, _step_dir(step), top_n)

    step += 1
    log.info("[%s] Step %d/%d: FWCI vs FWSI analysis...", label, step, total_steps)
    plot_fwci_vs_fwsi(pairs, _step_dir(step), top_n)

    step += 1
    log.info("[%s] Step %d/%d: Date relationships...", label, step, total_steps)
    plot_date_relationships(pairs, _step_dir(step))

    step += 1
    log.info("[%s] Step %d/%d: Lifecycle and survival analyses...", label, step, total_steps)
    plot_lifecycle_and_survival(pairs, _step_dir(step))

    step += 1
    log.info("[%s] Step %d/%d: License distribution analysis...", label, step, total_steps)
    analyze_license_distribution(pairs, _step_dir(step), top_n)

    step += 1
    log.info("[%s] Step %d/%d: Lifecycle phases vs FWCI...", label, step, total_steps)
    analyze_lifecycle_vs_fwci(pairs, _step_dir(step))

    step += 1
    log.info("[%s] Step %d/%d: Geographic diversity vs FWCI...", label, step, total_steps)
    analyze_geographic_diversity_vs_fwci(pairs, _step_dir(step), top_n)

    step += 1
    log.info("[%s] Step %d/%d: Code contributor ratio analysis...", label, step, total_steps)
    analyze_code_contributor_ratio(pairs, _step_dir(step), top_n)

    step += 1
    log.info("[%s] Step %d/%d: Pairwise correlation matrix...", label, step, total_steps)
    plot_correlation_matrix(pairs, _step_dir(step))

    step += 1
    log.info("[%s] Step %d/%d: Multi-language repository analysis...", label, step, total_steps)
    analyze_multilanguage_repositories(pairs, _step_dir(step), top_n)

    if run_network_analyses:
        step += 1
        log.info("[%s] Step %d/%d: Network coverage...", label, step, total_steps)
        analyze_network_coverage(pairs, _step_dir(step), n_shortest_path_iterations)

        step += 1
        log.info(
            "[%s] Step %d/%d: Network role by code-contribution status...",
            label,
            step,
            total_steps,
        )
        analyze_network_role_by_code_contribution_status(
            pairs,
            _step_dir(step),
            n_shortest_path_iterations,
        )


@app.command()
def analyze(
    top_n: int = typer.Option(9, help="Number of top categories (rest grouped as 'Other')."),
    n_shortest_path_iterations: int = typer.Option(
        400, help="Random shortest path iterations for network analysis."
    ),
    sample_size: int | None = typer.Option(
        None, help="Sample this many pairs for faster analysis."
    ),
    run_network_analyses: bool = typer.Option(
        True, help="Whether to run the (potentially time-consuming) network analyses."
    ),
    debug: bool = typer.Option(False, help="Enable debug logging."),
) -> None:
    """Run the full RQ1 analysis pipeline.

    Each pair-based analysis is run twice — once on **all** pairs and once on
    a **high-confidence** subset (doc-repo confidence null or >= 0.9994).
    Results are written to ``all/`` and ``high-conf/`` subdirectories.
    The iteration-expansion analysis (which reads directly from the DB)
    tracks both subsets in a single output.
    """
    setup_logger(debug=debug)

    results_dir = Path(__file__).parent / "rq1-results"

    # Delete existing results if present, to ensure clean slate for each run
    if results_dir.exists():
        log.info("Deleting existing results directory: %s", results_dir)
        shutil.rmtree(results_dir)

    # Remake the dir
    results_dir.mkdir(exist_ok=True)

    sns.set_palette(PALETTE)

    # -- Step 1: Load pairs (all + high-conf) --------------------------------
    log.info("Loading pairs (all)...")
    # pairs_all = load_pairs(sample_size=sample_size)
    log.info("Loading pairs (high-conf, >= 0.9994)...")
    pairs_high_conf = load_pairs(
        sample_size=sample_size,
        doc_repo_confidence_threshold=0.9994,
    )

    # -- Step 2: Iteration expansion (combined output) -----------------------
    log.info("Iteration expansion (tracks all + high-conf internally)...")
    iter_dir = results_dir / "iteration-expansion"
    iter_dir.mkdir(exist_ok=True)
    plot_iteration_expansion(iter_dir)

    # -- Steps 3+: Run all pair-based analyses for each subset ---------------
    # for label, pairs in [("all", pairs_all), ("high-conf", pairs_high_conf)]:
    for label, pairs in [("high-conf", pairs_high_conf)]:
        subset_dir = results_dir / label
        subset_dir.mkdir(exist_ok=True)
        log.info("Running pair-based analyses for subset: %s", label)
        _run_pair_analyses(
            pairs,
            subset_dir,
            top_n,
            n_shortest_path_iterations,
            label,
            run_network_analyses=run_network_analyses,
        )

    log.info("Analysis complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    app()
