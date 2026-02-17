import json
import logging
import random
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
import statsmodels.api as sm
import typer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.db import constants as db_constants

###############################################################################
# Constants
###############################################################################

PALETTE = cmaps.bold._colors.tolist()

SHARED_SOURCES = frozenset({"pwc", "plos", "joss", "softwarex", "softcite_2025"})
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
]

CATEGORICAL_FEATURES = [
    "document_field_name",
    "document_type",
    "repository_primary_language",
]

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
    return pl.read_database_uri(
        f"SELECT * FROM {table}",
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
    results_dir: Path,
    sample_size: int | None = None,
) -> pl.DataFrame:
    """Load document-repository pairs with provenance labels for RQ2."""
    log.debug("Reading database tables...")

    dataset_sources = _read_table("dataset_source")
    docs = _read_table("document")
    repos = _read_table("repository")
    pairs = _read_table("document_repository_link")
    doc_topics = _read_table("document_topic")
    topics = _read_table("topic")

    log.info("Raw document_repository_link rows: %d", pairs.height)

    # Drop predicted doc-repo pairs below confidence threshold
    pairs = pairs.filter(
        pl.col("predictive_model_confidence").is_null()
        | (pl.col("predictive_model_confidence") >= 0.995)
    )
    log.info("Pairs after confidence filter (>= 0.995): %d", pairs.height)

    # Join dataset source name for provenance labeling
    pairs = pairs.join(
        dataset_sources.select(
            pl.col("id").alias("dataset_source_id"),
            pl.col("name").alias("dataset_source_name"),
        ),
        on="dataset_source_id",
        how="left",
    )

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
            pl.col("dataset_source_names")
            .list.join(", ")
            .alias("dataset_source_names_str"),
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

    doc_contribs = _read_table("document_contributor")
    doc_contrib_institutions = _read_table("document_contributor_institution")
    institutions = _read_table("institution")

    doc_author_country_rows = (
        doc_contribs.select(
            pl.col("id").alias("document_contributor_id"),
            pl.col("researcher_id"),
            pl.col("document_id"),
        )
        .join(
            doc_contrib_institutions.select("document_contributor_id", "institution_id"),
            on="document_contributor_id",
            how="left",
        )
        .join(
            institutions.select(pl.col("id").alias("institution_id"), "country_code"),
            on="institution_id",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("country_code").is_null())
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("country_code"))
            .alias("country_code")
        )
    )

    doc_author_countries = (
        doc_author_country_rows.group_by("document_id")
        .agg(pl.len().alias("document_n_authors"))
    )

    repo_contribs = _read_table("repository_contributor")
    repo_contribs = repo_contribs.group_by("repository_id").len("repository_n_contributors")

    repo_files = _read_table("repository_file")
    repo_file_counts = (
        repo_files.filter(pl.col("tree_type") == "blob")
        .group_by("repository_id")
        .agg(pl.len().alias("repository_n_files"))
    )

    result = (
        pairs_deduped.join(
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
                pl.col("creation_datetime").alias("repository_creation_datetime"),
                pl.col("last_pushed_datetime").alias("repository_last_pushed_datetime"),
            ),
            on="repository_id",
            how="left",
        )
        .join(
            doc_topics.sort("score", descending=True)
            .unique("document_id", maintain_order=True)
            .select("document_id", pl.col("topic_id")),
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

    # Save outputs
    result.write_parquet(results_dir / "rq2-pairs.parquet")
    log.info("Saved rq2-pairs.parquet")

    # Missingness summary
    missingness_rows: list[dict[str, str | int | float]] = []
    for col_name in result.columns:
        n_total = result.height
        n_null = result[col_name].null_count()
        missingness_rows.append(
            {
                "column": col_name,
                "n_total": n_total,
                "n_null": n_null,
                "pct_null": round(100 * n_null / n_total, 2) if n_total > 0 else 0,
            }
        )
    pl.DataFrame(missingness_rows).write_csv(results_dir / "rq2-missingness.csv")

    # Schema
    schema_dict = {col: str(dtype) for col, dtype in result.schema.items()}
    with open(results_dir / "rq2-pairs-schema.json", "w") as f:
        json.dump(schema_dict, f, indent=2)

    return result


###############################################################################
# Workstream 2: Logistic Regression
###############################################################################


def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC using the trapezoidal rule (no sklearn)."""
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tpr_prev = 0.0
    fpr_prev = 0.0
    tp = 0
    fp = 0
    auc = 0.0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr

    return float(auc)


def run_logistic_regression(
    pairs: pl.DataFrame,
    results_dir: Path,
    top_n: int,
) -> None:
    """Logistic regression to distinguish shared vs mined pairs."""
    log.info("Starting logistic regression...")

    # Target: 1 = shared, 0 = mined
    model_df = pairs.with_columns(
        pl.when(pl.col("pair_source_label") == "shared")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("target")
    )

    n_shared = int(model_df.filter(pl.col("target") == 1).height)
    n_mined = int(model_df.filter(pl.col("target") == 0).height)
    log.info("Class counts — shared: %d, mined: %d", n_shared, n_mined)

    # --- Prepare numeric features ---
    numeric_cols = [c for c in NUMERIC_FEATURES if c in model_df.columns]

    # Median impute numeric features
    for col_name in numeric_cols:
        median_val = model_df[col_name].median()
        if median_val is None:
            median_val = 0.0
        model_df = model_df.with_columns(pl.col(col_name).fill_null(median_val).fill_nan(median_val))

    # Standardize numeric features
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    for col_name in numeric_cols:
        m = float(model_df[col_name].mean())  # type: ignore[arg-type]
        s = float(model_df[col_name].std())  # type: ignore[arg-type]
        if s == 0 or np.isnan(s):
            s = 1.0
        numeric_means[col_name] = m
        numeric_stds[col_name] = s
        model_df = model_df.with_columns(
            ((pl.col(col_name) - m) / s).alias(col_name)
        )

    # --- Prepare categorical features ---
    cat_dummies: list[str] = []
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col not in model_df.columns:
            continue
        model_df = model_df.with_columns(
            pl.col(cat_col).fill_null("Unknown")
        )
        model_df, _, top_col = _add_top_n_other_column(model_df, cat_col, top_n)
        unique_vals = sorted(model_df[top_col].unique().to_list())
        # Drop first value as reference level
        if len(unique_vals) > 1:
            reference = unique_vals[0]
            for val in unique_vals[1:]:
                dummy_name = f"{cat_col}__{val}"
                model_df = model_df.with_columns(
                    pl.when(pl.col(top_col) == val)
                    .then(pl.lit(1.0))
                    .otherwise(pl.lit(0.0))
                    .alias(dummy_name)
                )
                cat_dummies.append(dummy_name)
            log.debug("Categorical %s: reference=%s, dummies=%d", cat_col, reference, len(unique_vals) - 1)

    all_features = numeric_cols + cat_dummies

    # Filter to finite rows for all features
    model_df = _filter_finite(model_df, numeric_cols)
    log.info("Rows after finite filter: %d", model_df.height)

    if model_df.height < 50:
        log.warning("Too few rows (%d) for logistic regression. Skipping.", model_df.height)
        return

    # Build design matrix
    X = model_df.select(all_features).to_pandas().values.astype(np.float64)
    y = model_df["target"].to_numpy().astype(np.float64)

    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    feature_names = ["const"] + all_features

    # Fit logistic regression
    try:
        glm_model = sm.GLM(y, X_with_const, family=sm.families.Binomial())
        glm_result = glm_model.fit()
    except Exception as exc:
        log.error("GLM fitting failed: %s", exc)
        return

    log.info("GLM converged: %s", glm_result.converged)

    # Extract coefficients
    coef_df = pl.DataFrame(
        {
            "feature": feature_names,
            "coefficient": glm_result.params.tolist(),
            "std_err": glm_result.bse.tolist(),
            "z_value": glm_result.tvalues.tolist(),
            "p_value": glm_result.pvalues.tolist(),
            "ci_lower": glm_result.conf_int()[:, 0].tolist(),
            "ci_upper": glm_result.conf_int()[:, 1].tolist(),
        }
    ).with_columns(
        pl.col("coefficient").exp().alias("odds_ratio"),
        pl.col("ci_lower").exp().alias("or_ci_lower"),
        pl.col("ci_upper").exp().alias("or_ci_upper"),
    )
    coef_df.write_csv(results_dir / "logreg-coefficients.csv")
    log.info("Saved logreg-coefficients.csv")

    # Compute AUC
    y_pred = glm_result.predict(X_with_const)
    auc = _compute_auc(y, y_pred)

    performance = {
        "n_total": int(len(y)),
        "n_shared": int(np.sum(y == 1)),
        "n_mined": int(np.sum(y == 0)),
        "auc": round(auc, 4),
        "pseudo_r_squared": round(float(glm_result.pseudo_rsquared(kind="mcfadden")), 4),
        "aic": round(float(glm_result.aic), 2),
        "bic": round(float(glm_result.bic_llf), 2),
        "converged": bool(glm_result.converged),
    }
    with open(results_dir / "logreg-performance.json", "w") as f:
        json.dump(performance, f, indent=2)

    # VIF check (skip constant)
    try:
        vif_values = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif_df = pl.DataFrame({"feature": all_features, "vif": vif_values})
        vif_df.write_csv(results_dir / "logreg-vif.csv")
        high_vif = vif_df.filter(pl.col("vif") > 10)
        if high_vif.height > 0:
            log.warning("High VIF features (>10):\n%s", high_vif)
    except Exception as exc:
        log.warning("VIF computation failed: %s", exc)

    # Forest plot of odds ratios (skip constant)
    plot_df = coef_df.filter(pl.col("feature") != "const").sort("odds_ratio")

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))
    y_positions = list(range(plot_df.height))
    ors = plot_df["odds_ratio"].to_list()
    ci_low = plot_df["or_ci_lower"].to_list()
    ci_high = plot_df["or_ci_upper"].to_list()
    labels = plot_df["feature"].to_list()

    ax.errorbar(
        ors,
        y_positions,
        xerr=[
            [o - lo for o, lo in zip(ors, ci_low)],
            [hi - o for o, hi in zip(ors, ci_high)],
        ],
        fmt="o",
        color="steelblue",
        ecolor="gray",
        capsize=3,
    )
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Odds Ratio (Shared vs Mined)")
    ax.set_title(
        "Logistic Regression: Odds Ratios for Shared vs Mined Pairs\n"
        f"(AUC={auc:.3f}, N={len(y):,})",
        fontsize=13,
    )
    fig.savefig(results_dir / "logreg-coefficients-forest.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Plain-language summary of top features
    sig_features = coef_df.filter(
        pl.col("feature") != "const",
        pl.col("p_value") < 0.05,
    ).sort("odds_ratio", descending=True)

    lines = [
        "# Logistic Regression: Top Features (Shared vs Mined)\n",
        f"- **N**: {len(y):,} pairs ({int(np.sum(y == 1)):,} shared, {int(np.sum(y == 0)):,} mined)",
        f"- **AUC**: {auc:.3f}",
        f"- **Pseudo R-squared (McFadden)**: {performance['pseudo_r_squared']:.4f}",
        "",
        "## Significant Features (p < 0.05)\n",
        "Features with OR > 1 are more associated with **shared** pairs;",
        "features with OR < 1 are more associated with **mined** pairs.\n",
    ]
    for row in sig_features.iter_rows(named=True):
        direction = "shared" if row["odds_ratio"] > 1 else "mined"
        lines.append(
            f"- **{row['feature']}**: OR={row['odds_ratio']:.3f} "
            f"(95% CI: {row['or_ci_lower']:.3f}-{row['or_ci_upper']:.3f}), "
            f"p={row['p_value']:.4f} — more associated with **{direction}** pairs"
        )

    with open(results_dir / "logreg-top-features.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    log.info("Logistic regression complete.")


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
    n_shared_ad = (
        shared_ad.select("researcher_id", "developer_account_id").unique().height
    )

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
    axes[0].bar(
        ["Shared", "Mined"],
        [n_shared, n_mined],
        color=[PALETTE[0], PALETTE[1]],
    )
    axes[0].set_title(
        "Article-Repository Pair Coverage\n"
        f"(Shared: {overall_coverage['shared_pair_coverage']:.1%}, "
        f"Mined: {overall_coverage['mined_pair_coverage']:.1%})",
        fontsize=12,
    )
    axes[0].set_ylabel("Number of Pairs")

    # Author-developer coverage bar
    n_mined_only_ad = n_full_ad - n_shared_ad
    axes[1].bar(
        ["Shared-only", "Mined-added"],
        [n_shared_ad, n_mined_only_ad],
        color=[PALETTE[0], PALETTE[1]],
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


def analyze_network_components(
    pairs: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Build co-authorship graphs and compare components across subsets."""
    log.info("Starting network component analysis...")

    all_doc_contribs = _read_table("document_contributor")

    # Shared-only graph
    shared_doc_ids = (
        pairs.filter(pl.col("pair_source_label") == "shared")["document_id"]
        .unique()
        .to_list()
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
        pairs.filter(pl.col("pair_source_label") == "mined")["document_id"]
        .unique()
        .to_list()
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
    full_components = (
        rx.connected_components(graph_full) if graph_full.num_nodes() > 0 else []
    )

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
        len(shared_comps) for shared_comps in full_comp_to_shared_comps.values()
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

    # --- Comparison bar chart ---
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), constrained_layout=True)

    subset_labels = ["Shared", "Full", "Mined"]
    all_stats = [stats_shared, stats_full, stats_mined]

    # Nodes and edges
    axes[0].bar(
        subset_labels,
        [s["total_nodes"] for s in all_stats],
        color=PALETTE[:3],
    )
    axes[0].set_title("Total Nodes")
    axes[0].set_ylabel("Count")

    # Components
    axes[1].bar(
        subset_labels,
        [s["total_components"] for s in all_stats],
        color=PALETTE[:3],
    )
    axes[1].set_title("Connected Components")
    axes[1].set_ylabel("Count")

    # Largest CC coverage
    axes[2].bar(
        subset_labels,
        [s["coverage_pct"] for s in all_stats],
        color=PALETTE[:3],
    )
    axes[2].set_title("Largest Component Coverage (%)")
    axes[2].set_ylabel("Coverage (%)")

    fig.suptitle(
        "Co-Authorship Network Comparison: Shared vs Full vs Mined",
        fontsize=14,
    )
    fig.savefig(
        results_dir / "network-component-comparison.png", bbox_inches="tight", dpi=300
    )
    plt.close(fig)

    log.info("Network component analysis complete.")


###############################################################################
# Workstream 5: Shortest Path Analysis
###############################################################################


def _sample_shortest_paths(
    graph: rx.PyGraph,
    n_iterations: int,
    seed: int,
) -> dict[str, float | int]:
    """Sample random shortest paths from the largest connected component."""
    components = rx.connected_components(graph) if graph.num_nodes() > 0 else []
    if not components:
        return {
            "valid_paths": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }

    largest_cc = max(components, key=len)
    subgraph = graph.subgraph(list(largest_cc))

    if subgraph.num_nodes() < 2:
        return {
            "valid_paths": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }

    rng = random.Random(seed)
    subgraph_indices = list(range(subgraph.num_nodes()))
    dijkstra_lengths: list[int] = []

    for _ in range(n_iterations):
        source, target = rng.sample(subgraph_indices, 2)
        dijkstra_res = rx.dijkstra_shortest_path_lengths(
            subgraph,
            source,
            lambda _: 1,
            goal=target,
        )
        dijkstra_lengths.append(dijkstra_res[target])

    vec = np.array(dijkstra_lengths)
    return {
        "largest_cc_nodes": int(subgraph.num_nodes()),
        "largest_cc_edges": int(subgraph.num_edges()),
        "valid_paths": len(vec),
        "mean": round(float(np.mean(vec)), 4),
        "std": round(float(np.std(vec)), 4),
        "median": round(float(np.median(vec)), 4),
        "p10": round(float(np.quantile(vec, 0.10)), 4),
        "p90": round(float(np.quantile(vec, 0.90)), 4),
    }


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

        path_stats = _sample_shortest_paths(graph, n_iterations, seed=42)

        with open(results_dir / f"shortest-path-{subset_name}.json", "w") as f:
            json.dump(path_stats, f, indent=2)

        log.info("Shortest path stats (%s): %s", subset_name, path_stats)

        # Collect raw path lengths for distribution plot
        if path_stats["valid_paths"] > 0:
            components = rx.connected_components(graph)
            largest_cc = max(components, key=len)
            subgraph = graph.subgraph(list(largest_cc))
            rng = random.Random(42)
            subgraph_indices = list(range(subgraph.num_nodes()))
            lengths: list[int] = []
            for _ in range(n_iterations):
                source, target = rng.sample(subgraph_indices, 2)
                dijkstra_res = rx.dijkstra_shortest_path_lengths(
                    subgraph,
                    source,
                    lambda _: 1,
                    goal=target,
                )
                lengths.append(dijkstra_res[target])
            all_path_lengths[subset_name] = lengths

    # --- Distribution plot ---
    if all_path_lengths:
        plot_rows: list[dict[str, str | int]] = []
        for subset_name, lengths in all_path_lengths.items():
            for length in lengths:
                plot_rows.append({"subset": subset_name, "shortest_path_length": length})

        plot_df = pl.DataFrame(plot_rows)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=plot_df,
            x="subset",
            y="shortest_path_length",
            hue="subset",
            palette=PALETTE[:len(all_path_lengths)],
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


def write_summary(results_dir: Path) -> None:
    """Write a summary markdown file from saved result artifacts."""
    lines = ["# RQ2 Analysis Summary\n"]

    # Coverage
    coverage_path = results_dir / "coverage-overall.json"
    if coverage_path.exists():
        with open(coverage_path) as f:
            coverage = json.load(f)
        lines.extend([
            "## Coverage\n",
            f"- **Total pairs**: {coverage['n_total_pairs']:,}",
            f"- **Shared pairs**: {coverage['n_shared_pairs']:,} ({coverage['shared_pair_coverage']:.1%})",
            f"- **Mined pairs**: {coverage['n_mined_pairs']:,} ({coverage['mined_pair_coverage']:.1%})",
            f"- **Shared author-dev pairs**: {coverage['n_shared_author_dev_pairs']:,}",
            f"- **Full author-dev pairs**: {coverage['n_full_author_dev_pairs']:,}",
            f"- **Shared author-dev coverage**: {coverage['shared_author_dev_coverage']:.1%}",
            "",
        ])

    # Logistic regression
    perf_path = results_dir / "logreg-performance.json"
    if perf_path.exists():
        with open(perf_path) as f:
            perf = json.load(f)
        lines.extend([
            "## Logistic Regression\n",
            f"- **N**: {perf['n_total']:,} ({perf['n_shared']:,} shared, {perf['n_mined']:,} mined)",
            f"- **AUC**: {perf['auc']:.3f}",
            f"- **Pseudo R-squared**: {perf['pseudo_r_squared']:.4f}",
            "",
        ])

    # Network stats
    for label in ["shared", "full", "mined"]:
        stats_path = results_dir / f"network-stats-{label}.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            lines.extend([
                f"## Network: {label.capitalize()}\n",
                f"- **Nodes**: {stats['total_nodes']:,}",
                f"- **Edges**: {stats['total_edges']:,}",
                f"- **Components**: {stats['total_components']:,}",
                f"- **Largest CC coverage**: {stats['coverage_pct']:.1f}%",
                "",
            ])

    # Bridging
    bridging_path = results_dir / "network-bridging-summary.json"
    if bridging_path.exists():
        with open(bridging_path) as f:
            bridging = json.load(f)
        lines.extend([
            "## Bridging Analysis\n",
            f"- **Delta components**: {bridging['delta_components']:,}",
            f"- **Delta coverage**: {bridging['delta_coverage_pct']:.1f}%",
            f"- **Merging full components**: {bridging['n_merging_full_components']:,}",
            f"- **Shared components merged**: {bridging['n_shared_components_that_merged']:,}",
            "",
        ])

    # Shortest paths
    for label in ["shared", "mined", "full"]:
        sp_path = results_dir / f"shortest-path-{label}.json"
        if sp_path.exists():
            with open(sp_path) as f:
                sp = json.load(f)
            lines.extend([
                f"## Shortest Paths: {label.capitalize()}\n",
                f"- **Mean**: {sp['mean']:.2f} (std: {sp['std']:.2f})",
                f"- **Median**: {sp['median']:.2f}",
                f"- **p10-p90**: {sp['p10']:.2f} - {sp['p90']:.2f}",
                f"- **Sampled paths**: {sp['valid_paths']:,}",
                "",
            ])

    with open(results_dir / "rq2-summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    log.info("Saved rq2-summary.md")


###############################################################################
# CLI
###############################################################################


@app.command()
def analyze(
    top_n: int = typer.Option(9, help="Number of top categories (rest grouped as 'Other')."),
    n_shortest_path_iterations: int = typer.Option(
        5000, help="Random shortest path iterations for network analysis."
    ),
    sample_size: int | None = typer.Option(
        None, help="Sample this many pairs for faster analysis."
    ),
    debug: bool = typer.Option(False, help="Enable debug logging."),
) -> None:
    """Run the full RQ2 analysis pipeline."""
    setup_logger(debug=debug)

    results_dir = Path(__file__).parent / "rq2-results"
    results_dir.mkdir(exist_ok=True)

    sns.set_palette(PALETTE)

    total_steps = 6
    step = 0

    step += 1
    log.info("Step %d/%d: Loading RQ2 pairs...", step, total_steps)
    pairs = load_rq2_pairs(results_dir, sample_size=sample_size)

    step += 1
    log.info("Step %d/%d: Logistic regression...", step, total_steps)
    run_logistic_regression(pairs, results_dir, top_n)

    step += 1
    log.info("Step %d/%d: Coverage analysis...", step, total_steps)
    compute_coverage(pairs, results_dir)

    step += 1
    log.info("Step %d/%d: Network component analysis...", step, total_steps)
    analyze_network_components(pairs, results_dir)

    step += 1
    log.info("Step %d/%d: Shortest path analysis...", step, total_steps)
    analyze_shortest_paths(pairs, results_dir, n_shortest_path_iterations)

    step += 1
    log.info("Step %d/%d: Writing summary...", step, total_steps)
    write_summary(results_dir)

    log.info("RQ2 analysis complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    app()
