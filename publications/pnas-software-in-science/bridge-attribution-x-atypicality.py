import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

###############################################################################

app = typer.Typer()

THIS_FILE_PATH = Path(__file__).resolve()
THIS_DIR = THIS_FILE_PATH.parent
RESULTS_DIR = THIS_DIR / "results" / "bridge-attribution-x-atypicality"

ATTRIBUTION_RESULTS_DIR = THIS_DIR / "results" / "attribution-of-software"
ATYPICALITY_RESULTS_DIR = THIS_DIR / "results" / "software-atypicality-and-article-impact"

###############################################################################


# Helper to load a table as a polars DataFrame (zero-copy via Arrow)
def load_table(table: str) -> pl.DataFrame:
    ds = load_dataset("evamxb/rs-graph-v2-full", table, split="train")
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


def _load_our_dataset(
    top_n_fields: int = 5,
) -> pl.DataFrame:
    # Load all pair info
    documents = load_table("document")
    article_repo_links = load_table("document_repository_link")
    repositories = load_table("repository")
    document_topics = load_table("document_topic")
    topics = load_table("topic")
    document_contributors = load_table("document_contributor")
    researchers = load_table("researcher")

    # Create dataframe of document_id,
    # document_author_count, and document_author_mean_citations
    document_contributors = document_contributors.join(
        researchers.select(
            pl.col("id").alias("researcher_id"),
            pl.col("cited_by_count").alias("researcher_cited_by_count"),
        ),
        on="researcher_id",
        how="left",
    )
    document_author_stats = (
        document_contributors.group_by("document_id")
        .agg(
            pl.count("researcher_id").alias("document_author_count"),
            pl.mean("researcher_cited_by_count").alias("document_author_mean_citations"),
        )
        .with_columns(
            pl.col("document_author_mean_citations")
            .log1p()
            .alias("document_log_author_mean_citations")
        )
    )

    # Sort document topics by score (descending)
    # Drop duplicated by document_id to get the top topic for each document
    # Join topic info to get the field and domain name
    document_topics = document_topics.sort("score", descending=True)
    document_topics = document_topics.unique(subset="document_id", keep="first")
    document_topics = (
        document_topics.select(
            pl.col("document_id"),
            pl.col("topic_id"),
        )
        .join(
            topics.select(
                pl.col("id").alias("topic_id"),
                pl.col("field_name").alias("document_field_name"),
                pl.col("domain_name").alias("document_domain_name"),
            ),
            on="topic_id",
        )
        .select(
            pl.col("document_id"),
            pl.col("document_field_name").alias("document_field_name"),
            pl.col("document_domain_name").alias("document_domain_name"),
        )
    )

    # Construct the merged dataframe with all basic details
    merged = (
        article_repo_links.select(
            pl.col("id").alias("document_repository_link_id"),
            pl.col("document_id"),
            pl.col("repository_id"),
            pl.col("dataset_source_id"),
            pl.col("predictive_model_confidence"),
        )
        .join(
            documents.select(
                *[pl.col(col).alias(f"document_{col}") for col in documents.columns]
            ),
            on="document_id",
        )
        .join(
            repositories.select(
                *[pl.col(col).alias(f"repository_{col}") for col in repositories.columns]
            ),
            on="repository_id",
        )
        .join(
            document_topics,
            on="document_id",
        )
        .join(
            document_author_stats,
            on="document_id",
            how="left",
        )
    )

    # Create document publication year column as integer (extract year from date)
    merged = merged.with_columns(
        pl.col("document_publication_date")
        .str.to_date("%Y-%m-%d")
        .alias("document_publication_date_parsed"),
    ).with_columns(
        pl.col("document_publication_date_parsed").dt.year().alias("document_publication_year"),
    )

    # Filter to only pairs published after 2008 (the year GitHub was founded)
    merged = merged.filter(pl.col("document_publication_year") >= 2008).with_columns(
        (pl.col("document_publication_year") - pl.col("document_publication_year").min()).alias(
            "document_years_since_earliest"
        )
    )

    # Reduce to only pairs with confidence of 0.9994
    merged = merged.filter(
        (pl.col("predictive_model_confidence") > 0.9994)
        | (pl.col("predictive_model_confidence").is_null())
    )

    # Drop to unique 1:1 pairs
    merged = merged.unique(
        subset="document_id",
        keep="none",
    ).unique(
        subset="repository_id",
        keep="none",
    )

    # Create a "document_field_name_pruned" column
    # that takes the top N most common field names, and then labels the rest as "other"
    top_n_field_names = (
        merged.get_column("document_field_name")
        .value_counts(sort=True)
        .head(top_n_fields)
        .get_column("document_field_name")
        .to_list()
    )
    merged = merged.with_columns(
        pl.when(pl.col("document_field_name").is_in(top_n_field_names))
        .then(pl.col("document_field_name"))
        .otherwise(pl.lit("Other"))
        .alias("document_field_name_pruned")
    )

    return merged


###############################################################################
# Bridge analysis functions
###############################################################################


def _build_bridge_dataframe(
    pair_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """
    Load attribution long-frame and atypicality scores, merge at the paper level.
    Returns a paper-level dataframe with fraction_mentioned and atypicality scores.
    """
    # Load the attribution long-frame
    long_frame_path = ATTRIBUTION_RESULTS_DIR / "imports-and-mentions-long.parquet"
    if not long_frame_path.exists():
        raise FileNotFoundError(
            f"Attribution long-frame not found at {long_frame_path}. "
            "Run attribution-of-software.py first to generate it."
        )
    long_df = pl.read_parquet(long_frame_path)

    # Load the atypicality scores
    atypicality_path = ATYPICALITY_RESULTS_DIR / "article-atypicality-scores.parquet"
    if not atypicality_path.exists():
        raise FileNotFoundError(
            f"Atypicality scores not found at {atypicality_path}. "
            "Run software-atypicality-and-article-impact.py first to generate them."
        )
    atypicality_df = pl.read_parquet(atypicality_path)

    # Compute per-paper mention fractions from attribution long-frame
    imported_df = long_df.filter(pl.col("is_imported"))
    paper_mention_df = (
        imported_df.group_by("document_id")
        .agg(
            n_imported=pl.len(),
            n_mentioned=pl.col("is_mentioned").sum(),
        )
        .with_columns(
            fraction_mentioned=(pl.col("n_mentioned") / pl.col("n_imported")),
        )
    )
    print(f"Papers with import data: {len(paper_mention_df)}")

    # Select atypicality columns
    atypicality_selected = atypicality_df.select(
        "document_id",
        "document_atypicality_score",
        "document_atypicality_z_score",
        "ecosystem_label",
    ).unique(subset="document_id")
    print(f"Papers with atypicality scores: {len(atypicality_selected)}")

    # Inner join: papers with both mention data and atypicality scores
    bridge_df = paper_mention_df.join(
        atypicality_selected,
        on="document_id",
        how="inner",
    )
    print(f"Papers in bridge (inner join): {len(bridge_df)}")

    # Add control variables from pair_metadata
    bridge_df = bridge_df.join(
        pair_metadata.select(
            "document_id",
            "document_years_since_earliest",
            "document_author_count",
            "document_log_author_mean_citations",
            "document_field_name_pruned",
        ),
        on="document_id",
        how="inner",
    ).drop_nulls()
    print(f"Papers after joining controls and dropping nulls: {len(bridge_df)}")

    return bridge_df


def _descriptive_stats(bridge_df: pl.DataFrame) -> None:
    """Print and save descriptive statistics."""
    print()
    print("=" * 60)
    print("Descriptive Statistics")
    print("=" * 60)

    print(f"Total papers: {len(bridge_df)}")
    print(f"Mean fraction_mentioned: {bridge_df.get_column('fraction_mentioned').mean():.4f}")
    print(
        f"Median fraction_mentioned: {bridge_df.get_column('fraction_mentioned').median():.4f}"
    )
    print(
        f"Mean atypicality z-score: "
        f"{bridge_df.get_column('document_atypicality_z_score').mean():.4f}"
    )

    # By ecosystem
    stats_rows = []
    for eco in ["all", *sorted(bridge_df.get_column("ecosystem_label").unique().to_list())]:
        if eco == "all":
            subset = bridge_df
        else:
            subset = bridge_df.filter(pl.col("ecosystem_label") == eco)

        stats_rows.append(
            {
                "ecosystem": eco,
                "n_papers": len(subset),
                "mean_fraction_mentioned": subset.get_column("fraction_mentioned").mean(),
                "median_fraction_mentioned": subset.get_column("fraction_mentioned").median(),
                "mean_n_imported": subset.get_column("n_imported").mean(),
                "mean_atypicality_z": subset.get_column("document_atypicality_z_score").mean(),
                "std_atypicality_z": subset.get_column("document_atypicality_z_score").std(),
            }
        )
        if eco != "all":
            print(
                f"  {eco}: N={len(subset)}, "
                f"mean_frac={subset.get_column('fraction_mentioned').mean():.4f}"
            )

    pl.DataFrame(stats_rows).write_csv(RESULTS_DIR / "descriptive-stats.csv")
    print("Descriptive stats saved.")


def _collinearity_diagnostics(regression_pd: pd.DataFrame) -> None:
    """Compute and save collinearity diagnostics."""
    atyp_n_corr, atyp_n_pval = stats.pearsonr(
        regression_pd["document_atypicality_z_score"],
        regression_pd["n_imported"],
    )
    print(f"Pearson r(atypicality_z, n_imported) = {atyp_n_corr:.4f}, p = {atyp_n_pval:.2e}")

    continuous_controls = [
        "document_atypicality_z_score",
        "document_years_since_earliest",
        "n_imported",
        "document_author_count",
        "document_log_author_mean_citations",
    ]
    vif_x = sm.add_constant(regression_pd[continuous_controls])
    assert isinstance(vif_x, pd.DataFrame)
    vif_results = {
        vif_x.columns[i]: variance_inflation_factor(vif_x.values, i)
        for i in range(vif_x.shape[1])
    }

    collinearity_lines = [
        "Collinearity Diagnostics: Attribution x Atypicality Bridge",
        "=" * 60,
        "",
        "Pearson correlation (document_atypicality_z_score, n_imported):",
        f"  r = {atyp_n_corr:.4f}, p = {atyp_n_pval:.2e}",
        "",
        "Variance Inflation Factors (continuous predictors in controlled model):",
    ]
    for var, vif_val in vif_results.items():
        if var == "const":
            continue
        collinearity_lines.append(f"  {var}: {vif_val:.2f}")

    (RESULTS_DIR / "collinearity-diagnostics.txt").write_text("\n".join(collinearity_lines))
    print("Collinearity diagnostics saved.")


def _run_fractional_logit_regressions(bridge_df: pl.DataFrame) -> None:
    """Run fractional logit (GLM Binomial) regressions."""
    # Convert to pandas for statsmodels
    model_cols = [
        "fraction_mentioned",
        "document_atypicality_z_score",
        "ecosystem_label",
        "document_years_since_earliest",
        "n_imported",
        "document_author_count",
        "document_log_author_mean_citations",
        "document_field_name_pruned",
    ]
    regression_pd = bridge_df.select(model_cols).drop_nulls().to_pandas()

    print()
    print("=" * 60)
    print("Fractional Logit Regressions")
    print("=" * 60)
    print(f"N observations: {len(regression_pd)}")

    # Collinearity diagnostics
    _collinearity_diagnostics(regression_pd)

    # Define ecosystem groups
    ecosystem_groups = {"combined": regression_pd}
    for eco in sorted(regression_pd["ecosystem_label"].unique()):
        ecosystem_groups[eco] = regression_pd[regression_pd["ecosystem_label"] == eco]

    all_summary_rows = []

    for eco_label, eco_pd in ecosystem_groups.items():
        print(f"\n--- Ecosystem: {eco_label} (N={len(eco_pd)}) ---")

        ecosystem_suffix = " + C(ecosystem_label)" if eco_label == "combined" else ""

        # Raw model
        raw_formula = f"fraction_mentioned ~ document_atypicality_z_score{ecosystem_suffix}"
        raw_model = smf.glm(
            raw_formula,
            data=eco_pd,
            family=sm.families.Binomial(),
        ).fit(cov_type="HC1")
        (RESULTS_DIR / f"glm-raw-{eco_label}.txt").write_text(raw_model.summary().as_text())

        # Controlled model
        controlled_formula = (
            f"fraction_mentioned ~ document_atypicality_z_score"
            f" + document_years_since_earliest"
            f" + n_imported"
            f" + document_author_count"
            f" + document_log_author_mean_citations"
            f" + C(document_field_name_pruned)"
            f"{ecosystem_suffix}"
        )
        controlled_model = smf.glm(
            controlled_formula,
            data=eco_pd,
            family=sm.families.Binomial(),
        ).fit(cov_type="HC1")
        (RESULTS_DIR / f"glm-controlled-{eco_label}.txt").write_text(
            controlled_model.summary().as_text()
        )

        # Extract coefficients
        predictor = "document_atypicality_z_score"
        for model, model_name in [(raw_model, "raw"), (controlled_model, "controlled")]:
            conf_int = model.conf_int()
            all_summary_rows.append(
                {
                    "ecosystem": eco_label,
                    "model_type": model_name,
                    "n_obs": int(model.nobs),
                    "predictor": predictor,
                    "coefficient": model.params[predictor],
                    "std_err": model.bse[predictor],
                    "p_value": model.pvalues[predictor],
                    "ci_lower": conf_int.loc[predictor, 0],
                    "ci_upper": conf_int.loc[predictor, 1],
                }
            )

    summary_df = pl.DataFrame(all_summary_rows)
    summary_df.write_csv(RESULTS_DIR / "modeling-summary-stats.csv")
    print()
    print("Modeling summary stats:")
    print(summary_df)

    # Visualizations
    _plot_coefficient_forest(summary_df)


def _plot_coefficient_forest(summary_df: pl.DataFrame) -> None:
    """Forest plot of atypicality coefficients across ecosystems."""
    ecosystems = sorted(summary_df.get_column("ecosystem").unique().to_list())
    model_types = ["raw", "controlled"]
    colors = {
        "raw": "#2c7bb6",
        "controlled": "#d7191c",
    }

    _fig, ax = plt.subplots(figsize=(8, max(4, len(ecosystems) * 1.0)))
    offset = 0.15

    for i, eco in enumerate(ecosystems):
        for j, model_type in enumerate(model_types):
            rows = summary_df.filter(
                (pl.col("ecosystem") == eco) & (pl.col("model_type") == model_type)
            ).to_dicts()
            if not rows:
                continue
            row = rows[0]
            y_pos = i + (j - (len(model_types) - 1) / 2) * offset
            ax.errorbar(
                row["coefficient"],
                y_pos,
                xerr=[
                    [row["coefficient"] - row["ci_lower"]],
                    [row["ci_upper"] - row["coefficient"]],
                ],
                fmt="o",
                color=colors[model_type],
                capsize=4,
                label=model_type if i == 0 else None,
            )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(ecosystems)))
    ax.set_yticklabels(
        [
            e.upper() if e not in ("combined", "cross-ecosystem") else e.title()
            for e in ecosystems
        ]
    )
    ax.set_xlabel("GLM Binomial coefficient for atypicality z-score (95% CI)")
    ax.set_title(
        "Attribution x Atypicality: Effect of Software Atypicality\non Mention Fraction"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "coefficient-forest-plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Coefficient forest plot saved.")


def _plot_binned_scatter(bridge_df: pl.DataFrame) -> None:
    """Binned scatter: atypicality z-score deciles vs mean fraction_mentioned."""
    ecosystems = sorted(bridge_df.get_column("ecosystem_label").unique().to_list())

    _fig, axes = plt.subplots(1, len(ecosystems), figsize=(5 * len(ecosystems), 5), sharey=True)
    if len(ecosystems) == 1:
        axes = [axes]

    for ax, eco in zip(axes, ecosystems, strict=True):
        eco_df = bridge_df.filter(pl.col("ecosystem_label") == eco)

        # Create decile bins
        eco_pd = eco_df.select("document_atypicality_z_score", "fraction_mentioned").to_pandas()
        eco_pd["atypicality_decile"] = pd.qcut(
            eco_pd["document_atypicality_z_score"], 10, labels=False, duplicates="drop"
        )
        binned = (
            eco_pd.groupby("atypicality_decile")
            .agg(
                mean_atypicality=("document_atypicality_z_score", "mean"),
                mean_fraction=("fraction_mentioned", "mean"),
                count=("fraction_mentioned", "count"),
            )
            .reset_index()
        )

        ax.scatter(
            binned["mean_atypicality"],
            binned["mean_fraction"],
            s=binned["count"] / binned["count"].max() * 200,
            color="#2c7bb6",
            edgecolor="white",
            alpha=0.8,
        )

        # Add trend line
        z = np.polyfit(binned["mean_atypicality"], binned["mean_fraction"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(
            binned["mean_atypicality"].min(), binned["mean_atypicality"].max(), 50
        )
        ax.plot(x_line, p(x_line), "--", color="#d7191c", alpha=0.7)

        ax.set_xlabel("Atypicality z-score (decile mean)")
        ax.set_ylabel("Mean fraction mentioned")
        ax.set_title(f"{eco.title()} (N={len(eco_df)})")
        ax.grid(ls="--", lw=0.5)

    plt.suptitle(
        "Software atypicality vs mention fraction (binned scatter)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "atypicality-vs-mention-fraction-binned.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Binned scatter plot saved.")


def _plot_mention_fraction_by_tercile(bridge_df: pl.DataFrame) -> None:
    """Boxplot of fraction_mentioned by atypicality tercile, per ecosystem."""
    ecosystems = sorted(bridge_df.get_column("ecosystem_label").unique().to_list())

    _fig, axes = plt.subplots(1, len(ecosystems), figsize=(5 * len(ecosystems), 5), sharey=True)
    if len(ecosystems) == 1:
        axes = [axes]

    tercile_labels = ["Low", "Medium", "High"]

    for ax, eco in zip(axes, ecosystems, strict=True):
        eco_df = bridge_df.filter(pl.col("ecosystem_label") == eco)
        eco_pd = eco_df.select("document_atypicality_z_score", "fraction_mentioned").to_pandas()

        eco_pd["atypicality_tercile"] = pd.qcut(
            eco_pd["document_atypicality_z_score"],
            3,
            labels=tercile_labels,
            duplicates="drop",
        )

        data_for_box = [
            eco_pd[eco_pd["atypicality_tercile"] == label]["fraction_mentioned"].values
            for label in tercile_labels
        ]
        bp = ax.boxplot(
            data_for_box,
            labels=tercile_labels,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "#d7191c", "markersize": 6},
        )
        colors_box = ["#abdda4", "#ffffbf", "#fdae61"]
        for patch, color in zip(bp["boxes"], colors_box, strict=True):
            patch.set_facecolor(color)

        # Add N labels
        for i, label in enumerate(tercile_labels):
            n = len(eco_pd[eco_pd["atypicality_tercile"] == label])
            ax.text(i + 1, ax.get_ylim()[1] * 0.95, f"n={n:,}", ha="center", fontsize=8)

        ax.set_xlabel("Atypicality tercile")
        ax.set_ylabel("Fraction of imports mentioned")
        ax.set_title(f"{eco.title()} (N={len(eco_df)})")
        ax.grid(axis="y", ls="--", lw=0.5)

    plt.suptitle(
        "Mention fraction by software atypicality tercile",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "mention-fraction-by-atypicality-tercile.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Mention fraction by tercile plot saved.")


###############################################################################
# CLI entry point
###############################################################################


@app.command()
def main(
    top_n_fields: int = 5,
    sample: bool = False,
    sample_size: int = 5000,
) -> None:
    load_dotenv()
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load pair metadata (for control variables)
    pair_metadata = _load_our_dataset(top_n_fields=top_n_fields)

    # Take a sample to speed up development
    if sample:
        pair_metadata = pair_metadata.sample(sample_size, seed=42)

    print(f"Loaded {len(pair_metadata)} article-repository pairs")

    # Build the bridge dataframe
    bridge_df = _build_bridge_dataframe(pair_metadata)

    # Descriptive statistics
    _descriptive_stats(bridge_df)

    # Visualizations
    _plot_binned_scatter(bridge_df)
    _plot_mention_fraction_by_tercile(bridge_df)

    # Fractional logit regressions
    _run_fractional_logit_regressions(bridge_df)


###############################################################################

if __name__ == "__main__":
    app()
