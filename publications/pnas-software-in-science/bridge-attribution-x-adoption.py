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
RESULTS_DIR = THIS_DIR / "results" / "bridge-attribution-x-adoption"

ATTRIBUTION_RESULTS_DIR = THIS_DIR / "results" / "attribution-of-software"

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
# Library age computation (adapted from first-mover-advantage.py)
###############################################################################


def _get_imported_software(
    repository_ids: list[int],
    remove_extremely_rare_imports: bool = True,
    rare_import_threshold: int = 3,
) -> pl.DataFrame:
    repository_imports = load_table("repository_import")

    # Filter to only the repositories in our dataset
    repository_imports = repository_imports.filter(
        pl.col("repository_id").is_in(repository_ids)
    )

    # Determine ecosystem from file extensions
    check_for_py = (
        pl.col("file_paths_lower").str.contains(r"\.py\;")
        | pl.col("file_paths_lower").str.contains(r"\.py$")
        | pl.col("file_paths_lower").str.contains(r"\.ipynb\;")
        | pl.col("file_paths_lower").str.contains(r"\.ipynb$")
    )
    check_for_r = (
        pl.col("file_paths_lower").str.contains(r"\.r\;")
        | pl.col("file_paths_lower").str.contains(r"\.r$")
        | pl.col("file_paths_lower").str.contains(r"\.rmd\;")
        | pl.col("file_paths_lower").str.contains(r"\.rmd$")
    )

    repository_imports = (
        repository_imports.with_columns(
            pl.col("file_paths").str.to_lowercase().alias("file_paths_lower")
        )
        .with_columns(
            pl.when(check_for_py & check_for_r)
            .then(pl.lit("mixed"))
            .when(check_for_py)
            .then(pl.lit("py"))
            .when(check_for_r)
            .then(pl.lit("r"))
            .otherwise(pl.lit("other"))
            .alias("ecosystem")
        )
        .with_columns(
            (pl.col("ecosystem") + pl.lit(":") + pl.col("software_name_normalized")).alias(
                "ecosystem_normalized_software_name"
            )
        )
    )

    # Drop "mixed" and "other" ecosystems
    repository_imports = repository_imports.filter(pl.col("ecosystem").is_in(["py", "r"]))

    # Remove any imports that were imported less than threshold times
    if remove_extremely_rare_imports:
        import_counts = repository_imports.group_by("ecosystem_normalized_software_name").agg(
            pl.len().alias("import_count")
        )

        pre_filter_unique_package_count = repository_imports.get_column(
            "ecosystem_normalized_software_name"
        ).n_unique()
        non_rare_imports = (
            import_counts.filter(pl.col("import_count") >= rare_import_threshold)
            .get_column("ecosystem_normalized_software_name")
            .to_list()
        )

        repository_imports = repository_imports.filter(
            pl.col("ecosystem_normalized_software_name").is_in(non_rare_imports)
        )
        post_filter_unique_package_count = repository_imports.get_column(
            "ecosystem_normalized_software_name"
        ).n_unique()

        print(
            f"Removed {pre_filter_unique_package_count - post_filter_unique_package_count} "
            f"unique packages (below {rare_import_threshold} uses)"
        )
        print(
            f"Number of remaining unique imported packages: {post_filter_unique_package_count}"
        )

    return repository_imports


def _compute_library_metadata(
    repository_imports: pl.DataFrame,
    pair_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute library birth year and total adopters for each library.

    Uses ALL papers (before adopter-count threshold) so birth year reflects
    the library's true first appearance in the dataset.
    """
    import_with_year = (
        repository_imports.select("repository_id", "ecosystem_normalized_software_name")
        .join(
            pair_metadata.select("document_id", "repository_id", "document_publication_year"),
            on="repository_id",
            how="inner",
        )
        .unique(subset=["document_id", "ecosystem_normalized_software_name"])
    )

    library_metadata = import_with_year.group_by("ecosystem_normalized_software_name").agg(
        pl.col("document_publication_year").min().alias("library_birth_year"),
        pl.col("document_id").n_unique().alias("total_adopters"),
    )

    return library_metadata


def _compute_library_age_stats(
    repository_imports: pl.DataFrame,
    pair_metadata: pl.DataFrame,
    library_metadata: pl.DataFrame,
    min_adopters_per_library: int = 10,
    young_library_threshold: int = 2,
) -> pl.DataFrame:
    """
    For each (document, library) pair, compute how old the library was
    when the paper used it.
    """
    # Build deduplicated import-with-year rows
    import_with_year = (
        repository_imports.select("repository_id", "ecosystem_normalized_software_name")
        .join(
            pair_metadata.select("document_id", "repository_id", "document_publication_year"),
            on="repository_id",
            how="inner",
        )
        .unique(subset=["document_id", "ecosystem_normalized_software_name"])
    )

    # Join library metadata
    import_with_year = import_with_year.join(
        library_metadata,
        on="ecosystem_normalized_software_name",
        how="inner",
    )

    # Compute library age at use
    import_with_year = import_with_year.with_columns(
        (pl.col("document_publication_year") - pl.col("library_birth_year")).alias(
            "library_age_at_use"
        ),
    )

    # Filter to libraries with enough adopters
    pre_count = import_with_year.get_column("ecosystem_normalized_software_name").n_unique()
    import_with_year = import_with_year.filter(
        pl.col("total_adopters") >= min_adopters_per_library
    )
    post_count = import_with_year.get_column("ecosystem_normalized_software_name").n_unique()
    print(f"Libraries with >= {min_adopters_per_library} adopters: {post_count} / {pre_count}")

    # Add young library flag
    import_with_year = import_with_year.with_columns(
        (pl.col("library_age_at_use") <= young_library_threshold).alias("is_young_library"),
    )

    return import_with_year.select(
        "document_id",
        "ecosystem_normalized_software_name",
        "library_age_at_use",
        "is_young_library",
        "library_birth_year",
        "total_adopters",
    )


###############################################################################
# Bridge analysis functions
###############################################################################


def _build_bridge_dataframe(
    pair_metadata: pl.DataFrame,
    library_age_stats: pl.DataFrame,
) -> pl.DataFrame:
    """
    Load the attribution long-frame parquet and join with library age data.
    Returns a (document, library) level dataframe with is_mentioned and age.
    """
    # Load the attribution long-frame
    long_frame_path = ATTRIBUTION_RESULTS_DIR / "imports-and-mentions-long.parquet"
    if not long_frame_path.exists():
        raise FileNotFoundError(
            f"Attribution long-frame not found at {long_frame_path}. "
            "Run attribution-of-software.py first to generate it."
        )
    long_df = pl.read_parquet(long_frame_path)

    # Filter to imported rows only (mentions-only rows have ecosystem=None)
    imported_df = long_df.filter(pl.col("is_imported"))
    print(f"Attribution long-frame imported rows: {len(imported_df)}")

    # Create the join key: ecosystem + ":" + library_name_normalized
    # to match first-mover's ecosystem_normalized_software_name
    imported_df = imported_df.with_columns(
        (pl.col("ecosystem") + pl.lit(":") + pl.col("library_name_normalized")).alias(
            "ecosystem_normalized_software_name"
        )
    )

    # Compute per-document number of imports (for controls)
    doc_import_counts = imported_df.group_by("document_id").agg(
        n_imported=pl.col("library_name_normalized").n_unique()
    )

    # Print the software that are most common in both frames
    # Added for debug printing to make sure that the two strings follow the same pattern
    # as we are joining on them and they are generated from different code paths
    # top_software_in_imported_df = (
    #     imported_df.group_by("ecosystem_normalized_software_name")
    #     .agg(pl.len().alias("count_in_imported_df"))
    #     .sort("count_in_imported_df", descending=True)
    # )
    # print("Top software in imported dataframe:")
    # print(top_software_in_imported_df.head(10))

    # top_software_in_library_age_stats = (
    #     library_age_stats.group_by("ecosystem_normalized_software_name")
    #     .agg(pl.len().alias("count_in_library_age_stats"))
    #     .sort("count_in_library_age_stats", descending=True)
    # )
    # print("Top software in library age stats dataframe:")
    # print(top_software_in_library_age_stats.head(10))

    # Join with library age stats on (document_id, ecosystem_normalized_software_name)
    bridge_df = imported_df.join(
        library_age_stats,
        on=["document_id", "ecosystem_normalized_software_name"],
        how="inner",
    )
    print(
        f"After joining with library age data: {len(bridge_df)} (document, library) pairs "
        f"({bridge_df.get_column('document_id').n_unique()} unique documents)"
    )

    # Add log_total_adopters
    bridge_df = bridge_df.with_columns(
        pl.col("total_adopters").cast(pl.Float64).log().alias("log_total_adopters"),
    )

    # Add per-document import count and metadata
    bridge_df = bridge_df.join(doc_import_counts, on="document_id").join(
        pair_metadata.select(
            "document_id",
            "document_years_since_earliest",
            "document_author_count",
            "document_log_author_mean_citations",
            "document_field_name_pruned",
        ),
        on="document_id",
    )

    return bridge_df


def _descriptive_stats(bridge_df: pl.DataFrame) -> None:
    """Print and save descriptive statistics."""
    print()
    print("=" * 60)
    print("Descriptive Statistics")
    print("=" * 60)

    # Overall mention rate
    overall_mention_rate = bridge_df.get_column("is_mentioned").mean()
    print(f"Overall mention rate: {overall_mention_rate:.4f}")

    # By ecosystem
    for eco in sorted(bridge_df.get_column("ecosystem").unique().to_list()):
        eco_df = bridge_df.filter(pl.col("ecosystem") == eco)
        mention_rate = eco_df.get_column("is_mentioned").mean()
        print(f"  {eco}: N={len(eco_df)}, mention rate={mention_rate:.4f}")

    # Library age distribution
    print(
        f"\nLibrary age at use: "
        f"mean={bridge_df.get_column('library_age_at_use').mean():.2f}, "
        f"median={bridge_df.get_column('library_age_at_use').median():.1f}, "
        f"std={bridge_df.get_column('library_age_at_use').std():.2f}"
    )

    # Young library fraction
    young_frac = bridge_df.get_column("is_young_library").mean()
    print(f"Fraction young libraries (age <= threshold): {young_frac:.4f}")

    # Summary stats CSV
    stats_rows = []
    for eco in ["all", *sorted(bridge_df.get_column("ecosystem").unique().to_list())]:
        if eco == "all":
            subset = bridge_df
        else:
            subset = bridge_df.filter(pl.col("ecosystem") == eco)
        stats_rows.append(
            {
                "ecosystem": eco,
                "n_pairs": len(subset),
                "n_documents": subset.get_column("document_id").n_unique(),
                "n_unique_libraries": subset.get_column(
                    "ecosystem_normalized_software_name"
                ).n_unique(),
                "mention_rate": subset.get_column("is_mentioned").mean(),
                "mean_library_age": subset.get_column("library_age_at_use").mean(),
                "median_library_age": subset.get_column("library_age_at_use").median(),
                "frac_young_libraries": subset.get_column("is_young_library").mean(),
            }
        )
    pl.DataFrame(stats_rows).write_csv(RESULTS_DIR / "descriptive-stats.csv")
    print("Descriptive stats saved.")


def _plot_mention_rate_by_age_bin(bridge_df: pl.DataFrame) -> None:
    """Bar chart of mention rate by library age bins, faceted by ecosystem."""
    # Create age bins
    bridge_with_bins = bridge_df.with_columns(
        pl.when(pl.col("library_age_at_use") == 0)
        .then(pl.lit("0"))
        .when(pl.col("library_age_at_use") <= 2)
        .then(pl.lit("1-2"))
        .when(pl.col("library_age_at_use") <= 5)
        .then(pl.lit("3-5"))
        .when(pl.col("library_age_at_use") <= 10)
        .then(pl.lit("6-10"))
        .otherwise(pl.lit("11+"))
        .alias("age_bin")
    )

    bin_order = ["0", "1-2", "3-5", "6-10", "11+"]
    ecosystems = sorted(bridge_df.get_column("ecosystem").unique().to_list())

    _fig, axes = plt.subplots(1, len(ecosystems), figsize=(5 * len(ecosystems), 5), sharey=True)
    if len(ecosystems) == 1:
        axes = [axes]

    for ax, eco in zip(axes, ecosystems, strict=True):
        eco_df = bridge_with_bins.filter(pl.col("ecosystem") == eco)
        rates = []
        counts = []
        for bin_label in bin_order:
            bin_df = eco_df.filter(pl.col("age_bin") == bin_label)
            if len(bin_df) > 0:
                rates.append(bin_df.get_column("is_mentioned").mean())
                counts.append(len(bin_df))
            else:
                rates.append(0)
                counts.append(0)

        bars = ax.bar(bin_order, rates, color="#2c7bb6", edgecolor="white")
        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"n={count:,}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="gray",
            )

        ax.set_xlabel("Library age at use (years)")
        ax.set_ylabel("p(mention | import)")
        ax.set_title(f"{eco.upper()} ecosystem")
        ax.grid(axis="y", ls="--", lw=0.5)

    plt.suptitle("Mention rate by library age at use", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "mention-rate-by-age-bin.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Mention rate by age bin plot saved.")


def _plot_young_vs_old_mention_rate(bridge_df: pl.DataFrame) -> None:
    """Plot grouped bar chart comparing mention rate for young vs old libraries."""
    ecosystems = sorted(bridge_df.get_column("ecosystem").unique().to_list())

    young_rates = []
    old_rates = []
    young_counts = []
    old_counts = []
    for eco in ecosystems:
        eco_df = bridge_df.filter(pl.col("ecosystem") == eco)
        young = eco_df.filter(pl.col("is_young_library"))
        old = eco_df.filter(~pl.col("is_young_library"))
        young_rates.append(young.get_column("is_mentioned").mean() if len(young) > 0 else 0)
        old_rates.append(old.get_column("is_mentioned").mean() if len(old) > 0 else 0)
        young_counts.append(len(young))
        old_counts.append(len(old))

    x = np.arange(len(ecosystems))
    width = 0.35

    _fig, ax = plt.subplots(figsize=(6, 5))
    bars_young = ax.bar(x - width / 2, young_rates, width, label="Young", color="#fdae61")
    bars_old = ax.bar(x + width / 2, old_rates, width, label="Old", color="#2c7bb6")

    for bar, count in zip(bars_young, young_counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"n={count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, count in zip(bars_old, old_counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"n={count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Ecosystem")
    ax.set_ylabel("p(mention | import)")
    ax.set_xticks(x)
    ax.set_xticklabels([e.upper() for e in ecosystems])
    ax.set_title("Mention rate: Young vs Old libraries")
    ax.legend()
    ax.grid(axis="y", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "young-vs-old-mention-rate.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Young vs old mention rate plot saved.")


def _collinearity_diagnostics(regression_pd: pd.DataFrame) -> None:
    """Compute and save collinearity diagnostics."""
    age_pop_corr, age_pop_pval = stats.pearsonr(
        regression_pd["library_age_at_use"],
        regression_pd["log_total_adopters"],
    )
    print(
        f"Pearson r(library_age_at_use, log_total_adopters) = {age_pop_corr:.4f}, "
        f"p = {age_pop_pval:.2e}"
    )

    continuous_controls = [
        "library_age_at_use",
        "log_total_adopters",
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
        "Collinearity Diagnostics: Attribution x Adoption Bridge",
        "=" * 60,
        "",
        "Pearson correlation (library_age_at_use, log_total_adopters):",
        f"  r = {age_pop_corr:.4f}, p = {age_pop_pval:.2e}",
        "",
        "Variance Inflation Factors (continuous predictors in controlled model):",
    ]
    for var, vif_val in vif_results.items():
        if var == "const":
            continue
        collinearity_lines.append(f"  {var}: {vif_val:.2f}")

    (RESULTS_DIR / "collinearity-diagnostics.txt").write_text("\n".join(collinearity_lines))
    print("Collinearity diagnostics saved.")


def _run_logistic_regressions(bridge_df: pl.DataFrame) -> None:
    """Run logistic regressions: raw, controlled, binary, single-predictor models."""
    # Convert to pandas for statsmodels
    model_cols = [
        "is_mentioned",
        "library_age_at_use",
        "is_young_library",
        "log_total_adopters",
        "ecosystem",
        "document_years_since_earliest",
        "n_imported",
        "document_author_count",
        "document_log_author_mean_citations",
        "document_field_name_pruned",
        "ecosystem_normalized_software_name",
    ]
    regression_pd = (
        bridge_df.select(model_cols)
        .with_columns(
            pl.col("is_mentioned").cast(pl.Int8),
            pl.col("is_young_library").cast(pl.Int8),
        )
        .drop_nulls()
        .to_pandas()
    )

    print()
    print("=" * 60)
    print("Logistic Regressions")
    print("=" * 60)
    print(f"N observations: {len(regression_pd)}")
    print(
        f"N unique libraries: {regression_pd['ecosystem_normalized_software_name'].nunique()}"
    )
    print(f"N unique documents: {regression_pd['is_mentioned'].count()}")

    # Collinearity diagnostics
    _collinearity_diagnostics(regression_pd)

    # Define ecosystem groups: combined + per-ecosystem
    ecosystem_groups = {"combined": regression_pd}
    for eco in sorted(regression_pd["ecosystem"].unique()):
        ecosystem_groups[eco] = regression_pd[regression_pd["ecosystem"] == eco]

    all_summary_rows = []

    for eco_label, eco_pd in ecosystem_groups.items():
        print(f"\n--- Ecosystem: {eco_label} (N={len(eco_pd)}) ---")

        ecosystem_suffix = " + C(ecosystem)" if eco_label == "combined" else ""

        # Raw model
        raw_formula = f"is_mentioned ~ library_age_at_use{ecosystem_suffix}"
        raw_model = smf.logit(raw_formula, data=eco_pd).fit(
            cov_type="cluster",
            cov_kwds={"groups": eco_pd["ecosystem_normalized_software_name"]},
            maxiter=1000,
            disp=0,
        )
        (RESULTS_DIR / f"logit-raw-{eco_label}.txt").write_text(raw_model.summary().as_text())

        # Controlled model
        controlled_formula = (
            f"is_mentioned ~ library_age_at_use + log_total_adopters"
            f" + document_years_since_earliest"
            f" + n_imported"
            f" + document_author_count"
            f" + document_log_author_mean_citations"
            f" + C(document_field_name_pruned)"
            f"{ecosystem_suffix}"
        )
        controlled_model = smf.logit(controlled_formula, data=eco_pd).fit(
            cov_type="cluster",
            cov_kwds={"groups": eco_pd["ecosystem_normalized_software_name"]},
            maxiter=1000,
            disp=0,
        )
        (RESULTS_DIR / f"logit-controlled-{eco_label}.txt").write_text(
            controlled_model.summary().as_text()
        )

        # Binary model (is_young_library instead of library_age_at_use)
        binary_formula = (
            f"is_mentioned ~ is_young_library + log_total_adopters"
            f" + document_years_since_earliest"
            f" + n_imported"
            f" + document_author_count"
            f" + document_log_author_mean_citations"
            f" + C(document_field_name_pruned)"
            f"{ecosystem_suffix}"
        )
        binary_model = smf.logit(binary_formula, data=eco_pd).fit(
            cov_type="cluster",
            cov_kwds={"groups": eco_pd["ecosystem_normalized_software_name"]},
            maxiter=1000,
            disp=0,
        )
        (RESULTS_DIR / f"logit-binary-{eco_label}.txt").write_text(
            binary_model.summary().as_text()
        )

        # Single-predictor: age only
        age_only_formula = f"is_mentioned ~ library_age_at_use{ecosystem_suffix}"
        age_only_model = smf.logit(age_only_formula, data=eco_pd).fit(
            cov_type="cluster",
            cov_kwds={"groups": eco_pd["ecosystem_normalized_software_name"]},
            maxiter=1000,
            disp=0,
        )
        (RESULTS_DIR / f"logit-single-age-{eco_label}.txt").write_text(
            age_only_model.summary().as_text()
        )

        # Single-predictor: popularity only
        pop_only_formula = f"is_mentioned ~ log_total_adopters{ecosystem_suffix}"
        pop_only_model = smf.logit(pop_only_formula, data=eco_pd).fit(
            cov_type="cluster",
            cov_kwds={"groups": eco_pd["ecosystem_normalized_software_name"]},
            maxiter=1000,
            disp=0,
        )
        (RESULTS_DIR / f"logit-single-popularity-{eco_label}.txt").write_text(
            pop_only_model.summary().as_text()
        )

        # Extract key coefficients for summary CSV
        models_and_predictors = [
            (raw_model, "raw", ["library_age_at_use"]),
            (controlled_model, "controlled", ["library_age_at_use", "log_total_adopters"]),
            (binary_model, "binary", ["is_young_library", "log_total_adopters"]),
            (age_only_model, "age_only", ["library_age_at_use"]),
            (pop_only_model, "popularity_only", ["log_total_adopters"]),
        ]
        for model, model_name, predictors in models_and_predictors:
            conf_int = model.conf_int()
            for predictor in predictors:
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

    # Coefficient forest plot
    _plot_coefficient_forest(summary_df)


def _plot_coefficient_forest(summary_df: pl.DataFrame) -> None:
    """Forest plot of library_age_at_use coefficients across ecosystems and model types."""
    # Filter to library_age_at_use predictor only
    age_coefs = summary_df.filter(pl.col("predictor") == "library_age_at_use")

    ecosystems = sorted(age_coefs.get_column("ecosystem").unique().to_list())
    model_types = ["raw", "controlled", "age_only"]
    colors = {
        "raw": "#2c7bb6",
        "controlled": "#d7191c",
        "age_only": "#fdae61",
    }

    _fig, ax = plt.subplots(figsize=(8, max(4, len(ecosystems) * 1.2)))
    offset = 0.15

    for i, eco in enumerate(ecosystems):
        for j, model_type in enumerate(model_types):
            rows = age_coefs.filter(
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
    ax.set_yticklabels([e.upper() if e != "combined" else "Combined" for e in ecosystems])
    ax.set_xlabel("Logit coefficient for library_age_at_use (95% CI)")
    ax.set_title("Attribution x Adoption: Effect of Library Age on Mention Probability")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "coefficient-forest-plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Coefficient forest plot saved.")


###############################################################################
# CLI entry point
###############################################################################


@app.command()
def main(
    rare_import_threshold: int = 3,
    min_adopters_per_library: int = 10,
    young_library_threshold: int = 2,
    top_n_fields: int = 5,
    sample: bool = False,
    sample_size: int = 5000,
) -> None:
    load_dotenv()
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load pair metadata
    pair_metadata = _load_our_dataset(top_n_fields=top_n_fields)

    # Take a sample to speed up development
    if sample:
        pair_metadata = pair_metadata.sample(sample_size, seed=42)

    print(f"Loaded {len(pair_metadata)} article-repository pairs")

    # Load repository imports and compute library age stats
    repository_imports = _get_imported_software(
        repository_ids=pair_metadata.get_column("repository_id").unique().to_list(),
        remove_extremely_rare_imports=True,
        rare_import_threshold=rare_import_threshold,
    )

    library_metadata = _compute_library_metadata(repository_imports, pair_metadata)
    library_age_stats = _compute_library_age_stats(
        repository_imports,
        pair_metadata,
        library_metadata,
        min_adopters_per_library=min_adopters_per_library,
        young_library_threshold=young_library_threshold,
    )

    # Build the bridge dataframe
    bridge_df = _build_bridge_dataframe(pair_metadata, library_age_stats)

    # Descriptive statistics
    _descriptive_stats(bridge_df)

    # Visualizations
    _plot_mention_rate_by_age_bin(bridge_df)
    _plot_young_vs_old_mention_rate(bridge_df)

    # Logistic regressions
    _run_logistic_regressions(bridge_df)


###############################################################################

if __name__ == "__main__":
    app()
