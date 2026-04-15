import os
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.outliers_influence import variance_inflation_factor

###############################################################################

app = typer.Typer()

THIS_FILE_PATH = Path(__file__).resolve()
THIS_DIR = THIS_FILE_PATH.parent
RESULTS_DIR = THIS_DIR / "results" / "first-mover-advantage"

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


def _load_and_filter_pairs(
    top_n_fields: int = 5,
) -> pl.DataFrame:
    pair_metadata = _load_our_dataset(top_n_fields=top_n_fields)

    # Filter to papers with non-null FWCI > 0 and at least 2 citations
    pre_filter_count = len(pair_metadata)
    pair_metadata = (
        pair_metadata.filter(pl.col("document_fwci").is_not_null())
        .filter(pl.col("document_fwci") > 0)
        .filter(pl.col("document_cited_by_count") >= 2)
    )
    post_filter_count = len(pair_metadata)
    print(
        f"Removed {pre_filter_count - post_filter_count} pairs "
        f"with null FWCI or less than 2 citations"
    )

    # Select down to only 1:1 article-repository pairs
    pair_metadata = pair_metadata.unique(
        subset="document_id",
        keep="none",
    ).unique(
        subset="repository_id",
        keep="none",
    )
    print(f"Using {len(pair_metadata)} unique article-repository pairs")

    return pair_metadata


def _get_imported_software(
    repository_ids: list[int],
    remove_extremely_rare_imports: bool = True,
    rare_import_threshold: int = 3,
) -> pl.DataFrame:
    # Load all import data
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


def _compute_library_adoption_stats(
    repository_imports: pl.DataFrame,
    pair_metadata: pl.DataFrame,
    min_adopters_per_library: int = 10,
    early_adopter_percentile_threshold: float = 0.90,
) -> pl.DataFrame:
    """
    For each (library, paper) pair, compute the paper's adoption percentile
    for that library. 100th percentile = earliest adopter, 0 = latest.

    Returns a DataFrame with columns:
        document_id, ecosystem_normalized_software_name,
        adoption_percentile, is_early_adopter, log_total_adopters
    """
    # Join imports with paper metadata to get publication year
    import_with_year = repository_imports.select(
        "repository_id",
        "ecosystem_normalized_software_name",
    ).join(
        pair_metadata.select("document_id", "repository_id", "document_publication_year"),
        on="repository_id",
        how="inner",
    )

    # Deduplicate: one row per (document, library) pair
    import_with_year = import_with_year.unique(
        subset=["document_id", "ecosystem_normalized_software_name"]
    )

    # Count adopters per library and filter to those with enough
    library_adopter_counts = import_with_year.group_by(
        "ecosystem_normalized_software_name"
    ).agg(
        pl.col("document_id").n_unique().alias("total_adopters"),
    )

    pre_filter_count = library_adopter_counts.height
    eligible_libraries = (
        library_adopter_counts.filter(pl.col("total_adopters") >= min_adopters_per_library)
        .get_column("ecosystem_normalized_software_name")
        .to_list()
    )
    post_filter_count = len(eligible_libraries)
    print(
        f"Libraries with >= {min_adopters_per_library} adopters: "
        f"{post_filter_count} / {pre_filter_count}"
    )

    import_with_year = import_with_year.filter(
        pl.col("ecosystem_normalized_software_name").is_in(eligible_libraries)
    )

    # Join total_adopters back and compute log_total_adopters (Task 2)
    import_with_year = import_with_year.join(
        library_adopter_counts,
        on="ecosystem_normalized_software_name",
        how="left",
    ).with_columns(
        pl.col("total_adopters").log().alias("log_total_adopters"),
    )

    # Compute adoption percentile per (library, document) pair
    # Rank within each library by publication year (ascending: earliest gets rank 1)
    # Then convert to percentile where earliest = 100, latest = 0
    import_with_year = (
        import_with_year.with_columns(
            pl.col("document_publication_year")
            .rank(method="average")
            .over("ecosystem_normalized_software_name")
            .alias("adoption_rank"),
            pl.col("document_id")
            .count()
            .over("ecosystem_normalized_software_name")
            .alias("n_adopters"),
        )
        .with_columns(
            # Convert rank to percentile: 100 = earliest, 0 = latest
            (
                (1.0 - (pl.col("adoption_rank") - 1.0) / (pl.col("n_adopters") - 1.0)) * 100.0
            ).alias("adoption_percentile"),
        )
        .with_columns(
            # Handle libraries where all adopters share the same year (n_adopters == rank for all)
            pl.when(pl.col("n_adopters") == 1)
            .then(pl.lit(50.0))  # single adopter gets neutral score
            .otherwise(pl.col("adoption_percentile"))
            .alias("adoption_percentile"),
        )
        .with_columns(
            (pl.col("adoption_percentile") >= early_adopter_percentile_threshold * 100.0).alias(
                "is_early_adopter"
            ),
        )
    )

    return import_with_year.select(
        "document_id",
        "ecosystem_normalized_software_name",
        "adoption_percentile",
        "is_early_adopter",
        "log_total_adopters",
    )


def _compute_paper_first_mover_scores(
    library_adoption_stats: pl.DataFrame,
) -> pl.DataFrame:
    """
    Aggregate per-library adoption percentiles to paper-level first-mover scores.

    Returns a DataFrame with columns:
        document_id, mean_adoption_percentile, max_adoption_percentile,
        frac_early_adopter, num_libraries_scored, mean_log_library_popularity
    """
    paper_scores = library_adoption_stats.group_by("document_id").agg(
        pl.col("adoption_percentile").mean().alias("mean_adoption_percentile"),
        pl.col("adoption_percentile").max().alias("max_adoption_percentile"),
        pl.col("is_early_adopter").mean().alias("frac_early_adopter"),
        pl.col("ecosystem_normalized_software_name").n_unique().alias("num_libraries_scored"),
        pl.col("log_total_adopters").mean().alias("mean_log_library_popularity"),
    )

    return paper_scores


def _compute_library_metadata(
    repository_imports: pl.DataFrame,
    pair_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute library birth year and total adopters for each library.

    Uses ALL papers (before adopter-count threshold) so birth year reflects
    the library's true first appearance in the dataset.

    Returns a DataFrame with columns:
        ecosystem_normalized_software_name, library_birth_year, total_adopters
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

    Returns a DataFrame with columns:
        document_id, ecosystem_normalized_software_name,
        library_age_at_use, is_young_library, library_birth_year, total_adopters
    """
    # Build deduplicated import-with-year rows (same pattern as _compute_library_adoption_stats)
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
    print(
        f"[Q2] Libraries with >= {min_adopters_per_library} adopters: "
        f"{post_count} / {pre_count}"
    )

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


def _compute_paper_novelty_scores(
    library_age_stats: pl.DataFrame,
) -> pl.DataFrame:
    """
    Aggregate per-library age measures to paper-level "software novelty" scores.

    Returns a DataFrame with columns:
        document_id, mean_library_age, min_library_age, frac_young_libraries,
        num_libraries_age_scored, mean_log_library_popularity_q2, software_novelty
    """
    paper_scores = library_age_stats.group_by("document_id").agg(
        pl.col("library_age_at_use").mean().alias("mean_library_age"),
        pl.col("library_age_at_use").min().alias("min_library_age"),
        pl.col("is_young_library").mean().alias("frac_young_libraries"),
        pl.col("is_young_library").any().cast(pl.Int8).alias("has_any_young_library"),
        pl.col("ecosystem_normalized_software_name")
        .n_unique()
        .alias("num_libraries_age_scored"),
        pl.col("total_adopters").log().mean().alias("mean_log_library_popularity_q2"),
    )

    # software_novelty = -mean_library_age (higher = newer toolkit = more novelty)
    paper_scores = paper_scores.with_columns(
        (-pl.col("mean_library_age")).alias("software_novelty"),
    )

    return paper_scores


def _assign_ecosystem_labels(
    repository_imports: pl.DataFrame,
    pair_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """
    Assign ecosystem labels to documents using a Polars group-by approach.

    Returns a DataFrame with columns: document_id, ecosystem_label
    """
    doc_imports = repository_imports.select(
        "repository_id", "ecosystem_normalized_software_name"
    ).join(
        pair_metadata.select("document_id", "repository_id"),
        on="repository_id",
        how="inner",
    )

    # Extract the ecosystem prefix (everything before ":")
    ecosystem_label_df = (
        doc_imports.with_columns(
            pl.col("ecosystem_normalized_software_name")
            .str.split(":")
            .list.first()
            .alias("ecosystem_prefix")
        )
        .group_by("document_id")
        .agg(pl.col("ecosystem_prefix").unique().alias("ecosystems"))
        .with_columns(
            pl.when(
                pl.col("ecosystems").list.contains("py")
                & pl.col("ecosystems").list.contains("r")
            )
            .then(pl.lit("cross-ecosystem"))
            .when(pl.col("ecosystems").list.contains("py"))
            .then(pl.lit("py"))
            .when(pl.col("ecosystems").list.contains("r"))
            .then(pl.lit("r"))
            .otherwise(pl.lit("other"))
            .alias("ecosystem_label")
        )
        .select("document_id", "ecosystem_label")
    )

    return ecosystem_label_df


def _remove_outliers_per_ecosystem(results_df: pl.DataFrame) -> pl.DataFrame:
    """Remove top 1% software count outliers per ecosystem."""
    filtered_doc_ids: list[int] = []
    for eco_label in results_df.get_column("ecosystem_label").unique():
        eco_df = results_df.filter(pl.col("ecosystem_label") == eco_label)
        threshold = eco_df.get_column("num_libraries_scored").quantile(0.99)
        eco_filtered = eco_df.filter(pl.col("num_libraries_scored") <= threshold)
        filtered_doc_ids.extend(eco_filtered.get_column("document_id").to_list())

    pre_outlier_count = len(results_df)
    results_df = results_df.filter(pl.col("document_id").is_in(filtered_doc_ids))
    print(
        f"Removed {pre_outlier_count - len(results_df)} papers "
        f"in top 1% of library count per ecosystem"
    )
    return results_df


def _zscore_per_ecosystem(results_df: pl.DataFrame) -> pl.DataFrame:
    """Z-score first-mover measures per ecosystem."""
    z_scored_parts = []
    for eco_label in results_df.get_column("ecosystem_label").unique():
        eco_df = results_df.filter(pl.col("ecosystem_label") == eco_label)

        z_cols = [
            (
                (pl.col("mean_adoption_percentile") - pl.col("mean_adoption_percentile").mean())
                / pl.col("mean_adoption_percentile").std()
            ).alias("mean_adoption_percentile_z"),
            (
                (pl.col("max_adoption_percentile") - pl.col("max_adoption_percentile").mean())
                / pl.col("max_adoption_percentile").std()
            ).alias("max_adoption_percentile_z"),
            (
                (pl.col("frac_early_adopter") - pl.col("frac_early_adopter").mean())
                / pl.col("frac_early_adopter").std()
            ).alias("frac_early_adopter_z"),
        ]

        # Q2 z-scores (only if columns exist)
        if "mean_library_age" in eco_df.columns:
            z_cols.extend(
                [
                    (
                        (pl.col("mean_library_age") - pl.col("mean_library_age").mean())
                        / pl.col("mean_library_age").std()
                    ).alias("mean_library_age_z"),
                    (
                        (pl.col("software_novelty") - pl.col("software_novelty").mean())
                        / pl.col("software_novelty").std()
                    ).alias("software_novelty_z"),
                    (
                        (pl.col("min_library_age") - pl.col("min_library_age").mean())
                        / pl.col("min_library_age").std()
                    ).alias("min_library_age_z"),
                    (
                        (pl.col("frac_young_libraries") - pl.col("frac_young_libraries").mean())
                        / pl.col("frac_young_libraries").std()
                    ).alias("frac_young_libraries_z"),
                ]
            )

        eco_df = eco_df.with_columns(z_cols)
        z_scored_parts.append(eco_df)

    return pl.concat(z_scored_parts)


def _compute_temporal_confound_correlations(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """
    Compute Pearson r between each measure and document_publication_year per ecosystem.
    Print results and save as CSV.
    """
    from scipy.stats import pearsonr

    rows = []
    variables = [
        "mean_adoption_percentile",
        "mean_library_age",
        "software_novelty",
        "frac_young_libraries",
    ]

    for eco_label in sorted(results_df.get_column("ecosystem_label").unique().to_list()):
        eco_df = results_df.filter(pl.col("ecosystem_label") == eco_label)
        pub_year = eco_df.get_column("document_publication_year").to_numpy()

        print(f"\n[Temporal confound] Ecosystem: {eco_label}")
        for var in variables:
            if var not in eco_df.columns:
                continue
            values = eco_df.get_column(var).to_numpy()
            # Drop NaN pairs
            mask = ~(np.isnan(values) | np.isnan(pub_year))
            r, _p = pearsonr(values[mask], pub_year[mask])
            print(f"  {var} vs publication_year: r = {r:.4f}")
            rows.append(
                {
                    "ecosystem": eco_label,
                    "variable": var,
                    "pearson_r_with_pub_year": r,
                }
            )

    corr_df = pl.DataFrame(rows)
    corr_df.write_csv(results_dir / "temporal-confound-correlations.csv")
    print(
        f"\nSaved temporal confound correlations to {results_dir / 'temporal-confound-correlations.csv'}"
    )


def _plot_adoption_distribution(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Plot 1: Distribution of mean adoption percentile."""
    g = sns.displot(
        results_df,
        kind="hist",
        stat="percent",
        common_norm=False,
        x="mean_adoption_percentile",
        col="ecosystem_label",
        hue="ecosystem_label",
        bins=50,
        facet_kws={"sharey": False},
    )
    g.set_titles(col_template="{col_name}")
    g.fig.savefig(
        results_dir / "adoption-percentile-distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _plot_first_mover_vs_impact(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Plot 2: Scatter + trend (mean_adoption_percentile_z vs citation impact)."""
    analysis_long = (
        results_df.select(
            "document_id",
            "document_log_cited_by_count",
            "document_log_fwci",
        )
        .unpivot(
            on=["document_log_cited_by_count", "document_log_fwci"],
            index="document_id",
            variable_name="citation_impact_metric",
            value_name="citation_impact_value",
        )
        .join(
            results_df.select(
                "document_id",
                "mean_adoption_percentile_z",
                "ecosystem_label",
            ),
            on="document_id",
            how="left",
        )
        .sort("document_id")
    )

    g = sns.lmplot(
        data=analysis_long,
        x="mean_adoption_percentile_z",
        y="citation_impact_value",
        col="citation_impact_metric",
        hue="ecosystem_label",
        row="ecosystem_label",
        scatter_kws={"alpha": 0.3},
        facet_kws={"sharey": False, "sharex": True},
    )
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.savefig(
        results_dir / "first-mover-vs-citation-impact.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _plot_quintile_analysis(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Plot 3: Quintile analysis."""
    results_pd = results_df.to_pandas()
    results_pd["adoption_quintile"] = results_pd.groupby("ecosystem_label")[
        "mean_adoption_percentile"
    ].transform(lambda x: np.ceil(x.rank(pct=True) * 5).astype(int))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, outcome, label in zip(
        axes,
        ["document_log_cited_by_count", "document_log_fwci"],
        ["log(Citations)", "log(FWCI)"],
        strict=False,
    ):
        sns.pointplot(
            data=results_pd,
            x="adoption_quintile",
            y=outcome,
            hue="ecosystem_label",
            ax=ax,
            errorbar=("ci", 95),
        )
        ax.set_xlabel("Adoption percentile quintile (1=late, 5=early)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by adoption timing quintile")

    plt.tight_layout()
    fig.savefig(
        results_dir / "q1-sign-flip-quintile-analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _compute_collinearity_diagnostics(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Compute VIFs and correlation matrices for all continuous predictors."""
    predictor_cols = [
        "mean_adoption_percentile_z",
        "max_adoption_percentile_z",
        "frac_early_adopter_z",
        "document_years_since_earliest",
        "num_libraries_scored",
        "document_author_count",
        "document_log_author_mean_citations",
        "mean_log_library_popularity",
    ]

    for eco_label in sorted(results_df.get_column("ecosystem_label").unique().to_list()):
        eco_df = results_df.filter(pl.col("ecosystem_label") == eco_label)
        eco_pd = eco_df.select(predictor_cols).to_pandas().dropna()

        # Pearson correlation matrix
        corr_matrix = eco_pd.corr()
        corr_matrix.to_csv(results_dir / f"collinearity-correlations-{eco_label}.csv")

        # VIFs
        vif_data = pd.DataFrame(
            {
                "variable": predictor_cols,
                "VIF": [
                    variance_inflation_factor(eco_pd.values, i)
                    for i in range(len(predictor_cols))
                ],
            }
        )
        vif_data.to_csv(
            results_dir / f"collinearity-vifs-{eco_label}.csv",
            index=False,
        )

        # Warnings
        high_vifs = vif_data[vif_data["VIF"] > 5]
        if not high_vifs.empty:
            print(f"WARNING [{eco_label}] VIF > 5:")
            print(high_vifs.to_string(index=False))

        high_corrs = []
        for i in range(len(predictor_cols)):
            for j in range(i + 1, len(predictor_cols)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.7:
                    high_corrs.append((predictor_cols[i], predictor_cols[j], r))
        if high_corrs:
            print(f"WARNING [{eco_label}] |r| > 0.7:")
            for v1, v2, r in high_corrs:
                print(f"  {v1} <-> {v2}: {r:.3f}")


def _run_regressions(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Run Q1 regression models per ecosystem. Returns (summary_stats_rows, coef_plot_rows)."""
    summary_stats_rows: list[dict] = []
    coef_plot_rows: list[dict] = []

    # Controlled formula components (shared across models)
    controls = (
        "+ document_years_since_earliest "
        "+ num_libraries_scored "
        "+ document_author_count "
        "+ document_log_author_mean_citations "
        "+ mean_log_library_popularity "
        "+ C(document_field_name_pruned)"
    )
    controls_no_field = (
        "+ document_years_since_earliest "
        "+ num_libraries_scored "
        "+ document_author_count "
        "+ document_log_author_mean_citations "
        "+ mean_log_library_popularity"
    )
    for ecosystem_label in sorted(results_df.get_column("ecosystem_label").unique().to_list()):
        ecosystem_df = results_df.filter(
            pl.col("ecosystem_label") == ecosystem_label
        ).to_pandas()

        print(f"\n{'=' * 60}")
        print(f"[Q1] Ecosystem: {ecosystem_label} (N={len(ecosystem_df)})")
        print(f"{'=' * 60}")

        eco_dir = results_dir / "supplement" / "q1-adoption-percentile" / ecosystem_label
        eco_dir.mkdir(exist_ok=True, parents=True)

        # ---- Mean adoption percentile models ----

        ols_raw = smf.ols(
            "document_log_cited_by_count ~ mean_adoption_percentile_z",
            data=ecosystem_df,
        ).fit()

        ols_controlled = smf.ols(
            f"document_log_cited_by_count ~ mean_adoption_percentile_z {controls}",
            data=ecosystem_df,
        ).fit()

        negbin_raw = NegativeBinomial.from_formula(
            "document_cited_by_count ~ mean_adoption_percentile_z",
            data=ecosystem_df,
        ).fit(maxiter=1000, cov_type="HC1")

        negbin_controlled = NegativeBinomial.from_formula(
            f"document_cited_by_count ~ mean_adoption_percentile_z {controls}",
            data=ecosystem_df,
        ).fit(maxiter=1000, cov_type="HC1")

        ols_fwci = smf.ols(
            f"document_log_fwci ~ mean_adoption_percentile_z {controls_no_field}",
            data=ecosystem_df,
        ).fit()

        # ---- Binary (frac_early_adopter) robustness models ----

        ols_controlled_binary = smf.ols(
            f"document_log_cited_by_count ~ frac_early_adopter_z {controls}",
            data=ecosystem_df,
        ).fit()

        negbin_controlled_binary = NegativeBinomial.from_formula(
            f"document_cited_by_count ~ frac_early_adopter_z {controls}",
            data=ecosystem_df,
        ).fit(maxiter=1000, cov_type="HC1")

        ols_fwci_binary = smf.ols(
            f"document_log_fwci ~ frac_early_adopter_z {controls_no_field}",
            data=ecosystem_df,
        ).fit()

        # ---- Max adoption percentile models ----

        ols_controlled_max = smf.ols(
            f"document_log_cited_by_count ~ max_adoption_percentile_z {controls}",
            data=ecosystem_df,
        ).fit()

        negbin_controlled_max = NegativeBinomial.from_formula(
            f"document_cited_by_count ~ max_adoption_percentile_z {controls}",
            data=ecosystem_df,
        ).fit(maxiter=1000, cov_type="HC1")

        # ---- Save model summaries ----

        models_to_save = [
            ("ols-raw.txt", ols_raw),
            ("ols-controlled.txt", ols_controlled),
            ("negbin-raw.txt", negbin_raw),
            ("negbin-controlled.txt", negbin_controlled),
            ("ols-fwci.txt", ols_fwci),
            ("robustness-binary-ols-controlled.txt", ols_controlled_binary),
            ("robustness-binary-negbin-controlled.txt", negbin_controlled_binary),
            ("robustness-binary-ols-fwci.txt", ols_fwci_binary),
            ("ols-controlled-max.txt", ols_controlled_max),
            ("negbin-controlled-max.txt", negbin_controlled_max),
        ]

        for filename, model in models_to_save:
            with open(eco_dir / filename, "w") as f:
                f.write(model.summary().as_text())

        # ---- Collect summary stats ----

        summary_stats_rows.append(
            {
                "ecosystem": ecosystem_label,
                "n": len(ecosystem_df),
                "outcome_variable": "document_cited_by_count",
                "model_type": "Negative Binomial with controls",
                "predictor": "mean_adoption_percentile_z",
                "coefficient": negbin_controlled.params["mean_adoption_percentile_z"],
                "p_value": negbin_controlled.pvalues["mean_adoption_percentile_z"],
            }
        )
        summary_stats_rows.append(
            {
                "ecosystem": ecosystem_label,
                "n": len(ecosystem_df),
                "outcome_variable": "document_log_fwci",
                "model_type": "OLS with controls",
                "predictor": "mean_adoption_percentile_z",
                "coefficient": ols_fwci.params["mean_adoption_percentile_z"],
                "p_value": ols_fwci.pvalues["mean_adoption_percentile_z"],
            }
        )
        summary_stats_rows.append(
            {
                "ecosystem": ecosystem_label,
                "n": len(ecosystem_df),
                "outcome_variable": "document_cited_by_count",
                "model_type": "NegBin controlled (max)",
                "predictor": "max_adoption_percentile_z",
                "coefficient": negbin_controlled_max.params["max_adoption_percentile_z"],
                "p_value": negbin_controlled_max.pvalues["max_adoption_percentile_z"],
            }
        )

        # ---- Collect data for coefficient plot ----

        for model_label, model_obj, pred_name in [
            ("NegBin controlled", negbin_controlled, "mean_adoption_percentile_z"),
            ("OLS FWCI", ols_fwci, "mean_adoption_percentile_z"),
            (
                "NegBin controlled (binary)",
                negbin_controlled_binary,
                "frac_early_adopter_z",
            ),
            ("OLS FWCI (binary)", ols_fwci_binary, "frac_early_adopter_z"),
            (
                "NegBin controlled (max)",
                negbin_controlled_max,
                "max_adoption_percentile_z",
            ),
            (
                "OLS controlled (max)",
                ols_controlled_max,
                "max_adoption_percentile_z",
            ),
        ]:
            coef = model_obj.params[pred_name]
            ci = model_obj.conf_int().loc[pred_name]
            coef_plot_rows.append(
                {
                    "ecosystem": ecosystem_label,
                    "model": model_label,
                    "coefficient": coef,
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                }
            )

    return summary_stats_rows, coef_plot_rows


def _run_q2_regressions(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Run Q2 regression models per ecosystem. Returns (summary_stats_rows, coef_plot_rows)."""
    summary_stats_rows: list[dict] = []
    coef_plot_rows: list[dict] = []

    controls = (
        "+ document_years_since_earliest "
        "+ num_libraries_scored "
        "+ document_author_count "
        "+ document_log_author_mean_citations "
        "+ mean_log_library_popularity "
        "+ C(document_field_name_pruned)"
    )
    controls_no_field = (
        "+ document_years_since_earliest "
        "+ num_libraries_scored "
        "+ document_author_count "
        "+ document_log_author_mean_citations "
        "+ mean_log_library_popularity"
    )

    for ecosystem_label in sorted(results_df.get_column("ecosystem_label").unique().to_list()):
        ecosystem_df = results_df.filter(
            pl.col("ecosystem_label") == ecosystem_label
        ).to_pandas()

        print(f"\n{'=' * 60}")
        print(f"[Q2] Ecosystem: {ecosystem_label} (N={len(ecosystem_df)})")
        print(f"{'=' * 60}")

        eco_dir = results_dir / "modeling-results" / ecosystem_label
        eco_dir.mkdir(exist_ok=True, parents=True)

        # ---- Primary Q2 continuous: software_novelty_z ----

        q2_ols_raw = smf.ols(
            "document_log_cited_by_count ~ software_novelty_z",
            data=ecosystem_df,
        ).fit()

        q2_ols_controlled = smf.ols(
            f"document_log_cited_by_count ~ software_novelty_z {controls}",
            data=ecosystem_df,
        ).fit()

        q2_negbin_raw = NegativeBinomial.from_formula(
            "document_cited_by_count ~ software_novelty_z",
            data=ecosystem_df,
        ).fit(maxiter=1000, cov_type="HC1")

        q2_negbin_controlled = NegativeBinomial.from_formula(
            f"document_cited_by_count ~ software_novelty_z {controls}",
            data=ecosystem_df,
        ).fit(maxiter=1000, cov_type="HC1")

        q2_ols_fwci = smf.ols(
            f"document_log_fwci ~ software_novelty_z {controls_no_field}",
            data=ecosystem_df,
        ).fit()

        # ---- Primary Q2 fraction: frac_young_libraries_z ----

        q2_frac_ols_raw = smf.ols(
            "document_log_cited_by_count ~ frac_young_libraries_z",
            data=ecosystem_df,
        ).fit()

        q2_frac_ols_controlled = smf.ols(
            f"document_log_cited_by_count ~ frac_young_libraries_z {controls}",
            data=ecosystem_df,
        ).fit()

        q2_frac_negbin_raw = NegativeBinomial.from_formula(
            "document_cited_by_count ~ frac_young_libraries_z",
            data=ecosystem_df,
        ).fit(maxiter=1000, cov_type="HC1")

        q2_frac_negbin_controlled = NegativeBinomial.from_formula(
            f"document_cited_by_count ~ frac_young_libraries_z {controls}",
            data=ecosystem_df,
        ).fit(maxiter=1000, cov_type="HC1")

        q2_frac_ols_fwci = smf.ols(
            f"document_log_fwci ~ frac_young_libraries_z {controls_no_field}",
            data=ecosystem_df,
        ).fit()

        # ---- Q2 binary: has_any_young_library (not z-scored) ----

        q2_binary_ols_controlled = smf.ols(
            f"document_log_cited_by_count ~ has_any_young_library {controls}",
            data=ecosystem_df,
        ).fit()

        q2_binary_negbin_controlled = NegativeBinomial.from_formula(
            f"document_cited_by_count ~ has_any_young_library {controls}",
            data=ecosystem_df,
        ).fit(maxiter=1000, cov_type="HC1")

        q2_binary_ols_fwci = smf.ols(
            f"document_log_fwci ~ has_any_young_library {controls_no_field}",
            data=ecosystem_df,
        ).fit()

        # Report interpretable effect sizes for binary specification
        negbin_coef = q2_binary_negbin_controlled.params["has_any_young_library"]
        fwci_coef = q2_binary_ols_fwci.params["has_any_young_library"]
        pct_change = (np.exp(negbin_coef) - 1) * 100
        print(
            f"  [{ecosystem_label}] Papers with >=1 young library: "
            f"{pct_change:+.1f}% citations (NegBin), "
            f"{fwci_coef:+.4f} log-FWCI points (OLS)"
        )

        # ---- Q2 robustness: min_library_age_z ----

        q2_negbin_controlled_min = NegativeBinomial.from_formula(
            f"document_cited_by_count ~ min_library_age_z {controls}",
            data=ecosystem_df,
        ).fit(maxiter=1000, cov_type="HC1")

        # ---- Quadratic models ----

        ecosystem_df["software_novelty_z_sq"] = ecosystem_df["software_novelty_z"] ** 2
        ecosystem_df["frac_young_libraries_z_sq"] = ecosystem_df["frac_young_libraries_z"] ** 2
        ecosystem_df["mean_adoption_percentile_z_sq"] = (
            ecosystem_df["mean_adoption_percentile_z"] ** 2
        )

        q2_ols_fwci_quad = smf.ols(
            f"document_log_fwci ~ software_novelty_z + software_novelty_z_sq {controls_no_field}",
            data=ecosystem_df,
        ).fit()

        q2_frac_ols_fwci_quad = smf.ols(
            f"document_log_fwci ~ frac_young_libraries_z + frac_young_libraries_z_sq {controls_no_field}",
            data=ecosystem_df,
        ).fit()

        q1_ols_fwci_quad = smf.ols(
            f"document_log_fwci ~ mean_adoption_percentile_z + mean_adoption_percentile_z_sq {controls_no_field}",
            data=ecosystem_df,
        ).fit()

        # Report significant quadratic terms
        for label, model, sq_term in [
            ("Q2 software_novelty", q2_ols_fwci_quad, "software_novelty_z_sq"),
            ("Q2 frac_young_libraries", q2_frac_ols_fwci_quad, "frac_young_libraries_z_sq"),
            ("Q1 mean_adoption_percentile", q1_ols_fwci_quad, "mean_adoption_percentile_z_sq"),
        ]:
            p = model.pvalues[sq_term]
            coef = model.params[sq_term]
            if p < 0.05:
                print(f"  [{label}] Significant quadratic term: coef={coef:.4f}, p={p:.4f}")

        # ---- Save model summaries ----

        # Controlled models -> per-ecosystem modeling-results
        q2_controlled_models = [
            ("q2-ols-controlled.txt", q2_ols_controlled),
            ("q2-negbin-controlled.txt", q2_negbin_controlled),
            ("q2-ols-fwci.txt", q2_ols_fwci),
            ("q2-frac-ols-controlled.txt", q2_frac_ols_controlled),
            ("q2-frac-negbin-controlled.txt", q2_frac_negbin_controlled),
            ("q2-frac-ols-fwci.txt", q2_frac_ols_fwci),
            ("q2-binary-ols-controlled.txt", q2_binary_ols_controlled),
            ("q2-binary-negbin-controlled.txt", q2_binary_negbin_controlled),
            ("q2-binary-ols-fwci.txt", q2_binary_ols_fwci),
        ]
        for filename, model in q2_controlled_models:
            with open(eco_dir / filename, "w") as f:
                f.write(model.summary().as_text())

        # Raw/bivariate and robustness models -> supplement/robustness
        robustness_eco_dir = results_dir / "supplement" / "robustness" / ecosystem_label
        robustness_eco_dir.mkdir(exist_ok=True, parents=True)
        q2_robustness_models = [
            ("q2-ols-raw.txt", q2_ols_raw),
            ("q2-negbin-raw.txt", q2_negbin_raw),
            ("q2-frac-ols-raw.txt", q2_frac_ols_raw),
            ("q2-frac-negbin-raw.txt", q2_frac_negbin_raw),
            ("q2-negbin-controlled-min.txt", q2_negbin_controlled_min),
            ("q2-ols-fwci-quadratic.txt", q2_ols_fwci_quad),
            ("q2-frac-ols-fwci-quadratic.txt", q2_frac_ols_fwci_quad),
            ("q1-ols-fwci-quadratic.txt", q1_ols_fwci_quad),
        ]
        for filename, model in q2_robustness_models:
            with open(robustness_eco_dir / filename, "w") as f:
                f.write(model.summary().as_text())

        # ---- Collect summary stats ----

        n = len(ecosystem_df)
        for model_type, outcome, predictor, model_obj in [
            (
                "Q2: NegBin controlled (software novelty)",
                "document_cited_by_count",
                "software_novelty_z",
                q2_negbin_controlled,
            ),
            (
                "Q2: OLS FWCI (software novelty)",
                "document_log_fwci",
                "software_novelty_z",
                q2_ols_fwci,
            ),
            (
                "Q2: NegBin controlled (frac young libraries)",
                "document_cited_by_count",
                "frac_young_libraries_z",
                q2_frac_negbin_controlled,
            ),
            (
                "Q2: OLS FWCI (frac young libraries)",
                "document_log_fwci",
                "frac_young_libraries_z",
                q2_frac_ols_fwci,
            ),
            (
                "Q2: NegBin controlled (min library age)",
                "document_cited_by_count",
                "min_library_age_z",
                q2_negbin_controlled_min,
            ),
            (
                "Q2: NegBin controlled (binary: any young library)",
                "document_cited_by_count",
                "has_any_young_library",
                q2_binary_negbin_controlled,
            ),
            (
                "Q2: OLS FWCI (binary: any young library)",
                "document_log_fwci",
                "has_any_young_library",
                q2_binary_ols_fwci,
            ),
        ]:
            summary_stats_rows.append(
                {
                    "ecosystem": ecosystem_label,
                    "n": n,
                    "outcome_variable": outcome,
                    "model_type": model_type,
                    "predictor": predictor,
                    "coefficient": model_obj.params[predictor],
                    "p_value": model_obj.pvalues[predictor],
                }
            )

        # ---- Collect data for coefficient plot ----

        for model_label, model_obj, pred_name in [
            ("Q2: NegBin controlled (novelty)", q2_negbin_controlled, "software_novelty_z"),
            ("Q2: OLS FWCI (novelty)", q2_ols_fwci, "software_novelty_z"),
            (
                "Q2: NegBin controlled (frac young)",
                q2_frac_negbin_controlled,
                "frac_young_libraries_z",
            ),
            ("Q2: OLS FWCI (frac young)", q2_frac_ols_fwci, "frac_young_libraries_z"),
            (
                "Q2: NegBin controlled (binary)",
                q2_binary_negbin_controlled,
                "has_any_young_library",
            ),
            ("Q2: OLS FWCI (binary)", q2_binary_ols_fwci, "has_any_young_library"),
        ]:
            coef = model_obj.params[pred_name]
            ci = model_obj.conf_int().loc[pred_name]
            coef_plot_rows.append(
                {
                    "ecosystem": ecosystem_label,
                    "model": model_label,
                    "coefficient": coef,
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                }
            )

    return summary_stats_rows, coef_plot_rows


def _plot_coefficient_forest(
    coef_plot_rows: list[dict],
    results_dir: Path,
) -> None:
    """Plot 4: Coefficient forest plot."""
    coef_plot_df = pl.DataFrame(coef_plot_rows)
    models_in_plot = coef_plot_df.get_column("model").unique().sort().to_list()
    ecosystems_in_plot = coef_plot_df.get_column("ecosystem").unique().sort().to_list()

    colors = ["#2c7bb6", "#d7191c", "#fdae61", "#abdda4", "#756bb1", "#e6550d"]
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models_in_plot)}

    _fig, ax = plt.subplots(figsize=(10, 6))
    n_models = len(models_in_plot)
    offset = 0.12

    for i, eco in enumerate(ecosystems_in_plot):
        for j, model_name in enumerate(models_in_plot):
            rows = coef_plot_df.filter(
                (pl.col("ecosystem") == eco) & (pl.col("model") == model_name)
            ).to_dicts()
            if not rows:
                continue
            row = rows[0]
            y_pos = i + (j - (n_models - 1) / 2) * offset
            ax.errorbar(
                row["coefficient"],
                y_pos,
                xerr=[
                    [row["coefficient"] - row["ci_lower"]],
                    [row["ci_upper"] - row["coefficient"]],
                ],
                fmt="o",
                color=model_colors[model_name],
                capsize=4,
                label=model_name if i == 0 else None,
            )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(ecosystems_in_plot)))
    ax.set_yticklabels(ecosystems_in_plot)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title("First-mover advantage: effect on citation impact by ecosystem")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(
        results_dir / "coefficient-comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _save_extreme_cases(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Save top 20 and bottom 20 papers by mean_adoption_percentile per ecosystem."""
    output_cols = [
        "document_id",
        "repository_id",
        "mean_adoption_percentile",
        "max_adoption_percentile",
        "frac_early_adopter",
        "num_libraries_scored",
        "document_cited_by_count",
        "document_fwci",
        "document_field_name",
        "document_publication_year",
        "ecosystem_label",
    ]

    for eco_label in sorted(results_df.get_column("ecosystem_label").unique().to_list()):
        eco_df = results_df.filter(pl.col("ecosystem_label") == eco_label)
        sorted_df = eco_df.sort("mean_adoption_percentile", descending=True)

        sorted_df.head(20).select(output_cols).write_csv(
            results_dir / f"extreme-cases-high-{eco_label}.csv"
        )
        sorted_df.tail(20).select(output_cols).write_csv(
            results_dir / f"extreme-cases-low-{eco_label}.csv"
        )


def _plot_library_age_distribution(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Plot: Distribution of mean_library_age per ecosystem."""
    g = sns.displot(
        results_df,
        kind="hist",
        stat="percent",
        common_norm=False,
        x="mean_library_age",
        col="ecosystem_label",
        hue="ecosystem_label",
        bins=50,
        facet_kws={"sharey": False},
    )
    g.set_titles(col_template="{col_name}")
    g.fig.savefig(
        results_dir / "library-age-distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _plot_frac_young_distribution(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Plot: Distribution of frac_young_libraries per ecosystem."""
    g = sns.displot(
        results_df,
        kind="hist",
        stat="percent",
        common_norm=False,
        x="frac_young_libraries",
        col="ecosystem_label",
        hue="ecosystem_label",
        bins=50,
        facet_kws={"sharey": False},
    )
    g.set_titles(col_template="{col_name}")
    g.fig.savefig(
        results_dir / "frac-young-libraries-distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _plot_q2_quintile_analysis(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Plot: Quintile analysis for software_novelty."""
    results_pd = results_df.to_pandas()
    results_pd["novelty_quintile"] = results_pd.groupby("ecosystem_label")[
        "software_novelty"
    ].transform(lambda x: np.ceil(x.rank(pct=True) * 5).astype(int))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, outcome, label in zip(
        axes,
        ["document_log_cited_by_count", "document_log_fwci"],
        ["log(Citations)", "log(FWCI)"],
        strict=False,
    ):
        sns.pointplot(
            data=results_pd,
            x="novelty_quintile",
            y=outcome,
            hue="ecosystem_label",
            ax=ax,
            errorbar=("ci", 95),
        )
        ax.set_xlabel("Software novelty quintile (1=old toolkit, 5=new toolkit)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by software novelty quintile")

    plt.tight_layout()
    fig.savefig(
        results_dir / "q2-quintile-analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _plot_q2_frac_quintile_analysis(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Plot: Quintile analysis for frac_young_libraries."""
    results_pd = results_df.to_pandas()
    results_pd["frac_young_quintile"] = results_pd.groupby("ecosystem_label")[
        "frac_young_libraries"
    ].transform(lambda x: np.ceil(x.rank(pct=True) * 5).astype(int))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, outcome, label in zip(
        axes,
        ["document_log_cited_by_count", "document_log_fwci"],
        ["log(Citations)", "log(FWCI)"],
        strict=False,
    ):
        sns.pointplot(
            data=results_pd,
            x="frac_young_quintile",
            y=outcome,
            hue="ecosystem_label",
            ax=ax,
            errorbar=("ci", 95),
        )
        ax.set_xlabel("Fraction young libraries quintile (1=low, 5=high)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by fraction young libraries quintile")

    plt.tight_layout()
    fig.savefig(
        results_dir / "q2-frac-quintile-analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _plot_q2_binary_quintile_analysis(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Plot: Bar chart comparing outcomes between has_young_lib=0 vs 1."""
    results_pd = results_df.to_pandas()
    results_pd["has_young_library_label"] = cast(
        pd.Series, results_pd["has_any_young_library"]
    ).map({0: "No young library", 1: "Has young library"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, outcome, label in zip(
        axes,
        ["document_log_cited_by_count", "document_log_fwci"],
        ["log(Citations)", "log(FWCI)"],
        strict=False,
    ):
        sns.barplot(
            data=results_pd,
            x="has_young_library_label",
            y=outcome,
            hue="ecosystem_label",
            ax=ax,
            errorbar=("ci", 95),
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by young library usage")

    plt.tight_layout()
    fig.savefig(
        results_dir / "q2-binary-quintile-analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _plot_q2_coefficient_forest(
    coef_plot_rows: list[dict],
    results_dir: Path,
) -> None:
    """Plot: Q2-only coefficient forest plot."""
    coef_plot_df = pl.DataFrame(coef_plot_rows)
    models_in_plot = coef_plot_df.get_column("model").unique().sort().to_list()
    ecosystems_in_plot = coef_plot_df.get_column("ecosystem").unique().sort().to_list()

    colors = [
        "#2c7bb6",
        "#d7191c",
        "#fdae61",
        "#abdda4",
        "#756bb1",
        "#e6550d",
        "#1b9e77",
        "#d95f02",
    ]
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models_in_plot)}

    _fig, ax = plt.subplots(figsize=(10, 6))
    n_models = len(models_in_plot)
    offset = 0.10

    for i, eco in enumerate(ecosystems_in_plot):
        for j, model_name in enumerate(models_in_plot):
            rows = coef_plot_df.filter(
                (pl.col("ecosystem") == eco) & (pl.col("model") == model_name)
            ).to_dicts()
            if not rows:
                continue
            row = rows[0]
            y_pos = i + (j - (n_models - 1) / 2) * offset
            ax.errorbar(
                row["coefficient"],
                y_pos,
                xerr=[
                    [row["coefficient"] - row["ci_lower"]],
                    [row["ci_upper"] - row["coefficient"]],
                ],
                fmt="o",
                color=model_colors[model_name],
                capsize=4,
                label=model_name if i == 0 else None,
            )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(ecosystems_in_plot)))
    ax.set_yticklabels(ecosystems_in_plot)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title("Q2: Software novelty effect on citation impact by ecosystem")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(
        results_dir / "q2-coefficient-forest.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _run_sensitivity_analysis(
    results_df: pl.DataFrame,
    repository_imports: pl.DataFrame,
    pair_metadata: pl.DataFrame,
    library_metadata: pl.DataFrame,
    min_adopters_per_library: int,
    results_dir: Path,
) -> pl.DataFrame:
    """
    Sweep over young-library thresholds [1, 2, 3] and fit OLS FWCI models.

    Returns a DataFrame with sensitivity results.
    """
    controls_no_field = (
        "+ document_years_since_earliest "
        "+ num_libraries_scored "
        "+ document_author_count "
        "+ document_log_author_mean_citations "
        "+ mean_log_library_popularity"
    )

    sensitivity_rows: list[dict] = []

    for threshold in [1, 2, 3]:
        print(f"\n[Sensitivity] Threshold: {threshold} years")

        # Compute age stats for this threshold
        age_stats = _compute_library_age_stats(
            repository_imports=repository_imports,
            pair_metadata=pair_metadata,
            library_metadata=library_metadata,
            min_adopters_per_library=min_adopters_per_library,
            young_library_threshold=threshold,
        )
        novelty_scores = _compute_paper_novelty_scores(age_stats)

        # Join with results_df (use only the columns we need)
        sensitivity_df = results_df.select(
            "document_id",
            "ecosystem_label",
            "document_log_fwci",
            "document_cited_by_count",
            "document_log_cited_by_count",
            "document_years_since_earliest",
            "num_libraries_scored",
            "document_author_count",
            "document_log_author_mean_citations",
            "mean_log_library_popularity",
        ).join(
            novelty_scores.select(
                "document_id", "frac_young_libraries", "has_any_young_library"
            ),
            on="document_id",
            how="inner",
        )

        for eco_label in sorted(
            sensitivity_df.get_column("ecosystem_label").unique().to_list()
        ):
            eco_df = sensitivity_df.filter(pl.col("ecosystem_label") == eco_label)

            # Z-score frac_young_libraries within ecosystem
            eco_df = eco_df.with_columns(
                (
                    (pl.col("frac_young_libraries") - pl.col("frac_young_libraries").mean())
                    / pl.col("frac_young_libraries").std()
                ).alias("frac_young_libraries_z"),
            )
            eco_pd = eco_df.to_pandas()

            # OLS FWCI with frac_young_libraries_z
            frac_model = smf.ols(
                f"document_log_fwci ~ frac_young_libraries_z {controls_no_field}",
                data=eco_pd,
            ).fit()

            frac_ci = frac_model.conf_int().loc["frac_young_libraries_z"]
            sensitivity_rows.append(
                {
                    "ecosystem": eco_label,
                    "threshold_years": threshold,
                    "measure": "frac_young_libraries_z",
                    "outcome": "FWCI",
                    "coefficient": frac_model.params["frac_young_libraries_z"],
                    "ci_lower": frac_ci[0],
                    "ci_upper": frac_ci[1],
                    "p_value": frac_model.pvalues["frac_young_libraries_z"],
                    "n": len(eco_pd),
                }
            )

            # OLS FWCI with has_any_young_library (binary)
            binary_model = smf.ols(
                f"document_log_fwci ~ has_any_young_library {controls_no_field}",
                data=eco_pd,
            ).fit()

            binary_ci = binary_model.conf_int().loc["has_any_young_library"]
            sensitivity_rows.append(
                {
                    "ecosystem": eco_label,
                    "threshold_years": threshold,
                    "measure": "has_any_young_library",
                    "outcome": "FWCI",
                    "coefficient": binary_model.params["has_any_young_library"],
                    "ci_lower": binary_ci[0],
                    "ci_upper": binary_ci[1],
                    "p_value": binary_model.pvalues["has_any_young_library"],
                    "n": len(eco_pd),
                }
            )

    sensitivity_df = pl.DataFrame(sensitivity_rows)
    sensitivity_df.write_csv(results_dir / "sensitivity-young-threshold.csv")
    print(f"\nSaved sensitivity results to {results_dir / 'sensitivity-young-threshold.csv'}")
    return sensitivity_df


def _plot_sensitivity_threshold(
    sensitivity_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Plot: Sensitivity of FWCI coefficient across young-library thresholds."""
    sens_pd = sensitivity_df.to_pandas()

    ecosystems = sorted(sens_pd["ecosystem"].unique())
    n_eco = len(ecosystems)

    fig, axes = plt.subplots(1, n_eco, figsize=(5 * n_eco, 4), sharey=True)
    if n_eco == 1:
        axes = [axes]

    measure_styles = {
        "frac_young_libraries_z": ("Fraction (z-scored)", "o", "#2c7bb6"),
        "has_any_young_library": ("Binary (any young)", "s", "#d7191c"),
    }

    for ax, eco in zip(axes, ecosystems, strict=False):
        eco_data = sens_pd[sens_pd["ecosystem"] == eco]
        for measure, (label, marker, color) in measure_styles.items():
            m_data = cast(pd.DataFrame, eco_data[eco_data["measure"] == measure]).sort_values(
                "threshold_years"
            )
            ax.errorbar(
                m_data["threshold_years"],
                m_data["coefficient"],
                yerr=[
                    m_data["coefficient"] - m_data["ci_lower"],
                    m_data["ci_upper"] - m_data["coefficient"],
                ],
                fmt=f"{marker}-",
                color=color,
                capsize=4,
                label=label,
            )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Young library threshold (years)")
        ax.set_title(eco)
        ax.set_xticks([1, 2, 3])
        if ax == axes[0]:
            ax.set_ylabel("OLS FWCI coefficient (95% CI)")
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(
        results_dir / "sensitivity-young-threshold.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def _print_results_summary(
    summary_stats_df: pl.DataFrame,
    sensitivity_df: pl.DataFrame,
    temporal_confound_csv: Path,
) -> None:
    """Print a concise results summary to stdout."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Temporal confound correlations
    confound_df = pl.read_csv(temporal_confound_csv)
    print("\nTemporal confound (r with publication year):")
    for var in [
        "mean_adoption_percentile",
        "software_novelty",
        "frac_young_libraries",
    ]:
        var_rows = confound_df.filter(pl.col("variable") == var)
        if var_rows.is_empty():
            continue
        parts = []
        for eco in sorted(var_rows.get_column("ecosystem").to_list()):
            r = var_rows.filter(pl.col("ecosystem") == eco).get_column(
                "pearson_r_with_pub_year"
            )[0]
            parts.append(f"{eco}={r:.2f}")
        label = f"Q1 {var}" if "adoption" in var else f"Q2 {var}"
        print(f"  {label:40s} {', '.join(parts)}")

    # Primary Q2 results
    print("\nPrimary Q2 results (OLS FWCI, frac_young_libraries_z):")
    frac_fwci = summary_stats_df.filter(
        pl.col("model_type") == "Q2: OLS FWCI (frac young libraries)"
    )
    for eco in sorted(frac_fwci.get_column("ecosystem").to_list()):
        row = frac_fwci.filter(pl.col("ecosystem") == eco).to_dicts()[0]
        p_str = f"p={row['p_value']:.4f}" if row["p_value"] >= 0.001 else "p<0.001"
        print(f"  {eco:20s} coef={row['coefficient']:+.4f}, {p_str}, N={row['n']}")

    # Binary specification
    print("\nBinary specification (OLS FWCI, has_any_young_library):")
    binary_fwci = summary_stats_df.filter(
        pl.col("model_type") == "Q2: OLS FWCI (binary: any young library)"
    )
    for eco in sorted(binary_fwci.get_column("ecosystem").to_list()):
        row = binary_fwci.filter(pl.col("ecosystem") == eco).to_dicts()[0]
        p_str = f"p={row['p_value']:.4f}" if row["p_value"] >= 0.001 else "p<0.001"
        print(f"  {eco:20s} coef={row['coefficient']:+.4f}, {p_str}, N={row['n']}")

    # Threshold sensitivity
    print("\nThreshold sensitivity (OLS FWCI, frac_young_z):")
    frac_sens = sensitivity_df.filter(pl.col("measure") == "frac_young_libraries_z")
    for threshold in [1, 2, 3]:
        t_rows = frac_sens.filter(pl.col("threshold_years") == threshold)
        parts = []
        for eco in sorted(t_rows.get_column("ecosystem").to_list()):
            coef = t_rows.filter(pl.col("ecosystem") == eco).get_column("coefficient")[0]
            parts.append(f"{eco}={coef:+.4f}")
        print(f"  {threshold}yr: {', '.join(parts)}")


def _save_q2_extreme_cases(
    results_df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Save top 20 and bottom 20 papers by mean_library_age per ecosystem."""
    output_cols = [
        "document_id",
        "repository_id",
        "mean_library_age",
        "min_library_age",
        "frac_young_libraries",
        "software_novelty",
        "num_libraries_age_scored",
        "document_cited_by_count",
        "document_fwci",
        "document_field_name",
        "document_publication_year",
        "ecosystem_label",
    ]

    for eco_label in sorted(results_df.get_column("ecosystem_label").unique().to_list()):
        eco_df = results_df.filter(pl.col("ecosystem_label") == eco_label)
        sorted_df = eco_df.sort("mean_library_age", descending=False)

        # Newest toolkit (lowest mean_library_age)
        sorted_df.head(20).select(output_cols).write_csv(
            results_dir / f"q2-extreme-cases-newest-toolkit-{eco_label}.csv"
        )
        # Oldest toolkit (highest mean_library_age)
        sorted_df.tail(20).select(output_cols).write_csv(
            results_dir / f"q2-extreme-cases-oldest-toolkit-{eco_label}.csv"
        )


@app.command()
def main(
    remove_extremely_rare_imports: bool = True,
    rare_import_threshold: int = 3,
    min_adopters_per_library: int = 10,
    early_adopter_percentile_threshold: float = 0.90,
    top_n_fields: int = 5,
    sample: bool = False,
    sample_size: int = 5000,
) -> None:
    load_dotenv()
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Create results directory structure
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    (RESULTS_DIR / "main").mkdir(exist_ok=True)
    (RESULTS_DIR / "supplement" / "q1-adoption-percentile").mkdir(exist_ok=True, parents=True)
    (RESULTS_DIR / "supplement" / "collinearity").mkdir(exist_ok=True, parents=True)
    (RESULTS_DIR / "supplement" / "robustness").mkdir(exist_ok=True, parents=True)

    # Load and filter pairs
    pair_metadata = _load_and_filter_pairs(top_n_fields=top_n_fields)

    # Take a sample to speed up development
    if sample:
        pair_metadata = pair_metadata.sample(sample_size, seed=42)

    # Get the repository IDs in our dataset
    repository_ids = pair_metadata.get_column("repository_id").unique().to_list()

    # Load the repository imports
    repository_imports = _get_imported_software(
        repository_ids,
        remove_extremely_rare_imports=remove_extremely_rare_imports,
        rare_import_threshold=rare_import_threshold,
    )

    # Compute per-library adoption stats
    library_adoption_stats = _compute_library_adoption_stats(
        repository_imports=repository_imports,
        pair_metadata=pair_metadata,
        min_adopters_per_library=min_adopters_per_library,
        early_adopter_percentile_threshold=early_adopter_percentile_threshold,
    )

    # Assign ecosystem labels (Task 8)
    ecosystem_label_df = _assign_ecosystem_labels(repository_imports, pair_metadata)

    # Compute paper-level first-mover scores
    paper_scores = _compute_paper_first_mover_scores(library_adoption_stats)

    # Merge scores with metadata and ecosystem labels
    results_df = paper_scores.join(pair_metadata, on="document_id", how="inner").join(
        ecosystem_label_df, on="document_id", how="inner"
    )

    # Remove outliers and z-score
    results_df = _remove_outliers_per_ecosystem(results_df)
    results_df = _zscore_per_ecosystem(results_df)

    # Log final N per ecosystem
    print("Final counts per ecosystem:")
    print(results_df.get_column("ecosystem_label").value_counts(sort=True))

    # Save intermediate results
    results_df.write_parquet(RESULTS_DIR / "first-mover-scores.parquet")

    # Descriptive statistics
    print("\nDistribution of mean adoption percentile per ecosystem:")
    print(
        results_df.group_by("ecosystem_label").agg(
            pl.col("mean_adoption_percentile").min().alias("min"),
            pl.col("mean_adoption_percentile").quantile(0.25).alias("25%"),
            pl.col("mean_adoption_percentile").quantile(0.5).alias("50%"),
            pl.col("mean_adoption_percentile").quantile(0.75).alias("75%"),
            pl.col("mean_adoption_percentile").max().alias("max"),
            pl.col("mean_adoption_percentile").mean().alias("mean"),
            pl.col("mean_adoption_percentile").std().alias("std"),
        )
    )

    q1_supplement_dir = RESULTS_DIR / "supplement" / "q1-adoption-percentile"
    main_dir = RESULTS_DIR / "main"

    # Plot 1: Distribution of mean adoption percentile -> supplement
    print("Plotting adoption percentile distribution...")
    _plot_adoption_distribution(results_df, q1_supplement_dir)

    # Prepare data for modeling
    results_df = results_df.with_columns(
        (pl.col("document_cited_by_count").cast(pl.Float64).log()).alias(
            "document_log_cited_by_count"
        ),
        (pl.col("document_fwci").log()).alias("document_log_fwci"),
    ).filter(
        pl.col("document_log_fwci").is_not_nan(),
    )

    # Plot 2: Scatter + trend -> supplement
    print("Plotting first-mover vs citation impact...")
    _plot_first_mover_vs_impact(results_df, q1_supplement_dir)

    # Plot 3: Q1 quintile analysis -> main (shows sign flip motivating Q2)
    print("Plotting Q1 quintile analysis...")
    _plot_quintile_analysis(results_df, main_dir)

    # Collinearity diagnostics -> supplement/collinearity
    print("Computing collinearity diagnostics...")
    _compute_collinearity_diagnostics(results_df, RESULTS_DIR / "supplement" / "collinearity")

    # Extreme cases -> supplement
    print("Saving extreme cases...")
    _save_extreme_cases(results_df, q1_supplement_dir)

    # Q1 regression models
    print("Running Q1 regression models...")
    summary_stats_rows, coef_plot_rows = _run_regressions(
        results_df,
        RESULTS_DIR,
    )

    # Q1 coefficient forest plot -> supplement
    print("Plotting Q1 coefficient forest...")
    _plot_coefficient_forest(coef_plot_rows, q1_supplement_dir)

    # =========================================================================
    # Q2: Library Age / Software Novelty Analysis
    # =========================================================================

    print("\n" + "=" * 60)
    print("Starting Q2: Library Age / Software Novelty Analysis")
    print("=" * 60)

    # Compute library metadata (birth year and total adopters for all libraries)
    library_metadata = _compute_library_metadata(repository_imports, pair_metadata)
    print(f"Computed metadata for {len(library_metadata)} libraries")

    # Compute library age stats per (document, library) pair
    library_age_stats = _compute_library_age_stats(
        repository_imports=repository_imports,
        pair_metadata=pair_metadata,
        library_metadata=library_metadata,
        min_adopters_per_library=min_adopters_per_library,
        young_library_threshold=2,
    )

    # Compute paper-level novelty scores
    paper_novelty_scores = _compute_paper_novelty_scores(library_age_stats)

    # Join novelty scores into results_df
    results_df = results_df.join(paper_novelty_scores, on="document_id", how="inner")
    print(f"Papers with Q2 scores: {len(results_df)}")

    # Re-run z-scoring now that Q2 columns are present
    # (drop old z-score columns first to avoid conflicts)
    z_cols_to_drop = [c for c in results_df.columns if c.endswith("_z")]
    results_df = results_df.drop(z_cols_to_drop)
    results_df = _zscore_per_ecosystem(results_df)

    # Temporal confound correlations -> main
    print("Computing temporal confound correlations...")
    _compute_temporal_confound_correlations(results_df, main_dir)

    # Q2 descriptive statistics
    print("\nDistribution of mean_library_age per ecosystem:")
    print(
        results_df.group_by("ecosystem_label").agg(
            pl.col("mean_library_age").min().alias("min"),
            pl.col("mean_library_age").quantile(0.25).alias("25%"),
            pl.col("mean_library_age").quantile(0.5).alias("50%"),
            pl.col("mean_library_age").quantile(0.75).alias("75%"),
            pl.col("mean_library_age").max().alias("max"),
            pl.col("mean_library_age").mean().alias("mean"),
            pl.col("mean_library_age").std().alias("std"),
        )
    )

    robustness_dir = RESULTS_DIR / "supplement" / "robustness"

    # Q2 plots
    print("Plotting Q2 distributions...")
    _plot_library_age_distribution(results_df, robustness_dir)
    _plot_frac_young_distribution(results_df, main_dir)

    print("Plotting Q2 quintile analyses...")
    _plot_q2_quintile_analysis(results_df, robustness_dir)
    _plot_q2_frac_quintile_analysis(results_df, main_dir)
    _plot_q2_binary_quintile_analysis(results_df, main_dir)

    # Q2 regression models
    print("Running Q2 regression models...")
    q2_summary_stats_rows, q2_coef_plot_rows = _run_q2_regressions(
        results_df,
        RESULTS_DIR,
    )

    # Q2 coefficient forest plot -> main
    print("Plotting Q2 coefficient forest...")
    _plot_q2_coefficient_forest(q2_coef_plot_rows, main_dir)

    # Sensitivity analysis across thresholds
    print("\nRunning sensitivity analysis across young-library thresholds...")
    sensitivity_df = _run_sensitivity_analysis(
        results_df=results_df,
        repository_imports=repository_imports,
        pair_metadata=pair_metadata,
        library_metadata=library_metadata,
        min_adopters_per_library=min_adopters_per_library,
        results_dir=main_dir,
    )
    _plot_sensitivity_threshold(sensitivity_df, main_dir)

    # Save combined summary stats (Q1 + Q2) with question and is_primary columns
    for row in summary_stats_rows:
        row["question"] = "Q1"
        row["is_primary"] = False
    for row in q2_summary_stats_rows:
        row["question"] = "Q2"
        row["is_primary"] = row["model_type"] in {
            "Q2: NegBin controlled (software novelty)",
            "Q2: OLS FWCI (software novelty)",
            "Q2: NegBin controlled (frac young libraries)",
            "Q2: OLS FWCI (frac young libraries)",
            "Q2: NegBin controlled (binary: any young library)",
            "Q2: OLS FWCI (binary: any young library)",
        }
    all_summary_stats = summary_stats_rows + q2_summary_stats_rows
    summary_stats_df = pl.DataFrame(all_summary_stats).sort(
        "question", "ecosystem", "outcome_variable", "model_type"
    )
    summary_stats_df.write_csv(main_dir / "modeling-summary-stats.csv")

    # Q2 extreme cases -> supplement/robustness
    print("Saving Q2 extreme cases...")
    _save_q2_extreme_cases(results_df, robustness_dir)

    # Save updated results with Q2 columns
    results_df.write_parquet(RESULTS_DIR / "first-mover-scores.parquet")

    # =========================================================================
    # Results summary
    # =========================================================================
    _print_results_summary(
        summary_stats_df=summary_stats_df,
        sensitivity_df=sensitivity_df,
        temporal_confound_csv=main_dir / "temporal-confound-correlations.csv",
    )

    print("\nDone. Results saved to:", RESULTS_DIR)


###############################################################################

if __name__ == "__main__":
    app()
