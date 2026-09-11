#!/usr/bin/env python

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

###############################################################################

app = typer.Typer()

HF_DATASET = "sci-soft-collections/rs-graph-v2-full"

THIS_FILE_PATH = Path(__file__).resolve()
THIS_DIR = THIS_FILE_PATH.parent
load_dotenv(str(THIS_DIR.parents[1] / ".env"))

ANNOTATION_OUTPUT_PATH = THIS_DIR / "mentions-imports-deps-annotation.xlsx"

RANDOM_SEED = 42
SAMPLE_SIZE = 100
RARE_THRESHOLD = 3

# Generic terms excluded from mentions
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
    "software",
}

###############################################################################


def load_table(table: str) -> pl.DataFrame:
    """Load a single rs-graph table from HuggingFace as a polars DataFrame."""
    ds = load_dataset(HF_DATASET, table, split="train", token=os.environ.get("HF_TOKEN"))
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


def _filter_min_count(df: pl.DataFrame, col: str, threshold: int) -> pl.DataFrame:
    """Keep rows whose `col` value appears at least `threshold` times."""
    frequent_values = (
        df.group_by(col)
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= threshold)
        .get_column(col)
        .to_list()
    )
    return df.filter(pl.col(col).is_in(frequent_values))


def _set_diff_expr(col_a: str, col_b: str) -> pl.Expr:
    """Sorted semicolon-joined names present in the `col_a` list but not the `col_b` list."""
    return pl.struct([col_a, col_b]).map_elements(
        lambda x: ";".join(sorted(set(x[col_a] or []) - set(x[col_b] or []))),
        return_dtype=pl.String,
    )


def _load_our_dataset(
    top_n_fields: int = 5,
) -> pl.DataFrame:
    """Load and filter the article-repository pair table with document and repository metadata."""
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


def _remove_extremely_rare_software_usage(
    usage_df: pl.DataFrame,
    software_col: str = "software_name_normalized",
    rare_usage_threshold: int = 3,
    compute_ecosystem_prefix: bool = False,
    use_ecosystem_from_data: bool = False,
    exclude_generic_mentions: bool = False,
) -> pl.DataFrame:
    """Drop rows whose software name appears fewer than `rare_usage_threshold` times,
    optionally prefixing names with their ecosystem ("py:", "r:") first.
    """
    if compute_ecosystem_prefix:
        # Derive the ecosystem from the file extensions in the semicolon-separated
        # "file_paths" column, then prefix the software name with it

        # Pre-construct the py and r checks
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

        # Now compute ecosystem and ecosystem normalized software name
        usage_df = (
            usage_df.with_columns(
                pl.col("file_paths").str.to_lowercase().alias("file_paths_lower")
            )
            .with_columns(
                # Check contains at least one .py or .ipynb
                # AND at least one .r or .rmd to determine if it's mixed
                pl.when(check_for_py & check_for_r)
                .then(pl.lit("mixed"))
                # Check py and ipynb
                .when(check_for_py)
                .then(pl.lit("py"))
                # Check r and rmd
                .when(check_for_r)
                .then(pl.lit("r"))
                # Other
                .otherwise(pl.lit("other"))
                .alias("ecosystem")
            )
            .with_columns(
                (pl.col("ecosystem") + pl.lit(":") + pl.col(software_col)).alias(
                    f"ecosystem_{software_col}"
                )
            )
        )

        # Use the ecosystem-prefixed column for the rest of the function
        software_col = f"ecosystem_{software_col}"

        # Drop "mixed" and "other" ecosystems
        usage_df = usage_df.filter(pl.col("ecosystem").is_in(["py", "r"]))

    elif use_ecosystem_from_data:
        # Prefix the software name with the existing "ecosystem" column
        usage_df = usage_df.with_columns(
            (pl.col("ecosystem") + pl.lit(":") + pl.col(software_col)).alias(
                f"ecosystem_{software_col}"
            )
        )
        software_col = f"ecosystem_{software_col}"

    elif exclude_generic_mentions:
        # Drop generic terms like "code" or "software"
        usage_df = usage_df.filter(~pl.col(software_col).is_in(MENTION_EXCLUDE_NORMALIZED))

    # Keep only software used at least rare_usage_threshold times
    return _filter_min_count(usage_df, software_col, rare_usage_threshold)


@dataclass
class MetadataAndSoftwareUsageDataFrames:
    pair_metadata: pl.DataFrame
    repository_imports: pl.DataFrame
    repository_dependencies: pl.DataFrame
    document_software_mentions: pl.DataFrame


def _add_has_imports_dependencies_mentions_cols(
    pair_metadata: pl.DataFrame,
    repository_imports: pl.DataFrame,
    repository_dependencies: pl.DataFrame,
    document_software_mentions: pl.DataFrame,
) -> MetadataAndSoftwareUsageDataFrames:
    """Attach has_imports / has_dependencies / has_software_mentions flags to the pair table."""
    # Get the subset of each that have a repository_id or document_id in the merged set
    repository_imports = repository_imports.join(
        pair_metadata.select(
            pl.col("document_repository_link_id"),
            pl.col("document_id"),
            pl.col("repository_id"),
        ),
        on="repository_id",
        how="inner",
    )
    repository_dependencies = repository_dependencies.join(
        pair_metadata.select(
            pl.col("document_repository_link_id"),
            pl.col("document_id"),
            pl.col("repository_id"),
        ),
        on="repository_id",
        how="inner",
    )
    document_software_mentions = document_software_mentions.join(
        pair_metadata.select(
            pl.col("document_repository_link_id"),
            pl.col("document_id"),
            pl.col("repository_id"),
        ),
        on="document_id",
        how="inner",
    )

    # Flag pairs with at least one import / dependency / software mention
    has_imports_df = repository_imports.group_by("document_repository_link_id").agg(
        has_imports=pl.lit(True),
    )
    has_dependencies_df = repository_dependencies.group_by("document_repository_link_id").agg(
        has_dependencies=pl.lit(True),
    )
    has_software_mentions_df = document_software_mentions.group_by(
        "document_repository_link_id"
    ).agg(
        has_software_mentions=pl.lit(True),
    )

    # Add each of these columns to the merged table
    pair_metadata = (
        pair_metadata.join(
            has_imports_df,
            on="document_repository_link_id",
            how="left",
        )
        .join(
            has_dependencies_df,
            on="document_repository_link_id",
            how="left",
        )
        .join(
            has_software_mentions_df,
            on="document_repository_link_id",
            how="left",
        )
        .fill_null(False)
    )

    return MetadataAndSoftwareUsageDataFrames(
        pair_metadata=pair_metadata,
        repository_imports=repository_imports,
        repository_dependencies=repository_dependencies,
        document_software_mentions=document_software_mentions,
    )


def _link_ids_above_99th_percentile(usage_df: pl.DataFrame) -> set[int]:
    """Link ids whose per-pair usage-row count exceeds the 99th percentile count."""
    counts = usage_df.group_by("document_repository_link_id").agg(usage_count=pl.len())
    threshold = counts.get_column("usage_count").quantile(0.99)
    return set(
        counts.filter(pl.col("usage_count") > threshold)
        .get_column("document_repository_link_id")
        .to_list()
    )


def _remove_pairs_with_extreme_software_usage(
    pair_metadata: pl.DataFrame,
    repository_imports: pl.DataFrame,
    repository_dependencies: pl.DataFrame,
    document_software_mentions: pl.DataFrame,
) -> MetadataAndSoftwareUsageDataFrames:
    # Get link ids with imports, dependencies, or mentions above the 99th percentile count
    extreme_usage_ids = (
        _link_ids_above_99th_percentile(repository_imports)
        | _link_ids_above_99th_percentile(repository_dependencies)
        | _link_ids_above_99th_percentile(document_software_mentions)
    )

    # Remove extreme usage pairs from pair_metadata
    pair_metadata = pair_metadata.filter(
        ~pl.col("document_repository_link_id").is_in(extreme_usage_ids)
    )

    # Filter each usage table to only the remaining pairs
    remaining_link_ids = pair_metadata.get_column("document_repository_link_id").to_list()
    repository_imports = repository_imports.filter(
        pl.col("document_repository_link_id").is_in(remaining_link_ids)
    )
    repository_dependencies = repository_dependencies.filter(
        pl.col("document_repository_link_id").is_in(remaining_link_ids)
    )
    document_software_mentions = document_software_mentions.filter(
        pl.col("document_repository_link_id").is_in(remaining_link_ids)
    )

    return MetadataAndSoftwareUsageDataFrames(
        pair_metadata=pair_metadata,
        repository_imports=repository_imports,
        repository_dependencies=repository_dependencies,
        document_software_mentions=document_software_mentions,
    )


###############################################################################


@app.command()
def main() -> None:
    print("Loading tables and building pair metadata...")
    pair_metadata = _load_our_dataset()

    # Get repository imports, repository dependencies, and document software mentions
    repository_imports = load_table("repository_import")
    repository_dependencies = load_table("repository_dependency")
    document_software_mentions = load_table("document_software_mention")

    # Remove extremely rare software
    repository_imports = _remove_extremely_rare_software_usage(
        repository_imports,
        software_col="software_name_normalized",
        rare_usage_threshold=RARE_THRESHOLD,
        compute_ecosystem_prefix=True,
    )
    repository_dependencies = _remove_extremely_rare_software_usage(
        repository_dependencies,
        software_col="software_name_normalized",
        rare_usage_threshold=RARE_THRESHOLD,
        use_ecosystem_from_data=True,
    )
    document_software_mentions = _remove_extremely_rare_software_usage(
        document_software_mentions,
        software_col="software_name_normalized",
        rare_usage_threshold=RARE_THRESHOLD,
        exclude_generic_mentions=True,
    )

    # Add has_imports / has_dependencies / has_software_mentions flags
    software_usage_cols_result = _add_has_imports_dependencies_mentions_cols(
        pair_metadata,
        repository_imports,
        repository_dependencies,
        document_software_mentions,
    )
    pair_metadata = software_usage_cols_result.pair_metadata
    repository_imports = software_usage_cols_result.repository_imports
    repository_dependencies = software_usage_cols_result.repository_dependencies
    document_software_mentions = software_usage_cols_result.document_software_mentions

    # Remove pairs with extreme software usage counts
    non_extreme_pairs_result = _remove_pairs_with_extreme_software_usage(
        pair_metadata,
        repository_imports,
        repository_dependencies,
        document_software_mentions,
    )
    pair_metadata = non_extreme_pairs_result.pair_metadata
    repository_imports = non_extreme_pairs_result.repository_imports
    repository_dependencies = non_extreme_pairs_result.repository_dependencies
    document_software_mentions = non_extreme_pairs_result.document_software_mentions

    # Re-apply rare threshold on the analysis subset
    repository_imports = _filter_min_count(
        repository_imports, "ecosystem_software_name_normalized", RARE_THRESHOLD
    )
    document_software_mentions = _filter_min_count(
        document_software_mentions, "software_name_normalized", RARE_THRESHOLD
    )

    complete_cases = pair_metadata.filter(
        pl.col("has_imports") & pl.col("has_dependencies") & pl.col("has_software_mentions")
    )
    print(f"Total pairs: {len(pair_metadata)}")
    print(f"Complete cases: {len(complete_cases)}")

    # Aggregate software names for complete-case pairs only
    # Collect unique names into lists, then derive display strings and set diffs
    complete_link_ids = complete_cases.get_column("document_repository_link_id").to_list()

    mentions_agg = (
        document_software_mentions.filter(
            pl.col("document_repository_link_id").is_in(complete_link_ids)
        )
        .group_by("document_id")
        .agg(
            pl.col("software_name").unique().alias("_mentioned_raw"),
            pl.col("software_name_normalized").unique().alias("_mentioned_norm"),
            pl.col("software_name_normalized").n_unique().alias("mentioned_count"),
        )
    )
    imports_agg = (
        repository_imports.filter(
            pl.col("document_repository_link_id").is_in(complete_link_ids)
        )
        .group_by("repository_id")
        .agg(
            pl.col("software_name").unique().alias("_imported_raw"),
            pl.col("software_name_normalized").unique().alias("_imported_norm"),
            pl.col("software_name_normalized").n_unique().alias("imported_count"),
        )
    )
    dependencies_agg = (
        repository_dependencies.filter(
            pl.col("document_repository_link_id").is_in(complete_link_ids)
        )
        .group_by("repository_id")
        .agg(
            pl.col("software_name").unique().alias("_dependencies_raw"),
            pl.col("software_name_normalized").unique().alias("_dependencies_norm"),
            pl.col("software_name_normalized").n_unique().alias("dependencies_count"),
        )
    )

    complete_cases = (
        complete_cases.join(mentions_agg, on="document_id", how="left")
        .join(imports_agg, on="repository_id", how="left")
        .join(dependencies_agg, on="repository_id", how="left")
        .with_columns(
            ("https://doi.org/" + pl.col("document_doi")).alias("document_doi_url"),
            (
                "https://github.com/"
                + pl.col("repository_owner")
                + "/"
                + pl.col("repository_name")
            ).alias("repository_url"),
            # Sorted semicolon-joined display strings
            pl.col("_mentioned_raw").list.sort().list.join(";").alias("mentioned_software_raw"),
            pl.col("_mentioned_norm")
            .list.sort()
            .list.join(";")
            .alias("mentioned_software_normalized"),
            pl.col("_imported_raw").list.sort().list.join(";").alias("imported_software_raw"),
            pl.col("_imported_norm")
            .list.sort()
            .list.join(";")
            .alias("imported_software_normalized"),
            pl.col("_dependencies_raw")
            .list.sort()
            .list.join(";")
            .alias("dependencies_software_raw"),
            pl.col("_dependencies_norm")
            .list.sort()
            .list.join(";")
            .alias("dependencies_software_normalized"),
            # Set differences of normalized names (items in A not present in B)
            _set_diff_expr("_mentioned_norm", "_imported_norm").alias(
                "mentions_not_in_imports_normalized"
            ),
            _set_diff_expr("_imported_norm", "_mentioned_norm").alias(
                "imports_not_in_mentions_normalized"
            ),
            _set_diff_expr("_mentioned_norm", "_dependencies_norm").alias(
                "mentions_not_in_dependencies_normalized"
            ),
            _set_diff_expr("_dependencies_norm", "_mentioned_norm").alias(
                "dependencies_not_in_mentions_normalized"
            ),
            _set_diff_expr("_imported_norm", "_dependencies_norm").alias(
                "imports_not_in_dependencies_normalized"
            ),
            _set_diff_expr("_dependencies_norm", "_imported_norm").alias(
                "dependencies_not_in_imports_normalized"
            ),
        )
    )

    # Sample and write annotation file
    sample = complete_cases.sample(n=SAMPLE_SIZE, seed=RANDOM_SEED, shuffle=True)

    sample = sample.with_columns(
        pl.lit(None).cast(pl.String).alias("mentions_imports_differences_notes"),
        pl.lit(None).cast(pl.String).alias("mentions_dependencies_differences_notes"),
        pl.lit(None).cast(pl.String).alias("imports_dependencies_differences_notes"),
        pl.lit(None).cast(pl.String).alias("notes"),
    ).select(
        "document_id",
        "document_doi",
        "document_doi_url",
        "document_title",
        "document_field_name",
        "document_domain_name",
        "repository_id",
        "repository_url",
        "repository_primary_language",
        "mentioned_count",
        "imported_count",
        "dependencies_count",
        "mentioned_software_raw",
        "mentioned_software_normalized",
        "imported_software_raw",
        "imported_software_normalized",
        "dependencies_software_raw",
        "dependencies_software_normalized",
        "mentions_not_in_imports_normalized",
        "imports_not_in_mentions_normalized",
        "mentions_not_in_dependencies_normalized",
        "dependencies_not_in_mentions_normalized",
        "imports_not_in_dependencies_normalized",
        "dependencies_not_in_imports_normalized",
        "mentions_imports_differences_notes",
        "mentions_dependencies_differences_notes",
        "imports_dependencies_differences_notes",
        "notes",
    )

    print(f"Writing {len(sample)} rows to {ANNOTATION_OUTPUT_PATH}")
    sample.write_excel(ANNOTATION_OUTPUT_PATH)
    print("Done.")


###############################################################################

if __name__ == "__main__":
    app()
