from dataclasses import dataclass
from pathlib import Path

import polars as pl
import typer
from datasets import Dataset, load_dataset

###############################################################################

app = typer.Typer()

THIS_FILE_PATH = Path(__file__).resolve()
THIS_DIR = THIS_FILE_PATH.parent

ANNOTATION_CSV_PATH = THIS_DIR / "mentions-imports-deps-annotation.csv"

RANDOM_SEED = 42
SAMPLE_SIZE = 100
RARE_THRESHOLD = 3

# Generic terms excluded from mentions (mirrors attribution-of-software.py)
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


def _remove_extremely_rare_software_usage(
    usage_df: pl.DataFrame,
    usage_type: str,
    software_col: str = "software_name_normalized",
    rare_usage_threshold: int = 3,
    compute_ecosystem_prefix: bool = False,
    use_ecosystem_from_data: bool = False,
    exclude_generic_mentions: bool = False,
) -> pl.DataFrame:
    if compute_ecosystem_prefix:
        # Create column called "ecosystem_normalized_software_name"
        # Which prepends "py:", "r:", "mixed:" to the "software_name_normalized" column
        # based on the "file_paths" column
        # The "file_paths" column is a semi-colon separated list of file paths

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

        # Set software_col to the new ecosystem normalized column for the rest of the function
        software_col = f"ecosystem_{software_col}"

        # Drop "mixed" and "other" ecosystems
        usage_df = usage_df.filter(pl.col("ecosystem").is_in(["py", "r"]))

    elif use_ecosystem_from_data:
        # There is an "ecosystem" column in the data already
        # Use it to prefix the software name
        usage_df = usage_df.with_columns(
            (pl.col("ecosystem") + pl.lit(":") + pl.col(software_col)).alias(
                f"ecosystem_{software_col}"
            )
        )
        software_col = f"ecosystem_{software_col}"

    elif exclude_generic_mentions:
        # Exclude rows where the software_col is in the MENTION_EXCLUDE_NORMALIZED set
        usage_df = usage_df.filter(~pl.col(software_col).is_in(MENTION_EXCLUDE_NORMALIZED))

    # Count usage
    usage_counts = usage_df.group_by(software_col).agg(pl.len().alias("usage_count"))

    # Filter to only usage that were imported at least 3 times
    non_rare_imports = (
        usage_counts.filter(pl.col("usage_count") >= rare_usage_threshold)
        .get_column(software_col)
        .to_list()
    )

    # Filter usage_df to only non-rare usage
    usage_df = usage_df.filter(pl.col(software_col).is_in(non_rare_imports))

    return usage_df


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

    # Create three summary tables:
    # 1. "has_imports_df": document-repository pairs that have at least one repository import
    # 2. "has_dependencies_df": document-repository pairs that have at least one repository dependency
    # 3. "has_software_mentions_df": document-repository pairs that have at least one document software mention
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


def _remove_pairs_with_extreme_software_usage(
    pair_metadata: pl.DataFrame,
    repository_imports: pl.DataFrame,
    repository_dependencies: pl.DataFrame,
    document_software_mentions: pl.DataFrame,
) -> MetadataAndSoftwareUsageDataFrames:
    # Get the threshold value for 99th percentile of number of imports, dependencies, and mentions
    imports_threshold = (
        repository_imports.group_by("document_repository_link_id")
        .agg(imports_count=pl.len())
        .get_column("imports_count")
        .quantile(0.99)
    )
    dependencies_threshold = (
        repository_dependencies.group_by("document_repository_link_id")
        .agg(dependencies_count=pl.len())
        .get_column("dependencies_count")
        .quantile(0.99)
    )
    mentions_threshold = (
        document_software_mentions.group_by("document_repository_link_id")
        .agg(mentions_count=pl.len())
        .get_column("mentions_count")
        .quantile(0.99)
    )

    # Get any document_repository_link_ids that have imports, dependencies, or mentions above these thresholds
    extreme_imports_ids = (
        repository_imports.group_by("document_repository_link_id")
        .agg(imports_count=pl.len())
        .filter(pl.col("imports_count") > imports_threshold)
        .get_column("document_repository_link_id")
        .to_list()
    )
    extreme_dependencies_ids = (
        repository_dependencies.group_by("document_repository_link_id")
        .agg(dependencies_count=pl.len())
        .filter(pl.col("dependencies_count") > dependencies_threshold)
        .get_column("document_repository_link_id")
        .to_list()
    )
    extreme_mentions_ids = (
        document_software_mentions.group_by("document_repository_link_id")
        .agg(mentions_count=pl.len())
        .filter(pl.col("mentions_count") > mentions_threshold)
        .get_column("document_repository_link_id")
        .to_list()
    )

    # Join these lists together to get all extreme usage ids
    extreme_usage_ids = (
        set(extreme_imports_ids) | set(extreme_dependencies_ids) | set(extreme_mentions_ids)
    )

    # Filter the pair_metadata to remove these extreme usage ids
    pair_metadata = pair_metadata.filter(
        ~pl.col("document_repository_link_id").is_in(extreme_usage_ids)
    )

    # Filter the repository_imports, repository_dependencies, and document_software_mentions to only the remaining pairs
    repository_imports = repository_imports.filter(
        pl.col("document_repository_link_id").is_in(
            pair_metadata.get_column("document_repository_link_id").to_list()
        )
    )
    repository_dependencies = repository_dependencies.filter(
        pl.col("document_repository_link_id").is_in(
            pair_metadata.get_column("document_repository_link_id").to_list()
        )
    )
    document_software_mentions = document_software_mentions.filter(
        pl.col("document_repository_link_id").is_in(
            pair_metadata.get_column("document_repository_link_id").to_list()
        )
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

    # Remove extremely rare software (mirrors attribution-of-software.py)
    repository_imports = _remove_extremely_rare_software_usage(
        repository_imports,
        usage_type="imports",
        software_col="software_name_normalized",
        rare_usage_threshold=RARE_THRESHOLD,
        compute_ecosystem_prefix=True,
        use_ecosystem_from_data=False,
        exclude_generic_mentions=False,
    )
    repository_dependencies = _remove_extremely_rare_software_usage(
        repository_dependencies,
        usage_type="dependencies",
        software_col="software_name_normalized",
        rare_usage_threshold=RARE_THRESHOLD,
        compute_ecosystem_prefix=False,
        use_ecosystem_from_data=True,
        exclude_generic_mentions=False,
    )
    document_software_mentions = _remove_extremely_rare_software_usage(
        document_software_mentions,
        usage_type="mentions",
        software_col="software_name_normalized",
        rare_usage_threshold=RARE_THRESHOLD,
        compute_ecosystem_prefix=False,
        use_ecosystem_from_data=False,
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
    repository_imports = repository_imports.filter(
        pl.col("ecosystem_software_name_normalized").is_in(
            repository_imports.group_by("ecosystem_software_name_normalized")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") >= RARE_THRESHOLD)
            .get_column("ecosystem_software_name_normalized")
            .to_list()
        )
    )
    document_software_mentions = document_software_mentions.filter(
        pl.col("software_name_normalized").is_in(
            document_software_mentions.group_by("software_name_normalized")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") >= RARE_THRESHOLD)
            .get_column("software_name_normalized")
            .to_list()
        )
    )

    complete_cases_count = len(
        pair_metadata.filter(
            pl.col("has_imports") & pl.col("has_dependencies") & pl.col("has_software_mentions")
        )
    )
    print(f"Total pairs: {len(pair_metadata)}")
    print(f"Complete cases: {complete_cases_count}")


###############################################################################

if __name__ == "__main__":
    app()
