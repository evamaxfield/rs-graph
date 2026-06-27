import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
from tqdm import tqdm

from rs_graph.utils.software_alignment import align_software_names

###############################################################################

app = typer.Typer()

THIS_FILE_PATH = Path(__file__).resolve()
THIS_DIR = THIS_FILE_PATH.parent
RESULTS_DIR = THIS_DIR / "results" / "attribution-of-software"

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


def _get_counts_and_proportions_of_each_software_usage_type(
    pair_metadata: pl.DataFrame,
) -> pl.DataFrame:
    # We want a dataframe of "usage_type", "count", "proportion"
    # "usage_type" can be imports, dependencies, mentions, or combinations of those (e.g. imports+mentions)
    # we also want "complete-cases"
    imports_count = len(pair_metadata.filter(pl.col("has_imports")))
    dependencies_count = len(pair_metadata.filter(pl.col("has_dependencies")))
    mentions_count = len(pair_metadata.filter(pl.col("has_software_mentions")))
    imports_and_dependencies_count = len(
        pair_metadata.filter(pl.col("has_imports") & pl.col("has_dependencies"))
    )
    imports_and_mentions_count = len(
        pair_metadata.filter(pl.col("has_imports") & pl.col("has_software_mentions"))
    )
    dependencies_and_mentions_count = len(
        pair_metadata.filter(pl.col("has_dependencies") & pl.col("has_software_mentions"))
    )
    complete_cases_count = len(
        pair_metadata.filter(
            pl.col("has_imports") & pl.col("has_dependencies") & pl.col("has_software_mentions")
        )
    )
    total_count = len(pair_metadata)

    counts = pl.DataFrame(
        [
            {
                "usage_type": "imports",
                "count": imports_count,
                "proportion": imports_count / total_count,
            },
            {
                "usage_type": "dependencies",
                "count": dependencies_count,
                "proportion": dependencies_count / total_count,
            },
            {
                "usage_type": "mentions",
                "count": mentions_count,
                "proportion": mentions_count / total_count,
            },
            {
                "usage_type": "imports+dependencies",
                "count": imports_and_dependencies_count,
                "proportion": imports_and_dependencies_count / total_count,
            },
            {
                "usage_type": "imports+mentions",
                "count": imports_and_mentions_count,
                "proportion": imports_and_mentions_count / total_count,
            },
            {
                "usage_type": "dependencies+mentions",
                "count": dependencies_and_mentions_count,
                "proportion": dependencies_and_mentions_count / total_count,
            },
            {
                "usage_type": "complete-cases",
                "count": complete_cases_count,
                "proportion": complete_cases_count / total_count,
            },
        ]
    )

    return counts


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


def _match_imports_and_mentions_and_get_long_frame(
    pair_metadata: pl.DataFrame,
    repository_imports: pl.DataFrame,
    document_software_mentions: pl.DataFrame,
    matching_score_threshold: float = 75.0,
):
    # Get the subset of pairs that have at least one import and at least one mention
    pairs_with_imports_and_mentions = pair_metadata.filter(
        pl.col("has_imports") & pl.col("has_software_mentions")
    )

    # For each pair, get the subset of imports and the subset of mentions
    # run the alignment algorithm to find which imports were mentioned, then build a long-format table
    imports_and_mentions_matched_rows = []
    for pair_details in tqdm(
        pairs_with_imports_and_mentions.iter_rows(named=True),
        total=len(pairs_with_imports_and_mentions),
        desc="Aligning imports and mentions for each document-repository pair",
    ):
        # Get basic metadata for this document-repository pair
        this_pair_document_id = pair_details["document_id"]
        this_pair_publication_year = pair_details["document_publication_year"]
        this_pair_field = pair_details["document_field_name"]
        this_pair_pruned_field = pair_details["document_field_name_pruned"]
        this_pair_repository_id = pair_details["repository_id"]

        # Get the imports and mentions for this document-repository pair
        this_pair_imports = repository_imports.filter(
            pl.col("repository_id") == this_pair_repository_id
        )
        this_pair_mentions = document_software_mentions.filter(
            pl.col("document_id") == this_pair_document_id
        )

        # Align imports and mentions to find which imported libraries were mentioned
        # this only returns pairs that are matched, so we will need to add unmatched imports with is_mentioned=False later
        normalized_imported_software_names = this_pair_imports.get_column(
            "software_name_normalized"
        ).to_list()
        normalized_mentioned_software_names = this_pair_mentions.get_column(
            "software_name_normalized"
        ).to_list()
        matched_imports_and_mentions = align_software_names(
            items_a=normalized_imported_software_names,
            items_b=normalized_mentioned_software_names,
            source_a="import",
            source_b="mention",
            cutoff=matching_score_threshold,
        )

        # Add matched pairs to the long-format table
        for matched_import_and_mention in matched_imports_and_mentions:
            imports_and_mentions_matched_rows.append(
                {
                    "document_id": this_pair_document_id,
                    "publication_year": this_pair_publication_year,
                    "original_field": this_pair_field,
                    "pruned_field": this_pair_pruned_field,
                    "ecosystem": this_pair_imports.filter(
                        pl.col("software_name_normalized")
                        == matched_import_and_mention.normalized_item_one
                    )
                    .get_column("ecosystem")
                    .first(),
                    "library_name_normalized": matched_import_and_mention.normalized_item_one,
                    "is_imported": True,
                    "is_mentioned": True,
                }
            )

        # Add unmatched imports with is_mentioned=False
        unmatched_imports = (
            set(normalized_imported_software_names)
            - {match.normalized_item_one for match in matched_imports_and_mentions}
            - {match.normalized_item_two for match in matched_imports_and_mentions}
        )
        for unmatched_import in unmatched_imports:
            imports_and_mentions_matched_rows.append(
                {
                    "document_id": this_pair_document_id,
                    "publication_year": this_pair_publication_year,
                    "original_field": this_pair_field,
                    "pruned_field": this_pair_pruned_field,
                    "ecosystem": this_pair_imports.filter(
                        pl.col("software_name_normalized") == unmatched_import
                    )
                    .get_column("ecosystem")
                    .first(),
                    "library_name_normalized": unmatched_import,
                    "is_imported": True,
                    "is_mentioned": False,
                }
            )

        # Add unmatched mentions with is_imported=False
        unmatched_mentions = (
            set(normalized_mentioned_software_names)
            - {match.normalized_item_one for match in matched_imports_and_mentions}
            - {match.normalized_item_two for match in matched_imports_and_mentions}
        )
        for unmatched_mention in unmatched_mentions:
            imports_and_mentions_matched_rows.append(
                {
                    "document_id": this_pair_document_id,
                    "publication_year": this_pair_publication_year,
                    "original_field": this_pair_field,
                    "pruned_field": this_pair_pruned_field,
                    "ecosystem": None,
                    "library_name_normalized": unmatched_mention,
                    "is_imported": False,
                    "is_mentioned": True,
                }
            )

    # Convert the long-format table to a DataFrame
    imports_and_mentions_long_df = pl.DataFrame(imports_and_mentions_matched_rows)

    return imports_and_mentions_long_df


def _get_descriptive_stats_and_tables(
    imports_and_mentions_long_df: pl.DataFrame,
) -> None:
    # Print dataframe with example rows where software is imported but not mentioned, and where software is mentioned but not imported
    print("Examples of imported but not mentioned software:")
    print(imports_and_mentions_long_df.filter(pl.col("is_imported") & ~pl.col("is_mentioned")))
    print("Examples of mentioned but not imported software:")
    print(imports_and_mentions_long_df.filter(~pl.col("is_imported") & pl.col("is_mentioned")))

    # Compute number of unique packages imported, depended on, and mentioned for each document-repository pair
    total_unique_imports = (
        imports_and_mentions_long_df.filter(pl.col("is_imported"))
        .get_column("library_name_normalized")
        .n_unique()
    )
    total_unique_mentions = (
        imports_and_mentions_long_df.filter(pl.col("is_mentioned"))
        .get_column("library_name_normalized")
        .n_unique()
    )
    total_unique_matched = (
        imports_and_mentions_long_df.filter(pl.col("is_imported") & pl.col("is_mentioned"))
        .get_column("library_name_normalized")
        .n_unique()
    )
    per_pair_imports_counts = (
        imports_and_mentions_long_df.filter(pl.col("is_imported"))
        .group_by("document_id")
        .agg(unique_imports_count=pl.col("library_name_normalized").n_unique())
    )
    per_pair_mentions_counts = (
        imports_and_mentions_long_df.filter(pl.col("is_mentioned"))
        .group_by("document_id")
        .agg(unique_mentions_count=pl.col("library_name_normalized").n_unique())
    )
    per_pair_matched_counts = (
        imports_and_mentions_long_df.filter(pl.col("is_imported") & pl.col("is_mentioned"))
        .group_by("document_id")
        .agg(unique_matched_count=pl.col("library_name_normalized").n_unique())
    )
    print(f"Total unique imports: {total_unique_imports}")
    print(f"Total unique mentions: {total_unique_mentions}")
    print(f"Total unique matched imports and mentions: {total_unique_matched}")
    print(
        f"Average unique imports per pair -- "
        f"Median: {per_pair_imports_counts.get_column('unique_imports_count').median()}; "
        f"Mean: {per_pair_imports_counts.get_column('unique_imports_count').mean()} "
        f"({per_pair_imports_counts.get_column('unique_imports_count').std()})"
    )
    print(
        f"Average unique mentions per pair -- "
        f"Median: {per_pair_mentions_counts.get_column('unique_mentions_count').median()}; "
        f"Mean: {per_pair_mentions_counts.get_column('unique_mentions_count').mean()} "
        f"({per_pair_mentions_counts.get_column('unique_mentions_count').std()})"
    )
    print(
        f"Average unique matched imports and mentions per pair -- "
        f"Median: {per_pair_matched_counts.get_column('unique_matched_count').median()}; "
        f"Mean: {per_pair_matched_counts.get_column('unique_matched_count').mean()} "
        f"({per_pair_matched_counts.get_column('unique_matched_count').std()})"
    )

    # Get the top libraries that are imported and mentioned
    # Get the top libraries that are imported but not mentioned
    # Get the top libraries that are mentioned but not imported
    for filter_name, filter_expr, split_ecosystem in [
        ("imported-and-mentioned", pl.col("is_imported") & pl.col("is_mentioned"), True),
        ("imported-not-mentioned", pl.col("is_imported") & ~pl.col("is_mentioned"), True),
        ("mentioned-not-imported", ~pl.col("is_imported") & pl.col("is_mentioned"), False),
    ]:
        filtered_set_of_matched_imports_and_mentions = imports_and_mentions_long_df.filter(
            filter_expr
        )

        # Handle split by ecosystem or not
        if split_ecosystem:
            for ecosystem in filtered_set_of_matched_imports_and_mentions.get_column(
                "ecosystem"
            ).unique():
                total_documents = (
                    filtered_set_of_matched_imports_and_mentions.filter(
                        pl.col("ecosystem") == ecosystem
                    )
                    .get_column("document_id")
                    .n_unique()
                )

                top_libraries = (
                    filtered_set_of_matched_imports_and_mentions.filter(
                        pl.col("ecosystem") == ecosystem
                    )
                    .group_by("library_name_normalized")
                    .agg(pl.len().alias("count"))
                    .sort("count", descending=True)
                    .head(50)
                ).with_columns(
                    (pl.col("count") / total_documents).alias("proportion_of_documents")
                )
                top_libraries.write_csv(
                    RESULTS_DIR / f"top-fifty-libraries-{filter_name}-{ecosystem}.csv"
                )
        else:
            total_documents = filtered_set_of_matched_imports_and_mentions.get_column(
                "document_id"
            ).n_unique()

            top_libraries = (
                filtered_set_of_matched_imports_and_mentions.group_by("library_name_normalized")
                .agg(pl.len().alias("count"))
                .sort(
                    "count",
                    descending=True,
                )
                .with_columns(
                    (pl.col("count") / total_documents).alias("proportion_of_documents")
                )
                .head(50)
            )
            top_libraries.write_csv(RESULTS_DIR / f"top-fifty-libraries-{filter_name}.csv")

    # Get the libraries that have the highest ratios of imports to mentions
    # Get the libraries that have the highest ratios of mentions to imports
    for ecosystem in imports_and_mentions_long_df.get_column("ecosystem").unique():
        if ecosystem is None:
            ecosystem_filter = pl.col("ecosystem").is_null()
        else:
            ecosystem_filter = pl.col("ecosystem") == ecosystem
        library_imports_mentions_ratios = (
            imports_and_mentions_long_df.filter(ecosystem_filter)
            .group_by("library_name_normalized")
            .agg(
                imports_count=pl.col("is_imported").sum(),
                mentions_count=pl.col("is_mentioned").sum(),
            )
            .with_columns(
                import_to_mention_ratio=pl.when(pl.col("mentions_count") > 0)
                .then(pl.col("imports_count") / pl.col("mentions_count"))
                .otherwise(pl.lit(None)),
                mention_to_import_ratio=pl.when(pl.col("imports_count") > 0)
                .then(pl.col("mentions_count") / pl.col("imports_count"))
                .otherwise(pl.lit(None)),
            )
        )

        for threshold_name, threshold_value in [
            ("at-least-10", 10),
            ("at-least-100", 100),
        ]:
            top_import_to_mention_ratio = (
                library_imports_mentions_ratios.filter(
                    pl.col("import_to_mention_ratio").is_not_null()
                    & (pl.col("imports_count") >= threshold_value)
                )
                .sort("import_to_mention_ratio", descending=True)
                .head(50)
            )
            top_mention_to_import_ratio = (
                library_imports_mentions_ratios.filter(
                    pl.col("mention_to_import_ratio").is_not_null()
                    & (pl.col("mentions_count") >= threshold_value)
                )
                .sort("mention_to_import_ratio", descending=True)
                .head(50)
            )
            top_import_to_mention_ratio.write_csv(
                RESULTS_DIR
                / f"top-fifty-libraries-by-import-to-mention-ratio-{ecosystem}-{threshold_name}.csv"
            )
            top_mention_to_import_ratio.write_csv(
                RESULTS_DIR
                / f"top-fifty-libraries-by-mention-to-import-ratio-{ecosystem}-{threshold_name}.csv"
            )


def _compute_cumulative_imports_by_library_year(
    imports_and_mentions_long_df: pl.DataFrame,
) -> pl.DataFrame:
    # Cumulative import count per library through end of each publication_year.
    per_year = (
        imports_and_mentions_long_df.filter(pl.col("is_imported"))
        .group_by("library_name_normalized", "publication_year")
        .agg(n_imports_in_year=pl.len())
    )
    return per_year.sort("library_name_normalized", "publication_year").with_columns(
        cumulative_imports_at_year=pl.col("n_imports_in_year")
        .cum_sum()
        .over("library_name_normalized")
    )


def _compute_probability_of_mention_given_import_over_time(
    imports_and_mentions_long_df: pl.DataFrame,
) -> None:
    # Filter to libraries that were imported at least 20 times across all years
    library_import_counts = (
        imports_and_mentions_long_df.filter(pl.col("is_imported"))
        .group_by("library_name_normalized")
        .agg(total_imports=pl.len())
        .filter(pl.col("total_imports") >= 20)
    )
    libraries_to_investigate = library_import_counts.get_column(
        "library_name_normalized"
    ).to_list()

    # For each of these libraries, group by publication_year and calculate p(mention | import)
    # Add this calculation to the imports_and_mentions_long_df DataFrame as a new column "p_mention_given_import"
    imports_and_mentions_long_df = imports_and_mentions_long_df.filter(
        pl.col("library_name_normalized").is_in(libraries_to_investigate)
        & pl.col("is_imported")
    )
    per_library_year = (
        imports_and_mentions_long_df.group_by("library_name_normalized", "publication_year")
        .agg(
            n_imported=pl.len(),
            n_mentioned=pl.sum("is_mentioned"),
        )
        .with_columns(
            (pl.col("n_mentioned") / pl.col("n_imported")).alias("p_mention_given_import")
        )
    )

    # Average across libraries
    # (equal weight per library) for each year
    # Filter to library-year combos with enough observations
    aggregate_by_year = (
        per_library_year.filter(pl.col("n_imported") >= 5)  # min obs per library-year cell
        .group_by("publication_year")
        .agg(
            mean_p=pl.mean("p_mention_given_import"),
            std_p=pl.std("p_mention_given_import"),
            n_libraries=pl.len(),
        )
        .with_columns((pl.col("std_p") / pl.col("n_libraries").sqrt()).alias("se_p"))
        .sort("publication_year")
    )

    # Only include years that have at least 10 libraries with observations
    aggregate_by_year = aggregate_by_year.filter(pl.col("n_libraries") >= 10)

    # Plot the aggregate curve with confidence intervals
    # Also plot the trends in a few specific libraries
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: Aggregate curve (equal weight per library) ──
    agg = aggregate_by_year.sort("publication_year").to_pandas()

    ax1.plot(agg["publication_year"], agg["mean_p"], "o-", color="#2c7bb6", linewidth=2)
    ax1.fill_between(
        agg["publication_year"],
        agg["mean_p"] - 1.96 * agg["se_p"],
        agg["mean_p"] + 1.96 * agg["se_p"],
        alpha=0.2,
        color="#2c7bb6",
    )
    ax1.set_xlabel("Publication year")
    ax1.set_ylabel("p(mention | import)")
    ax1.set_title("Aggregate across all libraries\n(equal weight per library)")
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Annotate with number of libraries per year
    for _, row in agg.iterrows():
        ax1.annotate(
            f"n={int(row['n_libraries'])}",
            (row["publication_year"], row["mean_p"]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=7,
            ha="center",
            color="gray",
        )

    # ── Right panel: Individual library trajectories ──
    # Pick a few libraries to highlight
    python_spotlight_libraries = [
        "numpy",
        "pandas",
        "matplotlib",
        "tensorflow",
        "torch",
    ]
    r_spotlight_libraries = [
        "ggplot2",
        "dplyr",
        "mass",
        "survival",
        "lme4",
    ]
    spotlight_libraries = [
        *python_spotlight_libraries,
        *r_spotlight_libraries,
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(spotlight_libraries)))  # type: ignore

    for lib, color in zip(spotlight_libraries, colors, strict=True):
        lib_data = (
            per_library_year.filter(
                (pl.col("library_name_normalized") == lib)
                & pl.col("publication_year").is_between(2014, 2023)
                & (pl.col("n_imported") >= 5)
            )
            .sort("publication_year")
            .to_pandas()
        )
        if len(lib_data) > 0:
            ax2.plot(
                lib_data["publication_year"],
                lib_data["p_mention_given_import"],
                "o-" if lib in python_spotlight_libraries else "s-",
                label=lib,
                color=color,
                linewidth=1.5,
                markersize=4,
            )

    ax2.set_xlabel("Publication year")
    ax2.set_ylabel("p(mention | import)")
    ax2.set_title("Individual library trajectories")
    ax2.set_ylim(bottom=0)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.legend(fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "p-mention-given-import-two-panel.png", dpi=300, bbox_inches="tight"
    )


def _compute_probability_of_mention_given_popularity(
    imports_and_mentions_long_df: pl.DataFrame,
) -> None:
    # For libraries that were imported at least 20 times lifetime, compute
    # p(mention | import) per library-year and plot against popularity
    # (cumulative imports) as of that publication year.
    library_import_counts = (
        imports_and_mentions_long_df.filter(pl.col("is_imported"))
        .group_by("library_name_normalized")
        .agg(total_imports=pl.len())
        .filter(pl.col("total_imports") >= 20)
    )
    libraries_to_investigate = library_import_counts.get_column(
        "library_name_normalized"
    ).to_list()

    cumulative_imports_by_year = _compute_cumulative_imports_by_library_year(
        imports_and_mentions_long_df
    )

    min_imports_per_cell = 5

    per_library_year = (
        imports_and_mentions_long_df.filter(
            pl.col("library_name_normalized").is_in(libraries_to_investigate)
            & pl.col("is_imported")
        )
        .group_by("library_name_normalized", "publication_year")
        .agg(
            n_imports_in_year=pl.len(),
            n_mentions_in_year=pl.sum("is_mentioned"),
            ecosystem=pl.col("ecosystem").first(),
        )
        .join(
            cumulative_imports_by_year.select(
                "library_name_normalized", "publication_year", "cumulative_imports_at_year"
            ),
            on=["library_name_normalized", "publication_year"],
        )
        .with_columns(
            p_mention_given_import=(pl.col("n_mentions_in_year") / pl.col("n_imports_in_year")),
        )
        .filter(pl.col("n_imports_in_year") >= min_imports_per_cell)
    )

    # Plot p(mention | import) vs cumulative importing projects at publication year
    plt.figure(figsize=(6, 5))
    plt.scatter(
        per_library_year.get_column("cumulative_imports_at_year"),
        per_library_year.get_column("p_mention_given_import"),
        alpha=0.4,
        s=14,
    )

    # Label spotlight libraries at their most recent library-year point
    py_libraries_to_label = ["numpy", "pandas", "tensorflow", "torch"]
    r_libraries_to_label = ["data.table", "dplyr", "lme4"]
    labeled_libs = py_libraries_to_label + r_libraries_to_label
    latest_points = (
        per_library_year.filter(pl.col("library_name_normalized").is_in(labeled_libs))
        .sort("publication_year")
        .group_by("library_name_normalized")
        .last()
        .to_pandas()
    )
    for _, row in latest_points.iterrows():
        row_details = row.to_dict()
        plt.annotate(
            row_details["library_name_normalized"],
            (
                row_details["cumulative_imports_at_year"],
                row_details["p_mention_given_import"],
            ),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=7,
            ha="center",
        )

    plt.xscale("log")
    plt.xlabel("Cumulative importing projects through publication year (log scale)")
    plt.ylabel("p(mention | import)")
    plt.title(
        "p(mention | import) vs cumulative importing projects\n(one point per library-year)"
    )
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(
        RESULTS_DIR / "p-mention-given-import-vs-popularity.png", dpi=300, bbox_inches="tight"
    )

    # Faceted by field: p(mention | import) per library-field-year vs library-level
    # cumulative popularity at that publication year.
    per_library_field_year = (
        imports_and_mentions_long_df.filter(
            pl.col("library_name_normalized").is_in(libraries_to_investigate)
            & pl.col("is_imported")
        )
        .group_by("library_name_normalized", "pruned_field", "publication_year")
        .agg(
            n_imports_in_year=pl.len(),
            n_mentions_in_year=pl.sum("is_mentioned"),
        )
        .join(
            cumulative_imports_by_year.select(
                "library_name_normalized", "publication_year", "cumulative_imports_at_year"
            ),
            on=["library_name_normalized", "publication_year"],
        )
        .with_columns(
            p_mention_given_import=(pl.col("n_mentions_in_year") / pl.col("n_imports_in_year")),
        )
        .filter(pl.col("n_imports_in_year") >= min_imports_per_cell)
    )
    fields = per_library_field_year.get_column("pruned_field").unique().sort()
    n_fields = len(fields)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    _fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), sharex=True, sharey=True
    )
    for field, ax in zip(fields, axes.flatten(), strict=True):
        field_data = per_library_field_year.filter(pl.col("pruned_field") == field).to_pandas()
        ax.scatter(
            field_data["cumulative_imports_at_year"],
            field_data["p_mention_given_import"],
            alpha=0.4,
            s=14,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Cumulative importing projects through publication year (log scale)")
        ax.set_ylabel("p(mention | import)")
        ax.set_title(f"Field: {field}")
        ax.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "p-mention-given-import-vs-popularity-by-field.png",
        dpi=300,
        bbox_inches="tight",
    )


def _compute_probability_of_mention_since_year_of_first_import(
    imports_and_mentions_long_df: pl.DataFrame,
) -> None:
    # For libraries that were imported at least 20 times, compute p(mention | import)
    # and see how it varies by import popularity
    library_import_counts = (
        imports_and_mentions_long_df.filter(pl.col("is_imported"))
        .group_by("library_name_normalized")
        .agg(total_imports=pl.len())
        .filter(pl.col("total_imports") >= 20)
    )
    libraries_to_investigate = library_import_counts.get_column(
        "library_name_normalized"
    ).to_list()

    # Filter to these libraries and only imported rows
    imports_and_mentions_long_df = imports_and_mentions_long_df.filter(
        pl.col("library_name_normalized").is_in(libraries_to_investigate)
        & pl.col("is_imported")
    )

    # All uses in the first year (t=0) may have higher probablity
    # of mention than later years, so we want to look at trends in p(mention | import)
    # as a function of time since first usage (import)
    first_import_year = (
        imports_and_mentions_long_df.filter(pl.col("is_imported"))
        .group_by("library_name_normalized")
        .agg(first_import_year=pl.col("publication_year").min())
    )

    imports_and_mentions_long_df = imports_and_mentions_long_df.join(
        first_import_year,
        on="library_name_normalized",
        how="left",
    ).with_columns(
        years_since_first_import=pl.col("publication_year") - pl.col("first_import_year")
    )
    per_library_years_since_first_import = (
        imports_and_mentions_long_df.filter(pl.col("is_imported"))
        .group_by("library_name_normalized", "years_since_first_import")
        .agg(
            n_imported=pl.len(),
            n_mentioned=pl.sum("is_mentioned"),
        )
        .with_columns(
            p_mention_given_import=(pl.col("n_mentioned") / pl.col("n_imported")),
        )
    )

    # We want to plot the mean + confidence interval
    # of p(mention | import) as a function of years since first import
    # (equal weight per library)
    aggregate_by_years_since_first_import = (
        per_library_years_since_first_import.group_by("years_since_first_import")
        .agg(
            mean_p=pl.mean("p_mention_given_import"),
            std_p=pl.std("p_mention_given_import"),
            n_libraries=pl.len(),
        )
        .with_columns((pl.col("std_p") / pl.col("n_libraries").sqrt()).alias("se_p"))
        .sort("years_since_first_import")
    )

    # Plot the curve with confidence intervals
    plt.figure(figsize=(6, 5))
    agg = aggregate_by_years_since_first_import.to_pandas()
    plt.plot(
        agg["years_since_first_import"],
        agg["mean_p"],
        "o-",
        color="#2c7bb6",
        linewidth=2,
    )
    plt.fill_between(
        agg["years_since_first_import"],
        agg["mean_p"] - 1.96 * agg["se_p"],
        agg["mean_p"] + 1.96 * agg["se_p"],
        alpha=0.2,
        color="#2c7bb6",
    )

    # # Mark the number of observations per year since first import
    # for _, row in agg.iterrows():
    #     row_details = row.to_dict()
    #     years_since_first_import: float = row_details["years_since_first_import"]
    #     mean_p: float = row_details["mean_p"]
    #     plt.annotate(
    #         f"n={int(row['n_libraries'])}",
    #         (years_since_first_import, mean_p),
    #         textcoords="offset points",
    #         xytext=(0, 10),
    #         fontsize=7,
    #         ha="center",
    #         color="gray",
    #     )

    plt.xlabel("Years since first import")
    plt.ylabel("p(mention | import)")
    plt.title("p(mention | import) vs years since first import\n(equal weight per library)")
    plt.grid(True, ls="--", lw=0.5)
    plt.savefig(
        RESULTS_DIR / "p-mention-given-import-vs-years-since-first-import.png",
        dpi=300,
        bbox_inches="tight",
    )


def _plot_individual_library_trajectories_since_year_of_first_import(
    imports_and_mentions_long_df: pl.DataFrame,
) -> None:
    # Mirror the "individual library trajectories" panel of the two-panel figure,
    # but place years-since-first-import on the x-axis (matching the aggregate age plot
    # in p-mention-given-import-vs-years-since-first-import.png) instead of publication year.
    library_import_counts = (
        imports_and_mentions_long_df.filter(pl.col("is_imported"))
        .group_by("library_name_normalized")
        .agg(total_imports=pl.len())
        .filter(pl.col("total_imports") >= 20)
    )
    libraries_to_investigate = library_import_counts.get_column(
        "library_name_normalized"
    ).to_list()

    imported_df = imports_and_mentions_long_df.filter(
        pl.col("library_name_normalized").is_in(libraries_to_investigate)
        & pl.col("is_imported")
    )

    # Year of first import per library, then years since first import per row
    first_import_year = imported_df.group_by("library_name_normalized").agg(
        first_import_year=pl.col("publication_year").min()
    )
    per_library_years_since_first_import = (
        imported_df.join(first_import_year, on="library_name_normalized", how="left")
        .with_columns(
            years_since_first_import=pl.col("publication_year") - pl.col("first_import_year")
        )
        .group_by("library_name_normalized", "years_since_first_import")
        .agg(
            n_imported=pl.len(),
            n_mentioned=pl.sum("is_mentioned"),
        )
        .with_columns(
            p_mention_given_import=(pl.col("n_mentioned") / pl.col("n_imported")),
        )
    )

    # Spotlight libraries for this plot: Python giants only
    # (torch and all R libraries are intentionally excluded here)
    spotlight_libraries = [
        "numpy",
        "pandas",
        "matplotlib",
        "tensorflow",
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(spotlight_libraries)))  # type: ignore

    plt.figure(figsize=(6, 5))
    for lib, color in zip(spotlight_libraries, colors, strict=True):
        lib_data = (
            per_library_years_since_first_import.filter(
                (pl.col("library_name_normalized") == lib) & (pl.col("n_imported") >= 5)
            )
            .sort("years_since_first_import")
            .to_pandas()
        )
        if len(lib_data) > 0:
            plt.plot(
                lib_data["years_since_first_import"],
                lib_data["p_mention_given_import"],
                "o-",
                label=lib,
                color=color,
                linewidth=1.5,
                markersize=4,
            )

    plt.xlabel("Years since first import")
    plt.ylabel("p(mention | import)")
    plt.title("Individual library trajectories\n(by years since first import)")
    plt.ylim(bottom=0)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.legend(fontsize=8, loc="upper right", ncol=2)
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "p-mention-given-import-trajectories-vs-years-since-first-import.png",
        dpi=300,
        bbox_inches="tight",
    )


def _analysis_age_vs_popularity_logistic_regression(
    imports_and_mentions_long_df: pl.DataFrame,
    pair_metadata: pl.DataFrame,
) -> None:
    """Logistic regression disentangling library age from popularity."""
    # Start with imported rows only
    imported_df = imports_and_mentions_long_df.filter(pl.col("is_imported"))

    # Cumulative imports per (library, publication_year) through end of that year.
    # Popularity at the time of publication, not lifetime popularity.
    cumulative_imports_by_year = _compute_cumulative_imports_by_library_year(
        imports_and_mentions_long_df
    ).select("library_name_normalized", "publication_year", "cumulative_imports_at_year")

    # Compute first import year per library
    first_import_year = imported_df.group_by("library_name_normalized").agg(
        first_import_year=pl.col("publication_year").min()
    )

    # Compute per-document num_unique_software_imported
    doc_import_counts = imported_df.group_by("document_id").agg(
        num_unique_software_imported=pl.col("library_name_normalized").n_unique()
    )

    # Build the regression dataframe
    regression_df = (
        imported_df.join(
            cumulative_imports_by_year, on=["library_name_normalized", "publication_year"]
        )
        .join(first_import_year, on="library_name_normalized")
        .with_columns(
            log_cumulative_imports=pl.col("cumulative_imports_at_year").cast(pl.Float64).log(),
            years_since_first_import=(pl.col("publication_year") - pl.col("first_import_year")),
        )
        .join(doc_import_counts, on="document_id")
        .join(
            pair_metadata.select(
                "document_id",
                "document_years_since_earliest",
                "document_author_count",
                "document_log_author_mean_citations",
                "document_field_name_pruned",
            ),
            on="document_id",
        )
    )

    # Convert booleans for statsmodels and drop nulls
    model_cols = [
        "is_mentioned",
        "years_since_first_import",
        "log_cumulative_imports",
        "ecosystem",
        "document_years_since_earliest",
        "num_unique_software_imported",
        "document_author_count",
        "document_log_author_mean_citations",
        "document_field_name_pruned",
        "library_name_normalized",
    ]
    regression_pd = (
        regression_df.select(model_cols)
        .with_columns(pl.col("is_mentioned").cast(pl.Int8))
        .drop_nulls()
        .to_pandas()
    )

    print()
    print("Analysis 1: Age vs Popularity logistic regression")
    print(f"N observations: {len(regression_pd)}")
    print(f"N unique libraries: {regression_pd['library_name_normalized'].nunique()}")

    # Collinearity diagnostics
    age_pop_corr, age_pop_pval = stats.pearsonr(
        regression_pd["years_since_first_import"],
        regression_pd["log_cumulative_imports"],
    )
    print(f"Pearson r(age, log_popularity) = {age_pop_corr:.4f}, p = {age_pop_pval:.2e}")

    continuous_controls = [
        "years_since_first_import",
        "log_cumulative_imports",
        "document_years_since_earliest",
        "num_unique_software_imported",
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
        "Collinearity Diagnostics for Analysis 1",
        "=" * 50,
        "",
        "Pearson correlation (years_since_first_import, log_cumulative_imports):",
        f"  r = {age_pop_corr:.4f}, p = {age_pop_pval:.2e}",
        "",
        "Variance Inflation Factors (continuous predictors in controlled model):",
    ]
    for var, vif_val in vif_results.items():
        if var == "const":
            continue
        collinearity_lines.append(f"  {var}: {vif_val:.2f}")
    (RESULTS_DIR / "age-vs-popularity-collinearity.txt").write_text(
        "\n".join(collinearity_lines)
    )
    print("Collinearity diagnostics saved.")

    # Raw model (key predictors only)
    raw_model = smf.logit(
        "is_mentioned ~ years_since_first_import + log_cumulative_imports + C(ecosystem)",
        data=regression_pd,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": regression_pd["library_name_normalized"]},
        maxiter=1000,
    )

    # Controlled model (with standard controls)
    controlled_model = smf.logit(
        "is_mentioned ~ years_since_first_import + log_cumulative_imports"
        " + document_years_since_earliest"
        " + num_unique_software_imported"
        " + document_author_count"
        " + document_log_author_mean_citations"
        " + C(document_field_name_pruned)"
        " + C(ecosystem)",
        data=regression_pd,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": regression_pd["library_name_normalized"]},
        maxiter=1000,
    )

    # Single-predictor models (to probe collinearity)
    age_only_model = smf.logit(
        "is_mentioned ~ years_since_first_import + C(ecosystem)",
        data=regression_pd,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": regression_pd["library_name_normalized"]},
        maxiter=1000,
    )

    popularity_only_model = smf.logit(
        "is_mentioned ~ log_cumulative_imports + C(ecosystem)",
        data=regression_pd,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": regression_pd["library_name_normalized"]},
        maxiter=1000,
    )

    # Save model summaries
    (RESULTS_DIR / "age-vs-popularity-logit-raw.txt").write_text(raw_model.summary().as_text())
    (RESULTS_DIR / "age-vs-popularity-logit-raw-marginal-effects.txt").write_text(
        raw_model.get_margeff(at="overall").summary().as_text()
    )
    (RESULTS_DIR / "age-vs-popularity-logit-controlled.txt").write_text(
        controlled_model.summary().as_text()
    )
    (RESULTS_DIR / "age-vs-popularity-logit-controlled-marginal-effects.txt").write_text(
        controlled_model.get_margeff(at="overall").summary().as_text()
    )
    (RESULTS_DIR / "age-vs-popularity-logit-age-only.txt").write_text(
        age_only_model.summary().as_text()
    )
    (RESULTS_DIR / "age-vs-popularity-logit-age-only-marginal-effects.txt").write_text(
        age_only_model.get_margeff(at="overall").summary().as_text()
    )
    (RESULTS_DIR / "age-vs-popularity-logit-popularity-only.txt").write_text(
        popularity_only_model.summary().as_text()
    )
    (RESULTS_DIR / "age-vs-popularity-logit-popularity-only-marginal-effects.txt").write_text(
        popularity_only_model.get_margeff(at="overall").summary().as_text()
    )

    # Extract key coefficients for summary CSV
    summary_rows = []
    models_and_predictors = [
        (raw_model, "raw", ["years_since_first_import", "log_cumulative_imports"]),
        (
            controlled_model,
            "controlled",
            ["years_since_first_import", "log_cumulative_imports"],
        ),
        (age_only_model, "age_only", ["years_since_first_import"]),
        (popularity_only_model, "popularity_only", ["log_cumulative_imports"]),
    ]
    for model, model_name, predictors in models_and_predictors:
        conf_int = model.conf_int()
        for predictor in predictors:
            summary_rows.append(
                {
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
    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_csv(RESULTS_DIR / "age-vs-popularity-summary-stats.csv")
    print(summary_df)

    # Coefficient plot
    all_model_names = ["raw", "controlled", "age_only", "popularity_only"]
    colors = {
        "raw": "#2c7bb6",
        "controlled": "#d7191c",
        "age_only": "#fdae61",
        "popularity_only": "#abdda4",
    }
    key_predictors = ["years_since_first_import", "log_cumulative_imports"]
    _fig, ax = plt.subplots(figsize=(8, 4))
    n_models = len(all_model_names)
    offset = 0.12

    for i, predictor in enumerate(key_predictors):
        for j, model_name in enumerate(all_model_names):
            rows = summary_df.filter(
                (pl.col("predictor") == predictor) & (pl.col("model_type") == model_name)
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
                color=colors[model_name],
                capsize=4,
                label=model_name if i == 0 else None,
            )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(key_predictors)))
    ax.set_yticklabels([p.replace("_", " ") for p in key_predictors])
    ax.set_xlabel("Logit coefficient (95% CI)")
    ax.set_title("Age vs Popularity: Logistic Regression Coefficients")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "age-vs-popularity-coefficient-plot.png",
        dpi=300,
        bbox_inches="tight",
    )


def _analysis_per_paper_mention_fraction(
    imports_and_mentions_long_df: pl.DataFrame,
    pair_metadata: pl.DataFrame,
) -> None:
    """Per-paper mention fraction analysis with fractional logit models."""
    imported_df = imports_and_mentions_long_df.filter(pl.col("is_imported"))

    # Combined (all ecosystems pooled per paper
    combined_paper_df = (
        imported_df.group_by("document_id")
        .agg(
            n_imported=pl.len(),
            n_mentioned=pl.col("is_mentioned").sum(),
            dominant_ecosystem=pl.col("ecosystem").mode().first(),
        )
        .with_columns(
            fraction_mentioned=(pl.col("n_mentioned") / pl.col("n_imported")),
        )
        .join(
            pair_metadata.select(
                "document_id",
                "document_publication_year",
                "document_author_count",
                "document_log_author_mean_citations",
                "document_field_name_pruned",
                "document_years_since_earliest",
            ),
            on="document_id",
        )
        .drop_nulls()
    )

    # Per-ecosystem paper fractions
    per_eco_paper_df = (
        imported_df.group_by("document_id", "ecosystem")
        .agg(
            n_imported=pl.len(),
            n_mentioned=pl.col("is_mentioned").sum(),
        )
        .with_columns(
            fraction_mentioned=(pl.col("n_mentioned") / pl.col("n_imported")),
        )
        .join(
            pair_metadata.select(
                "document_id",
                "document_publication_year",
                "document_author_count",
                "document_log_author_mean_citations",
                "document_field_name_pruned",
                "document_years_since_earliest",
            ),
            on="document_id",
        )
        .drop_nulls()
    )

    print()
    print("Analysis 2: Per-paper mention fraction")

    # Mean per-paper mention fraction by field (descriptive)
    by_field_fraction = (
        combined_paper_df.group_by("document_field_name_pruned")
        .agg(
            n_papers=pl.len(),
            mean_fraction_mentioned=pl.col("fraction_mentioned").mean(),
            median_fraction_mentioned=pl.col("fraction_mentioned").median(),
            std_fraction_mentioned=pl.col("fraction_mentioned").std(),
        )
        .rename({"document_field_name_pruned": "field"})
    )
    overall_fraction = combined_paper_df.select(
        pl.lit("overall").alias("field"),
        pl.len().alias("n_papers"),
        pl.col("fraction_mentioned").mean().alias("mean_fraction_mentioned"),
        pl.col("fraction_mentioned").median().alias("median_fraction_mentioned"),
        pl.col("fraction_mentioned").std().alias("std_fraction_mentioned"),
    )
    by_field_fraction = pl.concat([by_field_fraction, overall_fraction]).sort(
        "mean_fraction_mentioned", descending=True
    )
    by_field_fraction.write_csv(RESULTS_DIR / "per-paper-fraction-by-field.csv")
    print("Per-paper mention fraction by field:")
    print(by_field_fraction)

    # Combined model
    combined_pd = combined_paper_df.to_pandas()
    print(f"Combined model N: {len(combined_pd)}")

    combined_model = smf.glm(
        "fraction_mentioned ~ C(document_field_name_pruned) + document_publication_year"
        " + document_author_count + document_log_author_mean_citations"
        " + n_imported + C(dominant_ecosystem)",
        data=combined_pd,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC1")
    (RESULTS_DIR / "per-paper-fraction-glm-combined.txt").write_text(
        combined_model.summary().as_text()
    )
    (RESULTS_DIR / "per-paper-fraction-glm-combined-marginal-effects.txt").write_text(
        combined_model.get_margeff(at="overall").summary().as_text()
    )

    # Per-ecosystem models
    summary_rows = []
    non_field_predictors = [
        "document_publication_year",
        "document_author_count",
        "document_log_author_mean_citations",
        "n_imported",
    ]

    # Add combined model coefficients
    for predictor in non_field_predictors:
        conf_int = combined_model.conf_int()
        summary_rows.append(
            {
                "model": "combined",
                "n_obs": int(combined_model.nobs),
                "predictor": predictor,
                "coefficient": combined_model.params[predictor],
                "std_err": combined_model.bse[predictor],
                "p_value": combined_model.pvalues[predictor],
                "ci_lower": conf_int.loc[predictor, 0],
                "ci_upper": conf_int.loc[predictor, 1],
            }
        )

    for ecosystem in ["py", "r"]:
        eco_df = per_eco_paper_df.filter(pl.col("ecosystem") == ecosystem).to_pandas()
        print(f"  {ecosystem} model N: {len(eco_df)}")

        eco_model = smf.glm(
            "fraction_mentioned ~ C(document_field_name_pruned) + document_publication_year"
            " + document_author_count + document_log_author_mean_citations"
            " + n_imported",
            data=eco_df,
            family=sm.families.Binomial(),
        ).fit(cov_type="HC1")
        (RESULTS_DIR / f"per-paper-fraction-glm-{ecosystem}.txt").write_text(
            eco_model.summary().as_text()
        )
        (RESULTS_DIR / f"per-paper-fraction-glm-{ecosystem}-marginal-effects.txt").write_text(
            eco_model.get_margeff(at="overall").summary().as_text()
        )

        conf_int = eco_model.conf_int()
        for predictor in non_field_predictors:
            summary_rows.append(
                {
                    "model": ecosystem,
                    "n_obs": int(eco_model.nobs),
                    "predictor": predictor,
                    "coefficient": eco_model.params[predictor],
                    "std_err": eco_model.bse[predictor],
                    "p_value": eco_model.pvalues[predictor],
                    "ci_lower": conf_int.loc[predictor, 0],
                    "ci_upper": conf_int.loc[predictor, 1],
                }
            )

    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_csv(RESULTS_DIR / "per-paper-fraction-summary-stats.csv")
    print(summary_df)

    # Histogram of per-paper mention fractions by ecosystem
    _fig, ax = plt.subplots(figsize=(7, 5))
    eco_colors = {"py": "#2c7bb6", "r": "#d7191c"}
    for ecosystem in ["py", "r"]:
        eco_data = combined_paper_df.filter(
            pl.col("dominant_ecosystem") == ecosystem
        ).get_column("fraction_mentioned")
        eco_mean = float(eco_data.mean())  # type: ignore[arg-type]
        ax.hist(
            eco_data.to_list(),
            bins=20,
            alpha=0.5,
            label=f"{'Python' if ecosystem == 'py' else 'R'} (mean={eco_mean:.2f})",
            color=eco_colors[ecosystem],
        )
        ax.axvline(
            eco_mean,
            color=eco_colors[ecosystem],
            linestyle="--",
            linewidth=1.5,
        )
    ax.set_xlabel("Fraction of imports mentioned")
    ax.set_ylabel("Number of papers")
    ax.set_title("Per-paper mention fraction by dominant ecosystem")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "per-paper-fraction-histogram.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Coefficient plot (per-ecosystem models side by side)
    _fig, ax = plt.subplots(figsize=(7, 4))
    offset = 0.15

    for i, predictor in enumerate(non_field_predictors):
        for j, model_name in enumerate(["py", "r"]):
            row = summary_df.filter(
                (pl.col("predictor") == predictor) & (pl.col("model") == model_name)
            ).to_dicts()[0]
            y_pos = i + (j - 0.5) * offset
            ax.errorbar(
                row["coefficient"],
                y_pos,
                xerr=[
                    [row["coefficient"] - row["ci_lower"]],
                    [row["ci_upper"] - row["coefficient"]],
                ],
                fmt="o",
                color=eco_colors[model_name],
                capsize=4,
                label=f"{'Python' if model_name == 'py' else 'R'}" if i == 0 else None,
            )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(non_field_predictors)))
    ax.set_yticklabels([p.replace("_", " ") for p in non_field_predictors])
    ax.set_xlabel("GLM coefficient (95% CI)")
    ax.set_title(
        "Per-paper mention fraction: Coefficients by ecosystem\n(field fixed effects included but omitted from plot)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "per-paper-fraction-coefficient-plot.png",
        dpi=300,
        bbox_inches="tight",
    )


def _analysis_ecosystem_differences(
    imports_and_mentions_long_df: pl.DataFrame,
) -> None:
    """Ecosystem-stratified attribution analysis."""
    imported_df = imports_and_mentions_long_df.filter(
        pl.col("is_imported") & pl.col("ecosystem").is_in(["py", "r"])
    )
    eco_colors = {"py": "#2c7bb6", "r": "#d7191c"}
    eco_labels = {"py": "Python", "r": "R"}
    min_library_imports = 20

    # 3a: p(mention|import) over time by ecosystem
    library_import_counts = (
        imported_df.group_by("library_name_normalized")
        .agg(total_imports=pl.len())
        .filter(pl.col("total_imports") >= min_library_imports)
    )
    libraries_to_investigate = library_import_counts.get_column(
        "library_name_normalized"
    ).to_list()

    filtered_df = imported_df.filter(
        pl.col("library_name_normalized").is_in(libraries_to_investigate)
    )

    # Per library-year-ecosystem aggregation
    per_lib_year_eco = (
        filtered_df.group_by("library_name_normalized", "publication_year", "ecosystem")
        .agg(
            n_imported=pl.len(),
            n_mentioned=pl.col("is_mentioned").sum(),
        )
        .with_columns(
            p_mention=(pl.col("n_mentioned") / pl.col("n_imported")),
        )
    )

    # Aggregate across libraries per ecosystem-year
    _fig, ax = plt.subplots(figsize=(7, 5))
    for ecosystem in ["py", "r"]:
        agg = (
            per_lib_year_eco.filter(pl.col("ecosystem") == ecosystem)
            .group_by("publication_year")
            .agg(
                mean_p=pl.mean("p_mention"),
                std_p=pl.std("p_mention"),
                n_libraries=pl.len(),
            )
            .with_columns(se_p=(pl.col("std_p") / pl.col("n_libraries").sqrt()))
            .filter(pl.col("n_libraries") >= 10)
            .sort("publication_year")
            .to_pandas()
        )
        ax.plot(
            agg["publication_year"],
            agg["mean_p"],
            "o-",
            color=eco_colors[ecosystem],
            linewidth=2,
            label=eco_labels[ecosystem],
        )
        ax.fill_between(
            agg["publication_year"],
            agg["mean_p"] - 1.96 * agg["se_p"],
            agg["mean_p"] + 1.96 * agg["se_p"],
            alpha=0.15,
            color=eco_colors[ecosystem],
        )
    ax.set_xlabel("Publication year")
    ax.set_ylabel("p(mention | import)")
    ax.set_title("p(mention | import) over time by ecosystem\n(equal weight per library)")
    ax.legend()
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "p-mention-given-import-over-time-by-ecosystem.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 3b: p(mention|import) vs years since first import by ecosystem
    first_import_year = filtered_df.group_by("library_name_normalized").agg(
        first_import_year=pl.col("publication_year").min()
    )
    filtered_with_age = filtered_df.join(
        first_import_year, on="library_name_normalized"
    ).with_columns(
        years_since_first_import=pl.col("publication_year") - pl.col("first_import_year")
    )

    per_lib_age_eco = (
        filtered_with_age.group_by(
            "library_name_normalized", "years_since_first_import", "ecosystem"
        )
        .agg(
            n_imported=pl.len(),
            n_mentioned=pl.col("is_mentioned").sum(),
        )
        .with_columns(
            p_mention=(pl.col("n_mentioned") / pl.col("n_imported")),
        )
    )

    _fig, ax = plt.subplots(figsize=(7, 5))
    for ecosystem in ["py", "r"]:
        agg = (
            per_lib_age_eco.filter(pl.col("ecosystem") == ecosystem)
            .group_by("years_since_first_import")
            .agg(
                mean_p=pl.mean("p_mention"),
                std_p=pl.std("p_mention"),
                n_libraries=pl.len(),
            )
            .with_columns(se_p=(pl.col("std_p") / pl.col("n_libraries").sqrt()))
            .filter(pl.col("n_libraries") >= 10)
            .sort("years_since_first_import")
            .to_pandas()
        )
        ax.plot(
            agg["years_since_first_import"],
            agg["mean_p"],
            "o-",
            color=eco_colors[ecosystem],
            linewidth=2,
            label=eco_labels[ecosystem],
        )
        ax.fill_between(
            agg["years_since_first_import"],
            agg["mean_p"] - 1.96 * agg["se_p"],
            agg["mean_p"] + 1.96 * agg["se_p"],
            alpha=0.15,
            color=eco_colors[ecosystem],
        )
    ax.set_xlabel("Years since first import")
    ax.set_ylabel("p(mention | import)")
    ax.set_title("p(mention | import) vs library age by ecosystem\n(equal weight per library)")
    ax.legend()
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "p-mention-given-import-vs-age-by-ecosystem.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 3c: p(mention|import) vs popularity (at publication year) by ecosystem
    cumulative_imports_by_year = _compute_cumulative_imports_by_library_year(
        imports_and_mentions_long_df
    )
    per_lib_year_popularity = (
        filtered_df.group_by("library_name_normalized", "publication_year")
        .agg(
            n_imports_in_year=pl.len(),
            n_mentions_in_year=pl.col("is_mentioned").sum(),
            ecosystem=pl.col("ecosystem").first(),
        )
        .join(
            cumulative_imports_by_year.select(
                "library_name_normalized", "publication_year", "cumulative_imports_at_year"
            ),
            on=["library_name_normalized", "publication_year"],
        )
        .with_columns(
            p_mention=(pl.col("n_mentions_in_year") / pl.col("n_imports_in_year")),
        )
        .filter(pl.col("n_imports_in_year") >= 5)
    )

    _fig, ax = plt.subplots(figsize=(7, 5))
    for ecosystem in ["py", "r"]:
        eco_data = per_lib_year_popularity.filter(pl.col("ecosystem") == ecosystem).to_pandas()
        ax.scatter(
            eco_data["cumulative_imports_at_year"],
            eco_data["p_mention"],
            alpha=0.4,
            color=eco_colors[ecosystem],
            label=eco_labels[ecosystem],
            s=14,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative importing projects through publication year (log scale)")
    ax.set_ylabel("p(mention | import)")
    ax.set_title(
        "p(mention | import) vs cumulative importing projects by ecosystem\n"
        "(one point per library-year)"
    )
    ax.legend()
    ax.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "p-mention-given-import-vs-popularity-by-ecosystem.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 3d: Overall + by-field proportion test
    # Overall chi-squared test
    contingency = (
        imported_df.group_by("ecosystem")
        .agg(
            n_imported=pl.len(),
            n_mentioned=pl.col("is_mentioned").sum(),
        )
        .with_columns(
            n_not_mentioned=(pl.col("n_imported") - pl.col("n_mentioned")),
            p_mention=(pl.col("n_mentioned") / pl.col("n_imported")),
        )
        .sort("ecosystem")
    )

    # Build 2x2 table for chi-squared
    table_2x2 = contingency.select("n_mentioned", "n_not_mentioned").to_numpy()
    chi2, p_value, _dof, _expected = stats.chi2_contingency(table_2x2)

    overall_result = contingency.with_columns(
        pl.lit("overall").alias("field"),
        pl.lit(chi2).alias("chi2_statistic"),
        pl.lit(p_value).alias("chi2_p_value"),
    )
    overall_result.write_csv(RESULTS_DIR / "ecosystem-proportion-test.csv")
    print("\nAnalysis 3d: Ecosystem proportion test (overall)")
    print(f"  Chi2={chi2:.2f}, p={p_value:.4g}")
    print(contingency)

    # By-field proportion tests
    by_field_rows = []
    for field in imported_df.get_column("pruned_field").unique().sort():
        field_df = imported_df.filter(pl.col("pruned_field") == field)
        field_contingency = (
            field_df.group_by("ecosystem")
            .agg(
                n_imported=pl.len(),
                n_mentioned=pl.col("is_mentioned").sum(),
            )
            .with_columns(
                n_not_mentioned=(pl.col("n_imported") - pl.col("n_mentioned")),
                p_mention=(pl.col("n_mentioned") / pl.col("n_imported")),
            )
            .sort("ecosystem")
        )

        # Only run chi-squared if both ecosystems present and have data
        if len(field_contingency) == 2:
            field_table = field_contingency.select("n_mentioned", "n_not_mentioned").to_numpy()
            field_chi2, field_p, _dof, _expected = stats.chi2_contingency(field_table)
        else:
            field_chi2, field_p = float("nan"), float("nan")

        for row in field_contingency.to_dicts():
            by_field_rows.append(
                {
                    "field": field,
                    "ecosystem": row["ecosystem"],
                    "n_imported": row["n_imported"],
                    "n_mentioned": row["n_mentioned"],
                    "p_mention": row["p_mention"],
                    "chi2_statistic": field_chi2,
                    "chi2_p_value": field_p,
                }
            )

    by_field_df = pl.DataFrame(by_field_rows)
    by_field_df.write_csv(RESULTS_DIR / "ecosystem-proportion-test-by-field.csv")
    print("Analysis 3d: Ecosystem proportion test (by field)")
    print(by_field_df)


@app.command()
def main(
    rare_import_threshold: int = 3,
    top_n_fields: int = 5,
    sample: bool = False,
    sample_size: int = 5000,
    matching_score_threshold: float = 75.0,
) -> None:
    load_dotenv()
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Create data dir
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load our dataset
    pair_metadata = _load_our_dataset(
        top_n_fields=top_n_fields,
    )

    # Take a sample to speed up development
    if sample:
        pair_metadata = pair_metadata.sample(sample_size, seed=42)

    # Get repository imports, repository dependencies, and document software mentions
    repository_imports = load_table("repository_import")
    repository_dependencies = load_table("repository_dependency")
    document_software_mentions = load_table("document_software_mention")

    # Remove extremely rare imports and dependencies if specified (these are likely noise and can skew results)
    repository_imports = _remove_extremely_rare_software_usage(
        repository_imports,
        usage_type="imports",
        software_col="software_name_normalized",
        rare_usage_threshold=rare_import_threshold,
        compute_ecosystem_prefix=True,
        use_ecosystem_from_data=False,
        exclude_generic_mentions=False,
    )
    repository_dependencies = _remove_extremely_rare_software_usage(
        repository_dependencies,
        usage_type="dependencies",
        software_col="software_name_normalized",
        rare_usage_threshold=rare_import_threshold,
        compute_ecosystem_prefix=False,
        use_ecosystem_from_data=True,
        exclude_generic_mentions=False,
    )
    document_software_mentions = _remove_extremely_rare_software_usage(
        document_software_mentions,
        usage_type="mentions",
        software_col="software_name_normalized",
        rare_usage_threshold=rare_import_threshold,
        compute_ecosystem_prefix=False,
        use_ecosystem_from_data=False,
        exclude_generic_mentions=True,
    )

    # Add "has_imports", "has_dependencies", and "has_software_mentions" columns to the merged table
    software_usage_cols_result = _add_has_imports_dependencies_mentions_cols(
        pair_metadata,
        repository_imports,
        repository_dependencies,
        document_software_mentions,
    )

    # Unpack the result
    pair_metadata = software_usage_cols_result.pair_metadata
    repository_imports = software_usage_cols_result.repository_imports
    repository_dependencies = software_usage_cols_result.repository_dependencies
    document_software_mentions = software_usage_cols_result.document_software_mentions

    # Remove any pairs with extreme numbers of software used
    non_extreme_pairs_result = _remove_pairs_with_extreme_software_usage(
        pair_metadata,
        repository_imports,
        repository_dependencies,
        document_software_mentions,
    )

    # Unpack the result
    pair_metadata = non_extreme_pairs_result.pair_metadata
    repository_imports = non_extreme_pairs_result.repository_imports
    repository_dependencies = non_extreme_pairs_result.repository_dependencies
    document_software_mentions = non_extreme_pairs_result.document_software_mentions

    # Re-apply rare threshold on the analysis subset.
    # The earlier filtering (lines 582-608) was on the full global tables; after inner-joining
    # to pair_metadata and removing extreme pairs, some libraries may now appear <3 times.
    repository_imports = repository_imports.filter(
        pl.col("ecosystem_software_name_normalized").is_in(
            repository_imports.group_by("ecosystem_software_name_normalized")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") >= rare_import_threshold)
            .get_column("ecosystem_software_name_normalized")
            .to_list()
        )
    )
    document_software_mentions = document_software_mentions.filter(
        pl.col("software_name_normalized").is_in(
            document_software_mentions.group_by("software_name_normalized")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") >= rare_import_threshold)
            .get_column("software_name_normalized")
            .to_list()
        )
    )

    # Get the top 50 most commonly imported libraries per ecosystem
    for ecosystem in repository_imports.get_column("ecosystem").unique():
        top_imports = (
            repository_imports.filter(pl.col("ecosystem") == ecosystem)
            .group_by("software_name_normalized")
            .agg(pl.len().alias("import_count"))
            .sort("import_count", descending=True)
        )

        # Get count of total repositories
        total_repositories = (
            repository_imports.filter(pl.col("ecosystem") == ecosystem)
            .get_column("repository_id")
            .n_unique()
        )

        # Add proportion of repositories with imports of this library, and then take top 50
        top_imports = top_imports.with_columns(
            # Add proportion of repositories with imports of this library
            (pl.col("import_count") / total_repositories).alias("import_proportion")
        ).head(50)

        top_imports.write_csv(RESULTS_DIR / f"top-fifty-imports-overall-{ecosystem}.csv")

    # Get counts and proportions for each type of software usage
    software_usage_counts = _get_counts_and_proportions_of_each_software_usage_type(
        pair_metadata,
    )
    software_usage_counts.write_csv(RESULTS_DIR / "software-usage-counts.csv")
    print(software_usage_counts)

    # Match imports and mentions for each pair
    # Create long-format dataframe
    # with one row per imported library, with columns for "is_mentioned", "publication_year", and "field"
    imports_and_mentions_long_df = _match_imports_and_mentions_and_get_long_frame(
        pair_metadata,
        repository_imports,
        document_software_mentions,
        matching_score_threshold=matching_score_threshold,
    )

    # Save long-frame parquet for downstream bridge analyses
    imports_and_mentions_long_df.write_parquet(
        RESULTS_DIR / "imports-and-mentions-long.parquet"
    )

    # Print descriptive stats and tables about the relationship between imports and mentions
    _get_descriptive_stats_and_tables(imports_and_mentions_long_df)

    # Compute probability of mention given import over time
    _compute_probability_of_mention_given_import_over_time(imports_and_mentions_long_df)

    # Compute probability of mention given import vs popularity
    _compute_probability_of_mention_given_popularity(imports_and_mentions_long_df)

    # Compute probability of mention given time since first import
    _compute_probability_of_mention_since_year_of_first_import(imports_and_mentions_long_df)

    # Plot individual spotlight-library trajectories on a years-since-first-import x-axis
    _plot_individual_library_trajectories_since_year_of_first_import(
        imports_and_mentions_long_df
    )

    # Analysis 1: Age vs popularity confounding (logistic regression)
    _analysis_age_vs_popularity_logistic_regression(imports_and_mentions_long_df, pair_metadata)

    # Analysis 2: Per-paper mention fraction (fractional logit)
    _analysis_per_paper_mention_fraction(imports_and_mentions_long_df, pair_metadata)

    # Analysis 3: Ecosystem differences in attribution
    _analysis_ecosystem_differences(imports_and_mentions_long_df)


###############################################################################

if __name__ == "__main__":
    app()
