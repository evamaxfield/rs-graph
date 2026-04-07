import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
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

    library_popularity = (
        imports_and_mentions_long_df.filter(
            pl.col("library_name_normalized").is_in(libraries_to_investigate)
            & pl.col("is_imported")
        )
        .group_by("library_name_normalized")
        .agg(
            total_imports=pl.len(),
            total_mentions=pl.sum("is_mentioned"),
            ecosystem=pl.col("ecosystem").first(),
        )
        .with_columns(
            p_mention_given_import=(pl.col("total_mentions") / pl.col("total_imports")),
        )
    )

    # Plot p(mention | import) vs total imports
    plt.figure(figsize=(6, 5))
    plt.scatter(
        library_popularity.get_column("total_imports"),
        library_popularity.get_column("p_mention_given_import"),
        alpha=0.7,
    )

    # Label the following points in the scatter
    py_libraries_to_label = ["numpy", "pandas", "tensorflow", "torch"]
    r_libraries_to_label = ["data.table", "dplyr", "lme4"]

    # Get the rows for these libraries to label
    selected_libraries = library_popularity.filter(
        pl.col("library_name_normalized").is_in(py_libraries_to_label + r_libraries_to_label)
    ).to_pandas()

    # Iter over select rows and annotate the points with the library name
    for _, row in selected_libraries.iterrows():
        row_details = row.to_dict()
        plt.annotate(
            row_details["library_name_normalized"],
            (row_details["total_imports"], row_details["p_mention_given_import"]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=7,
            ha="center",
        )

    plt.xscale("log")
    plt.xlabel("Total importing projects (log scale)")
    plt.ylabel("p(mention | import)")
    plt.title("p(mention | import) vs total importing projects")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(
        RESULTS_DIR / "p-mention-given-import-vs-popularity.png", dpi=300, bbox_inches="tight"
    )

    # Also plot this distribution in small multiples of pruned field
    # One ax per field, with a scatter plot of p(mention | import) vs total imports, faceted by pruned_field
    library_popularity_by_field = (
        imports_and_mentions_long_df.filter(
            pl.col("library_name_normalized").is_in(libraries_to_investigate)
            & pl.col("is_imported")
        )
        .group_by("library_name_normalized", "pruned_field")
        .agg(
            total_imports=pl.len(),
            total_mentions=pl.sum("is_mentioned"),
        )
        .with_columns(
            p_mention_given_import=(pl.col("total_mentions") / pl.col("total_imports")),
        )
    )
    fields = library_popularity_by_field.get_column("pruned_field").unique().sort()
    n_fields = len(fields)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    _fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), sharex=True, sharey=True
    )
    for field, ax in zip(fields, axes.flatten(), strict=True):
        field_data = library_popularity_by_field.filter(
            pl.col("pruned_field") == field
        ).to_pandas()
        ax.scatter(
            field_data["total_imports"],
            field_data["p_mention_given_import"],
            alpha=0.7,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Total importing projects (log scale)")
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

    # Print descriptive stats and tables about the relationship between imports and mentions
    _get_descriptive_stats_and_tables(imports_and_mentions_long_df)

    # Compute probability of mention given import over time
    _compute_probability_of_mention_given_import_over_time(imports_and_mentions_long_df)

    # Compute probability of mention given import vs popularity
    _compute_probability_of_mention_given_popularity(imports_and_mentions_long_df)

    # Compute probability of mention given time since first import
    _compute_probability_of_mention_since_year_of_first_import(imports_and_mentions_long_df)


###############################################################################

if __name__ == "__main__":
    app()
