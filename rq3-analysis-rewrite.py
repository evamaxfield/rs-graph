import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from rs_graph.utils.software_alignment import align_software_names

###############################################################################

load_dotenv()
os.environ["HF_DATASETS_OFFLINE"] = "1"

###############################################################################


# Helper to load a table as a polars DataFrame (zero-copy via Arrow)
def load_table(table: str) -> pl.DataFrame:
    ds = load_dataset("evamxb/rs-graph-v2", table, split="train")
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


# Load all article info
documents = load_table("document")

# Load and join article-repository links with repository info
article_repo_links = load_table("document_repository_link")
repositories = load_table("repository")

merged = (
    article_repo_links.select(
        pl.col("id").alias("document_repository_link_id"),
        pl.col("document_id"),
        pl.col("repository_id"),
        pl.col("predictive_model_confidence"),
    )
    .join(
        documents.select(*[pl.col(col).alias(f"document_{col}") for col in documents.columns]),
        on="document_id",
    )
    .join(
        repositories.select(
            *[pl.col(col).alias(f"repository_{col}") for col in repositories.columns]
        ),
        on="repository_id",
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
merged = merged.filter(pl.col("document_publication_year") >= 2008)
print(f"Count of all document-repository pairs: {len(merged)}")

# Drop to unique documents, then drop to unique repositories
merged = merged.unique(
    subset="document_id",
    keep="none",
).unique(
    subset="repository_id",
    keep="none",
)
print(f"Count of unique one-to-one document-repository pairs: {len(merged)}")

# Drop to predictive model confidence of 0.995
merged = merged.filter(
    (pl.col("predictive_model_confidence") >= 0.995)
    | (pl.col("predictive_model_confidence").is_null())
)
print(f"Count of high-confidence unique one-to-one document-repository pairs: {len(merged)}")

# Get repository imports, repository dependencies, and document software mentions
repository_imports = load_table("repository_import")
repository_dependencies = load_table("repository_dependency")
document_software_mentions = load_table("document_software_mention")

# Get the subset of each that have a repository_id or document_id in the merged set
repository_imports = repository_imports.join(
    merged.select(
        pl.col("document_repository_link_id"),
        pl.col("document_id"),
        pl.col("repository_id"),
    ),
    on="repository_id",
    how="inner",
)
print(
    f"Count of repositories with imports linked to merged repositories (subset of high-conf unique one-to-one pairs): "
    f"{len(repository_imports.unique(subset='repository_id'))}"
)
repository_dependencies = repository_dependencies.join(
    merged.select(
        pl.col("document_repository_link_id"),
        pl.col("document_id"),
        pl.col("repository_id"),
    ),
    on="repository_id",
    how="inner",
)
print(
    f"Count of repositories with dependencies linked to merged repositories (subset of high-conf unique one-to-one pairs): "
    f"{len(repository_dependencies.unique(subset='repository_id'))}"
)
document_software_mentions = document_software_mentions.join(
    merged.select(
        pl.col("document_repository_link_id"),
        pl.col("document_id"),
        pl.col("repository_id"),
    ),
    on="document_id",
    how="inner",
)
print(
    f"Count of documents with software mentions linked to merged documents (subset of high-conf unique one-to-one pairs): "
    f"{len(document_software_mentions.unique(subset='document_id'))}"
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
merged = (
    merged.join(
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

# Calculate counts and percentages of document-repository pairs that have imports, dependencies, and software mentions
# and the combinations thereof (e.g. have imports but not dependencies, etc.)
total_pairs = len(merged)
has_imports = len(merged.filter(pl.col("has_imports")))
has_dependencies = len(merged.filter(pl.col("has_dependencies")))
has_software_mentions = len(merged.filter(pl.col("has_software_mentions")))
has_imports_only = len(
    merged.filter(
        pl.col("has_imports") & ~pl.col("has_dependencies") & ~pl.col("has_software_mentions")
    )
)
has_dependencies_only = len(
    merged.filter(
        ~pl.col("has_imports") & pl.col("has_dependencies") & ~pl.col("has_software_mentions")
    )
)
has_software_mentions_only = len(
    merged.filter(
        ~pl.col("has_imports") & ~pl.col("has_dependencies") & pl.col("has_software_mentions")
    )
)
has_imports_and_dependencies = len(
    merged.filter(
        pl.col("has_imports") & pl.col("has_dependencies") & ~pl.col("has_software_mentions")
    )
)
has_imports_and_software_mentions = len(
    merged.filter(
        pl.col("has_imports") & ~pl.col("has_dependencies") & pl.col("has_software_mentions")
    )
)
has_dependencies_and_software_mentions = len(
    merged.filter(
        ~pl.col("has_imports") & pl.col("has_dependencies") & pl.col("has_software_mentions")
    )
)
complete_cases = len(
    merged.filter(
        pl.col("has_imports") & pl.col("has_dependencies") & pl.col("has_software_mentions")
    )
)

print()
print(f"Count of unique one-to-one document-repository pairs: {total_pairs}")
print(f"Pairs with imports: {has_imports} ({has_imports / total_pairs:.2%})")
print(f"Pairs with dependencies: {has_dependencies} ({has_dependencies / total_pairs:.2%})")
print(
    f"Pairs with software mentions: {has_software_mentions} ({has_software_mentions / total_pairs:.2%})"
)
print(f"Pairs with imports only: {has_imports_only} ({has_imports_only / total_pairs:.2%})")
print(
    f"Pairs with dependencies only: {has_dependencies_only} ({has_dependencies_only / total_pairs:.2%})"
)
print(
    f"Pairs with software mentions only: {has_software_mentions_only} ({has_software_mentions_only / total_pairs:.2%})"
)
print(
    f"Pairs with imports and dependencies: {has_imports_and_dependencies} ({has_imports_and_dependencies / total_pairs:.2%})"
)
print(
    f"Pairs with imports and software mentions: {has_imports_and_software_mentions} ({has_imports_and_software_mentions / total_pairs:.2%})"
)
print(
    f"Pairs with dependencies and software mentions: {has_dependencies_and_software_mentions} ({has_dependencies_and_software_mentions / total_pairs:.2%})"
)
print(f"Pairs with complete cases: {complete_cases} ({complete_cases / total_pairs:.2%})")

print()
print("-" * 80)
print()

# Probability of mention given import and year over time
pairs_with_imports_and_mentions = merged.filter(
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
    )

    # Add matched pairs to the long-format table
    for matched_import_and_mention in matched_imports_and_mentions:
        imports_and_mentions_matched_rows.append(
            {
                "document_id": this_pair_document_id,
                "publication_year": this_pair_publication_year,
                "library_name_normalized": matched_import_and_mention.normalized_item_one,
                "is_imported": True,
                "is_mentioned": True,
            }
        )

    # Add unmatched imports with is_mentioned=False
    unmatched_imports = (
        set(normalized_imported_software_names)
        - set([match.normalized_item_one for match in matched_imports_and_mentions])
        - set([match.normalized_item_two for match in matched_imports_and_mentions])
    )
    for unmatched_import in unmatched_imports:
        imports_and_mentions_matched_rows.append(
            {
                "document_id": this_pair_document_id,
                "publication_year": this_pair_publication_year,
                "library_name_normalized": unmatched_import,
                "is_imported": True,
                "is_mentioned": False,
            }
        )

# Convert the long-format table to a DataFrame
imports_and_mentions_long_df = pl.DataFrame(imports_and_mentions_matched_rows)
print(imports_and_mentions_long_df)

# Filter to libraries that were imported at least 100 times across all years
library_import_counts = (
    imports_and_mentions_long_df.group_by("library_name_normalized")
    .agg(total_imports=pl.sum("is_imported"))
    .filter(pl.col("total_imports") >= 100)
)
print(library_import_counts.sort(by="total_imports", descending=True))
libraries_to_investigate = library_import_counts.get_column("library_name_normalized").to_list()

# For each of these libraries, group by publication_year and calculate p(mention | import)
# Add this calculation to the imports_and_mentions_long_df DataFrame as a new column "p_mention_given_import"
imports_and_mentions_long_df = imports_and_mentions_long_df.filter(
    pl.col("library_name_normalized").is_in(libraries_to_investigate)
)
per_library_year = (
    imports_and_mentions_long_df.group_by("library_name_normalized", "publication_year")
    .agg(
        n_imported=pl.sum("is_imported"),
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
        n_libraries=pl.count(),
    )
    .with_columns((pl.col("std_p") / pl.col("n_libraries").sqrt()).alias("se_p"))
    .sort("publication_year")
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ── Left panel: Aggregate curve (equal weight per library) ──
agg = (
    aggregate_by_year.filter(pl.col("publication_year").is_between(2014, 2023))
    .sort("publication_year")
    .to_pandas()
)

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
spotlight_libraries = [
    "numpy",
    "pandas",
    "polars",
    "tensorflow",
    "torch",
    "ggplot2",
    "dplyr",
    "mass",
    "survival",
    "lme4",
]

colors = plt.cm.tab10(np.linspace(0, 1, len(spotlight_libraries)))  # type: ignore

for lib, color in zip(spotlight_libraries, colors):
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
            "o-",
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
plt.savefig("p_mention_given_import_two_panel.png", dpi=300, bbox_inches="tight")
