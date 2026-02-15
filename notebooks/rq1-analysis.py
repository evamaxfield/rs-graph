import random
from pathlib import Path

import colormaps as cmaps
import connectorx  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rustworkx as rx
import seaborn as sns
from tqdm import tqdm

from rs_graph.db import constants as db_constants

# Setup plotting style
PALETTE = cmaps.bold._colors.tolist()
sns.set_palette(PALETTE)

# Plotting/Data Selection constants
DEFAULT_TOP_N = 9

# Output directory for saved plots
RESULTS_DIR = Path(__file__).parent / "rq1-results"
RESULTS_DIR.mkdir(exist_ok=True)

#######################################################################################


def _read_table(table: str) -> pl.DataFrame:
    """Read a table from the v2 database."""
    return pl.read_database_uri(
        f"SELECT * FROM {table}",
        f"sqlite:///{db_constants.V2_DATABASE_PATHS.dev}",
    )


def load_pairs(sample_size: int | None = None) -> pl.DataFrame:
    """Load document-repository pairs with all relevant metadata."""
    # Read all the tables we need
    dataset_sources = _read_table("dataset_source")
    docs = _read_table("document")
    repos = _read_table("repository")
    pairs = _read_table("document_repository_link")
    doc_topics = _read_table("document_topic")
    topics = _read_table("topic")

    # Drop to unique doc and unique repo in pairs
    pairs = pairs.unique(
        subset="document_id",
        keep="none",
    ).unique(
        subset="repository_id",
        keep="none",
    )

    # Apply sampling if requested
    if sample_size is not None:
        sampled_doc_ids = (
            pairs.select("document_id")
            .unique()
            .sample(n=min(sample_size, pairs.height), seed=42)
        )
        pairs = pairs.filter(pl.col("document_id").is_in(sampled_doc_ids["document_id"]))

    # Load contributor/institution data
    doc_contribs = _read_table("document_contributor")
    doc_contrib_institutions = _read_table("document_contributor_institution")
    institutions = _read_table("institution")

    # Process authorship team countries
    doc_author_countries = (
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
        )
        .group_by("document_id")
        .agg(
            pl.len().alias("document_n_authors"),
            pl.when(pl.col("country_code").n_unique() == 1)
            .then(pl.col("country_code").first())
            .otherwise(pl.lit("Multiple")),
        )
        .with_columns(
            pl.when(pl.col("country_code").is_null())
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("country_code"))
            .alias("country_code")
        )
    )

    # Get repo contributors count
    repo_contribs = _read_table("repository_contributor")
    repo_contribs = repo_contribs.group_by("repository_id").len("repository_n_contributors")

    # Get repo file count
    repo_files = _read_table("repository_file")
    repo_file_counts = (
        repo_files.filter(pl.col("tree_type") == "blob")
        .group_by("repository_id")
        .agg(
            pl.len().alias("repository_n_files"),
            pl.col("bytes_of_code").sum().alias("repository_total_file_bytes"),
        )
    )

    # Get repo language breakdown
    repo_languages = _read_table("repository_language")
    repo_language_counts = repo_languages.group_by("repository_id").agg(
        pl.len().alias("repository_n_languages"),
        pl.col("bytes_of_code").sum().alias("repository_total_language_bytes"),
    )

    # Join all tables
    result = (
        pairs.select(
            "document_id",
            "repository_id",
            "dataset_source_id",
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
        .join(repo_language_counts, on="repository_id", how="left")
        .with_columns(
            pl.col("document_publication_date").dt.year().alias("document_publication_year"),
            (
                (
                    pl.col("repository_last_pushed_datetime")
                    - pl.col("repository_creation_datetime")
                ).dt.total_days()
            ).alias("repository_commit_duration_days"),
        )
    )

    # Calculate field-year-type normalized stargazer count (FWSI)
    # First, determine the expected stars for each field-year-type combination
    field_year_type_expected_stars = result.group_by(
        [
            "document_field_name",
            "document_publication_year",
            "document_type",
        ]
    ).agg(
        pl.col("repository_stargazers_count").mean().alias("expected_stars"),
    )

    # Join back to result
    result = result.join(
        field_year_type_expected_stars,
        on=["document_field_name", "document_publication_year", "document_type"],
        how="left",
    )

    # Calculate FWSI (Field-Weighted Star Impact)
    result = result.with_columns(
        pl.when(pl.col("expected_stars").is_not_null())
        .then(pl.col("repository_stargazers_count") / pl.col("expected_stars"))
        .otherwise(pl.lit(None))
        .alias("repository_fwsi")
    ).drop("expected_stars")

    return result


pairs = load_pairs()
pairs


# ## Descriptive Stats

field_count_plot_data = pairs.filter(
    pl.col("document_field_name").is_not_null(),
    pl.col("document_field_name") != "",
)

print(field_count_plot_data["document_field_name"].value_counts(sort=True))

# Countplot of field
sns.countplot(
    data=field_count_plot_data,
    y="document_field_name",
    order=field_count_plot_data["document_field_name"].value_counts(sort=True)[
        "document_field_name"
    ],
)
plt.savefig(RESULTS_DIR / "field_countplot.png", bbox_inches="tight", dpi=300)

# Get mean, std, 25th, 50th, 75th percentiles for selected fields
pairs[
    [
        "document_cited_by_count",
        "document_fwci",
        "document_n_authors",
        "repository_stargazers_count",
        "repository_fwsi",
        "repository_commits_count",
        "repository_n_contributors",
        "repository_n_files",
        "repository_n_languages",
        "repository_size_kb",
        "repository_commit_duration_days",
    ]
].describe(percentiles=[0.25, 0.5, 0.75]).filter(
    pl.col("statistic").is_in(["mean", "std", "25%", "50%", "75%"]),
).transpose(include_header=True, header_name="metric", column_names="statistic")

# Plot features by top N fields + Other
top_nine_fields = (
    pairs.filter(
        pl.col("document_field_name").is_not_null(),
    )["document_field_name"]
    .value_counts(sort=True)
    .head(DEFAULT_TOP_N)["document_field_name"]
    .to_list()
)
pairs = pairs.with_columns(
    pl.when(pl.col("document_field_name").is_in(top_nine_fields))
    .then(pl.col("document_field_name"))
    .otherwise(pl.lit("Other"))
    .alias("document_field_name_top_nine_plus_other")
)

# Select features then unpivot for plotting
features_to_plot = [
    "document_fwci",
    "repository_fwsi",
    "document_cited_by_count",
    "repository_stargazers_count",
    "document_n_authors",
    "repository_n_contributors",
    "repository_n_files",
    "repository_commits_count",
]
features_melted = pairs.select(
    [
        "document_id",
        "repository_id",
        "document_field_name_top_nine_plus_other",
        *features_to_plot,
    ]
).unpivot(
    on=features_to_plot,
    variable_name="feature",
    value_name="value",
    index=["document_id", "repository_id", "document_field_name_top_nine_plus_other"],
)

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

# Add a feature name sort value for consistent ordering
features_melted = features_melted.with_columns(
    pl.col("feature")
    .map_elements(
        lambda x: list(FEATURE_NAME_TO_VIZ_NAME_LUT.keys()).index(x),
        return_dtype=pl.Int32,
    )
    .alias("feature_sort_order")
)
features_melted = features_melted.sort("feature_sort_order").drop("feature_sort_order")

# Get consistent hue order across plots
field_hue_order = [*top_nine_fields, "Other"]

# Setup figure and axes
fig, axes = plt.subplots(
    nrows=4,
    ncols=2,
    figsize=(20, 10),
    constrained_layout=True,
)
for ax, ((feature_name,), group_df) in zip(
    axes.flat, features_melted.group_by("feature", maintain_order=True), strict=True
):
    # Determine order by median value
    y_order = (
        group_df.group_by("document_field_name_top_nine_plus_other")
        .agg(
            pl.col("value").median().alias("median_value"),
            pl.col("value").mean().alias("mean_value"),
        )
        .sort(
            ["median_value", "mean_value", "document_field_name_top_nine_plus_other"],
            descending=True,
        )["document_field_name_top_nine_plus_other"]
        .to_list()
    )

    sns.boxplot(
        data=group_df,
        y="document_field_name_top_nine_plus_other",
        x="value",
        hue="document_field_name_top_nine_plus_other",
        hue_order=field_hue_order,
        order=y_order,
        ax=ax,
        showfliers=False,
    )

    # Remove axis labels
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Make y-axis tick labels smaller
    ax.tick_params(axis="y", labelsize=10)

    # Set title
    ax.set_title(FEATURE_NAME_TO_VIZ_NAME_LUT.get(feature_name, feature_name), fontsize=16)

# Add more whitespace between subplots vertically
fig.tight_layout(h_pad=3.0)
fig.savefig(RESULTS_DIR / "features_by_field_boxplots.png", bbox_inches="tight", dpi=300)

# Plot count of pairs over time
ax = sns.countplot(
    data=pairs.filter(
        pl.col("document_publication_year") > 2010,
    ),
    x="document_publication_year",
)

# Rotate x-axis labels for readability
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.figure.savefig(RESULTS_DIR / "pairs_over_time.png", bbox_inches="tight", dpi=300)

# We want to plot two features normally: field name, primary language
# Then we want to also plot the top 9 fields + other over time
# And, we want to plot the top 9 primary languages + other over time

# Easiest method for doing this is probably fig axes and then add each plot individually
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(20, 10),
    constrained_layout=True,
)

feature_to_title_lut = {
    "document_field_name_top_nine_plus_other": "Field Counts",
    "field_count": "Field Count over Time",
    "repository_primary_language_top_nine_plus_other": "Primary Language Counts",
    "language_count": "Primary Language Count over Time",
}

# Field name normal
sns.countplot(
    data=pairs,
    x="document_field_name_top_nine_plus_other",
    order=pairs["document_field_name_top_nine_plus_other"].value_counts(sort=True)[
        "document_field_name_top_nine_plus_other"
    ],
    hue="document_field_name_top_nine_plus_other",
    hue_order=field_hue_order,
    ax=axes[0, 0],
)

# Field name over time
sns.lineplot(
    data=pairs.group_by(
        ["document_field_name_top_nine_plus_other", "document_publication_year"]
    )
    .agg(pl.len().alias("field_count"))
    .filter(
        pl.col("document_publication_year") < 2025,
    ),
    x="document_publication_year",
    y="field_count",
    hue="document_field_name_top_nine_plus_other",
    hue_order=field_hue_order,
    ax=axes[0, 1],
    legend=False,
)

# Add column for top N primary languages + other
top_nine_languages = (
    pairs.filter(pl.col("repository_primary_language").is_not_null())[
        "repository_primary_language"
    ]
    .value_counts(sort=True)
    .head(DEFAULT_TOP_N)["repository_primary_language"]
    .to_list()
)
top_nine_languages_hue_order = [*top_nine_languages, "Other"]
pairs = pairs.with_columns(
    pl.when(pl.col("repository_primary_language").is_in(top_nine_languages))
    .then(pl.col("repository_primary_language"))
    .otherwise(pl.lit("Other"))
    .alias("repository_primary_language_top_nine_plus_other")
)

# Primary language normal
sns.countplot(
    data=pairs,
    x="repository_primary_language_top_nine_plus_other",
    order=pairs["repository_primary_language_top_nine_plus_other"].value_counts(sort=True)[
        "repository_primary_language_top_nine_plus_other"
    ],
    hue="repository_primary_language_top_nine_plus_other",
    hue_order=top_nine_languages_hue_order,
    ax=axes[1, 0],
)

# Primary language over time
sns.lineplot(
    data=pairs.group_by(
        ["repository_primary_language_top_nine_plus_other", "document_publication_year"]
    )
    .agg(pl.len().alias("language_count"))
    .filter(
        pl.col("document_publication_year") < 2025,
    ),
    x="document_publication_year",
    y="language_count",
    hue="repository_primary_language_top_nine_plus_other",
    hue_order=top_nine_languages_hue_order,
    ax=axes[1, 1],
    legend=False,
)

for ax in axes.flat:
    # Rotate all x-tick labels for readability
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Remove axis labels
    feature_name = ax.get_xlabel()
    feature_name = (
        ax.get_ylabel() if feature_name == "document_publication_year" else feature_name
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Set title
    ax.set_title(feature_to_title_lut.get(feature_name, feature_name), fontsize=16)

fig.tight_layout(h_pad=3.0)
fig.savefig(RESULTS_DIR / "field_and_language_counts.png", bbox_inches="tight", dpi=300)

# Plot median repository commit duration, size, commits, and number of files over time
over_time_metrics = [
    "repository_commit_duration_days",
    "repository_size_kb",
    "repository_commits_count",
    "repository_n_files",
]

over_time_metrics_to_viz_name_lut = {
    "repository_commit_duration_days": "Repository Commit Duration (Days)",
    "repository_size_kb": "Repository Size (KB)",
    "repository_commits_count": "Repository Commits Count",
    "repository_n_files": "Repository Number of Files",
}

over_time_df = pairs.filter(
    pl.col("document_publication_year") < 2025,
    pl.col("document_publication_year") > 2010,
)

over_time_df = over_time_df.unpivot(
    on=over_time_metrics,
    variable_name="metric",
    value_name="median_value",
    index=["document_publication_year"],
)

g = sns.catplot(
    data=over_time_df,
    x="document_publication_year",
    y="median_value",
    hue="metric",
    col="metric",
    col_wrap=2,
    kind="bar",
    sharey=False,
    legend=False,
)

# Set titles and labels
g.set_titles("{col_name}")

# Rotate x-axis labels for readability
for i, ax in enumerate(g.axes):
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    ax.set_xlabel("")
    ax.set_ylabel("Median Value")
    ax.set_title(over_time_metrics_to_viz_name_lut.get(over_time_metrics[i]), fontsize=16)

g.figure.tight_layout(h_pad=3.0)
g.figure.savefig(RESULTS_DIR / "repo_metrics_over_time.png", bbox_inches="tight", dpi=300)


# ## FWCI vs FWSI

# FWCI vs FWSI scatter plot with top points labeled
fwci_fwsi_plot_data = (
    pairs.filter(
        pl.col("document_fwci").is_not_null(),
        pl.col("repository_fwsi").is_not_null(),
        pl.col("document_fwci").is_not_nan(),
        pl.col("repository_fwsi").is_not_nan(),
        pl.col("document_fwci").is_finite(),
        pl.col("repository_fwsi").is_finite(),
    )
    .with_columns(
        (pl.lit(1) + pl.col("document_fwci")).log10().alias("document_fwci_log10"),
        (pl.lit(1) + pl.col("repository_fwsi")).log10().alias("repository_fwsi_log10"),
    )
    .filter(
        pl.col("document_fwci_log10").is_not_null(),
        pl.col("repository_fwsi_log10").is_not_null(),
        pl.col("document_fwci_log10").is_not_nan(),
        pl.col("repository_fwsi_log10").is_not_nan(),
        pl.col("document_fwci_log10").is_finite(),
        pl.col("repository_fwsi_log10").is_finite(),
    )
)

fwci_fwsi_ax = sns.scatterplot(
    data=fwci_fwsi_plot_data,
    x="document_fwci_log10",
    y="repository_fwsi_log10",
    alpha=0.1,
)


def _plot_fwci_fwsi_point_data(row: dict, color: str, ax: plt.Axes) -> None:
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


# Label the top two points by document FWCI
top_fwci_points = fwci_fwsi_plot_data.sort("document_fwci_log10", descending=True).head(2)
print(
    top_fwci_points[
        [
            "document_fwci_log10",
            "document_title",
            "repository_owner",
            "repository_name",
        ]
    ]
)
colors = ["red", "blue"]
for row in top_fwci_points.iter_rows(named=True):
    _plot_fwci_fwsi_point_data(row, colors.pop(0), fwci_fwsi_ax)

# Label the top two points by repository FWSI
top_fwsi_points = fwci_fwsi_plot_data.sort("repository_fwsi_log10", descending=True)[[0, 4]]
print(
    top_fwsi_points[
        [
            "repository_fwsi_log10",
            "document_title",
            "repository_owner",
            "repository_name",
        ]
    ]
)
colors = ["green", "orange"]
for row in top_fwsi_points.iter_rows(named=True):
    _plot_fwci_fwsi_point_data(row, colors.pop(0), fwci_fwsi_ax)

# Find a repo with high product of FWCI and FWSI
fwci_fwsi_plot_data = fwci_fwsi_plot_data.with_columns(
    (pl.col("document_fwci_log10") * pl.col("repository_fwsi_log10")).alias(
        "fwci_fwsi_product_log10"
    )
)
selected_fwci_fwsi_points = fwci_fwsi_plot_data.sort(
    "fwci_fwsi_product_log10", descending=True
)[[1, 4]]
print(
    selected_fwci_fwsi_points[
        [
            "fwci_fwsi_product_log10",
            "document_title",
            "repository_owner",
            "repository_name",
        ]
    ]
)
colors = ["brown", "purple"]
for row in selected_fwci_fwsi_points.iter_rows(named=True):
    _plot_fwci_fwsi_point_data(row, colors.pop(0), fwci_fwsi_ax)
fwci_fwsi_ax.figure.savefig(
    RESULTS_DIR / "fwci_vs_fwsi_scatter.png", bbox_inches="tight", dpi=300
)

# Unpivot the plot data to only include doc id, repo id, fwci_log10, fwsi_log10
fwci_fwsi_plot_data_melted = fwci_fwsi_plot_data.select(
    [
        "document_id",
        "repository_id",
        "document_fwci_log10",
        "repository_fwsi_log10",
    ]
).unpivot(
    on=["document_fwci_log10", "repository_fwsi_log10"],
    index=["document_id", "repository_id"],
    variable_name="metric",
    value_name="log10_value",
)

# Plot the distribution of FWCI and FWSI log10 values
g = sns.displot(
    data=fwci_fwsi_plot_data_melted,
    x="log10_value",
    hue="metric",
    col="metric",
    bins=20,
    stat="proportion",
    legend=False,
)

# Update the subplot titles
g.set_titles("{col_name}")
g.figure.savefig(
    RESULTS_DIR / "fwci_fwsi_distributions.png", bbox_inches="tight", dpi=300
)


# ## Date Relationships

# Two plots, one for repo creation vs publication date and one for
# publication date vs most recent push date
date_relationships_df = pairs.select(
    [
        "document_publication_date",
        "repository_creation_datetime",
        "repository_last_pushed_datetime",
    ]
).filter(
    pl.col("document_publication_date").dt.year() < 2025,
    pl.col("document_publication_date").dt.year() > 2010,
    pl.col("repository_creation_datetime").dt.year() < 2025,
    pl.col("repository_creation_datetime").dt.year() > 2010,
    pl.col("repository_last_pushed_datetime").dt.year() > 2010,
)

# Create fig axes for 2 plots
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(15, 7),
    constrained_layout=True,
)

sns.scatterplot(
    data=date_relationships_df,
    x="document_publication_date",
    y="repository_creation_datetime",
    alpha=0.1,
    ax=axes[0],
)

# Add 45 degree line
min_date = min(
    date_relationships_df["document_publication_date"].min(),
    date_relationships_df["repository_creation_datetime"].dt.date().min(),
)
max_date = max(
    date_relationships_df["document_publication_date"].max(),
    date_relationships_df["repository_creation_datetime"].dt.date().max(),
)
_ = axes[0].plot(
    [min_date, max_date],
    [min_date, max_date],
    color="red",
    linestyle="--",
)

sns.scatterplot(
    data=date_relationships_df,
    x="repository_last_pushed_datetime",
    y="document_publication_date",
    alpha=0.1,
    ax=axes[1],
)
axes[1].invert_yaxis()

# Add 45 degree line
min_date = min(
    date_relationships_df["document_publication_date"].min(),
    date_relationships_df["repository_last_pushed_datetime"].dt.date().min(),
)
max_date = max(
    date_relationships_df["document_publication_date"].max(),
    date_relationships_df["repository_last_pushed_datetime"].dt.date().max(),
)
_ = axes[1].plot(
    [min_date, max_date],
    [min_date, max_date],
    color="red",
    linestyle="--",
)

features_to_viz_name_lut = {
    "repository_creation_datetime": "Repository Creation Date",
    "document_publication_date": "Document Publication Date",
    "repository_last_pushed_datetime": "Repository Last Pushed Date",
}

for ax in axes.flat:
    # Remove axis labels
    x_label = ax.get_xlabel()
    y_label = ax.get_ylabel()
    ax.set_xlabel(features_to_viz_name_lut.get(x_label, x_label))
    ax.set_ylabel(features_to_viz_name_lut.get(y_label, y_label))

# Tighten layout
fig.tight_layout(w_pad=3.0)
fig.savefig(RESULTS_DIR / "date_relationships_scatter.png", bbox_inches="tight", dpi=300)

# Create the same plot but as distribution of "Days from X to Y"
date_relationships_days_df = (
    date_relationships_df.with_columns(
        (
            (
                pl.col("document_publication_date") - pl.col("repository_creation_datetime")
            ).dt.total_days()
        ).alias("days_from_repo_creation_to_publication"),
        (
            (
                pl.col("repository_last_pushed_datetime") - pl.col("document_publication_date")
            ).dt.total_days()
        ).alias("days_from_publication_to_last_push"),
    )
    .select(
        [
            "days_from_repo_creation_to_publication",
            "days_from_publication_to_last_push",
        ]
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

g = sns.displot(
    data=date_relationships_days_df.filter(
        pl.col("days_difference")
        > date_relationships_days_df["days_difference"].quantile(0.01),
        pl.col("days_difference")
        < date_relationships_days_df["days_difference"].quantile(0.99),
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
    RESULTS_DIR / "date_difference_distributions.png", bbox_inches="tight", dpi=300
)


# ## Network Coverage

def build_coauthorship_network(
    df: pl.DataFrame, sample_size: int | None = None
) -> tuple[rx.PyGraph, dict[int, int], dict[int, int], pl.DataFrame]:
    """Build a co-authorship network from document contributors using rustworkx.

    Returns:
        Tuple of (graph, node_to_idx mapping, idx_to_node mapping, doc contribs DataFrame).
    """
    doc_contribs = _read_table("document_contributor")

    # Filter to documents in our dataset
    doc_ids_in_dataset = df["document_id"].unique().to_list()
    doc_contribs = doc_contribs.filter(pl.col("document_id").is_in(doc_ids_in_dataset))

    # Apply sampling if requested
    if sample_size is not None:
        sampled_doc_ids = (
            doc_contribs.select("document_id")
            .unique()
            .sample(n=min(sample_size, len(doc_ids_in_dataset)), seed=42)
        )
        doc_contribs = doc_contribs.filter(
            pl.col("document_id").is_in(sampled_doc_ids["document_id"])
        )

    # Build rustworkx graph
    coauthorship_graph = rx.PyGraph()
    node_to_idx: dict[int, int] = {}  # researcher_id -> rustworkx node index
    idx_to_node: dict[int, int] = {}  # rustworkx node index -> researcher_id

    # Add nodes and edges
    for _, group in tqdm(
        doc_contribs.group_by("document_id"),
        total=doc_contribs["document_id"].n_unique(),
        desc="Building co-authorship network",
    ):
        # Add nodes for all authors
        for author in group.iter_rows(named=True):
            this_author_researcher_id = author["researcher_id"]
            if this_author_researcher_id not in node_to_idx:
                rx_node_idx = coauthorship_graph.add_node(this_author_researcher_id)
                node_to_idx[this_author_researcher_id] = rx_node_idx
                idx_to_node[rx_node_idx] = this_author_researcher_id

        # Add edges between co-authors
        for a1 in group.iter_rows(named=True):
            author_one_researcher_id = a1["researcher_id"]
            for a2 in group.iter_rows(named=True):
                author_two_researcher_id = a2["researcher_id"]
                if author_one_researcher_id == author_two_researcher_id:
                    continue

                # Add edge
                # Lookup node indices
                node_idx_1 = node_to_idx[author_one_researcher_id]
                node_idx_2 = node_to_idx[author_two_researcher_id]
                coauthorship_graph.add_edge(node_idx_1, node_idx_2, 1)

    print("Network built:")
    print(f"Nodes (authors): {coauthorship_graph.num_nodes():,}")
    print(f"Edges (co-authorships): {coauthorship_graph.num_edges():,}")

    return coauthorship_graph, node_to_idx, idx_to_node, doc_contribs


coauthorship_graph, node_to_idx, idx_to_node, doc_contribs = build_coauthorship_network(pairs)

components = rx.connected_components(coauthorship_graph)
component_sizes = sorted([len(c) for c in components], reverse=True)

total_nodes = coauthorship_graph.num_nodes()
largest_component_size = component_sizes[0] if component_sizes else 0
coverage = largest_component_size / total_nodes if total_nodes > 0 else 0

print(f"\nTotal number of components: {len(components):,}")
print(f"Total authors (nodes): {total_nodes:,}")
print(f"Largest component size: {largest_component_size:,}")
print(f"Coverage (largest / total): {coverage:.2%}")

print("\nComponent size distribution:")
print(f"  Largest 5: {component_sizes[:5]}")
print(f"  Isolates (size 1): {component_sizes.count(1):,}")
print(f"  Size 2-10: {sum(1 for s in component_sizes if 2 <= s <= 10):,}")
print(f"  Size 11-100: {sum(1 for s in component_sizes if 11 <= s <= 100):,}")
print(f"  Size >100: {sum(1 for s in component_sizes if s > 100):,}")

N_ITERATIONS = 5000

# Get the largest connected component
largest_cc = max(components, key=len)
largest_cc_nodes = list(largest_cc)

# Create subgraph for largest component
coauthorship_graph_largest_component = coauthorship_graph.subgraph(largest_cc_nodes)

print("\nAnalyzing largest connected component:")
print(f"  Nodes: {coauthorship_graph_largest_component.num_nodes():,}")
print(f"  Edges: {coauthorship_graph_largest_component.num_edges():,}")
print()

# Get all node indices for sampling
subgraph_indices = list(range(coauthorship_graph_largest_component.num_nodes()))

# Iteration Loop
dijkstra_lengths = []
for _ in tqdm(range(N_ITERATIONS), desc="Getting random shortest paths"):
    # Randomly select two distinct nodes
    source, target = random.sample(subgraph_indices, 2)

    # Dijkstra Shortest Path
    # Returns a dictionary {target_node: length}
    dijkstra_res = rx.dijkstra_shortest_path_lengths(
        coauthorship_graph_largest_component,
        source,
        lambda _: 1,  # Weight function (1 = unweighted/hops)
        goal=target,
    )
    dijkstra_lengths.append(dijkstra_res[target])

# Calculate Statistics
dijkstra_vec = np.array(dijkstra_lengths)

print()
print("--- Results ---")
print(f"Valid paths found: {len(dijkstra_vec)}")
print(f"  Mean:   {np.mean(dijkstra_vec):.4f}")
print(f"  Std:    {np.std(dijkstra_vec):.4f}")
print(f"  Median: {np.median(dijkstra_vec):.4f}")
