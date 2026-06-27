#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import evaplot
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import typer

from data_utils import load_base_dataset, load_table

###############################################################################

# Utilities


def save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    """Save a figure as PNG, TIFF, and PDF to OUTPUTS_DIR at 300 dpi."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "tiff", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add a bold panel label (A, B, C…) to the upper-left corner of an axes."""
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_table(df: pl.DataFrame, stem: str, output_dir: Path) -> None:
    """Save a Polars DataFrame as CSV to OUTPUTS_DIR."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_dir / f"{stem}.csv")


###############################################################################

app = typer.Typer()


@app.command()
def dataset_coverage_proportions(
    sqlite_database_path: Path,
    one_to_one_only: bool = False,
    min_year: int = 2008,
    confidence_threshold: float = 0.9994,
    top_n_fields: int = 10,
    output_dir: Path = Path("outputs"),
) -> None:
    # Apply evaplot style
    evaplot.set_style("evaplot_rc")

    # Load the dataset
    merged = load_base_dataset(
        sqlite_database_path,
        one_to_one_only=one_to_one_only,
        min_year=min_year,
        confidence_threshold=confidence_threshold,
        top_n_fields=top_n_fields,
    )

    # Create 2x2 grid of subplots
    # Each subplot has a panel label
    # A: proportion of article-repository pairs by field
    # (Overall, Papers with Code, Ours - Papers with Code)
    # B: proportion of article-repository pairs by publication year
    # (Overall, Papers with Code, Ours - Papers with Code)
    # C: proportion of article-repository pairs by document type (top 5 + "Other")
    # (Overall, Papers with Code, Ours - Papers with Code)
    # D: proportion of article-repository pairs by repository language (top 5 + "Other")
    # (Overall, Papers with Code, Ours - Papers with Code)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    add_panel_label(axes[0, 0], "A")
    add_panel_label(axes[0, 1], "B")
    add_panel_label(axes[1, 0], "C")
    add_panel_label(axes[1, 1], "D")

    # Panel A: Proportion of pairs by field (pruned)
    # Before we plot, we need to compute the proportions for pairs from:
    # "Papers with Code"
    # "Ours - All Sources"
    # "Ours - Without Papers with Code"

    # Get total counts for the three subsets overall
    total_pwc_count = len(merged.filter(pl.col("dataset_source_name_canonical") == "Papers with Code"))
    total_ours_all_count = len(merged)
    total_ours_no_pwc_count = len(merged.filter(pl.col("dataset_source_name_canonical") != "Papers with Code"))

    # Iter over the top 10 fields + "Other" and compute the proportion of pairs
    # in each dataset source "category"
    field_stats = []
    for field in merged.get_column("document_field_name_pruned").unique().to_list():
        field_df = merged.filter(pl.col("document_field_name_pruned") == field)
        pwc_count = len(field_df.filter(pl.col("dataset_source_name_canonical") == "Papers with Code"))
        ours_all_count = len(field_df)
        ours_no_pwc_count = len(
            field_df.filter(pl.col("dataset_source_name_canonical") != "Papers with Code")
        )
        field_stats.append(
            {
                "field": field,
                "pwc_count": pwc_count,
                "pwc_proportion": pwc_count / total_pwc_count,
                "ours_all_count": ours_all_count,
                "ours_all_proportion": ours_all_count / total_ours_all_count,
                "ours_no_pwc_count": ours_no_pwc_count,
                "ours_no_pwc_proportion": ours_no_pwc_count / total_ours_no_pwc_count,
            }
        )

    # Convert to frame and unpivot to long format for plotting
    field_stats_df = pl.DataFrame(field_stats).select(
        "field",
        "pwc_proportion",
        "ours_all_proportion",
        # "ours_no_pwc_proportion",
    ).unpivot(
        on=[
            "pwc_proportion",
            "ours_all_proportion",
            # "ours_no_pwc_proportion",
        ],
        index="field",
        variable_name="dataset_source_category",
        value_name="proportion",
    )

    # Sort by proportion descending for better visualization
    field_stats_df = field_stats_df.sort("proportion", descending=True)

    # Rename dataset source categories for better legend labels
    field_stats_df = field_stats_df.with_columns(
        pl.when(pl.col("dataset_source_category") == "pwc_proportion")
        .then(pl.lit("Papers with Code"))
        .when(pl.col("dataset_source_category") == "ours_all_proportion")
        .then(pl.lit("Ours - All Sources"))
        # .when(pl.col("dataset_source_category") == "ours_no_pwc_proportion")
        # .then(pl.lit("Ours - Without Papers with Code"))
        .otherwise(pl.col("dataset_source_category"))
        .alias("dataset_source_category")
    )

    # Make plot
    sns.barplot(
        data=field_stats_df,
        x="field",
        y="proportion",
        hue="dataset_source_category",
        ax=axes[1, 0],
        legend=False,
    )

    # Get order by getting count of pairs in each field
    field_display_order = (
        merged.get_column("document_field_name_pruned")
        .value_counts(sort=True)
        .get_column("document_field_name_pruned")
        .to_list()
    )

    sns.countplot(
        data=merged,
        x="document_field_name_pruned",
        stat="proportion",
        order=field_display_order,
        ax=axes[1, 0],
        legend=False,
    )

    # Get the field proportions as a CSV table as well
    field_proportions = (
        merged.group_by("document_field_name_pruned")
        .agg(pl.count().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))
        .sort("proportion", descending=True)
    )
    save_table(field_proportions, "field_proportions", output_dir)

    evaplot.rotate_xticklabels(axes[1, 0], rotation=40)

    # Plot B: Proportion of pairs by publication year
    sns.histplot(
        data=merged,
        x="document_publication_year",
        stat="proportion",
        ax=axes[0, 1],
        legend=False,
        shrink=5.2,
    )

    print("n docs", merged.n_unique("document_id"))
    print("n repositories", merged.n_unique("repository_id"))

    # Load the document_contributors table
    authors_df = load_table("document_contributor", sqlite_database_path).filter(
        pl.col("document_id").is_in(merged.get_column("document_id").to_list())
    )
    print("n authors", authors_df.n_unique("researcher_id"))

    # Load the repository_contributors table
    repo_contributors_df = load_table("repository_contributor", sqlite_database_path).filter(
        pl.col("repository_id").is_in(merged.get_column("repository_id").to_list())
    )
    print("n repo contributors", repo_contributors_df.n_unique("developer_account_id"))

    print("----")
    print("edges")

    # Get the number of authorship edges
    print("n authorship edges", authors_df.height)
    print("n repository contribution edges", repo_contributors_df.height)
    print("n article-repository edges", merged.height)

    # Load the researcher_developer_account_link table to get the
    # number of edges
    researcher_developer_links_df = load_table(
        "researcher_developer_account_link", sqlite_database_path
    ).filter(
        pl.col("researcher_id").is_in(authors_df.get_column("researcher_id").to_list())
        | pl.col("developer_account_id").is_in(
            repo_contributors_df.get_column("developer_account_id").to_list()
        ),
        pl.col("predictive_model_confidence") >= 0.97,
    )
    print("n researcher-developer links", researcher_developer_links_df.height)

    evaplot.rotate_xticklabels(axes[0, 1], rotation=40)

    # Save the figure
    save_figure(fig, "dataset_coverage_proportions", output_dir)


###############################################################################

if __name__ == "__main__":
    app()
