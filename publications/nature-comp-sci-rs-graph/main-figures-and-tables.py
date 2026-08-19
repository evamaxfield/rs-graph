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

# Dependency classification sets for Plot 2
_TESTING_PKGS: set[str] = {
    "pytest",
    "pytest-cov",
    "coverage",
    "nose",
    "nose2",
    "unittest2",
    "tox",
    "nox",
    "hypothesis",
    "testthat",
    "covr",
    "tinytest",
    "jest",
    "mocha",
    "jasmine",
    "vitest",
}
_LINTING_PKGS: set[str] = {
    "flake8",
    "pylint",
    "ruff",
    "black",
    "isort",
    "autopep8",
    "pycodestyle",
    "bandit",
    "pylama",
    "pep8",
    "pre-commit",
    "lintr",
    "styler",
    "eslint",
    "prettier",
    "jshint",
    "rubocop",
    "hadolint",
}
_DOCS_PKGS: set[str] = {
    "sphinx",
    "mkdocs",
    "pdoc",
    "pdoc3",
    "docutils",
    "nbsphinx",
    "sphinx-rtd-theme",
    "myst-parser",
    "pkgdown",
    "roxygen2",
    "jsdoc",
    "typedoc",
    "readthedocs",
    "pydoc",
    "sphinx-autodoc-typehints",
}
_TYPE_CHECKING_PKGS: set[str] = {
    "mypy",
    "pyright",
    "pytype",
    "pyre-check",
    "typeguard",
    "beartype",
    "types-requests",
    "types-setuptools",
}

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
    total_pwc_count = len(
        merged.filter(pl.col("dataset_source_name_canonical") == "Papers with Code")
    )
    total_ours_all_count = len(merged)
    total_ours_no_pwc_count = len(
        merged.filter(pl.col("dataset_source_name_canonical") != "Papers with Code")
    )

    # Iter over the top 10 fields + "Other" and compute the proportion of pairs
    # in each dataset source "category"
    field_stats = []
    for field in merged.get_column("document_field_name_pruned").unique().to_list():
        field_df = merged.filter(pl.col("document_field_name_pruned") == field)
        pwc_count = len(
            field_df.filter(pl.col("dataset_source_name_canonical") == "Papers with Code")
        )
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
    field_stats_df = (
        pl.DataFrame(field_stats)
        .select(
            "field",
            "pwc_proportion",
            "ours_all_proportion",
            # "ours_no_pwc_proportion",
        )
        .unpivot(
            on=[
                "pwc_proportion",
                "ours_all_proportion",
                # "ours_no_pwc_proportion",
            ],
            index="field",
            variable_name="dataset_source_category",
            value_name="proportion",
        )
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


@app.command()
def development_duration_distribution(
    sqlite_database_path: Path,
    output_dir: Path = Path("outputs"),
) -> None:
    evaplot.set_style("evaplot_rc")
    colors = evaplot.set_cat_palette(n=11)

    df = load_base_dataset(sqlite_database_path)

    df = (
        df.with_columns(
            pl.col("repository_creation_datetime").cast(pl.Datetime("us")),
            pl.col("repository_last_pushed_datetime").cast(pl.Datetime("us")),
            pl.col("document_publication_date_parsed")
            .cast(pl.Datetime("us"))
            .alias("pub_datetime"),
        )
        .with_columns(
            (pl.col("pub_datetime") - pl.col("repository_creation_datetime"))
            .dt.total_days()
            .alias("days_before_pub"),
            (pl.col("repository_last_pushed_datetime") - pl.col("pub_datetime"))
            .dt.total_days()
            .alias("days_after_pub"),
        )
        .filter(
            pl.col("days_before_pub").is_not_null() & pl.col("days_after_pub").is_not_null()
        )
    )

    before_label = "Before Publication\n(neg.: created after pub.)"
    after_label = "After Publication\n(neg.: last commit before pub.)"

    box_df = pl.concat(
        [
            df.select(
                pl.lit(before_label).alias("period"),
                (pl.col("days_before_pub") / 365.25).alias("duration_years"),
            ),
            df.select(
                pl.lit(after_label).alias("period"),
                (pl.col("days_after_pub") / 365.25).alias("duration_years"),
            ),
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.boxplot(
        data=box_df,
        x="period",
        y="duration_years",
        ax=ax,
        palette=colors,
        order=[before_label, after_label],
        showfliers=False,
    )

    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("Duration (Years)")

    save_figure(fig, "development-duration-distribution", output_dir)


@app.command()
def dependency_manifest_adoption(
    sqlite_database_path: Path,
    output_dir: Path = Path("outputs"),
) -> None:
    evaplot.set_style("evaplot_rc")
    evaplot.set_cat_palette(n=5)

    base_df = load_base_dataset(sqlite_database_path)
    year_repo = base_df.select("repository_id", "document_publication_year").filter(
        (pl.col("document_publication_year") > 2014)
        & (pl.col("document_publication_year") <= 2025)
    )

    total_per_year = year_repo.group_by("document_publication_year").agg(
        pl.len().alias("total")
    )

    deps = load_table("repository_dependency", sqlite_database_path)

    categories: dict[str, pl.DataFrame] = {
        "Any Manifest": deps,
        "Testing": deps.filter(pl.col("software_name_normalized").is_in(list(_TESTING_PKGS))),
        "Linting": deps.filter(pl.col("software_name_normalized").is_in(list(_LINTING_PKGS))),
        "Documentation": deps.filter(
            pl.col("software_name_normalized").is_in(list(_DOCS_PKGS))
        ),
        "Type Checking": deps.filter(
            pl.col("software_name_normalized").is_in(list(_TYPE_CHECKING_PKGS))
        ),
    }

    all_frames: list[pl.DataFrame] = []
    for cat, cat_deps in categories.items():
        cat_repos = cat_deps.select("repository_id").unique()
        frame = (
            year_repo.join(cat_repos, on="repository_id", how="semi")
            .group_by("document_publication_year")
            .agg(pl.len().alias("count"))
            .join(total_per_year, on="document_publication_year")
            .with_columns(
                (pl.col("count") / pl.col("total") * 100).alias("pct_repos"),
                pl.lit(cat).alias("category"),
            )
            .select("document_publication_year", "category", "pct_repos")
        )
        all_frames.append(frame)

    adoption_df = pl.concat(all_frames).sort("document_publication_year")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=adoption_df,
        x="document_publication_year",
        y="pct_repos",
        hue="category",
        ax=ax,
    )

    all_years = sorted(adoption_df["document_publication_year"].unique().to_list())
    ax.set_xticks(all_years)
    ax.set_xticklabels([str(y) for y in all_years])

    ax.set_xlabel("Publication Year")
    ax.set_ylabel("% of Article-Repository Pairs")
    ax.legend(title="Dependency Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    save_figure(fig, "dependency-manifest-adoption", output_dir)


@app.command()
def article_fwci_distribution(
    sqlite_database_path: Path,
    output_dir: Path = Path("outputs"),
) -> None:
    evaplot.set_style("evaplot_rc")
    colors = evaplot.set_cat_palette(n=11)

    df = load_base_dataset(sqlite_database_path)
    # Filter nulls and zeros; log scale requires positive values
    df = df.filter(pl.col("document_fwci").is_not_null() & (pl.col("document_fwci") > 0))

    fwci_99 = df.get_column("document_fwci").quantile(0.99)
    df = df.filter(pl.col("document_fwci") <= fwci_99)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(
        data=df,
        x="document_fwci",
        ax=ax,
        color=colors[2],
        log_scale=True,
    )

    fwci_median = df.get_column("document_fwci").median()

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="FWCI = 1.0 (world avg.)")
    ax.axvline(
        fwci_median,
        color="red",
        linestyle="-.",
        linewidth=1.5,
        label=f"Our Median: {fwci_median:.2f}",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncols=2)

    ax.set_xlabel("Field-Weighted Citation Impact (FWCI, log scale)")
    ax.set_ylabel("Count")

    save_figure(fig, "article-fwci-distribution", output_dir)


###############################################################################

if __name__ == "__main__":
    app()
