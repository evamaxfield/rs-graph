#!/usr/bin/env python3
"""
Main figures and tables for the Nature Computational Science rs-graph paper.

Each typer command is self-contained: it loads all the data it needs from
scratch, produces its output, and saves it. Nothing is shared between commands
at runtime — re-running any single command is always safe and repeatable.

Usage:
    python main-figures-and-tables.py figure-3
    python main-figures-and-tables.py figure-4
    python main-figures-and-tables.py all
"""

from __future__ import annotations

from pathlib import Path

import evaplot
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import typer
from datasets import Dataset, load_dataset

###############################################################################
# CONSTANTS
###############################################################################

HF_DATASET = "sci-soft-collections/rs-graph-v2-full"

# Pairs below this confidence threshold are excluded (null = author-provided, always kept).
CONFIDENCE_THRESHOLD = 0.9994

# Researcher–developer identity links below this threshold are excluded.
RESEARCHER_DEVELOPER_LINK_CONFIDENCE = 0.97

# Earliest year to include (GitHub was founded in 2008).
MIN_YEAR = 2008

# Number of top fields to show explicitly; everything else is collapsed to "Other".
TOP_N_FIELDS = 10

OUTPUTS_DIR = Path(__file__).parent / "outputs"

# Engineering practice detection via declared dependency names.
# Keys are practice labels; values are lowercased package names that signal
# adoption of that practice.
#
# TODO: When repository_file is added to the HuggingFace dataset, supplement
# or replace this dependency-based detection with file-path patterns:
#   - CI/CD: .github/workflows/, .travis.yml, .circleci/
#   - Dependency manifest: requirements.txt, pyproject.toml, setup.py, DESCRIPTION (R)
#   - Testing files: tests/ directory, test_*.py, *_test.py
PRACTICE_PACKAGES: dict[str, set[str]] = {
    "Testing": {
        "pytest", "pytest-cov", "pytest-xdist", "pytest-asyncio",
        "coverage", "hypothesis", "tox", "nose", "unittest2",
        "testthat", "runit", "covr", "testit",
    },
    "Type Checking": {
        "mypy", "pyright", "pytype", "pyre-check",
    },
    "Linting": {
        "ruff", "flake8", "pylint", "pycodestyle", "pyflakes",
        "pylama", "pydocstyle", "mccabe",
        "lintr",
    },
    "Formatting": {
        "black", "isort", "autopep8", "yapf",
        "styler",
    },
    "Documentation": {
        "sphinx", "mkdocs", "pdoc", "numpydoc", "pydoc-markdown",
        "sphinx-rtd-theme", "sphinx-autodoc-typehints",
        "roxygen2", "pkgdown",
    },
}

# Inverted lookup: lowercased package name → practice label.
_PKG_TO_PRACTICE: dict[str, str] = {
    pkg: practice
    for practice, pkgs in PRACTICE_PACKAGES.items()
    for pkg in pkgs
}

# Reference medians from published papers, used as vertical dashed lines in
# Figure 4 Panel A distributions.
#   Kalliamvakou et al. (2014): general GitHub sample.
#   Trujillo, Hébert-Dufresne & Bagrow (2022): general and Penumbra (academic) samples.
REFERENCE_LINES: dict[str, dict[str, float]] = {
    "commits": {
        "General (Kalliamvakou 2014)": 6,
        "General (Trujillo 2022)": 4,
        "Academic/Penumbra (Trujillo 2022)": 8,
    },
    "active_days": {
        "General (Kalliamvakou 2014)": 9.9,
        "General (Trujillo 2022)": 3,
        "Academic/Penumbra (Trujillo 2022)": 36,
    },
    "contributors": {
        "General (Trujillo 2022)": 1,
        "Academic/Penumbra (Trujillo 2022)": 1,
    },
}

###############################################################################
# STYLING
#
# evaplot.set_style() applies the evaplot matplotlib RC and categorical palette.
# We then apply Nature publication overrides (7–8 pt fonts, thin axes lines).
###############################################################################

evaplot.set_style(n=11)  # 10 fields + "Other"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    }
)

# Retrieve the categorical palette so it can be reused explicitly in any plot
# that needs per-category color mapping (e.g. practice lines in Panel B).
PRACTICE_PALETTE: dict[str, str] = dict(
    zip(
        PRACTICE_PACKAGES.keys(),
        evaplot.set_cat_palette(n=len(PRACTICE_PACKAGES)),
    )
)

###############################################################################
# DATA LOADING UTILITIES
###############################################################################


def load_table(table: str) -> pl.DataFrame:
    """Load a single table from the HuggingFace dataset as a Polars DataFrame."""
    ds = load_dataset(HF_DATASET, table, split="train")
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


def load_base_dataset(
    one_to_one_only: bool = True,
    top_n_fields: int = TOP_N_FIELDS,
) -> pl.DataFrame:
    """Load and filter the core article-repository pair dataset.

    Joins document_repository_link + document + repository + topics + author stats.
    Applies: publication_year >= MIN_YEAR, confidence threshold, optional 1:1 dedup.

    Added columns:
        document_publication_year      -- int, extracted from publication_date
        document_publication_date_parsed -- date, parsed from the date string
        document_field_name_pruned     -- top N field names, rest → "Other"
        document_field_name            -- original field name from topics taxonomy
        document_domain_name           -- top-level OpenAlex domain
        document_author_count          -- number of authors on the paper
        document_author_mean_citations -- mean cited_by_count across authors
    """
    documents = load_table("document")
    article_repo_links = load_table("document_repository_link")
    repositories = load_table("repository")
    document_topics = load_table("document_topic")
    topics = load_table("topic")
    document_contributors = load_table("document_contributor")
    researchers = load_table("researcher")

    # Per-document author statistics (count, mean citations).
    document_author_stats = (
        document_contributors.join(
            researchers.select(
                pl.col("id").alias("researcher_id"),
                pl.col("cited_by_count").alias("researcher_cited_by_count"),
            ),
            on="researcher_id",
            how="left",
        )
        .group_by("document_id")
        .agg(
            pl.len().alias("document_author_count"),
            pl.mean("researcher_cited_by_count").alias("document_author_mean_citations"),
        )
    )

    # Top topic per document (highest score) → field name + domain name.
    document_top_topics = (
        document_topics.sort("score", descending=True)
        .unique(subset="document_id", keep="first")
        .join(
            topics.select(
                pl.col("id").alias("topic_id"),
                pl.col("field_name").alias("document_field_name"),
                pl.col("domain_name").alias("document_domain_name"),
            ),
            on="topic_id",
        )
        .select("document_id", "document_field_name", "document_domain_name")
    )

    # Build the merged base dataframe.
    merged = (
        article_repo_links.select(
            pl.col("id").alias("document_repository_link_id"),
            "document_id",
            "repository_id",
            "dataset_source_id",
            "predictive_model_confidence",
        )
        .join(
            documents.select(
                *[pl.col(c).alias(f"document_{c}") for c in documents.columns]
            ),
            on="document_id",
        )
        .join(
            repositories.select(
                *[pl.col(c).alias(f"repository_{c}") for c in repositories.columns]
            ),
            on="repository_id",
        )
        .join(document_top_topics, on="document_id")
        .join(document_author_stats, on="document_id", how="left")
    )

    # Parse publication date and extract year.
    merged = merged.with_columns(
        pl.col("document_publication_date")
        .str.to_date("%Y-%m-%d")
        .alias("document_publication_date_parsed"),
    ).with_columns(
        pl.col("document_publication_date_parsed")
        .dt.year()
        .alias("document_publication_year"),
    )

    # Filter: published after GitHub's founding (2008).
    merged = merged.filter(pl.col("document_publication_year") >= MIN_YEAR)

    # Filter: high-confidence or author-provided pairs (confidence > threshold or null).
    merged = merged.filter(
        (pl.col("predictive_model_confidence") > CONFIDENCE_THRESHOLD)
        | pl.col("predictive_model_confidence").is_null()
    )

    # Optional: restrict to strict 1:1 pairs (each document and repository appears once).
    if one_to_one_only:
        merged = (
            merged.unique(subset="document_id", keep="none")
            .unique(subset="repository_id", keep="none")
        )

    # Prune field names: top N kept as-is, everything else → "Other".
    top_field_names = (
        merged.get_column("document_field_name")
        .value_counts(sort=True)
        .head(top_n_fields)
        .get_column("document_field_name")
        .to_list()
    )
    merged = merged.with_columns(
        pl.when(pl.col("document_field_name").is_in(top_field_names))
        .then(pl.col("document_field_name"))
        .otherwise(pl.lit("Other"))
        .alias("document_field_name_pruned")
    )

    return merged


def load_repo_contributor_counts() -> pl.DataFrame:
    """Load repository_contributor and return per-repo contributor counts.

    Returns a DataFrame with columns: repository_id, contributor_count.
    """
    return (
        load_table("repository_contributor")
        .group_by("repository_id")
        .agg(pl.len().alias("contributor_count"))
    )


###############################################################################
# PLOT UTILITIES
###############################################################################


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure as PNG, TIFF, and PDF to OUTPUTS_DIR at 300 dpi."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "tiff", "pdf"):
        fig.savefig(OUTPUTS_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


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


###############################################################################
# TYPER APP
###############################################################################

app = typer.Typer()


@app.command()
def figure_3() -> None:
    """Dataset coverage: pairs by field, by year, software view coverage, identity link growth."""
    # -------------------------------------------------------------------------
    # 1. Load core pairs and join dataset source names.
    # -------------------------------------------------------------------------
    base = load_base_dataset()
    dataset_sources = load_table("dataset_source").select(
        pl.col("id").alias("dataset_source_id"),
        pl.col("name").alias("source_name"),
    )
    base = base.join(dataset_sources, on="dataset_source_id", how="left")

    total_pairs = len(base)

    # -------------------------------------------------------------------------
    # 2. Panel A — Pairs by field (horizontal bar chart, sorted descending).
    # -------------------------------------------------------------------------
    pairs_by_field = (
        base.group_by("document_field_name_pruned")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    # -------------------------------------------------------------------------
    # 3. Panel B — Pairs over time (bar chart by publication year).
    # -------------------------------------------------------------------------
    pairs_by_year = (
        base.group_by("document_publication_year")
        .agg(pl.len().alias("count"))
        .sort("document_publication_year")
    )

    # -------------------------------------------------------------------------
    # 4. Panel C — Software view coverage.
    #
    # For each of the three software views (mentions, imports, dependencies),
    # compute the fraction of base pairs that have at least one record.
    # "Complete cases" are pairs where all three views are present.
    # -------------------------------------------------------------------------
    # Mentions are indexed by document_id.
    docs_with_mentions = (
        load_table("document_software_mention")
        .select("document_id")
        .unique()
        .join(base.select("document_id"), on="document_id", how="inner")
    )

    # Imports and dependencies are indexed by repository_id.
    repos_with_imports = (
        load_table("repository_import")
        .select("repository_id")
        .unique()
        .join(base.select("repository_id"), on="repository_id", how="inner")
    )
    repos_with_deps = (
        load_table("repository_dependency")
        .select("repository_id")
        .unique()
        .join(base.select("repository_id"), on="repository_id", how="inner")
    )

    # Complete cases: pairs where the document has mentions AND the repo has
    # both imports and dependencies.
    complete_pairs = (
        base.select("document_id", "repository_id")
        .join(docs_with_mentions, on="document_id", how="inner")
        .join(repos_with_imports, on="repository_id", how="inner")
        .join(repos_with_deps, on="repository_id", how="inner")
    )

    coverage_df = pl.DataFrame(
        {
            "view": ["Mentions", "Imports", "Dependencies", "Complete (all three)"],
            "pct": [
                len(docs_with_mentions) / total_pairs * 100,
                len(repos_with_imports) / total_pairs * 100,
                len(repos_with_deps) / total_pairs * 100,
                len(complete_pairs) / total_pairs * 100,
            ],
        }
    )

    # -------------------------------------------------------------------------
    # 5. Panel D — Cumulative researcher–developer identity link growth.
    #
    # Load high-confidence identity links, map each researcher to the earliest
    # publication year they appear in (via document_contributor), then accumulate
    # cumulative unique links by year.
    # -------------------------------------------------------------------------
    identity_links = load_table("researcher_developer_account_link").filter(
        pl.col("predictive_model_confidence") > RESEARCHER_DEVELOPER_LINK_CONFIDENCE
    )

    # Earliest publication year for each researcher in our filtered base.
    researcher_earliest_year = (
        load_table("document_contributor")
        .join(
            base.select("document_id", "document_publication_year"),
            on="document_id",
            how="inner",
        )
        .group_by("researcher_id")
        .agg(pl.min("document_publication_year").alias("first_year"))
    )

    # Attach first_year to identity links; count new links per year, then cumsum.
    cumulative_links = (
        identity_links.select("researcher_id", "developer_account_id")
        .join(researcher_earliest_year, on="researcher_id", how="inner")
        .unique(subset=["researcher_id", "developer_account_id"])
        .group_by("first_year")
        .agg(pl.len().alias("new_links"))
        .sort("first_year")
        .with_columns(pl.col("new_links").cum_sum().alias("cumulative_links"))
    )

    # -------------------------------------------------------------------------
    # 6. Compose 2×2 figure and save.
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # Panel A — pairs by field
    sns.barplot(
        data=pairs_by_field.to_pandas(),
        y="document_field_name_pruned",
        x="count",
        ax=ax_a,
        orient="h",
        order=pairs_by_field.get_column("document_field_name_pruned").to_list(),
    )
    ax_a.set_xlabel("Number of pairs")
    ax_a.set_ylabel("")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    add_panel_label(ax_a, "A")

    # Panel B — pairs by year
    sns.barplot(
        data=pairs_by_year.to_pandas(),
        x="document_publication_year",
        y="count",
        ax=ax_b,
    )
    ax_b.set_xlabel("Publication year")
    ax_b.set_ylabel("Number of pairs")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    evaplot.rotate_xticklabels(ax_b, rotation=45)
    add_panel_label(ax_b, "B")

    # Panel C — software view coverage
    sns.barplot(
        data=coverage_df.to_pandas(),
        y="view",
        x="pct",
        ax=ax_c,
        orient="h",
        order=["Mentions", "Imports", "Dependencies", "Complete (all three)"],
    )
    ax_c.set_xlabel("Coverage (% of pairs)")
    ax_c.set_ylabel("")
    ax_c.set_xlim(0, 100)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    add_panel_label(ax_c, "C")

    # Panel D — cumulative identity link growth
    years = cumulative_links.get_column("first_year").to_list()
    cum_vals = cumulative_links.get_column("cumulative_links").to_list()
    ax_d.fill_between(years, cum_vals, alpha=0.25)
    ax_d.plot(years, cum_vals, linewidth=1)
    ax_d.set_xlabel("Year")
    ax_d.set_ylabel("Cumulative identity links")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    add_panel_label(ax_d, "D")

    evaplot.adjust_layout(fig)
    save_figure(fig, "figure-3")
    typer.echo("Saved figure-3 to outputs/")


@app.command()
def figure_4() -> None:
    """Repository characteristics & engineering practices (two saved panels)."""
    # =========================================================================
    # PANEL A — Repository metric distributions
    # =========================================================================

    # -------------------------------------------------------------------------
    # 1. Load base dataset and join repository contributor counts.
    # -------------------------------------------------------------------------
    base = load_base_dataset()
    contrib_counts = load_repo_contributor_counts()

    # Join contributor counts and compute derived timing metrics.
    base = (
        base.join(contrib_counts, on="repository_id", how="left")
        .with_columns(
            # Days from repo creation to last push (proxy for total active lifespan).
            (
                pl.col("repository_last_pushed_datetime").cast(pl.Date)
                - pl.col("repository_creation_datetime").cast(pl.Date)
            )
            .dt.total_days()
            .alias("active_days"),
            # Days from repo creation to paper publication (how long before pub was code started).
            (
                pl.col("document_publication_date_parsed")
                - pl.col("repository_creation_datetime").cast(pl.Date)
            )
            .dt.total_days()
            .alias("days_creation_to_pub"),
            # Days from paper publication to last push (how long code remained active post-pub).
            (
                pl.col("repository_last_pushed_datetime").cast(pl.Date)
                - pl.col("document_publication_date_parsed")
            )
            .dt.total_days()
            .alias("days_pub_to_last_push"),
        )
    )

    # -------------------------------------------------------------------------
    # 2. Build Panel A: four KDE distribution subplots (log-scaled x-axis).
    #
    # Subplot 1: contributor count
    # Subplot 2: commit count
    # Subplot 3: active days (creation → last push)
    # Subplot 4: timing relative to publication (two overlapping KDEs:
    #            creation→pub and pub→last_push, positive values only)
    #
    # Vertical dashed lines mark reference medians from published literature.
    # -------------------------------------------------------------------------
    fig_a, axes_a = plt.subplots(1, 4, figsize=(7.2, 2.2))

    # Helper: draw reference lines from REFERENCE_LINES on the given axes.
    ref_styles = [
        ("General (Kalliamvakou 2014)", "--", "gray"),
        ("General (Trujillo 2022)", ":", "dimgray"),
        ("Academic/Penumbra (Trujillo 2022)", "-.", "black"),
    ]

    def _draw_reference_lines(ax: plt.Axes, metric_key: str) -> None:
        for label, ls, color in ref_styles:
            val = REFERENCE_LINES.get(metric_key, {}).get(label)
            if val is not None:
                ax.axvline(val, linestyle=ls, color=color, linewidth=0.6, label=label)

    # Subplot 1: contributor count
    contributor_data = base.get_column("contributor_count").drop_nulls().to_list()
    sns.kdeplot(contributor_data, ax=axes_a[0], log_scale=True, fill=True, alpha=0.3)
    _draw_reference_lines(axes_a[0], "contributors")
    axes_a[0].set_xlabel("Contributors")
    axes_a[0].set_ylabel("Density")
    axes_a[0].spines["top"].set_visible(False)
    axes_a[0].spines["right"].set_visible(False)
    add_panel_label(axes_a[0], "A")

    # Subplot 2: commit count
    commit_data = (
        base.get_column("repository_commits_count").drop_nulls().to_list()
    )
    sns.kdeplot(commit_data, ax=axes_a[1], log_scale=True, fill=True, alpha=0.3)
    _draw_reference_lines(axes_a[1], "commits")
    axes_a[1].set_xlabel("Commits")
    axes_a[1].set_ylabel("")
    axes_a[1].spines["top"].set_visible(False)
    axes_a[1].spines["right"].set_visible(False)

    # Subplot 3: active days
    active_days_data = (
        base.filter(pl.col("active_days") > 0)
        .get_column("active_days")
        .to_list()
    )
    sns.kdeplot(active_days_data, ax=axes_a[2], log_scale=True, fill=True, alpha=0.3)
    _draw_reference_lines(axes_a[2], "active_days")
    axes_a[2].set_xlabel("Active days")
    axes_a[2].set_ylabel("")
    axes_a[2].spines["top"].set_visible(False)
    axes_a[2].spines["right"].set_visible(False)

    # Subplot 4: timing relative to publication (two KDEs, positive values only)
    before_pub_data = (
        base.filter(pl.col("days_creation_to_pub") > 0)
        .get_column("days_creation_to_pub")
        .to_list()
    )
    after_pub_data = (
        base.filter(pl.col("days_pub_to_last_push") > 0)
        .get_column("days_pub_to_last_push")
        .to_list()
    )
    sns.kdeplot(
        before_pub_data,
        ax=axes_a[3],
        log_scale=True,
        fill=True,
        alpha=0.3,
        label="Creation → publication",
    )
    sns.kdeplot(
        after_pub_data,
        ax=axes_a[3],
        log_scale=True,
        fill=True,
        alpha=0.3,
        label="Publication → last push",
    )
    axes_a[3].set_xlabel("Days relative to publication")
    axes_a[3].set_ylabel("")
    axes_a[3].spines["top"].set_visible(False)
    axes_a[3].spines["right"].set_visible(False)
    axes_a[3].legend(fontsize=5)

    # Shared reference-line legend for subplots 1–3
    handles, labels = [], []
    for label, ls, color in ref_styles:
        handles.append(
            plt.Line2D([0], [0], linestyle=ls, color=color, linewidth=0.8)
        )
        labels.append(label)
    fig_a.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=5,
        frameon=False,
        bbox_to_anchor=(0.42, -0.12),
    )

    evaplot.adjust_layout(fig_a)
    save_figure(fig_a, "figure-4-panel-a")

    # =========================================================================
    # PANEL B — Engineering practice adoption rates over time by OpenAlex domain
    #
    # Uses the 4 top-level OpenAlex domains (Physical Sciences, Life Sciences,
    # Social Sciences, Health Sciences) to keep the grid to 2×2.
    # =========================================================================

    # -------------------------------------------------------------------------
    # 3. Load repository dependencies and tag each with a practice category.
    # -------------------------------------------------------------------------
    deps = load_table("repository_dependency")

    # Assign a practice label to each dependency row; rows not matching any
    # known practice package are dropped.
    practice_tagged = (
        deps.select("repository_id", pl.col("name").str.to_lowercase().alias("name_lower"))
        .with_columns(
            pl.col("name_lower")
            .replace(
                old=list(_PKG_TO_PRACTICE.keys()),
                new=list(_PKG_TO_PRACTICE.values()),
                default=None,
            )
            .alias("practice")
        )
        .filter(pl.col("practice").is_not_null())
        .unique(subset=["repository_id", "practice"])  # one row per (repo, practice)
    )

    # -------------------------------------------------------------------------
    # 4. Join practices to base to get publication year and domain.
    # -------------------------------------------------------------------------
    base_slim = base.select(
        "repository_id", "document_publication_year", "document_domain_name"
    ).unique(subset="repository_id")

    repo_practices = practice_tagged.join(base_slim, on="repository_id", how="inner")

    # For each (year, domain, practice): fraction of repos with that practice.
    # Denominator = total repos published in that (year, domain).
    total_by_year_domain = (
        base_slim.group_by(["document_publication_year", "document_domain_name"])
        .agg(pl.len().alias("total_repos"))
    )

    adoption = (
        repo_practices.group_by(
            ["document_publication_year", "document_domain_name", "practice"]
        )
        .agg(pl.len().alias("repos_with_practice"))
        .join(
            total_by_year_domain,
            on=["document_publication_year", "document_domain_name"],
        )
        .with_columns(
            (pl.col("repos_with_practice") / pl.col("total_repos")).alias(
                "adoption_rate"
            )
        )
        .sort("document_publication_year")
    )

    # -------------------------------------------------------------------------
    # 5. Build Panel B: 2×2 grid, one subplot per domain.
    # -------------------------------------------------------------------------
    domains = (
        adoption.get_column("document_domain_name")
        .unique()
        .sort()
        .to_list()
    )

    fig_b, axes_b = plt.subplots(2, 2, figsize=(7.2, 4.5), sharex=True, sharey=True)

    for ax, domain in zip(axes_b.flat, domains):
        domain_data = (
            adoption.filter(pl.col("document_domain_name") == domain).to_pandas()
        )
        for practice, color in PRACTICE_PALETTE.items():
            practice_data = domain_data[domain_data["practice"] == practice]
            if practice_data.empty:
                continue
            ax.plot(
                practice_data["document_publication_year"],
                practice_data["adoption_rate"],
                label=practice,
                color=color,
                linewidth=0.8,
            )
        ax.set_title(domain, fontsize=7, pad=3)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots if fewer than 4 domains.
    for ax in axes_b.flat[len(domains):]:
        ax.set_visible(False)

    # Shared axis labels.
    for ax in axes_b[1]:
        ax.set_xlabel("Publication year")
    for ax in axes_b[:, 0]:
        ax.set_ylabel("Adoption rate")

    # Add panel label to first subplot only.
    add_panel_label(axes_b[0, 0], "B")

    # Shared legend below the grid.
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.2)
        for color in PRACTICE_PALETTE.values()
    ]
    fig_b.legend(
        handles,
        list(PRACTICE_PALETTE.keys()),
        loc="lower center",
        ncol=len(PRACTICE_PALETTE),
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    evaplot.adjust_layout(fig_b)
    save_figure(fig_b, "figure-4-panel-b")

    typer.echo("Saved figure-4-panel-a and figure-4-panel-b to outputs/")


@app.command()
def all() -> None:
    """Generate all main figures and tables."""
    figure_3()
    figure_4()


if __name__ == "__main__":
    app()
