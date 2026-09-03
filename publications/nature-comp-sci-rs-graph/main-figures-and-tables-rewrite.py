#!/usr/bin/env python3

"""
Nature Computational Science `rs-graph` manuscript -- figures and tables.

Every command loads directly from HuggingFace (`sci-soft-collections/rs-graph-v2-full`),
applies the standard filters itself, and is runnable standalone -- no dependency on a local
SQLite checkout. Builds Figures 1-4, Table 1, the mention-predictors logistic regression, and
the supporting statistics/tables cited in the manuscript text.
"""

from __future__ import annotations

import json
import random
from collections import deque
from datetime import date
from pathlib import Path

import evaplot
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rustworkx as rx
import seaborn as sns
import statsmodels.formula.api as smf
import typer
import utils as u
from datasets import DatasetDict, load_dataset
from matplotlib.collections import LineCollection
from scipy.stats import norm, pearsonr, spearmanr
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from rs_graph.utils.identifier_normalization import normalize_name
from rs_graph.utils.software_alignment import align_software_names

###############################################################################

app = typer.Typer()

OUTPUT_DIR = Path(__file__).parent / "outputs"

###############################################################################
# Figure 2 -- dataset coverage


def _pwc_coverage_statistic(df: pl.DataFrame) -> dict[str, float]:
    """
    Compute what fraction of PwC's official paper<->code links rs-graph's own PwC-sourced,
    filtered pair set covers, plus the manuscript's "% increase over the verified, accessible
    subset of Papers with Code" statistic (line 23; our filtered PwC-sourced pairs are that
    subset).
    """
    print("\nLoading pwc-archive/links-between-paper-and-code from HuggingFace...")
    pwc_ds = load_dataset("pwc-archive/links-between-paper-and-code")
    assert isinstance(pwc_ds, DatasetDict)
    pwc_df = pwc_ds["train"].to_polars()
    pwc_official = pwc_df.filter(pl.col("is_official"))
    print(f"  PwC full: {len(pwc_df):,} rows | PwC official-only: {len(pwc_official):,} rows")

    our_pwc_pairs = df.filter(pl.col("dataset_source_name") == "pwc")
    coverage_pct = 100 * len(our_pwc_pairs) / len(pwc_official)
    increase_pct = 100 * (len(df) - len(our_pwc_pairs)) / len(our_pwc_pairs)

    print("\n--- PwC coverage statistic ---")
    print(
        f"rs-graph includes {len(our_pwc_pairs):,} PwC-sourced, filtered pairs, "
        f"covering {coverage_pct:.1f}% of PwC's {len(pwc_official):,} official "
        f"paper-repository links."
    )
    print(
        f"Line 23 fill: {len(df):,} filtered pairs is a {increase_pct:.0f}% increase over the "
        f"verified, accessible PwC subset ({len(our_pwc_pairs):,} pairs)."
    )
    print("-------------------------------\n")
    return {
        "n_filtered_pairs_total": float(len(df)),
        "n_pwc_sourced_filtered_pairs": float(len(our_pwc_pairs)),
        "pct_increase_over_pwc_verified_subset": increase_pct,
        "n_pwc_official_rows": float(len(pwc_official)),
        "pwc_coverage_pct": coverage_pct,
    }


OPENALEX_FIELD_COUNT_CACHE = Path(__file__).parent / "data" / "openalex-field-work-counts.json"


def _field_literature_penetration(
    df: pl.DataFrame, field_order: list[str]
) -> pl.DataFrame | None:
    """
    Compute field-level literature penetration: rs-graph pair count per top field divided by
    that field's total OpenAlex work count over the same publication-year window. One `pyalex`
    count query per field ("Other" is skipped -- it isn't an OpenAlex field); fetched counts
    are cached to a local JSON so re-renders don't refetch. Returns None (and the figure skips
    the annotation) if OpenAlex is unreachable and no cache exists.
    """
    import pyalex

    year_min = int(df.get_column("document_publication_year").min())
    year_max = int(df.get_column("document_publication_year").max())
    window_key = f"{year_min}-{year_max}"

    cache: dict[str, dict[str, int]] = {}
    if OPENALEX_FIELD_COUNT_CACHE.exists():
        cache = json.loads(OPENALEX_FIELD_COUNT_CACHE.read_text())

    real_fields = [f for f in field_order if f != "Other"]
    if window_key not in cache or any(f not in cache[window_key] for f in real_fields):
        try:
            oa_fields = {
                f["display_name"]: f["id"].rsplit("/", 1)[-1]
                for f in pyalex.Fields().get(per_page=30)
            }
            window_counts = cache.setdefault(window_key, {})
            for fname in real_fields:
                if fname in window_counts:
                    continue
                if fname not in oa_fields:
                    print(f"  No OpenAlex field matches '{fname}' -- skipping penetration.")
                    continue
                n = (
                    pyalex.Works()
                    .filter(
                        **{"primary_topic.field.id": oa_fields[fname]},
                        from_publication_date=f"{year_min}-01-01",
                        to_publication_date=f"{year_max}-12-31",
                    )
                    .count()
                )
                window_counts[fname] = int(n)
                print(f"  OpenAlex works, {fname}, {window_key}: {n:,}")
            OPENALEX_FIELD_COUNT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            OPENALEX_FIELD_COUNT_CACHE.write_text(json.dumps(cache, indent=2))
        except Exception as e:
            print(f"OpenAlex field-count fetch failed ({e}) -- penetration column skipped.")
            if window_key not in cache:
                return None

    pair_counts = df.group_by("document_field_name_pruned").agg(
        pl.len().alias("rs_graph_pairs")
    )
    rows = []
    for fname in real_fields:
        oa_count = cache.get(window_key, {}).get(fname)
        if oa_count is None:
            continue
        n_pairs = int(
            pair_counts.filter(pl.col("document_field_name_pruned") == fname)
            .get_column("rs_graph_pairs")
            .sum()
        )
        rows.append(
            {
                "field": fname,
                "rs_graph_pairs": n_pairs,
                "openalex_total_works": oa_count,
                "penetration_pct": 100 * n_pairs / oa_count,
                "year_window": window_key,
            }
        )
    return pl.DataFrame(rows) if rows else None


@app.command()
def figure_2_dataset_coverage(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Build Figure 2: dataset coverage. (A) pair counts by field -- rs-graph, stacked seed vs.
    mined, beside the verified PwC subset; (B) pair counts per publication year stacked by
    top-6 fields + Other, with the mined share of each year's pairs overlaid on a secondary
    axis. Also saves the field-proportion table (line 23's percentages) and a seed-source x
    field supplement table.
    """
    evaplot.set_style("evaplot_rc")
    df = u.load_filtered_pairs(top_n_fields=10)
    df = df.with_columns(pl.col("link_processing_iteration").is_not_null().alias("is_mined"))

    field_order = (
        df.get_column("document_field_name_pruned")
        .value_counts(sort=True)
        .get_column("document_field_name_pruned")
        .to_list()
    )

    ours_field_props = (
        df.group_by("document_field_name_pruned")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))
        .sort("proportion", descending=True)
    )
    u.save_table(ours_field_props, "figure2_field_proportions", output_dir)

    # ---- Seed vs. mined counts by field (Panel A's rs-graph stack) ----
    seed_mined_by_field = (
        df.group_by("document_field_name_pruned")
        .agg(
            (~pl.col("is_mined")).sum().alias("seed_count"),
            pl.col("is_mined").sum().alias("mined_count"),
        )
        .with_columns(
            (
                100 * pl.col("mined_count") / (pl.col("seed_count") + pl.col("mined_count"))
            ).alias("mined_pct")
        )
        .sort("seed_count", descending=True)
    )
    u.save_table(seed_mined_by_field, "figure2_seed_vs_mined_by_field", output_dir)
    print("\nSeed vs. mined pairs by field:")
    print(seed_mined_by_field)

    pwc_by_field = (
        df.filter(pl.col("dataset_source_name") == "pwc")
        .group_by("document_field_name_pruned")
        .agg(pl.len().alias("pwc_count"))
    )

    # ---- Seed-source x field supplement table (full source detail, incl. mined bucket) ----
    source_by_field = (
        df.group_by(["dataset_source_name_canonical", "document_field_name_pruned"])
        .agg(pl.len().alias("count"))
        .sort(["dataset_source_name_canonical", "count"], descending=[False, True])
    )
    u.save_table(source_by_field, "supplemental_seed_source_by_field", output_dir)

    # ---- Panel B data: counts per year x top-6 field + Other, plus per-year mined share ----
    current_year = date.today().year
    partial_year_note = None
    year_df = df
    if df.get_column("document_publication_year").max() == current_year:
        partial_year_note = f"{current_year} excluded: current year, partial-year data only"
        year_df = df.filter(pl.col("document_publication_year") != current_year)

    top6_fields = [f for f in field_order if f != "Other"][:6]
    year_field = (
        year_df.with_columns(
            pl.when(pl.col("document_field_name_pruned").is_in(top6_fields))
            .then(pl.col("document_field_name_pruned"))
            .otherwise(pl.lit("Other"))
            .alias("field7")
        )
        .group_by(["document_publication_year", "field7"])
        .agg(pl.len().alias("count"))
        .sort(["document_publication_year", "field7"])
    )
    mined_share_by_year = (
        year_df.group_by("document_publication_year")
        .agg((100 * pl.col("is_mined").mean()).alias("mined_pct_of_year"))
        .sort("document_publication_year")
    )
    u.save_table(
        year_field.join(mined_share_by_year, on="document_publication_year"),
        "figure2_pairs_by_year_and_field",
        output_dir,
    )

    # ---- Main figure: 2 panels, raw counts ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"wspace": 0.45})
    u.add_panel_label(axes[0], "A")
    u.add_panel_label(axes[1], "B")

    _palette2 = evaplot.set_cat_palette(n=2)
    family_green, family_orange = _palette2[0], _palette2[1]
    mined_tint = "#8fd4bc"  # lighter tint of the family green, for the mined segment

    # Panel A: grouped horizontal bars -- rs-graph (stacked seed+mined) vs. PwC per field.
    a_frame = (
        pl.DataFrame({"document_field_name_pruned": field_order})
        .join(seed_mined_by_field, on="document_field_name_pruned", how="left")
        .join(pwc_by_field, on="document_field_name_pruned", how="left")
        .with_columns(
            pl.col("seed_count").fill_null(0),
            pl.col("mined_count").fill_null(0),
            pl.col("pwc_count").fill_null(0),
        )
    )
    y_pos = np.arange(len(field_order))
    bar_h = 0.38
    seed_counts = a_frame.get_column("seed_count").to_numpy()
    mined_counts = a_frame.get_column("mined_count").to_numpy()
    pwc_counts = a_frame.get_column("pwc_count").to_numpy()
    axes[0].barh(
        y_pos - bar_h / 2,
        seed_counts,
        height=bar_h,
        color=family_green,
        label="rs-graph (seed pairs)",
    )
    axes[0].barh(
        y_pos - bar_h / 2,
        mined_counts,
        left=seed_counts,
        height=bar_h,
        color=mined_tint,
        label="rs-graph (mined pairs)",
    )
    axes[0].barh(
        y_pos + bar_h / 2,
        pwc_counts,
        height=bar_h,
        color=family_orange,
        label="Papers with Code (verified subset)",
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(field_order)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Article-Repository Pairs")

    # Per-bar annotation: rs-graph pairs as a share of the field's total OpenAlex literature
    # over the same year window.
    penetration = _field_literature_penetration(df, field_order)
    if penetration is not None:
        u.save_table(penetration, "figure2_field_literature_penetration", output_dir)
        print("\nField-level literature penetration:")
        print(penetration)
        pen_by_field = {r["field"]: r["penetration_pct"] for r in penetration.to_dicts()}
        max_total = float((seed_counts + mined_counts).max())
        axes[0].set_xlim(0, max_total * 1.28)
        for i, fname in enumerate(field_order):
            if fname not in pen_by_field:
                continue
            axes[0].text(
                seed_counts[i] + mined_counts[i] + max_total * 0.015,
                y_pos[i] - bar_h / 2,
                f"{pen_by_field[fname]:.2f}% of field's works",
                va="center",
                ha="left",
                fontsize=5.5,
                color="#555555",
            )
    leg_a = axes[0].legend(
        fontsize=7, title="", loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2
    )
    u.style_legend(leg_a)
    u.shrink_ticks(axes[0], size=9)

    # Panel B: stacked yearly counts by field, with the mined share of each year's pairs
    # overlaid as a line on a secondary axis (the plan's default choice -- preserves the
    # field-by-year detail while still showing what mining contributed when).
    b_fields = [*top6_fields, "Other"]
    # Distinct categorical hues (the blended green-orange palette reads as a gradient at
    # seven categories); Other stays gray.
    b_colors = [*sns.color_palette("colorblind", n_colors=len(top6_fields)), "#bbbbbb"]
    years = sorted(year_field.get_column("document_publication_year").unique().to_list())
    bottoms = np.zeros(len(years))
    year_index = {y: i for i, y in enumerate(years)}
    for fname, color in zip(b_fields, b_colors, strict=True):
        counts = np.zeros(len(years))
        for row in year_field.filter(pl.col("field7") == fname).iter_rows(named=True):
            counts[year_index[row["document_publication_year"]]] = row["count"]
        axes[1].bar(years, counts, bottom=bottoms, color=color, width=0.8, label=fname)
        bottoms += counts
    axes[1].set_xlabel("Publication Year")
    axes[1].set_ylabel("Article-Repository Pairs")
    ax_b2 = axes[1].twinx()
    ax_b2.plot(
        mined_share_by_year.get_column("document_publication_year").to_numpy(),
        mined_share_by_year.get_column("mined_pct_of_year").to_numpy(),
        color="#333333",
        linestyle="--",
        marker="o",
        markersize=3.5,
        linewidth=1.3,
        label="Mined share of pairs (%)",
    )
    ax_b2.set_ylabel("Mined Share of Pairs (%)")
    ax_b2.set_ylim(0, 100)
    ax_b2.grid(False)
    # Integer year ticks -- bar() on numeric x otherwise produces fractional labels.
    axes[1].set_xticks([y for y in years if y % 2 == 0])
    handles_b, labels_b = axes[1].get_legend_handles_labels()
    handles_b2, labels_b2 = ax_b2.get_legend_handles_labels()
    # Legend outside the axes: no in-panel pocket stays clear of bars or line at every year.
    leg_b = axes[1].legend(
        handles_b + handles_b2,
        labels_b + labels_b2,
        fontsize=6.5,
        title="",
        loc="center left",
        bbox_to_anchor=(1.16, 0.5),
        borderaxespad=0.0,
    )
    u.style_legend(leg_b, fontsize=6.5)
    evaplot.rotate_xticklabels(axes[1], rotation=40)
    u.shrink_ticks(axes[1], size=9)
    u.shrink_ticks(ax_b2, size=9)
    if partial_year_note:
        u.add_footnote(axes[1], partial_year_note, loc="bottom center outside")

    evaplot.adjust_layout(fig, bottom=0.22)
    u.save_figure(fig, "figure2_dataset_coverage", output_dir)
    plt.close(fig)

    # ---- Standalone PwC coverage statistic + line 23/paragraph summary stats ----
    pwc_stats = _pwc_coverage_statistic(df)
    # Field counts for line 23's "spans more than 20 fields" claim -- unpruned field taxonomy.
    n_distinct_fields = df.get_column("document_field_name").drop_nulls().n_unique()
    print(f"Distinct (unpruned) OpenAlex fields represented: {n_distinct_fields}")
    summary_stats = pl.DataFrame(
        {
            "statistic": [*pwc_stats.keys(), "n_distinct_fields_unpruned"],
            "value": [*pwc_stats.values(), float(n_distinct_fields)],
        }
    )
    u.save_table(summary_stats, "figure2_summary_stats", output_dir)


###############################################################################
# Figure 3 -- software development characteristics

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
# ruff is both linter and formatter; kept under linting. pre-commit is orchestration; also
# kept under linting.
_LINTING_PKGS: set[str] = {
    "flake8",
    "pylint",
    "ruff",
    "pycodestyle",
    "pep8",
    "pylama",
    "bandit",
    "lintr",
    "eslint",
    "jshint",
    "rubocop",
    "hadolint",
    "pre-commit",
}
_FORMATTING_PKGS: set[str] = {
    "black",
    "isort",
    "autopep8",
    "yapf",
    "prettier",
    "styler",
    "formatR",
    "docformatter",
}
# R has no mainstream type checker -- this category is Python-dominated (noted in Methods).
_TYPECHECKING_PKGS: set[str] = {
    "mypy",
    "pyright",
    "pyre-check",
    "pytype",
    "typeguard",
    "beartype",
    "pandera",
}
_DOCUMENTATION_PKGS: set[str] = {
    "sphinx",
    "sphinx-rtd-theme",
    "furo",
    "mkdocs",
    "mkdocs-material",
    "pdoc",
    "pdoc3",
    "numpydoc",
    "myst-parser",
    "roxygen2",
    "pkgdown",
    "docutils",
}
_DEPENDENCY_CATEGORIES: dict[str, set[str]] = {
    "Testing": _TESTING_PKGS,
    "Linting": _LINTING_PKGS,
    "Formatting": _FORMATTING_PKGS,
    "Type Checking": _TYPECHECKING_PKGS,
    "Documentation": _DOCUMENTATION_PKGS,
}


def _dev_duration_frame(df: pl.DataFrame) -> pl.DataFrame:
    out = (
        df.with_columns(
            pl.col("repository_creation_datetime_parsed").alias("repo_created"),
            pl.col("repository_last_pushed_datetime")
            .str.to_datetime(strict=False)
            .alias("repo_last_pushed"),
            pl.col("document_publication_date_parsed").cast(pl.Datetime("us")).alias("pub_dt"),
        )
        .with_columns(
            (pl.col("pub_dt") - pl.col("repo_created"))
            .dt.total_days()
            .alias("days_before_pub"),
            (pl.col("repo_last_pushed") - pl.col("pub_dt"))
            .dt.total_days()
            .alias("days_after_pub"),
        )
        .filter(
            pl.col("days_before_pub").is_not_null() & pl.col("days_after_pub").is_not_null()
        )
    )
    before = out.select(
        "dataset_source_name_canonical",
        "document_type_bucket",
        pl.lit("Before Publication").alias("period"),
        (pl.col("days_before_pub") / 365.25).alias("duration_years"),
    )
    after = out.select(
        "dataset_source_name_canonical",
        "document_type_bucket",
        pl.lit("After Publication").alias("period"),
        (pl.col("days_after_pub") / 365.25).alias("duration_years"),
    )
    return pl.concat([before, after])


MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR = 100


def _manifest_adoption_over_time(df: pl.DataFrame, deps: pl.DataFrame) -> pl.DataFrame:
    # Dedup to one row per repository (first-seen publication year) -- the panel reports a
    # "% of Repos" statistic, so a repository linked to multiple papers must not be counted
    # once per paper.
    current_year = date.today().year
    year_repo = (
        df.select("repository_id", "document_publication_year", "repository_primary_language")
        .unique(subset="repository_id", keep="first")
        .filter(
            (pl.col("document_publication_year") >= u.DEFAULT_MIN_YEAR)
            & (pl.col("document_publication_year") < current_year)
        )
    )
    # Pooled series (any manifest, all repos -- the paper's cited number) plus Python/R
    # ecosystem reference series (primary-language repos vs. their own manifest ecosystems).
    series_specs: list[tuple[str, pl.DataFrame, pl.DataFrame]] = [
        ("All repositories", year_repo, deps.select("repository_id").unique())
    ]
    for eco, dep_ecosystems in TABLE1_DEPENDENCY_ECOSYSTEMS.items():
        series_specs.append(
            (
                f"{eco}-primary repositories",
                year_repo.filter(pl.col("repository_primary_language") == eco),
                deps.filter(pl.col("ecosystem").is_in(dep_ecosystems))
                .select("repository_id")
                .unique(),
            )
        )
    frames = []
    for label, repos, repos_with_manifest in series_specs:
        total_per_year = repos.group_by("document_publication_year").agg(
            pl.len().alias("total")
        )
        frame = (
            repos.join(repos_with_manifest, on="repository_id", how="semi")
            .group_by("document_publication_year")
            .agg(pl.len().alias("count"))
            .join(total_per_year, on="document_publication_year", how="right")
            .with_columns(
                pl.col("count").fill_null(0),
                pl.lit(label).alias("series"),
            )
            .with_columns((pl.col("count") / pl.col("total") * 100).alias("pct_repos"))
            .filter(pl.col("total") >= MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR)
            .sort("document_publication_year")
        )
        frames.append(frame)
    return pl.concat(frames)


def _dependency_category_adoption(
    df: pl.DataFrame, deps: pl.DataFrame, group_col: str | None, with_year: bool
) -> pl.DataFrame:
    """Adoption of the five dependency categories, as a long (category, [group], [year])
    frame with count/total/pct_repos. Repo-year attribution: first-seen publication year per
    repository (same dedup rule as `_manifest_adoption_over_time`).
    """
    base_cols = ["repository_id", "document_publication_year"]
    if group_col is not None:
        base_cols.append(group_col)
    repo_base = df.select(base_cols).unique(subset="repository_id", keep="first")
    key_cols = ([group_col] if group_col is not None else []) + (
        ["document_publication_year"] if with_year else []
    )
    if not key_cols:
        repo_base = repo_base.with_columns(pl.lit("all").alias("_all"))
        key_cols = ["_all"]
    total_per_group = repo_base.group_by(key_cols).agg(pl.len().alias("total"))

    frames = []
    for category, pkg_set in _DEPENDENCY_CATEGORIES.items():
        # software_name_normalized is hyphen/underscore-stripped, so match on normalized names
        # (raw "pytest-cov" / "pre-commit" could never match the normalized column).
        norm_pkgs = [normalize_name(p) for p in pkg_set]
        cat_repos = (
            deps.filter(pl.col("software_name_normalized").is_in(norm_pkgs))
            .select("repository_id")
            .unique()
        )
        frame = (
            repo_base.join(cat_repos, on="repository_id", how="semi")
            .group_by(key_cols)
            .agg(pl.len().alias("count"))
            .join(total_per_group, on=key_cols, how="right")
            .with_columns(
                pl.col("count").fill_null(0),
                pl.lit(category).alias("category"),
            )
            .with_columns((pl.col("count") / pl.col("total") * 100).alias("pct_repos"))
        )
        frames.append(frame)
    out = pl.concat(frames)
    if "_all" in out.columns:
        out = out.drop("_all")
    return out.sort(["category", *[c for c in key_cols if c != "_all"]])


def _fwci_distribution_frame(fwci_docs: pl.DataFrame) -> pl.DataFrame:
    # `> 0` is required for the log-scale histogram (log(0) is undefined), but it also drops
    # every zero-lifetime-citation document (a valid 0.0 ratio) from the distribution and
    # median line -- report that exclusion.
    n_before_zero_filter = fwci_docs.filter(pl.col("document_raw_fwci").is_not_null()).height
    d = fwci_docs.filter(
        pl.col("document_raw_fwci").is_not_null() & (pl.col("document_raw_fwci") > 0)
    )
    n_zero_citation_dropped = n_before_zero_filter - d.height
    print(
        f"Figure 3 Panel D: excluding {n_zero_citation_dropped:,} zero-FWCI documents "
        "(OpenAlex FWCI = 0.0) from the log-scale distribution/median"
    )
    p99 = d.get_column("document_raw_fwci").quantile(0.99)
    return d.filter(pl.col("document_raw_fwci") <= p99)


def _fwsi_vs_fwci_by_field(
    df: pl.DataFrame, fwci_docs: pl.DataFrame, fwsi_repos: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Re-prune to top-5 + Other so the breakdown matches the manuscript's six named fields.
    top5 = (
        df.get_column("document_field_name")
        .value_counts(sort=True)
        .head(5)
        .get_column("document_field_name")
        .to_list()
    )
    field_map = (
        df.select("document_id", "document_field_name")
        .unique(subset="document_id", keep="first")
        .with_columns(
            pl.when(pl.col("document_field_name").is_in(top5))
            .then(pl.col("document_field_name"))
            .otherwise(pl.lit("Other"))
            .alias("field6")
        )
    )

    pair_level = (
        df.select("document_id", "repository_id")
        .join(fwci_docs.select("document_id", "document_raw_fwci"), on="document_id")
        .join(
            fwsi_repos.select("repository_id", "repository_modified_fwsi"), on="repository_id"
        )
        .join(field_map.select("document_id", "field6"), on="document_id")
        .drop_nulls(["document_raw_fwci", "repository_modified_fwsi"])
    )

    rho_rows = []
    for field, grp in pair_level.group_by("field6"):
        if grp.height < 10:
            continue
        rho, pval = spearmanr(
            grp.get_column("document_raw_fwci").to_numpy(),
            grp.get_column("repository_modified_fwsi").to_numpy(),
        )
        # 95% CI on Spearman rho via Fisher z-transform -- n varies substantially by field,
        # so a bare rho with no uncertainty invites over-reading small differences.
        z = np.arctanh(rho)
        se = 1.0 / np.sqrt(grp.height - 3)
        ci_lo, ci_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        rho_rows.append(
            {
                "field": field[0],
                "rho": rho,
                "p_value": pval,
                "n": grp.height,
                "rho_ci_lo": ci_lo,
                "rho_ci_hi": ci_hi,
            }
        )
    rho_df = pl.DataFrame(rho_rows).sort("rho", descending=True)
    return pair_level, rho_df


def _spearman_with_ci(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Spearman rho with a Fisher-z 95% CI."""
    rho, pval = spearmanr(x, y)
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(len(x) - 3)
    return {
        "rho": float(rho),
        "p_value": float(pval),
        "n": len(x),
        "rho_ci_lo": float(np.tanh(z - 1.96 * se)),
        "rho_ci_hi": float(np.tanh(z + 1.96 * se)),
    }


def _stars_vs_citations_by_field(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute per-field Spearman rho between raw repository stargazer counts and raw document
    citation counts, at the pair level, top-5 fields + Other, plus a pooled/overall row.
    """
    top5 = (
        df.get_column("document_field_name")
        .value_counts(sort=True)
        .head(5)
        .get_column("document_field_name")
        .to_list()
    )
    pair_level = (
        df.select(
            "document_id",
            "repository_id",
            "document_cited_by_count",
            "repository_stargazers_count",
            "document_field_name",
        )
        .with_columns(
            pl.when(pl.col("document_field_name").is_in(top5))
            .then(pl.col("document_field_name"))
            .otherwise(pl.lit("Other"))
            .alias("field6")
        )
        .drop_nulls(["document_cited_by_count", "repository_stargazers_count"])
    )
    rows = [
        {
            "field": "All fields (pooled)",
            **_spearman_with_ci(
                pair_level.get_column("repository_stargazers_count").to_numpy(),
                pair_level.get_column("document_cited_by_count").to_numpy(),
            ),
        }
    ]
    for field, grp in pair_level.group_by("field6"):
        if grp.height < 10:
            continue
        rows.append(
            {
                "field": field[0],
                **_spearman_with_ci(
                    grp.get_column("repository_stargazers_count").to_numpy(),
                    grp.get_column("document_cited_by_count").to_numpy(),
                ),
            }
        )
    return pl.DataFrame(rows)


# License-category keyword lists for Panel F.
# Matched against GitHub's license display names as stored in `repository_license`; checked
# copyleft-first so share-alike CC variants never fall through to the permissive CC match.
_COPYLEFT_LICENSE_KEYWORDS: tuple[str, ...] = (
    "GNU",  # GPL / AGPL / LGPL / FDL
    "Mozilla Public",
    "Eclipse Public",
    "European Union Public",
    "CeCILL",
    "Open Software License",
    "Share Alike",
    "CERN Open Hardware",
)
_PERMISSIVE_LICENSE_KEYWORDS: tuple[str, ...] = (
    "MIT",
    "Apache",
    "BSD",
    "ISC License",
    "Unlicense",
    "Zero v1.0",  # CC0
    "Attribution 4.0",  # CC-BY
    "Artistic License",
    "Boost Software",
    "What The F",  # WTFPL
    "zlib",
    "NCSA",
    "Universal Permissive",
    "Mulan Permissive",
    "Blue Oak",
    "Microsoft Public",
    "Academic Free",
)

LICENSE_ADOPTION_MIN_REPOS_PER_YEAR = 100


def _license_adoption_over_time(df: pl.DataFrame) -> pl.DataFrame:
    """Per first-seen publication year: % of repos with any license, a permissive license,
    or a copyleft license (same repo-dedup rule as `_manifest_adoption_over_time`).
    Licenses matching neither keyword list (incl. GitHub's literal "Other") count toward
    "any" only.
    """
    current_year = date.today().year
    repo_year = (
        df.select("repository_id", "document_publication_year", "repository_license")
        .unique(subset="repository_id", keep="first")
        .filter(
            (pl.col("document_publication_year") >= u.DEFAULT_MIN_YEAR)
            & (pl.col("document_publication_year") < current_year)
        )
    )
    copyleft_expr = pl.any_horizontal(
        [
            pl.col("repository_license").str.contains(k, literal=True)
            for k in _COPYLEFT_LICENSE_KEYWORDS
        ]
    )
    permissive_expr = pl.any_horizontal(
        [
            pl.col("repository_license").str.contains(k, literal=True)
            for k in _PERMISSIVE_LICENSE_KEYWORDS
        ]
    )
    repo_year = repo_year.with_columns(
        pl.col("repository_license").is_not_null().alias("has_license"),
        (pl.col("repository_license").is_not_null() & copyleft_expr)
        .fill_null(False)
        .alias("is_copyleft"),
    ).with_columns(
        (pl.col("repository_license").is_not_null() & ~pl.col("is_copyleft") & permissive_expr)
        .fill_null(False)
        .alias("is_permissive")
    )
    out = (
        repo_year.group_by("document_publication_year")
        .agg(
            pl.len().alias("n_repos"),
            (100 * pl.col("has_license").mean()).alias("pct_any_license"),
            (100 * pl.col("is_permissive").mean()).alias("pct_permissive"),
            (100 * pl.col("is_copyleft").mean()).alias("pct_copyleft"),
        )
        .sort("document_publication_year")
    )
    return out


@app.command()
def figure_3_software_development_characteristics(
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Build Figure 3: software development characteristics, one consolidated 6-panel figure.
    (A) dev activity duration, (B) dependency-manifest adoption over time, (C)
    dependency-category adoption over time, (D) raw-FWCI distribution, (E)
    raw-stars-vs-raw-citations Spearman rho by field as a horizontal forest plot with a
    pooled row, (F) license adoption over time (any / permissive / copyleft). Also produces
    the article-vs-preprint supplemental split from the same computations.
    """
    evaplot.set_style("evaplot_rc")
    df = u.load_filtered_pairs(top_n_fields=10)

    print("\nLoading repository_dependency from HuggingFace...")
    deps = u.load_table("repository_dependency")
    print(f"  repository_dependency: {len(deps):,} rows")
    deps = u.clean_dependency_names(deps)

    # Raw OpenAlex FWCI everywhere FWCI appears; modified (in-sample) FWCI stays available in
    # utils but is not on the paper path.
    fwci_docs = u.raw_fwci_docs(df)
    fwsi_repos = u.compute_modified_fwsi(df)

    dev_duration = _dev_duration_frame(df)
    manifest_adoption = _manifest_adoption_over_time(df, deps)
    category_by_year = _dependency_category_adoption(df, deps, group_col=None, with_year=True)
    category_by_domain = _dependency_category_adoption(
        df, deps, group_col="document_domain_name", with_year=False
    )
    category_by_field_year = _dependency_category_adoption(
        df, deps, group_col="document_field_name_pruned", with_year=True
    )
    fwci_dist = _fwci_distribution_frame(fwci_docs)
    _pair_level_fwsi_fwci, rho_df = _fwsi_vs_fwci_by_field(df, fwci_docs, fwsi_repos)

    # FWSI-vs-FWCI rho is saved as CSV only; Panel E plots raw stars vs. raw citations.
    u.save_table(rho_df, "figure3_fwsi_fwci_spearman_by_field", output_dir)
    print("\nFWSI-vs-raw-FWCI Spearman rho by field (CSV only, no longer plotted):")
    print(rho_df)

    stars_rho_df = _stars_vs_citations_by_field(df)
    u.save_table(stars_rho_df, "figure3_stars_citations_spearman_by_field", output_dir)
    print("\nRaw-stars-vs-raw-citations Spearman rho by field (Panel E):")
    print(stars_rho_df)

    license_adoption = _license_adoption_over_time(df)
    u.save_table(license_adoption, "figure3_license_adoption_by_year", output_dir)
    print("\nLicense adoption by year (Panel F):")
    print(license_adoption)

    # Panel B/C data as labeled tables -- the manuscript cites exact adoption percentages.
    u.save_table(manifest_adoption, "figure3_manifest_adoption_by_year", output_dir)
    current_year = date.today().year
    category_by_year_plotted = category_by_year.filter(
        pl.col("document_publication_year") < current_year
    )
    u.save_table(
        category_by_year_plotted, "figure3_dependency_category_adoption_by_year", output_dir
    )
    u.save_table(category_by_domain, "figure3_dependency_category_by_domain", output_dir)
    u.save_table(
        category_by_field_year,
        "supplemental_dependency_category_by_field_and_year",
        output_dir,
    )

    # FWCI summary stats -- the paper's headline median and per-field medians.
    fwci_nonnull = fwci_docs.filter(pl.col("document_raw_fwci").is_not_null())
    fwci_summary = pl.DataFrame(
        {
            "statistic": [
                "median_raw_fwci_incl_zero_citation_docs",
                "median_raw_fwci_excl_zero_citation_docs",
                "median_raw_fwci_plotted_distribution_zero_dropped_p99_capped",
                "n_documents_with_raw_fwci",
                "pct_documents_with_raw_fwci",
            ],
            "value": [
                float(fwci_nonnull.get_column("document_raw_fwci").median()),
                float(
                    fwci_nonnull.filter(pl.col("document_raw_fwci") > 0)
                    .get_column("document_raw_fwci")
                    .median()
                ),
                float(fwci_dist.get_column("document_raw_fwci").median()),
                float(fwci_nonnull.height),
                100 * fwci_nonnull.height / fwci_docs.height,
            ],
        }
    )
    u.save_table(fwci_summary, "figure3_fwci_summary_stats", output_dir)
    fwci_by_field = (
        fwci_nonnull.group_by("document_field_name_pruned")
        .agg(
            pl.len().alias("n_documents"),
            pl.median("document_raw_fwci").alias("median_raw_fwci"),
        )
        .sort("median_raw_fwci", descending=True)
    )
    u.save_table(fwci_by_field, "figure3_fwci_median_by_field", output_dir)
    print("\nMedian raw OpenAlex FWCI by field (incl. zero-citation docs):")
    print(fwci_by_field)

    # ---- 2x3 grid, six panels ----
    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 6, hspace=0.75, wspace=1.3)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2:4])
    ax_f = fig.add_subplot(gs[1, 4:6])

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], "ABCDEF", strict=True):
        u.add_panel_label(ax, label)

    # A: dev activity duration by source -- horizontal boxplots (duration on x); the
    # "Negative = ..." explainer lives in the paper's figure caption.
    sns.boxplot(
        data=dev_duration.to_pandas(),
        y="dataset_source_name_canonical",
        x="duration_years",
        hue="period",
        ax=ax_a,
        showfliers=False,
    )
    ax_a.set_ylabel("")
    ax_a.set_xlabel("Duration (Years)")
    leg = ax_a.legend(
        fontsize=7, title="", loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2
    )
    u.style_legend(leg)
    ax_a.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
    # Cap x-limits to the 2nd-98th percentile so long-tail whiskers don't squeeze the boxes.
    u.cap_ylim_to_quantiles(ax_a, dev_duration.to_pandas()["duration_years"], axis="x")
    u.shrink_ticks(ax_a, size=8)

    # B: dependency manifest adoption over time -- pooled line plus Python/R ecosystem
    # reference lines, full 2008+ window with an n-floor for sparse early years.
    sns.lineplot(
        data=manifest_adoption.to_pandas(),
        x="document_publication_year",
        y="pct_repos",
        hue="series",
        style="series",
        ax=ax_b,
        marker="o",
    )
    ax_b.set_xlabel("Publication Year")
    ax_b.set_ylabel("% of Repos with Manifest")
    ax_b.set_ylim(0, 100)
    u.style_legend(ax_b.legend(fontsize=6.5, title="", loc="upper left"), fontsize=6.5)
    u.add_footnote(
        ax_b,
        f"Years with < {MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR} repos in a series excluded",
        loc="upper right",
    )
    u.shrink_ticks(ax_b, size=8)

    # C: dependency-category adoption over time -- five lines, pooled across domains; the
    # per-domain snapshot goes to the backing CSV / supplement.
    sns.lineplot(
        data=category_by_year_plotted.filter(
            pl.col("total") >= MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR
        ).to_pandas(),
        x="document_publication_year",
        y="pct_repos",
        hue="category",
        style="category",
        hue_order=list(_DEPENDENCY_CATEGORIES),
        style_order=list(_DEPENDENCY_CATEGORIES),
        palette=u.field_palette(len(_DEPENDENCY_CATEGORIES)),
        markers=True,
        dashes=True,
        ax=ax_c,
    )
    ax_c.set_xlabel("Publication Year")
    ax_c.set_ylabel("% of Repos")
    u.style_legend(ax_c.legend(fontsize=6.5, title="", loc="upper left"), fontsize=6.5)
    u.shrink_ticks(ax_c, size=8)

    # D: raw OpenAlex FWCI distribution. bins=25: seaborn's log-scale "auto" rule produced
    # jagged bin-to-bin noise unrelated to real signal.
    sns.histplot(
        data=fwci_dist.to_pandas(), x="document_raw_fwci", ax=ax_d, log_scale=True, bins=25
    )
    fwci_median = fwci_dist.get_column("document_raw_fwci").median()
    ax_d.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Field avg.")
    ax_d.axvline(
        fwci_median,
        color="red",
        linestyle="-.",
        linewidth=1.5,
        label=f"Median: {fwci_median:.2f}",
    )
    ax_d.set_xlabel("OpenAlex FWCI (log scale)")
    ax_d.set_ylabel("Count")
    u.style_legend(ax_d.legend(fontsize=8))
    u.shrink_ticks(ax_d, size=9)
    u.add_footnote(
        ax_d,
        "Log scale cannot show 0: zero-FWCI documents\nare excluded (see console output for count).",
        loc="upper right",
    )

    # E: raw-stars-vs-raw-citations Spearman rho by field, drawn as a horizontal forest plot
    # with the pooled/overall rho as the top row and a reference line at its value.
    pooled_row = stars_rho_df.filter(pl.col("field") == "All fields (pooled)")
    field_rows = stars_rho_df.filter(pl.col("field") != "All fields (pooled)").sort(
        "rho", descending=True
    )
    forest = pl.concat([pooled_row, field_rows]).to_pandas()
    point_color = evaplot.set_cat_palette(n=1)[0]
    ys = np.arange(len(forest))
    xerr_lo = (forest["rho"] - forest["rho_ci_lo"]).to_numpy()
    xerr_hi = (forest["rho_ci_hi"] - forest["rho"]).to_numpy()
    ax_e.errorbar(
        x=forest["rho"],
        y=ys,
        xerr=[xerr_lo, xerr_hi],
        fmt="o",
        color=point_color,
        ecolor="black",
        elinewidth=1,
        capsize=3,
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=0.6,
    )
    pooled_rho = float(pooled_row.get_column("rho")[0])
    ax_e.axvline(pooled_rho, color="#888888", linewidth=0.9, linestyle="--", zorder=0)
    ax_e.axvline(0, color="#bbbbbb", linewidth=0.8, linestyle=":", zorder=0)
    ax_e.set_yticks(ys)
    ax_e.set_yticklabels(
        [f"{f} (n={n:,})" for f, n in zip(forest["field"], forest["n"], strict=True)]
    )
    ax_e.invert_yaxis()
    ax_e.set_ylim(len(forest) - 0.5, -0.5)
    ax_e.set_xlabel("Spearman rho (stars vs. citations)")
    u.shrink_ticks(ax_e, size=7)
    u.add_footnote(
        ax_e,
        "Raw stargazer and citation counts; dashed line = pooled rho.\nTop 5 fields + Other (differs from the top-10 grouping used elsewhere)",
        loc="top center outside",
    )

    # F: license adoption over time -- any / permissive / copyleft license share of repos by
    # first-seen publication year.
    license_plotted = license_adoption.filter(
        pl.col("n_repos") >= LICENSE_ADOPTION_MIN_REPOS_PER_YEAR
    )
    lic_colors = u.field_palette(3)
    for col, label, color, ls in [
        ("pct_any_license", "Any license", lic_colors[0], "-"),
        ("pct_permissive", "Permissive", lic_colors[1], "--"),
        ("pct_copyleft", "Copyleft", lic_colors[2], "-."),
    ]:
        ax_f.plot(
            license_plotted.get_column("document_publication_year").to_numpy(),
            license_plotted.get_column(col).to_numpy(),
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            linestyle=ls,
            color=color,
            label=label,
        )
    ax_f.set_xlabel("Publication Year")
    ax_f.set_ylabel("% of Repos")
    ax_f.set_ylim(0, 100)
    u.style_legend(ax_f.legend(fontsize=6.5, title="", loc="upper right"), fontsize=6.5)
    u.add_footnote(
        ax_f,
        f"Years with < {LICENSE_ADOPTION_MIN_REPOS_PER_YEAR} repos excluded; unclassifiable "
        "licenses (e.g. GitHub's 'Other')\ncount toward 'Any license' only",
        loc="bottom center outside",
    )
    u.shrink_ticks(ax_f, size=8)

    u.save_figure(fig, "figure3_software_development_characteristics", output_dir)
    plt.close(fig)

    # ---- Bonus supplemental: article vs preprint split ----
    fig_supp, axes_supp = plt.subplots(1, 2, figsize=(11, 4.5))
    u.add_panel_label(axes_supp[0], "A")
    u.add_panel_label(axes_supp[1], "B")

    dev_duration_split = dev_duration.filter(
        pl.col("document_type_bucket").is_in(["article", "preprint"])
    )
    # Gold/magenta pair -- distinct from the teal/orange and blue/purple binaries used in the
    # main Figure 3, so this unrelated article/preprint binary isn't pattern-matched onto
    # either. Horizontal, matching main Panel A.
    sns.boxplot(
        data=dev_duration_split.to_pandas(),
        y="period",
        x="duration_years",
        hue="document_type_bucket",
        palette=u.TERTIARY_BINARY_PALETTE,
        ax=axes_supp[0],
        showfliers=False,
    )
    axes_supp[0].set_ylabel("")
    axes_supp[0].set_xlabel("Duration (Years)")
    u.style_legend(axes_supp[0].legend(fontsize=7, title=""))
    axes_supp[0].axvline(0, color="#888888", linewidth=0.8, linestyle=":")
    # Same 2nd-98th percentile x-limit cap as main Figure 3 Panel A.
    u.cap_ylim_to_quantiles(
        axes_supp[0], dev_duration_split.to_pandas()["duration_years"], axis="x"
    )
    u.shrink_ticks(axes_supp[0], size=9)

    fwci_dist_split = fwci_dist.join(
        df.select("document_id", "document_type_bucket").unique(subset="document_id"),
        on="document_id",
        how="left",
    ).filter(pl.col("document_type_bucket").is_in(["article", "preprint"]))
    # Stepped histograms alone leave the overlapping article/preprint peaks hard to
    # distinguish -- low histogram alpha for context, with a KDE line per series on top.
    sns.histplot(
        data=fwci_dist_split.to_pandas(),
        x="document_raw_fwci",
        hue="document_type_bucket",
        palette=u.TERTIARY_BINARY_PALETTE,
        ax=axes_supp[1],
        log_scale=True,
        common_norm=False,
        stat="density",
        element="step",
        fill=True,
        alpha=0.3,
        legend=False,
    )
    sns.kdeplot(
        data=fwci_dist_split.to_pandas(),
        x="document_raw_fwci",
        hue="document_type_bucket",
        palette=u.TERTIARY_BINARY_PALETTE,
        ax=axes_supp[1],
        log_scale=True,
        common_norm=False,
        linewidth=2,
    )
    axes_supp[1].set_xlabel("OpenAlex FWCI (log scale)")
    axes_supp[1].set_ylabel("Density")
    if axes_supp[1].get_legend() is not None:
        axes_supp[1].get_legend().set_title("")
        u.style_legend(axes_supp[1].get_legend())
    u.shrink_ticks(axes_supp[1], size=9)

    evaplot.adjust_layout(fig_supp)
    u.save_figure(fig_supp, "figure3_supplemental_article_vs_preprint", output_dir)
    plt.close(fig_supp)


###############################################################################
# Figure 4 -- software mention rate by field, over time

# A lower floor lets single-digit-n field-year cells through, producing single-year
# percentage spikes that are statistical noise; 30 pushes every plotted field's effective
# start year to where cell sizes make the rate stable, without a hard-coded cutoff year.
MIN_PAIRS_PER_CELL = 30


@app.command()
def figure_4_mention_rate_by_field_and_year(
    output_dir: Path = OUTPUT_DIR,
    cutoff: float = 85.0,
    min_pairs_per_cell: int = MIN_PAIRS_PER_CELL,
    top_n_fields_plotted: int = 6,
) -> None:
    """
    Build Figure 4: rate at which imported software is also explicitly mentioned in the
    paper's text, by field and publication year. Uses
    `align_software_names(method="global_min_diff")` per document-repository pair (two views
    at a time: imports vs. mentions), with the import name always taken as canonical.
    """
    evaplot.set_style("evaplot_rc")
    df = u.load_filtered_pairs(top_n_fields=10)

    print("\nLoading repository_import and document_software_mention from HuggingFace...")
    imports = u.load_table("repository_import")
    mentions = u.load_table("document_software_mention")
    print(f"  repository_import: {len(imports):,} rows")
    print(f"  document_software_mention: {len(mentions):,} rows")

    # Only imports/dependency name is ever canonical -- align per pair with imports as items_a.
    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    eligible = df.filter(pl.col("repository_id").is_in(repo_with_import))
    print(
        f"After restricting to pairs whose repository has >=1 import: "
        f"{eligible.height:,} of {df.height:,} pairs remain"
    )

    imports_by_repo: dict[int, list[str]] = {
        rid[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for rid, grp in imports.group_by("repository_id")
    }
    mentions_by_doc: dict[int, list[str]] = {
        did[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for did, grp in mentions.group_by("document_id")
    }

    rows = []
    for row in eligible.select(
        "document_id",
        "repository_id",
        "document_field_name_pruned",
        "document_publication_year",
        "repository_primary_language",
    ).iter_rows(named=True):
        repo_id = row["repository_id"]
        doc_id = row["document_id"]
        pair_imports = imports_by_repo.get(repo_id, [])
        if not pair_imports:
            continue
        pair_mentions = mentions_by_doc.get(doc_id, [])

        matches = align_software_names(
            items_a=pair_imports,
            items_b=pair_mentions,
            source_a="import",
            source_b="mention",
            cutoff=cutoff,
            method="global_min_diff",
        )
        n_total_imports = len(set(pair_imports))
        n_matched = len({m.normalized_item_one for m in matches})
        rows.append(
            {
                "document_field_name_pruned": row["document_field_name_pruned"],
                "document_publication_year": row["document_publication_year"],
                "repository_primary_language": row["repository_primary_language"],
                "has_any_mention": bool(pair_mentions),
                "n_total_imports": n_total_imports,
                "n_matched_imports": n_matched,
            }
        )

    pair_rates = pl.DataFrame(rows)
    print(f"\nProcessed {pair_rates.height:,} pairs with >=1 import through alignment.")

    field_year_agg = (
        pair_rates.group_by(["document_field_name_pruned", "document_publication_year"])
        .agg(
            pl.len().alias("n_pairs"),
            pl.sum("n_total_imports").alias("total_imports"),
            pl.sum("n_matched_imports").alias("matched_imports"),
        )
        .with_columns(
            (100 * pl.col("matched_imports") / pl.col("total_imports")).alias(
                "mention_rate_pct"
            )
        )
        .sort(["document_field_name_pruned", "document_publication_year"])
    )
    u.save_table(field_year_agg, "figure4_mention_rate_by_field_year_full", output_dir)

    # Software-mention extraction lags behind the most recent publication years: zero or
    # sharply depressed rates despite large import volume are the signature of
    # partially-populated extraction, not a real behavioral shift. Walk backward from the most
    # recent year, dropping any year whose overall rate falls below 40% of the next-older
    # (more mature) year's rate, until the series stabilizes.
    yearly_totals = (
        field_year_agg.group_by("document_publication_year")
        .agg(pl.sum("matched_imports").alias("matched"), pl.sum("total_imports").alias("total"))
        .with_columns((pl.col("matched") / pl.col("total")).alias("rate"))
        .sort("document_publication_year", descending=True)
    )
    yearly_rows = yearly_totals.to_dicts()
    stale_years: list[int] = []
    idx = 0
    while idx < len(yearly_rows) and yearly_rows[idx]["matched"] == 0:
        stale_years.append(yearly_rows[idx]["document_publication_year"])
        idx += 1
    while idx < len(yearly_rows) - 1:
        this_rate = yearly_rows[idx]["rate"]
        prior_rate = yearly_rows[idx + 1]["rate"]
        if prior_rate > 0 and this_rate < 0.4 * prior_rate:
            stale_years.append(yearly_rows[idx]["document_publication_year"])
            idx += 1
        else:
            break
    max_plot_year = yearly_rows[idx]["document_publication_year"]
    if stale_years:
        print(
            f"\nYears {sorted(stale_years)} dropped (zero or sharply depressed mention rates; "
            f"extraction not caught up). Capping the plotted year range at {max_plot_year}."
        )
    field_year_agg = field_year_agg.filter(pl.col("document_publication_year") <= max_plot_year)

    n_before_floor = field_year_agg.height
    plotted = field_year_agg.filter(pl.col("n_pairs") >= min_pairs_per_cell)
    print(
        f"Applying minimum-observation floor (n_pairs >= {min_pairs_per_cell} per "
        f"field-year cell): {plotted.height:,} of {n_before_floor:,} cells remain"
    )

    top_fields_by_local_count = (
        eligible.get_column("document_field_name_pruned")
        .value_counts(sort=True)
        .filter(pl.col("document_field_name_pruned") != "Other")
        .head(top_n_fields_plotted)
        .get_column("document_field_name_pruned")
        .to_list()
    )
    # Which fields are plotted is decided by this figure's own eligible-pair counts, but the
    # order (legend, line style/marker assignment) follows Figure 2 Panel A's global
    # prevalence order for continuity across the figure set.
    canonical_field_order = (
        df.get_column("document_field_name_pruned")
        .value_counts(sort=True)
        .filter(pl.col("document_field_name_pruned") != "Other")
        .get_column("document_field_name_pruned")
        .to_list()
    )
    top_fields = [f for f in canonical_field_order if f in top_fields_by_local_count]
    plotted = plotted.filter(pl.col("document_field_name_pruned").is_in(top_fields))
    u.save_table(plotted, "figure4_mention_rate_by_field_year_plotted", output_dir)

    overall_rate = (
        100
        * pair_rates.get_column("n_matched_imports").sum()
        / pair_rates.get_column("n_total_imports").sum()
    )
    print(f"\nOverall software mention rate across all eligible pairs: {overall_rate:.1f}%")

    # Per-field aggregate rates (post stale-year cap) -- the paper's per-field percentages.
    field_overall = (
        field_year_agg.group_by("document_field_name_pruned")
        .agg(
            pl.sum("n_pairs").alias("n_pairs"),
            pl.sum("total_imports").alias("total_imports"),
            pl.sum("matched_imports").alias("matched_imports"),
        )
        .with_columns(
            (100 * pl.col("matched_imports") / pl.col("total_imports")).alias(
                "mention_rate_pct"
            )
        )
        .sort("mention_rate_pct", descending=True)
    )
    u.save_table(field_overall, "figure4_mention_rate_by_field_overall", output_dir)
    print("\nOverall mention rate by field (post stale-year cap):")
    print(field_overall)

    # Headline summary stats -- overall rate, per-language (Python/R) rates, and the share of
    # pairs mentioning none of their imports. Every rate is reported twice: over ALL pairs
    # with >=1 import (the figure's denominator), and CONDITIONAL on the document having >=1
    # extracted mention -- the two denominators differ by ~4x.
    with_mention = pair_rates.filter(pl.col("has_any_mention"))

    def _rate(frame: pl.DataFrame, lang: str | None = None) -> float:
        sub = (
            frame
            if lang is None
            else frame.filter(pl.col("repository_primary_language") == lang)
        )
        total = sub.get_column("n_total_imports").sum()
        return (
            100 * sub.get_column("n_matched_imports").sum() / total if total else float("nan")
        )

    def _pct_zero(frame: pl.DataFrame) -> float:
        return 100 * frame.filter(pl.col("n_matched_imports") == 0).height / frame.height

    summary = pl.DataFrame(
        {
            "statistic": [
                "overall_mention_rate_pct",
                "python_primary_language_mention_rate_pct",
                "r_primary_language_mention_rate_pct",
                "pct_pairs_with_zero_mentioned_imports",
                "n_pairs_processed",
                "overall_mention_rate_pct_conditional_on_any_mention",
                "python_mention_rate_pct_conditional_on_any_mention",
                "r_mention_rate_pct_conditional_on_any_mention",
                "pct_pairs_zero_mentioned_conditional_on_any_mention",
                "n_pairs_with_any_mention",
                "max_plot_year_after_stale_cap",
            ],
            "value": [
                overall_rate,
                _rate(pair_rates, "Python"),
                _rate(pair_rates, "R"),
                _pct_zero(pair_rates),
                float(pair_rates.height),
                _rate(with_mention),
                _rate(with_mention, "Python"),
                _rate(with_mention, "R"),
                _pct_zero(with_mention),
                float(with_mention.height),
                float(max_plot_year),
            ],
        }
    )
    u.save_table(summary, "figure4_mention_rate_summary", output_dir)
    print(summary)

    # Per-field conditional rates (same stale-year cap as the plotted figure), alongside the
    # all-pairs per-field table saved above.
    field_overall_conditional = (
        with_mention.filter(pl.col("document_publication_year") <= max_plot_year)
        .group_by("document_field_name_pruned")
        .agg(
            pl.len().alias("n_pairs_with_any_mention"),
            pl.sum("n_total_imports").alias("total_imports"),
            pl.sum("n_matched_imports").alias("matched_imports"),
        )
        .with_columns(
            (100 * pl.col("matched_imports") / pl.col("total_imports")).alias(
                "mention_rate_pct_conditional"
            )
        )
        .sort("mention_rate_pct_conditional", descending=True)
    )
    u.save_table(
        field_overall_conditional,
        "figure4_mention_rate_by_field_overall_conditional",
        output_dir,
    )
    print("\nOverall mention rate by field, conditional on >=1 extracted mention:")
    print(field_overall_conditional)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    u.add_panel_label(ax, "A")
    # Anchored on the same green/orange family as Figures 2 and 3 (see `u.field_palette`).
    field_colors = u.field_palette(len(top_fields))
    # Adjacent fields land on near-identical greens in the blended palette, so `style=` gives
    # every field a distinct marker + dash pattern in addition to its color.
    sns.lineplot(
        data=plotted.to_pandas(),
        x="document_publication_year",
        y="mention_rate_pct",
        hue="document_field_name_pruned",
        hue_order=top_fields,
        style="document_field_name_pruned",
        style_order=top_fields,
        palette=field_colors,
        markers=True,
        dashes=True,
        ax=ax,
    )
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Software Mention Rate (%)")
    # Legend outside the axes: no pocket inside these axes stays clear of data at every
    # plotted year.
    leg = ax.legend(
        title="Field",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6.5,
        title_fontsize=7,
        handlelength=1.5,
        labelspacing=0.3,
        borderaxespad=0.6,
    )
    u.style_legend(leg, fontsize=6.5)
    if stale_years:
        u.add_footnote(
            ax,
            f"Years after {max_plot_year} excluded: mention extraction hasn't caught up to these publication years yet (see Methods).",
            loc="bottom center outside",
        )

    evaplot.adjust_layout(fig)
    u.save_figure(fig, "figure4_mention_rate_by_field_and_year", output_dir)
    plt.close(fig)


###############################################################################
# Unit 5 -- Predictors of Software Mentioning (logistic regression)

RARE_SOFTWARE_MIN_COUNT = 3
# Generic software names excluded from the regression population.
GENERIC_SOFTWARE_NAME_EXCLUDE: set[str] = {
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
UNIT5_TOP_N_FIELDS_FOR_CONTROL = 5


def _unit5_trace(
    trace: list[dict] | None, step_name: str, frame: pl.DataFrame, note: str = ""
) -> None:
    """Append a filter-chain checkpoint (row/document/library counts) to the trace list."""
    if trace is None:
        return
    trace.append(
        {
            "step_name": step_name,
            "n_rows": frame.height,
            "n_documents": (
                frame.n_unique("document_id") if "document_id" in frame.columns else None
            ),
            "n_libraries": (
                frame.n_unique("library_name_normalized")
                if "library_name_normalized" in frame.columns
                else None
            ),
            "note": note,
        }
    )


def _remove_rare_and_generic_software(
    df: pl.DataFrame,
    min_count: int = RARE_SOFTWARE_MIN_COUNT,
    trace: list[dict] | None = None,
) -> pl.DataFrame:
    """Exclude libraries with usage count < `min_count` plus a small generic-name exclude
    list.
    """
    n_before = df.height
    df = df.filter(~pl.col("library_name_normalized").is_in(GENERIC_SOFTWARE_NAME_EXCLUDE))
    print(
        f"Excluding generic software names {sorted(GENERIC_SOFTWARE_NAME_EXCLUDE)}: "
        f"{df.height:,} of {n_before:,} rows remain"
    )
    _unit5_trace(trace, "after_generic_name_exclusion", df)

    n_before_rare = df.height
    usage_counts = df.group_by("library_name_normalized").agg(pl.len().alias("usage_count"))
    non_rare = usage_counts.filter(pl.col("usage_count") >= min_count).get_column(
        "library_name_normalized"
    )
    df = df.filter(pl.col("library_name_normalized").is_in(non_rare))
    print(
        f"Excluding rare software (usage_count < {min_count}): "
        f"{df.height:,} of {n_before_rare:,} rows remain"
    )
    _unit5_trace(
        trace,
        "after_rare_software_exclusion",
        df,
        note=(
            f"FLAG: floor of {min_count} mirrors the pnas reference design; "
            f"removed {n_before_rare - df.height:,} rows -- loosen only if judged worth it"
        ),
    )
    return df


def _trim_extreme_usage_pairs(
    df: pl.DataFrame, upper_q: float = 0.99, trace: list[dict] | None = None
) -> pl.DataFrame:
    """Drop entire (document, repository) pairs whose per-pair library count exceeds the
    `upper_q` percentile.
    """
    pair_counts = df.group_by(["document_id", "repository_id"]).agg(
        pl.len().alias("n_libraries")
    )
    threshold = pair_counts.get_column("n_libraries").quantile(upper_q)
    extreme_pairs = pair_counts.filter(pl.col("n_libraries") > threshold).select(
        "document_id", "repository_id"
    )
    n_before = df.height
    df = df.join(extreme_pairs, on=["document_id", "repository_id"], how="anti")
    print(
        f"Trimming pairs with extreme per-pair library usage "
        f"(> {upper_q:.0%} percentile = {threshold:.0f} libraries): "
        f"{df.height:,} of {n_before:,} rows remain"
    )
    _unit5_trace(
        trace,
        "after_extreme_usage_pair_trim",
        df,
        note=(
            f"FLAG: >p{upper_q * 100:.0f} trim mirrors the pnas reference design; "
            f"removed {n_before - df.height:,} rows"
        ),
    )
    return df


def _fit_and_cluster(formula: str, data, doc_groups: np.ndarray, lib_groups: np.ndarray):
    """Fit a logit model, then replace its covariance with the two-way (document x library)
    cluster-robust covariance (Cameron-Gelbach-Miller estimator) via
    `statsmodels.stats.sandwich_covariance.cov_cluster_2groups` -- fit first, then swap in
    the two-way covariance for SEs/p-values/CIs rather than passing a `cov_type=` to `.fit()`.
    """
    model = smf.logit(formula, data=data).fit(disp=0, maxiter=1000)
    cov_both, _cov_doc, _cov_lib = cov_cluster_2groups(
        model, doc_groups, lib_groups, use_correction=True
    )
    se = np.sqrt(np.diag(cov_both))
    z = model.params.to_numpy() / se
    pvals = 2 * (1 - norm.cdf(np.abs(z)))
    ci_lo = model.params.to_numpy() - 1.96 * se
    ci_hi = model.params.to_numpy() + 1.96 * se
    return model, se, pvals, ci_lo, ci_hi


# Figure 4's saved stale-cap year (`max_plot_year_after_stale_cap` in
# figure4_mention_rate_summary.csv) -- mention extraction is absent/partial after this year,
# so later rows are structurally is_mentioned=False (see the mentions-coverage diagnostic).
UNIT5_YEAR_CAP = 2022


def _unit5_prepare_features(
    long_df: pl.DataFrame,
    doc_meta: pl.DataFrame,
    top_n_fields_for_control: int,
    trace: list[dict] | None,
) -> pl.DataFrame:
    """Join document metadata and derive age/popularity/field-control columns on a
    (document, repository, library) long frame.
    """
    model_cols = [
        "is_mentioned",
        "age_years",
        "log_cumulative_imports",
        "document_field_name_pruned",
        "document_type_bucket",
        "document_publication_year",
    ]
    long_df = long_df.join(doc_meta, on="document_id", how="left")

    # ---- Age: years since a library's first-ever appearance in the corpus ----
    first_appearance = long_df.group_by("library_name_normalized").agg(
        pl.min("document_publication_year").alias("first_appearance_year")
    )
    long_df = long_df.join(first_appearance, on="library_name_normalized", how="left")
    long_df = long_df.with_columns(
        (pl.col("document_publication_year") - pl.col("first_appearance_year")).alias(
            "age_years"
        )
    )

    # ---- Popularity: log cumulative imports through the paper's own publication year ----
    # Popularity-at-time-of-publication, not lifetime popularity -- lifetime popularity uses
    # post-publication information to explain the paper's own behavior.
    per_lib_year = (
        long_df.group_by(["library_name_normalized", "document_publication_year"])
        .agg(pl.len().alias("n_in_year"))
        .sort(["library_name_normalized", "document_publication_year"])
        .with_columns(
            pl.col("n_in_year")
            .cum_sum()
            .over("library_name_normalized")
            .alias("cumulative_imports_at_year")
        )
    )
    long_df = long_df.join(
        per_lib_year.select(
            "library_name_normalized", "document_publication_year", "cumulative_imports_at_year"
        ),
        on=["library_name_normalized", "document_publication_year"],
        how="left",
    ).with_columns(
        pl.col("cumulative_imports_at_year")
        .cast(pl.Float64)
        .log()
        .alias("log_cumulative_imports")
    )

    # ---- Field-variable collapse: top-N most common fields + "Other" catch-all ----
    # Computed on unique documents (doc_meta), not the (document, library) long frame -- the
    # long frame has many rows per document, which would over-weight prolific-import documents
    # in the field ranking.
    top_fields = (
        doc_meta.get_column("document_field_name")
        .value_counts(sort=True)
        .head(top_n_fields_for_control)
        .get_column("document_field_name")
        .to_list()
    )
    long_df = long_df.with_columns(
        pl.when(pl.col("document_field_name").is_in(top_fields))
        .then(pl.col("document_field_name"))
        .otherwise(pl.lit("Other"))
        .alias("document_field_name_pruned")
    )

    if trace is not None:
        n_any_null = long_df.select(
            pl.any_horizontal([pl.col(c).is_null() for c in model_cols]).sum()
        ).item()
        _unit5_trace(
            trace,
            "after_doc_metadata_join",
            long_df,
            note=f"{n_any_null:,} rows carry >=1 null across the full model-column set",
        )
    return long_df


@app.command()
def unit5_predictors_of_software_mentioning(
    output_dir: Path = OUTPUT_DIR,
    cutoff: float = 85.0,
    top_n_fields_for_control: int = UNIT5_TOP_N_FIELDS_FOR_CONTROL,
    year_cap: int = UNIT5_YEAR_CAP,
) -> None:
    """
    Fit logistic regressions modeling whether an imported library is explicitly mentioned.
    Rows are at the (document, library) level -- the "why" companion to Figure 4's
    "how often." Age = years since a library's first-ever corpus appearance; popularity = log
    cumulative imports through the paper's own publication year. Fits four specifications
    (age_only, popularity_only, raw, controlled) with two-way (document x library)
    cluster-robust SEs, across a labeled grid of variants.

      - alignment_variant: grouped_hungarian (per-pair one-to-one assignment) vs. independent
        (each import scored against every mention name independently).
      - denominator_variant: all_pairs_with_imports vs. conditional_on_any_mention (documents
        with >=1 extracted mention only).
      - year_cap_applied: with and without Figure 4's stale-year cap (default 2022) -- rows
        after the cap are structurally is_mentioned=False because mention extraction hasn't
        covered those publication years.

    Nulls are dropped per-spec on only the columns each spec actually uses, so simpler specs
    keep documents with e.g. no OpenAlex topic. Also writes a filter-chain trace
    (unit5_filter_chain_row_counts.csv) recording data loss at every step.
    """
    df = u.load_filtered_pairs(top_n_fields=10)

    print("\nLoading repository_import and document_software_mention from HuggingFace...")
    imports = u.load_table("repository_import")
    mentions = u.load_table("document_software_mention")
    print(f"  repository_import: {len(imports):,} rows")
    print(f"  document_software_mention: {len(mentions):,} rows")

    docs_with_mention = set(mentions.get_column("document_id").unique().to_list())

    trace: list[dict] = []
    _unit5_trace(trace, "standard_filtered_pairs", df)
    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    _unit5_trace(
        trace,
        "pairs_with_gte1_import",
        df.filter(pl.col("repository_id").is_in(repo_with_import)),
    )

    doc_meta = df.select(
        "document_id",
        "document_publication_year",
        "document_field_name",
        "document_type_bucket",
    ).unique(subset="document_id", keep="first")

    specs: dict[str, tuple[str, list[str], list[str]]] = {
        "age_only": ("is_mentioned ~ age_years", ["age_years"], ["age_years"]),
        "popularity_only": (
            "is_mentioned ~ log_cumulative_imports",
            ["log_cumulative_imports"],
            ["log_cumulative_imports"],
        ),
        "raw": (
            "is_mentioned ~ age_years + log_cumulative_imports",
            ["age_years", "log_cumulative_imports"],
            ["age_years", "log_cumulative_imports"],
        ),
        "controlled": (
            "is_mentioned ~ age_years + log_cumulative_imports"
            " + C(document_field_name_pruned) + C(document_type_bucket)"
            " + document_publication_year",
            ["age_years", "log_cumulative_imports"],
            [
                "age_years",
                "log_cumulative_imports",
                "document_field_name_pruned",
                "document_type_bucket",
                "document_publication_year",
            ],
        ),
    }

    summary_rows = []
    for alignment in ("grouped_hungarian", "independent"):
        # Trace only the first (original) alignment's chain -- the second follows the same
        # filters and would just duplicate the counts with slightly different is_mentioned.
        variant_trace = trace if alignment == "grouped_hungarian" else None
        long_df = u.build_import_mention_pair_library_frame(
            df, imports, mentions, cutoff=cutoff, alignment=alignment
        )
        _unit5_trace(trace if alignment == "grouped_hungarian" else None, "long_frame", long_df)
        long_df = _remove_rare_and_generic_software(long_df, trace=variant_trace)
        long_df = _trim_extreme_usage_pairs(long_df, trace=variant_trace)
        long_df = _unit5_prepare_features(
            long_df, doc_meta, top_n_fields_for_control, trace=variant_trace
        )

        for denominator in ("all_pairs_with_imports", "conditional_on_any_mention"):
            den_df = (
                long_df
                if denominator == "all_pairs_with_imports"
                else long_df.filter(pl.col("document_id").is_in(docs_with_mention))
            )
            for cap_applied in (True, False):
                grid_df = (
                    den_df.filter(pl.col("document_publication_year") <= year_cap)
                    if cap_applied
                    else den_df
                )
                print(
                    f"\n=== Unit 5 grid cell: alignment={alignment}, "
                    f"denominator={denominator}, year_cap_applied={cap_applied} "
                    f"({grid_df.height:,} rows before per-spec drop_nulls) ==="
                )
                for name, (formula, key_predictors, spec_cols) in specs.items():
                    # Per-spec drop_nulls: only the columns this spec actually uses, so
                    # simpler specs keep more data.
                    model_df = (
                        grid_df.select(
                            "is_mentioned",
                            "document_id",
                            "library_name_normalized",
                            *spec_cols,
                        )
                        .with_columns(pl.col("is_mentioned").cast(pl.Int8))
                        .drop_nulls()
                    )
                    if (
                        alignment == "grouped_hungarian"
                        and denominator == "all_pairs_with_imports"
                        and not cap_applied
                    ):
                        _unit5_trace(
                            trace,
                            f"after_per_spec_drop_nulls[{name}]",
                            model_df,
                            note="per-spec drop_nulls (uncapped, all-pairs, grouped frame)",
                        )
                    regression_pd = model_df.to_pandas()
                    if name == "raw":
                        corr_r, corr_p = pearsonr(
                            regression_pd["age_years"],
                            regression_pd["log_cumulative_imports"],
                        )
                        print(
                            f"Pearson r(age_years, log_cumulative_imports) = {corr_r:.4f}, "
                            f"p = {corr_p:.2e}"
                        )
                    # `cov_cluster_2groups` needs plain numeric group arrays -- library names
                    # are integer-coded (factorized) purely for this clustering step.
                    doc_groups = regression_pd["document_id"].to_numpy()
                    lib_groups = (
                        regression_pd["library_name_normalized"]
                        .astype("category")
                        .cat.codes.to_numpy()
                    )
                    model, se, pvals, ci_lo, ci_hi = _fit_and_cluster(
                        formula, regression_pd, doc_groups, lib_groups
                    )
                    print(f"[{name}] N={int(model.nobs):,}")
                    for predictor in key_predictors:
                        idx = list(model.params.index).index(predictor)
                        coef = model.params.iloc[idx]
                        odds_pct_per_unit = (np.exp(coef) - 1) * 100
                        print(
                            f"  {predictor}: coef={coef:.4f}, SE={se[idx]:.4f}, "
                            f"p={pvals[idx]:.4g}, 95% CI=({ci_lo[idx]:.4f}, {ci_hi[idx]:.4f}), "
                            f"odds change per unit = {odds_pct_per_unit:+.1f}%"
                        )
                        summary_rows.append(
                            {
                                "alignment_variant": alignment,
                                "denominator_variant": denominator,
                                "year_cap_applied": cap_applied,
                                "year_cap": year_cap if cap_applied else None,
                                "model_type": name,
                                "n_obs": int(model.nobs),
                                "predictor": predictor,
                                "coefficient": coef,
                                "std_err_2way_cluster": se[idx],
                                "p_value_2way_cluster": pvals[idx],
                                "ci_lower_2way_cluster": ci_lo[idx],
                                "ci_upper_2way_cluster": ci_hi[idx],
                                "odds_pct_change_per_unit": odds_pct_per_unit,
                            }
                        )

    summary_df = pl.DataFrame(summary_rows)
    u.save_table(summary_df, "unit5_age_popularity_logit_summary", output_dir)

    trace_df = pl.DataFrame(trace)
    u.save_table(trace_df, "unit5_filter_chain_row_counts", output_dir)
    print("\n--- Unit 5 filter-chain trace ---")
    print(trace_df)
    print(
        "\nSensitivity notes: rare-software floor of 3 and p99 usage trim (rows removed "
        "reported above); year-cap variants comparable in the summary CSV.\n"
    )


###############################################################################
# Unit 6 -- Table 1: top software by mentions / imports / dependents

TABLE1_TOP_N_PER_ECOSYSTEM = 15
# Dependency-manifest ecosystems considered part of each language bucket -- restricts the
# import-vs-dependency alignment to the manifests that actually belong to that language, rather
# than e.g. matching an npm frontend-tooling dependency declared inside a Python repo.
TABLE1_DEPENDENCY_ECOSYSTEMS: dict[str, list[str]] = {
    "Python": ["pypi", "conda"],
    "R": ["cran"],
}


@app.command()
def table1_top_software_by_usage(
    output_dir: Path = OUTPUT_DIR,
    cutoff: float = 85.0,
    top_n_per_ecosystem: int = TABLE1_TOP_N_PER_ECOSYSTEM,
) -> None:
    """
    Build Table 1: one row per software, anchored on the import-normalized software name,
    split by ecosystem (Python / R) and ranked within each by import count. Dependency count
    comes from a second, separate import-vs-dependency Hungarian alignment -- same tool,
    cutoff, and two-views-at-a-time constraint as Figure 4's import-vs-mention pass, never
    combined into one three-way alignment. Mention count reuses Figure 4's per-pair
    import-vs-mention alignment logic, aggregated per software instead of per field/year.
    """
    df = u.load_filtered_pairs(top_n_fields=10)

    print(
        "\nLoading repository_import, repository_dependency, document_software_mention from "
        "HuggingFace..."
    )
    imports = u.load_table("repository_import")
    deps = u.load_table("repository_dependency")
    mentions = u.load_table("document_software_mention")
    print(f"  repository_import: {len(imports):,} rows")
    print(f"  repository_dependency: {len(deps):,} rows")
    print(f"  document_software_mention: {len(mentions):,} rows")
    deps = u.clean_dependency_names(deps)

    # ---- Restrict to Python/R ecosystem repositories ----
    # `repository_primary_language` is GitHub's byte-count-based language classification.
    # "Jupyter Notebook"-classified repos are likely Python-ecosystem in practice, but are
    # excluded rather than assumed -- the count is reported below.
    n_before_lang = df.n_unique("repository_id")
    lang_df = df.filter(pl.col("repository_primary_language").is_in(["Python", "R"]))
    n_after_lang = lang_df.n_unique("repository_id")
    n_jupyter = df.filter(pl.col("repository_primary_language") == "Jupyter Notebook").n_unique(
        "repository_id"
    )
    print(
        f"\nRestricting to repositories with primary_language in {{Python, R}}: "
        f"{n_after_lang:,} of {n_before_lang:,} repositories remain "
        f"({n_jupyter:,} additional 'Jupyter Notebook'-primary repos excluded -- likely "
        "Python-heavy in practice but not assumed here)"
    )

    repo_ecosystem: dict[int, str] = dict(
        lang_df.select("repository_id", "repository_primary_language")
        .unique(subset="repository_id")
        .iter_rows()
    )

    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    eligible_repo_ids = set(repo_ecosystem.keys()) & repo_with_import
    print(
        f"Of those, {len(eligible_repo_ids):,} repositories have >=1 extracted import and are "
        "eligible for this table."
    )

    imports_by_repo: dict[int, list[str]] = {
        rid[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for rid, grp in imports.filter(
            pl.col("repository_id").is_in(eligible_repo_ids)
        ).group_by("repository_id")
    }
    mentions_by_doc: dict[int, list[str]] = {
        did[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for did, grp in mentions.group_by("document_id")
    }
    # Pre-grouped per ecosystem's own manifest ecosystems, so the per-repo loop below never has
    # to re-filter the (multi-million-row) full deps table.
    deps_by_repo_by_ecosystem: dict[str, dict[int, list[str]]] = {
        eco: {
            rid[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
            for rid, grp in deps.filter(
                pl.col("repository_id").is_in(eligible_repo_ids)
                & pl.col("ecosystem").is_in(dep_ecosystems)
            ).group_by("repository_id")
        }
        for eco, dep_ecosystems in TABLE1_DEPENDENCY_ECOSYSTEMS.items()
    }

    # ---- Import counts: # of eligible repos importing each canonical (import-normalized) name ----
    # Keyed by (name, ecosystem), not bare name -- the same normalized name can legitimately
    # appear as an import in both a Python-primary and an R-primary repository (e.g. short
    # utility-package names), and this table is explicitly split by ecosystem, so counts must
    # never merge across ecosystems.
    import_counts: dict[tuple[str, str], int] = {}
    for repo_id, names in imports_by_repo.items():
        eco = repo_ecosystem[repo_id]
        for name in set(names):
            key = (name, eco)
            import_counts[key] = import_counts.get(key, 0) + 1

    # ---- Dependency counts: second, separate import-vs-dependency Hungarian alignment, per repo ----
    dependency_counts: dict[tuple[str, str], int] = {}
    n_repo_dep_aligned = 0
    for repo_id, import_names in imports_by_repo.items():
        eco = repo_ecosystem[repo_id]
        dep_names = deps_by_repo_by_ecosystem[eco].get(repo_id, [])
        if not dep_names:
            continue
        matches = align_software_names(
            items_a=import_names,
            items_b=dep_names,
            source_a="import",
            source_b="dependency",
            cutoff=cutoff,
            method="global_min_diff",
        )
        n_repo_dep_aligned += 1
        for name in {m.normalized_item_one for m in matches}:
            key = (name, eco)
            dependency_counts[key] = dependency_counts.get(key, 0) + 1
    print(
        f"\nImport-vs-dependency alignment ran on {n_repo_dep_aligned:,} repositories with both "
        "imports and manifest dependencies in their ecosystem."
    )

    # ---- Mention counts: reuse Figure 4's per-pair import-vs-mention alignment, aggregated per software ----
    mention_doc_sets: dict[tuple[str, str], set[int]] = {}
    all_mention_names_seen: set[str] = set()
    matched_mention_names: set[str] = set()
    eligible_pairs = (
        df.filter(pl.col("repository_id").is_in(eligible_repo_ids))
        .select("document_id", "repository_id")
        .unique()
    )
    for row in eligible_pairs.iter_rows(named=True):
        repo_id, doc_id = row["repository_id"], row["document_id"]
        eco = repo_ecosystem[repo_id]
        import_names = imports_by_repo.get(repo_id, [])
        if not import_names:
            continue
        mention_names = mentions_by_doc.get(doc_id, [])
        all_mention_names_seen.update(mention_names)
        if not mention_names:
            continue
        matches = align_software_names(
            items_a=import_names,
            items_b=mention_names,
            source_a="import",
            source_b="mention",
            cutoff=cutoff,
            method="global_min_diff",
        )
        for m in matches:
            mention_doc_sets.setdefault((m.normalized_item_one, eco), set()).add(doc_id)
            matched_mention_names.add(m.normalized_item_two)
    mention_counts = {k: len(v) for k, v in mention_doc_sets.items()}

    n_unmatched_mentions = len(all_mention_names_seen - matched_mention_names)
    print(
        f"\n{n_unmatched_mentions:,} distinct mentioned-software names never matched any "
        f"import at cutoff={cutoff} (software mentioned but never imported cannot appear in "
        "this import-anchored table)."
    )

    # ---- Assemble table ----
    rows = [
        {
            "ecosystem": eco,
            "software_name": name,
            "import_count": n_import,
            "dependency_count": dependency_counts.get((name, eco), 0),
            "mention_count": mention_counts.get((name, eco), 0),
        }
        for (name, eco), n_import in import_counts.items()
    ]
    table = pl.DataFrame(rows).with_columns(
        (pl.col("mention_count") / pl.col("import_count")).alias("mentions_per_import"),
        (pl.col("dependency_count") / pl.col("import_count")).alias("dependents_per_import"),
    )

    top_tables = []
    for eco in ["Python", "R"]:
        eco_table = (
            table.filter(pl.col("ecosystem") == eco)
            .sort("import_count", descending=True)
            .head(top_n_per_ecosystem)
        )
        top_tables.append(eco_table)
        print(f"\nTop {top_n_per_ecosystem} {eco} software by import count:")
        print(eco_table)

    final_table = pl.concat(top_tables)
    u.save_table(final_table, "table1_top_software_by_usage", output_dir)


###############################################################################
# Unit 7 -- grouped statistic: per-source article-repository pair counts

SEED_SOURCE_NAMES: list[str] = ["joss", "plos", "pwc", "softcite_2025", "softwarex"]


@app.command()
def seed_source_pair_counts(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Fill the five `[N]` seed-pair-count placeholders in the manuscript (lines 101, 103,
    105 x2, 107, 125) -- all from one group-by of `document_repository_link` by source,
    restricted to seed rows (`iteration IS NULL`). Reports both raw (unfiltered) counts and
    standard-filtered counts, since the filtered-pairs table is what every other figure/table
    in this paper is built from.
    """
    print("Loading document_repository_link and dataset_source from HuggingFace...")
    raw_links = u.load_table("document_repository_link")
    dataset_sources = u.load_table("dataset_source")
    raw_links = raw_links.join(
        dataset_sources.select(
            pl.col("id").alias("dataset_source_id"), pl.col("name").alias("dataset_source_name")
        ),
        on="dataset_source_id",
        how="left",
    )

    # ---- Raw (unfiltered) seed-source counts, iteration IS NULL ----
    raw_seed = raw_links.filter(
        pl.col("iteration").is_null() & pl.col("dataset_source_name").is_in(SEED_SOURCE_NAMES)
    )
    raw_counts = (
        raw_seed.group_by("dataset_source_name")
        .agg(pl.len().alias("n_pairs"))
        .sort("dataset_source_name")
    )
    print("\n--- Raw (unfiltered) seed-source pair counts (iteration IS NULL) ---")
    print(raw_counts)
    raw_total = int(raw_counts.get_column("n_pairs").sum())
    print(f"Raw total across the five seed sources: {raw_total:,}")

    # ---- Standard-filtered seed-source counts ----
    filtered = u.load_filtered_pairs()
    filtered_seed = filtered.filter(
        pl.col("link_processing_iteration").is_null()
        & pl.col("dataset_source_name").is_in(SEED_SOURCE_NAMES)
    )
    filtered_counts = (
        filtered_seed.group_by("dataset_source_name")
        .agg(pl.len().alias("n_pairs"))
        .sort("dataset_source_name")
    )
    print("\n--- Standard-filtered seed-source pair counts (iteration IS NULL) ---")
    print(filtered_counts)
    filtered_total = int(filtered_counts.get_column("n_pairs").sum())
    print(f"Filtered total across the five seed sources: {filtered_total:,}")

    u.save_table(
        raw_counts.rename({"n_pairs": "n_pairs_raw"}),
        "unit7_seed_source_pair_counts_raw",
        output_dir,
    )
    u.save_table(
        filtered_counts.rename({"n_pairs": "n_pairs_filtered"}),
        "unit7_seed_source_pair_counts_filtered",
        output_dir,
    )

    def _get(counts: pl.DataFrame, source: str) -> int:
        row = counts.filter(pl.col("dataset_source_name") == source)
        return int(row.get_column("n_pairs")[0]) if row.height else 0

    print("\n--- Manuscript placeholder fills (raw / filtered) ---")
    print(f"Line 101 (combined total, five seed sources): {raw_total:,} / {filtered_total:,}")
    print(
        f"Line 103 (PwC author-provided-only seed pairs): "
        f"{_get(raw_counts, 'pwc'):,} / {_get(filtered_counts, 'pwc'):,}"
    )
    print(
        f"Line 105 (JOSS pairs): {_get(raw_counts, 'joss'):,} / {_get(filtered_counts, 'joss'):,}"
    )
    print(
        f"Line 105 (SoftwareX pairs): "
        f"{_get(raw_counts, 'softwarex'):,} / {_get(filtered_counts, 'softwarex'):,}"
    )
    print(
        f"Line 107 (PLOS pairs): {_get(raw_counts, 'plos'):,} / {_get(filtered_counts, 'plos'):,}"
    )
    print(
        f"Line 125 (SoftCite-2025 rows classified as 'match', seeded pairs): "
        f"{_get(raw_counts, 'softcite_2025'):,} / {_get(filtered_counts, 'softcite_2025'):,}"
    )
    print("---------------------------------------------------------\n")


###############################################################################
# Unit 8 -- statistic: median repository contributor count


@app.command()
def median_repository_contributor_count(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Fill line 33's `X%` placeholder -- "The median scientific repository has only a single
    contributor (X%)...". Filters at the pair level first (standard filters), derives the
    surviving repository set, then computes per-repository contributor counts from
    `repository_contributor` (repositories with no `repository_contributor` rows at all count
    as 0 contributors, not dropped).
    """
    df = u.load_filtered_pairs()
    filtered_repo_ids = df.get_column("repository_id").unique()
    print(
        f"\nRepositories surviving standard pair-level filtering: {filtered_repo_ids.len():,}"
    )

    print("Loading repository_contributor from HuggingFace...")
    repo_contributors = u.load_table("repository_contributor")
    print(f"  repository_contributor: {len(repo_contributors):,} rows")

    contributor_counts = (
        repo_contributors.filter(pl.col("repository_id").is_in(filtered_repo_ids.implode()))
        .group_by("repository_id")
        .agg(pl.n_unique("developer_account_id").alias("n_contributors"))
    )

    repos_frame = (
        pl.DataFrame({"repository_id": filtered_repo_ids})
        .join(contributor_counts, on="repository_id", how="left")
        .with_columns(pl.col("n_contributors").fill_null(0))
    )

    n_with_zero = repos_frame.filter(pl.col("n_contributors") == 0).height
    print(
        f"Repositories with no repository_contributor rows at all (treated as 0 contributors): "
        f"{n_with_zero:,}"
    )

    median_contributors = repos_frame.get_column("n_contributors").median()
    n_single = repos_frame.filter(pl.col("n_contributors") == 1).height
    pct_single = 100 * n_single / repos_frame.height

    u.save_table(repos_frame, "unit8_repository_contributor_counts", output_dir)

    print("\n--- Median repository contributor count ---")
    print(f"Median contributors per repository: {median_contributors}")
    print(
        f"Repositories with exactly one contributor: {n_single:,} of {repos_frame.height:,} "
        f"({pct_single:.1f}%)"
    )
    print(f"Line 33 placeholder fill: X = {pct_single:.1f}%")
    print("---------------------------------------------\n")

    # ---- Repository development characteristics (line 33's medians + FOOTNOTE 3's tables) ----
    # Line 33 cites median commits, development days, days created before publication, and days
    # last-updated after publication; FOOTNOTE 3 promises these split by document type with
    # additional percentiles. Commit/development metrics are repo-level (dedup to first-seen
    # pair); publication-relative deltas are pair-level.
    pair_char = df.with_columns(
        pl.col("repository_last_pushed_datetime")
        .str.to_datetime(strict=False)
        .alias("_last_pushed"),
        pl.col("document_publication_date_parsed").cast(pl.Datetime("us")).alias("_pub_dt"),
    ).with_columns(
        (pl.col("_pub_dt") - pl.col("repository_creation_datetime_parsed"))
        .dt.total_days()
        .alias("days_created_before_publication"),
        (pl.col("_last_pushed") - pl.col("_pub_dt"))
        .dt.total_days()
        .alias("days_last_push_after_publication"),
        (pl.col("_last_pushed") - pl.col("repository_creation_datetime_parsed"))
        .dt.total_days()
        .alias("development_days_creation_to_last_push"),
        pl.col("repository_commits_count").cast(pl.Float64).alias("commit_count"),
    )
    repo_char = pair_char.unique(subset="repository_id", keep="first").join(
        repos_frame.select("repository_id", "n_contributors"), on="repository_id", how="left"
    )

    metric_specs = [
        ("n_contributors", repo_char),
        ("commit_count", repo_char),
        ("development_days_creation_to_last_push", repo_char),
        ("days_created_before_publication", pair_char),
        ("days_last_push_after_publication", pair_char),
    ]
    summary_rows = []
    for doctype in [None, "article", "preprint", "other"]:
        for metric, frame in metric_specs:
            sub = (
                frame
                if doctype is None
                else frame.filter(pl.col("document_type_bucket") == doctype)
            )
            vals = sub.get_column(metric).drop_nulls()
            if vals.len() == 0:
                continue
            summary_rows.append(
                {
                    "document_type": doctype or "all",
                    "metric": metric,
                    "n": vals.len(),
                    "median": float(vals.median()),
                    "p10": float(vals.quantile(0.10)),
                    "p25": float(vals.quantile(0.25)),
                    "p75": float(vals.quantile(0.75)),
                    "p90": float(vals.quantile(0.90)),
                }
            )
    char_summary = pl.DataFrame(summary_rows)
    u.save_table(char_summary, "unit8_repository_characteristics_summary", output_dir)
    print("Repository development characteristics (line 33 / FOOTNOTE 3):")
    print(char_summary.filter(pl.col("document_type") == "all"))


###############################################################################
# Unit 9 -- statistic: classification model table verification
#
# Model training/eval artifacts stay in `sci-soft-models` -- this reads the three deployed
# models' saved `results.json` files directly from a sibling `sci-soft-models` checkout
# rather than retraining or reimplementing any classifier here. The JSON files are read
# directly (not through the package) because `binary_article_repo_em/data/__init__.py`
# resolves a symlinked local DB path at import time.

SCI_SOFT_MODELS_REPO = Path(__file__).resolve().parents[3] / "sci-soft-models"

# (results.json relative path, manuscript-stated precision/recall/f1, metric key prefix)
CLASSIFICATION_MODELS_TABLE: list[dict] = [
    {
        "name": "Article-Repository Matching Model (ARMM)",
        "results_path": SCI_SOFT_MODELS_REPO
        / "sci_soft_models/binary_article_repo_em/data/files/final-model-training-data-optimized/results.json",
        "stated_precision": 0.975,
        "stated_recall": 0.975,
        "stated_f1": 0.975,
        "metric_prefix": "macro",
    },
    {
        "name": "Researcher-Developer-Account Matching Model",
        "results_path": SCI_SOFT_MODELS_REPO
        / "sci_soft_models/dev_author_em/data/files/final-model-training-data/results.json",
        "stated_precision": 0.938,
        "stated_recall": 0.950,
        "stated_f1": 0.944,
        "metric_prefix": None,  # flat precision/recall/f1 keys, not macro_/binary_-prefixed
        # The paper cites the published Brown/Slaughter/Weber figures (0.938/0.950/0.944);
        # the sci-soft-models artifact currently reports higher numbers. The gap is recorded
        # as a flagged note, not treated as a table error.
        "known_discrepancy_note": (
            "Cited value = published Brown/Slaughter/Weber figures; the sci-soft-models "
            "artifact's current saved eval differs (see 'actual'). Known discrepancy -- "
            "the published/cited value stands (round-4 decision)."
        ),
    },
    {
        "name": "Software Repository Sharing Statement Classifier",
        "results_path": SCI_SOFT_MODELS_REPO
        / "sci_soft_models/software_mentions_repo_clf/data/files/final-model-training-data/results.json",
        "stated_precision": 0.860,
        "stated_recall": 0.812,
        "stated_f1": 0.827,
        "metric_prefix": "macro",
    },
]

MISMATCH_TOLERANCE = 0.005


@app.command()
def classification_models_table_verification(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Verify the numbers in the manuscript's "Table X. Classification models trained and
    utilized..." (line 81) against each deployed model's saved evaluation results in
    `sci-soft-models`, confirming the stated precision/recall/F1 are still accurate.
    """
    print("--- Classification models table verification (Unit 9) ---\n")
    any_mismatch = False
    verification_rows = []
    for entry in CLASSIFICATION_MODELS_TABLE:
        results_path: Path = entry["results_path"]
        print(f"{entry['name']}")
        print(f"  Reading: {results_path}")
        if not results_path.exists():
            print("  MISSING -- results.json not found at this path, cannot verify.\n")
            any_mismatch = True
            continue

        results = json.loads(results_path.read_text())
        prefix = entry["metric_prefix"]
        if prefix is None:
            actual_precision = results["precision"]
            actual_recall = results["recall"]
            actual_f1 = results["f1"]
        else:
            actual_precision = results[f"{prefix}_precision"]
            actual_recall = results[f"{prefix}_recall"]
            actual_f1 = results[f"{prefix}_f1"]

        discrepancy_note = entry.get("known_discrepancy_note", "")
        for label, stated, actual in [
            ("Precision", entry["stated_precision"], actual_precision),
            ("Recall", entry["stated_recall"], actual_recall),
            ("F1", entry["stated_f1"], actual_f1),
        ]:
            delta = actual - stated
            if abs(delta) <= MISMATCH_TOLERANCE:
                status = "OK"
            elif discrepancy_note:
                # Known, decided discrepancy: recorded/flagged, but not a table error.
                status = "FLAGGED_KNOWN_DISCREPANCY"
            else:
                status = "MISMATCH"
                any_mismatch = True
            print(
                f"  {label}: manuscript states {stated:.3f}, actual is {actual:.4f} "
                f"(delta {delta:+.4f}) -- {status}"
            )
            verification_rows.append(
                {
                    "model": entry["name"],
                    "detail_group": "headline",
                    "metric": label,
                    "manuscript_stated": stated,
                    "actual": actual,
                    "delta": delta,
                    "status": status,
                    "note": discrepancy_note if status != "OK" else "",
                }
            )
        print()

    # ---- ARMM detail verification (manuscript lines 207-211): per-field, per-period,
    # per-source, and README-length numbers, from sci-soft-models' single-feature-eval
    # accessors (imported lazily -- see the ARMM diagnostics section header). ----
    from sci_soft_models.binary_article_repo_em import (
        load_performance_by_readme_length,
        load_single_feature_eval,
    )

    detail_specs: list[tuple[str, str, float, float]] = []

    field_eval = load_single_feature_eval("document_topic_primary_field_pruned")
    field_f1 = field_eval.get_column("macro_f1")
    field_min = field_eval.sort("macro_f1").row(0, named=True)
    detail_specs += [
        ("per_field", "mean macro F1 across fields", 0.972, float(field_f1.mean())),
        ("per_field", "SD macro F1 across fields", 0.010, float(field_f1.std())),
        (
            "per_field",
            f"lowest field macro F1 ({field_min['feature_value']})",
            0.951,
            float(field_min["macro_f1"]),
        ),
    ]

    period_eval = load_single_feature_eval("document_publication_date_bin")
    period_f1 = period_eval.get_column("macro_f1")
    period_min = period_eval.sort("macro_f1").row(0, named=True)
    detail_specs += [
        ("per_period", "mean macro F1 across periods", 0.968, float(period_f1.mean())),
        ("per_period", "SD macro F1 across periods", 0.012, float(period_f1.std())),
        (
            "per_period",
            f"lowest period macro F1 ({period_min['feature_value']})",
            0.950,
            float(period_min["macro_f1"]),
        ),
    ]

    source_eval = load_single_feature_eval("dataset_source_name")

    def _source_binary_f1(source_name: str) -> float:
        return float(
            source_eval.filter(pl.col("feature_value") == source_name)
            .get_column("binary_f1")
            .item()
        )

    detail_specs += [
        ("per_source", "SoftCite-2025 binary F1", 0.968, _source_binary_f1("softcite_2025")),
        ("per_source", "JOSS binary F1", 0.998, _source_binary_f1("joss")),
        (
            "per_source",
            "same-author-different-article hard-negative binary F1",
            0.964,
            _source_binary_f1("same-author-different-article-negative"),
        ),
        (
            "per_source",
            "same-contributor-different-repo hard-negative binary F1",
            0.966,
            _source_binary_f1("same-contributor-different-repo-negative"),
        ),
    ]

    readme_perf = load_performance_by_readme_length()

    def _readme_bin_f1(bin_name: str) -> float:
        return float(
            readme_perf.filter(pl.col("repository_readme_length_bin") == bin_name)
            .get_column("macro_f1")
            .item()
        )

    detail_specs += [
        ("readme_length", "1601-3200 chars macro F1", 0.979, _readme_bin_f1("1601-3200")),
        ("readme_length", "<=100 chars macro F1", 0.957, _readme_bin_f1("<=100")),
    ]

    print("ARMM detail verification (lines 207-211):")
    for detail_group, metric, stated, actual in detail_specs:
        delta = actual - stated
        status = "OK" if abs(delta) <= MISMATCH_TOLERANCE else "MISMATCH"
        if status == "MISMATCH":
            any_mismatch = True
        print(
            f"  [{detail_group}] {metric}: manuscript states {stated:.3f}, actual is "
            f"{actual:.4f} (delta {delta:+.4f}) -- {status}"
        )
        verification_rows.append(
            {
                "model": "Article-Repository Matching Model (ARMM)",
                "detail_group": detail_group,
                "metric": metric,
                "manuscript_stated": stated,
                "actual": actual,
                "delta": delta,
                "status": status,
                "note": "",
            }
        )
    print()

    if verification_rows:
        u.save_table(
            pl.DataFrame(verification_rows),
            "unit9_classification_models_verification",
            output_dir,
        )
    if any_mismatch:
        print(
            "CAVEAT: at least one model's manuscript-stated figure is outside the "
            f"+/-{MISMATCH_TOLERANCE} tolerance of its current saved eval results -- see MISMATCH "
            "lines above. Table needs updating before submission.\n"
        )
    else:
        print("All manuscript-stated figures are within tolerance of current eval results.\n")
    print("-----------------------------------------------------------\n")


###############################################################################
# Unit 10 -- ARMM model diagnostics (confusion matrices, README-length performance)
#
# Model training/eval artifacts stay in `sci-soft-models` -- this imports the held-out test
# predictions and eval tables through `sci_soft_models.binary_article_repo_em`'s accessor
# functions rather than retraining or reimplementing the model. Imports are done lazily
# inside the command rather than at module level: `binary_article_repo_em`'s data module
# resolves a symlinked local DB path at import time that only exists in a full
# rs-graph-plus-data checkout, so an import failure should only break this one command.


@app.command()
def unit10_armm_model_diagnostics(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Build the ARMM held-out-test diagnostics.

      - the pooled confusion matrix (line 207, `Blues` cmap by standard convention);
      - per-field confusion matrices for the top-8 fields + the held-out "Other"/"Unknown"
        buckets (line 207's promised figure), each panel annotated with n and macro F1;
      - the performance-by-README-length figure (line ~211's placeholder), from
        sci-soft-models' `load_performance_by_readme_length` accessor.
    """
    from sci_soft_models.binary_article_repo_em import (
        load_final_model_test_predictions,
        load_performance_by_readme_length,
    )
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

    evaplot.set_style("evaplot_rc")

    test_preds = load_final_model_test_predictions()
    print(f"Loaded ARMM held-out test predictions: {test_preds.height:,} rows")

    y_true = test_preds.get_column("label").to_list()
    y_pred = test_preds.get_column("predicted_label").to_list()
    labels_order = ["no-match", "match"]

    # ---- Pooled confusion matrix (line 207 cross-reference) ----
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False, values_format=",")
    # evaplot_rc's default gridlines cut straight through the cell-value text on a heatmap-style
    # plot like this one (readable on line/bar charts, not here) -- disable for this figure only.
    ax_cm.grid(False)
    ax_cm.set_title("ARMM Held-Out Test Set Confusion Matrix")
    u.shrink_ticks(ax_cm, size=9)

    evaplot.adjust_layout(fig_cm)
    u.save_figure(fig_cm, "unit10_armm_confusion_matrix", output_dir)
    plt.close(fig_cm)

    # ---- Per-field confusion matrices (3x3 grid: top-8 fields + Other/Unknown buckets) ----
    field_col = "document_topic_primary_field_pruned"
    field_order = (
        test_preds.get_column(field_col).value_counts(sort=True).get_column(field_col).to_list()
    )
    n_fields = len(field_order)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    # Taller rows + explicit hspace so the grid's rows aren't cramped.
    fig_ff, axes_ff = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.6 * n_rows),
        gridspec_kw={"hspace": 0.45},
    )
    axes_flat = axes_ff.flatten()
    for i, field in enumerate(field_order):
        grp = test_preds.filter(pl.col(field_col) == field)
        g_true = grp.get_column("label").to_list()
        g_pred = grp.get_column("predicted_label").to_list()
        g_cm = confusion_matrix(g_true, g_pred, labels=labels_order)
        g_macro_f1 = f1_score(g_true, g_pred, labels=labels_order, average="macro")
        g_disp = ConfusionMatrixDisplay(confusion_matrix=g_cm, display_labels=labels_order)
        g_disp.plot(ax=axes_flat[i], cmap="Blues", colorbar=False, values_format=",")
        axes_flat[i].grid(False)
        axes_flat[i].set_title(
            f"{field}\n(n={grp.height:,}, macro F1={g_macro_f1:.3f})", fontsize=9
        )
        u.shrink_ticks(axes_flat[i], size=8)
        print(f"  {field}: n={grp.height:,}, macro F1={g_macro_f1:.4f}")
    for j in range(n_fields, len(axes_flat)):
        axes_flat[j].set_axis_off()
    fig_ff.suptitle("ARMM Held-Out Test Confusion Matrices by Field", y=1.005)
    # Row spacing passed through adjust_layout so tight_layout doesn't collapse it again.
    evaplot.adjust_layout(fig_ff, hspace=0.45)
    fig_ff.subplots_adjust(hspace=0.45)
    u.save_figure(fig_ff, "unit10_armm_confusion_matrix_by_field", output_dir)
    plt.close(fig_ff)

    # ---- Performance by README length (line ~211's placeholder figure) ----
    readme_perf = load_performance_by_readme_length()
    bin_order = ["<=100", "101-200", "201-400", "401-800", "801-1600", "1601-3200", ">3200"]
    readme_perf = (
        readme_perf.with_columns(
            pl.col("repository_readme_length_bin")
            .replace_strict({b: i for i, b in enumerate(bin_order)}, return_dtype=pl.Int64)
            .alias("_order")
        )
        .sort("_order")
        .drop("_order")
    )
    print("\nARMM performance by README length bin:")
    print(readme_perf.select("repository_readme_length_bin", "macro_f1", "support"))

    fig_rl, ax_rl = plt.subplots(figsize=(8, 5))
    point_color = evaplot.set_cat_palette(n=1)[0]
    ax_rl.plot(
        readme_perf.get_column("repository_readme_length_bin").to_list(),
        readme_perf.get_column("macro_f1").to_numpy(),
        marker="o",
        color=point_color,
        linewidth=1.6,
        markersize=7,
        markeredgecolor="black",
        markeredgewidth=0.6,
    )
    for x, (f1v, supp) in enumerate(readme_perf.select("macro_f1", "support").iter_rows()):
        ax_rl.annotate(
            f"n={supp:,}",
            (x, f1v),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=6.5,
        )
    ax_rl.set_xlabel("Repository README Length (Characters)")
    ax_rl.set_ylabel("Macro F1")
    # Zero-based y-axis: a tight autoscale makes 0.96 vs. 0.98 look like a large gap; full
    # scale shows performance is uniformly high.
    ax_rl.set_ylim(0, 1.02)
    u.shrink_ticks(ax_rl, size=9)
    evaplot.adjust_layout(fig_rl)
    u.save_figure(fig_rl, "unit10_armm_performance_by_readme_length", output_dir)
    plt.close(fig_rl)

    # ---- Summary performance-metric tables: three breakdowns -- by publication-year bin,
    # by corresponding-author country, and by first-author country -- with missingness
    # buckets disclosed as their own rows. ----
    year_bin_labels = {
        "pub-year-bin-01": "< 2016",
        "pub-year-bin-02": "2016-2020",
        "pub-year-bin-03": "2021-2024",
        "pub-year-bin-04": ">= 2025",
    }
    table_specs = [
        (
            "document_publication_date_bin",
            "unit10_armm_performance_by_publication_year_bin",
        ),
        (
            "document_corresponding_author_institution_country_code_pruned",
            "unit10_armm_performance_by_corresponding_author_country",
        ),
        (
            "document_first_author_institution_country_code_pruned",
            "unit10_armm_performance_by_first_author_country",
        ),
    ]
    from sci_soft_models.binary_article_repo_em import load_single_feature_eval

    for feature, stem in table_specs:
        ev = load_single_feature_eval(feature)
        total_support = int(ev.get_column("support").sum())
        table = ev.select(
            pl.col("feature_value").alias("group"),
            pl.col("support").alias("n_test_rows"),
            (100 * pl.col("support") / total_support).round(1).alias("pct_of_test_rows"),
            "binary_precision",
            "binary_recall",
            "binary_f1",
            "macro_f1",
        )
        if feature == "document_publication_date_bin":
            table = table.sort("group").with_columns(pl.col("group").replace(year_bin_labels))
        else:
            # Fix the source data's spelling; keep missingness buckets as disclosed rows.
            table = table.with_columns(
                pl.col("group").replace(
                    {
                        "No affliation": "No affiliation",
                        "Multiple affliations": "Multiple affiliations",
                    }
                )
            ).sort("n_test_rows", descending=True)
        u.save_table(table, stem, output_dir)
        print(f"\n{stem}:")
        print(table)


###############################################################################
# Unit 11 -- date-delta figure (repository creation vs. publication date)


def _date_delta_percentiles(diffs: pl.Series) -> dict[str, float]:
    """Median + 1st/5th/10th/90th/95th/99th percentiles (days) of a date-delta series."""
    return {
        "median": float(diffs.median()),
        "p1": float(diffs.quantile(0.01)),
        "p5": float(diffs.quantile(0.05)),
        "p10": float(diffs.quantile(0.10)),
        "p90": float(diffs.quantile(0.90)),
        "p95": float(diffs.quantile(0.95)),
        "p99": float(diffs.quantile(0.99)),
    }


def _plot_date_delta_panel(
    ax, diffs: pl.Series, title: str | None = None, legend: bool = True
) -> dict[str, float]:
    """Histogram of a date-delta series, clipped to the 1st/99th percentile for readability,
    with vertical marker lines at the median and the 10th/90th percentiles (the window-defining
    percentiles -- p5/p95 stay in the percentile CSV but are no longer plotted ink).
    """
    stats = _date_delta_percentiles(diffs)
    clipped = diffs.filter((diffs >= stats["p1"]) & (diffs <= stats["p99"]))
    sns.histplot(clipped.to_pandas(), ax=ax, bins=60)
    ax.axvline(
        stats["median"],
        color="black",
        linestyle="-",
        linewidth=1.5,
        label=f"Median ({stats['median']:.0f}d)",
    )
    ax.axvline(
        stats["p10"],
        color="blue",
        linestyle="--",
        linewidth=1.2,
        label=f"10th/90th pct. ({stats['p10']:.0f}/{stats['p90']:.0f}d)",
    )
    ax.axvline(stats["p90"], color="blue", linestyle="--", linewidth=1.2)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("Days")
    ax.set_ylabel("Count")
    if legend:
        leg = ax.legend(fontsize=7)
        u.style_legend(leg)
    u.shrink_ticks(ax, size=8)
    return stats


@app.command()
def unit11_date_delta_figure(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Build line 229's date-delta figure -- justifies the 90th/10th-percentile
    repository-creation-vs-publication-date inclusion window used elsewhere in the pipeline.
    Uses strict one-to-one linking, matching
    `notebooks/snowball-sampling-discovery-prep.ipynb`'s dedup (drop every document linked to
    more than one repository and every repository linked to more than one document), which
    isolates the clearest paper-to-repo match at little coverage cost. Mined pairs appear
    only as a sixth panel on the by-source supplemental; the pooled figure and its
    window-defining percentiles stay seed-only (mined pairs were selected *through* the
    window, so pooling them would be circular). The by-document-type supplemental is also
    seed-only.
    """
    evaplot.set_style("evaplot_rc")
    df = u.load_filtered_pairs(top_n_fields=10)

    # ---- Strict one-to-one linking (same dedup as the snowball-sampling prep notebook) ----
    n_before_dedup = df.height
    df = df.unique(subset="document_id", keep="none").unique(
        subset="repository_id", keep="none"
    )
    print(
        f"One-to-one dedup (drop all rows with a duplicated document_id, then all rows with "
        f"a duplicated repository_id): {df.height:,} of {n_before_dedup:,} pairs remain"
    )

    n_all = df.height
    mined_df = df.filter(pl.col("link_processing_iteration").is_not_null())
    df = df.filter(pl.col("link_processing_iteration").is_null())
    print(
        f"Seed pairs (link_processing_iteration IS NULL): {df.height:,}; "
        f"mined pairs (sixth supplemental panel only): {mined_df.height:,} "
        f"of {n_all:,} one-to-one pairs"
    )

    _delta_expr = (
        (
            pl.col("document_publication_date_parsed").cast(pl.Datetime)
            - pl.col("repository_creation_datetime_parsed")
        )
        .dt.total_days()
        .alias("publication_date_creation_date_diff")
    )
    df = df.with_columns(_delta_expr)
    mined_df = mined_df.with_columns(_delta_expr)
    n_before = df.height
    df = df.drop_nulls(subset=["publication_date_creation_date_diff"])
    mined_df = mined_df.drop_nulls(subset=["publication_date_creation_date_diff"])
    print(
        f"Dropped {n_before - df.height:,} seed pairs with unparseable publication/creation "
        f"date; {df.height:,} seed pairs remain for the date-delta analysis"
    )

    all_diffs = df.get_column("publication_date_creation_date_diff")

    # ---- Main figure: pooled, unsplit ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pooled_stats = _plot_date_delta_panel(ax, all_diffs, legend=True)
    ax.set_xlabel("Publication Date - Repository Creation Date (Days)")
    evaplot.adjust_layout(fig)
    u.save_figure(fig, "unit11_date_delta_pooled", output_dir)
    plt.close(fig)

    print("\nPooled date-delta percentiles (days):")
    for k, v in pooled_stats.items():
        print(f"  {k}: {v:.1f}")
    print(
        "Reference (notebook `snowball-sampling-discovery-prep.ipynb`): 90th=556.0, 10th=-73.0. "
        f"This run: 90th={pooled_stats['p90']:.1f}, 10th={pooled_stats['p10']:.1f}."
    )

    percentile_rows = [{"breakdown": "pooled", "group": "all", "n": df.height, **pooled_stats}]

    # ---- Supplemental 1: by seed source, plus mined pairs as a sixth panel ----
    sources = sorted(
        df.get_column("dataset_source_name_canonical").drop_nulls().unique().to_list()
    )
    panel_specs: list[tuple[str, str, pl.Series]] = [
        (
            "seed_source",
            source,
            df.filter(pl.col("dataset_source_name_canonical") == source).get_column(
                "publication_date_creation_date_diff"
            ),
        )
        for source in sources
    ]
    panel_specs.append(
        (
            "mined",
            "Mined (iterative discovery)",
            mined_df.get_column("publication_date_creation_date_diff"),
        )
    )
    fig_src, axes_src = plt.subplots(2, 3, figsize=(16, 9))
    axes_src_flat = axes_src.flatten()
    for i, (ax_i, (breakdown, label, sub_diffs)) in enumerate(
        zip(axes_src_flat, panel_specs, strict=False)
    ):
        sub_stats = _plot_date_delta_panel(
            ax_i, sub_diffs, title=f"{label} (n={sub_diffs.len():,})", legend=(i == 0)
        )
        u.add_panel_label(ax_i, chr(65 + i))
        percentile_rows.append(
            {"breakdown": breakdown, "group": label, "n": sub_diffs.len(), **sub_stats}
        )
    for j in range(len(panel_specs), len(axes_src_flat)):
        axes_src_flat[j].set_axis_off()
    fig_src.suptitle(
        "Publication Date - Repository Creation Date Difference, by Seed Source "
        "(+ Mined Pairs)",
        y=1.02,
    )
    evaplot.adjust_layout(fig_src)
    u.save_figure(fig_src, "unit11_date_delta_supplemental_by_seed_source", output_dir)
    plt.close(fig_src)

    # ---- Supplemental 2: by document type bucket (article / preprint / other) ----
    doctypes = ["article", "preprint", "other"]
    fig_dt, axes_dt = plt.subplots(1, 3, figsize=(15, 5))
    for i, (ax_i, doctype) in enumerate(zip(axes_dt, doctypes, strict=True)):
        sub_diffs = df.filter(pl.col("document_type_bucket") == doctype).get_column(
            "publication_date_creation_date_diff"
        )
        sub_stats = _plot_date_delta_panel(
            ax_i, sub_diffs, title=f"{doctype} (n={sub_diffs.len():,})", legend=(i == 0)
        )
        u.add_panel_label(ax_i, chr(65 + i))
        percentile_rows.append(
            {"breakdown": "document_type", "group": doctype, "n": sub_diffs.len(), **sub_stats}
        )
    fig_dt.suptitle(
        "Publication Date - Repository Creation Date Difference, by Document Type", y=1.02
    )
    evaplot.adjust_layout(fig_dt)
    u.save_figure(fig_dt, "unit11_date_delta_supplemental_by_document_type", output_dir)
    plt.close(fig_dt)

    percentile_table = pl.DataFrame(percentile_rows)
    u.save_table(percentile_table, "unit11_date_delta_percentiles", output_dir)
    print("\nAll date-delta percentiles by breakdown:")
    print(percentile_table)


###############################################################################
# Unit 12 -- mining-rounds table (iterations 1-5)


@app.command()
def mining_rounds_table(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Fill Table X (line 237) and line 254's `X` placeholder. Two halves:
      (a) new article-repository pairs per source/iteration -- a group-by on
          `document_repository_link`'s (dataset_source_id, iteration).
      (b) new researcher-developer-account identity links per iteration -- a structural join
          (not timestamp-based) attributing each identity link to the earliest iteration
          whose document-repository pair could have produced it.
    """
    print("Loading document_repository_link, dataset_source from HuggingFace...")
    raw_links = u.load_table("document_repository_link")
    dataset_sources = u.load_table("dataset_source")
    raw_links = raw_links.join(
        dataset_sources.select(
            pl.col("id").alias("dataset_source_id"), pl.col("name").alias("dataset_source_name")
        ),
        on="dataset_source_id",
        how="left",
    )

    # ---- (a) New article-repository pairs per source/iteration -- simple group-by ----
    pairs_by_source_iteration = (
        raw_links.with_columns(pl.col("iteration").fill_null(-1).alias("iteration_bucket"))
        .group_by(["dataset_source_name", "iteration_bucket"])
        .agg(pl.len().alias("n_pairs"))
        .sort(["iteration_bucket", "dataset_source_name"])
    )
    print("\n--- (a) New article-repository pairs by source and iteration (raw group-by) ---")
    print(pairs_by_source_iteration)

    pairs_by_iteration_total = (
        raw_links.with_columns(pl.col("iteration").fill_null(-1).alias("iteration_bucket"))
        .group_by("iteration_bucket")
        .agg(pl.len().alias("n_pairs"))
        .sort("iteration_bucket")
    )
    print("\nNew article-repository pairs per iteration, totaled across sources:")
    print(pairs_by_iteration_total)

    mining_rounds_total = int(
        pairs_by_iteration_total.filter(pl.col("iteration_bucket") != -1)
        .get_column("n_pairs")
        .sum()
    )
    extended_mining_total = int(
        pairs_by_iteration_total.filter(pl.col("iteration_bucket").is_in([4, 5]))
        .get_column("n_pairs")
        .sum()
    )
    print(
        f"\nTotal new pairs across all mining iterations (1-5), RAW candidates: {mining_rounds_total:,}"
    )
    print(
        f"Line 254 placeholder fill, RAW candidates (Extended Mining Round, iterations 4+5): "
        f"{extended_mining_total:,}"
    )

    u.save_table(
        pairs_by_source_iteration, "unit12a_new_pairs_by_source_and_iteration_raw", output_dir
    )

    # ---- Same group-by, restricted to the standard-filtered pairs table. The raw group-by
    # above counts every candidate row regardless of confidence; most predicted (non-seed)
    # rows do NOT meet 0.9994, so raw per-iteration counts overstate what's retained. The
    # filtered version is what "new pairs added to the dataset" in Table X and line 254
    # actually means.
    filtered_pairs_for_iteration = u.load_filtered_pairs()
    pairs_by_iteration_filtered = (
        filtered_pairs_for_iteration.with_columns(
            pl.col("link_processing_iteration").fill_null(-1).alias("iteration_bucket")
        )
        .group_by("iteration_bucket")
        .agg(pl.len().alias("n_pairs"))
        .sort("iteration_bucket")
    )
    print(
        "\nNew article-repository pairs per iteration, RETAINED (standard-filtered, "
        ">=0.9994-or-NULL confidence + post-2008):"
    )
    print(pairs_by_iteration_filtered)
    mining_rounds_total_filtered = int(
        pairs_by_iteration_filtered.filter(pl.col("iteration_bucket") != -1)
        .get_column("n_pairs")
        .sum()
    )
    extended_mining_total_filtered = int(
        pairs_by_iteration_filtered.filter(pl.col("iteration_bucket").is_in([4, 5]))
        .get_column("n_pairs")
        .sum()
    )
    print(
        f"Total new RETAINED pairs across all mining iterations (1-5): "
        f"{mining_rounds_total_filtered:,}"
    )
    print(
        f"Line 254 placeholder fill, RETAINED (Extended Mining Round, iterations 4+5): "
        f"{extended_mining_total_filtered:,}"
    )
    u.save_table(
        pairs_by_iteration_filtered, "unit12a_new_pairs_by_iteration_filtered", output_dir
    )

    # ---- (b) New researcher-developer-account identity links per iteration -- structural join ----
    print(
        "\nLoading researcher_developer_account_link, document_contributor, "
        "repository_contributor from HuggingFace..."
    )
    rdal = u.load_table("researcher_developer_account_link")
    document_contributors = u.load_table("document_contributor")
    repository_contributors = u.load_table("repository_contributor")
    print(f"  researcher_developer_account_link: {len(rdal):,} rows")
    print(f"  document_contributor: {len(document_contributors):,} rows")
    print(f"  repository_contributor: {len(repository_contributors):,} rows")

    rdal_filtered = rdal.filter(
        pl.col("predictive_model_confidence") >= u.DEFAULT_RDAL_CONFIDENCE_THRESHOLD
    ).select("researcher_id", "developer_account_id")
    print(
        f"After filtering researcher_developer_account_link to confidence >= "
        f"{u.DEFAULT_RDAL_CONFIDENCE_THRESHOLD}: {rdal_filtered.height:,} identity links remain"
    )

    # Reuse the already-loaded standard-filtered pairs table instead of re-running the full
    # HuggingFace load/join/filter pipeline for the same default args.
    drl_filtered = filtered_pairs_for_iteration.select(
        "document_id", "repository_id", "link_processing_iteration"
    )
    print(
        f"Using the standard-filtered article-repository pairs table as the join's candidate "
        f"pair pool: {drl_filtered.height:,} pairs"
    )

    # Expand each identity to its researcher's candidate documents, then narrow to the
    # document-repository pairs whose repository is also in that identity's developer
    # account's contribution list.
    rdal_docs = rdal_filtered.join(
        document_contributors.select("researcher_id", "document_id"), on="researcher_id"
    )
    rdal_docs_pairs = rdal_docs.join(drl_filtered, on="document_id")
    candidate_pairs = rdal_docs_pairs.join(
        repository_contributors.select("developer_account_id", "repository_id"),
        on=["developer_account_id", "repository_id"],
        how="inner",
    )
    print(
        f"Candidate document-repository pairs connecting an identity's author and developer "
        f"account: {candidate_pairs.height:,} rows (before dedup/earliest-iteration reduction)"
    )

    attribution = (
        candidate_pairs.with_columns(
            pl.col("link_processing_iteration").fill_null(-1).alias("iteration_bucket")
        )
        .group_by(["researcher_id", "developer_account_id"])
        .agg(pl.min("iteration_bucket").alias("earliest_iteration_bucket"))
    )

    n_identities_total = rdal_filtered.height
    n_identities_attributed = attribution.height
    n_identities_unattributed = n_identities_total - n_identities_attributed
    print(
        f"\nOf {n_identities_total:,} filtered identity links, {n_identities_attributed:,} "
        f"had >=1 qualifying candidate document-repository pair and could be attributed to an "
        f"iteration; {n_identities_unattributed:,} had none (not in the standard-filtered pairs "
        f"table's candidate pool -- e.g. discovered via a pair that didn't clear the 0.9994 "
        f"confidence or post-2008 filters) and are excluded from the per-iteration counts below."
    )

    identities_by_iteration = (
        attribution.group_by("earliest_iteration_bucket")
        .agg(pl.len().alias("n_new_identities"))
        .sort("earliest_iteration_bucket")
    )
    print("\n--- (b) New researcher-developer-account identity links per iteration ---")
    print(identities_by_iteration)

    identities_mining_total = int(
        identities_by_iteration.filter(pl.col("earliest_iteration_bucket") != -1)
        .get_column("n_new_identities")
        .sum()
    )
    print(
        f"\nTotal new identity links attributed to mining iterations (1-5): {identities_mining_total:,}"
    )

    u.save_table(identities_by_iteration, "unit12b_new_identities_by_iteration", output_dir)

    print(
        "\nAttribution caveats:\n"
        "  (a) An identity connected through pairs spanning multiple iterations attributes "
        "to the earliest ('first possible discovery', not a certainty).\n"
        "  (b) An identity whose earliest qualifying pair came from a seed source attributes "
        "to the seed even if it also links via later mining rounds.\n"
        "  (c) Ties within one iteration are unambiguous (min() is well-defined).\n"
    )

    print(
        "--- Combined mining-rounds table (Table X), using RETAINED (standard-filtered) pair "
        "counts -- this is the version that matches 'new pairs added to the dataset' ---"
    )
    combined = (
        pairs_by_iteration_filtered.rename({"n_pairs": "new_pairs"})
        .join(
            identities_by_iteration.rename({"n_new_identities": "new_identities"}),
            left_on="iteration_bucket",
            right_on="earliest_iteration_bucket",
            how="full",
            coalesce=True,
        )
        .sort("iteration_bucket")
    )
    print(combined)
    u.save_table(combined, "unit12_mining_rounds_table", output_dir)


###############################################################################
# Figure 1, panel 1 -- full quad-partite network visualization


def _pack_components_layout(
    graph: rx.PyGraph, seed: int = 42
) -> tuple[dict[int, tuple[float, float]], float, float]:
    """
    Lay out a graph by connected component instead of one global force layout.

    A random-pair sample's topology is a forest of thousands of small disjoint
    article-repository-author-developer "stars". A global `rx.graph_spring_layout` call has
    no attractive force between disconnected components, so at large sample sizes they
    collapse into one dense, illegible disc. Packing instead: lay out each component with a
    local spring layout, then place components into a grid (largest first, sized by each
    component's bounding box) with padding proportional to cell size, so the sample's real
    structure stays visible at any sample size. Returns positions plus the packed canvas's
    width and height (in data units) so the caller can size the figure to match.
    """
    rng = random.Random(seed)
    components = [list(c) for c in rx.connected_components(graph)]

    cell_boxes: list[tuple[list[int], float, float, dict[int, tuple[float, float]]]] = []
    for comp in components:
        if len(comp) == 1:
            local_pos = {comp[0]: (0.0, 0.0)}
            w = h = 1.0
        else:
            sub = graph.subgraph(comp)
            k = 1.5 / (len(comp) ** 0.5)
            # Spring layout is ~quadratic in node count (5k nodes ~5s, 30k ~176s at 60
            # iters); iteration count steps down as size grows to keep total layout time in
            # single-digit minutes.
            if len(comp) < 50:
                n_iter = 60
            elif len(comp) < 5_000:
                n_iter = 100
            elif len(comp) < 20_000:
                n_iter = 50
            else:
                n_iter = 30
            sub_pos = rx.graph_spring_layout(sub, k=k, num_iter=n_iter, seed=seed)  # type: ignore[call-arg]
            xs = [p[0] for p in sub_pos.values()]
            ys = [p[1] for p in sub_pos.values()]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            w = max(maxx - minx, 0.3)
            h = max(maxy - miny, 0.3)
            local_pos = {
                orig_i: (sub_pos[local_i][0] - minx, sub_pos[local_i][1] - miny)
                for local_i, orig_i in enumerate(comp)
            }
        cell_boxes.append((comp, w, h, local_pos))

    # Largest components first so they anchor the top-left of the grid; row width targets a
    # roughly square canvas overall.
    cell_boxes.sort(key=lambda cb: -len(cb[0]))
    total_area = sum((w * 1.9 + 0.25) * (h * 1.9 + 0.25) for _, w, h, _ in cell_boxes)
    target_row_width = (total_area**0.5) * 1.1

    positions: dict[int, tuple[float, float]] = {}
    cursor_x, cursor_y, row_height = 0.0, 0.0, 0.0
    for _comp, w, h, local_pos in cell_boxes:
        pad = max(0.25, 0.9 * max(w, h))
        if cursor_x + w + pad > target_row_width and cursor_x > 0:
            cursor_x = 0.0
            cursor_y += row_height + pad
            row_height = 0.0
        jitter_x = rng.uniform(-0.15, 0.15) * pad
        jitter_y = rng.uniform(-0.15, 0.15) * pad
        for orig_i, (lx, ly) in local_pos.items():
            positions[orig_i] = (cursor_x + lx + jitter_x, cursor_y + ly + jitter_y)
        cursor_x += w + pad
        row_height = max(row_height, h)

    canvas_width = target_row_width
    canvas_height = cursor_y + row_height
    return positions, canvas_width, canvas_height


def _global_spring_layout(
    graph: rx.PyGraph, seed: int = 42
) -> tuple[dict[int, tuple[float, float]], float, float]:
    """
    Lay out the whole sampled graph with one global spring layout. The snowball sample puts
    ~99% of pairs in one giant component, which a conventional force layout renders directly
    (the packed-by-component layout above exists for forest-of-tiny-stars topologies).
    Iteration count steps down with node count (spring layout is ~quadratic); k is slightly
    above rustworkx's 1/sqrt(n) default to spread the giant component's dense core.
    """
    n = len(graph.node_indices())
    if n < 5_000:
        n_iter = 150
    elif n < 20_000:
        n_iter = 80
    else:
        n_iter = 40
    k = 1.5 / (n**0.5)
    pos = rx.graph_spring_layout(graph, k=k, num_iter=n_iter, seed=seed)  # type: ignore[call-arg]
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, miny = min(xs), min(ys)
    positions = {i: (p[0] - minx, p[1] - miny) for i, p in pos.items()}
    return positions, max(xs) - minx, max(ys) - miny


def _forceatlas2_positions(
    graph: rx.PyGraph,
    init: dict[int, tuple[float, float]],
    seed: int = 42,
    max_iter: int = 60,
    repulsion_sample: int = 8192,
) -> dict[int, tuple[float, float]]:
    """
    ForceAtlas2 layout, implemented here because the installed
    `networkx.forceatlas2_layout` materializes full O(n^2) pairwise matrices (~50+GB at this
    graph's ~59k nodes) and no other FA2/igraph/graphviz package is installed. Follows the
    FA2 paper's forces (degree+1 node mass, linear edge attraction,
    mass-scaled repulsion, gravity) and its adaptive global-speed/swing update. Repulsion
    uses a fresh random node sample per iteration (scaled by n/sample) when the graph is
    large -- the standard sampling approximation, keeping each iteration O(n * sample)
    instead of O(n^2). Warm-started from the spring positions.
    """
    rng = np.random.default_rng(seed)
    idx = list(graph.node_indices())
    n = len(idx)
    row_of = {node: k for k, node in enumerate(idx)}
    pos = np.array([init[i] for i in idx], dtype=np.float32)
    edges = np.array([(row_of[a], row_of[b]) for a, b in graph.edge_list()], dtype=np.int64)
    deg = np.zeros(n, dtype=np.float32)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    mass = deg + 1.0

    scaling_ratio, gravity, jitter_tolerance = 2.0, 1.0, 1.0
    speed, speed_efficiency = 1.0, 1.0
    prev_forces = np.zeros_like(pos)
    use_sampling = n > 20_000
    for it in range(max_iter):
        forces = np.zeros_like(pos)
        # Repulsion: k_r * m_i * m_j / d^2 * delta (sampled approximation on large graphs).
        if use_sampling:
            sample = rng.choice(n, size=min(repulsion_sample, n), replace=False)
            scale = n / len(sample)
        else:
            sample = np.arange(n)
            scale = 1.0
        s_pos, s_mass = pos[sample], mass[sample]
        chunk = 4096
        for s in range(0, n, chunk):
            delta = pos[s : s + chunk, None, :] - s_pos[None, :, :]
            d2 = (delta**2).sum(-1) + 1e-6
            f = scaling_ratio * (mass[s : s + chunk, None] * s_mass[None, :]) / d2
            forces[s : s + chunk] += scale * (f[..., None] * delta).sum(1)
        # Attraction (linear) along edges.
        dvec = pos[edges[:, 0]] - pos[edges[:, 1]]
        np.add.at(forces, edges[:, 0], -dvec)
        np.add.at(forces, edges[:, 1], dvec)
        # Gravity toward the centroid.
        gd = pos - pos.mean(0)
        gdist = np.sqrt((gd**2).sum(-1))[:, None] + 1e-6
        forces -= gravity * mass[:, None] * gd / gdist
        # FA2 adaptive speed (swing/traction).
        swing = mass * np.sqrt(((forces - prev_forces) ** 2).sum(-1))
        traction = mass * np.sqrt(((forces + prev_forces) ** 2).sum(-1)) / 2
        total_swing, total_traction = float(swing.sum()), float(traction.sum())
        if total_swing > 0:
            est_jt = 0.05 * np.sqrt(n)
            jt = jitter_tolerance * float(
                np.clip(est_jt * total_traction / n**2, np.sqrt(est_jt), 10.0)
            )
            if total_swing / total_traction > 2.0:
                speed_efficiency = max(0.05, speed_efficiency * 0.5)
                jt = max(jt, jitter_tolerance)
            target_speed = jt * speed_efficiency * total_traction / total_swing
            if total_swing > jt * total_traction:
                speed_efficiency = max(0.05, speed_efficiency * 0.7)
            elif speed < 1000:
                speed_efficiency *= 1.3
            speed = speed + min(target_speed - speed, 0.5 * speed)
        factor = speed / (1.0 + np.sqrt(speed * swing))
        pos = pos + forces * factor[:, None]
        prev_forces = forces
        if (it + 1) % 10 == 0:
            print(f"  ForceAtlas2 iteration {it + 1}/{max_iter} (speed {speed:.3f})")
    minx, miny = pos[:, 0].min(), pos[:, 1].min()
    return {
        node: (float(pos[k, 0] - minx), float(pos[k, 1] - miny)) for node, k in row_of.items()
    }


def _build_quadpartite_graph(
    sampled_pairs: pl.DataFrame,
    doc_authors: pl.DataFrame,
    repo_devs: pl.DataFrame,
    identity_edges: pl.DataFrame,
    doc_fields: dict[int, str] | None = None,
) -> rx.PyGraph:
    """Build the quad-partite `rx.PyGraph` from the already-sampled/capped/filtered frames.
    `doc_fields` attaches the field color-encoding attribute to article nodes -- the only
    color-encoded node type.
    """
    graph: rx.PyGraph = rx.PyGraph()
    node_idx: dict[tuple[str, int], int] = {}
    doc_fields = doc_fields or {}

    def _add_node(node_type: str, node_id: int) -> int:
        key = (node_type, node_id)
        if key not in node_idx:
            payload = {"type": node_type, "id": node_id}
            if node_type == "article":
                payload["group"] = doc_fields.get(node_id, "Other")
            node_idx[key] = graph.add_node(payload)
        return node_idx[key]

    for row in sampled_pairs.iter_rows(named=True):
        doc_i = _add_node("article", row["document_id"])
        repo_i = _add_node("repository", row["repository_id"])
        if not graph.has_edge(doc_i, repo_i):
            graph.add_edge(doc_i, repo_i, {"type": "article_repository_link"})

    for row in doc_authors.iter_rows(named=True):
        doc_i = node_idx.get(("article", row["document_id"]))
        if doc_i is None:
            continue
        res_i = _add_node("researcher", row["researcher_id"])
        if not graph.has_edge(doc_i, res_i):
            graph.add_edge(doc_i, res_i, {"type": "authored_by"})

    for row in repo_devs.iter_rows(named=True):
        repo_i = node_idx.get(("repository", row["repository_id"]))
        if repo_i is None:
            continue
        dev_i = _add_node("developer", row["developer_account_id"])
        if not graph.has_edge(repo_i, dev_i):
            graph.add_edge(repo_i, dev_i, {"type": "contributed_to"})

    for row in identity_edges.iter_rows(named=True):
        res_i = node_idx.get(("researcher", row["researcher_id"]))
        dev_i = node_idx.get(("developer", row["developer_account_id"]))
        if res_i is None or dev_i is None:
            continue
        if not graph.has_edge(res_i, dev_i):
            graph.add_edge(res_i, dev_i, {"type": "identity"})

    return graph


@app.command()
def figure_1_quadpartite_network(
    output_dir: Path = OUTPUT_DIR,
    n_pairs: int = 10000,
    n_top_hub_anchors: int = 25,
    n_random_hub_anchors: int = 75,
    min_anchor_pair_degree: int = 3,
    max_pairs_per_entity: int = 10,
    max_contributors_per_side: int = 2,
    rdal_confidence_threshold: float = 0.97,
    random_seed: int = 42,
    layout: str = "both",
) -> None:
    """
    Build Figure 1, panel 1: the quad-partite network -- articles, repositories, researchers
    (authors), and developer accounts (contributors) -- rendered together with rustworkx.
    Panels 2/3 of Figure 1 (the growth-model and workflow diagrams) are hand-made images and
    out of scope here.

    Sampling: SNOWBALL from well-linked identity hubs, rather than a uniform-random pair
    sample (two randomly sampled pairs are bridged only if both happen to be drawn AND share
    an entity, which is rare, so random samples look artificially fragmented). Design:

      - Anchors: the top `n_top_hub_anchors` identity links (researcher, developer) by
        pair-degree (pairs reachable through either side), plus `n_random_hub_anchors`
        uniformly sampled identities with pair-degree >= `min_anchor_pair_degree` (seeded) --
        the stratified tail keeps the figure from over-representing anomalous mega-hubs.
      - Growth: BFS over pairs, mediated by entities. Each popped pair contributes up to
        `max_contributors_per_side` authors/contributors PLUS, always, any identity-linked
        author/contributor (a hard cap alone would sever identity edges whenever the linked
        person isn't among the first contributor rows). Each collected entity (and its
        identity counterpart) contributes up to `max_pairs_per_entity` new pairs (seeded
        subsample), which keeps hubs from dominating.
      - Stop the instant the `n_pairs` budget fills; if all anchors exhaust below budget, top
        up with uniform-random pairs and report the top-up count.

    Confidence filters: pair confidence >= 0.9994 or NULL (standard); identity links at
    >= `rdal_confidence_threshold` (default 0.97, the paper's stated threshold).
    """
    evaplot.set_style("evaplot_rc")
    rng = random.Random(random_seed)

    pairs = u.load_filtered_pairs()

    print("Loading contributor/identity tables from HuggingFace...")
    document_contributors = u.load_table("document_contributor")
    repository_contributors = u.load_table("repository_contributor")
    rdal = u.load_table("researcher_developer_account_link")

    n_rdal_before = len(rdal)
    rdal = rdal.filter(pl.col("predictive_model_confidence") >= rdal_confidence_threshold)
    print(
        f"Researcher-developer-account identity links: {n_rdal_before:,} total, "
        f"{len(rdal):,} remain after confidence >= {rdal_confidence_threshold} filter."
    )

    # ---- Lookup structures for the snowball walk ----
    pair_frame = pairs.select(
        pl.col("document_repository_link_id").alias("pair_id"), "document_id", "repository_id"
    )
    pair_doc: dict[int, int] = {}
    pair_repo: dict[int, int] = {}
    doc_pairs: dict[int, list[int]] = {}
    repo_pairs: dict[int, list[int]] = {}
    for pid, doc_id, repo_id in pair_frame.iter_rows():
        pair_doc[pid] = doc_id
        pair_repo[pid] = repo_id
        doc_pairs.setdefault(doc_id, []).append(pid)
        repo_pairs.setdefault(repo_id, []).append(pid)

    doc_authors_ordered: dict[int, list[int]] = {}
    for doc_id, researcher_id in document_contributors.select(
        "document_id", "researcher_id"
    ).iter_rows():
        if doc_id in doc_pairs:
            doc_authors_ordered.setdefault(doc_id, []).append(researcher_id)
    repo_devs_ordered: dict[int, list[int]] = {}
    for repo_id, dev_id in repository_contributors.select(
        "repository_id", "developer_account_id"
    ).iter_rows():
        if repo_id in repo_pairs:
            repo_devs_ordered.setdefault(repo_id, []).append(dev_id)

    researcher_pairs: dict[int, list[int]] = {}
    for doc_id, authors in doc_authors_ordered.items():
        for researcher_id in authors:
            researcher_pairs.setdefault(researcher_id, []).extend(doc_pairs[doc_id])
    developer_pairs: dict[int, list[int]] = {}
    for repo_id, devs in repo_devs_ordered.items():
        for dev_id in devs:
            developer_pairs.setdefault(dev_id, []).extend(repo_pairs[repo_id])

    identity_r2d: dict[int, list[int]] = {}
    identity_d2r: dict[int, list[int]] = {}
    identity_links: list[tuple[int, int]] = []
    for researcher_id, dev_id in (
        rdal.select("researcher_id", "developer_account_id").unique().iter_rows()
    ):
        identity_r2d.setdefault(researcher_id, []).append(dev_id)
        identity_d2r.setdefault(dev_id, []).append(researcher_id)
        identity_links.append((researcher_id, dev_id))

    # ---- Hub selection: pair-degree per identity link, stratified anchors ----
    degree_by_link = [
        (
            len(researcher_pairs.get(r, [])) + len(developer_pairs.get(d, [])),
            r,
            d,
        )
        for r, d in identity_links
    ]
    degree_by_link.sort(key=lambda t: (-t[0], t[1], t[2]))
    top_anchors = [(r, d) for _deg, r, d in degree_by_link[:n_top_hub_anchors]]
    tail_candidates = [
        (r, d)
        for deg, r, d in degree_by_link[n_top_hub_anchors:]
        if deg >= min_anchor_pair_degree
    ]
    rng.shuffle(tail_candidates)
    anchor_queue = top_anchors + tail_candidates  # first 25+75 planned; rest is the reserve
    print(
        f"Anchors: {len(top_anchors)} top hubs (max pair-degree "
        f"{degree_by_link[0][0] if degree_by_link else 0:,}) + "
        f"{min(n_random_hub_anchors, len(tail_candidates)):,} stratified random of "
        f"{len(tail_candidates):,} identities with pair-degree >= {min_anchor_pair_degree} "
        "(remainder held as reserve)."
    )

    # ---- Snowball BFS over pairs ----
    budget = min(n_pairs, len(pair_doc))
    sampled: set[int] = set()
    frontier: deque[int] = deque()

    def _add_pairs(candidate_pairs: list[int]) -> None:
        new = [p for p in dict.fromkeys(candidate_pairs) if p not in sampled]
        if len(new) > max_pairs_per_entity:
            new = rng.sample(new, max_pairs_per_entity)
        for p in new:
            if len(sampled) >= budget:
                return
            sampled.add(p)
            frontier.append(p)

    def _collect_entities(ordered: list[int], identity_map: dict[int, list[int]]) -> list[int]:
        # Cap at max_contributors_per_side, but always include identity-linked entities so
        # the cap never severs identity edges.
        kept = list(ordered[:max_contributors_per_side])
        kept += [e for e in ordered[max_contributors_per_side:] if e in identity_map]
        return list(dict.fromkeys(kept))

    # Seed ALL planned anchors' pairs up front so the sample spans every anchor neighborhood
    # rather than exhausting the budget on the first hub's BFS; the remaining tail is a
    # reserve drawn only if the frontier empties below budget.
    n_anchors_used = 0
    for anchor_r, anchor_d in anchor_queue[: n_top_hub_anchors + n_random_hub_anchors]:
        if len(sampled) >= budget:
            break
        n_anchors_used += 1
        _add_pairs(researcher_pairs.get(anchor_r, []))
        _add_pairs(developer_pairs.get(anchor_d, []))
    anchor_queue = anchor_queue[n_anchors_used:]

    while len(sampled) < budget:
        if not frontier:
            if not anchor_queue:
                break
            anchor_r, anchor_d = anchor_queue.pop(0)
            n_anchors_used += 1
            _add_pairs(researcher_pairs.get(anchor_r, []))
            if len(sampled) >= budget:
                break
            _add_pairs(developer_pairs.get(anchor_d, []))
            continue
        pid = frontier.popleft()
        authors = _collect_entities(doc_authors_ordered.get(pair_doc[pid], []), identity_r2d)
        devs = _collect_entities(repo_devs_ordered.get(pair_repo[pid], []), identity_d2r)
        for researcher_id in authors:
            _add_pairs(researcher_pairs.get(researcher_id, []))
            if len(sampled) >= budget:
                break
            for dev_id in identity_r2d.get(researcher_id, []):
                _add_pairs(developer_pairs.get(dev_id, []))
                if len(sampled) >= budget:
                    break
        if len(sampled) >= budget:
            break
        for dev_id in devs:
            _add_pairs(developer_pairs.get(dev_id, []))
            if len(sampled) >= budget:
                break
            for researcher_id in identity_d2r.get(dev_id, []):
                _add_pairs(researcher_pairs.get(researcher_id, []))
                if len(sampled) >= budget:
                    break

    n_snowball = len(sampled)
    n_topup = 0
    if n_snowball < budget:
        remainder = [p for p in pair_doc if p not in sampled]
        topup = rng.sample(remainder, min(budget - n_snowball, len(remainder)))
        sampled.update(topup)
        n_topup = len(topup)
    print(
        f"\nSnowball sample: {n_snowball:,} pairs from {n_anchors_used:,} anchors"
        + (f" + {n_topup:,} uniform-random top-up pairs" if n_topup else "")
        + f" = {len(sampled):,} of {len(pair_doc):,} filtered pairs."
    )

    sampled_pairs = pair_frame.filter(pl.col("pair_id").is_in(sampled)).rename(
        {"pair_id": "document_repository_link_id"}
    )

    # ---- Entity frames for graph construction (same cap + always-include-identity rule) ----
    author_rows: list[tuple[int, int]] = []
    for doc_id in sampled_pairs.get_column("document_id").unique().to_list():
        for researcher_id in _collect_entities(
            doc_authors_ordered.get(doc_id, []), identity_r2d
        ):
            author_rows.append((doc_id, researcher_id))
    dev_rows: list[tuple[int, int]] = []
    for repo_id in sampled_pairs.get_column("repository_id").unique().to_list():
        for dev_id in _collect_entities(repo_devs_ordered.get(repo_id, []), identity_d2r):
            dev_rows.append((repo_id, dev_id))
    doc_authors = pl.DataFrame(
        {
            "document_id": [r[0] for r in author_rows],
            "researcher_id": [r[1] for r in author_rows],
        }
    )
    repo_devs = pl.DataFrame(
        {
            "repository_id": [r[0] for r in dev_rows],
            "developer_account_id": [r[1] for r in dev_rows],
        }
    )
    print(
        f"Entity rows (cap {max_contributors_per_side} + always-include-identity-linked): "
        f"{len(doc_authors):,} authorship rows, {len(repo_devs):,} contribution rows."
    )

    seed_researcher_ids = set(doc_authors.get_column("researcher_id").unique().to_list())
    seed_developer_ids = set(repo_devs.get_column("developer_account_id").unique().to_list())
    identity_edges = rdal.filter(
        pl.col("researcher_id").is_in(seed_researcher_ids)
        & pl.col("developer_account_id").is_in(seed_developer_ids)
    )
    print(
        f"Identity edges drawn: {len(identity_edges):,} "
        f"(both endpoints present in the sampled graph)."
    )

    # ---- Node color-encoding metadata: only articles carry color, by field (top 6 +
    # Other); all other node types are grey outline-only shapes. ----
    fig1_fields = (
        pairs.get_column("document_field_name_pruned")
        .value_counts(sort=True)
        .filter(pl.col("document_field_name_pruned") != "Other")
        .head(6)
        .get_column("document_field_name_pruned")
        .to_list()
    )
    doc_fields = {
        did: (f if f in fig1_fields else "Other")
        for did, f in pairs.select("document_id", "document_field_name_pruned")
        .unique(subset="document_id")
        .iter_rows()
    }
    graph = _build_quadpartite_graph(
        sampled_pairs,
        doc_authors,
        repo_devs,
        identity_edges,
        doc_fields=doc_fields,
    )

    n_nodes = len(graph.nodes())
    n_edges = len(graph.edges())
    counts_by_type: dict[str, int] = {}
    for node in graph.nodes():
        counts_by_type[node["type"]] = counts_by_type.get(node["type"], 0) + 1
    print(f"\nFinal sampled network: {n_nodes:,} nodes, {n_edges:,} edges.")
    print(f"  Node counts by type: {counts_by_type}")

    # ---- Connectivity reporting (the point of the snowball change) ----
    components = rx.connected_components(graph)
    comp_sizes = sorted((len(c) for c in components), reverse=True)
    largest_comp = max(components, key=len) if components else set()
    doc_node_ids = {
        node_i for node_i in graph.node_indices() if graph[node_i]["type"] == "article"
    }
    largest_doc_ids = {graph[node_i]["id"] for node_i in largest_comp if node_i in doc_node_ids}
    n_pairs_in_largest = sampled_pairs.filter(
        pl.col("document_id").is_in(largest_doc_ids)
    ).height
    pct_pairs_in_largest = 100 * n_pairs_in_largest / len(sampled)
    print(
        f"Connectivity: {len(comp_sizes):,} components; largest holds "
        f"{comp_sizes[0]:,} nodes and {n_pairs_in_largest:,} sampled pairs "
        f"({pct_pairs_in_largest:.1f}% of the sample); "
        f"top-5 component sizes: {comp_sizes[:5]}."
    )

    # ---- Subset to the largest connected component; the caption reports how many
    # components existed. ----
    n_dropped_nodes = n_nodes - len(largest_comp)
    graph = graph.subgraph(list(largest_comp))
    print(
        f"Restricting to the largest connected component: dropped {len(comp_sizes) - 1:,} "
        f"satellite components ({n_dropped_nodes:,} nodes); drawing {len(graph.nodes()):,} "
        "nodes."
    )

    suptitle_base = (
        f"Snowball-sampled quad-partite network (n={len(sampled):,} pairs from "
        f"{n_anchors_used:,} identity-hub anchors"
        + (f" + {n_topup:,} random top-up" if n_topup else "")
        + f"; {len(comp_sizes):,} components existed, only the largest is drawn = "
        f"{pct_pairs_in_largest:.0f}% of pairs)"
    )

    # Spring is the primary output; ForceAtlas2 is an additional variant; "packed" keeps the
    # per-component layout available.
    if layout in ("both", "spring", "forceatlas2"):
        print("\nComputing global spring layout...")
        positions, canvas_width, canvas_height = _global_spring_layout(graph, seed=random_seed)
        if layout in ("both", "spring"):
            _draw_quadpartite_network(
                graph,
                positions,
                canvas_width,
                canvas_height,
                suptitle_base + " -- spring layout",
                output_dir,
                fig1_fields,
                stem="figure1_quadpartite_network",
            )
        if layout in ("both", "forceatlas2"):
            print("\nComputing ForceAtlas2 layout (warm-started from spring positions)...")
            fa2_positions = _forceatlas2_positions(graph, positions, seed=random_seed)
            xs = [p[0] for p in fa2_positions.values()]
            ys = [p[1] for p in fa2_positions.values()]
            _draw_quadpartite_network(
                graph,
                fa2_positions,
                max(xs),
                max(ys),
                suptitle_base + " -- ForceAtlas2 layout",
                output_dir,
                fig1_fields,
                stem="figure1_quadpartite_network_forceatlas2",
            )
    else:
        positions, canvas_width, canvas_height = _pack_components_layout(
            graph, seed=random_seed
        )
        _draw_quadpartite_network(
            graph,
            positions,
            canvas_width,
            canvas_height,
            suptitle_base + " -- packed layout",
            output_dir,
            fig1_fields,
            stem="figure1_quadpartite_network",
        )


def _draw_quadpartite_network(
    graph: rx.PyGraph,
    positions: dict[int, tuple[float, float]],
    canvas_width: float,
    canvas_height: float,
    suptitle: str,
    output_dir: Path,
    field_order: list[str],
    stem: str = "figure1_quadpartite_network",
) -> None:
    """Draw and save one Figure 1 render: only articles carry color (by field, matching
    Figure 2 Panel B's palette); repositories/researchers/developers are grey outline-only
    shapes; all edges uniform grey differentiated by alpha/linewidth; LineCollections and
    scatters rasterized so the PDF stays small and fast to open.
    """
    # Field colors match Figure 2 Panel B's colorblind palette.
    field_colors = dict(
        zip(
            [*field_order, "Other"],
            [*sns.color_palette("colorblind", n_colors=len(field_order)), "#bbbbbb"],
            strict=True,
        )
    )
    repo_grey = "#999999"
    people_grey = "#cccccc"

    # Uniform grey edges: edge type is differentiated only by alpha/linewidth, never by hue.
    edge_grey = "#888888"
    edge_style = {
        "authored_by": {"color": edge_grey, "lw": 0.35, "ls": "-", "zorder": 1, "alpha": 0.12},
        "contributed_to": {
            "color": edge_grey,
            "lw": 0.35,
            "ls": "-",
            "zorder": 1,
            "alpha": 0.12,
        },
        "article_repository_link": {
            "color": edge_grey,
            "lw": 0.7,
            "ls": "-",
            "zorder": 2,
            "alpha": 0.3,
        },
        "identity": {"color": edge_grey, "lw": 0.9, "ls": "-", "zorder": 3, "alpha": 0.25},
    }
    marker_size = 6.0
    people_marker_size = 4.0
    node_alpha = 0.6
    marker_lw = 0.25

    ref_canvas_extent = 143.2  # canvas width measured at the 2,000-seed-pair calibration run
    canvas_extent = max(canvas_width, canvas_height)
    figsize_in = min(22.0, max(14.0, 14.0 * canvas_extent / ref_canvas_extent))

    fig, ax = plt.subplots(figsize=(figsize_in, figsize_in))

    for etype, style in edge_style.items():
        segments = [
            (positions[src], positions[tgt])
            for edge_idx in graph.edge_indices()
            if graph.get_edge_data_by_index(edge_idx)["type"] == etype
            for src, tgt in [graph.get_edge_endpoints_by_index(edge_idx)]
        ]
        if not segments:
            continue
        ax.add_collection(
            LineCollection(
                segments,
                colors=style["color"],
                linewidths=style["lw"],
                linestyles=style["ls"],
                zorder=style["zorder"],
                alpha=style["alpha"],
                rasterized=True,
            )
        )
    ax.autoscale_view()

    # Colored, filled article nodes (grouped by field) -- the only color-encoded node type.
    xs_by_group: dict[str, list[float]] = {}
    ys_by_group: dict[str, list[float]] = {}
    for node_i in graph.node_indices():
        nd = graph[node_i]
        if nd["type"] != "article":
            continue
        group = nd.get("group", "Other")
        x, y = positions[node_i]
        xs_by_group.setdefault(group, []).append(x)
        ys_by_group.setdefault(group, []).append(y)
    for group, xs in xs_by_group.items():
        ax.scatter(
            xs,
            ys_by_group[group],
            c=field_colors.get(group, "#bbbbbb"),
            marker="o",
            s=marker_size,
            alpha=node_alpha,
            edgecolors="none",
            linewidths=marker_lw,
            zorder=4,
            rasterized=True,
        )

    # Grey outline-only shapes for everything else: repositories keep article-node size and a
    # darker grey so they read above the people-nodes, which stay smallest/lightest.
    for ntype, marker, edge_color, size, alpha in [
        ("repository", "s", repo_grey, marker_size, node_alpha),
        ("researcher", "D", people_grey, people_marker_size, 0.5),
        ("developer", "^", people_grey, people_marker_size, 0.5),
    ]:
        xs, ys = [], []
        for node_i in graph.node_indices():
            nd = graph[node_i]
            if nd["type"] == ntype:
                x, y = positions[node_i]
                xs.append(x)
                ys.append(y)
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            facecolors="none",
            edgecolors=edge_color,
            marker=marker,
            s=size,
            alpha=alpha,
            linewidths=marker_lw,
            zorder=4,
            rasterized=True,
        )

    def _marker_handle(marker, color, label, hollow=False, size=7):
        return mlines.Line2D(
            [],
            [],
            marker=marker,
            color="none",
            markerfacecolor="none" if hollow else color,
            markeredgecolor=color,
            linestyle="None",
            markersize=size,
            label=label,
        )

    legend_handles = [
        _marker_handle("o", field_colors[f], f"Article: {f}") for f in [*field_order, "Other"]
    ]
    legend_handles += [
        _marker_handle("s", repo_grey, "Repository", hollow=True),
        _marker_handle("D", people_grey, "Researcher (author)", hollow=True),
        _marker_handle("^", people_grey, "Developer account", hollow=True),
        mlines.Line2D([], [], color=edge_grey, lw=1.2, label="Authorship / contribution"),
        mlines.Line2D([], [], color=edge_grey, lw=1.5, label="Article-repository link"),
        mlines.Line2D([], [], color=edge_grey, lw=1.8, label="Researcher-developer identity"),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        fontsize=8,
    )
    u.style_legend(legend, fontsize=8)

    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.suptitle(suptitle, fontsize=11, y=1.03)

    u.save_figure(fig, stem, output_dir)
    plt.close(fig)


###############################################################################
# Unit 13 -- co-authorship network (fresh rebuild)
#
# The manuscript's Results text cites component statistics originally sourced from the
# docs-site's `web/data-prep/queries/coauthorship_network.py` pipeline; those numbers must be
# rebuilt inside this replication package before being cited. The docs-site pipeline is
# reference only for what the computation needs to do (edge-construction rule, author-count
# bound).


@app.command()
def unit13_coauthorship_network(
    output_dir: Path = OUTPUT_DIR,
    min_authors: int = 2,
    max_authors: int = 12,
) -> None:
    """
    Compute co-authorship network statistics -- connected component count, largest
    component's size/percentage, and the next-largest component's size (the three numbers the
    manuscript cites).

    Edge-construction rule: nodes are researchers; one undirected edge per co-authoring
    researcher pair (not one edge per shared document), weighted by the number of documents
    that pair co-authored together (`n_shared_docs`). Documents are bounded to
    `min_authors`-`max_authors` listed authors before generating all-pairs edges within a
    document -- without this bound, large-consortium papers would each contribute up to
    C(n_authors, 2) edges of combinatorial noise.

    The researcher-developer identity-link confidence filter is deliberately NOT applied:
    it constrains researcher<->developer identity, and co-authorship is a purely
    researcher<->researcher relationship, so there are no identities in scope for it.
    """
    pairs = u.load_filtered_pairs()
    filtered_document_ids = pairs.get_column("document_id").unique().to_list()
    print(
        f"\nDocuments surviving standard pair-level filtering (confidence >= "
        f"{u.DEFAULT_CONFIDENCE_THRESHOLD} or NULL, post-{u.DEFAULT_MIN_YEAR}): "
        f"{len(filtered_document_ids):,}"
    )

    print("Loading document_contributor from HuggingFace...")
    document_contributors = u.load_table("document_contributor")
    print(f"  document_contributor: {len(document_contributors):,} rows")

    authors = (
        document_contributors.select("document_id", "researcher_id")
        .unique()
        .filter(pl.col("document_id").is_in(filtered_document_ids))
    )
    print(
        f"Authorship rows for filtered documents: {len(authors):,} "
        f"({authors.get_column('document_id').n_unique():,} documents, "
        f"{authors.get_column('researcher_id').n_unique():,} researchers)"
    )

    author_counts = authors.group_by("document_id").agg(n_authors=pl.len())
    qualifying_docs = author_counts.filter(
        pl.col("n_authors").is_between(min_authors, max_authors)
    ).select("document_id")
    n_docs_before_bound = authors.get_column("document_id").n_unique()
    # The author-count bound caps combinatorial all-pairs EDGE generation for large-consortium
    # papers -- it must not also shrink the node set. `authors` (unbounded) still holds every
    # researcher with an authorship row on a filtered document, so single-author-only
    # researchers (min_authors=2 excludes their only paper) remain graph nodes, appearing as
    # their own isolated one-node component rather than being dropped from the network entirely.
    bounded_authors = authors.join(qualifying_docs, on="document_id", how="inner")
    print(
        f"After bounding to {min_authors}-{max_authors} listed authors/document (caps "
        f"combinatorial all-pairs edge generation for large-consortium papers): "
        f"{bounded_authors.get_column('document_id').n_unique():,} of {n_docs_before_bound:,} "
        f"documents remain, {len(bounded_authors):,} authorship rows, "
        f"{bounded_authors.get_column('researcher_id').n_unique():,} unique researchers "
        "contribute edges (all researchers, including those excluded here, still become nodes)"
    )

    # All co-author pairs within each qualifying document, one direction only
    # (researcher_id < researcher_id_b), weighted by shared-document count.
    edges = (
        bounded_authors.join(bounded_authors, on="document_id", suffix="_b")
        .filter(pl.col("researcher_id") < pl.col("researcher_id_b"))
        .group_by(["researcher_id", "researcher_id_b"])
        .agg(n_shared_docs=pl.col("document_id").n_unique())
    )
    print(f"Co-authorship edges (unique researcher pairs): {len(edges):,}")

    node_ids = authors.get_column("researcher_id").unique().to_list()
    graph: rx.PyGraph = rx.PyGraph()
    index_by_researcher: dict[int, int] = {}
    for researcher_id in node_ids:
        index_by_researcher[researcher_id] = graph.add_node(researcher_id)
    for row in edges.iter_rows(named=True):
        graph.add_edge(
            index_by_researcher[row["researcher_id"]],
            index_by_researcher[row["researcher_id_b"]],
            row["n_shared_docs"],
        )

    n_nodes = graph.num_nodes()
    n_edges = graph.num_edges()
    print(f"\nCo-authorship network: {n_nodes:,} researcher nodes, {n_edges:,} edges")

    components = rx.connected_components(graph)
    component_sizes = sorted((len(c) for c in components), reverse=True)
    n_components = len(component_sizes)
    largest_size = component_sizes[0] if component_sizes else 0
    next_largest_size = component_sizes[1] if len(component_sizes) > 1 else 0
    largest_pct = 100 * largest_size / n_nodes if n_nodes else 0.0

    results = pl.DataFrame(
        {
            "component_rank": list(range(1, n_components + 1)),
            "component_size": component_sizes,
        }
    )
    u.save_table(results, "unit13_coauthorship_component_sizes", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_researcher_nodes": n_nodes,
        "n_coauthorship_edges": n_edges,
        "n_connected_components": n_components,
        "largest_component_size": largest_size,
        "largest_component_pct": round(largest_pct, 1),
        "next_largest_component_size": next_largest_size,
        "manuscript_currently_cites": {
            "n_connected_components": 11568,
            "largest_component_pct": 91,
            "next_largest_component_size": 46,
            "source": "web/data-prep/queries/coauthorship_network.py (docs-site pipeline, "
            "not to be cited per replication-package policy -- comparison only)",
        },
    }
    with open(output_dir / "unit13_coauthorship_network_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {output_dir / 'unit13_coauthorship_network_summary.json'}")

    print("\n--- Co-authorship network statistics (fresh rebuild) ---")
    print(f"Connected components: {n_components:,}")
    print(f"Largest component: {largest_size:,} researchers ({largest_pct:.1f}%)")
    print(f"Next-largest component: {next_largest_size:,} researchers")
    print(
        "Manuscript currently cites: 11,568 components / 91% in largest component / "
        "46 in next-largest (docs-site pipeline numbers -- not to be cited per "
        "replication-package policy; comparison only)."
    )
    print("---------------------------------------------------------\n")


###############################################################################
# Unit 14 -- full-network entity/edge counts


@app.command()
def unit14_network_entity_edge_counts(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Fill the manuscript's full-network paragraph (line 29) -- article, repository,
    researcher, and developer-account node counts, plus authorship, contribution,
    article-repository-link, and researcher-developer identity edge counts. Entities are
    derived from the standard-filtered pairs (pair-level filtering first); identity-link
    counts are reported at both the >=0.9 and the manuscript-Methods-stated >=0.97
    thresholds, restricted to identities whose researcher and developer account both appear
    in the network.
    """
    pairs = u.load_filtered_pairs()
    n_links = pairs.height
    n_articles = pairs.n_unique("document_id")
    n_repositories = pairs.n_unique("repository_id")
    doc_ids = pairs.get_column("document_id").unique()
    repo_ids = pairs.get_column("repository_id").unique()

    print("\nLoading contributor/identity tables from HuggingFace...")
    document_contributors = u.load_table("document_contributor")
    repository_contributors = u.load_table("repository_contributor")
    rdal = u.load_table("researcher_developer_account_link")

    authorship = (
        document_contributors.select("document_id", "researcher_id")
        .unique()
        .filter(pl.col("document_id").is_in(doc_ids.implode()))
    )
    contribution = (
        repository_contributors.select("repository_id", "developer_account_id")
        .unique()
        .filter(pl.col("repository_id").is_in(repo_ids.implode()))
    )
    n_researchers = authorship.n_unique("researcher_id")
    n_developer_accounts = contribution.n_unique("developer_account_id")

    researcher_ids = authorship.get_column("researcher_id").unique()
    developer_ids = contribution.get_column("developer_account_id").unique()
    in_network_rdal = rdal.filter(
        pl.col("researcher_id").is_in(researcher_ids.implode())
        & pl.col("developer_account_id").is_in(developer_ids.implode())
    )
    identity_counts = {
        f"n_identity_links_confidence_gte_{thr}": int(
            in_network_rdal.filter(pl.col("predictive_model_confidence") >= thr)
            .select("researcher_id", "developer_account_id")
            .unique()
            .height
        )
        # Both thresholds: 0.9 (the repo-wide convention) and 0.97 (the paper-local
        # default) -- kept as two columns so either can be cited.
        for thr in (0.9, 0.97)
    }

    summary = {
        "n_articles": n_articles,
        "n_repositories": n_repositories,
        "n_researchers": n_researchers,
        "n_developer_accounts": n_developer_accounts,
        "n_authorship_edges": authorship.height,
        "n_contribution_edges": contribution.height,
        "n_article_repository_links": n_links,
        **identity_counts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "unit14_network_entity_edge_counts.json", "w") as f:
        json.dump(summary, f, indent=2)
    u.save_table(
        pl.DataFrame(
            {"statistic": list(summary.keys()), "value": [float(v) for v in summary.values()]}
        ),
        "unit14_network_entity_edge_counts",
        output_dir,
    )

    print("\n--- Full-network entity/edge counts (line 29) ---")
    for k, v in summary.items():
        print(f"  {k}: {v:,}")
    print("-------------------------------------------------\n")


###############################################################################
# Unit 15 -- import-vs-dependency IoU over time

# Minimum pairs per year for plotted lines; CSVs keep every year.
UNIT15_MIN_PAIRS_PER_YEAR = 50


def _top5_unmatched_rows(
    direction_specs: list[tuple[str, dict]],
    weight_by_key: dict | None,
    n_eligible_pairs: int,
    top_n: int = 5,
) -> list[dict]:
    """Top-N unmatched names per direction. `direction_specs` maps a direction label
    to {key: set-of-unmatched-terms}; `weight_by_key` gives each key's pair count (None =
    each key is itself one pair). Counts = number of pairs in which the term appears
    unmatched.
    """
    rows: list[dict] = []
    for direction, unmatched_by_key in direction_specs:
        counts: dict[str, int] = {}
        for key, terms in unmatched_by_key.items():
            weight = 1 if weight_by_key is None else weight_by_key.get(key, 0)
            if not weight:
                continue
            for term in terms:
                counts[term] = counts.get(term, 0) + weight
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        for rank, (term, n) in enumerate(top, start=1):
            rows.append(
                {
                    "direction": direction,
                    "rank": rank,
                    "term": term,
                    "n_pairs_unmatched": n,
                    "n_eligible_pairs": n_eligible_pairs,
                }
            )
    return rows


# Cohort year for the fixed-cohort robustness re-run: repos already manifest-eligible by this
# year are held fixed and the conditional IoU trend is recomputed within them.
UNIT15_FIXED_COHORT_YEAR = 2017


@app.command()
def unit15_import_dependency_iou_over_time(
    output_dir: Path = OUTPUT_DIR,
    cutoff: float = 85.0,
    min_pairs_per_year: int = UNIT15_MIN_PAIRS_PER_YEAR,
) -> None:
    """
    Compute import-vs-dependency IoU over time. Builds: (1) the conditional metric (pairs
    whose repository has >=1 import AND >=1 pypi/conda/cran manifest dependency); (2) an
    unconditional companion -- same population minus the both-views-present requirement, with
    single-view pairs entering as IoU=0 and neither-view pairs excluded; (3) a three-band
    composition decomposition (no manifest / manifest-no-overlap / manifest-with-overlap) per
    year among import-bearing pairs; (4) a fixed-cohort robustness re-run (repos already
    manifest-eligible by `UNIT15_FIXED_COHORT_YEAR`); (5) directional top-5 unmatched-name
    lists. Dependency names are cleaned of residual comparator/junk characters before
    alignment.
    """
    evaplot.set_style("evaplot_rc")
    df = u.load_filtered_pairs(top_n_fields=10)

    print("\nLoading repository_import and repository_dependency from HuggingFace...")
    imports = u.load_table("repository_import")
    deps = u.load_table("repository_dependency")
    print(f"  repository_import: {len(imports):,} rows")
    print(f"  repository_dependency: {len(deps):,} rows")
    deps = u.clean_dependency_names(deps)

    manifest_ecosystems = sorted(
        {e for ecos in TABLE1_DEPENDENCY_ECOSYSTEMS.values() for e in ecos}
    )
    deps_pr = deps.filter(pl.col("ecosystem").is_in(manifest_ecosystems))
    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    repo_with_dep = set(deps_pr.get_column("repository_id").unique().to_list())
    both_view_repos = repo_with_import & repo_with_dep
    either_view_repos = repo_with_import | repo_with_dep
    eligible = df.filter(pl.col("repository_id").is_in(both_view_repos))
    print(
        f"Pairs whose repository has >=1 import AND >=1 {manifest_ecosystems} dependency: "
        f"{eligible.height:,} of {df.height:,}"
    )

    imports_by_repo: dict[int, list[str]] = {
        rid[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for rid, grp in imports.filter(pl.col("repository_id").is_in(both_view_repos)).group_by(
            "repository_id"
        )
    }
    deps_by_repo: dict[int, list[str]] = {
        rid[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for rid, grp in deps_pr.filter(pl.col("repository_id").is_in(both_view_repos)).group_by(
            "repository_id"
        )
    }

    # IoU is a repository-level property, so compute one alignment per unique repository and
    # attach it to each of that repository's pairs (pairs re-weight repos by publication
    # year). Unmatched import/dependency names ride along for the directional top-5 lists.
    iou_by_repo: dict[int, float] = {}
    unmatched_imports_by_repo: dict[int, set[str]] = {}
    unmatched_deps_by_repo: dict[int, set[str]] = {}
    for repo_id, import_names in imports_by_repo.items():
        dep_names = deps_by_repo.get(repo_id, [])
        unique_imports = set(import_names)
        unique_deps = set(dep_names)
        if not unique_imports or not unique_deps:
            continue
        matches = align_software_names(
            items_a=import_names,
            items_b=dep_names,
            source_a="import",
            source_b="dependency",
            cutoff=cutoff,
            method="global_min_diff",
        )
        matched_imports = {m.normalized_item_one for m in matches}
        matched_deps = {m.normalized_item_two for m in matches}
        n_matched = len(matched_imports)
        union_size = len(unique_imports) + len(unique_deps) - n_matched
        iou_by_repo[repo_id] = n_matched / union_size if union_size else float("nan")
        unmatched_imports_by_repo[repo_id] = unique_imports - matched_imports
        unmatched_deps_by_repo[repo_id] = unique_deps - matched_deps
    print(f"Aligned imports vs. dependencies for {len(iou_by_repo):,} repositories.")

    current_year = date.today().year
    pair_iou = (
        eligible.select(
            "document_id",
            "repository_id",
            "document_publication_year",
            "repository_primary_language",
        )
        .unique(subset=["document_id", "repository_id"])
        .with_columns(
            pl.col("repository_id").replace_strict(iou_by_repo, default=None).alias("iou")
        )
        .drop_nulls("iou")
        .filter(pl.col("document_publication_year") < current_year)
    )

    # Unconditional companion: pairs whose repo has imports OR manifest deps; a
    # single-view pair enters as IoU=0 (IoU on two sets, one empty, is 0); pairs with
    # neither view stay excluded (IoU undefined on two empty sets).
    single_view_pairs = (
        df.filter(pl.col("repository_id").is_in(either_view_repos - both_view_repos))
        .select(
            "document_id",
            "repository_id",
            "document_publication_year",
            "repository_primary_language",
        )
        .unique(subset=["document_id", "repository_id"])
        .with_columns(pl.lit(0.0).alias("iou"))
        .filter(pl.col("document_publication_year") < current_year)
    )
    pair_iou_unconditional = pl.concat([pair_iou, single_view_pairs])
    print(
        f"Unconditional population: {pair_iou_unconditional.height:,} pairs "
        f"({single_view_pairs.height:,} single-view pairs enter as IoU=0)"
    )

    def _agg_by_year(frame: pl.DataFrame, language: str, variant: str) -> pl.DataFrame:
        return (
            frame.group_by("document_publication_year")
            .agg(
                pl.len().alias("n_pairs"),
                pl.median("iou").alias("median_iou"),
                pl.mean("iou").alias("mean_iou"),
                pl.col("iou").quantile(0.25).alias("p25_iou"),
                pl.col("iou").quantile(0.75).alias("p75_iou"),
            )
            .with_columns(pl.lit(language).alias("language"), pl.lit(variant).alias("variant"))
            .sort("document_publication_year")
        )

    by_year = pl.concat(
        [
            _agg_by_year(pair_iou, "pooled", "conditional"),
            _agg_by_year(
                pair_iou.filter(pl.col("repository_primary_language") == "Python"),
                "Python",
                "conditional",
            ),
            _agg_by_year(
                pair_iou.filter(pl.col("repository_primary_language") == "R"),
                "R",
                "conditional",
            ),
            _agg_by_year(pair_iou_unconditional, "pooled", "unconditional"),
        ]
    )
    u.save_table(by_year, "unit15_import_dependency_iou_by_year", output_dir)

    # ---- Fixed-cohort robustness re-run: hold the repos already manifest-eligible by the
    # cohort year fixed, recompute the conditional trend within them -- separates real
    # per-pair decline from manifest-adoption composition shift. ----
    repo_first_year = (
        eligible.group_by("repository_id")
        .agg(pl.min("document_publication_year").alias("first_year"))
        .filter(pl.col("first_year") <= UNIT15_FIXED_COHORT_YEAR)
    )
    cohort_repo_ids = set(repo_first_year.get_column("repository_id").to_list())
    cohort_pair_iou = pair_iou.filter(pl.col("repository_id").is_in(cohort_repo_ids))
    fixed_cohort_by_year = _agg_by_year(
        cohort_pair_iou, "pooled", f"fixed_cohort_le_{UNIT15_FIXED_COHORT_YEAR}"
    ).with_columns(
        pl.lit(
            f"Repositories with >=1 import and >=1 pypi/conda/cran manifest dependency "
            f"whose first linked pair published <= {UNIT15_FIXED_COHORT_YEAR}"
        ).alias("cohort_definition")
    )
    u.save_table(fixed_cohort_by_year, "unit15_fixed_cohort_iou_by_year", output_dir)
    print(
        f"\nFixed-cohort robustness re-run ({len(cohort_repo_ids):,} repos eligible by "
        f"{UNIT15_FIXED_COHORT_YEAR}, {cohort_pair_iou.height:,} pairs):"
    )
    print(fixed_cohort_by_year.select(pl.exclude("cohort_definition")))

    # ---- Composition decomposition: per year, among import-bearing pairs, the
    # share with (i) no manifest deps, (ii) manifest deps but zero overlap with imports,
    # (iii) manifest deps with >=1 overlap. ----
    band_by_repo: dict[int, str] = {}
    for rid in repo_with_import:
        if rid not in both_view_repos:
            band_by_repo[rid] = "no_manifest"
        elif rid in iou_by_repo and iou_by_repo[rid] > 0:
            band_by_repo[rid] = "manifest_with_overlap"
        else:
            band_by_repo[rid] = "manifest_no_overlap"
    composition = (
        df.filter(pl.col("repository_id").is_in(repo_with_import))
        .select("document_id", "repository_id", "document_publication_year")
        .unique(subset=["document_id", "repository_id"])
        .filter(pl.col("document_publication_year") < current_year)
        .with_columns(
            pl.col("repository_id")
            .replace_strict(band_by_repo, default="no_manifest")
            .alias("band")
        )
        .group_by(["document_publication_year", "band"])
        .agg(pl.len().alias("n_pairs"))
        .sort(["document_publication_year", "band"])
    )
    composition = composition.join(
        composition.group_by("document_publication_year").agg(
            pl.sum("n_pairs").alias("n_pairs_year_total")
        ),
        on="document_publication_year",
    ).with_columns(
        (100 * pl.col("n_pairs") / pl.col("n_pairs_year_total")).alias("pct_of_pairs")
    )
    u.save_table(composition, "unit15_composition_decomposition_by_year", output_dir)

    band_order = ["no_manifest", "manifest_no_overlap", "manifest_with_overlap"]
    band_labels = {
        "no_manifest": "No manifest dependencies",
        "manifest_no_overlap": "Manifest, zero overlap with imports",
        "manifest_with_overlap": "Manifest, >=1 overlap with imports",
    }
    comp_plotted = composition.filter(pl.col("n_pairs_year_total") >= min_pairs_per_year)
    comp_years = sorted(comp_plotted.get_column("document_publication_year").unique().to_list())
    band_matrix = []
    for band in band_order:
        row = []
        for year in comp_years:
            cell = comp_plotted.filter(
                (pl.col("document_publication_year") == year) & (pl.col("band") == band)
            )
            row.append(float(cell.get_column("pct_of_pairs")[0]) if cell.height else 0.0)
        band_matrix.append(row)
    fig_comp, ax_comp = plt.subplots(figsize=(9, 5.5))
    comp_colors = u.field_palette(3)
    ax_comp.stackplot(
        comp_years,
        band_matrix,
        labels=[band_labels[b] for b in band_order],
        colors=comp_colors,
        alpha=0.85,
    )
    ax_comp.set_xlabel("Publication Year")
    ax_comp.set_ylabel("% of Import-Bearing Pairs")
    ax_comp.set_ylim(0, 100)
    leg_comp = ax_comp.legend(
        fontsize=7, title="", loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2
    )
    u.style_legend(leg_comp)
    u.add_footnote(
        ax_comp,
        f"Years with < {min_pairs_per_year} import-bearing pairs and the current partial "
        "year excluded",
        loc="bottom center outside",
    )
    u.shrink_ticks(ax_comp, size=9)
    evaplot.adjust_layout(fig_comp, bottom=0.18)
    u.save_figure(fig_comp, "unit15_composition_decomposition", output_dir)
    plt.close(fig_comp)

    # ---- Directional top-5 unmatched lists ----
    pair_counts_by_repo = dict(
        pair_iou.group_by("repository_id").agg(pl.len().alias("n")).iter_rows()
    )
    unmatched_rows = _top5_unmatched_rows(
        [
            ("imports_not_matched_to_dependencies", unmatched_imports_by_repo),
            ("dependencies_not_matched_to_imports", unmatched_deps_by_repo),
        ],
        pair_counts_by_repo,
        n_eligible_pairs=pair_iou.height,
    )
    u.save_table(pl.DataFrame(unmatched_rows), "unit15_unmatched_top5_by_direction", output_dir)

    plotted = by_year.filter(
        (pl.col("language") == "pooled")
        & (pl.col("variant") == "conditional")
        & (pl.col("n_pairs") >= min_pairs_per_year)
    )
    print("\nMedian import-vs-dependency IoU by year (pooled conditional, n-floor applied):")
    print(plotted)
    plotted_uncond = by_year.filter(
        (pl.col("language") == "pooled")
        & (pl.col("variant") == "unconditional")
        & (pl.col("n_pairs") >= min_pairs_per_year)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    years = plotted.get_column("document_publication_year").to_numpy()
    palette2 = evaplot.set_cat_palette(n=2)
    line_color, uncond_color = palette2[0], palette2[1]
    ax.fill_between(
        years,
        plotted.get_column("p25_iou").to_numpy(),
        plotted.get_column("p75_iou").to_numpy(),
        color=line_color,
        alpha=0.18,
        label="Interquartile range (conditional)",
    )
    ax.plot(
        years,
        plotted.get_column("median_iou").to_numpy(),
        marker="o",
        color=line_color,
        linewidth=1.8,
        label="Median IoU (conditional: both views present)",
    )
    ax.plot(
        plotted_uncond.get_column("document_publication_year").to_numpy(),
        plotted_uncond.get_column("median_iou").to_numpy(),
        marker="s",
        markersize=4,
        color=uncond_color,
        linewidth=1.4,
        linestyle="--",
        label="Median IoU (unconditional: single-view pairs = 0)",
    )
    ax.plot(
        plotted_uncond.get_column("document_publication_year").to_numpy(),
        plotted_uncond.get_column("mean_iou").to_numpy(),
        marker="^",
        markersize=4,
        color=uncond_color,
        linewidth=1.2,
        linestyle=":",
        label="Mean IoU (unconditional)",
    )
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Import-vs-Dependency IoU")
    ax.set_ylim(0, 1)
    u.style_legend(ax.legend(fontsize=7, loc="upper left"), fontsize=7)
    u.add_footnote(
        ax,
        f"Years with < {min_pairs_per_year} eligible pairs and the current partial year "
        "excluded",
        loc="lower right",
    )
    u.shrink_ticks(ax, size=9)
    evaplot.adjust_layout(fig)
    u.save_figure(fig, "unit15_import_dependency_iou_over_time", output_dir)
    plt.close(fig)


###############################################################################
# Unit 16 -- mentions-extraction coverage by publication year (diagnostic)


@app.command()
def unit16_mentions_coverage_by_year(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Diagnose whether the post-2022 mentions gap is real or an rs-graph filtering artifact.
    Loads `document` and `document_software_mention` UNFILTERED
    (no confidence/year filters -- the question is about the raw extraction, not the analysis
    subset) and reports, per publication year: document count, documents with >=1 extracted
    mention, mention-row count, and % of documents with a mention. If coverage collapses at a
    hard year in the unfiltered table, the gap is an upstream SoftCite-2025 extraction-horizon
    cutoff, not an rs-graph join artifact -- the verdict is printed and written to the CSV.
    """
    print("Loading document and document_software_mention (UNFILTERED) from HuggingFace...")
    documents = u.load_table("document")
    mentions = u.load_table("document_software_mention")
    print(f"  document: {len(documents):,} rows")
    print(f"  document_software_mention: {len(mentions):,} rows")

    doc_years = documents.select(
        "id",
        pl.col("publication_date")
        .str.to_date("%Y-%m-%d", strict=False)
        .dt.year()
        .alias("publication_year"),
    )
    mention_counts = mentions.group_by("document_id").agg(pl.len().alias("n_mention_rows"))
    joined = doc_years.join(
        mention_counts, left_on="id", right_on="document_id", how="left"
    ).with_columns(pl.col("n_mention_rows").fill_null(0))

    by_year = (
        joined.drop_nulls("publication_year")
        .group_by("publication_year")
        .agg(
            pl.len().alias("n_documents"),
            (pl.col("n_mention_rows") > 0).sum().alias("n_documents_with_mention"),
            pl.sum("n_mention_rows").alias("n_mention_rows"),
        )
        .with_columns(
            (100 * pl.col("n_documents_with_mention") / pl.col("n_documents")).alias(
                "pct_docs_with_mention"
            )
        )
        .sort("publication_year")
    )

    # Verdict: find the last year with meaningful coverage (>= 20% of the peak coverage rate
    # among years with >= 1,000 documents), then check whether later years collapse to ~zero.
    substantive = by_year.filter(pl.col("n_documents") >= 1000)
    peak_pct = float(substantive.get_column("pct_docs_with_mention").max())
    covered_years = substantive.filter(
        pl.col("pct_docs_with_mention") >= 0.2 * peak_pct
    ).get_column("publication_year")
    last_covered_year = int(covered_years.max())
    post = substantive.filter(pl.col("publication_year") > last_covered_year)
    verdict = (
        f"Coverage collapses after {last_covered_year} in the UNFILTERED mention table -- "
        "consistent with an upstream SoftCite-2025 extraction-horizon cutoff, not an "
        "rs-graph filtering artifact."
        if post.height == 0 or post.get_column("pct_docs_with_mention").max() < 0.2 * peak_pct
        else "No hard coverage collapse detected -- investigate rs-graph joins."
    )
    by_year = by_year.with_columns(pl.lit(verdict).alias("verdict"))

    u.save_table(by_year, "unit16_mentions_coverage_by_year", output_dir)
    print("\n--- Mentions-extraction coverage by publication year (unfiltered) ---")
    print(by_year.select(pl.exclude("verdict")))
    print(f"\nVERDICT: {verdict}")
    print("----------------------------------------------------------------------\n")


###############################################################################
# Unit 17 -- mentions-vs-imports and mentions-vs-dependencies alignment over time

# Minimum pairs per year for plotted lines; CSVs keep every year.
UNIT17_MIN_PAIRS_PER_YEAR = 50
# Same mentions-extraction horizon as the regression's cap -- mention extraction is
# absent/partial after this year, so later years would show a coverage artifact, not an
# alignment trend.
UNIT17_YEAR_CAP = UNIT5_YEAR_CAP

# Generic mention terms removed for the generic-filtered IoU variant. Deliberately NOT
# listed: named software and languages (matlab, samtools, python, r, bioconductor,
# jupyter) -- real referents whose non-matching is the construct-mismatch signal, not noise.
# All names compared post-normalize_name.
GENERIC_MENTION_STOPLIST: set[str] = {
    "code",
    "scripts",
    "script",
    "latex",
    "software",
    "github",
    "gitlab",
    "bitbucket",
    "library",
    "libraries",
    "package",
    "packages",
    "tool",
    "tools",
    "toolbox",
    "pipeline",
    "workflow",
    "api",
    "database",
    "website",
    "server",
    "notebook",
}


@app.command()
def unit17_mentions_alignment_over_time(
    output_dir: Path = OUTPUT_DIR,
    cutoff: float = 85.0,
    min_pairs_per_year: int = UNIT17_MIN_PAIRS_PER_YEAR,
    year_cap: int = UNIT17_YEAR_CAP,
) -> None:
    """
    Compute mentions-vs-imports and mentions-vs-dependencies alignment over time. Per
    (document, repository) pair with >=1 extracted mention and >=1 import (resp. >=1 pooled
    pypi/conda/cran manifest dependency), mentions are Hungarian-aligned against imports
    (resp. dependencies), imports/deps as `items_a` (canonical); IoU = |matched| / |union|.
    Every mentions-based IoU is reported twice -- as-is, and with `GENERIC_MENTION_STOPLIST`
    terms removed from the mention sets before alignment (per-term removed counts saved
    alongside) -- plus an unconditional companion variant where single-view pairs enter as
    IoU=0, and directional top-5 unmatched lists (from the as-is variant). Dependency names
    are cleaned before alignment. Capped at `year_cap` for the mentions-extraction horizon.
    """
    evaplot.set_style("evaplot_rc")
    df = u.load_filtered_pairs(top_n_fields=10)

    print("\nLoading imports/dependencies/mentions from HuggingFace...")
    imports = u.load_table("repository_import")
    deps = u.load_table("repository_dependency")
    mentions = u.load_table("document_software_mention")
    deps = u.clean_dependency_names(deps)

    manifest_ecosystems = sorted(
        {e for ecos in TABLE1_DEPENDENCY_ECOSYSTEMS.values() for e in ecos}
    )
    deps_pr = deps.filter(pl.col("ecosystem").is_in(manifest_ecosystems))

    imports_by_repo: dict[int, list[str]] = {
        rid[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for rid, grp in imports.group_by("repository_id")
    }
    deps_by_repo: dict[int, list[str]] = {
        rid[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for rid, grp in deps_pr.group_by("repository_id")
    }
    mentions_by_doc: dict[int, list[str]] = {
        did[0]: grp.get_column("software_name_normalized").drop_nulls().to_list()
        for did, grp in mentions.group_by("document_id")
    }

    df = df.filter(pl.col("document_publication_year") <= year_cap)

    stoplist_removed_rows: list[dict] = []
    unmatched_specs: list[tuple[str, dict]] = []

    def _pair_iou_frame(
        repo_names: dict[int, list[str]], source_b_label: str, comparison: str
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return (conditional pair frame with as-is + generic-filtered IoU, zero-pad frame
        of single-view pairs for the unconditional variant).
        """
        eligible = df.filter(
            pl.col("repository_id").is_in(set(repo_names))
            & pl.col("document_id").is_in(set(mentions_by_doc))
        )
        print(
            f"\nmentions-vs-{source_b_label}: {eligible.height:,} of {df.height:,} pairs have "
            f">=1 extracted mention and >=1 {source_b_label}"
        )
        removed_counts: dict[str, int] = {}
        unmatched_mentions: dict[int, set[str]] = {}
        unmatched_items: dict[int, set[str]] = {}
        rows = []
        for pair_idx, row in enumerate(
            eligible.select("document_id", "repository_id", "document_publication_year")
            .unique(subset=["document_id", "repository_id"])
            .iter_rows(named=True)
        ):
            pair_b = repo_names[row["repository_id"]]
            pair_mentions = mentions_by_doc[row["document_id"]]
            matches = align_software_names(
                items_a=pair_b,
                items_b=pair_mentions,
                source_a=source_b_label,
                source_b="mention",
                cutoff=cutoff,
                method="global_min_diff",
            )
            matched_items = {m.normalized_item_one for m in matches}
            matched_mentions = {m.normalized_item_two for m in matches}
            unique_b = set(pair_b)
            unique_mentions = set(pair_mentions)
            n_matched = len(matched_items)
            union_size = len(unique_b) + len(unique_mentions) - n_matched
            iou_as_is = n_matched / union_size if union_size else float("nan")
            unmatched_mentions[pair_idx] = unique_mentions - matched_mentions
            unmatched_items[pair_idx] = unique_b - matched_items

            # Generic-filtered variant: same pair, stoplist terms removed from the mention
            # set before alignment. A pair whose mentions are all generic stays in the
            # population with IoU=0 (eligibility is defined on the raw mention set).
            generic_here = unique_mentions & GENERIC_MENTION_STOPLIST
            for term in generic_here:
                removed_counts[term] = removed_counts.get(term, 0) + 1
            if generic_here:
                filtered_mentions = [
                    m for m in pair_mentions if m not in GENERIC_MENTION_STOPLIST
                ]
                if filtered_mentions:
                    f_matches = align_software_names(
                        items_a=pair_b,
                        items_b=filtered_mentions,
                        source_a=source_b_label,
                        source_b="mention",
                        cutoff=cutoff,
                        method="global_min_diff",
                    )
                    f_matched = len({m.normalized_item_one for m in f_matches})
                    f_union = len(unique_b) + len(set(filtered_mentions)) - f_matched
                    iou_filtered = f_matched / f_union if f_union else float("nan")
                else:
                    iou_filtered = 0.0
            else:
                iou_filtered = iou_as_is
            rows.append(
                {
                    "document_publication_year": row["document_publication_year"],
                    "iou_as_is": iou_as_is,
                    "iou_generic_filtered": iou_filtered,
                }
            )

        for term, n in removed_counts.items():
            stoplist_removed_rows.append(
                {
                    "comparison": comparison,
                    "term": term,
                    "n_pairs_with_term_removed": n,
                    "n_eligible_pairs": len(rows),
                }
            )
        unmatched_specs.append(
            (f"mentions_not_matched_to_{source_b_label}s", unmatched_mentions)
        )
        unmatched_specs.append((f"{source_b_label}s_not_matched_to_mentions", unmatched_items))

        # Unconditional zero-pads: repo has the b-side view but the document has no extracted
        # mention, or the document has mentions but the repo lacks the b-side view.
        zero_pad = (
            df.filter(
                (
                    pl.col("repository_id").is_in(set(repo_names))
                    & ~pl.col("document_id").is_in(set(mentions_by_doc))
                )
                | (
                    pl.col("document_id").is_in(set(mentions_by_doc))
                    & ~pl.col("repository_id").is_in(set(repo_names))
                )
            )
            .select("document_id", "repository_id", "document_publication_year")
            .unique(subset=["document_id", "repository_id"])
            .select(
                pl.col("document_publication_year").cast(pl.Int64),
                pl.lit(0.0).alias("iou_as_is"),
                pl.lit(0.0).alias("iou_generic_filtered"),
            )
        )
        print(f"  unconditional companion adds {zero_pad.height:,} single-view pairs as IoU=0")
        return pl.DataFrame(rows), zero_pad

    frames: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {
        "mentions_vs_imports": _pair_iou_frame(
            imports_by_repo, "import", "mentions_vs_imports"
        ),
        "mentions_vs_dependencies": _pair_iou_frame(
            deps_by_repo, "dependency", "mentions_vs_dependencies"
        ),
    }

    if stoplist_removed_rows:
        u.save_table(
            pl.DataFrame(stoplist_removed_rows).sort(
                ["comparison", "n_pairs_with_term_removed"], descending=[False, True]
            ),
            "unit17_generic_stoplist_removed_counts",
            output_dir,
        )
    unmatched_rows = _top5_unmatched_rows(
        unmatched_specs,
        weight_by_key=None,
        n_eligible_pairs=-1,  # per-direction populations differ; overwritten below
    )
    # Fix per-direction eligible-pair counts (each direction's population is its comparison's
    # conditional population).
    n_eligible = {
        "mentions_not_matched_to_imports": frames["mentions_vs_imports"][0].height,
        "imports_not_matched_to_mentions": frames["mentions_vs_imports"][0].height,
        "mentions_not_matched_to_dependencys": frames["mentions_vs_dependencies"][0].height,
        "dependencys_not_matched_to_mentions": frames["mentions_vs_dependencies"][0].height,
    }
    for r in unmatched_rows:
        r["n_eligible_pairs"] = n_eligible.get(r["direction"], -1)
        r["direction"] = r["direction"].replace("dependencys", "dependencies")
    u.save_table(pl.DataFrame(unmatched_rows), "unit17_unmatched_top5_by_direction", output_dir)
    print("\nDirectional top-5 unmatched lists (as-is variant):")
    print(pl.DataFrame(unmatched_rows))

    def _agg(frame: pl.DataFrame, iou_col: str) -> pl.DataFrame:
        return (
            frame.group_by("document_publication_year")
            .agg(
                pl.len().alias("n_pairs"),
                pl.median(iou_col).alias("median_iou"),
                pl.mean(iou_col).alias("mean_iou"),
                pl.col(iou_col).quantile(0.25).alias("p25_iou"),
                pl.col(iou_col).quantile(0.75).alias("p75_iou"),
            )
            .sort("document_publication_year")
        )

    by_year_frames, trend_rows = [], []
    for comparison, (cond_frame, zero_pad) in frames.items():
        uncond_frame = pl.concat([cond_frame, zero_pad])
        for variant, frame in [("conditional", cond_frame), ("unconditional", uncond_frame)]:
            for mention_filter, iou_col in [
                ("as_is", "iou_as_is"),
                ("generic_filtered", "iou_generic_filtered"),
            ]:
                by_year_frames.append(
                    _agg(frame, iou_col).with_columns(
                        pl.lit(comparison).alias("comparison"),
                        pl.lit(variant).alias("variant"),
                        pl.lit(mention_filter).alias("mention_filter"),
                    )
                )
                rho, pval = spearmanr(
                    frame.get_column("document_publication_year").to_numpy(),
                    frame.get_column(iou_col).to_numpy(),
                )
                trend_rows.append(
                    {
                        "comparison": comparison,
                        "variant": variant,
                        "mention_filter": mention_filter,
                        "n_pairs": frame.height,
                        "spearman_rho_iou_vs_year": float(rho),
                        "p_value": float(pval),
                    }
                )
    by_year = pl.concat(by_year_frames)
    u.save_table(by_year, "unit17_mentions_alignment_iou_by_year", output_dir)
    trend = pl.DataFrame(trend_rows)
    u.save_table(trend, "unit17_mentions_alignment_trend_summary", output_dir)
    print("\nPer-pair IoU-vs-year Spearman trends:")
    print(trend)

    # ---- Figure: conditional variant, both comparisons, as-is (solid) vs.
    # generic-filtered (dashed) mean lines. Medians are ~0 everywhere (most pairs mention
    # none of their imports/deps), so the mean carries the trend signal; medians/IQR stay in
    # the CSV. ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = evaplot.set_cat_palette(n=2)
    labels = {
        "mentions_vs_imports": "Mentions vs. imports",
        "mentions_vs_dependencies": "Mentions vs. manifest dependencies",
    }
    for (comparison, label), color in zip(labels.items(), colors[: len(labels)], strict=True):
        for mention_filter, ls, marker, suffix in [
            ("as_is", "-", "o", "as-is"),
            ("generic_filtered", "--", "s", "generic terms filtered"),
        ]:
            plotted = by_year.filter(
                (pl.col("comparison") == comparison)
                & (pl.col("variant") == "conditional")
                & (pl.col("mention_filter") == mention_filter)
                & (pl.col("n_pairs") >= min_pairs_per_year)
            )
            ax.plot(
                plotted.get_column("document_publication_year").to_numpy(),
                plotted.get_column("mean_iou").to_numpy(),
                marker=marker,
                markersize=4,
                color=color,
                linewidth=1.6,
                linestyle=ls,
                label=f"{label} (mean, {suffix})",
            )
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Mentions Alignment IoU (mean)")
    y_hi = 1.4 * float(
        by_year.filter(pl.col("variant") == "conditional").get_column("mean_iou").max()
    )
    ax.set_ylim(0, max(0.05, y_hi))
    u.style_legend(ax.legend(fontsize=7, loc="upper left"), fontsize=7)
    u.add_footnote(
        ax,
        f"Pairs with >=1 extracted mention only (conditional); years after {year_cap} "
        f"(mentions-extraction horizon) and years with < {min_pairs_per_year} pairs "
        "excluded; medians (~0 throughout) in the CSV",
        loc="lower right",
    )
    u.shrink_ticks(ax, size=9)
    evaplot.adjust_layout(fig)
    u.save_figure(fig, "unit17_mentions_alignment_over_time", output_dir)
    plt.close(fig)


###############################################################################
# Supplemental -- package-shaped vs. script-shaped repository diagnostic (R + Python)

# Definitions on record: deterministic full-population computations, no sampling.
R_PACKAGE_DEFINITION = (
    "R-primary repository is 'package-shaped' iff it has >=1 dependency row with "
    "ecosystem == 'cran' and manifest_paths == 'DESCRIPTION' (root-level); everything else "
    "is 'script-shaped'. Repo year = first-seen linked publication year. Deterministic, "
    "full population (no sampling)."
)
PYTHON_PACKAGE_FILES = ("setup.py", "setup.cfg", "pyproject.toml")
PYTHON_PACKAGE_DEFINITION = (
    "Python-primary repository is 'package-shaped' iff it has >=1 dependency row with "
    "ecosystem == 'pypi' and manifest_paths in {setup.py, setup.cfg, pyproject.toml} "
    "(root-level); requirements-only/lockfile-only/no-manifest repos are 'script-shaped'. "
    "Caveats: setup.py historically doubled as a dependency-listing mechanism and "
    "pyproject.toml is increasingly used by non-packages for tool config, so this flag is "
    "noisier than R's CRAN DESCRIPTION and biases AGAINST finding a package-share decline. "
    "Repo year = first-seen linked publication year. Deterministic, full population."
)


@app.command()
def supplemental_package_vs_script_diagnostics(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Compute package-shaped vs. script-shaped repository composition over time, for R-primary
    and Python-primary repositories -- per-first-seen-publication-year shares plus median
    commit counts per shape, saved as a definition-labeled CSV.
    """
    df = u.load_filtered_pairs(top_n_fields=10)

    print("\nLoading repository_dependency from HuggingFace...")
    deps = u.load_table("repository_dependency")
    print(f"  repository_dependency: {len(deps):,} rows")

    repo_level = df.group_by("repository_id").agg(
        pl.min("document_publication_year").alias("first_seen_publication_year"),
        pl.first("repository_primary_language").alias("primary_language"),
        pl.first("repository_commits_count").alias("commits_count"),
    )

    r_package_repos = set(
        deps.filter(
            (pl.col("ecosystem") == "cran") & (pl.col("manifest_paths") == "DESCRIPTION")
        )
        .get_column("repository_id")
        .unique()
        .to_list()
    )
    py_package_repos = set(
        deps.filter(
            (pl.col("ecosystem") == "pypi")
            & pl.col("manifest_paths").is_in(list(PYTHON_PACKAGE_FILES))
        )
        .get_column("repository_id")
        .unique()
        .to_list()
    )

    frames = []
    for language, package_repo_ids, definition in [
        ("R", r_package_repos, R_PACKAGE_DEFINITION),
        ("Python", py_package_repos, PYTHON_PACKAGE_DEFINITION),
    ]:
        lang_repos = repo_level.filter(pl.col("primary_language") == language).with_columns(
            pl.col("repository_id").is_in(package_repo_ids).alias("package_shaped")
        )
        by_year = (
            lang_repos.group_by("first_seen_publication_year")
            .agg(
                pl.len().alias("n_repos"),
                pl.col("package_shaped").sum().alias("n_package_shaped"),
                pl.col("commits_count")
                .filter(pl.col("package_shaped"))
                .median()
                .alias("median_commits_package_shaped"),
                pl.col("commits_count")
                .filter(~pl.col("package_shaped"))
                .median()
                .alias("median_commits_script_shaped"),
            )
            .with_columns(
                (100 * pl.col("n_package_shaped") / pl.col("n_repos")).alias(
                    "pct_package_shaped"
                ),
                pl.lit(language).alias("primary_language"),
                pl.lit(definition).alias("definition"),
            )
            .sort("first_seen_publication_year")
        )
        frames.append(by_year)
        print(f"\n{language}-primary package-vs-script composition by first-seen year:")
        print(by_year.select(pl.exclude("definition")))

    out = pl.concat(frames).select(
        "primary_language",
        "first_seen_publication_year",
        "n_repos",
        "n_package_shaped",
        "pct_package_shaped",
        "median_commits_package_shaped",
        "median_commits_script_shaped",
        "definition",
    )
    u.save_table(out, "supplemental_package_vs_script_diagnostic", output_dir)


###############################################################################

if __name__ == "__main__":
    app()
