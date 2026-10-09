#!/usr/bin/env python3

"""Figure 3: software development characteristics (six panels plus the article-vs-preprint
supplemental), and the package-shaped vs. script-shaped repository diagnostic.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import evaplot
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import utils as u
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr

from rs_graph.utils.identifier_normalization import normalize_name

###############################################################################
# Dependency-category package lists

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

MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR = 100
LICENSE_ADOPTION_MIN_REPOS_PER_YEAR = 100

# License-category keyword lists for Panel D.
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

###############################################################################
# Panel data helpers


FIELD_RHO_MIN_PAIRS = 10


def _top_pruned_field_names(df: pl.DataFrame, n: int) -> list[str]:
    """Top-`n` pruned field names by pair count, excluding the "Other" bucket."""
    return (
        df.filter(pl.col("document_field_name_pruned") != "Other")
        .get_column("document_field_name_pruned")
        .value_counts(sort=True)
        .head(n)
        .get_column("document_field_name_pruned")
        .to_list()
    )


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


def _manifest_adoption_over_time(df: pl.DataFrame, deps: pl.DataFrame) -> pl.DataFrame:
    # Dedup to one row per repository (first-seen publication year) -- the panel reports a
    # "% of Repos" statistic, so a repository must not be counted once per linked paper.
    current_year = date.today().year
    year_repo = (
        df.select("repository_id", "document_publication_year", "repository_primary_language")
        .unique(subset="repository_id", keep="first")
        .filter(
            (pl.col("document_publication_year") >= u.DEFAULT_MIN_YEAR)
            & (pl.col("document_publication_year") < current_year)
        )
    )
    # Pooled series (all repos, Python/R manifests only -- pypi/conda/cran, matching this
    # analysis's Python/R scope) plus Python/R ecosystem reference series.
    series_specs: list[tuple[str, pl.DataFrame, pl.DataFrame]] = [
        (
            "All repositories",
            year_repo,
            deps.filter(pl.col("ecosystem").is_in(u.ALL_MANIFEST_ECOSYSTEMS))
            .select("repository_id")
            .unique(),
        )
    ]
    for eco, dep_ecosystems in u.MANIFEST_ECOSYSTEMS_BY_LANGUAGE.items():
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


PYTHON_SHARE_TOP_N_FIELDS = 5
PYTHON_SHARE_OVERALL_LABEL = "Overall (all fields)"


def _python_share_by_field_over_time(
    df: pl.DataFrame, top_n_fields: int = PYTHON_SHARE_TOP_N_FIELDS
) -> pl.DataFrame:
    """Python's share of repositories over time, per field. One row per (series, year) giving
    the % of that field's repositories whose primary language is Python, out of repositories
    with any known primary language. Series are the top-`top_n_fields` pruned fields, "Other",
    and a pooled all-fields line. Repo-year attribution and the per-year repository floor match
    `_manifest_adoption_over_time`.
    """
    current_year = date.today().year
    repo_year = (
        df.select(
            "repository_id",
            "document_publication_year",
            "repository_primary_language",
            "document_field_name_pruned",
        )
        .unique(subset="repository_id", keep="first")
        .filter(
            (pl.col("document_publication_year") >= u.DEFAULT_MIN_YEAR)
            & (pl.col("document_publication_year") < current_year)
        )
        .drop_nulls("repository_primary_language")
    )
    top_fields = _top_pruned_field_names(df, top_n_fields)
    repo_year = repo_year.with_columns(
        pl.when(pl.col("document_field_name_pruned").is_in(top_fields))
        .then(pl.col("document_field_name_pruned"))
        .otherwise(pl.lit("Other"))
        .alias("field")
    )

    def series_frame(repos: pl.DataFrame, label: str) -> pl.DataFrame:
        return (
            repos.group_by("document_publication_year")
            .agg(
                pl.len().alias("total"),
                (pl.col("repository_primary_language") == "Python")
                .sum()
                .alias("n_python_primary"),
            )
            .with_columns(
                pl.lit(label).alias("series"),
                (pl.col("n_python_primary") / pl.col("total") * 100).alias("pct_python"),
            )
            .filter(pl.col("total") >= MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR)
            .sort("document_publication_year")
        )

    frames = [series_frame(repo_year, PYTHON_SHARE_OVERALL_LABEL)]
    for field in [*top_fields, "Other"]:
        frames.append(series_frame(repo_year.filter(pl.col("field") == field), field))
    return pl.concat(frames).select(
        "series", "document_publication_year", "n_python_primary", "total", "pct_python"
    )


def _manifest_adoption_by_field_over_time(
    df: pl.DataFrame, deps: pl.DataFrame, top_n_fields: int = PYTHON_SHARE_TOP_N_FIELDS
) -> pl.DataFrame:
    """Per-field split of the pooled "All repositories" series from
    `_manifest_adoption_over_time`: % of repositories with a parsed Python/R (pypi/conda/cran)
    manifest, by first-seen publication year. Same repo-year attribution, numerator and
    per-year repository floor as that series; field buckets match
    `_python_share_by_field_over_time` (top-`top_n_fields` pruned fields, "Other", and a
    pooled all-fields line).
    """
    current_year = date.today().year
    repos_with_manifest = (
        deps.filter(pl.col("ecosystem").is_in(u.ALL_MANIFEST_ECOSYSTEMS))
        .select("repository_id")
        .unique()
        .with_columns(pl.lit(True).alias("has_manifest"))
    )
    top_fields = _top_pruned_field_names(df, top_n_fields)
    repo_year = (
        df.select("repository_id", "document_publication_year", "document_field_name_pruned")
        .unique(subset="repository_id", keep="first")
        .filter(
            (pl.col("document_publication_year") >= u.DEFAULT_MIN_YEAR)
            & (pl.col("document_publication_year") < current_year)
        )
        .join(repos_with_manifest, on="repository_id", how="left")
        .with_columns(
            pl.col("has_manifest").fill_null(False),
            pl.when(pl.col("document_field_name_pruned").is_in(top_fields))
            .then(pl.col("document_field_name_pruned"))
            .otherwise(pl.lit("Other"))
            .alias("field"),
        )
    )

    def series_frame(repos: pl.DataFrame, label: str) -> pl.DataFrame:
        return (
            repos.group_by("document_publication_year")
            .agg(pl.len().alias("total"), pl.col("has_manifest").sum().alias("count"))
            .with_columns(
                pl.lit(label).alias("series"),
                (pl.col("count") / pl.col("total") * 100).alias("pct_repos"),
            )
            .filter(pl.col("total") >= MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR)
            .sort("document_publication_year")
        )

    frames = [series_frame(repo_year, PYTHON_SHARE_OVERALL_LABEL)]
    for field in [*top_fields, "Other"]:
        frames.append(series_frame(repo_year.filter(pl.col("field") == field), field))
    return pl.concat(frames).select(
        "series", "document_publication_year", "count", "total", "pct_repos"
    )


def _dependency_category_adoption(
    df: pl.DataFrame,
    deps: pl.DataFrame,
    group_col: str | None,
    with_year: bool,
    repo_filter: pl.DataFrame | None = None,
    numerator_ecosystems: list[str] | None = None,
) -> pl.DataFrame:
    """Adoption of the five dependency categories, as a long (category, [group], [year])
    frame with count/total/pct_repos. Repo-year attribution: first-seen publication year per
    repository (same dedup rule as `_manifest_adoption_over_time`).

    `repo_filter`, if given, restricts the denominator population to a `repository_id` frame
    (e.g. repositories with any parsed manifest). `numerator_ecosystems`, if given, restricts
    which `repository_dependency` ecosystem rows can count toward a category hit.
    """
    base_cols = ["repository_id", "document_publication_year"]
    if group_col is not None:
        base_cols.append(group_col)
    repo_base = df.select(base_cols).unique(subset="repository_id", keep="first")
    if repo_filter is not None:
        repo_base = repo_base.join(repo_filter, on="repository_id", how="semi")
    key_cols = ([group_col] if group_col is not None else []) + (
        ["document_publication_year"] if with_year else []
    )
    if not key_cols:
        repo_base = repo_base.with_columns(pl.lit("all").alias("_all"))
        key_cols = ["_all"]
    total_per_group = repo_base.group_by(key_cols).agg(pl.len().alias("total"))

    numerator_deps = deps
    if numerator_ecosystems is not None:
        numerator_deps = deps.filter(pl.col("ecosystem").is_in(numerator_ecosystems))

    frames = []
    for category, pkg_set in _DEPENDENCY_CATEGORIES.items():
        # software_name_normalized is hyphen/underscore-stripped, so match on normalized names.
        norm_pkgs = [normalize_name(p) for p in pkg_set]
        cat_repos = (
            numerator_deps.filter(pl.col("software_name_normalized").is_in(norm_pkgs))
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


def _fwci_distribution_frame(fwci_docs: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Panel E's positive-FWCI frame plus the zero-FWCI count (drawn as its own bar)."""
    n_before_zero_filter = fwci_docs.filter(pl.col("document_raw_fwci").is_not_null()).height
    d = fwci_docs.filter(
        pl.col("document_raw_fwci").is_not_null() & (pl.col("document_raw_fwci") > 0)
    )
    n_zero_citation = n_before_zero_filter - d.height
    print(
        f"Figure 3 Panel E: {n_zero_citation:,} zero-FWCI documents (OpenAlex FWCI = 0.0) "
        "drawn as a dedicated bar left of the log-scale distribution"
    )
    p99 = d.get_column("document_raw_fwci").quantile(0.99)
    return d.filter(pl.col("document_raw_fwci") <= p99), n_zero_citation


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


def _fwsi_vs_fwci_by_field(
    df: pl.DataFrame, fwci_docs: pl.DataFrame, fwsi_repos: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-field Spearman rho between raw document FWCI and modified repository FWSI, at the
    pair level, over every pruned field with at least `FIELD_RHO_MIN_PAIRS` pairs.
    """
    field_map = df.select("document_id", "document_field_name_pruned").unique(
        subset="document_id", keep="first"
    )

    pair_level = (
        df.select("document_id", "repository_id")
        .join(fwci_docs.select("document_id", "document_raw_fwci"), on="document_id")
        .join(
            fwsi_repos.select("repository_id", "repository_modified_fwsi"), on="repository_id"
        )
        .join(field_map, on="document_id")
        .drop_nulls(["document_raw_fwci", "repository_modified_fwsi"])
    )
    print("\nFWSI-vs-FWCI pair counts per field, before the minimum-n floor:")
    print(
        pair_level.group_by("document_field_name_pruned")
        .agg(pl.len().alias("n_pairs"))
        .sort("n_pairs", descending=True)
    )

    rho_rows = []
    for field, grp in pair_level.group_by("document_field_name_pruned"):
        if grp.height < FIELD_RHO_MIN_PAIRS:
            print(f"  Excluding {field[0]}: only {grp.height} pairs")
            continue
        rho_rows.append(
            {
                "field": field[0],
                **_spearman_with_ci(
                    grp.get_column("document_raw_fwci").to_numpy(),
                    grp.get_column("repository_modified_fwsi").to_numpy(),
                ),
            }
        )
    rho_df = pl.DataFrame(rho_rows).sort("rho", descending=True)
    return pair_level, rho_df


def _stars_vs_citations_by_field(df: pl.DataFrame) -> pl.DataFrame:
    """Per-field Spearman rho between raw repository stargazer counts and raw document
    citation counts, at the pair level, over every pruned field, plus a pooled/overall row.
    """
    pair_level = df.select(
        "document_id",
        "repository_id",
        "document_cited_by_count",
        "repository_stargazers_count",
        "document_field_name_pruned",
    ).drop_nulls(["document_cited_by_count", "repository_stargazers_count"])
    rows = [
        {
            "field": "All fields (pooled)",
            **_spearman_with_ci(
                pair_level.get_column("repository_stargazers_count").to_numpy(),
                pair_level.get_column("document_cited_by_count").to_numpy(),
            ),
        }
    ]
    for field, grp in pair_level.group_by("document_field_name_pruned"):
        if grp.height < FIELD_RHO_MIN_PAIRS:
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


def _license_adoption_over_time(df: pl.DataFrame) -> pl.DataFrame:
    """Per first-seen publication year: % of repos with any license, a permissive license,
    or a copyleft license. Licenses matching neither keyword list (incl. GitHub's literal
    "Other") count toward "any" only.
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


def _plot_dependency_category_adoption(
    ax: Axes, category_by_year: pl.DataFrame, legend_loc: str = "upper left"
) -> None:
    """Draw the five dependency-category adoption lines by publication year onto `ax`, over
    years with at least `MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR` repositories in the denominator.
    """
    sns.lineplot(
        data=category_by_year.filter(
            pl.col("total") >= MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR
        ).to_pandas(),
        x="document_publication_year",
        y="pct_repos",
        hue="category",
        style="category",
        hue_order=list(_DEPENDENCY_CATEGORIES),
        style_order=list(_DEPENDENCY_CATEGORIES),
        palette=u.general_palette(len(_DEPENDENCY_CATEGORIES)),
        markers=True,
        dashes=True,
        ax=ax,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Publication Year")
    u.style_legend(ax.legend(fontsize=8, title="", loc=legend_loc), fontsize=8)
    u.shrink_ticks(ax, size=9)


FIELD_LINE_ALPHA = 0.4


def _plot_field_lines(
    ax: Axes,
    frame: pl.DataFrame,
    value_col: str,
    fields: list[str],
    field_colors: dict[str, str],
) -> None:
    """Panel B/C style: one translucent line per field plus an opaque black dashed pooled
    line drawn on top, with the field legend in the (data-free) upper-left corner.
    """
    for series_label in [*fields, PYTHON_SHARE_OVERALL_LABEL]:
        plotted = frame.filter(pl.col("series") == series_label)
        is_overall = series_label == PYTHON_SHARE_OVERALL_LABEL
        ax.plot(
            plotted.get_column("document_publication_year").to_numpy(),
            plotted.get_column(value_col).to_numpy(),
            marker="o",
            markersize=3,
            linewidth=2.2 if is_overall else 1.5,
            linestyle="--" if is_overall else "-",
            color="black" if is_overall else field_colors[series_label],
            alpha=1.0 if is_overall else FIELD_LINE_ALPHA,
            label="Overall" if is_overall else u.abbreviate_field(series_label),
            zorder=3 if is_overall else 2,
        )
    u.style_legend(
        ax.legend(fontsize=6, title="", loc="upper left", ncol=2, columnspacing=0.8),
        fontsize=6,
    )


###############################################################################


def figure_3_software_development_characteristics(
    output_dir: Path = u.OUTPUT_DIR,
) -> None:
    """
    Build Figure 3: software development characteristics, one consolidated 6-panel figure.
    (A) dev activity duration, (B) Python's share of repositories over time by field, (C)
    Python/R dependency-manifest adoption over time by field, with a pooled line, (D) license adoption over time
    (any / permissive / copyleft), (E) raw-FWCI distribution, (F) Spearman rho by field as a
    horizontal forest plot, raw stars vs. raw citations (with a pooled row) alongside modified
    FWSI vs. raw FWCI. Also produces the dependency-category (tooling) adoption and
    article-vs-preprint supplementals from the same computations.
    """
    evaplot.set_style("evaplot_rc")
    # Stable year sort so every keep="first" dedup below attributes the earliest publication.
    df = u.load_filtered_pairs(top_n_fields=10).sort(
        "document_publication_year", maintain_order=True
    )

    print("\nLoading repository_dependency from HuggingFace...")
    deps = u.load_table("repository_dependency")
    print(f"  repository_dependency: {len(deps):,} rows")
    deps = u.clean_dependency_names(deps)

    # Raw OpenAlex FWCI everywhere FWCI appears.
    fwci_docs = u.raw_fwci_docs(df)
    fwsi_repos = u.compute_modified_fwsi(df)

    dev_duration = _dev_duration_frame(df)
    manifest_adoption = _manifest_adoption_over_time(df, deps)
    python_share = _python_share_by_field_over_time(df)
    manifest_by_field = _manifest_adoption_by_field_over_time(df, deps)
    # Tooling denominator: repositories with any parsed manifest, not every linked repository
    # -- otherwise the tooling series partly just measures manifest adoption itself. Numerator
    # restricted to pypi/cran/conda ecosystems, matching the manifest-adoption measure.
    repos_with_any_manifest = deps.select("repository_id").unique()
    category_by_year_all_repos = _dependency_category_adoption(
        df, deps, group_col=None, with_year=True
    )
    category_by_year = _dependency_category_adoption(
        df,
        deps,
        group_col=None,
        with_year=True,
        repo_filter=repos_with_any_manifest,
        numerator_ecosystems=u.ALL_MANIFEST_ECOSYSTEMS,
    )
    category_by_domain = _dependency_category_adoption(
        df, deps, group_col="document_domain_name", with_year=False
    )
    category_by_field_year = _dependency_category_adoption(
        df, deps, group_col="document_field_name_pruned", with_year=True
    )
    fwci_dist, n_zero_fwci_dropped = _fwci_distribution_frame(fwci_docs)
    _pair_level_fwsi_fwci, rho_df = _fwsi_vs_fwci_by_field(df, fwci_docs, fwsi_repos)

    # Vanishingly small p-values print as a bound rather than a literal 0.0.
    format_p = pl.col("p_value").map_elements(u.format_p_value, return_dtype=pl.String)

    u.save_table(
        rho_df.with_columns(format_p), "figure3_fwsi_fwci_spearman_by_field", output_dir
    )
    print("\nFWSI-vs-raw-FWCI Spearman rho by field (Panel F, square markers):")
    print(rho_df)

    stars_rho_df = _stars_vs_citations_by_field(df)
    u.save_table(
        stars_rho_df.with_columns(format_p),
        "figure3_stars_citations_spearman_by_field",
        output_dir,
    )
    print("\nRaw-stars-vs-raw-citations Spearman rho by field (Panel F, circle markers):")
    print(stars_rho_df)

    license_adoption = _license_adoption_over_time(df)
    u.save_table(license_adoption, "figure3_license_adoption_by_year", output_dir)
    print("\nLicense adoption by year (Panel D):")
    print(license_adoption)

    # Panel B/C data as labeled tables; Panel C's pooled line is the "All repositories"
    # series, checked against the per-field frame's pooled line below.
    u.save_table(manifest_adoption, "figure3_manifest_adoption_by_year", output_dir)
    u.save_table(python_share, "figure3_python_share_by_field_over_year", output_dir)
    u.save_table(manifest_by_field, "figure3_manifest_adoption_by_field_over_year", output_dir)
    manifest_by_field_pooled = manifest_by_field.filter(
        pl.col("series") == PYTHON_SHARE_OVERALL_LABEL
    ).select("document_publication_year", "count", "total")
    manifest_all_repos = manifest_adoption.filter(pl.col("series") == "All repositories")
    assert manifest_by_field_pooled.equals(
        manifest_all_repos.select("document_publication_year", "count", "total")
    ), "Panel C per-field pooled line diverges from the 'All repositories' series"
    current_year = date.today().year
    category_by_year_plotted = category_by_year.filter(
        pl.col("document_publication_year") < current_year
    )
    category_by_year_all_repos_plotted = category_by_year_all_repos.filter(
        pl.col("document_publication_year") < current_year
    )
    # Tooling data (former Panel C, now a supplementary candidate): with-manifest
    # denominator, pypi/cran/conda-only numerator. Filename kept for continuity.
    u.save_table(
        category_by_year_plotted, "figure3_dependency_category_adoption_by_year", output_dir
    )
    # All-repositories/all-ecosystem version -- Supplement only.
    u.save_table(
        category_by_year_all_repos_plotted,
        "supplemental_dependency_category_adoption_all_repos_by_year",
        output_dir,
    )
    print("\nTooling adoption (Supplement), with-manifest denominator vs. all-repos:")
    endpoint_years = [
        category_by_year_plotted.get_column("document_publication_year").min(),
        category_by_year_plotted.get_column("document_publication_year").max(),
    ]
    for cat_frame, cat_label in [
        (category_by_year_plotted, "with-manifest denominator, pypi/cran/conda numerator"),
        (category_by_year_all_repos_plotted, "all repos, all-ecosystem numerator"),
    ]:
        for category in _DEPENDENCY_CATEGORIES:
            row = cat_frame.filter(
                (pl.col("category") == category)
                & pl.col("document_publication_year").is_in(endpoint_years)
            ).sort("document_publication_year")
            vals = ", ".join(
                f"{r['document_publication_year']}: {r['pct_repos']:.1f}% (n={r['total']:,})"
                for r in row.iter_rows(named=True)
            )
            print(f"  [{cat_label}] {category}: {vals}")
    # Documents with no OpenAlex domain are labeled "Unknown" rather than left blank.
    category_by_domain = category_by_domain.with_columns(
        pl.col("document_domain_name").fill_null("Unknown").replace({"": "Unknown"})
    )
    u.save_table(category_by_domain, "figure3_dependency_category_by_domain", output_dir)
    u.save_table(
        category_by_field_year,
        "supplemental_dependency_category_by_field_and_year",
        output_dir,
    )

    # FWCI summary stats -- headline median and per-field medians.
    fwci_nonnull = fwci_docs.filter(pl.col("document_raw_fwci").is_not_null())
    fwci_median_incl_zero = fwci_nonnull.get_column("document_raw_fwci").median()
    fwci_median_excl_zero = (
        fwci_nonnull.filter(pl.col("document_raw_fwci") > 0)
        .get_column("document_raw_fwci")
        .median()
    )
    fwci_median_plotted = fwci_dist.get_column("document_raw_fwci").median()
    assert isinstance(fwci_median_incl_zero, float)
    assert isinstance(fwci_median_excl_zero, float)
    assert isinstance(fwci_median_plotted, float)
    fwci_summary = pl.DataFrame(
        {
            "statistic": [
                "median_raw_fwci_incl_zero_citation_docs",
                "median_raw_fwci_excl_zero_citation_docs",
                "median_raw_fwci_plotted_distribution_zero_dropped_p99_capped",
                "n_documents_with_raw_fwci",
                "pct_documents_with_raw_fwci",
                "n_zero_fwci_documents_in_panel_d_zero_bar",
            ],
            "value": [
                fwci_median_incl_zero,
                fwci_median_excl_zero,
                fwci_median_plotted,
                float(fwci_nonnull.height),
                100 * fwci_nonnull.height / fwci_docs.height,
                float(n_zero_fwci_dropped),
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

    # ---- 2x3 grid, six panels; sized so fonts stay legible at Nature's 183mm print width ----
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=2.6)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2:4])
    ax_f = fig.add_subplot(gs[1, 4:6])

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], "ABCDEF", strict=True):
        u.add_panel_label(ax, label)

    # A: dev activity duration by source -- horizontal boxplots; the "Negative = ..."
    # explainer lives in the figure caption.
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
        fontsize=8, title="", loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2
    )
    u.style_legend(leg)
    ax_a.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
    # Cap x-limits to the 2nd-98th percentile so long-tail whiskers don't squeeze the boxes.
    u.cap_ylim_to_quantiles(ax_a, dev_duration.to_pandas()["duration_years"], axis="x")
    u.shrink_ticks(ax_a, size=9)

    # B and C: one line per field plus a pooled line, sharing fields, colors and legend.
    python_share_fields = [
        s
        for s in python_share.get_column("series").unique(maintain_order=True).to_list()
        if s != PYTHON_SHARE_OVERALL_LABEL
    ]
    field_colors = u.field_color_map(python_share_fields)
    _plot_field_lines(ax_b, python_share, "pct_python", python_share_fields, field_colors)
    _plot_field_lines(ax_c, manifest_by_field, "pct_repos", python_share_fields, field_colors)
    ax_b.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_b.set_xlabel("Publication Year")
    ax_b.set_ylabel("% of Repos Python-Primary")
    ax_b.set_ylim(0, 100)
    u.print_caption_note(
        "figure3 Panel B",
        f"Share of repositories with a known primary language whose primary language is "
        f"Python, by first-seen publication year; top {PYTHON_SHARE_TOP_N_FIELDS} fields + "
        f"Other + a pooled all-fields line; years with "
        f"< {MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR} repos in a series excluded. Abbreviated "
        "field labels: " + u.field_abbreviation_caption(python_share_fields),
    )
    u.shrink_ticks(ax_b, size=9)

    # C: manifest adoption over time -- share of repositories with a parsed Python/R
    # (pypi/conda/cran) manifest, per field plus pooled (plotted above with B). Per-language
    # series stay in the Supplement.
    ax_c.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_c.set_xlabel("Publication Year")
    ax_c.set_ylabel("% of Repos with Manifest")
    ax_c.set_ylim(0, 100)
    u.shrink_ticks(ax_c, size=9)
    u.print_caption_note(
        "figure3 Panel C",
        "Share of repositories with a parsed Python or R dependency manifest (pypi, conda or "
        f"cran), by first-seen publication year; top {PYTHON_SHARE_TOP_N_FIELDS} fields + "
        "Other (translucent, same field colors as Panel B) + a pooled all-repositories "
        f"line (black dashed); years with < {MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR} repos in a "
        "series excluded",
    )

    # D: license adoption over time -- any / permissive / copyleft license share of repos by
    # first-seen publication year.
    license_plotted = license_adoption.filter(
        pl.col("n_repos") >= LICENSE_ADOPTION_MIN_REPOS_PER_YEAR
    )
    lic_colors = u.general_palette(3)
    for col, label, color, ls in [
        ("pct_any_license", "Any license", lic_colors[0], "-"),
        ("pct_permissive", "Permissive", lic_colors[1], "--"),
        ("pct_copyleft", "Copyleft", lic_colors[2], "-."),
    ]:
        ax_d.plot(
            license_plotted.get_column("document_publication_year").to_numpy(),
            license_plotted.get_column(col).to_numpy(),
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            linestyle=ls,
            color=color,
            label=label,
        )
    ax_d.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_d.set_xlabel("Publication Year")
    ax_d.set_ylabel("% of Repos")
    ax_d.set_ylim(0, 100)
    u.style_legend(ax_d.legend(fontsize=8, title="", loc="upper right"), fontsize=8)
    u.print_caption_note(
        "figure3 Panel D",
        f"Years with < {LICENSE_ADOPTION_MIN_REPOS_PER_YEAR} repos excluded; unclassifiable "
        "licenses (e.g. GitHub's 'Other') count toward 'Any license' only",
    )
    u.shrink_ticks(ax_d, size=9)

    # E: raw OpenAlex FWCI distribution. bins=25: seaborn's log-scale "auto" rule produced
    # jagged bin-to-bin noise unrelated to real signal.
    sns.histplot(
        data=fwci_dist.to_pandas(), x="document_raw_fwci", ax=ax_e, log_scale=True, bins=25
    )
    # Zero-FWCI docs can't sit on a log axis; drawn as a detached bar left of the
    # positive distribution so all documents stay visible. Chosen over log1p, which would
    # move FWCI=1 off the axis's natural reference point and break the field-avg line.
    pos = fwci_dist.get_column("document_raw_fwci")
    pos_min, pos_max = pos.min(), pos.max()
    assert isinstance(pos_min, float)
    assert isinstance(pos_max, float)
    bin_ratio = (pos_max / pos_min) ** (1 / 25)
    zero_x = pos_min / bin_ratio**2.5
    ax_e.bar(
        zero_x,
        n_zero_fwci_dropped,
        width=zero_x * (bin_ratio - 1),
        color=u.general_palette(2)[1],
        label=f"FWCI = 0 (n={n_zero_fwci_dropped:,})",
    )
    ax_e.set_xlim(left=zero_x / bin_ratio)
    # Median over every document with an FWCI, zeros included, matching the plotted data.
    ax_e.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Field avg.")
    ax_e.axvline(
        fwci_median_incl_zero,
        color="red",
        linestyle="-.",
        linewidth=1.5,
        label=f"Median: {fwci_median_incl_zero:.2f}",
    )
    ax_e.set_xlabel("OpenAlex FWCI (log scale)")
    ax_e.set_ylabel("Count")
    # Upper left keeps the legend clear of Panel F's long y-tick labels.
    u.style_legend(ax_e.legend(fontsize=8, loc="upper left"))
    u.shrink_ticks(ax_e, size=9)
    u.print_caption_note(
        "figure3 Panel E",
        f"Zero-FWCI documents (n={n_zero_fwci_dropped:,}) drawn as the detached bar left of "
        "the log axis; median includes them",
    )

    # F: Spearman rho by field as a horizontal forest plot -- raw stars vs. raw citations
    # (circles, pooled row on top, dashed line at the pooled value) and modified FWSI vs. raw
    # FWCI (squares), the two markers offset vertically within each field row.
    pooled_row = stars_rho_df.filter(pl.col("field") == "All fields (pooled)")
    field_rows = stars_rho_df.filter(pl.col("field") != "All fields (pooled)").sort(
        "rho", descending=True
    )
    forest_fields = pl.concat([pooled_row, field_rows]).get_column("field").to_list()
    forest_row = {field: i for i, field in enumerate(forest_fields)}
    fwsi_unplotted = set(rho_df.get_column("field").to_list()) - set(forest_fields)
    if fwsi_unplotted:
        print(
            f"  Panel F: FWSI-vs-FWCI fields with no stars row, not plotted: {fwsi_unplotted}"
        )
    forest_colors = u.general_palette(2)
    forest_offset = 0.18
    for rho_frame, label, marker, color, offset in [
        (stars_rho_df, "Stars vs. citations", "o", forest_colors[0], -forest_offset),
        (rho_df, "FWSI vs. FWCI", "s", forest_colors[1], forest_offset),
    ]:
        plotted = rho_frame.filter(pl.col("field").is_in(forest_fields))
        rho = plotted.get_column("rho").to_numpy()
        ax_f.errorbar(
            x=rho,
            y=np.array([forest_row[f] for f in plotted.get_column("field")]) + offset,
            xerr=[
                rho - plotted.get_column("rho_ci_lo").to_numpy(),
                plotted.get_column("rho_ci_hi").to_numpy() - rho,
            ],
            fmt=marker,
            color=color,
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
            markersize=4.5,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=label,
        )
    pooled_rho = float(pooled_row.get_column("rho")[0])
    ax_f.axvline(pooled_rho, color="#888888", linewidth=0.9, linestyle="--", zorder=0)
    ax_f.axvline(0, color="#bbbbbb", linewidth=0.8, linestyle=":", zorder=0)
    ax_f.set_yticks(np.arange(len(forest_fields)))
    ax_f.set_yticklabels([u.abbreviate_field(f) for f in forest_fields])
    ax_f.set_ylim(len(forest_fields) - 0.5, -0.5)
    ax_f.set_xlabel("Spearman rho")
    u.style_legend(ax_f.legend(fontsize=7, title="", loc="lower right"), fontsize=7)
    u.shrink_ticks(ax_f, size=7)
    u.print_caption_note(
        "figure3 Panel F",
        "Circles: raw stargazer vs. raw citation counts; squares: modified FWSI vs. raw OpenAlex "
        "FWCI (repositories >= 2 years old). Bars = 95% CI; dashed line = pooled stars-vs-"
        "citations rho. Every pruned field plus the Other bucket; per-series n in the backing "
        "CSVs. Abbreviated field labels: " + u.field_abbreviation_caption(forest_fields),
    )

    u.save_figure(fig, "figure3_software_development_characteristics", output_dir)
    plt.close(fig)

    # ---- Supplemental candidate: tooling adoption, with-manifest denominator (Figure 3
    # Panel C before the pooled manifest-adoption panel replaced it) ----
    fig_tooling_manifest, ax_tooling_manifest = plt.subplots(figsize=(7, 5))
    _plot_dependency_category_adoption(
        ax_tooling_manifest, category_by_year_plotted, legend_loc="upper right"
    )
    ax_tooling_manifest.set_ylabel("% of Repos with a Parsed Manifest")
    ax_tooling_manifest.set_ylim(bottom=0)
    u.print_caption_note(
        "supplemental_dependency_category_adoption_with_manifest",
        "Share of repositories with any parsed manifest that declare a package from each of "
        "five tooling categories (pypi/cran/conda numerator only), by first-seen publication "
        f"year; years with < {MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR} repos excluded.",
    )
    evaplot.adjust_layout(fig_tooling_manifest)
    u.save_figure(
        fig_tooling_manifest,
        "supplemental_dependency_category_adoption_with_manifest",
        output_dir,
    )
    plt.close(fig_tooling_manifest)

    # ---- Supplemental: tooling adoption, all-repositories denominator ----
    fig_tooling_supp, ax_tooling_supp = plt.subplots(figsize=(7, 5))
    _plot_dependency_category_adoption(ax_tooling_supp, category_by_year_all_repos_plotted)
    ax_tooling_supp.set_ylabel("% of Repos (all linked repositories)")
    u.print_caption_note(
        "supplemental_dependency_category_adoption_all_repos",
        "Same five tooling categories as the with-manifest supplemental, but over every linked "
        "repository (not only those with a parsed manifest) and with no ecosystem restriction "
        "on the numerator.",
    )
    evaplot.adjust_layout(fig_tooling_supp)
    u.save_figure(
        fig_tooling_supp, "supplemental_dependency_category_adoption_all_repos", output_dir
    )
    plt.close(fig_tooling_supp)

    # ---- Supplemental: manifest adoption, Python-primary vs. R-primary repositories (Q10 --
    # moved out of the main text; the R-primary decline is counter-intuitive and needs its own
    # explanation, which belongs in the Supplement, not Results) ----
    manifest_by_language = manifest_adoption.filter(pl.col("series") != "All repositories")
    u.save_table(manifest_by_language, "supplemental_manifest_adoption_by_language", output_dir)
    fig_manifest_supp, ax_manifest_supp = plt.subplots(figsize=(7, 5))
    language_colors = u.general_palette(manifest_by_language.get_column("series").n_unique())
    for series_label, color in zip(
        manifest_by_language.get_column("series").unique(maintain_order=True).to_list(),
        language_colors,
        strict=True,
    ):
        plotted = manifest_by_language.filter(pl.col("series") == series_label)
        ax_manifest_supp.plot(
            plotted.get_column("document_publication_year").to_numpy(),
            plotted.get_column("pct_repos").to_numpy(),
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=color,
            label=series_label,
        )
    ax_manifest_supp.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_manifest_supp.set_xlabel("Publication Year")
    ax_manifest_supp.set_ylabel("% of Repos with a Manifest")
    ax_manifest_supp.set_ylim(0, 100)
    u.style_legend(ax_manifest_supp.legend(fontsize=8, title="", loc="upper left"), fontsize=8)
    u.shrink_ticks(ax_manifest_supp, size=9)
    u.print_caption_note(
        "supplemental_manifest_adoption_by_language",
        "Manifest-adoption share of Python-primary and R-primary repositories by first-seen "
        "publication year; years with "
        f"< {MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR} repos in a series excluded.",
    )
    evaplot.adjust_layout(fig_manifest_supp)
    u.save_figure(fig_manifest_supp, "supplemental_manifest_adoption_by_language", output_dir)
    plt.close(fig_manifest_supp)

    # ---- Supplemental: article vs preprint split ----
    fig_supp, axes_supp = plt.subplots(1, 2, figsize=(9.5, 4))
    u.add_panel_label(axes_supp[0], "A")
    u.add_panel_label(axes_supp[1], "B")

    dev_duration_split = dev_duration.filter(
        pl.col("document_type_bucket").is_in(["article", "preprint"])
    )
    # Pinned hue order so gold/magenta mean the same document type in both panels.
    doctype_hue_order = ["article", "preprint"]
    # Gold/magenta pair, distinct from the binary palettes used in the main Figure 3.
    sns.boxplot(
        data=dev_duration_split.to_pandas(),
        y="period",
        x="duration_years",
        hue="document_type_bucket",
        hue_order=doctype_hue_order,
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
    # Low-alpha histograms for context, with a KDE line per series on top -- stepped
    # histograms alone leave the overlapping peaks hard to distinguish.
    sns.histplot(
        data=fwci_dist_split.to_pandas(),
        x="document_raw_fwci",
        hue="document_type_bucket",
        hue_order=doctype_hue_order,
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
        hue_order=doctype_hue_order,
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
# FWCI-vs-FWSI comparison table


def fwsi_fwci_comparison_table(output_dir: Path = u.OUTPUT_DIR, top_n: int = 5) -> None:
    """
    Build the FWCI-vs-FWSI comparison table: per ecosystem (Python / R primary language),
    the top `top_n` pairs by OpenAlex FWCI and the top `top_n` by modified FWSI, each row
    carrying both metrics -- surfaces articles with high citation impact but low repository
    recognition, and the reverse.
    """
    # Same stable year sort as Figure 3 so FWSI peer groups come from the earliest pair.
    df = u.load_filtered_pairs(top_n_fields=10).sort(
        "document_publication_year", maintain_order=True
    )
    fwci_docs = u.raw_fwci_docs(df)
    fwsi_repos = u.compute_modified_fwsi(df)

    lang_pairs = df.filter(pl.col("repository_primary_language").is_in(["Python", "R"]))
    n_before_source_filter = lang_pairs.unique(subset=["document_id", "repository_id"]).height
    # SoftCite mention-based links over-match famous tools (paper uses the tool, not the
    # tool's own paper), so this table excludes them.
    lang_pairs = lang_pairs.filter(pl.col("dataset_source_name_canonical") != "SoftCite 2025")
    pair_level = lang_pairs.select(
        "document_id",
        "repository_id",
        "document_doi",
        "repository_owner",
        "repository_name",
        pl.col("repository_primary_language").alias("ecosystem"),
    ).unique(subset=["document_id", "repository_id"])
    n_after_source_filter = pair_level.height
    print(
        f"Excluding SoftCite 2025-sourced pairs: {n_after_source_filter:,} of "
        f"{n_before_source_filter:,} Python/R-primary pairs remain "
        f"({n_before_source_filter - n_after_source_filter:,} dropped)"
    )
    # Strictly one-to-one pairs only, applied BEFORE the metric joins: a repo/document with
    # multiple raw pairs is ambiguous even when only one pair has complete metrics.
    n_before_one_to_one = pair_level.height
    pair_level = pair_level.filter(
        pl.col("document_id").is_unique() & pl.col("repository_id").is_unique()
    )
    print(
        f"Restricting to strictly one-to-one document-repository pairs: "
        f"{pair_level.height:,} remain ({n_before_one_to_one - pair_level.height:,} dropped)"
    )
    pair_level = (
        pair_level.join(fwci_docs.select("document_id", "document_raw_fwci"), on="document_id")
        .join(
            fwsi_repos.select("repository_id", "repository_modified_fwsi"), on="repository_id"
        )
        .drop_nulls(["document_raw_fwci", "repository_modified_fwsi"])
    )
    print(
        f"Pairs with both an OpenAlex FWCI and a modified FWSI, Python/R-primary "
        f"repositories: {pair_level.height:,}"
    )

    blocks = []
    for ecosystem in ["Python", "R"]:
        eco_pairs = pair_level.filter(pl.col("ecosystem") == ecosystem)
        for ranked_by, rank_col in [
            ("top_by_fwci", "document_raw_fwci"),
            ("top_by_fwsi", "repository_modified_fwsi"),
        ]:
            blocks.append(
                eco_pairs.sort(rank_col, descending=True)
                .head(top_n)
                .select(
                    pl.lit(ecosystem).alias("ecosystem"),
                    pl.lit(ranked_by).alias("ranked_by"),
                    pl.col("document_doi").alias("doi"),
                    (pl.col("repository_owner") + "/" + pl.col("repository_name")).alias(
                        "repository"
                    ),
                    pl.col("document_raw_fwci").round(2).alias("openalex_fwci"),
                    pl.col("repository_modified_fwsi").round(2).alias("modified_fwsi"),
                )
            )
    table = pl.concat(blocks)
    u.save_table(table, "fwsi_fwci_comparison_table", output_dir)
    print("\nFWCI-vs-FWSI comparison table (top pairs per ecosystem and metric):")
    print(table)
    u.print_caption_note(
        "fwsi_fwci_comparison_table",
        "The google-deepmind/alphafold3 row's DOI (10.1038/s41586-024-08416-7) is the "
        "AlphaFold 3 Nature addendum published alongside the code release, not the main "
        "article DOI, so its FWCI understates the main paper's; ideally both would be "
        "linked. Addendum-vs-main-article ambiguity is a general limitation of "
        "DOI-repository linking",
    )


###############################################################################
# Package-shaped vs. script-shaped repository diagnostic (R + Python)

# Package-shape definitions, saved verbatim into the diagnostic CSV.
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


# Root-level packaging/manifest files checked for presence per language.
MANIFEST_FILES_BY_LANGUAGE: dict[str, list[str]] = {
    "R": ["DESCRIPTION"],
    "Python": ["setup.py", "setup.cfg", "pyproject.toml"],
}
MANIFEST_FILE_ADOPTION_MIN_REPOS_PER_YEAR = 100


def supplemental_manifest_file_adoption(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Compute per-manifest-file adoption over time -- DESCRIPTION (R), setup.py, setup.cfg,
    pyproject.toml (Python) -- each as % of that language's primary-language repositories by
    first-seen publication year. Presence comes from the raw `repository_file` tree listing
    (root-level path, tree_type == "blob"), so a file counts even when the dependency parser
    extracted no dependency rows from it (e.g. a setup.cfg holding only tool config).
    """
    df = u.load_filtered_pairs(top_n_fields=10)

    print("\nLoading repository_file from HuggingFace...")
    repo_files = u.load_table("repository_file")
    print(f"  repository_file: {len(repo_files):,} rows")

    all_manifest_files = sorted(
        {f for files in MANIFEST_FILES_BY_LANGUAGE.values() for f in files}
    )
    # Exact path match = root-level only (no "/" in path); blobs only, not trees.
    manifest_blobs = repo_files.filter(
        pl.col("path").is_in(all_manifest_files) & (pl.col("tree_type") == "blob")
    )
    repos_with_file: dict[str, set[int]] = {
        fname: set(
            manifest_blobs.filter(pl.col("path") == fname)
            .get_column("repository_id")
            .unique()
            .to_list()
        )
        for fname in all_manifest_files
    }
    for fname in all_manifest_files:
        print(f"Repositories with a root-level {fname} blob: {len(repos_with_file[fname]):,}")

    current_year = date.today().year
    repo_level = (
        df.group_by("repository_id")
        .agg(
            pl.min("document_publication_year").alias("first_seen_publication_year"),
            pl.first("repository_primary_language").alias("primary_language"),
        )
        .filter(
            (pl.col("first_seen_publication_year") >= u.DEFAULT_MIN_YEAR)
            & (pl.col("first_seen_publication_year") < current_year)
        )
    )

    frames = []
    for language, manifest_files in MANIFEST_FILES_BY_LANGUAGE.items():
        lang_repos = repo_level.filter(pl.col("primary_language") == language)
        n_lang_repos = lang_repos.height
        total_per_year = lang_repos.group_by("first_seen_publication_year").agg(
            pl.len().alias("n_repos")
        )
        for fname in manifest_files:
            n_with_file_overall = lang_repos.filter(
                pl.col("repository_id").is_in(repos_with_file[fname])
            ).height
            print(
                f"{language}-primary repos with root-level {fname}: "
                f"{n_with_file_overall:,} of {n_lang_repos:,} "
                f"({100 * n_with_file_overall / n_lang_repos:.1f}%)"
            )
            by_year = (
                lang_repos.filter(pl.col("repository_id").is_in(repos_with_file[fname]))
                .group_by("first_seen_publication_year")
                .agg(pl.len().alias("n_with_file"))
                .join(total_per_year, on="first_seen_publication_year", how="right")
                .with_columns(
                    pl.col("n_with_file").fill_null(0),
                    pl.lit(language).alias("primary_language"),
                    pl.lit(fname).alias("manifest_file"),
                )
                .with_columns(
                    (100 * pl.col("n_with_file") / pl.col("n_repos")).alias("pct_repos")
                )
                .sort("first_seen_publication_year")
            )
            frames.append(by_year)
    out = pl.concat(frames).select(
        "primary_language",
        "manifest_file",
        "first_seen_publication_year",
        "n_repos",
        "n_with_file",
        "pct_repos",
    )
    u.save_table(out, "supplemental_manifest_file_adoption_by_year", output_dir)

    # ---- Figure: one line per manifest file ----
    evaplot.set_style("evaplot_rc")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    line_specs = [
        ("R", "DESCRIPTION", "-", "o"),
        ("Python", "setup.py", "-", "s"),
        ("Python", "setup.cfg", "--", "^"),
        ("Python", "pyproject.toml", "-.", "D"),
    ]
    line_colors = u.general_palette(len(line_specs))
    for (language, fname, ls, marker), color in zip(line_specs, line_colors, strict=True):
        plotted = out.filter(
            (pl.col("primary_language") == language)
            & (pl.col("manifest_file") == fname)
            & (pl.col("n_repos") >= MANIFEST_FILE_ADOPTION_MIN_REPOS_PER_YEAR)
        )
        ax.plot(
            plotted.get_column("first_seen_publication_year").to_numpy(),
            plotted.get_column("pct_repos").to_numpy(),
            marker=marker,
            markersize=4,
            linewidth=1.6,
            linestyle=ls,
            color=color,
            label=f"{fname} ({language})",
        )
    ax.set_xlabel("First-Seen Publication Year")
    ax.set_ylabel("% of Language's Repos with File")
    ax.set_ylim(0, 100)
    u.style_legend(ax.legend(fontsize=8, title="", loc="upper left"), fontsize=8)
    u.print_caption_note(
        "supplemental_manifest_file_adoption",
        f"Presence from the raw repository file tree (root-level blob); years with "
        f"< {MANIFEST_FILE_ADOPTION_MIN_REPOS_PER_YEAR} repos excluded",
    )
    u.shrink_ticks(ax, size=9)
    evaplot.adjust_layout(fig)
    u.save_figure(fig, "supplemental_manifest_file_adoption", output_dir)
    plt.close(fig)


def supplemental_package_vs_script_diagnostics(output_dir: Path = u.OUTPUT_DIR) -> None:
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
# Optional read-only check: does seed-source composition explain the with-manifest test/docs
# tooling decline, or a package/script-shape effect?

TOOLING_COMPOSITION_MIN_REPOS_PER_CELL = 20
SOFTWARE_PAPER_SEED_SOURCES: tuple[str, ...] = ("JOSS", "SoftwareX")


def tooling_composition_by_seed_source_diagnostic(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Among with-manifest repositories, check whether the test/docs tooling decline (the
    with-manifest tooling supplemental) is explained by seed-source composition shifting away from software papers (JOSS,
    SoftwareX), which are more likely to declare test/docs tooling than the other sources
    (PLOS, Papers with Code, SoftCite, mined pairs). Read-only analysis, not a manuscript
    figure: (1) the software-paper share of with-manifest repositories by year; (2) test and
    documentation tooling shares split by seed group and year, among with-manifest
    repositories.
    """
    df = u.load_filtered_pairs(top_n_fields=10).sort(
        "document_publication_year", maintain_order=True
    )
    deps = u.clean_dependency_names(u.load_table("repository_dependency"))
    repos_with_any_manifest = deps.select("repository_id").unique()

    repo_base = (
        df.select("repository_id", "document_publication_year", "dataset_source_name_canonical")
        .unique(subset="repository_id", keep="first")
        .join(repos_with_any_manifest, on="repository_id", how="semi")
        .with_columns(
            pl.when(pl.col("dataset_source_name_canonical").is_in(SOFTWARE_PAPER_SEED_SOURCES))
            .then(pl.lit("Software paper (JOSS/SoftwareX)"))
            .otherwise(pl.lit("Other seed source"))
            .alias("seed_group")
        )
    )

    # ---- (1) software-paper share of with-manifest repos, by year ----
    software_paper_share = (
        repo_base.group_by("document_publication_year")
        .agg(
            pl.len().alias("total"),
            (pl.col("seed_group") == "Software paper (JOSS/SoftwareX)")
            .sum()
            .alias("n_software_paper"),
        )
        .with_columns(
            (100 * pl.col("n_software_paper") / pl.col("total")).alias("pct_software_paper")
        )
        .filter(pl.col("total") >= TOOLING_COMPOSITION_MIN_REPOS_PER_CELL)
        .sort("document_publication_year")
    )
    u.save_table(
        software_paper_share,
        "supplemental_software_paper_share_of_manifest_repos_by_year",
        output_dir,
    )
    print("\nSoftware-paper share of with-manifest repositories, by year:")
    print(software_paper_share)

    # ---- (2) test/docs tooling shares by seed group and year ----
    frames = []
    for category in ("Testing", "Documentation"):
        norm_pkgs = [normalize_name(p) for p in _DEPENDENCY_CATEGORIES[category]]
        cat_repos = (
            deps.filter(pl.col("software_name_normalized").is_in(norm_pkgs))
            .select("repository_id")
            .unique()
        )
        key_cols = ["seed_group", "document_publication_year"]
        total_per_cell = repo_base.group_by(key_cols).agg(pl.len().alias("total"))
        frame = (
            repo_base.join(cat_repos, on="repository_id", how="semi")
            .group_by(key_cols)
            .agg(pl.len().alias("count"))
            .join(total_per_cell, on=key_cols, how="right")
            .with_columns(pl.col("count").fill_null(0), pl.lit(category).alias("category"))
            .with_columns((100 * pl.col("count") / pl.col("total")).alias("pct_repos"))
            .filter(pl.col("total") >= TOOLING_COMPOSITION_MIN_REPOS_PER_CELL)
        )
        frames.append(frame)
    out = pl.concat(frames).sort(["category", "seed_group", "document_publication_year"])
    u.save_table(out, "supplemental_tooling_by_seed_group_and_year", output_dir)
    print("\nTest/documentation tooling share among with-manifest repos, by seed group + year:")
    print(out)


###############################################################################
# Seed vs. mined characteristics, licence categories, FWCI coverage profile


def _license_category_expr() -> pl.Expr:
    """Per-repository licence category with the Panel D keyword lists: copyleft first,
    then permissive, then any other recorded licence, else none.
    """
    lic = pl.col("repository_license")
    copyleft = pl.any_horizontal(
        [lic.str.contains(k, literal=True) for k in _COPYLEFT_LICENSE_KEYWORDS]
    )
    permissive = pl.any_horizontal(
        [lic.str.contains(k, literal=True) for k in _PERMISSIVE_LICENSE_KEYWORDS]
    )
    return (
        pl.when(lic.is_null())
        .then(pl.lit("none"))
        .when(copyleft)
        .then(pl.lit("copyleft"))
        .when(permissive)
        .then(pl.lit("permissive"))
        .otherwise(pl.lit("other_or_unclassified"))
        .alias("license_category")
    )


def _characteristics_row(
    group: str, pairs: pl.DataFrame, repos_with_manifest: pl.DataFrame
) -> dict[str, float | int | str]:
    """One row of seed-vs-mined characteristics, each metric on the same population as its
    main-text counterpart (FWCI over all documents; repository metrics over repositories
    first seen 2008 to last year, attributed to their earliest publication year).
    """
    current_year = date.today().year
    docs = pairs.unique(subset="document_id", keep="first")
    fwci = docs.select("document_fwci").drop_nulls()
    repos = pairs.unique(subset="repository_id", keep="first").filter(
        pl.col("document_publication_year") < current_year
    )
    repos = repos.with_columns(
        _license_category_expr(),
        pl.col("repository_id")
        .is_in(repos_with_manifest.get_column("repository_id").implode())
        .alias("has_manifest"),
    )
    known_language = repos.drop_nulls("repository_primary_language")
    last_year = repos.filter(pl.col("document_publication_year") == current_year - 1)
    timing = _dev_duration_frame(pairs)

    def pct(frame: pl.DataFrame, expr: pl.Expr) -> float:
        return 100 * frame.select(expr.mean()).item() if frame.height else float("nan")

    def median_days(period: str) -> float:
        vals = timing.filter(pl.col("period") == period)
        return vals.select(pl.col("duration_years").median() * 365.25).item()

    return {
        "group": group,
        "n_pairs": pairs.height,
        "n_documents": docs.height,
        "pct_documents_with_fwci": 100 * fwci.height / docs.height,
        "median_fwci_incl_zero": fwci.select(pl.col("document_fwci").median()).item(),
        "median_fwci_excl_zero": fwci.filter(pl.col("document_fwci") > 0)
        .select(pl.col("document_fwci").median())
        .item(),
        "pct_fwci_above_1": pct(fwci, pl.col("document_fwci") > 1),
        "n_repositories_2008_to_last_year": repos.height,
        "pct_permissive": pct(repos, pl.col("license_category") == "permissive"),
        "pct_copyleft": pct(repos, pl.col("license_category") == "copyleft"),
        "pct_other_or_unclassified": pct(
            repos, pl.col("license_category") == "other_or_unclassified"
        ),
        "pct_no_license": pct(repos, pl.col("license_category") == "none"),
        "pct_with_manifest_2008_to_last_year": pct(repos, pl.col("has_manifest")),
        f"pct_with_manifest_{current_year - 1}": pct(last_year, pl.col("has_manifest")),
        "n_repositories_known_language": known_language.height,
        "pct_python_of_known_language": pct(
            known_language, pl.col("repository_primary_language") == "Python"
        ),
        "median_days_created_before_publication": median_days("Before Publication"),
        "median_days_last_push_after_publication": median_days("After Publication"),
    }


def seed_vs_mined_characteristics(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Supplementary table: FWCI, licence, manifest share, Python share and publication timing
    split by how pairs entered RS-Graph (seed vs. mined, with Papers with Code separated from
    the other seeds). The "All pairs" row reproduces the pooled main-text figures. Also
    writes every recorded licence string with its category and repository count, so the
    permissive/copyleft keyword lists can be checked against the data.
    """
    df = u.load_filtered_pairs(top_n_fields=10).sort(
        "document_publication_year", maintain_order=True
    )
    deps = u.clean_dependency_names(u.load_table("repository_dependency"))
    repos_with_manifest = (
        deps.filter(pl.col("ecosystem").is_in(u.ALL_MANIFEST_ECOSYSTEMS))
        .select("repository_id")
        .unique()
    )

    is_mined = pl.col("link_processing_iteration").is_not_null()
    is_pwc = pl.col("dataset_source_name") == "pwc"
    groups = [
        ("All pairs", df),
        ("Seed (all sources)", df.filter(~is_mined)),
        ("Seed: Papers with Code", df.filter(~is_mined & is_pwc)),
        ("Seed: other four sources", df.filter(~is_mined & ~is_pwc)),
        ("Mined (rounds 1-5)", df.filter(is_mined)),
        (
            "Mined, excluding Computer Science",
            df.filter(is_mined).filter(pl.col("document_field_name") != "Computer Science"),
        ),
        (
            "Seed, excluding Computer Science",
            df.filter(~is_mined).filter(pl.col("document_field_name") != "Computer Science"),
        ),
    ]
    table = pl.DataFrame(
        [_characteristics_row(g, frame, repos_with_manifest) for g, frame in groups]
    )
    n_repos_both = (
        df.group_by("repository_id")
        .agg(is_mined.any().alias("m"), (~is_mined).any().alias("s"))
        .filter(pl.col("m") & pl.col("s"))
        .height
    )
    print(f"Repositories linked through both a seed pair and a mined pair: {n_repos_both:,}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(table)
    u.save_table(table, "seed_vs_mined_characteristics", output_dir)

    current_year = date.today().year
    license_strings = (
        df.unique(subset="repository_id", keep="first")
        .filter(pl.col("document_publication_year") < current_year)
        .with_columns(_license_category_expr())
        .group_by("license_category", "repository_license")
        .agg(pl.len().alias("n_repositories"))
        .sort("license_category", "n_repositories", descending=[False, True])
    )
    with pl.Config(tbl_rows=60, fmt_str_lengths=60):
        print(license_strings)
    u.save_table(license_strings, "license_category_strings", output_dir)


def fwci_coverage_profile(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    How documents with an OpenAlex FWCI differ from those without one: document type,
    publication year, route into RS-Graph, field and open-access status, one row per group.
    """
    df = u.load_filtered_pairs(top_n_fields=10)
    docs = df.group_by("document_id").agg(
        pl.col("document_fwci").first().is_not_null().alias("has_fwci"),
        pl.col("document_type_bucket").first(),
        pl.col("document_publication_year").first(),
        pl.col("document_field_name").first(),
        pl.col("document_is_open_access").first(),
        pl.col("document_cited_by_count").first(),
        pl.col("link_processing_iteration").is_not_null().all().alias("only_mined"),
        (pl.col("dataset_source_name") == "pwc").any().alias("any_pwc"),
    )
    current_year = date.today().year
    profile = (
        docs.group_by("has_fwci")
        .agg(
            pl.len().alias("n_documents"),
            (100 * (pl.col("document_type_bucket") == "article").mean()).alias("pct_article"),
            (100 * (pl.col("document_type_bucket") == "preprint").mean()).alias("pct_preprint"),
            (100 * (pl.col("document_type_bucket") == "other").mean()).alias("pct_other_type"),
            pl.col("document_publication_year").median().alias("median_publication_year"),
            (100 * (pl.col("document_publication_year") >= current_year - 1).mean()).alias(
                f"pct_published_{current_year - 1}_or_later"
            ),
            (100 * pl.col("any_pwc").mean()).alias("pct_with_pwc_pair"),
            (100 * pl.col("only_mined").mean()).alias("pct_only_mined_pairs"),
            (100 * (pl.col("document_field_name") == "Computer Science").mean()).alias(
                "pct_computer_science"
            ),
            (100 * pl.col("document_is_open_access").cast(pl.Float64).mean()).alias(
                "pct_open_access"
            ),
            pl.col("document_cited_by_count").median().alias("median_cited_by_count"),
        )
        .sort("has_fwci", descending=True)
    )
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(profile)
    by_type = (
        docs.group_by("document_type_bucket")
        .agg(
            pl.len().alias("n_documents"),
            (100 * pl.col("has_fwci").mean()).alias("pct_with_fwci"),
        )
        .sort("n_documents", descending=True)
    )
    print(by_type)
    u.save_table(profile, "fwci_coverage_profile", output_dir)
    u.save_table(by_type, "fwci_coverage_by_document_type", output_dir)
