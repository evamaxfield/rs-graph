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
    # Pooled series (any manifest, all repos) plus Python/R ecosystem reference series.
    series_specs: list[tuple[str, pl.DataFrame, pl.DataFrame]] = [
        ("All repositories", year_repo, deps.select("repository_id").unique())
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
        # software_name_normalized is hyphen/underscore-stripped, so match on normalized names.
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


def _fwci_distribution_frame(fwci_docs: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Panel D's positive-FWCI frame plus the zero-FWCI count (drawn as its own bar)."""
    n_before_zero_filter = fwci_docs.filter(pl.col("document_raw_fwci").is_not_null()).height
    d = fwci_docs.filter(
        pl.col("document_raw_fwci").is_not_null() & (pl.col("document_raw_fwci") > 0)
    )
    n_zero_citation = n_before_zero_filter - d.height
    print(
        f"Figure 3 Panel D: {n_zero_citation:,} zero-FWCI documents (OpenAlex FWCI = 0.0) "
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


###############################################################################


def figure_3_software_development_characteristics(
    output_dir: Path = u.OUTPUT_DIR,
) -> None:
    """
    Build Figure 3: software development characteristics, one consolidated 6-panel figure.
    (A) dev activity duration, (B) Python's share of repositories over time by field, (C)
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
    # Manifest adoption is CSV-only now -- cited as prose, not plotted.
    manifest_adoption = _manifest_adoption_over_time(df, deps)
    python_share = _python_share_by_field_over_time(df)
    category_by_year = _dependency_category_adoption(df, deps, group_col=None, with_year=True)
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

    # FWSI-vs-FWCI rho is saved as CSV only; Panel E plots raw stars vs. raw citations.
    u.save_table(
        rho_df.with_columns(format_p), "figure3_fwsi_fwci_spearman_by_field", output_dir
    )
    print("\nFWSI-vs-raw-FWCI Spearman rho by field (CSV only, not plotted):")
    print(rho_df)

    stars_rho_df = _stars_vs_citations_by_field(df)
    u.save_table(
        stars_rho_df.with_columns(format_p),
        "figure3_stars_citations_spearman_by_field",
        output_dir,
    )
    print("\nRaw-stars-vs-raw-citations Spearman rho by field (Panel E):")
    print(stars_rho_df)

    license_adoption = _license_adoption_over_time(df)
    u.save_table(license_adoption, "figure3_license_adoption_by_year", output_dir)
    print("\nLicense adoption by year (Panel F):")
    print(license_adoption)

    # Panel B/C data as labeled tables -- the manuscript cites exact adoption percentages.
    u.save_table(manifest_adoption, "figure3_manifest_adoption_by_year", output_dir)
    u.save_table(python_share, "figure3_python_share_by_field_over_year", output_dir)
    current_year = date.today().year
    category_by_year_plotted = category_by_year.filter(
        pl.col("document_publication_year") < current_year
    )
    u.save_table(
        category_by_year_plotted, "figure3_dependency_category_adoption_by_year", output_dir
    )
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

    # FWCI summary stats -- the paper's headline median and per-field medians.
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

    # B: Python's share of repositories over time, one line per field plus a pooled line.
    python_share_fields = [
        s
        for s in python_share.get_column("series").unique(maintain_order=True).to_list()
        if s != PYTHON_SHARE_OVERALL_LABEL
    ]
    field_colors = u.field_color_map(python_share_fields)
    for series_label in [*python_share_fields, PYTHON_SHARE_OVERALL_LABEL]:
        plotted = python_share.filter(pl.col("series") == series_label)
        is_overall = series_label == PYTHON_SHARE_OVERALL_LABEL
        ax_b.plot(
            plotted.get_column("document_publication_year").to_numpy(),
            plotted.get_column("pct_python").to_numpy(),
            marker="o",
            markersize=3,
            linewidth=2.2 if is_overall else 1.5,
            linestyle="--" if is_overall else "-",
            color="black" if is_overall else field_colors[series_label],
            label="Overall" if is_overall else u.abbreviate_field(series_label),
            zorder=3 if is_overall else 2,
        )
    ax_b.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_b.set_xlabel("Publication Year")
    ax_b.set_ylabel("% of Repos Python-Primary")
    ax_b.set_ylim(0, 100)
    u.style_legend(
        ax_b.legend(fontsize=6, title="", loc="upper left", ncol=2, columnspacing=0.8),
        fontsize=6,
    )
    u.print_caption_note(
        "figure3 Panel B",
        f"Share of repositories with a known primary language whose primary language is "
        f"Python, by first-seen publication year; top {PYTHON_SHARE_TOP_N_FIELDS} fields + "
        f"Other + a pooled all-fields line; years with "
        f"< {MANIFEST_ADOPTION_MIN_REPOS_PER_YEAR} repos in a series excluded. Abbreviated "
        "field labels: " + u.field_abbreviation_caption(python_share_fields),
    )
    u.shrink_ticks(ax_b, size=9)

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
        palette=u.general_palette(len(_DEPENDENCY_CATEGORIES)),
        markers=True,
        dashes=True,
        ax=ax_c,
    )
    ax_c.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_c.set_xlabel("Publication Year")
    ax_c.set_ylabel("% of Repos")
    # Headroom above the data so the legend doesn't sit on the early Testing line.
    ax_c.set_ylim(0, 12)
    u.style_legend(ax_c.legend(fontsize=8, title="", loc="upper left"), fontsize=8)
    u.shrink_ticks(ax_c, size=9)

    # D: raw OpenAlex FWCI distribution. bins=25: seaborn's log-scale "auto" rule produced
    # jagged bin-to-bin noise unrelated to real signal.
    sns.histplot(
        data=fwci_dist.to_pandas(), x="document_raw_fwci", ax=ax_d, log_scale=True, bins=25
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
    ax_d.bar(
        zero_x,
        n_zero_fwci_dropped,
        width=zero_x * (bin_ratio - 1),
        color=u.general_palette(2)[1],
        label=f"FWCI = 0 (n={n_zero_fwci_dropped:,})",
    )
    ax_d.set_xlim(left=zero_x / bin_ratio)
    # Median over every document with an FWCI, zeros included, matching the plotted data.
    ax_d.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Field avg.")
    ax_d.axvline(
        fwci_median_incl_zero,
        color="red",
        linestyle="-.",
        linewidth=1.5,
        label=f"Median: {fwci_median_incl_zero:.2f}",
    )
    ax_d.set_xlabel("OpenAlex FWCI (log scale)")
    ax_d.set_ylabel("Count")
    # Upper left keeps the legend clear of Panel E's long y-tick labels.
    u.style_legend(ax_d.legend(fontsize=8, loc="upper left"))
    u.shrink_ticks(ax_d, size=9)
    u.print_caption_note(
        "figure3 Panel D",
        f"Zero-FWCI documents (n={n_zero_fwci_dropped:,}) drawn as the detached bar left of "
        "the log axis; median includes them",
    )

    # E: raw-stars-vs-raw-citations Spearman rho by field -- horizontal forest plot with the
    # pooled rho as the top row and a reference line at its value.
    pooled_row = stars_rho_df.filter(pl.col("field") == "All fields (pooled)")
    field_rows = stars_rho_df.filter(pl.col("field") != "All fields (pooled)").sort(
        "rho", descending=True
    )
    forest = pl.concat([pooled_row, field_rows]).to_pandas()
    point_color = u.general_palette(1)[0]
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
        [
            f"{u.abbreviate_field(f)} (n={n:,})"
            for f, n in zip(forest["field"], forest["n"], strict=True)
        ]
    )
    ax_e.invert_yaxis()
    ax_e.set_ylim(len(forest) - 0.5, -0.5)
    ax_e.set_xlabel("Spearman rho (stars vs. citations)")
    # Smaller than the other panels: the field+n y-tick labels are the longest text in the grid.
    u.shrink_ticks(ax_e, size=7)
    u.print_caption_note(
        "figure3 Panel E",
        "Raw stargazer and citation counts; dashed line = pooled rho. Every pruned field "
        "plus the Other bucket. Abbreviated field labels: "
        + u.field_abbreviation_caption(forest["field"].tolist()),
    )

    # F: license adoption over time -- any / permissive / copyleft license share of repos by
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
    ax_f.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_f.set_xlabel("Publication Year")
    ax_f.set_ylabel("% of Repos")
    ax_f.set_ylim(0, 100)
    u.style_legend(ax_f.legend(fontsize=8, title="", loc="upper right"), fontsize=8)
    u.print_caption_note(
        "figure3 Panel F",
        f"Years with < {LICENSE_ADOPTION_MIN_REPOS_PER_YEAR} repos excluded; unclassifiable "
        "licenses (e.g. GitHub's 'Other') count toward 'Any license' only",
    )
    u.shrink_ticks(ax_f, size=9)

    u.save_figure(fig, "figure3_software_development_characteristics", output_dir)
    plt.close(fig)

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
    df = u.load_filtered_pairs(top_n_fields=10)
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
