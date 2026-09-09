#!/usr/bin/env python3

"""Software-mention analysis: Figure 4's mention rate by field and year, the
mention-predictors logistic regression, and the mentions-extraction coverage diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import evaplot
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
import utils as u
from scipy.stats import norm, pearsonr
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from rs_graph.utils.software_alignment import align_software_names

###############################################################################
# Figure 4 -- software mention rate by field, over time

# A lower floor lets single-digit-n field-year cells through, producing single-year
# percentage spikes that are statistical noise; 30 pushes every plotted field's effective
# start year to where cell sizes make the rate stable, without a hard-coded cutoff year.
MIN_PAIRS_PER_CELL = 30


def figure_4_mention_rate_by_field_and_year(
    output_dir: Path = u.OUTPUT_DIR,
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
    mentions = u.clean_mention_names(mentions)

    # Only the import name is ever canonical -- align per pair with imports as items_a.
    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    eligible = df.filter(pl.col("repository_id").is_in(repo_with_import))
    print(
        f"After restricting to pairs whose repository has >=1 import: "
        f"{eligible.height:,} of {df.height:,} pairs remain"
    )

    imports_by_repo = u.normalized_names_by_id(imports, "repository_id")
    mentions_by_doc = u.normalized_names_by_id(mentions, "document_id")

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

    # Mention extraction lags the most recent publication years: zero or sharply depressed
    # rates despite large import volume signal partially-populated extraction, not a real
    # behavioral shift. Walk backward from the most recent year, dropping any year whose
    # overall rate falls below 40% of the next-older year's rate, until the series stabilizes.
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
    # Plotted fields are decided by this figure's own eligible-pair counts, but the order
    # (legend, line style/marker assignment) follows Figure 2 Panel A's global prevalence
    # order for continuity across the figure set.
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

    # Summary stats respect the same stale-year cap as the plotted figure -- post-cap years
    # have structurally-zero mentions and would depress every rate below.
    pair_rates_capped = pair_rates.filter(pl.col("document_publication_year") <= max_plot_year)
    overall_rate = (
        100
        * pair_rates_capped.get_column("n_matched_imports").sum()
        / pair_rates_capped.get_column("n_total_imports").sum()
    )
    print(
        f"\nOverall software mention rate across eligible pairs "
        f"(<= {max_plot_year}): {overall_rate:.1f}%"
    )

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

    # Headline summary stats. Every rate is reported twice: over ALL pairs with >=1 import
    # (the figure's denominator), and CONDITIONAL on the document having >=1 extracted
    # mention -- the two denominators differ by ~4x.
    with_mention = pair_rates_capped.filter(pl.col("has_any_mention"))

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
                _rate(pair_rates_capped, "Python"),
                _rate(pair_rates_capped, "R"),
                _pct_zero(pair_rates_capped),
                float(pair_rates_capped.height),
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

    # Per-field conditional rates (same stale-year cap as the plotted figure).
    field_overall_conditional = (
        with_mention.group_by("document_field_name_pruned")
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

    fig, ax = plt.subplots(figsize=(9, 5))
    # Shared field-to-color assignment used by every per-field figure; `style=` additionally
    # gives every field a distinct marker + dash pattern.
    field_color_lookup = u.field_color_map(canonical_field_order)
    field_colors = [field_color_lookup[f] for f in top_fields]
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
    # Legend outside the axes: no pocket inside stays clear of data at every plotted year.
    leg = ax.legend(
        title="",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        handlelength=1.5,
        labelspacing=0.3,
        borderaxespad=0.6,
    )
    u.style_legend(leg, fontsize=8)
    if stale_years:
        u.print_caption_note(
            "figure4_mention_rate_by_field_and_year",
            f"Years after {max_plot_year} excluded: mention extraction hasn't caught up to "
            "these publication years yet (see Methods)",
        )
    u.print_caption_note(
        "figure4_mention_rate_by_field_and_year",
        "Year-cap rationale: SoftCite-2025 mention-extraction coverage is stable through "
        "April 2023, degrades over May-June 2023, and is exactly zero from July 2023 onward, "
        "so mention-based analyses are capped at 2022 -- the last fully covered publication "
        "year -- giving mentions the fairest representation by using only reliably extracted "
        "years, just as imports and dependencies each use their own full reliable range",
    )

    evaplot.adjust_layout(fig)
    u.save_figure(fig, "figure4_mention_rate_by_field_and_year", output_dir)
    plt.close(fig)


###############################################################################
# Predictors of software mentioning (logistic regression)

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
TOP_N_FIELDS_FOR_CONTROL = 5


def _trace_filter_step(
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
    _trace_filter_step(trace, "after_generic_name_exclusion", df)

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
    _trace_filter_step(
        trace,
        "after_rare_software_exclusion",
        df,
        note=(
            f"floor of {min_count} mirrors the PNAS reference design; "
            f"removed {n_before_rare - df.height:,} rows"
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
    _trace_filter_step(
        trace,
        "after_extreme_usage_pair_trim",
        df,
        note=(
            f">p{upper_q * 100:.0f} trim mirrors the PNAS reference design; "
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


def _prepare_regression_features(
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
    # Computed on unique documents (doc_meta), not the long frame -- the long frame has many
    # rows per document, which would over-weight prolific-import documents in the ranking.
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
        _trace_filter_step(
            trace,
            "after_doc_metadata_join",
            long_df,
            note=f"{n_any_null:,} rows carry >=1 null across the full model-column set",
        )
    return long_df


def predictors_of_software_mentioning(
    output_dir: Path = u.OUTPUT_DIR,
    cutoff: float = 85.0,
    top_n_fields_for_control: int = TOP_N_FIELDS_FOR_CONTROL,
    year_cap: int = u.MENTION_EXTRACTION_YEAR_CAP,
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
      - year_cap_applied: with and without the mentions-extraction year cap (default 2022) --
        rows after the cap are structurally is_mentioned=False because mention extraction
        hasn't covered those publication years.

    Nulls are dropped per-spec on only the columns each spec actually uses, so simpler specs
    keep documents with e.g. no OpenAlex topic. Also writes a filter-chain trace
    (mention_predictors_filter_chain_row_counts.csv) recording data loss at every step.
    """
    df = u.load_filtered_pairs(top_n_fields=10)

    print("\nLoading repository_import and document_software_mention from HuggingFace...")
    imports = u.load_table("repository_import")
    mentions = u.load_table("document_software_mention")
    print(f"  repository_import: {len(imports):,} rows")
    print(f"  document_software_mention: {len(mentions):,} rows")
    mentions = u.clean_mention_names(mentions)

    docs_with_mention = set(mentions.get_column("document_id").unique().to_list())

    trace: list[dict] = []
    _trace_filter_step(trace, "standard_filtered_pairs", df)
    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    _trace_filter_step(
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
        _trace_filter_step(variant_trace, "long_frame", long_df)
        long_df = _remove_rare_and_generic_software(long_df, trace=variant_trace)
        long_df = _trim_extreme_usage_pairs(long_df, trace=variant_trace)
        long_df = _prepare_regression_features(
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
                    f"\n=== Mention-predictors grid cell: alignment={alignment}, "
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
                        _trace_filter_step(
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
    u.save_table(summary_df, "mention_predictors_logit_summary", output_dir)

    trace_df = pl.DataFrame(trace)
    u.save_table(trace_df, "mention_predictors_filter_chain_row_counts", output_dir)
    print("\n--- Mention-predictors filter-chain trace ---")
    print(trace_df)
    print(
        "\nSensitivity notes: rare-software floor of 3 and p99 usage trim (rows removed "
        "reported above); year-cap variants comparable in the summary CSV.\n"
    )


###############################################################################
# Mentions-extraction coverage by publication year (diagnostic)


def mentions_coverage_by_year(output_dir: Path = u.OUTPUT_DIR) -> None:
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

    u.save_table(by_year, "mentions_coverage_by_year", output_dir)
    print("\n--- Mentions-extraction coverage by publication year (unfiltered) ---")
    print(by_year.select(pl.exclude("verdict")))
    print(f"\nVERDICT: {verdict}")
    print("----------------------------------------------------------------------\n")
