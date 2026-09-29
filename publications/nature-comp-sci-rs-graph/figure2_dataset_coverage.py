#!/usr/bin/env python3

"""Figure 2: dataset coverage by field and publication year, plus the PwC coverage and
summary statistics.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import evaplot
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import utils as u
from datasets import DatasetDict, load_dataset

###############################################################################


def _pwc_coverage_statistic(df: pl.DataFrame) -> dict[str, float]:
    """Compute rs-graph's coverage of PwC's official paper-code links and the "% increase
    over the verified, accessible PwC subset" statistic.
    """
    print("\nLoading pwc-archive/links-between-paper-and-code from HuggingFace...")
    pwc_ds = load_dataset("pwc-archive/links-between-paper-and-code")
    assert isinstance(pwc_ds, DatasetDict)
    pwc_df = pwc_ds["train"].to_polars()
    assert isinstance(pwc_df, pl.DataFrame)
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
        f"{len(df):,} filtered pairs is a {increase_pct:.0f}% increase over the "
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


def figure_2_dataset_coverage(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Build Figure 2: dataset coverage. (A) pair counts by field -- rs-graph, stacked seed vs.
    mined, beside the verified PwC subset; (B) pair counts per publication year stacked by
    top-6 fields + Other, with the mined share of each year's pairs overlaid on a secondary
    axis. Also saves the field-proportion table and a seed-source x field supplement table.
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

    # ---- Seed-source x field supplement table ----
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={"wspace": 0.45})
    u.add_panel_label(axes[0], "A")
    u.add_panel_label(axes[1], "B")

    palette2 = u.general_palette(2)
    family_green, family_orange = palette2[0], palette2[1]
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
    axes[0].set_yticklabels([u.abbreviate_field(f) for f in field_order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Article-Repository Pairs")

    # ncol=2 keeps the two rs-graph entries adjacent on the top row; anchored right of the
    # panel label so the two don't overlap.
    leg_a = axes[0].legend(
        fontsize=8, title="", loc="lower left", bbox_to_anchor=(0.05, 1.02), ncol=2
    )
    u.style_legend(leg_a)
    u.shrink_ticks(axes[0], size=9)

    # Panel B: stacked yearly counts by field, with each year's mined share overlaid as a
    # line on a secondary axis.
    b_fields = [*top6_fields, "Other"]
    # Shared field-to-color assignment used by every per-field figure.
    field_colors = u.field_color_map(b_fields)
    b_colors = [field_colors[f] for f in b_fields]
    years = sorted(year_field.get_column("document_publication_year").unique().to_list())
    bottoms = np.zeros(len(years))
    year_index = {y: i for i, y in enumerate(years)}
    for fname, color in zip(b_fields, b_colors, strict=True):
        counts = np.zeros(len(years))
        for row in year_field.filter(pl.col("field7") == fname).iter_rows(named=True):
            counts[year_index[row["document_publication_year"]]] = row["count"]
        axes[1].bar(
            years,
            counts,
            bottom=bottoms,
            color=color,
            width=0.8,
            label=u.abbreviate_field(fname),
        )
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
    # Legend outside the axes; every in-panel position collides with data in some year.
    leg_b = axes[1].legend(
        handles_b + handles_b2,
        labels_b + labels_b2,
        fontsize=8,
        title="",
        loc="center left",
        bbox_to_anchor=(1.16, 0.5),
        borderaxespad=0.0,
    )
    u.style_legend(leg_b, fontsize=8)
    evaplot.rotate_xticklabels(axes[1], rotation=40)
    u.shrink_ticks(axes[1], size=9)
    u.shrink_ticks(ax_b2, size=9)
    if partial_year_note:
        u.print_caption_note("figure2_dataset_coverage Panel B", partial_year_note)

    panel_a_abbrevs = u.field_abbreviation_caption(field_order)
    if panel_a_abbrevs:
        u.print_caption_note(
            "figure2_dataset_coverage Panel A",
            "Abbreviated field labels: " + panel_a_abbrevs,
        )
    panel_b_abbrevs = u.field_abbreviation_caption(b_fields)
    if panel_b_abbrevs:
        u.print_caption_note(
            "figure2_dataset_coverage Panel B",
            "Abbreviated field labels: " + panel_b_abbrevs,
        )

    evaplot.adjust_layout(fig, bottom=0.22)
    u.save_figure(fig, "figure2_dataset_coverage", output_dir)
    plt.close(fig)

    # ---- Standalone PwC coverage statistic + summary stats ----
    pwc_stats = _pwc_coverage_statistic(df)
    # Field counts for the "spans more than 20 fields" claim -- unpruned field taxonomy.
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
# Candidate Figure 2C -- mined share of pairs by field over time (decision aid, not wired
# into the main Figure 2 composite; go/no-go pending)

CANDIDATE_2C_MIN_PAIRS_PER_CELL = 100


def candidate_figure2c_mined_share_by_field(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Render two candidate versions of a possible Figure 2 Panel C -- mined share of pairs by
    field, over time -- as standalone PNGs for a go/no-go decision. Not called from
    `figure_2_dataset_coverage`. (a) top-6 fields + Other, 7 lines. (b) Computer Science vs.
    every other field pooled, 2 lines. Both apply the same
    `CANDIDATE_2C_MIN_PAIRS_PER_CELL` per-cell floor Figure 3 uses for its per-year series.
    """
    evaplot.set_style("evaplot_rc")
    df = u.load_filtered_pairs(top_n_fields=10)
    df = df.with_columns(pl.col("link_processing_iteration").is_not_null().alias("is_mined"))

    current_year = date.today().year
    year_df = df.filter(pl.col("document_publication_year") != current_year)

    field_order = (
        df.get_column("document_field_name_pruned")
        .value_counts(sort=True)
        .get_column("document_field_name_pruned")
        .to_list()
    )
    top6_fields = [f for f in field_order if f != "Other"][:6]

    # ---- (a) 7 lines: top-6 fields + Other ----
    b_fields = [*top6_fields, "Other"]
    field_mined_share = (
        year_df.with_columns(
            pl.when(pl.col("document_field_name_pruned").is_in(top6_fields))
            .then(pl.col("document_field_name_pruned"))
            .otherwise(pl.lit("Other"))
            .alias("field7")
        )
        .group_by(["document_publication_year", "field7"])
        .agg(pl.len().alias("total"), pl.col("is_mined").sum().alias("mined_count"))
        .with_columns((100 * pl.col("mined_count") / pl.col("total")).alias("mined_pct"))
        .filter(pl.col("total") >= CANDIDATE_2C_MIN_PAIRS_PER_CELL)
        .sort(["field7", "document_publication_year"])
    )
    u.save_table(field_mined_share, "candidate_fig2c_mined_share_by_field_7lines", output_dir)

    field_colors = u.field_color_map(b_fields)
    fig_a, ax_a = plt.subplots(figsize=(8, 5.5))
    for fname in b_fields:
        plotted = field_mined_share.filter(pl.col("field7") == fname)
        ax_a.plot(
            plotted.get_column("document_publication_year").to_numpy(),
            plotted.get_column("mined_pct").to_numpy(),
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=field_colors[fname],
            label=u.abbreviate_field(fname),
        )
    ax_a.set_xlabel("Publication Year")
    ax_a.set_ylabel("Mined Share of Pairs (%)")
    ax_a.set_ylim(0, 100)
    u.style_legend(ax_a.legend(fontsize=8, title="", loc="upper left", ncol=2), fontsize=8)
    u.shrink_ticks(ax_a, size=9)
    abbrevs = u.field_abbreviation_caption(b_fields)
    if abbrevs:
        u.print_caption_note("candidate_fig2c_7lines", "Abbreviated field labels: " + abbrevs)
    u.print_caption_note(
        "candidate_fig2c_7lines",
        f"Years with < {CANDIDATE_2C_MIN_PAIRS_PER_CELL} pairs in a field-year cell excluded.",
    )
    evaplot.adjust_layout(fig_a)
    u.save_figure(fig_a, "candidate-fig2c-7lines", output_dir)
    plt.close(fig_a)

    # ---- (b) 2 lines: Computer Science vs. every other field pooled ----
    cs_vs_rest_share = (
        year_df.with_columns(
            pl.when(pl.col("document_field_name_pruned") == "Computer Science")
            .then(pl.lit("Computer Science"))
            .otherwise(pl.lit("All other fields"))
            .alias("field_group")
        )
        .group_by(["document_publication_year", "field_group"])
        .agg(pl.len().alias("total"), pl.col("is_mined").sum().alias("mined_count"))
        .with_columns((100 * pl.col("mined_count") / pl.col("total")).alias("mined_pct"))
        .filter(pl.col("total") >= CANDIDATE_2C_MIN_PAIRS_PER_CELL)
        .sort(["field_group", "document_publication_year"])
    )
    u.save_table(cs_vs_rest_share, "candidate_fig2c_mined_share_cs_vs_rest", output_dir)

    palette2 = u.general_palette(2)
    fig_b, ax_b = plt.subplots(figsize=(8, 5.5))
    for group_label, color in zip(
        ["Computer Science", "All other fields"], palette2, strict=True
    ):
        plotted = cs_vs_rest_share.filter(pl.col("field_group") == group_label)
        ax_b.plot(
            plotted.get_column("document_publication_year").to_numpy(),
            plotted.get_column("mined_pct").to_numpy(),
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            label=group_label,
        )
    ax_b.set_xlabel("Publication Year")
    ax_b.set_ylabel("Mined Share of Pairs (%)")
    ax_b.set_ylim(0, 100)
    u.style_legend(ax_b.legend(fontsize=8, title="", loc="upper left"), fontsize=8)
    u.shrink_ticks(ax_b, size=9)
    u.print_caption_note(
        "candidate_fig2c_cs_vs_rest",
        f"Years with < {CANDIDATE_2C_MIN_PAIRS_PER_CELL} pairs in a group-year cell excluded.",
    )
    evaplot.adjust_layout(fig_b)
    u.save_figure(fig_b, "candidate-fig2c-cs-vs-rest", output_dir)
    plt.close(fig_b)

    # ---- Diagnostic: is the 2025 uptick a real trend or a provisional/partial-year artifact? ----
    n_2025 = df.filter(pl.col("document_publication_year") == 2025).height
    n_2024 = df.filter(pl.col("document_publication_year") == 2024).height
    print(
        f"\n2025 vs. 2024 pair counts (2025 excluded above only if it equals the current "
        f"year): 2025 n={n_2025:,}, 2024 n={n_2024:,}. Current year excluded from the "
        f"candidates: {current_year}."
    )
