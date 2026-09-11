#!/usr/bin/env python3

"""Alignment IoU trends over time: imports vs. manifest dependencies, and mentions vs.
imports/dependencies.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import evaplot
import matplotlib.pyplot as plt
import polars as pl
import utils as u
from scipy.stats import spearmanr

from rs_graph.utils.software_alignment import align_software_names

###############################################################################

# Minimum pairs per year for plotted lines; CSVs keep every year.
IOU_MIN_PAIRS_PER_YEAR = 50

# Generic mention terms removed for the generic-filtered IoU variant. Named software and
# languages (matlab, samtools, python, r, bioconductor, jupyter) are intentionally kept:
# their non-matching is signal, not noise. All names compared post-normalize_name.
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


def _top5_unmatched_rows(
    direction_specs: list[tuple[str, dict[int, set[str]]]],
    weight_by_key: dict[int, int] | None,
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


def _iou_agg_by_year(frame: pl.DataFrame, iou_col: str = "iou") -> pl.DataFrame:
    """Per-year pair count plus median/mean/p25/p75 of an IoU column."""
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


###############################################################################
# Import-vs-dependency IoU over time


def import_dependency_iou_over_time(
    output_dir: Path = u.OUTPUT_DIR,
    cutoff: float = 85.0,
    min_pairs_per_year: int = IOU_MIN_PAIRS_PER_YEAR,
) -> None:
    """
    Compute import-vs-dependency IoU over time. Builds: (1) the conditional metric (pairs
    whose repository has >=1 import AND >=1 pypi/conda/cran manifest dependency); (2) an
    unconditional companion -- same population minus the both-views-present requirement, with
    single-view pairs entering as IoU=0 and neither-view pairs excluded; (3) a three-band
    composition decomposition (no manifest / manifest-no-overlap / manifest-with-overlap) per
    year among import-bearing pairs; (4) directional top-5 unmatched-name lists. Dependency
    names are cleaned of residual comparator/junk characters before alignment.
    """
    evaplot.set_style("evaplot_rc")
    df = u.load_filtered_pairs(top_n_fields=10)

    print("\nLoading repository_import and repository_dependency from HuggingFace...")
    imports = u.load_table("repository_import")
    deps = u.load_table("repository_dependency")
    print(f"  repository_import: {len(imports):,} rows")
    print(f"  repository_dependency: {len(deps):,} rows")
    deps = u.clean_dependency_names(deps)

    deps_pr = deps.filter(pl.col("ecosystem").is_in(u.ALL_MANIFEST_ECOSYSTEMS))
    repo_with_import = set(imports.get_column("repository_id").unique().to_list())
    repo_with_dep = set(deps_pr.get_column("repository_id").unique().to_list())
    both_view_repos = repo_with_import & repo_with_dep
    either_view_repos = repo_with_import | repo_with_dep
    eligible = df.filter(pl.col("repository_id").is_in(both_view_repos))
    print(
        f"Pairs whose repository has >=1 import AND >=1 {u.ALL_MANIFEST_ECOSYSTEMS} "
        f"dependency: {eligible.height:,} of {df.height:,}"
    )

    imports_by_repo = u.normalized_names_by_id(
        imports.filter(pl.col("repository_id").is_in(both_view_repos)), "repository_id"
    )
    deps_by_repo = u.normalized_names_by_id(
        deps_pr.filter(pl.col("repository_id").is_in(both_view_repos)), "repository_id"
    )

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

    def _agg_labeled(frame: pl.DataFrame, language: str, variant: str) -> pl.DataFrame:
        return _iou_agg_by_year(frame).with_columns(
            pl.lit(language).alias("language"), pl.lit(variant).alias("variant")
        )

    by_year = pl.concat(
        [
            _agg_labeled(pair_iou, "pooled", "conditional"),
            _agg_labeled(
                pair_iou.filter(pl.col("repository_primary_language") == "Python"),
                "Python",
                "conditional",
            ),
            _agg_labeled(
                pair_iou.filter(pl.col("repository_primary_language") == "R"),
                "R",
                "conditional",
            ),
            _agg_labeled(pair_iou_unconditional, "pooled", "unconditional"),
        ]
    )
    u.save_table(by_year, "import_dependency_iou_by_year", output_dir)

    # ---- Composition decomposition: per year, among import-bearing pairs, the share with
    # (i) no manifest deps, (ii) manifest deps but zero overlap with imports, (iii) manifest
    # deps with >=1 overlap. ----
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
    u.save_table(composition, "import_dependency_composition_decomposition_by_year", output_dir)

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
    comp_colors = u.general_palette(3)
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
    u.print_caption_note(
        "import_dependency_composition_decomposition",
        f"Years with < {min_pairs_per_year} import-bearing pairs and the current partial "
        "year excluded",
    )
    u.shrink_ticks(ax_comp, size=9)
    evaplot.adjust_layout(fig_comp)
    u.save_figure(fig_comp, "import_dependency_composition_decomposition", output_dir)
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
    u.save_table(
        pl.DataFrame(unmatched_rows),
        "import_dependency_unmatched_top5_by_direction",
        output_dir,
    )

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
    palette2 = u.general_palette(2)
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
    u.style_legend(ax.legend(fontsize=8, loc="upper left"), fontsize=8)
    u.print_caption_note(
        "import_dependency_iou_over_time",
        f"Years with < {min_pairs_per_year} eligible pairs and the current partial year "
        "excluded",
    )
    u.shrink_ticks(ax, size=9)
    evaplot.adjust_layout(fig)
    u.save_figure(fig, "import_dependency_iou_over_time", output_dir)
    plt.close(fig)


###############################################################################
# Mentions-vs-imports and mentions-vs-dependencies alignment over time


def _mentions_pair_iou_frames(
    df: pl.DataFrame,
    repo_names: dict[int, list[str]],
    mentions_by_doc: dict[int, list[str]],
    source_b_label: str,
    cutoff: float,
) -> tuple[
    pl.DataFrame, pl.DataFrame, dict[str, int], dict[int, set[str]], dict[int, set[str]]
]:
    """Per-pair mentions-vs-`source_b_label` alignment. Returns the conditional pair frame
    (as-is + generic-filtered IoU per pair), the zero-pad frame of single-view pairs for the
    unconditional variant, per-term stoplist-removed counts, and per-pair unmatched mention /
    unmatched item sets (keyed by pair index).
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
            filtered_mentions = [m for m in pair_mentions if m not in GENERIC_MENTION_STOPLIST]
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
    return pl.DataFrame(rows), zero_pad, removed_counts, unmatched_mentions, unmatched_items


def _plot_mentions_alignment_figure(
    by_year: pl.DataFrame,
    output_dir: Path,
    min_pairs_per_year: int,
    year_cap: int,
) -> None:
    """Conditional-variant mean-IoU lines, both comparisons, as-is (solid) vs.
    generic-filtered (dashed). Medians are ~0 everywhere (most pairs mention none of their
    imports/deps), so the mean carries the trend signal; medians/IQR stay in the CSV.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Shared data-view colors -- imports and dependencies keep one hue each across figures.
    colors = [u.DATA_VIEW_COLORS["imports"], u.DATA_VIEW_COLORS["dependencies"]]
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
    max_mean_iou = (
        by_year.filter(pl.col("variant") == "conditional").get_column("mean_iou").max()
    )
    assert isinstance(max_mean_iou, float)
    ax.set_ylim(0, max(0.05, 1.4 * max_mean_iou))
    u.style_legend(ax.legend(fontsize=8, loc="upper left"), fontsize=8)
    u.print_caption_note(
        "mentions_alignment_over_time",
        f"Pairs with >=1 extracted mention only (conditional); years after {year_cap} "
        f"(mentions-extraction horizon) and years with < {min_pairs_per_year} pairs "
        "excluded; medians (~0 throughout) in the CSV",
    )
    u.shrink_ticks(ax, size=9)
    evaplot.adjust_layout(fig)
    u.save_figure(fig, "mentions_alignment_over_time", output_dir)
    plt.close(fig)


def mentions_alignment_over_time(
    output_dir: Path = u.OUTPUT_DIR,
    cutoff: float = 85.0,
    min_pairs_per_year: int = IOU_MIN_PAIRS_PER_YEAR,
    year_cap: int = u.MENTION_EXTRACTION_YEAR_CAP,
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
    mentions = u.clean_mention_names(mentions)

    deps_pr = deps.filter(pl.col("ecosystem").is_in(u.ALL_MANIFEST_ECOSYSTEMS))

    imports_by_repo = u.normalized_names_by_id(imports, "repository_id")
    deps_by_repo = u.normalized_names_by_id(deps_pr, "repository_id")
    mentions_by_doc = u.normalized_names_by_id(mentions, "document_id")

    df = df.filter(pl.col("document_publication_year") <= year_cap)

    stoplist_removed_rows: list[dict] = []
    unmatched_specs: list[tuple[str, dict[int, set[str]]]] = []
    frames: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}
    for comparison, repo_names, source_b_label in [
        ("mentions_vs_imports", imports_by_repo, "import"),
        ("mentions_vs_dependencies", deps_by_repo, "dependency"),
    ]:
        cond_frame, zero_pad, removed_counts, unmatched_mentions, unmatched_items = (
            _mentions_pair_iou_frames(df, repo_names, mentions_by_doc, source_b_label, cutoff)
        )
        frames[comparison] = (cond_frame, zero_pad)
        stoplist_removed_rows.extend(
            {
                "comparison": comparison,
                "term": term,
                "n_pairs_with_term_removed": n,
                "n_eligible_pairs": cond_frame.height,
            }
            for term, n in removed_counts.items()
        )
        unmatched_specs.append(
            (f"mentions_not_matched_to_{source_b_label}s", unmatched_mentions)
        )
        unmatched_specs.append((f"{source_b_label}s_not_matched_to_mentions", unmatched_items))

    if stoplist_removed_rows:
        u.save_table(
            pl.DataFrame(stoplist_removed_rows).sort(
                ["comparison", "n_pairs_with_term_removed"], descending=[False, True]
            ),
            "mentions_alignment_generic_stoplist_removed_counts",
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
    u.save_table(
        pl.DataFrame(unmatched_rows),
        "mentions_alignment_unmatched_top5_by_direction",
        output_dir,
    )
    print("\nDirectional top-5 unmatched lists (as-is variant):")
    print(pl.DataFrame(unmatched_rows))

    by_year_frames, trend_rows = [], []
    for comparison, (cond_frame, zero_pad) in frames.items():
        uncond_frame = pl.concat([cond_frame, zero_pad])
        for variant, frame in [("conditional", cond_frame), ("unconditional", uncond_frame)]:
            for mention_filter, iou_col in [
                ("as_is", "iou_as_is"),
                ("generic_filtered", "iou_generic_filtered"),
            ]:
                by_year_frames.append(
                    _iou_agg_by_year(frame, iou_col).with_columns(
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
    u.save_table(by_year, "mentions_alignment_iou_by_year", output_dir)
    trend = pl.DataFrame(trend_rows)
    # Vanishingly small p-values print as a bound rather than a literal 0.0.
    u.save_table(
        trend.with_columns(
            pl.col("p_value").map_elements(u.format_p_value, return_dtype=pl.String)
        ),
        "mentions_alignment_trend_summary",
        output_dir,
    )
    print("\nPer-pair IoU-vs-year Spearman trends:")
    print(trend)

    _plot_mentions_alignment_figure(by_year, output_dir, min_pairs_per_year, year_cap)


###############################################################################
# Top software per view (own lists, independent of any cross-view alignment)

TOP_SOFTWARE_PER_VIEW_N = 10
RARE_SOFTWARE_MIN_PAIRS = 3
PAIR_USAGE_TRIM_QUANTILE = 0.99


def _trim_and_count_view(
    view_frame: pl.DataFrame,
    view_label: str,
    top_n: int,
    min_pairs: int = RARE_SOFTWARE_MIN_PAIRS,
    trim_quantile: float = PAIR_USAGE_TRIM_QUANTILE,
) -> pl.DataFrame:
    """Rank a view's software names by the number of pairs each appears in. `view_frame` is a
    long (document_id, repository_id, software_name_normalized) frame, one row per pair-name.
    Cleaning mirrors the mention-predictors regression: drop entire pairs whose per-pair
    unique-name count exceeds the `trim_quantile` percentile, then drop names appearing in
    fewer than `min_pairs` pairs.
    """
    pair_counts = view_frame.group_by(["document_id", "repository_id"]).agg(
        pl.len().alias("n_names")
    )
    threshold = pair_counts.get_column("n_names").quantile(trim_quantile)
    extreme_pairs = pair_counts.filter(pl.col("n_names") > threshold).select(
        "document_id", "repository_id"
    )
    trimmed = view_frame.join(extreme_pairs, on=["document_id", "repository_id"], how="anti")
    n_eligible_pairs = trimmed.select("document_id", "repository_id").unique().height
    print(
        f"{view_label}: trimmed {extreme_pairs.height:,} pairs above the "
        f"p{trim_quantile * 100:.0f} per-pair name count ({threshold:.0f}); "
        f"{n_eligible_pairs:,} pairs remain"
    )
    return (
        trimmed.group_by("software_name_normalized")
        .agg(pl.len().alias("n_pairs"))
        .filter(pl.col("n_pairs") >= min_pairs)
        .sort(["n_pairs", "software_name_normalized"], descending=[True, False])
        .head(top_n)
        .with_columns(
            pl.lit(view_label).alias("view"),
            pl.int_range(1, pl.len() + 1).alias("rank"),
            pl.lit(n_eligible_pairs).alias("n_eligible_pairs"),
        )
        .select("view", "rank", "software_name_normalized", "n_pairs", "n_eligible_pairs")
    )


def top_software_by_view(
    output_dir: Path = u.OUTPUT_DIR,
    top_n: int = TOP_SOFTWARE_PER_VIEW_N,
    year_cap: int = u.MENTION_EXTRACTION_YEAR_CAP,
) -> None:
    """
    Rank each data view's own most common software, independent of any cross-view alignment:
    top mentioned software (as-is and with `GENERIC_MENTION_STOPLIST` terms removed), top
    imported software, and top depended-upon (pypi/conda/cran manifest) software. Counts are
    the number of standard-filtered (document, repository) pairs the name appears in for that
    view; mention counts are capped at `year_cap` for the mentions-extraction horizon.
    Name cleaning matches the rest of the paper (dependency/mention cleaning, per-pair
    usage-count outlier trimming, rare-name floor).
    """
    df = u.load_filtered_pairs(top_n_fields=10)

    print("\nLoading imports/dependencies/mentions from HuggingFace...")
    imports = u.load_table("repository_import")
    deps = u.load_table("repository_dependency")
    mentions = u.load_table("document_software_mention")
    deps = u.clean_dependency_names(deps)
    mentions = u.clean_mention_names(mentions)
    deps_pr = deps.filter(pl.col("ecosystem").is_in(u.ALL_MANIFEST_ECOSYSTEMS))

    pair_pool = df.select("document_id", "repository_id").unique(
        subset=["document_id", "repository_id"]
    )
    mention_pair_pool = (
        df.filter(pl.col("document_publication_year") <= year_cap)
        .select("document_id", "repository_id")
        .unique(subset=["document_id", "repository_id"])
    )

    import_long = pair_pool.join(
        imports.select("repository_id", "software_name_normalized").unique(),
        on="repository_id",
    )
    dep_long = pair_pool.join(
        deps_pr.select("repository_id", "software_name_normalized").unique(),
        on="repository_id",
    )
    mention_long = mention_pair_pool.join(
        mentions.select("document_id", "software_name_normalized").unique(),
        on="document_id",
    )
    mention_long_filtered = mention_long.filter(
        ~pl.col("software_name_normalized").is_in(GENERIC_MENTION_STOPLIST)
    )

    top_frames = [
        _trim_and_count_view(mention_long, "mentions_as_is", top_n),
        _trim_and_count_view(mention_long_filtered, "mentions_stoplist_filtered", top_n),
        _trim_and_count_view(import_long, "imports", top_n),
        _trim_and_count_view(dep_long, "dependencies", top_n),
    ]
    top_table = pl.concat(top_frames)
    u.save_table(top_table, "top_software_by_view", output_dir)
    print("\nTop software per view (own lists):")
    for frame in top_frames:
        print(frame)
