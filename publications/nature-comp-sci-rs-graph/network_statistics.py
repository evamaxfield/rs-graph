#!/usr/bin/env python3

"""Linking and network statistics: the repository-creation-vs-publication date-delta figure,
the co-authorship network rebuild, and full-network entity/edge counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import evaplot
import matplotlib.pyplot as plt
import polars as pl
import rustworkx as rx
import seaborn as sns
import utils as u

###############################################################################
# Date-delta figure (repository creation vs. publication date)


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
    with vertical marker lines at the median and the window-defining 10th/90th percentiles
    (p5/p95 stay in the percentile CSV, not plotted).
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
        leg = ax.legend(fontsize=8)
        u.style_legend(leg)
    u.shrink_ticks(ax, size=8)
    return stats


def _one_to_one(df: pl.DataFrame) -> pl.DataFrame:
    """Strict one-to-one linking: drop every row whose document_id or repository_id is
    duplicated within `df` (same dedup as the snowball-sampling prep notebook).
    """
    return df.unique(subset="document_id", keep="none").unique(
        subset="repository_id", keep="none"
    )


def date_delta_figure(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Build the date-delta figure justifying the 10th/90th-percentile
    repository-creation-vs-publication-date inclusion window used elsewhere in the pipeline.

    Uses strict one-to-one linking (drop every document linked to more than one repository
    and every repository linked to more than one document), matching the dedup in
    `notebooks/snowball-sampling-discovery-prep.ipynb`, applied once over the whole filtered
    population (seed and mined together). The pooled figure and its window-defining
    percentiles use seed pairs only -- mined pairs were selected *through* the window, so
    pooling them would be circular. Mined pairs appear only as a sixth panel on the
    by-source supplemental; the by-document-type supplemental is also seed-only.
    """
    evaplot.set_style("evaplot_rc")
    df = u.load_filtered_pairs(top_n_fields=10)

    # ---- Strict one-to-one linking (same dedup as the snowball-sampling prep notebook) ----
    n_before_dedup = df.height
    df = _one_to_one(df)
    print(
        f"One-to-one dedup (drop all rows with a duplicated document_id, then all rows with "
        f"a duplicated repository_id): {df.height:,} of {n_before_dedup:,} filtered pairs "
        f"remain"
    )

    # ---- Split seed from mined for the per-source / mined breakdown ----
    mined_df = df.filter(pl.col("link_processing_iteration").is_not_null())
    df = df.filter(pl.col("link_processing_iteration").is_null())
    print(
        f"Seed pairs (link_processing_iteration IS NULL): {df.height:,}; "
        f"mined pairs (sixth supplemental panel only): {mined_df.height:,}"
    )

    delta_expr = (
        (
            pl.col("document_publication_date_parsed").cast(pl.Datetime)
            - pl.col("repository_creation_datetime_parsed")
        )
        .dt.total_days()
        .alias("publication_date_creation_date_diff")
    )
    df = df.with_columns(delta_expr)
    mined_df = mined_df.with_columns(delta_expr)
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
    u.save_figure(fig, "date_delta_pooled", output_dir)
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
    fig_src, axes_src = plt.subplots(2, 3, figsize=(12.5, 8))
    axes_src_flat = axes_src.flatten()
    # Every panel gets its own legend -- the marker-line values differ per panel.
    for i, (ax_i, (breakdown, label, sub_diffs)) in enumerate(
        zip(axes_src_flat, panel_specs, strict=False)
    ):
        sub_stats = _plot_date_delta_panel(
            ax_i, sub_diffs, title=f"{label} (n={sub_diffs.len():,})", legend=True
        )
        u.add_panel_label(ax_i, chr(65 + i))
        percentile_rows.append(
            {"breakdown": breakdown, "group": label, "n": sub_diffs.len(), **sub_stats}
        )
    for j in range(len(panel_specs), len(axes_src_flat)):
        axes_src_flat[j].set_axis_off()
    u.print_caption_note(
        "date_delta_supplemental_by_seed_source",
        "Publication Date - Repository Creation Date Difference, by Seed Source "
        "(+ Mined Pairs)",
    )
    evaplot.adjust_layout(fig_src)
    u.save_figure(fig_src, "date_delta_supplemental_by_seed_source", output_dir)
    plt.close(fig_src)

    # ---- Supplemental 2: by document type bucket (article / preprint / other) ----
    doctypes = ["article", "preprint", "other"]
    fig_dt, axes_dt = plt.subplots(1, 3, figsize=(12, 4.5))
    for i, (ax_i, doctype) in enumerate(zip(axes_dt, doctypes, strict=True)):
        sub_diffs = df.filter(pl.col("document_type_bucket") == doctype).get_column(
            "publication_date_creation_date_diff"
        )
        sub_stats = _plot_date_delta_panel(
            ax_i, sub_diffs, title=f"{doctype} (n={sub_diffs.len():,})", legend=True
        )
        u.add_panel_label(ax_i, chr(65 + i))
        percentile_rows.append(
            {"breakdown": "document_type", "group": doctype, "n": sub_diffs.len(), **sub_stats}
        )
    u.print_caption_note(
        "date_delta_supplemental_by_document_type",
        "Publication Date - Repository Creation Date Difference, by Document Type",
    )
    evaplot.adjust_layout(fig_dt)
    u.save_figure(fig_dt, "date_delta_supplemental_by_document_type", output_dir)
    plt.close(fig_dt)

    percentile_table = pl.DataFrame(percentile_rows)
    u.save_table(percentile_table, "date_delta_percentiles", output_dir)
    print("\nAll date-delta percentiles by breakdown:")
    print(percentile_table)


###############################################################################
# Co-authorship network
#
# Rebuilds the component statistics cited in the manuscript's Results text. The docs-site
# pipeline (`web/data-prep/queries/coauthorship_network.py`) defines the edge-construction
# rule and author-count bound reproduced here.


def coauthorship_network(
    output_dir: Path = u.OUTPUT_DIR,
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
    C(n_authors, 2) edges of combinatorial noise. Single-author-only isolates (degree 0)
    are excluded from the node population before component statistics.

    The researcher-developer identity-link confidence filter is not applied here: it
    constrains researcher<->developer identity, and co-authorship is a purely
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
    # The bound caps edge generation only -- the node set stays unbounded (`authors`);
    # degree-0 isolates are excluded later, before component statistics.
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
    graph = rx.PyGraph()
    index_by_researcher = dict(zip(node_ids, graph.add_nodes_from(node_ids), strict=True))
    graph.add_edges_from(
        [
            (index_by_researcher[researcher_id], index_by_researcher[researcher_id_b], weight)
            for researcher_id, researcher_id_b, weight in edges.iter_rows()
        ]
    )

    n_nodes_with_isolates = graph.num_nodes()
    n_edges = graph.num_edges()
    print(
        f"\nCo-authorship network (before isolate exclusion): "
        f"{n_nodes_with_isolates:,} researcher nodes, {n_edges:,} edges"
    )

    # Exclude single-author-only isolates (degree 0) before computing components.
    isolates = [idx for idx in graph.node_indices() if graph.degree(idx) == 0]
    graph.remove_nodes_from(isolates)
    n_nodes = graph.num_nodes()
    print(
        f"Excluded {len(isolates):,} single-author-only isolates (degree 0); "
        f"{n_nodes:,} researcher nodes remain"
    )

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
    u.save_table(results, "coauthorship_component_sizes", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_researchers_before_isolate_exclusion": n_nodes_with_isolates,
        "n_isolates_excluded": len(isolates),
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
    with open(output_dir / "coauthorship_network_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {output_dir / 'coauthorship_network_summary.json'}")

    u.print_caption_note(
        "coauthorship_network",
        "Component statistics are computed over researchers with at least one co-authorship "
        "edge; single-author-only researchers (degree 0) are excluded from the node "
        "population.",
    )
    print("\n--- Co-authorship network statistics ---")
    print(f"Connected components: {n_components:,}")
    print(f"Largest component: {largest_size:,} researchers ({largest_pct:.1f}%)")
    print(f"Next-largest component: {next_largest_size:,} researchers")
    print(
        "Docs-site pipeline reference (comparison only): 11,568 components / "
        "91% in largest component / 46 in next-largest."
    )
    print("----------------------------------------\n")


###############################################################################
# Full-network entity/edge counts


def network_entity_edge_counts(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Compute the manuscript's full-network counts -- article, repository, researcher, and
    developer-account node counts, plus authorship, contribution, article-repository-link,
    and researcher-developer identity edge counts. Entities are derived from the
    standard-filtered pairs (pair-level filtering first). Identity-link counts use the
    connection definition (same rule as `mining_rounds_table`'s iteration attribution): an
    identity counts only if at least one retained article-repository pair directly connects
    the researcher (as an author of the article) to the developer account (as a contributor
    to the repository). Counts are reported unfiltered and at both the >=0.9 and >=0.97
    confidence thresholds.
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

    # Connection definition: the (researcher, developer account) pairs directly connected
    # through a retained article-repository pair -- the researcher authored the article and
    # the developer account contributed to that same pair's repository.
    pair_pool = pairs.select("document_id", "repository_id").unique()
    connected_rd = (
        authorship.join(pair_pool, on="document_id")
        .join(contribution, on="repository_id")
        .select("researcher_id", "developer_account_id")
        .unique()
    )
    print(
        f"Researcher-developer pairs connected through a retained article-repository pair: "
        f"{connected_rd.height:,}"
    )

    def _connected_identity_count(threshold: float | None) -> int:
        candidates = (
            rdal
            if threshold is None
            else rdal.filter(pl.col("predictive_model_confidence") >= threshold)
        )
        return (
            candidates.select("researcher_id", "developer_account_id")
            .unique()
            .join(connected_rd, on=["researcher_id", "developer_account_id"], how="semi")
            .height
        )

    identity_counts = {
        # Unfiltered baseline: every connected identity link, at any confidence.
        "n_identity_links_connected_unfiltered": _connected_identity_count(None),
        "n_identity_links_connected_confidence_gte_0.97": _connected_identity_count(0.97),
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
    with open(output_dir / "network_entity_edge_counts.json", "w") as f:
        json.dump(summary, f, indent=2)
    u.save_table(
        pl.DataFrame(
            {"statistic": list(summary.keys()), "value": [float(v) for v in summary.values()]}
        ),
        "network_entity_edge_counts",
        output_dir,
    )

    print("\n--- Full-network entity/edge counts ---")
    for k, v in summary.items():
        print(f"  {k}: {v:,}")
    print("---------------------------------------\n")
