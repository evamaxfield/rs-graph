#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import evaplot
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl
import rustworkx as rx
import seaborn as sns
import typer
from datasets import Dataset, load_dataset

###############################################################################

app = typer.Typer()

THIS_FILE_PATH = Path(__file__).resolve()
THIS_DIR = THIS_FILE_PATH.parent

HF_DATASET = "sci-soft-collections/rs-graph-v2-full"
CONFIDENCE_THRESHOLD = 0.9994
RDAL_CONFIDENCE_THRESHOLD = 0.99
RARE_THRESHOLD = 3
TOP_N_FIELDS = 10
RANDOM_SEED = 42

# Panel background / text colors for three-views subfigure
PANEL_BG_LIGHT = "#f8f4ee"
PANEL_BG_DARK = "#1e1e2e"
PANEL_BG_MID = "#2a2a3e"
TEXT_LIGHT = "#cccccc"
TEXT_DARK = "#1a1a2e"

MENTION_EXCLUDE_NORMALIZED: set[str] = {
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

###############################################################################
# Data loading


def load_table(table: str) -> pl.DataFrame:
    ds = load_dataset(HF_DATASET, table, split="train")
    assert isinstance(ds, Dataset)
    df = pl.from_arrow(ds.data.table)
    assert isinstance(df, pl.DataFrame)
    return df


def _load_pair_metadata() -> pl.DataFrame:
    documents = load_table("document")
    article_repo_links = load_table("document_repository_link")
    repositories = load_table("repository")
    document_topics = load_table("document_topic")
    topics = load_table("topic")
    document_contributors = load_table("document_contributor")
    researchers = load_table("researcher")

    document_contributors_with_cites = document_contributors.join(
        researchers.select(
            pl.col("id").alias("researcher_id"),
            pl.col("cited_by_count").alias("researcher_cited_by_count"),
        ),
        on="researcher_id",
        how="left",
    )
    document_author_stats = (
        document_contributors_with_cites.group_by("document_id")
        .agg(
            pl.count("researcher_id").alias("document_author_count"),
            pl.mean("researcher_cited_by_count").alias("document_author_mean_citations"),
        )
        .with_columns(
            pl.col("document_author_mean_citations")
            .log1p()
            .alias("document_log_author_mean_citations")
        )
    )

    document_topics = document_topics.sort("score", descending=True)
    document_topics = document_topics.unique(subset="document_id", keep="first")
    document_topics = (
        document_topics.select(pl.col("document_id"), pl.col("topic_id"))
        .join(
            topics.select(
                pl.col("id").alias("topic_id"),
                pl.col("field_name").alias("document_field_name"),
                pl.col("domain_name").alias("document_domain_name"),
            ),
            on="topic_id",
        )
        .select(
            pl.col("document_id"),
            pl.col("document_field_name"),
            pl.col("document_domain_name"),
        )
    )

    merged = (
        article_repo_links.select(
            pl.col("id").alias("document_repository_link_id"),
            pl.col("document_id"),
            pl.col("repository_id"),
            pl.col("dataset_source_id"),
            pl.col("predictive_model_confidence"),
        )
        .join(
            documents.select(
                *[pl.col(col).alias(f"document_{col}") for col in documents.columns]
            ),
            on="document_id",
        )
        .join(
            repositories.select(
                *[pl.col(col).alias(f"repository_{col}") for col in repositories.columns]
            ),
            on="repository_id",
        )
        .join(document_topics, on="document_id")
        .join(document_author_stats, on="document_id", how="left")
    )

    merged = merged.with_columns(
        pl.col("document_publication_date")
        .str.to_date("%Y-%m-%d")
        .alias("document_publication_date_parsed"),
    ).with_columns(
        pl.col("document_publication_date_parsed").dt.year().alias("document_publication_year"),
    )

    # Core filter 1: published after GitHub founding
    merged = merged.filter(pl.col("document_publication_year") >= 2008).with_columns(
        (pl.col("document_publication_year") - pl.col("document_publication_year").min()).alias(
            "document_years_since_earliest"
        )
    )

    # Core filter 2: confidence threshold
    merged = merged.filter(
        (pl.col("predictive_model_confidence") > CONFIDENCE_THRESHOLD)
        | (pl.col("predictive_model_confidence").is_null())
    )

    # Core filter 3: one-to-one pairs only
    merged = merged.unique(subset="document_id", keep="none").unique(
        subset="repository_id", keep="none"
    )

    # Compute document_field_name_pruned (top 10 + "Other")
    top_field_names = (
        merged.get_column("document_field_name")
        .value_counts(sort=True)
        .head(TOP_N_FIELDS)
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


###############################################################################
# Software usage helpers (mirrors mentions-imports-deps-annotation.py)


def _remove_extremely_rare_software_usage(
    usage_df: pl.DataFrame,
    rare_usage_threshold: int = RARE_THRESHOLD,
    compute_ecosystem_prefix: bool = False,
    use_ecosystem_from_data: bool = False,
    exclude_generic_mentions: bool = False,
) -> pl.DataFrame:
    software_col = "software_name_normalized"

    if compute_ecosystem_prefix:
        check_for_py = (
            pl.col("file_paths_lower").str.contains(r"\.py\;")
            | pl.col("file_paths_lower").str.contains(r"\.py$")
            | pl.col("file_paths_lower").str.contains(r"\.ipynb\;")
            | pl.col("file_paths_lower").str.contains(r"\.ipynb$")
        )
        check_for_r = (
            pl.col("file_paths_lower").str.contains(r"\.r\;")
            | pl.col("file_paths_lower").str.contains(r"\.r$")
            | pl.col("file_paths_lower").str.contains(r"\.rmd\;")
            | pl.col("file_paths_lower").str.contains(r"\.rmd$")
        )
        usage_df = (
            usage_df.with_columns(
                pl.col("file_paths").str.to_lowercase().alias("file_paths_lower")
            )
            .with_columns(
                pl.when(check_for_py & check_for_r)
                .then(pl.lit("mixed"))
                .when(check_for_py)
                .then(pl.lit("py"))
                .when(check_for_r)
                .then(pl.lit("r"))
                .otherwise(pl.lit("other"))
                .alias("ecosystem")
            )
            .with_columns(
                (pl.col("ecosystem") + pl.lit(":") + pl.col(software_col)).alias(
                    "ecosystem_software_name_normalized"
                )
            )
        )
        software_col = "ecosystem_software_name_normalized"
        usage_df = usage_df.filter(pl.col("ecosystem").is_in(["py", "r"]))

    elif use_ecosystem_from_data:
        usage_df = usage_df.with_columns(
            (pl.col("ecosystem") + pl.lit(":") + pl.col(software_col)).alias(
                "ecosystem_software_name_normalized"
            )
        )
        software_col = "ecosystem_software_name_normalized"

    elif exclude_generic_mentions:
        usage_df = usage_df.filter(~pl.col(software_col).is_in(MENTION_EXCLUDE_NORMALIZED))

    usage_counts = usage_df.group_by(software_col).agg(pl.len().alias("usage_count"))
    non_rare = (
        usage_counts.filter(pl.col("usage_count") >= rare_usage_threshold)
        .get_column(software_col)
        .to_list()
    )
    return usage_df.filter(pl.col(software_col).is_in(non_rare))


@dataclass
class SoftwareUsageTables:
    pair_metadata: pl.DataFrame
    repository_imports: pl.DataFrame
    repository_dependencies: pl.DataFrame
    document_software_mentions: pl.DataFrame


def _build_software_usage_tables(pair_metadata: pl.DataFrame) -> SoftwareUsageTables:
    repository_imports = load_table("repository_import")
    repository_dependencies = load_table("repository_dependency")
    document_software_mentions = load_table("document_software_mention")

    # Remove extremely rare software
    repository_imports = _remove_extremely_rare_software_usage(
        repository_imports,
        compute_ecosystem_prefix=True,
    )
    repository_dependencies = _remove_extremely_rare_software_usage(
        repository_dependencies,
        use_ecosystem_from_data=True,
    )
    document_software_mentions = _remove_extremely_rare_software_usage(
        document_software_mentions,
        exclude_generic_mentions=True,
    )

    # Inner-join to pair_metadata (restricts to filtered pairs, downstream of core filters)
    pair_cols = pair_metadata.select(
        pl.col("document_repository_link_id"),
        pl.col("document_id"),
        pl.col("repository_id"),
    )
    repository_imports = repository_imports.join(pair_cols, on="repository_id", how="inner")
    repository_dependencies = repository_dependencies.join(
        pair_cols, on="repository_id", how="inner"
    )
    document_software_mentions = document_software_mentions.join(
        pair_cols, on="document_id", how="inner"
    )

    # Add has_* flags
    has_imports = repository_imports.group_by("document_repository_link_id").agg(
        has_imports=pl.lit(True)
    )
    has_deps = repository_dependencies.group_by("document_repository_link_id").agg(
        has_dependencies=pl.lit(True)
    )
    has_mentions = document_software_mentions.group_by("document_repository_link_id").agg(
        has_software_mentions=pl.lit(True)
    )

    pair_metadata = (
        pair_metadata.join(has_imports, on="document_repository_link_id", how="left")
        .join(has_deps, on="document_repository_link_id", how="left")
        .join(has_mentions, on="document_repository_link_id", how="left")
        .fill_null(False)
    )

    return SoftwareUsageTables(
        pair_metadata=pair_metadata,
        repository_imports=repository_imports,
        repository_dependencies=repository_dependencies,
        document_software_mentions=document_software_mentions,
    )


###############################################################################
# Subfigure A: Network diagram


def _build_network_graph(
    pair_metadata: pl.DataFrame,
    document_contributors: pl.DataFrame,
    researchers: pl.DataFrame,
    repository_contributors: pl.DataFrame,
    developer_accounts: pl.DataFrame,
    researcher_dev_links: pl.DataFrame,
    dataset_sources: pl.DataFrame,
) -> rx.PyGraph:
    # Filter researcher-developer links to high confidence
    rdal = researcher_dev_links.filter(
        pl.col("predictive_model_confidence") > RDAL_CONFIDENCE_THRESHOLD
    )

    # Classify each dataset_source_id as seed vs mined
    mined_source_ids = set(
        dataset_sources.filter(pl.col("name").str.to_lowercase().str.contains("snowball"))
        .get_column("id")
        .to_list()
    )
    seed_source_ids = set(
        dataset_sources.filter(~pl.col("name").str.to_lowercase().str.contains("snowball"))
        .get_column("id")
        .to_list()
    )

    # Find the researcher with the most articles in pair_metadata who has a high-conf match
    researcher_article_counts = (
        document_contributors.join(
            pair_metadata.select(pl.col("document_id")), on="document_id", how="inner"
        )
        .group_by("researcher_id")
        .agg(pl.len().alias("article_count"))
    )
    # Only keep researchers who have a high-conf developer link
    linked_researcher_ids = rdal.get_column("researcher_id").to_list()
    researcher_article_counts = researcher_article_counts.filter(
        pl.col("researcher_id").is_in(linked_researcher_ids)
    )

    # Prefer researchers who appear in BOTH seed and mined pairs
    researcher_doc_ids = document_contributors.join(
        pair_metadata.select(pl.col("document_id"), pl.col("dataset_source_id")),
        on="document_id",
        how="inner",
    )
    has_seed = (
        researcher_doc_ids.filter(pl.col("dataset_source_id").is_in(seed_source_ids))
        .get_column("researcher_id")
        .unique()
        .to_list()
    )
    has_mined = (
        researcher_doc_ids.filter(pl.col("dataset_source_id").is_in(mined_source_ids))
        .get_column("researcher_id")
        .unique()
        .to_list()
    )
    both_types_ids = set(has_seed) & set(has_mined)
    candidates_both = researcher_article_counts.filter(
        pl.col("researcher_id").is_in(both_types_ids)
    )
    candidates = candidates_both if len(candidates_both) > 0 else researcher_article_counts

    ego_researcher_id = (
        candidates.sort("article_count", descending=True)
        .head(1)
        .get_column("researcher_id")
        .item()
    )

    # Get ego researcher's matched developer account (highest confidence)
    ego_dev_link = (
        rdal.filter(pl.col("researcher_id") == ego_researcher_id)
        .sort("predictive_model_confidence", descending=True)
        .head(1)
    )
    ego_dev_account_id = ego_dev_link.get_column("developer_account_id").item()

    # Get ego researcher's articles, ensuring both seed and mined types are represented
    ego_all_docs = document_contributors.filter(
        pl.col("researcher_id") == ego_researcher_id
    ).join(
        pair_metadata.select(pl.col("document_id"), pl.col("dataset_source_id")),
        on="document_id",
        how="inner",
    )
    seed_doc_ids = (
        ego_all_docs.filter(pl.col("dataset_source_id").is_in(seed_source_ids))
        .get_column("document_id")
        .to_list()[:4]
    )
    mined_doc_ids = (
        ego_all_docs.filter(pl.col("dataset_source_id").is_in(mined_source_ids))
        .get_column("document_id")
        .to_list()[:4]
    )
    ego_article_doc_ids = seed_doc_ids + mined_doc_ids

    # Get linked repos for those articles
    ego_pairs = pair_metadata.filter(pl.col("document_id").is_in(ego_article_doc_ids))

    # Source name lookup for seed vs mined
    source_name_lut = dict(
        zip(
            dataset_sources.get_column("id").to_list(),
            dataset_sources.get_column("name").to_list(),
            strict=False,
        )
    )

    # Build graph
    graph: rx.PyGraph = rx.PyGraph()
    node_idx_map: dict[tuple[str, int], int] = {}

    # Add ego researcher and developer
    ego_res_row = researchers.filter(pl.col("id") == ego_researcher_id).row(0, named=True)
    _get_or_add_graph_node(
        graph, node_idx_map, "researcher", ego_researcher_id, ego_res_row["name"].split()[-1]
    )

    ego_dev_row = developer_accounts.filter(pl.col("id") == ego_dev_account_id).row(
        0, named=True
    )
    _get_or_add_graph_node(
        graph, node_idx_map, "developer", ego_dev_account_id, ego_dev_row["username"]
    )

    graph.add_edge(
        node_idx_map[("researcher", ego_researcher_id)],
        node_idx_map[("developer", ego_dev_account_id)],
        {"type": "identity"},
    )

    for pair_row in ego_pairs.iter_rows(named=True):
        _add_pair_to_graph(
            pair_row,
            graph,
            node_idx_map,
            ego_researcher_id,
            document_contributors,
            researchers,
            repository_contributors,
            developer_accounts,
            source_name_lut,
        )

    return graph


def _get_or_add_graph_node(
    graph: rx.PyGraph,
    node_idx_map: dict[tuple[str, int], int],
    node_type: str,
    node_id: int,
    label: str,
) -> int:
    key = (node_type, node_id)
    if key not in node_idx_map:
        idx = graph.add_node({"type": node_type, "id": node_id, "label": label})
        node_idx_map[key] = idx
    return node_idx_map[key]


def _add_pair_to_graph(
    pair_row: dict,
    graph: rx.PyGraph,
    node_idx_map: dict[tuple[str, int], int],
    ego_researcher_id: int,
    document_contributors: pl.DataFrame,
    researchers: pl.DataFrame,
    repository_contributors: pl.DataFrame,
    developer_accounts: pl.DataFrame,
    source_name_lut: dict[int, str],
) -> None:
    doc_id = pair_row["document_id"]
    repo_id = pair_row["repository_id"]
    source_name = source_name_lut.get(pair_row["dataset_source_id"], "")
    is_mined = "snowball" in source_name.lower()

    doc_idx = _get_or_add_graph_node(
        graph, node_idx_map, "article", doc_id, pair_row["document_title"][:20] + "..."
    )
    repo_idx = _get_or_add_graph_node(
        graph, node_idx_map, "repository", repo_id, pair_row["repository_name"][:20]
    )

    graph.add_edge(doc_idx, repo_idx, {"type": "mined" if is_mined else "seed"})

    res_idx = node_idx_map[("researcher", ego_researcher_id)]
    if not graph.has_edge(res_idx, doc_idx):
        graph.add_edge(res_idx, doc_idx, {"type": "authored_by"})

    for author_row in (
        document_contributors.filter(
            (pl.col("document_id") == doc_id) & (pl.col("researcher_id") != ego_researcher_id)
        )
        .head(3)
        .iter_rows(named=True)
    ):
        rid = author_row["researcher_id"]
        res_info = researchers.filter(pl.col("id") == rid)
        if len(res_info) == 0:
            continue
        name = res_info.row(0, named=True)["name"].split()[-1]
        r_idx = _get_or_add_graph_node(graph, node_idx_map, "researcher", rid, name)
        if not graph.has_edge(r_idx, doc_idx):
            graph.add_edge(r_idx, doc_idx, {"type": "authored_by"})

    for contrib_row in (
        repository_contributors.filter(pl.col("repository_id") == repo_id)
        .head(3)
        .iter_rows(named=True)
    ):
        dev_id = contrib_row["developer_account_id"]
        dev_info = developer_accounts.filter(pl.col("id") == dev_id)
        if len(dev_info) == 0:
            continue
        uname = dev_info.row(0, named=True)["username"]
        d_idx = _get_or_add_graph_node(graph, node_idx_map, "developer", dev_id, uname)
        if not graph.has_edge(repo_idx, d_idx):
            graph.add_edge(repo_idx, d_idx, {"type": "contributed_to"})


def _draw_network(ax: mpl.axes.Axes, graph: rx.PyGraph, colors: list[str]) -> None:
    positions = rx.graph_spring_layout(graph, seed=RANDOM_SEED)  # type: ignore[call-arg]

    node_type_cfg = {
        "article": {
            "color": colors[0],
            "marker": "o",
            "size": 120,
            "zorder": 3,
            "hollow": False,
        },
        "repository": {
            "color": colors[1],
            "marker": "s",
            "size": 120,
            "zorder": 3,
            "hollow": False,
        },
        "researcher": {
            "color": colors[2],
            "marker": "D",
            "size": 100,
            "zorder": 3,
            "hollow": True,
        },
        "developer": {
            "color": colors[3],
            "marker": "^",
            "size": 100,
            "zorder": 3,
            "hollow": True,
        },
    }
    edge_type_cfg = {
        "authored_by": {"color": "#aaaaaa", "lw": 0.8, "ls": "-", "zorder": 1},
        "seed": {"color": colors[4], "lw": 1.5, "ls": "-", "zorder": 2},
        "mined": {"color": colors[5], "lw": 1.5, "ls": "--", "zorder": 2},
        "contributed_to": {"color": "#aaaaaa", "lw": 0.8, "ls": "-", "zorder": 1},
        "identity": {"color": colors[6], "lw": 2.5, "ls": "-", "zorder": 4},
    }

    # Draw edges
    for edge_idx in graph.edge_indices():
        src, tgt = graph.get_edge_endpoints_by_index(edge_idx)
        edge_data = graph.get_edge_data_by_index(edge_idx)
        etype = edge_data["type"]
        cfg = edge_type_cfg.get(etype, edge_type_cfg["authored_by"])
        x0, y0 = positions[src]
        x1, y1 = positions[tgt]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=cfg["color"],
            lw=cfg["lw"],
            ls=cfg["ls"],
            zorder=cfg["zorder"],
            alpha=0.7,
        )

    # Draw nodes
    for ntype, cfg in node_type_cfg.items():
        xs, ys, labels = [], [], []
        for node_idx in graph.node_indices():
            nd = graph[node_idx]
            if nd["type"] == ntype:
                x, y = positions[node_idx]
                xs.append(x)
                ys.append(y)
                labels.append(nd["label"])
        if xs:
            node_color = str(cfg["color"])
            if cfg["hollow"]:
                ax.scatter(
                    xs,
                    ys,
                    facecolors="none",
                    edgecolors=node_color,
                    marker=str(cfg["marker"]),
                    s=cfg["size"],
                    zorder=cfg["zorder"],
                    linewidths=1.5,
                )
            else:
                ax.scatter(
                    xs,
                    ys,
                    c=node_color,
                    marker=str(cfg["marker"]),
                    s=cfg["size"],
                    zorder=cfg["zorder"],
                    edgecolors=node_color,
                    linewidths=1.5,
                )
            for x, y, lbl in zip(xs, ys, labels, strict=False):
                ax.annotate(
                    lbl,
                    (x, y),
                    fontsize=5,
                    ha="center",
                    va="bottom",
                    xytext=(0, 8),
                    textcoords="offset points",
                )

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=colors[0], edgecolor=colors[0], label="Article"),
        mpatches.Patch(facecolor=colors[1], edgecolor=colors[1], label="Repository"),
        mpatches.Patch(facecolor="none", edgecolor=colors[2], label="Researcher"),
        mpatches.Patch(facecolor="none", edgecolor=colors[3], label="Developer"),
        mpl.lines.Line2D([], [], color=colors[4], lw=1.5, label="Seed link"),
        mpl.lines.Line2D([], [], color=colors[5], lw=1.5, ls="--", label="Mined link"),
        mpl.lines.Line2D([], [], color=colors[6], lw=2.5, label="Matched identity"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=6,
        framealpha=0.8,
        ncol=1,
    )
    ax.set_axis_off()
    ax.set_title("a", fontweight="bold", loc="left", fontsize=10)


###############################################################################
# Subfigure B: Descriptive overview


def _draw_descriptive_overview(
    ax_b1: mpl.axes.Axes,
    ax_b2: mpl.axes.Axes,
    ax_b3: mpl.axes.Axes,
    ax_b4: mpl.axes.Axes,
    pair_metadata: pl.DataFrame,
    dataset_sources: pl.DataFrame,
    software: SoftwareUsageTables,
    colors_8: list[str],
    colors_11: list[str],
) -> None:
    # B1: Pairs per year
    year_counts = (
        pair_metadata.group_by("document_publication_year")
        .agg(pl.len().alias("count"))
        .sort("document_publication_year")
        .to_pandas()
    )
    sns.barplot(
        data=year_counts,
        x="document_publication_year",
        y="count",
        color=colors_8[0],
        ax=ax_b1,
    )
    ax_b1.set_xlabel("Publication Year")
    ax_b1.set_ylabel("Pairs")
    ax_b1.set_title("b", fontweight="bold", loc="left", fontsize=10)
    tick_years = sorted(year_counts["document_publication_year"].unique())
    tick_positions = [i for i, y in enumerate(tick_years) if y % 2 == 0]
    tick_labels = [tick_years[i] for i in tick_positions]
    ax_b1.set_xticks(tick_positions)
    ax_b1.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)

    # B2: Pairs by field (top 10 + Other)
    field_counts = (
        pair_metadata.group_by("document_field_name_pruned")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .to_pandas()
    )
    palette_b2 = {
        row["document_field_name_pruned"]: colors_11[i]
        for i, (_, row) in enumerate(field_counts.iterrows())
    }
    sns.barplot(
        data=field_counts,
        y="document_field_name_pruned",
        x="count",
        hue="document_field_name_pruned",
        palette=palette_b2,
        legend=False,
        ax=ax_b2,
        orient="h",
    )
    ax_b2.set_xlabel("Pairs")
    ax_b2.set_ylabel("")
    ax_b2.set_title("c", fontweight="bold", loc="left", fontsize=10)
    ax_b2.tick_params(axis="y", labelsize=7)

    # B3: Pairs by dataset source
    source_counts = (
        pair_metadata.join(
            dataset_sources.select(
                pl.col("id").alias("dataset_source_id"),
                pl.col("name").alias("source_name"),
            ),
            on="dataset_source_id",
            how="left",
        )
        .group_by("source_name")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .to_pandas()
    )
    source_names_ordered = source_counts["source_name"].tolist()
    palette_d = {
        name: colors_8[i % len(colors_8)] for i, name in enumerate(source_names_ordered)
    }
    sns.barplot(
        data=source_counts,
        y="source_name",
        x="count",
        hue="source_name",
        palette=palette_d,
        legend=False,
        ax=ax_b3,
        orient="h",
    )
    ax_b3.set_xlabel("Pairs")
    ax_b3.set_ylabel("")
    ax_b3.set_title("d", fontweight="bold", loc="left", fontsize=10)
    ax_b3.tick_params(axis="y", labelsize=7)

    # B4: Coverage of three views
    pm = software.pair_metadata
    total = len(pm)
    coverage_data = {
        "view": ["Mentions", "Imports", "Dependencies", "All Three"],
        "proportion": [
            pm["has_software_mentions"].sum() / total,
            pm["has_imports"].sum() / total,
            pm["has_dependencies"].sum() / total,
            (pm["has_imports"] & pm["has_dependencies"] & pm["has_software_mentions"]).sum()
            / total,
        ],
        "n": [
            pm["has_software_mentions"].sum(),
            pm["has_imports"].sum(),
            pm["has_dependencies"].sum(),
            (pm["has_imports"] & pm["has_dependencies"] & pm["has_software_mentions"]).sum(),
        ],
    }
    import pandas as pd

    cov_df = pd.DataFrame(coverage_data)
    bar_colors = [colors_8[0], colors_8[1], colors_8[2], colors_8[3]]
    bars = ax_b4.bar(cov_df["view"], cov_df["proportion"], color=bar_colors)
    for bar, n_val in zip(bars, cov_df["n"], strict=False):
        ax_b4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={n_val:,}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax_b4.set_xlabel("Software Usage View")
    ax_b4.set_ylabel("Proportion of Pairs")
    ax_b4.set_ylim(0, 1.0)
    ax_b4.set_title("e", fontweight="bold", loc="left", fontsize=10)
    ax_b4.tick_params(axis="x", labelsize=7)


###############################################################################
# Subfigure C: Three views


def _find_best_three_views_pair(
    software: SoftwareUsageTables,
) -> tuple[dict, pl.DataFrame, pl.DataFrame, pl.DataFrame] | None:
    pm = software.pair_metadata

    # Filter out extreme outliers (>97th percentile for any view count)
    imports_per_pair = software.repository_imports.group_by("document_repository_link_id").agg(
        pl.len().alias("imports_count")
    )
    deps_per_pair = software.repository_dependencies.group_by(
        "document_repository_link_id"
    ).agg(pl.len().alias("deps_count"))
    mentions_per_pair = software.document_software_mentions.group_by(
        "document_repository_link_id"
    ).agg(pl.len().alias("mentions_count"))

    imports_p97 = imports_per_pair.get_column("imports_count").quantile(0.85)
    deps_p97 = deps_per_pair.get_column("deps_count").quantile(0.85)
    mentions_p97 = mentions_per_pair.get_column("mentions_count").quantile(0.85)

    extreme_ids = (
        set(
            imports_per_pair.filter(pl.col("imports_count") > imports_p97)
            .get_column("document_repository_link_id")
            .to_list()
        )
        | set(
            deps_per_pair.filter(pl.col("deps_count") > deps_p97)
            .get_column("document_repository_link_id")
            .to_list()
        )
        | set(
            mentions_per_pair.filter(pl.col("mentions_count") > mentions_p97)
            .get_column("document_repository_link_id")
            .to_list()
        )
    )

    complete_cases = pm.filter(
        pl.col("has_imports")
        & pl.col("has_dependencies")
        & pl.col("has_software_mentions")
        & ~pl.col("document_repository_link_id").is_in(extreme_ids)
    )

    if len(complete_cases) == 0:
        return None

    # Prefer Python or R repositories
    preferred = complete_cases.filter(
        pl.col("repository_primary_language").is_in(["Python", "R"])
    )
    pool = preferred if len(preferred) > 0 else complete_cases

    # Score: n_mentions*2 + n_imports + n_deps + 10 if any mention_context non-null
    pool = pool.join(imports_per_pair, on="document_repository_link_id", how="left")
    pool = pool.join(deps_per_pair, on="document_repository_link_id", how="left")
    pool = pool.join(mentions_per_pair, on="document_repository_link_id", how="left")

    has_context = (
        software.document_software_mentions.filter(pl.col("mention_context").is_not_null())
        .select("document_repository_link_id")
        .unique()
        .with_columns(pl.lit(10).alias("context_bonus"))
    )
    pool = pool.join(has_context, on="document_repository_link_id", how="left").fill_null(0)

    # Bonus for pairs with many unique software names across mentions
    unique_mention_names = software.document_software_mentions.group_by(
        "document_repository_link_id"
    ).agg(pl.col("software_name").n_unique().alias("unique_mention_names"))
    pool = pool.join(
        unique_mention_names, on="document_repository_link_id", how="left"
    ).fill_null(0)

    pool = pool.with_columns(
        (
            pl.col("mentions_count") * 2
            + pl.col("imports_count")
            + pl.col("deps_count")
            + pl.col("context_bonus")
            + pl.col("unique_mention_names")
        ).alias("score")
    )

    best_row = pool.sort("score", descending=True).row(0, named=True)
    link_id = best_row["document_repository_link_id"]
    doc_id = best_row["document_id"]

    best_imports = software.repository_imports.filter(
        pl.col("document_repository_link_id") == link_id
    ).unique(subset=["software_name"], keep="first")
    all_deps = (
        software.repository_dependencies.filter(
            pl.col("document_repository_link_id") == link_id
        )
        .sort("version_spec", nulls_last=True)
        .unique(subset=["software_name"], keep="first")
    )
    py_r_deps = all_deps.filter(pl.col("ecosystem").str.to_lowercase().is_in(["pypi", "cran"]))
    best_deps = py_r_deps if len(py_r_deps) > 0 else all_deps
    best_mentions = software.document_software_mentions.filter(
        pl.col("document_id") == doc_id
    ).unique(subset=["software_name"], keep="first")

    return best_row, best_imports, best_deps, best_mentions


def _render_highlighted_line(
    ax: mpl.axes.Axes,
    x_start: float,
    y: float,
    context: str,
    name: str,
    base_color: str,
    text_color: str,
    fontsize: float,
) -> None:
    lower_ctx = context.lower()
    lower_name = name.lower()
    idx = lower_ctx.find(lower_name)
    char_w = 0.011  # axes-fraction width per character at typical panel width
    x = x_start
    if idx == -1:
        ax.text(
            x, y, context, transform=ax.transAxes, fontsize=fontsize, color=text_color, va="top"
        )
        return
    before = context[:idx]
    match = context[idx : idx + len(name)]
    after = context[idx + len(name) :]
    # Cap before/after so the name stays near the left edge and the line fits the panel
    max_before = 15
    if len(before) > max_before:
        before = "..." + before[-max_before:]
    max_after = 50
    if len(after) > max_after:
        after = after[:max_after] + "..."
    if before:
        ax.text(
            x, y, before, transform=ax.transAxes, fontsize=fontsize, color=text_color, va="top"
        )
        x += len(before) * char_w
    ax.text(
        x,
        y,
        match,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=base_color,
        va="top",
        fontweight="bold",
    )
    x += (len(match) * char_w) + 0.04  # small gap after highlighted name
    if after:
        ax.text(
            x, y, after, transform=ax.transAxes, fontsize=fontsize, color=text_color, va="top"
        )


def _draw_mentions_panel(
    ax: mpl.axes.Axes,
    title_text: str,
    doi: str,
    best_mentions: pl.DataFrame,
    color: str,
    max_items: int = 8,
) -> None:
    ax.set_facecolor(PANEL_BG_LIGHT)
    ax.set_axis_off()
    ax.set_title("f", fontweight="bold", loc="left", fontsize=10)

    y = 0.97
    ax.text(
        0.05,
        y,
        "Mentions",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=TEXT_DARK,
        va="top",
    )
    y -= 0.06
    ax.text(
        0.05,
        y,
        "software named in article text",
        transform=ax.transAxes,
        fontsize=7,
        color="#666666",
        va="top",
        style="italic",
    )
    y -= 0.05
    # title_display = (title_text[:60] + "...") if len(title_text) > 60 else title_text
    # ax.text(
    #     0.05,
    #     y,
    #     title_display,
    #     transform=ax.transAxes,
    #     fontsize=6.5,
    #     color=TEXT_DARK,
    #     va="top",
    # )
    # y -= 0.05
    ax.text(
        0.05, y, f"doi:{doi}", transform=ax.transAxes, fontsize=6, color="#666666", va="top"
    )
    y -= 0.06

    # Prioritize rows where the software name appears in the context
    mentions_with_context = best_mentions.filter(
        pl.col("mention_context").is_not_null()
        & pl.col("mention_context")
        .str.to_lowercase()
        .str.contains(pl.col("software_name_normalized").str.to_lowercase())
    )
    mentions_fallback = best_mentions.filter(
        ~(
            pl.col("mention_context").is_not_null()
            & pl.col("mention_context")
            .str.to_lowercase()
            .str.contains(pl.col("software_name_normalized").str.to_lowercase())
        )
    )
    mentions_sorted = pl.concat([mentions_with_context, mentions_fallback])
    total = len(mentions_sorted)
    shown = 0

    for row in mentions_sorted.iter_rows(named=True):
        if shown >= max_items or y < 0.05:
            break
        context = row.get("mention_context") or ""
        name = row["software_name"]
        if not context:
            continue
        lower_ctx = context.lower()
        lower_name = name.lower()
        name_idx = lower_ctx.find(lower_name)
        if name_idx == -1:
            continue
        # Extract a window of ~90 chars around the match
        win_start = max(0, name_idx - 42)
        win_end = min(len(context), name_idx + len(name) + 42)
        prefix = "..." if win_start > 0 else '"'
        suffix = "..." if win_end < len(context) else '"'
        context = prefix + context[win_start:win_end] + suffix

        _render_highlighted_line(ax, 0.02, y, context, name, color, TEXT_DARK, 6.5)
        y -= 0.09
        shown += 1

    if total > max_items:
        ax.text(
            0.05,
            max(y, 0.03),
            f"(+{total - max_items} more)",
            transform=ax.transAxes,
            fontsize=6,
            color="#999999",
            va="top",
            style="italic",
        )


def _draw_imports_panel(
    ax: mpl.axes.Axes,
    repo_url: str,
    best_imports: pl.DataFrame,
    color: str,
    max_items: int = 8,
) -> None:
    ax.set_facecolor(PANEL_BG_DARK)
    ax.set_axis_off()
    ax.set_title("g", fontweight="bold", loc="left", fontsize=10)

    y = 0.97
    ax.text(
        0.05,
        y,
        "Imports",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=TEXT_DARK,
        va="top",
    )
    y -= 0.06
    ax.text(
        0.05,
        y,
        "libraries called directly in code",
        transform=ax.transAxes,
        fontsize=7,
        color=TEXT_LIGHT,
        va="top",
        style="italic",
    )
    y -= 0.05
    ax.text(
        0.05,
        y,
        repo_url,
        transform=ax.transAxes,
        fontsize=6.5,
        color=TEXT_LIGHT,
        va="top",
    )
    y -= 0.08

    py_imports = best_imports.filter(pl.col("ecosystem") == "py")
    r_imports = best_imports.filter(pl.col("ecosystem") == "r")
    total = len(best_imports)
    shown = 0

    def _render_import_row(
        row: dict, ax: mpl.axes.Axes, y: float, color: str, is_r: bool
    ) -> float:
        name = row["software_name"]
        if is_r:
            prefix, suffix = "library(", ")"
        else:
            prefix, suffix = "import ", ""
        offset = len(prefix) * 0.018
        ax.text(
            0.05,
            y,
            prefix,
            transform=ax.transAxes,
            fontsize=7.5,
            color=TEXT_LIGHT,
            va="top",
            fontfamily="monospace",
        )
        ax.text(
            0.05 + offset,
            y,
            name,
            transform=ax.transAxes,
            fontsize=7.5,
            color=color,
            va="top",
            fontfamily="monospace",
            fontweight="bold",
        )
        if suffix:
            ax.text(
                0.05 + offset + len(name) * 0.018,
                y,
                suffix,
                transform=ax.transAxes,
                fontsize=7.5,
                color=TEXT_LIGHT,
                va="top",
                fontfamily="monospace",
            )
        file_paths_raw = row.get("file_paths") or ""
        if file_paths_raw:
            first_path = file_paths_raw.split(";")[0].strip()
            filename = Path(first_path).name
            ax.text(
                0.4,
                y,
                filename,
                transform=ax.transAxes,
                fontsize=5.5,
                color="#888888",
                va="top",
                style="italic",
                fontfamily="monospace",
            )
        return y - 0.07

    for section_df, section_label, is_r in (
        (py_imports, "# Python", False),
        (r_imports, "# R", True),
    ):
        if len(section_df) == 0 or shown >= max_items or y < 0.05:
            continue
        ax.text(
            0.05,
            y,
            section_label,
            transform=ax.transAxes,
            fontsize=6.5,
            color="#888888",
            va="top",
            style="italic",
            fontfamily="monospace",
        )
        y -= 0.07
        for row in section_df.iter_rows(named=True):
            if shown >= max_items or y < 0.05:
                break
            y = _render_import_row(row, ax, y, color, is_r)
            shown += 1

    if total > max_items:
        ax.text(
            0.05,
            max(y, 0.03),
            f"(+{total - max_items} more)",
            transform=ax.transAxes,
            fontsize=6,
            color="#888888",
            va="top",
            style="italic",
        )


def _draw_deps_panel(
    ax: mpl.axes.Axes,
    repo_url: str,
    best_deps: pl.DataFrame,
    color: str,
    max_items: int = 8,
) -> None:
    ax.set_facecolor(PANEL_BG_MID)
    ax.set_axis_off()
    ax.set_title("h", fontweight="bold", loc="left", fontsize=10)

    y = 0.97
    ax.text(
        0.05,
        y,
        "Dependencies",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=TEXT_DARK,
        va="top",
    )
    y -= 0.06
    ax.text(
        0.05,
        y,
        "packages declared in manifests",
        transform=ax.transAxes,
        fontsize=7,
        color=TEXT_LIGHT,
        va="top",
        style="italic",
    )
    y -= 0.05
    ax.text(
        0.05,
        y,
        repo_url,
        transform=ax.transAxes,
        fontsize=6.5,
        color=TEXT_LIGHT,
        va="top",
    )
    y -= 0.08

    pypi_deps = best_deps.filter(pl.col("ecosystem").str.to_lowercase() == "pypi")
    cran_deps = best_deps.filter(pl.col("ecosystem").str.to_lowercase() == "cran")
    other_deps = best_deps.filter(
        ~pl.col("ecosystem").str.to_lowercase().is_in(["pypi", "cran"])
    )
    total = len(best_deps)
    shown = 0

    def _render_dep_row(row: dict, ax: mpl.axes.Axes, y: float, color: str) -> float:
        name = row["software_name"]
        version = row.get("version_spec") or ""
        ax.text(
            0.05,
            y,
            name,
            transform=ax.transAxes,
            fontsize=7.5,
            color=color,
            va="top",
            fontfamily="monospace",
            fontweight="bold",
        )
        if version:
            print(version)
            ax.text(
                0.05 + len(name) * 0.018,
                y,
                f"  {version}",
                transform=ax.transAxes,
                fontsize=7.5,
                color=TEXT_LIGHT,
                va="top",
                fontfamily="monospace",
            )
        return y - 0.07

    for section_df, section_label in (
        (pypi_deps, "# Python / PyPI"),
        (cran_deps, "# R / CRAN"),
        (other_deps, "# Other"),
    ):
        if len(section_df) == 0 or shown >= max_items or y < 0.05:
            continue
        ax.text(
            0.05,
            y,
            section_label,
            transform=ax.transAxes,
            fontsize=6.5,
            color="#888888",
            va="top",
            style="italic",
            fontfamily="monospace",
        )
        y -= 0.07
        for row in section_df.iter_rows(named=True):
            if shown >= max_items or y < 0.05:
                break
            y = _render_dep_row(row, ax, y, color)
            shown += 1

    if total > max_items:
        ax.text(
            0.05,
            max(y, 0.03),
            f"(+{total - max_items} more)",
            transform=ax.transAxes,
            fontsize=6,
            color="#888888",
            va="top",
            style="italic",
        )


def _draw_three_views(
    ax_c1: mpl.axes.Axes,
    ax_c2: mpl.axes.Axes,
    ax_c3: mpl.axes.Axes,
    best_row: dict,
    best_imports: pl.DataFrame,
    best_deps: pl.DataFrame,
    best_mentions: pl.DataFrame,
    colors: list[str],
    max_items: int = 8,
) -> None:
    title_text = best_row["document_title"]
    if len(title_text) > 70:
        title_text = title_text[:70] + "..."
    repo_url = f"github.com/{best_row['repository_owner']}/{best_row['repository_name']}"
    doi = best_row["document_doi"]

    _draw_mentions_panel(ax_c1, title_text, doi, best_mentions, colors[0], max_items=max_items)
    _draw_imports_panel(ax_c2, repo_url, best_imports, colors[1], max_items=max_items)
    _draw_deps_panel(ax_c3, repo_url, best_deps, colors[2], max_items=max_items)


###############################################################################
# Figure assembly


def _build_figure(
    layout: Literal["horizontal", "vertical"],
    pair_metadata: pl.DataFrame,
    dataset_sources: pl.DataFrame,
    software: SoftwareUsageTables,
    network_graph: rx.PyGraph,
    three_views_result: tuple[dict, pl.DataFrame, pl.DataFrame, pl.DataFrame] | None,
    colors_8: list[str],
    colors_11: list[str],
) -> mpl.figure.Figure:
    if layout == "horizontal":
        fig = plt.figure(figsize=(22, 9))
        # 3 main columns: network | B(2x2) | C(3 rows)
        outer_gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[1, 2, 1],
            hspace=0.05,
            wspace=0.35,
            left=0.05,
            right=0.97,
            top=0.95,
            bottom=0.08,
        )
        ax_a = fig.add_subplot(outer_gs[0])

        b_gs = outer_gs[1].subgridspec(2, 2, hspace=0.45, wspace=0.6)
        ax_b1 = fig.add_subplot(b_gs[0, 0])
        ax_b2 = fig.add_subplot(b_gs[0, 1])
        ax_b3 = fig.add_subplot(b_gs[1, 0])
        ax_b4 = fig.add_subplot(b_gs[1, 1])

        c_gs = outer_gs[2].subgridspec(3, 1, hspace=0.1)
        ax_c1 = fig.add_subplot(c_gs[0])
        ax_c2 = fig.add_subplot(c_gs[1])
        ax_c3 = fig.add_subplot(c_gs[2])
        three_views_max_items = 8

    else:  # vertical
        fig = plt.figure(figsize=(12, 22))
        outer_gs = fig.add_gridspec(
            3,
            1,
            height_ratios=[1.5, 1.3, 1],
            hspace=0.35,
            left=0.08,
            right=0.97,
            top=0.97,
            bottom=0.04,
        )
        a_row_gs = outer_gs[0].subgridspec(1, 1)
        ax_a = fig.add_subplot(a_row_gs[0])

        b_gs = outer_gs[1].subgridspec(2, 2, hspace=0.55, wspace=0.5)
        ax_b1 = fig.add_subplot(b_gs[0, 0])
        ax_b2 = fig.add_subplot(b_gs[0, 1])
        ax_b3 = fig.add_subplot(b_gs[1, 0])
        ax_b4 = fig.add_subplot(b_gs[1, 1])

        c_gs = outer_gs[2].subgridspec(1, 3, wspace=0.1)
        ax_c1 = fig.add_subplot(c_gs[0])
        ax_c2 = fig.add_subplot(c_gs[1])
        ax_c3 = fig.add_subplot(c_gs[2])
        three_views_max_items = 6

    _draw_network(ax_a, network_graph, colors_8)
    _draw_descriptive_overview(
        ax_b1,
        ax_b2,
        ax_b3,
        ax_b4,
        pair_metadata,
        dataset_sources,
        software,
        colors_8,
        colors_11,
    )

    if three_views_result is not None:
        best_row, best_imports, best_deps, best_mentions = three_views_result
        _draw_three_views(
            ax_c1,
            ax_c2,
            ax_c3,
            best_row,
            best_imports,
            best_deps,
            best_mentions,
            colors_8,
            max_items=three_views_max_items,
        )
    else:
        for ax in [ax_c1, ax_c2, ax_c3]:
            ax.text(
                0.5,
                0.5,
                "No complete-case pair found",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_axis_off()

    return fig


###############################################################################
# Main


@app.command()
def main() -> None:
    evaplot.set_style()
    colors_8 = evaplot.set_cat_palette(8)
    colors_11 = evaplot.set_cat_palette(11)

    print("Loading core tables...")
    pair_metadata = _load_pair_metadata()
    print(f"  pair_metadata: {len(pair_metadata):,} rows")

    dataset_sources = load_table("dataset_source")
    document_contributors = load_table("document_contributor")
    researchers = load_table("researcher")
    repository_contributors = load_table("repository_contributor")
    developer_accounts = load_table("developer_account")
    researcher_dev_links = load_table("researcher_developer_account_link")

    print("Building software usage tables...")
    software = _build_software_usage_tables(pair_metadata)
    # Swap in pair_metadata with has_* flags
    pair_metadata = software.pair_metadata

    print("Building network graph...")
    network_graph = _build_network_graph(
        pair_metadata,
        document_contributors,
        researchers,
        repository_contributors,
        developer_accounts,
        researcher_dev_links,
        dataset_sources,
    )
    print(f"  network nodes: {len(network_graph.nodes())}, edges: {len(network_graph.edges())}")

    print("Finding best three-views pair...")
    three_views_result = _find_best_three_views_pair(software)
    if three_views_result:
        best_row = three_views_result[0]
        print(
            f"  selected pair: doi={best_row['document_doi']}, repo={best_row['repository_owner']}/{best_row['repository_name']}"
        )

    for layout in ("horizontal", "vertical"):
        layout_typed: Literal["horizontal", "vertical"] = layout  # type: ignore[assignment]
        print(f"Rendering {layout_typed} layout...")
        fig = _build_figure(
            layout_typed,
            pair_metadata,
            dataset_sources,
            software,
            network_graph,
            three_views_result,
            colors_8,
            colors_11,
        )
        for ext in ("pdf", "png"):
            out_path = THIS_DIR / f"star-figure-{layout}.{ext}"
            dpi = 300 if ext == "png" else None
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"  saved → {out_path}")
        plt.close(fig)

    print("Done.")


###############################################################################

if __name__ == "__main__":
    app()
