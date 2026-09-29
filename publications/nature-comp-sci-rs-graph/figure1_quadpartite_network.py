#!/usr/bin/env python3

"""Figure 1, panel 1: the snowball-sampled quad-partite network of articles, repositories,
researchers, and developer accounts.
"""

from __future__ import annotations

import random
from collections import Counter, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import evaplot
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rustworkx as rx
import utils as u
from matplotlib.collections import LineCollection

###############################################################################
# Layouts


def _global_spring_layout(
    graph: rx.PyGraph, seed: int = 42
) -> tuple[dict[int, tuple[float, float]], float, float]:
    """
    Lay out the whole sampled graph with one global spring layout. The snowball sample puts
    ~99% of pairs in one giant component, which a conventional force layout renders directly.
    Used only to warm-start ForceAtlas2 below -- not rendered on its own.
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
    FA2 paper's forces (degree+1 node mass, linear edge attraction, mass-scaled repulsion,
    gravity) and its adaptive global-speed/swing update. Repulsion uses a fresh random node
    sample per iteration (scaled by n/sample) when the graph is large -- the standard
    sampling approximation, keeping each iteration O(n * sample) instead of O(n^2).
    Warm-started from the spring positions.
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


###############################################################################
# Graph construction


def _add_provenance_colored_pair_edges(
    graph: rx.PyGraph,
    resolve_node: Callable[[str, int, bool], int | None],
    sampled_pairs: pl.DataFrame,
    pair_is_mined: dict[int, bool],
) -> None:
    """Add article-repository edges, split into a seed/mined edge type per pair (from
    `pair_is_mined`) rather than one fixed type for the whole frame -- kept out of
    `_build_quadpartite_graph`'s generic edge loop for that reason.
    """
    for row in sampled_pairs.iter_rows(named=True):
        # create=True on both sides, so these are never actually None.
        src_i = resolve_node("article", row["document_id"], True)
        tgt_i = resolve_node("repository", row["repository_id"], True)
        if src_i is None or tgt_i is None or graph.has_edge(src_i, tgt_i):
            continue
        is_mined = pair_is_mined.get(row["document_repository_link_id"], False)
        etype = "article_repository_link_mined" if is_mined else "article_repository_link_seed"
        graph.add_edge(src_i, tgt_i, {"type": etype})


def _build_quadpartite_graph(
    sampled_pairs: pl.DataFrame,
    doc_authors: pl.DataFrame,
    repo_devs: pl.DataFrame,
    identity_edges: pl.DataFrame,
    pair_is_mined: dict[int, bool],
    doc_fields: dict[int, str] | None = None,
) -> rx.PyGraph:
    """Build the quad-partite `rx.PyGraph` from the already-sampled/capped/filtered frames.
    `doc_fields` attaches the field color-encoding attribute to article nodes (unused for
    drawing today, kept for the strata/mix reporting below). `pair_is_mined` splits each
    article-repository edge into a seed/mined type so the draw step can color by provenance.
    """
    graph: rx.PyGraph = rx.PyGraph()
    node_idx: dict[tuple[str, int], int] = {}
    doc_fields = doc_fields or {}

    def _resolve_node(node_type: str, node_id: int, create: bool) -> int | None:
        key = (node_type, node_id)
        if key in node_idx:
            return node_idx[key]
        if not create:
            return None
        payload: dict[str, str | int] = {"type": node_type, "id": node_id}
        if node_type == "article":
            payload["group"] = doc_fields.get(node_id, "Other")
        node_idx[key] = graph.add_node(payload)
        return node_idx[key]

    _add_provenance_colored_pair_edges(graph, _resolve_node, sampled_pairs, pair_is_mined)

    # (frame, (source type, source column, create source), (target ditto), edge type);
    # existing-only endpoints (create=False) skip rows whose entity was never sampled.
    edge_specs: list[tuple[pl.DataFrame, tuple[str, str, bool], tuple[str, str, bool], str]] = [
        (
            doc_authors,
            ("article", "document_id", False),
            ("researcher", "researcher_id", True),
            "authored_by",
        ),
        (
            repo_devs,
            ("repository", "repository_id", False),
            ("developer", "developer_account_id", True),
            "contributed_to",
        ),
        (
            identity_edges,
            ("researcher", "researcher_id", False),
            ("developer", "developer_account_id", False),
            "identity",
        ),
    ]
    for frame, (src_type, src_col, create_src), (
        tgt_type,
        tgt_col,
        create_tgt,
    ), etype in edge_specs:
        for row in frame.iter_rows(named=True):
            src_i = _resolve_node(src_type, row[src_col], create_src)
            if src_i is None:
                continue
            tgt_i = _resolve_node(tgt_type, row[tgt_col], create_tgt)
            if tgt_i is None:
                continue
            if not graph.has_edge(src_i, tgt_i):
                graph.add_edge(src_i, tgt_i, {"type": etype})

    return graph


###############################################################################
# Snowball sampling


@dataclass
class _SnowballLookups:
    """Adjacency lookups over the filtered pairs, keyed for the snowball walk."""

    pair_doc: dict[int, int]
    pair_repo: dict[int, int]
    doc_authors_ordered: dict[int, list[int]]
    repo_devs_ordered: dict[int, list[int]]
    researcher_pairs: dict[int, list[int]]
    developer_pairs: dict[int, list[int]]
    identity_r2d: dict[int, list[int]]
    identity_d2r: dict[int, list[int]]
    identity_links: list[tuple[int, int]]


def _ordered_membership(
    frame: pl.DataFrame, key_col: str, value_col: str, allowed_keys: dict[int, list[int]]
) -> dict[int, list[int]]:
    """Group `value_col` values per `key_col`, keeping frame row order, for allowed keys."""
    membership: dict[int, list[int]] = {}
    for key, value in frame.select(key_col, value_col).iter_rows():
        if key in allowed_keys:
            membership.setdefault(key, []).append(value)
    return membership


def _entity_pair_reach(
    membership: dict[int, list[int]], pairs_by_key: dict[int, list[int]]
) -> dict[int, list[int]]:
    """Invert membership into pairs reachable per entity."""
    reach: dict[int, list[int]] = {}
    for key, entities in membership.items():
        for entity_id in entities:
            reach.setdefault(entity_id, []).extend(pairs_by_key[key])
    return reach


def _build_snowball_lookups(
    pair_frame: pl.DataFrame,
    document_contributors: pl.DataFrame,
    repository_contributors: pl.DataFrame,
    rdal: pl.DataFrame,
) -> _SnowballLookups:
    """Build every adjacency lookup the snowball walk needs from the raw frames."""
    pair_doc: dict[int, int] = {}
    pair_repo: dict[int, int] = {}
    doc_pairs: dict[int, list[int]] = {}
    repo_pairs: dict[int, list[int]] = {}
    for pid, doc_id, repo_id in pair_frame.iter_rows():
        pair_doc[pid] = doc_id
        pair_repo[pid] = repo_id
        doc_pairs.setdefault(doc_id, []).append(pid)
        repo_pairs.setdefault(repo_id, []).append(pid)

    doc_authors_ordered = _ordered_membership(
        document_contributors, "document_id", "researcher_id", doc_pairs
    )
    repo_devs_ordered = _ordered_membership(
        repository_contributors, "repository_id", "developer_account_id", repo_pairs
    )
    researcher_pairs = _entity_pair_reach(doc_authors_ordered, doc_pairs)
    developer_pairs = _entity_pair_reach(repo_devs_ordered, repo_pairs)

    identity_r2d: dict[int, list[int]] = {}
    identity_d2r: dict[int, list[int]] = {}
    identity_links: list[tuple[int, int]] = []
    for researcher_id, dev_id in (
        rdal.select("researcher_id", "developer_account_id")
        .unique(maintain_order=True)
        .iter_rows()
    ):
        identity_r2d.setdefault(researcher_id, []).append(dev_id)
        identity_d2r.setdefault(dev_id, []).append(researcher_id)
        identity_links.append((researcher_id, dev_id))

    return _SnowballLookups(
        pair_doc=pair_doc,
        pair_repo=pair_repo,
        doc_authors_ordered=doc_authors_ordered,
        repo_devs_ordered=repo_devs_ordered,
        researcher_pairs=researcher_pairs,
        developer_pairs=developer_pairs,
        identity_r2d=identity_r2d,
        identity_d2r=identity_d2r,
        identity_links=identity_links,
    )


def _identity_link_field(
    lookups: _SnowballLookups, r: int, d: int, pair_field: dict[int, str]
) -> str | None:
    """Modal article field over the pairs reachable through either side of an identity link."""
    fields = Counter(
        pair_field[p]
        for p in lookups.researcher_pairs.get(r, []) + lookups.developer_pairs.get(d, [])
    )
    return fields.most_common(1)[0][0] if fields else None


def _select_anchors(
    lookups: _SnowballLookups,
    pair_field: dict[int, str],
    strata: list[str],
    n_top_hub_anchors_per_field: int,
    n_random_hub_anchors_per_field: int,
    min_anchor_pair_degree: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], int]:
    """Field-stratified anchors: per stratum, the top identity links by pair-degree plus a
    seeded random draw of the qualifying rest; the remaining qualifying links (shuffled) form
    the reserve. Returns (anchor queue, number of planned anchors).
    """
    degree_by_link = [
        (
            len(lookups.researcher_pairs.get(r, [])) + len(lookups.developer_pairs.get(d, [])),
            r,
            d,
        )
        for r, d in lookups.identity_links
    ]
    degree_by_link.sort(key=lambda t: (-t[0], t[1], t[2]))
    by_stratum: dict[str, list[tuple[int, int]]] = {s: [] for s in strata}
    for deg, r, d in degree_by_link:
        if deg < min_anchor_pair_degree:
            continue
        field = _identity_link_field(lookups, r, d, pair_field)
        if field in by_stratum:
            by_stratum[field].append((r, d))

    top_anchors: list[tuple[int, int]] = []
    random_anchors: list[tuple[int, int]] = []
    reserve: list[tuple[int, int]] = []
    for stratum in strata:
        links = by_stratum[stratum]
        top_anchors += links[:n_top_hub_anchors_per_field]
        rest = links[n_top_hub_anchors_per_field:]
        rng.shuffle(rest)
        random_anchors += rest[:n_random_hub_anchors_per_field]
        reserve += rest[n_random_hub_anchors_per_field:]
        print(
            f"  Anchor stratum {stratum}: {len(links):,} identities with pair-degree >= "
            f"{min_anchor_pair_degree}; {len(links[:n_top_hub_anchors_per_field])} top hubs + "
            f"{len(rest[:n_random_hub_anchors_per_field])} random."
        )
    rng.shuffle(random_anchors)
    rng.shuffle(reserve)
    planned = top_anchors + random_anchors
    print(
        f"Anchors: {len(planned)} planned across {len(strata)} field strata "
        f"({len(top_anchors)} top hubs + {len(random_anchors)} random); "
        f"{len(reserve):,} held as reserve."
    )
    return planned + reserve, len(planned)


def _capped_entities(
    ordered: list[int], identity_map: dict[int, list[int]], cap: int
) -> list[int]:
    """First `cap` entities plus any identity-linked ones, so the cap never severs
    identity edges.
    """
    kept = list(ordered[:cap])
    kept += [e for e in ordered[cap:] if e in identity_map]
    return list(dict.fromkeys(kept))


def _pair_expansions(pid: int, lookups: _SnowballLookups, cap: int) -> Iterator[list[int]]:
    """Yield the candidate-pair lists one popped pair contributes: per capped author, that
    author's pairs then each identity-linked developer's pairs; then likewise per capped
    developer.
    """
    for researcher_id in _capped_entities(
        lookups.doc_authors_ordered.get(lookups.pair_doc[pid], []), lookups.identity_r2d, cap
    ):
        yield lookups.researcher_pairs.get(researcher_id, [])
        for dev_id in lookups.identity_r2d.get(researcher_id, []):
            yield lookups.developer_pairs.get(dev_id, [])
    for dev_id in _capped_entities(
        lookups.repo_devs_ordered.get(lookups.pair_repo[pid], []), lookups.identity_d2r, cap
    ):
        yield lookups.developer_pairs.get(dev_id, [])
        for researcher_id in lookups.identity_d2r.get(dev_id, []):
            yield lookups.researcher_pairs.get(researcher_id, [])


def _admit_pairs(
    candidate_pairs: list[int],
    sampled: set[int],
    frontier: deque[int],
    budget: int,
    max_pairs_per_entity: int,
    rng: random.Random,
    pair_field: dict[int, str],
    admitted_per_field: Counter[str],
    max_pairs_per_field: int,
) -> None:
    """Admit up to `max_pairs_per_entity` unseen candidates (seeded subsample) into the
    sample and frontier, stopping at the budget; candidates whose article field already holds
    `max_pairs_per_field` sampled pairs are skipped so no field can dominate the sample.
    """
    new = [p for p in dict.fromkeys(candidate_pairs) if p not in sampled]
    if len(new) > max_pairs_per_entity:
        new = rng.sample(new, max_pairs_per_entity)
    for p in new:
        if len(sampled) >= budget:
            return
        if admitted_per_field[pair_field[p]] >= max_pairs_per_field:
            continue
        sampled.add(p)
        frontier.append(p)
        admitted_per_field[pair_field[p]] += 1


def _snowball_sample(
    lookups: _SnowballLookups,
    anchor_queue: list[tuple[int, int]],
    n_planned_anchors: int,
    budget: int,
    max_pairs_per_entity: int,
    max_contributors_per_side: int,
    rng: random.Random,
    pair_field: dict[int, str],
    max_pairs_per_field: int,
) -> tuple[set[int], int]:
    """BFS over pairs mediated by entities; returns the sampled pair ids and the number of
    anchors used.
    """
    sampled: set[int] = set()
    frontier: deque[int] = deque()
    admitted_per_field: Counter[str] = Counter()

    def _add_pairs(candidate_pairs: list[int]) -> None:
        _admit_pairs(
            candidate_pairs,
            sampled,
            frontier,
            budget,
            max_pairs_per_entity,
            rng,
            pair_field,
            admitted_per_field,
            max_pairs_per_field,
        )

    # Seed all planned anchors' pairs up front so the sample spans every anchor neighborhood
    # rather than exhausting the budget on the first hub's BFS; the remaining tail is a
    # reserve drawn only if the frontier empties below budget.
    n_anchors_used = 0
    for anchor_r, anchor_d in anchor_queue[:n_planned_anchors]:
        if len(sampled) >= budget:
            break
        n_anchors_used += 1
        _add_pairs(lookups.researcher_pairs.get(anchor_r, []))
        _add_pairs(lookups.developer_pairs.get(anchor_d, []))
    anchor_queue = anchor_queue[n_anchors_used:]

    while len(sampled) < budget:
        if not frontier:
            if not anchor_queue:
                break
            anchor_r, anchor_d = anchor_queue.pop(0)
            n_anchors_used += 1
            _add_pairs(lookups.researcher_pairs.get(anchor_r, []))
            if len(sampled) >= budget:
                break
            _add_pairs(lookups.developer_pairs.get(anchor_d, []))
            continue
        pid = frontier.popleft()
        for candidate_pairs in _pair_expansions(pid, lookups, max_contributors_per_side):
            _add_pairs(candidate_pairs)
            if len(sampled) >= budget:
                break

    return sampled, n_anchors_used


def _sampled_entity_frames(
    sampled_pairs: pl.DataFrame, lookups: _SnowballLookups, cap: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Authorship and contribution frames for the sampled pairs, with the same
    cap-plus-always-include-identity rule the walk used.
    """
    author_rows = [
        (doc_id, researcher_id)
        for doc_id in sampled_pairs.get_column("document_id")
        .unique(maintain_order=True)
        .to_list()
        for researcher_id in _capped_entities(
            lookups.doc_authors_ordered.get(doc_id, []), lookups.identity_r2d, cap
        )
    ]
    dev_rows = [
        (repo_id, dev_id)
        for repo_id in sampled_pairs.get_column("repository_id")
        .unique(maintain_order=True)
        .to_list()
        for dev_id in _capped_entities(
            lookups.repo_devs_ordered.get(repo_id, []), lookups.identity_d2r, cap
        )
    ]
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
    return doc_authors, repo_devs


###############################################################################
# Drawing


def _edge_segments(
    graph: rx.PyGraph, positions: dict[int, tuple[float, float]], etype: str
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Position segments for every edge of one type."""
    return [
        (positions[src], positions[tgt])
        for edge_idx in graph.edge_indices()
        if graph.get_edge_data_by_index(edge_idx)["type"] == etype
        for src, tgt in [graph.get_edge_endpoints_by_index(edge_idx)]
    ]


def _node_positions_by_type(
    graph: rx.PyGraph, positions: dict[int, tuple[float, float]], ntype: str
) -> tuple[list[float], list[float]]:
    """X and y coordinate lists for every node of one type."""
    xs: list[float] = []
    ys: list[float] = []
    for node_i in graph.node_indices():
        if graph[node_i]["type"] == ntype:
            x, y = positions[node_i]
            xs.append(x)
            ys.append(y)
    return xs, ys


def _marker_handle(
    marker: str, color: str, label: str, hollow: bool = False, size: int = 7
) -> mlines.Line2D:
    """Legend handle for a node marker."""
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


def _draw_quadpartite_network(
    graph: rx.PyGraph,
    positions: dict[int, tuple[float, float]],
    canvas_width: float,
    canvas_height: float,
    caption: str,
    output_dir: Path,
    stem: str = "figure1_quadpartite_network",
    show_legend: bool = False,
) -> None:
    """Draw and save one Figure 1 render: article/researcher share one blue hue (article
    filled, researcher hollow, lighter tint); repository/developer share one vermillion hue
    the same way. Article-repository edges are coloured by provenance (seed vs. mined, from
    `link_processing_iteration`); every other edge is light grey, differentiated only by
    alpha/linewidth. No per-node labels. `show_legend=False` (the main output) omits the
    boxed legend -- panel B carries the encoding instead; `show_legend=True` draws a small
    reference legend for review. LineCollections and scatters rasterized so the PDF stays
    small and fast to open.
    """
    # Two hue families, dark (filled) + light (hollow) tint each: blue for
    # article/researcher, vermillion for repository/developer. Full separation from the Figs
    # 2-4 field palette isn't achievable -- see the figure's memory-doc discussion.
    node_type_colors: dict[str, str] = {
        "article": "#1B4F72",
        "researcher": "#7FB3D5",
        "repository": "#B7472A",
        "developer": "#F0B27A",
    }
    hollow_node_types = {"researcher", "developer"}
    node_type_markers: dict[str, str] = {
        "article": "o",
        "repository": "s",
        "researcher": "D",
        "developer": "^",
    }
    node_type_labels: dict[str, str] = {
        "article": "Article",
        "repository": "Repository",
        "researcher": "Researcher (author)",
        "developer": "Developer account",
    }

    # Edge provenance hues (ColorBrewer Dark2 purple/gold -- a blue-yellow pair, robust under
    # red-green colourblindness and distinct from the blue/vermillion node hues above and the
    # field/general/data-view palettes used elsewhere): purple for seed pairs, gold for mined.
    # Every other edge type stays uniform light grey, differentiated only by alpha/linewidth.
    edge_grey = "#999999"
    seed_color, mined_color = "#7570B3", "#E6AB02"
    edge_style = {
        "authored_by": {"color": edge_grey, "lw": 0.25, "ls": "-", "zorder": 1, "alpha": 0.06},
        "contributed_to": {
            "color": edge_grey,
            "lw": 0.25,
            "ls": "-",
            "zorder": 1,
            "alpha": 0.06,
        },
        "article_repository_link_seed": {
            "color": seed_color,
            "lw": 0.9,
            "ls": "-",
            "zorder": 2,
            "alpha": 0.38,
        },
        "article_repository_link_mined": {
            "color": mined_color,
            "lw": 0.9,
            "ls": "-",
            "zorder": 2,
            "alpha": 0.38,
        },
        "identity": {"color": edge_grey, "lw": 0.7, "ls": "-", "zorder": 3, "alpha": 0.15},
    }
    # Marker area bumped ~2.5x over this figure's original 3.5pt^2 calibration so the two hue
    # families read at panel size. Researcher/developer now draw ~0.7x that area -- people
    # nodes outnumber article/repository nodes heavily at this sample size, so a slight size
    # cut (on top of already being hollow) keeps them from dominating.
    marker_size = 8.75
    identity_marker_size = 8.75 * 0.7
    node_alpha = 0.5
    identity_alpha = 0.4
    marker_lw = 0.25

    ref_canvas_extent = 143.2  # canvas width measured at the 2,000-seed-pair calibration run
    canvas_extent = max(canvas_width, canvas_height)
    figsize_in = min(22.0, max(14.0, 14.0 * canvas_extent / ref_canvas_extent))

    fig, ax = plt.subplots(figsize=(figsize_in, figsize_in))
    ax.margins(0.02)  # crop the whitespace border down from matplotlib's default 5%

    for etype, style in edge_style.items():
        segments = _edge_segments(graph, positions, etype)
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

    # Article/repository filled; researcher/developer hollow (lighter tint of the same hue)
    # and slightly smaller.
    for ntype, color in node_type_colors.items():
        xs, ys = _node_positions_by_type(graph, positions, ntype)
        if not xs:
            continue
        hollow = ntype in hollow_node_types
        ax.scatter(
            xs,
            ys,
            facecolors="none" if hollow else color,
            edgecolors=color if hollow else "none",
            marker=node_type_markers[ntype],
            s=identity_marker_size if hollow else marker_size,
            alpha=identity_alpha if hollow else node_alpha,
            linewidths=marker_lw,
            zorder=4,
            rasterized=True,
        )

    if show_legend:
        legend_handles = [
            _marker_handle(
                node_type_markers[ntype],
                node_type_colors[ntype],
                label,
                hollow=ntype in hollow_node_types,
            )
            for ntype, label in node_type_labels.items()
        ]
        # Column-first order (matplotlib fills legends down each column, not across rows) for
        # a 4-col x 2-row layout: nodes in cols 1-2, edges in cols 3-4.
        legend_handles += [
            mlines.Line2D(
                [], [], color=seed_color, lw=1.5, label="Article-repository link (seed)"
            ),
            mlines.Line2D(
                [], [], color=mined_color, lw=1.5, label="Article-repository link (mined)"
            ),
            mlines.Line2D([], [], color=edge_grey, lw=1.2, label="Authorship / contribution"),
            mlines.Line2D(
                [], [], color=edge_grey, lw=1.8, label="Researcher-developer identity"
            ),
        ]
        legend = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=4,
            fontsize=8,
        )
        u.style_legend(legend, fontsize=8)

    ax.set_axis_off()
    ax.set_aspect("equal")
    u.print_caption_note(stem, caption)

    u.save_figure(fig, stem, output_dir)
    plt.close(fig)


def _render_layouts(
    graph: rx.PyGraph,
    seed: int,
    caption_base: str,
    output_dir: Path,
    stem_suffix: str = "",
) -> None:
    """Compute ForceAtlas2 (the only layout this figure draws now -- warm-started from a
    global spring layout that is never itself rendered) and draw two files: the main output
    with no legend (panel B carries the encoding) and a `_review-legend` copy with a small
    reference legend for sign-off. `stem_suffix` (e.g. "_n15000") writes a trial render to its
    own files without touching the standard-name outputs.
    """
    base_stem = f"figure1_quadpartite_network_forceatlas2{stem_suffix}"
    print("\nComputing global spring layout (ForceAtlas2 warm-start)...")
    positions, _canvas_width, _canvas_height = _global_spring_layout(graph, seed=seed)
    print("\nComputing ForceAtlas2 layout (warm-started from spring positions)...")
    fa2_positions = _forceatlas2_positions(graph, positions, seed=seed)
    xs = [p[0] for p in fa2_positions.values()]
    ys = [p[1] for p in fa2_positions.values()]
    _draw_quadpartite_network(
        graph,
        fa2_positions,
        max(xs),
        max(ys),
        caption_base,
        output_dir,
        stem=base_stem,
        show_legend=False,
    )
    _draw_quadpartite_network(
        graph,
        fa2_positions,
        max(xs),
        max(ys),
        caption_base,
        output_dir,
        stem=f"{base_stem}_review-legend",
        show_legend=True,
    )


###############################################################################
# Command


def figure_1_quadpartite_network(
    output_dir: Path = u.OUTPUT_DIR,
    n_pairs: int = 5000,
    n_top_hub_anchors_per_field: int = 2,
    n_random_hub_anchors_per_field: int = 3,
    min_anchor_pair_degree: int = 3,
    max_pairs_per_entity: int = 6,
    max_contributors_per_side: int = 2,
    field_share_cap: float = 0.18,
    rdal_confidence_threshold: float = 0.97,
    random_seed: int = 42,
    output_stem_suffix: str = "",
) -> None:
    """
    Build Figure 1, panel A: the quad-partite network -- articles, repositories, researchers
    (authors), and developer accounts (contributors) -- rendered together with rustworkx.
    The workflow diagram (Figure 1, panel B) is a hand-made image and out of scope here.
    `output_stem_suffix` (e.g. "_n15000") routes a trial render to its own filenames.

    Sampling: field-stratified snowball from well-linked identity hubs, rather than a
    uniform-random pair sample (two randomly sampled pairs are bridged only if both happen to
    be drawn and share an entity, which is rare, so random samples look artificially
    fragmented). Design:

      - Strata: the six most common article fields plus a pooled "Other" (the same grouping
        that colors the article nodes). Each identity link (researcher, developer) is
        assigned the modal field of the pairs reachable through either side.
      - Anchors: per stratum, the top `n_top_hub_anchors_per_field` identity links by
        pair-degree plus `n_random_hub_anchors_per_field` seeded-random identities with
        pair-degree >= `min_anchor_pair_degree`; the random tail keeps the figure from
        over-representing anomalous mega-hubs, and the per-field split keeps the
        Computer-Science hubs (by far the largest) from seeding the whole sample.
      - Growth: BFS over pairs, mediated by entities. Each popped pair contributes up to
        `max_contributors_per_side` authors/contributors plus, always, any identity-linked
        author/contributor (a hard cap alone would sever identity edges whenever the linked
        person isn't among the first contributor rows). Each collected entity (and its
        identity counterpart) contributes up to `max_pairs_per_entity` new pairs (seeded
        subsample), which keeps hubs from dominating. No stratum may exceed
        `field_share_cap` of the `n_pairs` budget: once a field is full, its pairs are
        skipped and the walk continues through the other fields' pairs.
      - Stop the instant the `n_pairs` budget fills; if all anchors exhaust below budget, top
        up with uniform-random pairs and report the top-up count.

    Confidence filters: pair confidence >= 0.9994 or NULL (standard); identity links at
    >= `rdal_confidence_threshold` (default 0.97, the dataset-wide threshold).
    """
    evaplot.set_style("evaplot_rc")
    rng = random.Random(random_seed)

    pairs = u.load_filtered_pairs()

    # Seed vs. mined provenance per pair (`link_processing_iteration` IS NULL == seed), for the
    # article-repository edge colouring below -- same convention as figure2_dataset_coverage.py.
    pair_is_mined: dict[int, bool] = {
        pid: it is not None
        for pid, it in pairs.select(
            "document_repository_link_id", "link_processing_iteration"
        ).iter_rows()
    }

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

    # ---- Field strata: top 6 fields + Other. Used both to stratify the sample and to color
    # the article nodes (the only color-encoded node type). ----
    fig1_fields = (
        pairs.get_column("document_field_name_pruned")
        .value_counts(sort=True)
        .filter(pl.col("document_field_name_pruned") != "Other")
        .head(6)
        .get_column("document_field_name_pruned")
        .to_list()
    )
    strata = [*fig1_fields, "Other"]
    doc_fields = {
        did: (f if f in fig1_fields else "Other")
        for did, f in pairs.select("document_id", "document_field_name_pruned")
        .unique(subset="document_id")
        .iter_rows()
    }

    pair_frame = pairs.select(
        pl.col("document_repository_link_id").alias("pair_id"), "document_id", "repository_id"
    )
    lookups = _build_snowball_lookups(
        pair_frame, document_contributors, repository_contributors, rdal
    )
    pair_field = {pid: doc_fields[doc_id] for pid, doc_id in lookups.pair_doc.items()}
    population_mix = Counter(pair_field.values())
    print(
        "Filtered-population field mix: "
        + ", ".join(f"{f} {100 * population_mix[f] / len(pair_field):.1f}%" for f in strata)
    )
    anchor_queue, n_planned_anchors = _select_anchors(
        lookups,
        pair_field,
        strata,
        n_top_hub_anchors_per_field,
        n_random_hub_anchors_per_field,
        min_anchor_pair_degree,
        rng,
    )

    budget = min(n_pairs, len(lookups.pair_doc))
    max_pairs_per_field = int(field_share_cap * budget)
    sampled, n_anchors_used = _snowball_sample(
        lookups,
        anchor_queue,
        n_planned_anchors,
        budget,
        max_pairs_per_entity,
        max_contributors_per_side,
        rng,
        pair_field,
        max_pairs_per_field,
    )

    n_snowball = len(sampled)
    n_topup = 0
    if n_snowball < budget:
        remainder = [p for p in lookups.pair_doc if p not in sampled]
        topup = rng.sample(remainder, min(budget - n_snowball, len(remainder)))
        sampled.update(topup)
        n_topup = len(topup)
    print(
        f"\nSnowball sample: {n_snowball:,} pairs from {n_anchors_used:,} anchors"
        + (f" + {n_topup:,} uniform-random top-up pairs" if n_topup else "")
        + f" = {len(sampled):,} of {len(lookups.pair_doc):,} filtered pairs "
        f"(per-field cap {max_pairs_per_field:,} pairs)."
    )
    sample_mix = Counter(pair_field[p] for p in sampled)
    print(
        "Sampled field mix: "
        + ", ".join(
            f"{f} {sample_mix[f]:,} ({100 * sample_mix[f] / len(sampled):.1f}%)" for f in strata
        )
    )

    sampled_pairs = pair_frame.filter(pl.col("pair_id").is_in(sampled)).rename(
        {"pair_id": "document_repository_link_id"}
    )

    doc_authors, repo_devs = _sampled_entity_frames(
        sampled_pairs, lookups, max_contributors_per_side
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

    graph = _build_quadpartite_graph(
        sampled_pairs,
        doc_authors,
        repo_devs,
        identity_edges,
        pair_is_mined,
        doc_fields=doc_fields,
    )

    n_nodes = len(graph.nodes())
    n_edges = len(graph.edges())
    counts_by_type: dict[str, int] = {}
    for node in graph.nodes():
        counts_by_type[node["type"]] = counts_by_type.get(node["type"], 0) + 1
    print(f"\nFinal sampled network: {n_nodes:,} nodes, {n_edges:,} edges.")
    print(f"  Node counts by type: {counts_by_type}")

    # ---- Connectivity reporting ----
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

    drawn_mix = Counter(
        graph[node_i]["group"]
        for node_i in graph.node_indices()
        if graph[node_i]["type"] == "article"
    )
    n_drawn_articles = sum(drawn_mix.values())
    print(
        "Drawn-component article field mix: "
        + ", ".join(
            f"{f} {drawn_mix[f]:,} ({100 * drawn_mix[f] / n_drawn_articles:.1f}%)"
            for f in strata
        )
    )

    drawn_link_types = Counter(
        graph.get_edge_data_by_index(idx)["type"] for idx in graph.edge_indices()
    )
    print(
        "Drawn article-repository links by provenance: "
        f"seed {drawn_link_types['article_repository_link_seed']:,}, "
        f"mined {drawn_link_types['article_repository_link_mined']:,}."
    )

    caption_base = (
        f"Field-stratified snowball sample of the quad-partite network (n={len(sampled):,} "
        f"article-repository pairs, at most {field_share_cap:.0%} per field stratum; "
        f"{len(comp_sizes):,} connected components existed, only the largest is drawn = "
        f"{pct_pairs_in_largest:.1f}% of pairs)"
    )

    _render_layouts(graph, random_seed, caption_base, output_dir, output_stem_suffix)
