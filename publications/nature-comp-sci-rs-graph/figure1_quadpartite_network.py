#!/usr/bin/env python3

"""Figure 1, panel 1: the snowball-sampled quad-partite network of articles, repositories,
researchers, and developer accounts.
"""

from __future__ import annotations

import random
from collections import deque
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
    component's bounding box) with padding proportional to cell size. Returns positions plus
    the packed canvas's width and height (in data units) so the caller can size the figure.
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
            # Spring layout is ~quadratic in node count; iteration count steps down as size
            # grows to keep total layout time in single-digit minutes.
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


###############################################################################
# Command


def figure_1_quadpartite_network(
    output_dir: Path = u.OUTPUT_DIR,
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

    caption_base = (
        f"Snowball-sampled quad-partite network (n={len(sampled):,} article-repository pairs; "
        f"{len(comp_sizes):,} connected components existed, only the largest is drawn = "
        f"{pct_pairs_in_largest:.1f}% of pairs)"
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
                caption_base,
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
                caption_base,
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
            caption_base,
            output_dir,
            fig1_fields,
            stem="figure1_quadpartite_network",
        )


def _draw_quadpartite_network(
    graph: rx.PyGraph,
    positions: dict[int, tuple[float, float]],
    canvas_width: float,
    canvas_height: float,
    caption: str,
    output_dir: Path,
    field_order: list[str],
    stem: str = "figure1_quadpartite_network",
) -> None:
    """Draw and save one Figure 1 render: only articles carry color (by field, matching
    Figure 2 Panel B's palette); repositories/researchers/developers are grey outline-only
    shapes; all edges uniform grey differentiated by alpha/linewidth; LineCollections and
    scatters rasterized so the PDF stays small and fast to open.
    """
    # Shared field-to-color assignment used by every per-field figure.
    field_colors = u.field_color_map([*field_order, "Other"])
    # Repositories get their own light, non-grey color -- muted sky blue, colorblind-safe and
    # unused by any of the six field colors; people nodes stay very light transparent grey.
    repo_color = "#56B4E9"
    people_grey = "#cccccc"

    # Uniform grey edges: edge type is differentiated only by alpha/linewidth, never by hue.
    edge_grey = "#999999"
    edge_style = {
        "authored_by": {"color": edge_grey, "lw": 0.25, "ls": "-", "zorder": 1, "alpha": 0.06},
        "contributed_to": {
            "color": edge_grey,
            "lw": 0.25,
            "ls": "-",
            "zorder": 1,
            "alpha": 0.06,
        },
        "article_repository_link": {
            "color": edge_grey,
            "lw": 0.5,
            "ls": "-",
            "zorder": 2,
            "alpha": 0.15,
        },
        "identity": {"color": edge_grey, "lw": 0.7, "ls": "-", "zorder": 3, "alpha": 0.15},
    }
    marker_size = 3.5
    people_marker_size = 2.5
    node_alpha = 0.5
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

    # Outline-only shapes for everything else: repositories in their own light blue at
    # article-node size; people-nodes stay smallest, very light, and transparent.
    for ntype, marker, edge_color, size, alpha in [
        ("repository", "s", repo_color, marker_size, node_alpha),
        ("researcher", "D", people_grey, people_marker_size, 0.3),
        ("developer", "^", people_grey, people_marker_size, 0.3),
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
        _marker_handle("s", repo_color, "Repository", hollow=True),
        _marker_handle("D", people_grey, "Researcher (author)", hollow=True),
        _marker_handle("^", people_grey, "Developer account", hollow=True),
        mlines.Line2D([], [], color=edge_grey, lw=1.2, label="Authorship / contribution"),
        mlines.Line2D([], [], color=edge_grey, lw=1.5, label="Article-repository link"),
        mlines.Line2D([], [], color=edge_grey, lw=1.8, label="Researcher-developer identity"),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        fontsize=12,
    )
    u.style_legend(legend, fontsize=12)

    ax.set_axis_off()
    ax.set_aspect("equal")
    u.print_caption_note(stem, caption)

    u.save_figure(fig, stem, output_dir)
    plt.close(fig)
