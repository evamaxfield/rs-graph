"""New question: does co-authorship network structure differ by scientific domain?

Why this question, out of the "brainstorm one" brief: `document_contributor`
(researcher_id <-> document_id, with `position`/`is_corresponding`) is the only
author-identity structure the schema actually has at the document level --
there's no richer social-network table (no explicit "collaborator" edge, no
institution-level affiliation graph beyond `document_contributor_institution`,
which is a different question). A co-authorship graph -- an edge between two
researchers whenever they share a document -- is the one network genuinely
implied by the data as it exists, not assumed on top of it. Domain (the same
4-value `Topic.domain_name` used elsewhere on the site) is a clean, already-
validated categorical split to compare that graph's shape across, and "is
collaboration structurally different in Physical Sciences vs. Life Sciences"
is a real, checkable science-of-science question (large multi-author physics
collaborations vs. smaller wet-lab groups is a documented pattern in the
literature) rather than a novel claim invented for this site.

Scope, restricted throughout to high-precision article-repository links (same
population as the rest of the site) and to documents with 2-12 listed authors
-- the upper bound exists because a handful of large-consortium papers (some
research-software papers list 100+ authors) would otherwise blow up the
per-document all-pairs edge count combinatorially without adding real
co-authorship *structure* insight, just star-shaped noise around one paper.

Graph construction and every statistic below uses rustworkx (not networkx),
per an explicit requirement for this pass: one undirected PyGraph per domain,
nodes = researchers with a qualifying document in that domain, edges = a
researcher pair sharing at least one qualifying document (weighted by shared
document count). Reports, per domain: node/edge counts, graph density,
largest-connected-component fraction, and global transitivity (clustering
coefficient) -- the three standard shape statistics for "is this network
denser/more fragmented than that one." Also precomputes a small illustrative
subgraph (one domain's largest connected component, BFS-sampled down to a
size a static SVG can render) with a rustworkx spring layout, so the site can
show an actual picture alongside the per-domain numbers.
"""

import json
import os
from collections import deque

import polars as pl
import rustworkx as rx
from lib.confidence import filter_high_precision_document_repository_links
from lib.hf_loader import load_table

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "output", "coauthorship_network.json"
)

MIN_AUTHORS = 2
MAX_AUTHORS = 12
MAX_ILLUSTRATIVE_NODES = 150


def _build_domain_graph(pairs: pl.DataFrame, node_ids: list[int]) -> rx.PyGraph:
    graph = rx.PyGraph()
    index_by_researcher = {}
    for researcher_id in node_ids:
        index_by_researcher[researcher_id] = graph.add_node(researcher_id)
    for row in pairs.iter_rows(named=True):
        graph.add_edge(
            index_by_researcher[row["researcher_id_a"]],
            index_by_researcher[row["researcher_id_b"]],
            row["n_shared_docs"],
        )
    return graph


def _bfs_sample(graph: rx.PyGraph, start: int, max_nodes: int) -> list[int]:
    """BFS out from `start`, capped at `max_nodes` -- keeps the illustrative
    subgraph visibly connected, unlike a uniform random node sample would."""
    visited = {start}
    order = [start]
    queue = deque([start])
    while queue and len(visited) < max_nodes:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            order.append(neighbor)
            queue.append(neighbor)
            if len(visited) >= max_nodes:
                break
    return order


def run() -> dict:
    # --- site-snippet:start ---
    hp_doc_ids = (
        filter_high_precision_document_repository_links(load_table("document_repository_link"))
        .select("document_id")
        .unique()
    )

    authors = (
        load_table("document_contributor")
        .select("document_id", "researcher_id")
        .unique()
        .join(hp_doc_ids, on="document_id", how="inner")
    )

    author_counts = authors.group_by("document_id").agg(n_authors=pl.len())
    qualifying_docs = author_counts.filter(
        pl.col("n_authors").is_between(MIN_AUTHORS, MAX_AUTHORS)
    ).select("document_id")
    authors = authors.join(qualifying_docs, on="document_id", how="inner")

    # domain per document: best-scoring topic -> domain_name (same join
    # pattern used by ml_tooling_adoption.py and dependency_manifest_growth.py)
    document_topic = load_table("document_topic")
    topic = load_table("topic").select("id", "domain_name").rename({"id": "topic_id"})
    best_topic = (
        document_topic.sort("score", descending=True)
        .group_by("document_id")
        .head(1)
        .join(topic, on="topic_id")
        .select("document_id", "domain_name")
    )
    authors = authors.join(best_topic, on="document_id", how="inner")

    domain_stats = []
    graphs_by_domain: dict[str, rx.PyGraph] = {}
    for domain_name in sorted(authors["domain_name"].unique().to_list()):
        domain_authors = authors.filter(pl.col("domain_name") == domain_name)

        # all co-author pairs within each document, one direction only
        # (researcher_id_a < researcher_id_b), weighted by shared doc count.
        pairs = (
            domain_authors.join(domain_authors, on="document_id", suffix="_b")
            .filter(pl.col("researcher_id") < pl.col("researcher_id_b"))
            .group_by(["researcher_id", "researcher_id_b"])
            .agg(n_shared_docs=pl.col("document_id").n_unique())
            .rename({"researcher_id": "researcher_id_a", "researcher_id_b": "researcher_id_b"})
        )
        node_ids = domain_authors["researcher_id"].unique().to_list()
        graph = _build_domain_graph(pairs, node_ids)
        graphs_by_domain[domain_name] = graph

        n_nodes = graph.num_nodes()
        n_edges = graph.num_edges()
        density = (2 * n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
        components = rx.connected_components(graph)
        largest_component_frac = (
            max(len(c) for c in components) / n_nodes if n_nodes else 0.0
        )
        transitivity = rx.transitivity(graph) if n_nodes > 2 else 0.0

        domain_stats.append(
            {
                "domain_name": domain_name,
                "n_researchers": n_nodes,
                "n_coauthorship_edges": n_edges,
                "n_qualifying_documents": domain_authors["document_id"].n_unique(),
                "density": round(density, 5),
                "largest_component_frac": round(largest_component_frac, 4),
                "largest_component_pct": round(largest_component_frac * 100, 1),
                "transitivity": round(transitivity, 4),
            }
        )

    # Illustrative subgraph: the largest connected component of whichever
    # domain has the most researchers, BFS-sampled down to a renderable size.
    illustrative_domain = max(domain_stats, key=lambda d: d["n_researchers"])["domain_name"]
    illustrative_graph = graphs_by_domain[illustrative_domain]
    components = rx.connected_components(illustrative_graph)
    largest_component = max(components, key=len)
    start_node = max(largest_component, key=lambda n: len(illustrative_graph.neighbors(n)))
    sample_indices = _bfs_sample(illustrative_graph, start_node, MAX_ILLUSTRATIVE_NODES)
    subgraph = illustrative_graph.subgraph(sample_indices)
    layout = rx.spring_layout(subgraph, seed=42)
    # --- site-snippet:end ---

    illustrative_nodes = [
        {
            "id": int(subgraph.get_node_data(i)),
            "x": float(layout[i][0]),
            "y": float(layout[i][1]),
        }
        for i in subgraph.node_indices()
    ]
    illustrative_edges = [
        {"source": int(a), "target": int(b)} for a, b in subgraph.edge_list()
    ]

    result = {
        "question": (
            "Does co-authorship network structure -- how densely connected "
            "authors are -- differ across scientific domains?"
        ),
        "methodology": (
            "document_repository_link filtered to NULL OR confidence >= 0.9994; "
            f"documents restricted to {MIN_AUTHORS}-{MAX_AUTHORS} listed authors "
            "(bounds the per-document all-pairs edge count so large-consortium "
            "papers don't dominate); domain via each document's best-scoring "
            "topic -> Topic.domain_name; graphs built and analyzed with "
            "rustworkx.PyGraph (undirected, one node per researcher, one edge "
            "per co-authoring pair, edge weight = number of shared qualifying "
            "documents)."
        ),
        "min_authors": MIN_AUTHORS,
        "max_authors": MAX_AUTHORS,
        "domains": domain_stats,
        "illustrative_domain": illustrative_domain,
        "illustrative_max_nodes": MAX_ILLUSTRATIVE_NODES,
        "illustrative_nodes": illustrative_nodes,
        "illustrative_edges": illustrative_edges,
    }
    return result


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"wrote {len(result['domains'])} domains, "
        f"{len(result['illustrative_nodes'])}-node illustrative subgraph "
        f"({result['illustrative_domain']}) to {OUTPUT_PATH}"
    )
