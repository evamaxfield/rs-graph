"""Q_libraries supplement: a per-library lookup table joining all three views.

Each of the three library-usage rankings (imported, manifest-declared
dependencies, mentioned-in-text) uses a different naming convention for the
same underlying library -- torch/pytorch being the canonical example -- and
top_libraries.py only ever surfaces the top 10 of each in isolation. Exact
string matching on `software_name_normalized` therefore misses real matches
across views (torch vs pytorch, sklearn vs scikit-learn, etc).

This builds one row per *canonical library identity*, not one row per raw
name, by reusing `rs_graph.utils.software_alignment.align_software_names()`
-- the same Hungarian-algorithm + rapidfuzz + alternates-table machinery
already used to align software names within a single article-repo pair
elsewhere in rs_graph -- extended here to a global, three-way reconciliation
across the whole dataset:

1. Pairwise-align the top-200-by-count name lists for each pair of views
   (imports<->dependencies, imports<->mentions, dependencies<->mentions).
   Each pairwise alignment is a one-to-one (Hungarian) assignment, so a name
   is matched to at most one partner per other view.
2. Union the three pairwise alignments into canonical groups via a greedy,
   highest-score-first union-find, with one guard: a merge is only applied
   if it would not put two different names from the *same* view into one
   group. This is what keeps the merge well-defined despite fuzzy alignment
   being non-transitive (A~B and B~C at the cutoff doesn't guarantee A~C) --
   processing edges strongest-first and refusing same-view collisions means
   a weak indirect chain can never silently fuse two genuinely different
   libraries into one row, since the strong direct edges are locked in
   first and a later, weaker edge that would violate the one-name-per-view
   invariant is simply skipped rather than forcing a merge.

Dependencies are pooled across ecosystems into one count/rank (this is a
lookup table, not a chart needing ecosystem color), with the top-contributing
ecosystem retained as a side signal.
"""

import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from lib.confidence import filter_high_precision_document_repository_links  # noqa: E402
from lib.hf_loader import load_table  # noqa: E402
from rs_graph.utils.identifier_normalization import normalize_name  # noqa: E402
from rs_graph.utils.software_alignment import align_software_names  # noqa: E402
from rs_graph.utils.software_alternates import load_alternate_groups  # noqa: E402

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "library_cross_view.json")

TOP_N_PER_VIEW = 200
# align_software_names()'s own default (75.0) is tuned for per-article-repo-pair
# alignment, where each side is a short, specific list -- a coincidental collision
# is unlikely. Here the candidate pool is up to 200 names per view, so short
# tokens collide easily at 75 on plain rapidfuzz.ratio (e.g. "mass"/"mask",
# "blas"/"blast", "rsa"/"rstan" all score >= 75 and are not the same library).
# Raised to 90 for this global pass -- checked empirically this pass: every
# fuzzy-only match kept at 90 is either exact or a genuine near-duplicate
# (e.g. "torch"=="torch"); known short-name aliases (torch/pytorch, pil/pillow,
# cv2/opencv, bs4/beautifulsoup4, skimage/scikit-image, yaml/pyyaml,
# absl/absl-py, huggingfacehub/huggingface) still merge because
# are_alternates() overrides the score to 100 regardless of cutoff -- the
# curated alternates table, not the fuzzy score, is what should be trusted for
# short/dissimilar-but-known-same names at this scale.
ALIGNMENT_CUTOFF = 90.0


def _collapse_intra_view_alternates(
    df: pl.DataFrame, id_col: str, name_col: str
) -> pl.DataFrame:
    """Collapse same-view raw spellings that the alternates table already knows
    are the same library (e.g. a pypi dependency named "torch" and a conda
    dependency named "pytorch") to one representative spelling before ranking.

    This matters specifically for the dependency view, which pools multiple
    package ecosystems into one column -- different ecosystems can use a
    genuinely different canonical name for the same real library, so without
    this step the cross-view alignment below (which matches at most one name
    per view into a canonical group) can only ever pick up one of the two
    spellings, silently leaving the other stranded as its own low-ranked row.
    Deliberately alternates-table-only (not fuzzy-scored) for this intra-view
    pass -- an exact, curated equivalence is safe to collapse blindly; a fuzzy
    score is not, since two genuinely different top-N libraries in the same
    view are far more likely to collide by accident than across-view pairs.
    """
    alternate_groups = load_alternate_groups()
    counts = df.group_by(name_col).agg(n=pl.col(id_col).n_unique())

    group_key_by_name: dict[str, frozenset[str]] = {}
    for name in counts[name_col].to_list():
        norm = normalize_name(name)
        group = alternate_groups.get(norm)
        group_key_by_name[name] = group if group else frozenset({norm})

    # Representative spelling per group = whichever raw name has the highest
    # count in this view (the most recognizable/common form here).
    best_for_group: dict[frozenset[str], tuple[str, int]] = {}
    for row in counts.iter_rows(named=True):
        name, n = row[name_col], row["n"]
        key = group_key_by_name[name]
        if key not in best_for_group or n > best_for_group[key][1]:
            best_for_group[key] = (name, n)

    rename_map = {name: best_for_group[key][0] for name, key in group_key_by_name.items()}
    return df.with_columns(pl.col(name_col).replace(rename_map).alias(name_col))


def _ranked_counts(df: pl.DataFrame, id_col: str, name_col: str) -> pl.DataFrame:
    return (
        df.group_by(name_col)
        .agg(count=pl.col(id_col).n_unique())
        .sort("count", descending=True)
        .with_columns(rank=pl.int_range(1, pl.len() + 1))
    )


class _UnionFind:
    """Union-find over (view, name) nodes, refusing merges that would put two
    different names from the same view into one group."""

    def __init__(self) -> None:
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}
        self.members: dict[tuple[str, str], list[tuple[str, str]]] = {}

    def make(self, node: tuple[str, str]) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.members[node] = [node]

    def find(self, node: tuple[str, str]) -> tuple[str, str]:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def try_union(self, a: tuple[str, str], b: tuple[str, str]) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        views_a = {n[0] for n in self.members[ra]}
        views_b = {n[0] for n in self.members[rb]}
        if views_a & views_b:
            return False
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].extend(self.members[rb])
        del self.members[rb]
        return True


def _canonical_groups(
    import_names: list[str],
    dependency_names: list[str],
    mention_names: list[str],
    cutoff: float = ALIGNMENT_CUTOFF,
) -> tuple[list[dict[str, str | None]], int, int]:
    """Reconcile three name lists into canonical (cross-view) library groups.

    Returns (groups, n_merges_applied, n_merges_skipped_for_view_conflict),
    where each group is a dict of {"import": name_or_None, "dependency":
    name_or_None, "mention": name_or_None}.
    """
    uf = _UnionFind()
    for view, names in (
        ("import", import_names),
        ("dependency", dependency_names),
        ("mention", mention_names),
    ):
        for name in names:
            uf.make((view, name))

    edges: list[tuple[tuple[str, str], tuple[str, str], float]] = []
    for view_a, names_a, view_b, names_b in (
        ("import", import_names, "dependency", dependency_names),
        ("import", import_names, "mention", mention_names),
        ("dependency", dependency_names, "mention", mention_names),
    ):
        for result in align_software_names(
            names_a, names_b, view_a, view_b, cutoff=cutoff, use_alternates=True
        ):
            edges.append(
                (
                    (result.item_one_source, result.item_one),
                    (result.item_two_source, result.item_two),
                    result.score,
                )
            )
    # Highest-confidence matches win the right to merge first -- this is what
    # makes the union-find's same-view guard actually protect against
    # non-transitive chains rather than just merging in arbitrary edge order.
    edges.sort(key=lambda e: e[2], reverse=True)

    n_applied = 0
    n_skipped = 0
    for node_a, node_b, _score in edges:
        if uf.try_union(node_a, node_b):
            n_applied += 1
        else:
            n_skipped += 1

    seen_roots: set[tuple[str, str]] = set()
    groups: list[dict[str, str | None]] = []
    for node in uf.parent:
        root = uf.find(node)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        group: dict[str, str | None] = {"import": None, "dependency": None, "mention": None}
        for view, name in uf.members[root]:
            group[view] = name
        groups.append(group)

    return groups, n_applied, n_skipped


def run() -> dict:
    # --- site-snippet:start ---
    hp_links = filter_high_precision_document_repository_links(
        load_table("document_repository_link")
    ).select("document_id", "repository_id")
    hp_repo_ids = hp_links.select("repository_id").unique()
    hp_doc_ids = hp_links.select("document_id").unique()

    imports = load_table("repository_import").join(hp_repo_ids, on="repository_id", how="inner")
    dependencies = load_table("repository_dependency").join(
        hp_repo_ids, on="repository_id", how="inner"
    )
    mentions = load_table("document_software_mention").join(
        hp_doc_ids, on="document_id", how="inner"
    )

    # Collapse same-view alternate spellings (e.g. a conda dependency named
    # "pytorch" vs a pypi dependency named "torch") before ranking -- see
    # _collapse_intra_view_alternates for why this has to happen per-view,
    # ahead of the cross-view fuzzy alignment below.
    imports = _collapse_intra_view_alternates(
        imports, "repository_id", "software_name_normalized"
    )
    mentions = _collapse_intra_view_alternates(
        mentions, "document_id", "software_name_normalized"
    )

    import_ranked = _ranked_counts(imports, "repository_id", "software_name_normalized")
    mention_ranked = _ranked_counts(mentions, "document_id", "software_name_normalized")

    # Dependencies pooled across ecosystems -- one row per (name, repository_id)
    # regardless of which ecosystem(s) declared it, so a repo declaring the same
    # normalized name in two ecosystems only counts once. Alternates collapse
    # happens first so "torch" (pypi) and "pytorch" (conda) pool as one name.
    dependencies_pooled = dependencies.select(
        "repository_id", "software_name_normalized", "ecosystem"
    )
    dependencies_pooled = _collapse_intra_view_alternates(
        dependencies_pooled, "repository_id", "software_name_normalized"
    ).unique(subset=["repository_id", "software_name_normalized"], keep="first")
    dependency_ranked = _ranked_counts(
        dependencies_pooled, "repository_id", "software_name_normalized"
    )
    # Top-contributing ecosystem per library: whichever ecosystem accounts for
    # the most distinct declaring repos for that name.
    dependency_top_ecosystem = (
        dependencies_pooled.group_by(["software_name_normalized", "ecosystem"])
        .agg(n=pl.col("repository_id").n_unique())
        .sort("n", descending=True)
        .group_by("software_name_normalized")
        .head(1)
        .select(
            "software_name_normalized",
            pl.col("ecosystem").alias("dependency_top_ecosystem"),
        )
    )

    import_top = import_ranked.head(TOP_N_PER_VIEW)["software_name_normalized"].to_list()
    dependency_top = dependency_ranked.head(TOP_N_PER_VIEW)[
        "software_name_normalized"
    ].to_list()
    mention_top = mention_ranked.head(TOP_N_PER_VIEW)["software_name_normalized"].to_list()

    # The core fix: reconcile the three top-N name lists into canonical,
    # cross-view library identities via fuzzy alignment instead of an exact
    # string join -- see module docstring for the algorithm.
    groups, n_merges_applied, n_merges_skipped = _canonical_groups(
        import_top, dependency_top, mention_top
    )

    lookups = {
        "import": {
            r["software_name_normalized"]: (r["count"], r["rank"])
            for r in import_ranked.head(TOP_N_PER_VIEW).iter_rows(named=True)
        },
        "dependency": {
            r["software_name_normalized"]: (r["count"], r["rank"])
            for r in dependency_ranked.head(TOP_N_PER_VIEW).iter_rows(named=True)
        },
        "mention": {
            r["software_name_normalized"]: (r["count"], r["rank"])
            for r in mention_ranked.head(TOP_N_PER_VIEW).iter_rows(named=True)
        },
    }
    dep_ecosystem_lookup = {
        r["software_name_normalized"]: r["dependency_top_ecosystem"]
        for r in dependency_top_ecosystem.iter_rows(named=True)
    }

    n_hp_repos = hp_repo_ids.height
    n_hp_docs = hp_doc_ids.height
    # --- site-snippet:end ---

    rows = []
    n_multi_spelling_groups = 0
    for group in groups:
        import_name = group["import"]
        dependency_name = group["dependency"]
        mention_name = group["mention"]

        import_count, import_rank = lookups["import"].get(import_name, (None, None))
        dependency_count, dependency_rank = lookups["dependency"].get(
            dependency_name, (None, None)
        )
        mention_count, mention_rank = lookups["mention"].get(mention_name, (None, None))

        # Canonical display name: whichever view's spelling has the highest
        # count (i.e. the most recognizable/common form).
        candidates = [
            (import_name, import_count or -1),
            (dependency_name, dependency_count or -1),
            (mention_name, mention_count or -1),
        ]
        display_name = max((c for c in candidates if c[0] is not None), key=lambda c: c[1])[0]

        distinct_spellings = {n for n in (import_name, dependency_name, mention_name) if n}
        if len(distinct_spellings) > 1:
            n_multi_spelling_groups += 1

        rows.append(
            {
                "name": display_name,
                "aliases": {
                    "import": import_name,
                    "dependency": dependency_name,
                    "mention": mention_name,
                },
                "import_count": int(import_count) if import_count is not None else None,
                "import_rank": int(import_rank) if import_rank is not None else None,
                "dependency_count": (
                    int(dependency_count) if dependency_count is not None else None
                ),
                "dependency_rank": (
                    int(dependency_rank) if dependency_rank is not None else None
                ),
                "dependency_top_ecosystem": dep_ecosystem_lookup.get(dependency_name),
                "mention_count": int(mention_count) if mention_count is not None else None,
                "mention_rank": int(mention_rank) if mention_rank is not None else None,
            }
        )

    rows.sort(key=lambda r: (r["import_count"] is None, -(r["import_count"] or 0)))

    result = {
        "question": (
            "For a given library, where does it land across imports, declared "
            "dependencies, and paper-text mentions?"
        ),
        "methodology": (
            "Each view first has same-view alternate spellings collapsed via the "
            "curated software-name-alternates table (e.g. a pypi dependency named "
            "\"torch\" and a conda dependency named \"pytorch\"), then ranked to "
            f"top-{TOP_N_PER_VIEW}-by-distinct-count in each of the three views "
            "(imports, dependencies pooled across ecosystems, mentions), restricted "
            "to high-precision article-repository links. The three top-N lists are "
            "then reconciled into canonical cross-view library identities via "
            "rs_graph.utils.software_alignment.align_software_names() (Hungarian "
            f"assignment + rapidfuzz, cutoff={ALIGNMENT_CUTOFF}) pairwise across all "
            "three views, unioned greedily highest-score-first with a same-view "
            "collision guard (see module docstring for why that guard matters). A "
            f"null count/rank means the library fell outside that view's top "
            f"{TOP_N_PER_VIEW} -- not that it was measured and found at zero."
        ),
        "top_n_per_view": TOP_N_PER_VIEW,
        "alignment_cutoff": ALIGNMENT_CUTOFF,
        "n_hp_repos": n_hp_repos,
        "n_hp_docs": n_hp_docs,
        "n_canonical_groups": len(rows),
        "n_groups_with_multiple_spellings": n_multi_spelling_groups,
        "n_fuzzy_merges_applied": n_merges_applied,
        "n_fuzzy_merges_skipped_for_view_conflict": n_merges_skipped,
        "rows": rows,
    }
    return result


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"wrote {len(result['rows'])} rows to {OUTPUT_PATH} "
        f"({result['n_groups_with_multiple_spellings']} multi-spelling groups)"
    )
