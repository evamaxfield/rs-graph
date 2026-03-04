# Task 7: Three-Way Alignment and Reconciliation

## Overall Project Context

RQ3 of the rs-graph research project asks: **how do three observable views of scientific software use — mentions in articles, imports in code, and dependency manifests — overlap and diverge?**

Task 5 provides a reusable `align_software_names()` function for pairwise alignment of any two sets of software names. This task extends that into three-way alignment and reconciliation, producing unified software entities that span all three sources.

> **Note**: This is a future/deferred task. The current analysis (Task 6) applies `align_software_names()` pairwise directly. This task is for when unified entity reconciliation across all three sources is needed.

## End Goal

A reconciliation layer on top of `align_software_names()` that:
1. Runs all three pairwise alignments (imports↔deps, imports↔mentions, deps↔mentions)
2. Applies transitive closure to merge overlapping pairwise matches into unified software entities
3. Returns structured records showing which sources each entity appeared in and all pairwise scores

## This Task

### Reconciliation via transitive closure

Given pairwise results from all three alignments:

1. Build a graph where nodes are `(source, name)` tuples and edges are accepted pairwise matches
2. Find connected components — each component is one software entity
3. Example: if `sklearn` (import) matches `scikit-learn` (dependency), and `scikit-learn` (dependency) matches `Scikit-Learn` (mention), all three are one entity — even if `sklearn` and `Scikit-Learn` don't directly match well

### Output structure

Each reconciled entity is a `ReconciledSoftwareRecord`:

```python
@dataclass
class ReconciledSoftwareRecord:
    import_name: str | None               # import name if present
    dependency_name: str | None           # dependency name if present
    mention_name: str | None              # mention name if present
    normalized_name: str                  # best normalized form
    import_dependency_score: float | None # pairwise match score
    import_mention_score: float | None
    dependency_mention_score: float | None
    sources: set[str]                     # e.g. {"import", "dependency"}
```

### Functions

```python
def reconcile_pairwise(
    imports_deps: list[PairwiseAlignmentResult],
    imports_mentions: list[PairwiseAlignmentResult],
    deps_mentions: list[PairwiseAlignmentResult],
) -> list[ReconciledSoftwareRecord]:
    ...

def align_three_way(
    imports: list[str],
    dependencies: list[str],
    mentions: list[str],
    cutoff: float = 75.0,
) -> list[ReconciledSoftwareRecord]:
    """Convenience wrapper: runs all three pairwise alignments then reconciles."""
    ...
```

### Key design decisions

- **Reuse `align_software_names()`** from `rs_graph/utils/software_alignment.py` (Task 5) for all three pairwise steps
- **Transitive closure**: Simple union-find or `networkx` connected components on the match graph
- **Handle missing sources gracefully**: Not all pairs will have all three sources (e.g., a repo might have no SoftCite mentions); single-source-only entities should still appear in the output with the other name fields as `None`
- **Normalized name selection**: Pick the import name's normalized form as canonical if available, otherwise dependency, otherwise mention

## Files to Read

- `rs_graph/utils/software_alignment.py` — `PairwiseAlignmentResult` and `align_software_names()` (from Task 5)
- `rs_graph/utils/identifier_normalization.py` — `normalize_name()` (from Task 2)
- `notebooks/example-used-software-comparison.py` — prototype reconciliation logic if any

## Files to Edit

- `rs_graph/utils/software_alignment.py` — add `ReconciledSoftwareRecord`, `reconcile_pairwise()`, and `align_three_way()`

## Commands to Run

- `just lint` — ensure code passes linting and type checking
- Test with 5–10 document-repository pairs that have all three sources populated; inspect that transitive matches are resolved correctly (e.g., `sklearn` ↔ `scikit-learn` ↔ `Scikit-Learn` unified into one entity)
