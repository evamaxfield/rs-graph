# Task 5: Pairwise Alignment and Reconciliation

## Overall Project Context

RQ3 of the rs-graph research project asks: **how do three observable views of scientific software use — mentions in articles, imports in code, and dependency manifests — overlap and diverge?**

The prototype script contains an `align_dependencies()` function that matches imported libraries to paper mentions using the Hungarian algorithm with fuzzy string matching. This task refactors and extends that logic to support three-way pairwise comparison (imports vs dependencies vs mentions) with reconciliation into unified software entities.

## End Goal

A reusable alignment module that:
1. Takes three sets of software names (imports, dependencies, mentions) for a single document-repository pair
2. Runs three pairwise alignments using fuzzy matching + Hungarian algorithm
3. Reconciles the pairwise results into unified software entities via transitive closure
4. Produces structured records showing which sources each software entity appeared in

## This Task

### Three-way pairwise alignment approach

For a given document-repository pair with imports `I`, dependencies `D`, and mentions `M`:

1. **Pairwise alignments** (reusing the existing `align_dependencies()` logic):
   - `align(I, D)` → which imports match which dependencies
   - `align(I, M)` → which imports match which mentions
   - `align(D, M)` → which dependencies match which mentions

2. **Reconciliation via transitive closure**:
   - Build a graph where nodes are (source, name) tuples and edges are pairwise matches
   - Find connected components — each component is one software entity
   - Example: if `sklearn` (import) matches `scikit-learn` (dependency), and `scikit-learn` (dependency) matches `Scikit-Learn` (mention), all three are one entity — even if `sklearn` and `Scikit-Learn` don't directly match well

3. **Output structure** — a `ReconciledSoftwareRecord` (or similar) per entity:
   - `import_name` (str | None) — the import name if present
   - `dependency_name` (str | None) — the dependency name if present
   - `mention_name` (str | None) — the mention name if present
   - `normalized_name` (str) — best normalized form
   - `import_dependency_score` (float | None) — pairwise match score
   - `import_mention_score` (float | None)
   - `dependency_mention_score` (float | None)
   - `sources` (set[str]) — e.g., `{"import", "dependency"}` or `{"import", "dependency", "mention"}`

### Key design decisions

- **Reuse existing Hungarian algorithm matching**: The `align_dependencies()` function and its cost matrix approach work well; parameterize it to accept any two sets
- **Transitive closure for reconciliation**: Simple union-find or connected components algorithm on the match graph
- **Score cutoff**: Same configurable cutoff (default 75.0) as the prototype; may need tuning for dependency-vs-import matching which should be higher confidence
- **Handle missing sources gracefully**: Not all pairs will have all three sources (e.g., a repo might have no SoftCite mentions). The alignment should degrade gracefully to two-way or even report single-source-only entities

## Files to Read

- `notebooks/example-used-software-comparison.py` — source of existing alignment logic:
  - `align_dependencies()` (lines 541-619) — Hungarian algorithm matching with fuzzy scores
  - `_create_software_records()` (lines 276-333) — creating structured records from match results
  - `SoftwareRecord` dataclass (lines 257-273) — current record structure (will be extended)
  - `normalize_name()` (lines 228-245) — used during alignment
- `rs_graph/utils/normalization.py` — shared normalization functions (from Task 2)

## Files to Create

- `rs_graph/utils/software_alignment.py` — contains:
  - `align_software_names()` — generalized pairwise alignment (refactored from `align_dependencies`)
  - `reconcile_pairwise()` — transitive closure reconciliation
  - `align_three_way()` — convenience function that runs all three alignments and reconciles
  - `ReconciledSoftwareRecord` dataclass

## Files to Edit

- `notebooks/example-used-software-comparison.py` — update to import from shared module (if still used as a standalone script)

## Commands to Run

- `just lint` — ensure code passes linting and type checking
- Test with sample data: pick 5-10 document-repository pairs that have all three sources populated, run the three-way alignment, and manually inspect results for correctness
