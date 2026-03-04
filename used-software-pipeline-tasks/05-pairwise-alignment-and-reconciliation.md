# Task 5: Pairwise Alignment Function

## Overall Project Context

RQ3 of the rs-graph research project asks: **how do three observable views of scientific software use — mentions in articles, imports in code, and dependency manifests — overlap and diverge?**

The prototype script contains an `align_dependencies()` function that matches imported libraries to paper mentions using the Hungarian algorithm with fuzzy string matching. This task refactors that logic into a reusable pairwise alignment function that can be applied during analysis.

## End Goal

A single reusable function `align_software_names()` that:
1. Accepts any two sets of software names (with source labels)
2. Runs fuzzy matching + Hungarian algorithm to find the best one-to-one alignment
3. Returns structured records with both original and normalized names and a match score

The function can be called during analysis with any pair of sources (imports↔mentions, imports↔dependencies, dependencies↔mentions).

## This Task

### Output structure

Each matched pair is a `PairwiseAlignmentResult`:

```python
@dataclass
class PairwiseAlignmentResult:
    item_one_source: str          # e.g. "import"
    item_one: str                 # original name from source one
    normalized_item_one: str      # normalized form
    item_two_source: str          # e.g. "mention"
    item_two: str                 # original name from source two
    normalized_item_two: str      # normalized form
    score: float                  # fuzzy match score (0–100)
```

### Function signature

```python
def align_software_names(
    items_a: list[str],
    items_b: list[str],
    source_a: str,
    source_b: str,
    cutoff: float = 75.0,
) -> list[PairwiseAlignmentResult]:
```

- Normalizes both lists using `normalize_name()` from `rs_graph/utils/identifier_normalization.py`
- Builds a cost matrix of fuzzy match scores (using `rapidfuzz` or `thefuzz`)
- Runs the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) to find optimal one-to-one assignments
- Filters out assignments below `cutoff`
- Returns one `PairwiseAlignmentResult` per accepted match

### Key design decisions

- **Reuse existing Hungarian algorithm matching**: The `align_dependencies()` function (prototype lines 541–619) has the cost matrix + Hungarian logic; extract and generalize it
- **Score cutoff**: Same configurable default (75.0) as the prototype
- **Unmatched items are not returned**: The caller can find unmatched items by diffing the input lists against the returned results

## Files to Read

- `notebooks/example-used-software-comparison.py` — source of existing alignment logic:
  - `align_dependencies()` (lines 541–619) — Hungarian algorithm matching with fuzzy scores
  - `normalize_name()` (lines 228–245) — used during alignment
- `rs_graph/utils/identifier_normalization.py` — shared normalization functions (from Task 2); use `normalize_name()` from here

### Actual DB field names (from Task 1 implementation)

When querying the three tables, use these exact field names:

- `RepositoryImport`: `repository_id`, `software_name`, `software_name_normalized`, `file_paths` (semicolon-separated, nullable)
- `RepositoryDependency`: `repository_id`, `software_name`, `software_name_normalized`, `version_spec`, `ecosystem` (e.g. "PyPI", "CRAN"), `manifest_paths` (semicolon-separated), `dependency_type` (e.g. "runtime", "development")
- `DocumentSoftwareMention`: `document_id`, `software_name`, `software_name_normalized`, `mention_context`

## Files to Create

- `rs_graph/utils/software_alignment.py` — contains:
  - `PairwiseAlignmentResult` dataclass
  - `align_software_names()` function

## Files to Edit

- `notebooks/example-used-software-comparison.py` — update to import from shared module (if still used as a standalone script)

## Commands to Run

- `just lint` — ensure code passes linting and type checking
- Test by calling `align_software_names()` with sample imports and mentions from a real document-repository pair and inspecting the results
