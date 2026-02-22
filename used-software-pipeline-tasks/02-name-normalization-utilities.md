# Task 2: Name Normalization Utilities

## Overall Project Context

RQ3 of the rs-graph research project asks: **how do three observable views of scientific software use — mentions in articles, imports in code, and dependency manifests — overlap and diverge?**

Several functions for normalizing software names, DOIs, and display text currently live in the prototype script (`notebooks/example-used-software-comparison.py`). These are needed by the extraction pipeline (Task 3), the ingestion script (Task 4), and the alignment logic (Task 5). This task extracts them into a shared utility module.

## End Goal

A reusable utility module containing name normalization functions, importable by all pipeline scripts and analysis notebooks. The prototype script is updated to import from the shared location instead of defining its own copies.

## This Task

### Functions to extract

1. **`normalize_name(name: str) -> str`** (prototype lines 228-245)
   - Lowercases, removes hyphens/underscores/spaces/newlines
   - Used for comparing software names across sources

2. **`prep_name_for_printing(name: str) -> str`** (prototype lines 248-254)
   - Strips newlines and whitespace for display purposes

3. **`normalize_doi_expr(col_name: str) -> pl.Expr`** (prototype lines 55-72)
   - Polars expression for DOI normalization (strip, lowercase, remove URL prefixes)
   - Used in both extraction and ingestion pipelines
   - Note: currently prefixed with `_` as private; make it public in the shared module

### Where to put them

Create `rs_graph/utils/normalization.py` or add to an existing utils module. Check if there's already a natural home in `rs_graph/utils/`.

### Updates to existing code

- Update `notebooks/example-used-software-comparison.py` to import from the shared module
- Any new pipeline scripts (Tasks 3, 4) should import from the shared module

### Note on future enhancement

A more sophisticated normalization approach (canonical name mapping from import names to PyPI/CRAN package names) would improve accuracy for the three-way comparison. This is explicitly **out of scope** for now — the current simple normalization is sufficient for the pairwise alignment approach. If needed later, it can be added to this same utility module.

## Files to Read

- `notebooks/example-used-software-comparison.py` — source of the functions to extract (lines 55-72, 228-254)
- `rs_graph/utils/` — check existing utility modules for a natural home
- `rs_graph/utils/code_host_parsing.py` — example of existing utility module patterns
- `rs_graph/utils/dt_and_td.py` — another example

## Files to Create

- `rs_graph/utils/normalization.py` — new utility module

## Files to Edit

- `notebooks/example-used-software-comparison.py` — update imports to use shared module
- `rs_graph/utils/__init__.py` — add export if needed (check existing pattern)

## Commands to Run

- `just lint` — ensure code passes linting and type checking
