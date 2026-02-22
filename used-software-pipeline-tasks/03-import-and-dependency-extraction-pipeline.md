# Task 3: Import + Dependency Extraction Pipeline

## Overall Project Context

RQ3 of the rs-graph research project asks: **how do three observable views of scientific software use — mentions in articles, imports in code, and dependency manifests — overlap and diverge?**

The current prototype (`notebooks/example-used-software-comparison.py`) handles import extraction for a sample of ~200 repos. This task builds the production extraction pipeline that clones repos once and extracts both imports (via `eil`) and dependencies (via `git-pkgs`), storing results in the database tables created in Task 1.

## End Goal

A CLI script that:
1. Queries the database for document-repository pairs
2. Clones each repository locally
3. Extracts imports via `eil` and dependencies via `git-pkgs` in a single clone session
4. Stores results in `RepositoryImport` and `RepositoryDependency` tables
5. Supports checkpointing/resumption for large batch runs

## This Task

### Pipeline flow per repository

1. Query DB for document-repository pairs (with filtering options: language, dataset source, etc.)
2. Skip pairs that already have extraction results in the DB
3. For each pair:
   a. Clone repo to a temp directory
   b. Convert notebooks to scripts via `nb_to_src` (for import extraction)
   c. Run `eil` to extract imports → list of `(software_name, file_path(s))` tuples
   d. Run `git-pkgs` CLI via `subprocess` to extract dependencies → list of `(software_name, version_spec)` tuples
   e. Normalize all software names using shared `normalize_name()` function
   f. Store imports in `RepositoryImport` table
   g. Store dependencies in `RepositoryDependency` table
   h. Clean up the cloned repo
4. Save progress periodically (cache to parquet or use DB as checkpoint)

### `git-pkgs` integration

`git-pkgs` is installed via `brew install git-pkgs/tap/git-pkgs` and invoked as a CLI tool. Key details to investigate:
- What is the exact CLI invocation and output format?
- Does it need to run inside the repo directory?
- What package ecosystems does it support (pip, npm, cargo, etc.)?
- How does it handle multiple manifest files in a single repo?

Run `git-pkgs --help` and test it on a sample repo to understand the output format before writing the parsing logic.

### Key design decisions

- **Clone once, extract both**: The git clone is the expensive operation; both extractors run on the local clone
- **Batch processing with checkpointing**: Support `--cache-every-n` flag (like the prototype) and skip already-processed pairs
- **Error handling per repo**: A failed clone or extraction for one repo should not stop the batch; log the error and continue
- **Language filtering**: Initially filter to Python/R/Jupyter Notebook repos (matching prototype); `git-pkgs` may enable broader coverage later
- **Use prefect and coiled**: Use prefect for task management and Coiled (with a parameter to enable) to distribute processing to multiple machines

## Files to Read

- `notebooks/example-used-software-comparison.py` — the prototype script; reuse patterns from:
  - `load_pairs()` (lines 75-225) — loading and joining document-repository pairs from DB
  - `get_imported_libraries()` (lines 659-784) — clone + extract loop with checkpointing
  - `_TqdmProgress` class (lines 622-656) — git clone progress display
  - `normalize_name()` (lines 228-245) — name normalization
- `rs_graph/db/models.py` — the new tables from Task 1, plus existing models for querying pairs
- `rs_graph/db/utils.py` — DB session/engine utilities, upsert patterns
- `rs_graph/db/constants.py` — database paths
- `rs_graph/bin/prelinked_dataset_ingestion.py` — reference for pipeline CLI patterns (Typer, batch processing, error tracking)
- `rs_graph/bin/pipeline_utils.py` — shared pipeline utilities

## Files to Create

- `rs_graph/bin/used_software_extraction.py` — main CLI script with Typer commands

## Files to Edit

- `pyproject.toml` — add `git-pkgs`-related dependencies if any Python wrapper is needed; add script entry point

## Commands to Run

- `brew install git-pkgs/tap/git-pkgs` — install git-pkgs (if not already installed)
- `git-pkgs --help` — understand CLI interface and output format
- `just lint` — ensure code passes linting and type checking
- Test on a small sample (5 repos) to verify both extractors work end-to-end
