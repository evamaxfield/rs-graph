# Task 1: Database Schema — Three Software Tables + Migration

## Overall Project Context

RQ3 of the rs-graph research project asks: **how do three observable views of scientific software use — mentions in articles, imports in code, and dependency manifests — overlap and diverge?**

The current prototype (`notebooks/example-used-software-comparison.py`) handles import extraction and import-vs-mention comparison for a sample of ~200 repos. This larger effort generalizes and productionizes that work by:

1. Adding database tables to store all three software sources (this task)
2. Building an extraction pipeline for imports and dependencies (combined during a single git clone)
3. Ingesting SoftCite mentions into the database
4. Extracting shared normalization and alignment utilities
5. Building pairwise alignment and reconciliation logic
6. Creating the RQ3 analysis notebook

## End Goal

Three new database tables exist in the rs-graph schema, one for each software source type. Imports and dependencies link to `repository`, while mentions link to `document`. An Alembic migration is created and applied successfully.

## This Task

Add three new SQLModel tables to the database:

### Table: `RepositoryImport`
Stores software libraries imported in repository source code (extracted via `eil`).

- `id` (int, PK)
- `repository_id` (FK → `repository.id`)
- `software_name` (str) — original import name as found in code (e.g., `sklearn`)
- `software_name_normalized` (str) — normalized form for comparison
- `file_path` (str, nullable) — file(s) where the import was found

### Table: `RepositoryDependency`
Stores dependencies declared in repository manifests (extracted via `git-pkgs`).

- `id` (int, PK)
- `repository_id` (FK → `repository.id`)
- `software_name` (str) — package name as declared in manifest (e.g., `scikit-learn`)
- `software_name_normalized` (str) — normalized form for comparison
- `version_spec` (str, nullable) — version specifier if present (e.g., `>=1.0,<2.0`)

### Table: `DocumentSoftwareMention`
Stores software mentions found in the accompanying academic paper (from SoftCite dataset).

- `id` (int, PK)
- `document_id` (FK → `document.id`)
- `software_name` (str) — mention text as found in paper
- `software_name_normalized` (str) — normalized form for comparison
- `mention_context` (str, nullable) — surrounding text or SoftCite mention ID for traceability

### Design rationale: three tables vs one

While the three-way comparison is the immediate use case, future analyses will likely focus on just one or two sources. Separate tables avoid loading unnecessary data and allow each table to evolve independently with source-specific columns.

## Files to Read

- `rs_graph/db/models.py` — existing SQLModel definitions; follow the same patterns (see `RepositoryLanguage`, `RepositoryFile`, `DocumentTopic` for similar FK-linked tables)
- `rs_graph/db/constants.py` — database path constants
- `rs_graph/db/utils.py` — `get_unique_first_model()` and `store_full_details()` patterns for upserting
- `notebooks/example-used-software-comparison.py` — the `normalize_name()` function (lines 228-245) defines how normalization works; the new tables should store the result of this normalization

## Files to Edit

- `rs_graph/db/models.py` — add the three new model classes

## Commands to Run

- `just lint` — ensure code passes linting and type checking

## Ask User to Run

- `just db-migrate` — create the Alembic migration
- `just db-upgrade` — apply migration to dev database
