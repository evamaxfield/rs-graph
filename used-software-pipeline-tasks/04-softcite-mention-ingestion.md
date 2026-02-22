# Task 4: SoftCite Mention Ingestion

## Overall Project Context

RQ3 of the rs-graph research project asks: **how do three observable views of scientific software use — mentions in articles, imports in code, and dependency manifests — overlap and diverge?**

SoftCite provides software mention annotations for academic papers. Currently, the prototype loads these mentions from parquet files at analysis time. This task ingests them into the `DocumentSoftwareMention` database table (created in Task 1) so all three software sources are queryable from one place.

## End Goal

SoftCite software mentions are stored in the `DocumentSoftwareMention` table, linked to their corresponding `document_id` records via DOI matching. The ingestion is idempotent (can be re-run without creating duplicates).

## This Task

### Ingestion flow

1. Load SoftCite parquet files:
   - `papers.parquet` — contains `paper_id` and `doi`
   - `mentions.pdf.parquet` — contains `software_mention_id`, `paper_id`, `software_normalized`
2. Join mentions to papers to get DOIs
3. Normalize DOIs using the shared `_normalize_doi_expr()` function
4. Match DOIs to existing `document` records in the rs-graph database
5. Look up the corresponding `document` records (also check the `DocumentAlternateDOI` table)
6. For each mention:
   - `software_name` = the `software_normalized` value from SoftCite
   - `software_name_normalized` = result of `normalize_name(software_name)`
   - `mention_context` = the `software_mention_id` from SoftCite (for traceability back to the original annotation)
7. Store in `DocumentSoftwareMention` table

### Key design decisions

- **DOI-based linkage**: Mentions are linked to document-repository pairs via DOI matching, same approach as the prototype
- **Idempotent ingestion**: Check for existing records before inserting to support re-runs
- **One mention per row**: Each SoftCite mention becomes one row, even if the same software is mentioned multiple times in the same paper (this preserves the granularity of the SoftCite annotations)
- **Normalization**: Use the same `normalize_name()` function as imports and dependencies for consistency

### Considerations

- Some SoftCite mentions may be generic terms (e.g., "code", "script", "library"). These should still be ingested — filtering happens at analysis time using the `MENTION_EXCLUDE_NORMALIZED` set from the prototype.

## Files to Read

- `notebooks/example-used-software-comparison.py` — the prototype's mention loading logic:
  - `compare_imported_vs_mentioned()` (lines 786-1032) — how mentions are currently loaded and joined
  - `_normalize_doi_expr()` (lines 55-72) — DOI normalization
  - `normalize_name()` (lines 228-245) — software name normalization
  - `MENTION_EXCLUDE_NORMALIZED` (lines 28-39) — exclusion set (for reference, not for ingestion filtering)
- `rs_graph/sources/softcite_2025.py` — existing SoftCite data source adapter; understand how SoftCite data is already used in the project
- `rs_graph/db/models.py` — the `DocumentSoftwareMention` table from Task 1, plus `Document` model
- `rs_graph/db/utils.py` — DB session utilities and upsert patterns

## Files to Create or Edit

- Add as a Typer command in `rs_graph/bin/used_software_extraction.py` (created in Task 2), OR create a separate script if Task 2 is not yet complete

## Commands to Run

- `just lint` — ensure code passes linting and type checking
- Test with a small subset of SoftCite data to verify DOI matching and storage
