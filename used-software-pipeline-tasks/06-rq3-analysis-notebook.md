# Task 6: RQ3 Analysis Notebook

## Overall Project Context

RQ3 of the rs-graph research project asks: **how do three observable views of scientific software use — mentions in articles, imports in code, and dependency manifests — overlap and diverge?**

After Tasks 1-5 have populated the database with imports, dependencies, and mentions, and provided alignment utilities, this task creates the analysis notebook that produces the RQ3 findings.

## End Goal

A complete analysis notebook that:
1. Queries all three software tables from the database
2. Runs three-way pairwise alignment and reconciliation per document-repository pair
3. Produces summary statistics, visualizations, and breakdowns answering RQ3
4. Follows the conventions of existing analysis notebooks (`rq1-analysis.py`, `rq2-analysis.py`)

## This Task

### Key analyses (informed by the proposal)

1. **Overall overlap/divergence**
   - For each document-repository pair: what fraction of software appears in all three sources, two sources, or only one?
   - Aggregate: distribution of overlap categories across the full dataset
   - Visualization: Venn diagram or UpSet plot showing three-way overlap

2. **Breakdown by metadata**
   - By domain and field (OpenAlex topics)
   - By programming language
   - By dataset source (JOSS, PLOS, PWC, SoftwareX, SoftCite)
   - By publication year
   - By country affiliation

3. **Top software by overlap category**
   - Most common software that appears in all three sources
   - Most common import-only software (used but not declared or mentioned)
   - Most common mention-only software (cited in paper but not found in code)
   - Most common dependency-only software (declared but not directly imported — transitive deps, build tools, etc.)

4. **Systematic biases**
   - Are certain types of software systematically missed by certain views?
   - Do mentions favor "marquee" software while imports capture utility libraries?
   - Do dependency manifests include framework/tooling software not captured by imports or mentions?

5. **Match quality analysis**
   - Distribution of pairwise match scores
   - Cases where transitive closure resolved matches that direct comparison missed

### Reuse from prototype

The prototype's `compare_imported_vs_mentioned` command (lines 786-1032) has significant analysis code that can be adapted:
- Grouped statistics computation (lines 994-1031)
- Top-N analysis functions (`_get_top_items_by_status`, `_print_top_n_analysis`)
- `MENTION_EXCLUDE_NORMALIZED` filtering for generic terms

The key difference: extend from two-way (import vs mention) to three-way comparison.

## Files to Read

- `notebooks/example-used-software-comparison.py` — prototype analysis logic to adapt
- `notebooks/rq1-analysis.py` — conventions for analysis notebooks in this project
- `notebooks/rq2-analysis.py` — another reference for notebook structure and style
- `rs_graph/db/models.py` — the three software tables (from Task 1) plus metadata tables
- `rs_graph/db/constants.py` — database path constants
- `rs_graph/utils/software_alignment.py` — three-way alignment utilities (from Task 5)
- `rs_graph/utils/normalization.py` — shared normalization functions (from Task 2)
- `notebooks/proposal-draft.md` — RQ3 description and expected analyses

## Files to Create

- `notebooks/rq3-analysis.py` — Jupyter-compatible analysis script (percent-format, matching existing conventions)

## Commands to Run

- `just lint` — ensure code passes linting and type checking
- Run the notebook end-to-end and review outputs for correctness and completeness
