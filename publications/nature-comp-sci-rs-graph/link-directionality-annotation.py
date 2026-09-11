#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import typer
import utils as u
from statsmodels.stats.inter_rater import aggregate_raters, cohens_kappa, fleiss_kappa

################################################################################

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATORS = ["eva", "sarah", "anna"]
ANNOTATION_COLUMNS = [
    "label",
    "this-repo-linked-in-paper",
    "this-article-linked-in-repo",
    "diff-repo-linked-in-paper",
    "diff-article-linked-in-repo",
]
FULL_TO_ANNOTATE_FILENAME_TEMPLATE = "link-directionality-to-annotate-{annotator}.csv"
TRAINING_ANNOTATION_FILENAME_PATH = DATA_DIR / "link-directionality-training-annotation-set.csv"
TRAINING_ANNOTATION_FILENAME_TEMPLATE = "link-directionality-training-annotated-{annotator}.csv"
TRAINING_ANNOTATION_FILENAMES = [
    DATA_DIR / TRAINING_ANNOTATION_FILENAME_TEMPLATE.format(annotator=annotator)
    for annotator in ANNOTATORS
]

app = typer.Typer()

################################################################################


@app.command()
def create_annotation_set() -> None:
    """Sample pairs per field and write the agreement + per-annotator annotation CSVs."""
    # Load dataset with top 5 fields (5 + Other)
    df = u.load_filtered_pairs(top_n_fields=5)

    # Filter to only rows with a predictive model confidence that isn't null
    df = df.filter(pl.col("predictive_model_confidence").is_not_null())

    # Iter top fields
    top_fields = df.get_column("document_field_name_pruned").unique().to_list()

    # We annotate 600 overall
    # 24 from training agreement set, 576 from all independent annotations
    agreement_subsets = []
    independent_subsets = []
    for field in top_fields:
        field_subset = df.filter(pl.col("document_field_name_pruned") == field)
        agreement_subset = field_subset.sample(
            n=4,
            seed=1,
        )
        independent_subset = field_subset.sample(
            n=96,
            seed=2,
        )
        agreement_subsets.append(agreement_subset)
        independent_subsets.append(independent_subset)

    # Prep for annotation with select column
    select_statements = [
        pl.col("document_repository_link_id"),
        pl.col("document_id"),
        pl.col("repository_id"),
        pl.col("predictive_model_confidence"),
        pl.col("document_field_name_pruned"),
        (pl.lit("https://doi.org/") + pl.col("document_doi")).alias("document_url"),
        (
            pl.lit("https://github.com/")
            + pl.col("repository_owner")
            + pl.lit("/")
            + pl.col("repository_name")
        ).alias("repository_url"),
        pl.lit(None).alias("label"),
        pl.lit(None).alias("this-repo-linked-in-paper"),
        pl.lit(None).alias("this-article-linked-in-repo"),
        pl.lit(None).alias("diff-repo-linked-in-paper"),
        pl.lit(None).alias("diff-article-linked-in-repo"),
        pl.lit(None).alias("notes"),
    ]

    # Combine subsets to single dataframes
    # Shuffle the rows of both
    agreement_df = (
        pl.concat(agreement_subsets)
        .select(*select_statements)
        .sample(fraction=1.0, seed=3, shuffle=True)
    )
    independent_df = (
        pl.concat(independent_subsets)
        .select(*select_statements)
        .sample(fraction=1.0, seed=4, shuffle=True)
    )

    # Save training set to be annotated by all annotators for agreement analysis
    agreement_df.write_csv(TRAINING_ANNOTATION_FILENAME_PATH)

    # Get the number of rows each annotator should annotate from the independent set
    num_annotators = len(ANNOTATORS)
    num_rows_per_annotator = len(independent_df) // num_annotators
    for i, annotator in enumerate(ANNOTATORS):
        start_idx = i * num_rows_per_annotator
        end_idx = (i + 1) * num_rows_per_annotator
        annotator_subset = independent_df[start_idx:end_idx]
        print(
            f"Annotator {annotator} assigned {len(annotator_subset)} rows for independent annotation."
        )

        # Save output CSV for this annotator
        output_path = DATA_DIR / FULL_TO_ANNOTATE_FILENAME_TEMPLATE.format(annotator=annotator)
        annotator_subset.write_csv(output_path)


def _pairwise_kappa_str(va: list[str | None], vb: list[str | None]) -> str:
    """Format Cohen's kappa (with N) for two annotators' labels, skipping unlabeled rows."""
    valid = [(a, b) for a, b in zip(va, vb, strict=False) if a is not None and b is not None]
    if len(valid) < 2:
        return "N/A"
    a_arr = np.array([v[0] for v in valid])
    b_arr = np.array([v[1] for v in valid])
    categories = sorted(set(a_arr) | set(b_arr))
    cat_idx = {c: i for i, c in enumerate(categories)}
    table = np.zeros((len(categories), len(categories)), dtype=int)
    for a, b in zip(a_arr, b_arr, strict=False):
        table[cat_idx[a], cat_idx[b]] += 1
    kappa = cohens_kappa(table).kappa
    kappa_str = f"{kappa:.3f}" if not np.isnan(kappa) else "N/A (trivially perfect)"
    return f"{kappa_str}  (N={len(valid)})"


def _fleiss_kappa_str(shared_rows: list[list[str]]) -> str:
    """Format Fleiss' kappa (with N) for the rows every annotator labeled."""
    if len(shared_rows) < 2:
        return "N/A (insufficient shared non-null rows)"
    # aggregate_raters expects an (N_subjects x N_raters) array of category indices
    categories = sorted({v for row in shared_rows for v in row})
    cat_idx = {c: j for j, c in enumerate(categories)}
    ratings_matrix = np.array([[cat_idx[v] for v in row] for row in shared_rows])
    table, _ = aggregate_raters(ratings_matrix)
    fk = fleiss_kappa(table)
    fk_str = f"{fk:.3f}" if not np.isnan(fk) else "N/A (trivially perfect)"
    return f"{fk_str}  (N={len(shared_rows)})"


@app.command()
def compare_annotation_sets() -> None:
    """Report inter-rater agreement and disagreements across the annotated training sets."""
    # Load all annotated CSVs
    # Inner join on the shared row identifier keeps only rows present in all files
    dfs = {
        annotator: pl.read_csv(
            DATA_DIR / TRAINING_ANNOTATION_FILENAME_TEMPLATE.format(annotator=annotator)
        )
        for annotator in ANNOTATORS
    }

    base = (
        dfs["eva"]
        .select(
            [
                "document_repository_link_id",
                "document_url",
                "repository_url",
                *ANNOTATION_COLUMNS,
            ]
        )
        .rename({col: f"{col}_eva" for col in ANNOTATION_COLUMNS})
    )

    for annotator in ["sarah", "anna"]:
        other = (
            dfs[annotator]
            .select(["document_repository_link_id", *ANNOTATION_COLUMNS])
            .rename({col: f"{col}_{annotator}" for col in ANNOTATION_COLUMNS})
        )
        base = base.join(other, on="document_repository_link_id", how="inner")

    doc_urls = base["document_url"].to_list()
    repo_urls = base["repository_url"].to_list()

    for col in ANNOTATION_COLUMNS:
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Column: {col}")
        typer.echo("=" * 60)

        # Normalize empty strings to None for consistent handling.
        annotator_vals: dict[str, list[str | None]] = {}
        for annotator in ANNOTATORS:
            raw = base[f"{col}_{annotator}"].to_list()
            annotator_vals[annotator] = [
                str(v).strip() if (v is not None and str(v).strip()) else None for v in raw
            ]

        # --- Pairwise Cohen's kappa ---
        typer.echo("\n--- Inter-Rater Agreement ---")
        typer.echo("Pairwise Cohen's Kappa:")
        pairs = [("eva", "sarah"), ("eva", "anna"), ("sarah", "anna")]
        for name_a, name_b in pairs:
            kappa_str = _pairwise_kappa_str(annotator_vals[name_a], annotator_vals[name_b])
            typer.echo(f"  {name_a} vs {name_b}: {kappa_str}")

        # --- Three-way Fleiss' kappa ---
        # Keep only rows every annotator labeled
        shared_rows: list[list[str]] = []
        for i in range(len(doc_urls)):
            row_vals = [annotator_vals[ann][i] for ann in ANNOTATORS]
            non_null = [v for v in row_vals if v is not None]
            if len(non_null) == len(ANNOTATORS):
                shared_rows.append(non_null)
        typer.echo(f"Three-way Fleiss' Kappa: {_fleiss_kappa_str(shared_rows)}")

        # --- Disagreements ---
        disagreements = [
            i
            for i in range(len(doc_urls))
            if len({annotator_vals[ann][i] for ann in ANNOTATORS}) > 1
        ]
        typer.echo(f"\n--- Disagreements ({len(disagreements)} rows) ---")
        for i in disagreements:
            typer.echo(f"- document doi: {doc_urls[i]}")
            typer.echo(f"- repository url: {repo_urls[i]}")
            for ann in ANNOTATORS:
                v = annotator_vals[ann][i]
                typer.echo(f"\t- {ann}: {v if v is not None else '(empty)'}")
            typer.echo("")


if __name__ == "__main__":
    app()
