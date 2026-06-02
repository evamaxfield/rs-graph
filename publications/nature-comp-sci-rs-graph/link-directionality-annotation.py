#!/usr/bin/env python3

from __future__ import annotations

import polars as pl
import typer
from data_utils import DATA_DIR, load_base_dataset

################################################################################

ANNOTATORS = ["eva", "sarah", "anna"]
FULL_TO_ANNOTATE_FILENAME_TEMPLATE = "link-directionality-to-annotate-{annotator}.csv"
TRAINING_ANNOTATION_FILENAME_PATH = DATA_DIR / "link-directionality-training-annotation-set.csv"

app = typer.Typer()

################################################################################


@app.command()
def create_annotation_set() -> None:
    # Load dataset with top 5 fields (5 + Other)
    df = load_base_dataset(top_n_fields=5)

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


if __name__ == "__main__":
    app()
