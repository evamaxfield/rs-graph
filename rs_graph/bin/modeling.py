#!/usr/bin/env python

import logging

import dedupe
import pandas as pd
import numpy as np
import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import DATA_FILES_DIR, load_rs_graph_author_contributions_dataset

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################

def _cosine_diff_or_none_comparator(
    a: np.ndarray | None,
    b: np.ndarray | None,
) -> float:
    if a is None or b is None:
        return 0.5  # "random"
    
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.command()
def train_author_deduper(debug: bool = False) -> None:
    # Setup logging
    setup_logger(debug=debug)

    # Load the author contributions dataset
    author_contributions = load_rs_graph_author_contributions_dataset()

    # Format as records dict
    author_records = {}
    for _, row in author_contributions.sample(500).iterrows():
        # Get all co authors from all contributions
        co_authors = set()
        for contribution in row.contributions:
            co_authors.update(set(contribution["co_authors"]))

        # Get all embeddings of contributed papers as np arrays
        contribution_embeddings = []
        for contribution in row.contributions:
            if contribution["embedding"] is not None:
                contribution_embeddings.append(np.array(contribution["embedding"]))

        # Take mean
        if len(contribution_embeddings) > 0:
            mean_embedding = np.mean(np.stack(contribution_embeddings), axis=0)
        else:
            mean_embedding = None

        # Create new record
        author_records[row.author_id] = {
            "name": row["name"],
            "aliases": tuple(row.aliases),
            "co_authors": tuple(co_authors),
            "mean_embedding": mean_embedding,
        }
    
    # Variables for dedupe
    variables = [
        {"field": "name", "type": "Name"},
        {"field": "aliases", "type": "Set"},
        {"field": "co_authors", "type": "Set"},
        {
            "field": "mean_embedding",
            "type": "Custom",
            "comparator": _cosine_diff_or_none_comparator,
        },
    ]

    # Create deduper
    deduper = dedupe.Dedupe(variables)

    # Create training pairs
    deduper.prepare_training(author_records, sample_size=400)

    # Start training
    dedupe.console_label(deduper)


@app.command()
def test():
    print("blah")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
