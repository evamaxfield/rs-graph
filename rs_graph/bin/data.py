#!/usr/bin/env python

import logging

import typer

from rs_graph.data.joss import get_joss_dataset
from rs_graph.bin.typer_utils import setup_logger

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################

@app.command()
def joss(
    output_filepath: str = "joss-short-paper-details.parquet",
    start_page: int = 1,
    debug: bool = False,
) -> None:
    """Download the JOSS dataset."""
    # Setup logger
    setup_logger(debug=debug)

    # Download JOSS dataset
    final_stored_dataset = get_joss_dataset(
        output_filepath=output_filepath,
        start_page=start_page,
    )
    log.info(f"Stored JOSS dataset to: '{final_stored_dataset}'")

@app.command()
def f1000(
    output_filepath: str = "f1000-short-paper-details.parquet",
    debug: bool = False,
) -> None:
    """Download the F1000 dataset."""
    # Setup logger
    setup_logger(debug=debug)

    raise NotImplementedError()

###############################################################################

def main() -> None:
    app()

if __name__ == "__main__":
    app()