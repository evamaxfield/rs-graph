#!/usr/bin/env python

import logging
from pathlib import Path

import pandas as pd
import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import DATA_FILES_DIR
from rs_graph.enrichment import open_alex
from rs_graph.sources.joss import JOSSDataSource
from rs_graph.sources.plos import PLOSDataSource
from rs_graph.sources.softwarex import SoftwareXDataSource

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


SOURCE_MAP = {
    "joss": JOSSDataSource,
    "plos": PLOSDataSource,
    "softwarex": SoftwareXDataSource,
}


@app.command()
def process_papers(
    source: str,
    success_results_file: str = "",
    errored_results_file: str = "",
    prod: bool = False,
    use_dask: bool = False,
    debug: bool = False,
) -> None:
    """Get data from OpenAlex."""
    # Setup logger
    setup_logger(debug=debug)

    # If no filepaths provided, create them
    if len(success_results_file) == 0:
        success_results_filepath = (
            DATA_FILES_DIR / f"{source}-paper-processing-success-results.parquet"
        )
    else:
        success_results_filepath = Path(success_results_file)

    if len(errored_results_file) == 0:
        errored_results_filepath = (
            DATA_FILES_DIR / f"{source}-paper-processing-errored-results.parquet"
        )
    else:
        errored_results_filepath = Path(errored_results_file)

    # Create dask client and cluster
    pairs = SOURCE_MAP[source].get_dataset(
        use_dask=use_dask,
    )

    # Process with open_alex
    results = open_alex.process_pairs(pairs, prod=prod)

    # Store to files
    success_results = pd.DataFrame([r.to_dict() for r in results.successful_results])
    success_results.to_parquet(success_results_filepath)
    errored_results = pd.DataFrame([r.to_dict() for r in results.errored_results])
    errored_results.to_parquet(errored_results_filepath)


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()