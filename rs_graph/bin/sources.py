#!/usr/bin/env python

import logging
import pkgutil

import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import DATA_FILES_DIR, DATASET_SOURCE_FILE_PATTERN, sources

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

ALL_DATASET_SOURCE_MODULES = [i.name for i in pkgutil.iter_modules(sources.__path__)]

###############################################################################

app = typer.Typer()

###############################################################################


def _get_single_dataset(source: str) -> None:
    """Download a single dataset."""
    # Get the dataset loader function from the module
    source_module = getattr(sources, source)

    # Download the dataset
    final_stored_dataset = source_module.get_dataset(
        output_filepath=DATA_FILES_DIR / f"{source}{DATASET_SOURCE_FILE_PATTERN}",
    )
    log.info(f"Stored {source} dataset to: '{final_stored_dataset}'")


@app.command()
def get_single(source: str, debug: bool = False) -> None:
    """Download the JOSS dataset."""
    # Setup logger
    setup_logger(debug=debug)

    # Run download
    _get_single_dataset(source)


###############################################################################


@app.command()
def get_all(debug: bool = False) -> None:
    """Download all source datasets."""
    # Setup logger
    setup_logger(debug=debug)

    # Run download
    for source in ALL_DATASET_SOURCE_MODULES:
        _get_single_dataset(source)


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
