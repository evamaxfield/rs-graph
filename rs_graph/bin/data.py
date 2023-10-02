#!/usr/bin/env python

import logging
import shutil
from datetime import datetime

import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data.joss import get_joss_dataset

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


@app.command()
def joss(
    output_filepath: str = "joss-short-paper-details.parquet",
    start_page: int = 1,
    copy_to_lib: bool = False,
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

    # Copy the final to the repo / library
    if copy_to_lib:
        current_date_str = datetime.now().date().isoformat()
        lib_storage_path = f"rs_graph/data/files/joss-{current_date_str}.parquet"
        shutil.copy2(
            final_stored_dataset,
            lib_storage_path,
        )
        log.info(f"Copied JOSS dataset to: '{lib_storage_path}'")


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
