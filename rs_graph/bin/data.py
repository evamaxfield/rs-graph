#!/usr/bin/env python

import logging
from pathlib import Path

import typer

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import DATA_FILES_DIR

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


@app.command()
def upload(debug: bool = False) -> None:
    """Upload all local files to remote storage."""
    # Setup logger
    setup_logger(debug=debug)

    print("TODO: upload")


###############################################################################


@app.command()
def download(debug: bool = False) -> None:
    """Download all files from remote storage."""
    # Setup logger
    setup_logger(debug=debug)

    print("TODO: download")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
