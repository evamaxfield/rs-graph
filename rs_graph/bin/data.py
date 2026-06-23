#!/usr/bin/env python

import logging
import os

import typer
from dotenv import load_dotenv

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import DATA_FILES_DIR

DATAVERSE_HOST = "https://dataverse.harvard.edu/"
DATAVERSE_RS_GRAPH_V1_DATASET_DOI = "10.7910/DVN/KPYVI1"

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


def _dataverse_download(dataverse_token: str | None = None) -> None:
    from easyDataverse import Dataverse

    # Handle token
    if dataverse_token is None:
        # Load env
        load_dotenv()

        # Get token
        dataverse_token = os.getenv("DATAVERSE_TOKEN", None)

    # Raise is no token
    if dataverse_token is None:
        raise ValueError(
            "No Dataverse token provided. "
            "Please provide one via --dataverse-token or set the "
            "DATAVERSE_TOKEN env variable."
        )

    # Init Dataverse
    dv = Dataverse(
        DATAVERSE_HOST,  # type: ignore[arg-type]
        api_token=dataverse_token,  # type: ignore[arg-type]
    )

    # Download all files
    log.info(
        f"Downloading all files from {DATAVERSE_RS_GRAPH_V1_DATASET_DOI} to {DATA_FILES_DIR}"
    )
    dv.load_dataset(
        pid=f"doi:{DATAVERSE_RS_GRAPH_V1_DATASET_DOI}",
        filedir=str(DATA_FILES_DIR),
        filenames=[
            "rs-graph-v1-prod.db",
            "rs-graph-v1-redacted.db",
        ],
    )


@app.command()
def download(
    dataverse_token: str = "",
    debug: bool = False,
) -> None:
    """
    Download all files from the published Dataverse dataset.

    Parameters
    ----------
    dataverse_token: str
        Your Dataverse API token
        (or none if you want to use the env variable DATAVERSE_TOKEN).
    debug: bool
        Whether to enable debug logging.
    """
    # Setup logger
    setup_logger(debug=debug)

    # Update token to None if empty
    if len(dataverse_token) == 0:
        resolved_dataverse_token = None
    else:
        resolved_dataverse_token = dataverse_token

    _dataverse_download(dataverse_token=resolved_dataverse_token)


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
