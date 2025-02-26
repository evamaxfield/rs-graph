#!/usr/bin/env python

import logging
import os
from dataclasses import dataclass
from datetime import date

import typer
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from easyDataverse import Dataverse

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import DATA_FILES_DIR, GCS_PROJECT_ID, REMOTE_STORAGE_BUCKET

DATAVERSE_HOST = "https://dataverse.harvard.edu/"
DATAVERSE_RS_GRAPH_V1_DATASET_DOI = "10.7910/DVN/KPYVI1"

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


@app.command()
def upload(debug: bool = False) -> None:
    """
    Upload all local files to remote storage.

    Note: to use this command you need to have the
    `GOOGLE_APPLICATION_CREDENTIALS` environment variable set
    to the path to a service account key JSON file.
    """
    from gcsfs import GCSFileSystem

    # Setup logger
    setup_logger(debug=debug)

    # Load env
    load_dotenv()

    # Init GCSFileSystem
    fs = GCSFileSystem(project=GCS_PROJECT_ID)

    # Copy all files to remote and place within a folder with the current date
    current_date_str = date.today().isoformat()
    remote_storage_folder = f"{REMOTE_STORAGE_BUCKET}/{current_date_str}"
    log.info(f"Uploading all files from {DATA_FILES_DIR} to {remote_storage_folder}")
    local_files = [
        str(f.resolve()) for f in DATA_FILES_DIR.glob("*") if f.name != ".metadata.json"
    ]
    fs.put(local_files, remote_storage_folder)


###############################################################################


@dataclass
class CacheStateMetadata(DataClassJsonMixin):
    """Metadata for the cache state."""

    last_updated: str


def _gcs_download(force: bool = False) -> None:
    from gcsfs import GCSFileSystem

    # Load env
    load_dotenv()

    # Init GCSFileSystem
    fs = GCSFileSystem(project=GCS_PROJECT_ID)

    # Make the DATA_FILES_DIR if it doesn't exist
    DATA_FILES_DIR.mkdir(exist_ok=True)

    # Get the last updated date from the metadata file
    metadata_file = DATA_FILES_DIR / ".metadata.json"
    if metadata_file.exists():
        # Open and read
        with open(metadata_file) as open_f:
            metadata = CacheStateMetadata.from_json(open_f.read())

        # Get the last updated date
        last_updated = date.fromisoformat(metadata.last_updated)
    else:
        # No metadata file, so set the last updated date to the epoch
        # There will always be a remote folder with a newer date
        last_updated = date(1970, 1, 1)

    # Get all remote folders
    remote_folder_dates = [
        # Convert only the directory name to a date
        date.fromisoformat(d.split("/")[-1])
        for d in fs.ls(REMOTE_STORAGE_BUCKET)
        if "remote" not in d
    ]

    # Get most recent date
    most_recent_date = max(remote_folder_dates)

    # If the most recent date is newer than the last updated date,
    # then download the new files
    if most_recent_date > last_updated or force:
        # Get the remote folder
        remote_folder = f"{REMOTE_STORAGE_BUCKET}/{most_recent_date.isoformat()}"

        # Download all files
        log.info(f"Downloading all files from {remote_folder} to {DATA_FILES_DIR}")
        all_files = fs.ls(remote_folder)
        fs.get(all_files, str(DATA_FILES_DIR.resolve()))

        # Update the metadata file
        metadata = CacheStateMetadata(last_updated=most_recent_date.isoformat())
        with open(metadata_file, "w") as open_f:
            open_f.write(metadata.to_json())
    else:
        log.info("No new files to download")


def _dataverse_download(dataverse_token: str | None = None) -> None:
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
    dv = Dataverse(DATAVERSE_HOST, api_token=dataverse_token)

    # Download all files
    log.info(
        f"Downloading all files from "
        f"{DATAVERSE_RS_GRAPH_V1_DATASET_DOI} to {DATA_FILES_DIR}"
    )
    dv.load_dataset(
        pid=f"doi:{DATAVERSE_RS_GRAPH_V1_DATASET_DOI}",
        filedir=DATA_FILES_DIR,
        filenames=[
            "rs-graph-v1-prod.db",
            "rs-graph-v1-redacted.db",
        ],
    )


@app.command()
def download(
    dataverse_token: str = "",
    from_gcs: bool = False,
    force: bool = False,
    debug: bool = False,
) -> None:
    """
    Download all files from remote storage.

    Parameters
    ----------
    dataverse_token: str
        Your Dataverse API token
        (or none if you want to use the env variable DATAVERSE_TOKEN).
    from_gcs: bool
        Whether to download from Google Cloud Storage.
    force: bool
        Whether to force download all files
        (only for GCS)
    debug: bool
        Whether to enable debug logging.
    """
    # Setup logger
    setup_logger(debug=debug)

    # Route to GCS download
    if from_gcs:
        _gcs_download(force=force)
    else:
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
