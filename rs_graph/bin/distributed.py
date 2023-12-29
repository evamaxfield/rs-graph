#!/usr/bin/env python

from pathlib import Path
import json
import logging
from datetime import datetime
from dataclasses import dataclass

import typer
from prefect import task, Flow
from dotenv import load_dotenv
import os

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import sources, REMOTE_STORAGE_BUCKET
from dask_cloudprovider.gcp import GCPCluster
from prefect_dask.task_runners import DaskTaskRunner

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

@dataclass
class DatasetSource:
    name: str
    required_parameters: list[str]

ALL_DATASET_SOURCES_DETAILS_LUT = {
    "joss": DatasetSource(
        name="joss",
        required_parameters=[],
    ),
    "softwarex": DatasetSource(
        name="softwarex",
        required_parameters=[
            "github_api_keys",
            "elsevier_api_key",
        ],
    ),
}

###############################################################################

app = typer.Typer()

###############################################################################

def _get_all_dataset_sources(
    gcp_project_id: str = "sci-software-graph",
    gh_api_keys_file: str = ".github-tokens.json",
    remote_storage_bucket: str = REMOTE_STORAGE_BUCKET,
    remote_storage_prefix: str = "",
    debug: bool = False,
) -> None:
    """
    Download all source datasets.
    
    Parameters
    ----------
    gcp_project_id: str
        The GCP project ID to use.
    gh_api_keys_file: str
        The path to the JSON file which stores a list of GitHub PATs.
    remote_storage_bucket: str
        The cloud storage bucket to store the datasets in.
    remote_storage_prefix: str
        The prefix to add to the stored datasets.
    debug: bool
        Whether or not to run in debug mode.
    """
    # Very github api keys file
    if not Path(gh_api_keys_file).exists():
        raise FileNotFoundError(f"GitHub API keys file not found: '{gh_api_keys_file}'")

    # Handle debug
    setup_logger(debug=debug)

    # Determine prefix
    if len(remote_storage_prefix) == 0:
        utcnow = datetime.utcnow().replace(microsecond=0).isoformat()
        remote_storage_prefix = "distributed-{utcnow}"
    elif remote_storage_prefix.startswith("/"):
        remote_storage_prefix = remote_storage_prefix.strip("/")
    elif remote_storage_prefix.endswith("/"):
        remote_storage_prefix = remote_storage_prefix.rstrip("/")

    # Handle final path
    remote_storage_path = f"{remote_storage_bucket}/sources/{remote_storage_prefix}/"
    log.info(f"Storing datasets to: '{remote_storage_path}'")

    # Load global parameters
    log.info("Loading global parameters")
    global_parameters = {}

    # Load github api keys
    with open(gh_api_keys_file) as f:
        global_parameters["github_api_keys"] = json.load(f)
    
    # Load dotenv and get elsevier api key
    load_dotenv()
    global_parameters["elsevier_api_key"] = os.getenv("ELSEVIER_API_KEY")

    # Create dask cloudprovider gcp
    log.info("Creating GCP cluster")
    cluster = GCPCluster(
        project=gcp_project_id,
        zone="us-central1-a",
        n_workers=1,
        machine_type="n1-standard-1",
        preemptible=True,
        docker_image="gcr.io/evamaxfield/rs-graph:latest",
    )

    # Adaptive between 1 and the number of github api keys
    cluster.adapt(
        minimum=1,
        maximum=len(global_parameters["github_api_keys"]),
    )

    # Log cluster dashboard
    log.info(f"Cluster dashboard: {cluster.dashboard_link}")

    # Setup prefect
    stored_datasets = {}
    with Flow(
        "get_all_dataset_sources",
        task_runner=DaskTaskRunner(cluster=cluster),
    ) as flow:
        # Get all dataset sources
        for source_name, source_details in ALL_DATASET_SOURCES_DETAILS_LUT.items():
            # Get the dataset loader function from the module
            source_module = getattr(sources, source_name)

            # Get the dataset
            dataset_path = source_module.get_dataset(
                output_filepath=remote_storage_path,
                **{
                    k: v
                    for k, v in global_parameters.items()
                    if k in source_details.required_parameters
                },
            )
            stored_datasets[source_name] = dataset_path

    # Run flow
    flow.run()

    # Return stored datasets
    return stored_datasets

@app.command()
def get_all_dataset_sources(
    gcp_project_id: str = "sci-software-graph",
    gh_api_keys_file: str = ".github-tokens.json",
    remote_storage_bucket: str = REMOTE_STORAGE_BUCKET,
    remote_storage_prefix: str = "",
    debug: bool = False,
) -> None:
    """
    Download all source datasets.
    
    Parameters
    ----------
    gcp_project_id: str
        The GCP project ID to use.
    gh_api_keys_file: str
        The path to the JSON file which stores a list of GitHub PATs.
    remote_storage_bucket: str
        The cloud storage bucket to store the datasets in.
    remote_storage_prefix: str
        The prefix to add to the stored datasets.
    debug: bool
        Whether or not to run in debug mode.
    """
    _get_all_dataset_sources(
        gcp_project_id=gcp_project_id,
        gh_api_keys_file=gh_api_keys_file,
        remote_storage_bucket=remote_storage_bucket,
        remote_storage_prefix=remote_storage_prefix,
        debug=debug,
    )

###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
