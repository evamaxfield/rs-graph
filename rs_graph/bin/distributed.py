#!/usr/bin/env python

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import coiled
import typer
from dotenv import load_dotenv

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.data import REMOTE_STORAGE_BUCKET, sources

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@dataclass
class DatasetSource:
    name: str
    required_parameters: list[str]
    result_filename: str


ALL_DATASET_SOURCES_DETAILS_LUT = {
    "joss": DatasetSource(
        name="joss",
        required_parameters=[],
        result_filename="data-source-joss.parquet",
    ),
    "softwarex": DatasetSource(
        name="softwarex",
        required_parameters=[
            "github_api_keys",
            "elsevier_api_key",
            "use_distributed",
        ],
        result_filename="data-source-softwarex.parquet",
    ),
}

###############################################################################

app = typer.Typer()

###############################################################################


def _get_dataset_sources_download_functions(
    agg_params: dict[str, Any],
    remote_storage_prefix: str,
) -> list[partial]:
    """
    Download all source datasets.

    Parameters
    ----------
    agg_params: dict[str, Any]
        The preprocessed global parameters to pass to various functions
    remote_storage_prefix: str
        The prefix to add to the stored datasets.

    Returns
    -------
    tasks: list[partial]
        The individual get dataset functions ready to run.
    """
    # Get all dataset sources
    partial_funcs = []
    for source_name, source_details in ALL_DATASET_SOURCES_DETAILS_LUT.items():
        # Get the dataset loader function from the module
        source_module = getattr(sources, source_name)

        # Create full path
        this_func_out_fp = f"{remote_storage_prefix}/{source_details.result_filename}"

        # Get the dataset
        partial_funcs.append(
            partial(
                source_module.get_dataset,
                output_filepath=this_func_out_fp,
                **{
                    k: v
                    for k, v in agg_params.items()
                    if k in source_details.required_parameters
                },
            )
        )

    return partial_funcs


@app.command()
def get_all_dataset_sources(
    gh_api_keys_file: str = ".github-tokens.json",
    remote_storage_bucket: str = REMOTE_STORAGE_BUCKET,
    remote_storage_prefix: str = "",
    debug: bool = False,
) -> None:
    """
    Download all source datasets.

    Parameters
    ----------
    gh_api_keys_file: str
        The path to the JSON file which stores a list of GitHub PATs.
    remote_storage_bucket: str
        The cloud storage bucket to store the datasets in.
    remote_storage_prefix: str
        The prefix to add to the stored datasets.
    debug: bool
        Whether or not to run in debug mode.
    """
    # Handle debug
    setup_logger(debug=debug)

    # Very github api keys file
    if not Path(gh_api_keys_file).exists():
        raise FileNotFoundError(f"GitHub API keys file not found: '{gh_api_keys_file}'")

    # Determine prefix
    if len(remote_storage_prefix) == 0:
        utcnow = datetime.utcnow().replace(microsecond=0).isoformat()
        remote_storage_prefix = f"distributed-{utcnow}".replace(":", "-")
    elif remote_storage_prefix.startswith("/"):
        remote_storage_prefix = remote_storage_prefix.strip("/")
    elif remote_storage_prefix.endswith("/"):
        remote_storage_prefix = remote_storage_prefix.rstrip("/")

    # Handle final path
    full_storage_prefix = f"{remote_storage_bucket}/sources/{remote_storage_prefix}"
    log.info(f"Storing datasets to: '{full_storage_prefix}'")

    # Load global parameters
    log.info("Loading global parameters")
    agg_params = {}

    # Load github api keys
    with open(gh_api_keys_file) as f:
        agg_params["github_api_keys"] = json.load(f)
        log.info(f"Loaded {len(agg_params['github_api_keys'])} GitHub API keys")

    # Load dotenv and get elsevier api key
    load_dotenv()
    agg_params["elsevier_api_key"] = os.getenv("ELSEVIER_API_KEY")

    # Add use_distributed flag
    agg_params["use_distributed"] = True

    # Get the list of functions to submit to cluster
    partial_funcs = _get_dataset_sources_download_functions(
        agg_params=agg_params,
        remote_storage_prefix=full_storage_prefix,
    )

    # Create flow
    log.info("Creating GCP cluster")
    with coiled.Cluster(
        n_workers=1,
        worker_vm_types=["n1-standard-1"],
        container="ghcr.io/evamaxfield/rs-graph:distributed",
    ) as cluster:
        # with LocalCluster(
        #     n_workers=len(agg_params["github_api_keys"]),
        #     threads_per_worker=1,
        #     processes=False,
        # ) as cluster:
        # Set up adaptive
        log.info("Setting up adaptive")
        cluster.adapt(minimum=1, maximum=len(agg_params["github_api_keys"]))

        # Log dashboard address
        log.info(f"Dask dashboard address: {cluster.dashboard_link}")

        # Get client
        client = cluster.get_client()

        # Submit all funcs
        log.info("Submitting all tasks")
        tasks = [client.submit(func) for func in partial_funcs]

        # Gather / wait for all
        log.info("Gathering all tasks")
        client.gather(tasks)


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    main()
