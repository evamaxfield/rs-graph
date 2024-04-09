#!/usr/bin/env python

import logging
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import typer
from distributed import LocalCluster

from rs_graph.bin.typer_utils import setup_logger
from rs_graph.enrichment import open_alex
from rs_graph.sources.joss import JOSSDataSource
from rs_graph.sources.plos import PLOSDataSource
from rs_graph.sources.proto import DataSource
from rs_graph.sources.softwarex import SoftwareXDataSource
from rs_graph.types import SuccessAndErroredResultsLists
from rs_graph.utils import code_host_parsing

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

DEFAULT_RESULTS_DIR = Path("processing-results")

###############################################################################


SOURCE_MAP: dict[str, type[DataSource]] = {
    "joss": JOSSDataSource,
    "plos": PLOSDataSource,
    "softwarex": SoftwareXDataSource,
}


def _split_and_store_results(
    new_results: SuccessAndErroredResultsLists,
    old_results: SuccessAndErroredResultsLists,
    success_results_filepath: str,
    errored_results_filepath: str,
) -> SuccessAndErroredResultsLists:
    # Combine results
    results = SuccessAndErroredResultsLists(
        successful_results=new_results.successful_results,
        errored_results=old_results.errored_results + new_results.errored_results,
    )

    # Split and store results
    success_results = pd.DataFrame([r.to_dict() for r in results.successful_results])
    success_results.to_parquet(success_results_filepath)
    errored_results = pd.DataFrame([r.to_dict() for r in results.errored_results])
    errored_results.to_parquet(errored_results_filepath)

    return results


@app.command()
def process(
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

    # Create current datetime without microseconds
    current_datetime = datetime.now().replace(microsecond=0)
    # Convert to isoformat and replace colons with dashes
    current_datetime_str = current_datetime.isoformat().replace(":", "-")

    # If no filepaths provided, create them
    if len(success_results_file) == 0:
        # Create "results" dir
        DEFAULT_RESULTS_DIR.mkdir(exist_ok=True)

        success_results_filepath = str(
            DEFAULT_RESULTS_DIR
            / f"process-results-{source}-{current_datetime_str}-success.parquet"
        )
    else:
        success_results_filepath = success_results_file

    if len(errored_results_file) == 0:
        # Create "results" dir
        DEFAULT_RESULTS_DIR.mkdir(exist_ok=True)

        errored_results_filepath = str(
            DEFAULT_RESULTS_DIR
            / f"process-results-{source}-{current_datetime_str}-errored.parquet"
        )
    else:
        errored_results_filepath = errored_results_file

    # Split and store partial
    split_and_store_results_partial = partial(
        _split_and_store_results,
        success_results_filepath=success_results_filepath,
        errored_results_filepath=errored_results_filepath,
    )

    # Create dask cluster and client
    if use_dask:
        cluster = LocalCluster(
            processes=True,
            n_workers=4,
            threads_per_worker=1,
        )

        # Log dask dashboard link
        log.info(f"Dask dashboard link: {cluster.dashboard_link}")

        cluster_address = cluster.scheduler_address
    else:
        cluster_address = None

    # Create dask client and cluster
    get_dataset_results = SOURCE_MAP[source].get_dataset(
        cluster_address=cluster_address,
    )
    split_and_store_results_partial(
        new_results=get_dataset_results,
        old_results=SuccessAndErroredResultsLists([], []),
    )

    # Filter out non-GitHub Repo pairs
    code_filtering_results = code_host_parsing.filter_repo_paper_pairs(
        get_dataset_results.successful_results,
        cluster_address=cluster_address,
    )
    split_and_store_results_partial(
        new_results=code_filtering_results,
        old_results=get_dataset_results,
    )

    # Process with open_alex
    open_alex_processing_results = open_alex.process_pairs(
        code_filtering_results.successful_results,
        prod=prod,
        cluster_address=cluster_address,
    )
    split_and_store_results_partial(
        new_results=open_alex_processing_results,
        old_results=code_filtering_results,
    )


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
