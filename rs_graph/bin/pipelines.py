#!/usr/bin/env python

import logging

import pandas as pd
from prefect import flow, serve, task
from prefect_dask.task_runners import DaskTaskRunner

from rs_graph.enrichment import open_alex
from rs_graph.sources.joss import JOSSDataSource
from rs_graph.sources.proto import RepositoryDocumentPair

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


SOURCE_MAP = {
    "joss": JOSSDataSource,
}


@task
def get_data(source: str) -> list[RepositoryDocumentPair]:
    return SOURCE_MAP[source].get_dataset()


@flow(
    task_runner=DaskTaskRunner(
        # Make threaded
        cluster_kwargs={"n_workers": 1, "threads_per_worker": 4, "processes": False},
    ),
)
def _process_papers(
    source: str,
    success_results_file: str | None = None,
    errored_results_file: str | None = None,
    prod: bool = False,
) -> None:
    """Get data from OpenAlex."""
    # If no filepaths provided, create them
    if success_results_file is None:
        success_results_file = f"{source}_success_results.parquet"
    if errored_results_file is None:
        errored_results_file = f"{source}_errored_results.parquet"

    # Get the dataset
    dataset = get_data(source=source)

    # Process with open_alex
    results = open_alex.process_pairs(
        pairs=dataset,
        prod=prod,
    )

    # Store to files
    success_results = pd.DataFrame([r.to_dict() for r in results.successful_results])
    success_results.to_parquet(success_results_file)
    errored_results = pd.DataFrame([r.to_dict() for r in results.errored_results])
    errored_results.to_parquet(errored_results_file)


def serve_pipelines() -> None:
    """Serve the pipelines."""
    open_alex_processing_deploy = _process_papers.to_deployment(name="Paper Processing")
    serve(open_alex_processing_deploy)


###############################################################################


def main() -> None:
    serve_pipelines()


if __name__ == "__main__":
    serve_pipelines()
