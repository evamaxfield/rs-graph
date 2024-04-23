#!/usr/bin/env python

from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from prefect import Flow, unmapped
from prefect.task_runners import SequentialTaskRunner
from prefect_dask.task_runners import DaskTaskRunner

from rs_graph.db import utils as db_utils
from rs_graph.enrichment import entity_matching, github, open_alex
from rs_graph.sources import joss, plos, proto, softwarex
from rs_graph.types import SuccessAndErroredResultsLists
from rs_graph.utils import code_host_parsing

###############################################################################

app = typer.Typer()

DEFAULT_RESULTS_DIR = Path("processing-results")

###############################################################################


SOURCE_MAP: dict[str, proto.DatasetRetrievalFunction] = {
    "joss": joss.get_dataset,
    "plos": plos.get_dataset,
    "softwarex": softwarex.get_dataset,
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

    # If no successful results, raise error
    if len(results.successful_results) == 0:
        raise ValueError("No successful results to store.")

    return results


def _standard_ingest_flow(
    source: str,
    prod: bool,
) -> None:
    # Get dataset
    source_func = SOURCE_MAP[source]
    source_results = source_func()

    # Filter dataset
    filtered_results = code_host_parsing.filter_repo_paper_pairs(
        source_results.successful_results,
    )

    # Process open alex
    open_alex_futures = open_alex.process_open_alex_work_task.map(
        pair=filtered_results.successful_results,
    )

    # Process github
    github_futures = github.process_github_repo_task.map(
        pair=open_alex_futures,
    )

    # Store everything
    stored_futures = db_utils.store_full_details_task.map(
        pair=github_futures,
        prod=unmapped(prod),
    )

    # Match devs and researchers
    dev_researcher_futures = entity_matching.match_devs_and_researchers.map(
        pair=stored_futures,
    )

    # Store the dev-researcher links
    db_utils.store_dev_research_em_links_task.map(
        pair=dev_researcher_futures,
        prod=unmapped(prod),
    )


@app.command()
def standard_ingest(
    source: str,
    success_results_file: str = "",
    errored_results_file: str = "",
    prod: bool = False,
    use_dask: bool = False,
) -> None:
    """Get data from OpenAlex."""
    # Create current datetime without microseconds
    current_datetime = datetime.now().replace(microsecond=0)
    # Convert to isoformat and replace colons with dashes
    current_datetime_str = current_datetime.isoformat().replace(":", "-")

    # Create dir for this datetime
    current_datetime_dir = DEFAULT_RESULTS_DIR / current_datetime_str

    # If no filepaths provided, create them
    if len(success_results_file) == 0:
        # Create "results" dir
        current_datetime_dir.mkdir(exist_ok=True, parents=True)

        str(current_datetime_dir / f"process-results-{source}-success.parquet")
    else:
        pass

    if len(errored_results_file) == 0:
        # Create "results" dir
        current_datetime_dir.mkdir(exist_ok=True, parents=True)

        str(current_datetime_dir / f"process-results-{source}-errored.parquet")
    else:
        pass

    # If using dask, use DaskTaskRunner
    if use_dask:
        task_runner = DaskTaskRunner(
            cluster_class="distributed.LocalCluster",
            cluster_kwargs={"n_workers": 3, "threads_per_worker": 1},
        )
    else:
        task_runner = SequentialTaskRunner()

    # Create the flow
    ingest_flow = Flow(
        _standard_ingest_flow,
        name="ingest-flow",
        task_runner=task_runner,
        log_prints=True,
    )

    # Keep track of duration
    start_dt = datetime.now()
    start_dt = start_dt.replace(microsecond=0)

    # Start the flow
    ingest_flow(
        source=source,
        prod=prod,
    )

    # End duration
    end_dt = datetime.now()
    end_dt = end_dt.replace(microsecond=0)

    # Log time taken
    print(f"Total Processing Duration: {end_dt - start_dt}")


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
