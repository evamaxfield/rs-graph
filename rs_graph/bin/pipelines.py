#!/usr/bin/env python

import math
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from prefect import Flow, get_run_logger, unmapped
from prefect.task_runners import SequentialTaskRunner
from prefect_dask.task_runners import DaskTaskRunner
from tqdm import tqdm

from rs_graph import types
from rs_graph.db import utils as db_utils
from rs_graph.enrichment import entity_matching, github, open_alex
from rs_graph.sources import joss, plos, proto, pwc
from rs_graph.utils import code_host_parsing

###############################################################################

app = typer.Typer()

DEFAULT_RESULTS_DIR = Path("processing-results")

###############################################################################


SOURCE_MAP: dict[str, proto.DatasetRetrievalFunction] = {
    "joss": joss.get_dataset,
    "plos": plos.get_dataset,
    # "softwarex": softwarex.get_dataset,
    "pwc": pwc.get_dataset,
}


def _store_errored_results(
    results: list[types.StoredRepositoryDocumentPair | types.ErrorResult],
    store_path: Path,
) -> None:
    # Get only errors
    errored_results = [
        result for result in results if isinstance(result, types.ErrorResult)
    ]

    # Store errored results
    errored_df = pd.DataFrame(errored_results)
    errored_df.to_parquet(store_path)


def _prelinked_dataset_ingestion_flow(
    source: str,
    batch_size: int,
    prod: bool,
    errored_store_path: Path,
) -> None:
    # Get dataset
    source_func = SOURCE_MAP[source]
    source_results = source_func()

    # Filter dataset
    filtered_results = code_host_parsing.filter_repo_paper_pairs(
        source_results.successful_results,
    )

    # Set logging to Error
    logger = get_run_logger()
    logger.setLevel("ERROR")

    # Create chunks of batch_size of the results to process
    n_batches = math.ceil(len(filtered_results.successful_results) / batch_size)
    for i in tqdm(
        range(0, len(filtered_results.successful_results), batch_size),
        desc="Batches",
        total=n_batches,
    ):
        chunk = filtered_results.successful_results[i : i + batch_size]

        # Process open alex
        open_alex_futures = open_alex.process_open_alex_work_task.map(
            pair=chunk,
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
        stored_dev_researcher_futures = db_utils.store_dev_researcher_em_links_task.map(
            pair=dev_researcher_futures,
            prod=unmapped(prod),
        )

        # Store this batch's errored results
        # Update errored store path with batch index
        this_batch_store_path = errored_store_path.with_name(
            errored_store_path.stem + f"-{i // batch_size}.parquet"
        )
        _store_errored_results(
            # Wait for all futures to complete
            results=[f.result() for f in stored_dev_researcher_futures],
            store_path=this_batch_store_path,
        )

        # Sleep for a second before next chunk
        time.sleep(1)

    # Cooldown
    time.sleep(3)


@app.command()
def prelinked_dataset_ingestion(
    source: str,
    prod: bool = False,
    use_dask: bool = False,
    batch_size: int = 200,
) -> None:
    """Get data from OpenAlex."""
    # Create current datetime without microseconds
    current_datetime = datetime.now().replace(microsecond=0)
    # Convert to isoformat and replace colons with dashes
    current_datetime_str = current_datetime.isoformat().replace(":", "-")

    # Create dir for this datetime
    current_datetime_dir = DEFAULT_RESULTS_DIR / current_datetime_str
    # Create "results" dir
    current_datetime_dir.mkdir(exist_ok=True, parents=True)
    errored_store_path = (
        current_datetime_dir / f"process-results-{source}-errored.parquet"
    )

    # If using dask, use DaskTaskRunner
    if use_dask:
        task_runner = DaskTaskRunner(
            cluster_class="distributed.LocalCluster",
            cluster_kwargs={"n_workers": 5, "threads_per_worker": 1},
        )
    else:
        task_runner = SequentialTaskRunner()

    # Create the flow
    ingest_flow = Flow(
        _prelinked_dataset_ingestion_flow,
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
        batch_size=batch_size,
        prod=prod,
        errored_store_path=errored_store_path,
    )

    # End duration
    end_dt = datetime.now()
    end_dt = end_dt.replace(microsecond=0)

    # Log time taken
    print(f"Total Processing Duration: {end_dt - start_dt}")


@app.command()
def get_random_sample_of_prelinked_source_data(
    seed: int = 12,
    outfile_path: str = "random-sample-prelinked-sources.csv",
) -> None:
    """Get a random sample of pre-linked source data."""
    # Set seed
    random.seed(seed)

    # Iter over sources, take random samples of their "get_dataset" function results
    results = []
    for source, source_func in SOURCE_MAP.items():
        print("Working on source:", source)
        source_results = source_func()

        # Filter dataset
        filtered_results = code_host_parsing.filter_repo_paper_pairs(
            source_results.successful_results,
        )

        # Take random sample
        random_sample = random.sample(
            filtered_results.successful_results,
            10,
        )

        # Unpack each result and then append to results
        for result in random_sample:
            results.append(
                {
                    "source": source,
                    **result.to_dict(),
                }
            )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(outfile_path, index=False)


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
