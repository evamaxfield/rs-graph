#!/usr/bin/env python

import itertools
import math
import os
import random
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import coiled
import pandas as pd
import typer
import yaml
from prefect import Task, flow, task, unmapped
from tqdm import tqdm

from rs_graph import types
from rs_graph.bin.data import download as download_rs_graph_data_files
from rs_graph.bin.data import upload as upload_rs_graph_data_files
from rs_graph.db import utils as db_utils
from rs_graph.enrichment import entity_matching, github, open_alex
from rs_graph.sources import joss, plos, proto, pwc
from rs_graph.utils import code_host_parsing

###############################################################################

app = typer.Typer()

DEFAULT_RESULTS_DIR = Path("processing-results")
DEFAULT_GITHUB_TOKENS_FILE = ".github-tokens.yml"

###############################################################################


SOURCE_MAP: dict[str, proto.DatasetRetrievalFunction] = {
    "joss": joss.get_dataset,
    "plos": plos.get_dataset,
    # "softwarex": softwarex.get_dataset,
    "pwc": pwc.get_dataset,
}


def _wrap_func_with_coiled_prefect_task(
    func: Callable,
    prefect_kwargs: dict[str, Any] | None = None,
    coiled_kwargs: dict[str, Any] | None = None,
) -> Task:
    if coiled_kwargs is None:
        coiled_kwargs = {}
    if prefect_kwargs is None:
        prefect_kwargs = {}

    @task(
        **prefect_kwargs,
        name=func.__name__,
        log_prints=True,
    )
    @coiled.function(
        **coiled_kwargs,
        name=func.__name__,
    )
    def wrapped_func(*args, **kwargs):  # type: ignore
        return func(*args, **kwargs)

    return wrapped_func


def _load_github_tokens(
    github_tokens_file: str,
) -> list[str]:
    # Load tokens
    try:
        with open(github_tokens_file) as f:
            tokens_file = yaml.safe_load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"GitHub tokens file not found at path: {github_tokens_file}"
        ) from e

    # Get tokens
    tokens_list = tokens_file["tokens"].values()

    return tokens_list


def _upload_db(
    prod: bool,
) -> None:
    if prod:
        print("Uploading files to GCS...")
        upload_rs_graph_data_files()


def _store_batch_results(
    results: list[types.StoredRepositoryDocumentPair | types.ErrorResult],
    store_path: Path,
) -> None:
    print("Storing batch results...")

    # Get only errors
    errored_results = [
        result for result in results if isinstance(result, types.ErrorResult)
    ]

    # Log this batch counts
    print(f"This Batch Success: {len(results) - len(errored_results)}")
    print(f"This Batch Errored: {len(errored_results)}")

    # Store errored results
    errored_df = pd.DataFrame(errored_results)
    errored_df.to_parquet(store_path)


@flow(
    log_prints=True,
)
def _prelinked_dataset_ingestion_flow(
    source: str,
    prod: bool,
    github_tokens: list[str],
    use_coiled: bool,
    batch_size: int,
    errored_store_path: Path,
) -> None:
    # Get dataset
    source_func = SOURCE_MAP[source]
    source_results = source_func()

    # Filter dataset
    code_filtered_results = code_host_parsing.filter_repo_paper_pairs(
        source_results.successful_results,
    )

    # Filter out already processed pairs
    stored_filtered_results = db_utils.filter_stored_pairs(
        code_filtered_results.successful_results,
        prod=prod,
    )

    # Get an infinite cycle of github tokens
    cycled_github_tokens = itertools.cycle(github_tokens)

    # Workers is the number of github tokens
    n_github_tokens = len(github_tokens)

    # Construct different cluster parameters
    github_cluster_config = {
        "keepalive": "15m",
        "cpu": [2, 4],
        "memory": ["2GiB", "8GiB"],
        "n_workers": 1,
        "threads_per_worker": n_github_tokens,
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
    }
    print(github_cluster_config)
    open_alex_cluster_config = {
        "keepalive": "15m",
        "cpu": [4, 8],
        "memory": ["4GiB", "8GiB"],
        "n_workers": 1,
        "threads_per_worker": 6,
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
    }
    gpu_cluster_kwargs = {
        "keepalive": "15m",
        "cpu": [4, 8],
        "memory": ["4GiB", "8GiB"],
        "n_workers": 3,
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
    }

    # Create chunks of batch_size of the results to process
    n_batches = math.ceil(len(stored_filtered_results) / batch_size)
    for i in tqdm(
        range(0, len(stored_filtered_results), batch_size),
        desc="Batches",
        total=n_batches,
    ):
        chunk = stored_filtered_results[i : i + batch_size]

        # Handle any timeouts and such
        try:
            # Process open alex
            process_open_alex_wrapped_task = _wrap_func_with_coiled_prefect_task(
                open_alex.process_open_alex_work_task,
                coiled_kwargs=open_alex_cluster_config,
            )
            open_alex_futures = process_open_alex_wrapped_task.map(
                pair=chunk,
            )

            # Process github
            process_github_wrapped_task = _wrap_func_with_coiled_prefect_task(
                github.process_github_repo_task,
                coiled_kwargs=github_cluster_config,
            )
            github_futures = process_github_wrapped_task.map(
                pair=open_alex_futures,
                github_api_key=[
                    next(cycled_github_tokens) for _ in range(len(open_alex_futures))
                ],
            )

            # Store everything
            stored_futures = db_utils.store_full_details_task.map(
                pair=github_futures,
                prod=unmapped(prod),
            )

            # Match devs and researchers
            match_devs_and_researchers_wrapped_task = (
                _wrap_func_with_coiled_prefect_task(
                    entity_matching.match_devs_and_researchers,
                    coiled_kwargs=gpu_cluster_kwargs,
                )
            )
            dev_researcher_futures = match_devs_and_researchers_wrapped_task.map(
                pair=stored_futures,
            )

            # Store the dev-researcher links
            stored_dev_researcher_futures = (
                db_utils.store_dev_researcher_em_links_task.map(
                    pair=dev_researcher_futures,
                    prod=unmapped(prod),
                )
            )

            # Store this batch's errored results
            # Update errored store path with batch index
            this_batch_store_path = errored_store_path.with_name(
                errored_store_path.stem + f"-{i // batch_size}.parquet"
            )
            _store_batch_results(
                results=[f.result() for f in stored_dev_researcher_futures],
                store_path=this_batch_store_path,
            )

        except Exception as e:
            print("Error processing chunk, skipping storage of errors...")
            print(f"Error: {e}")

        # Always upload the database
        _upload_db(prod)

        # Sleep for a second before next chunk
        time.sleep(1)

    # Cooldown
    time.sleep(3)


@app.command()
def prelinked_dataset_ingestion(
    source: str,
    prod: bool = False,
    use_coiled: bool = False,
    github_tokens_file: str = DEFAULT_GITHUB_TOKENS_FILE,
    batch_size: int = 50,
) -> None:
    """
    Process and ingest a stored pre-linked dataset of
    scientific articles and source code repositories.
    """
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

    # Download latest if prod
    if prod:
        print("Downloading latest data files...")
        download_rs_graph_data_files(force=True)

    # Keep track of duration
    start_dt = datetime.now()
    start_dt = start_dt.replace(microsecond=0)

    # Ignore prefect task introspection warnings
    os.environ["PREFECT_TASK_INTROSPECTION_WARN_THRESHOLD"] = "0"

    # Load GitHub tokens
    github_tokens = _load_github_tokens(github_tokens_file)

    # Start the flow
    _prelinked_dataset_ingestion_flow(
        source=source,
        prod=prod,
        github_tokens=github_tokens,
        use_coiled=use_coiled,
        batch_size=batch_size,
        errored_store_path=errored_store_path,
    )

    # End duration
    end_dt = datetime.now()
    end_dt = end_dt.replace(microsecond=0)

    # Upload latest if prod
    if prod:
        upload_rs_graph_data_files()

    # Sum errors
    errored_df = pd.concat(
        [pd.read_parquet(path) for path in current_datetime_dir.glob("*.parquet")]
    )
    print(f"Total Errored: {len(errored_df)}")

    # Log time taken
    print(f"Total Processing Duration: {end_dt - start_dt}")


@app.command()
def get_random_sample_of_prelinked_source_data(
    n: int = 50,
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
            n,
        )

        # Unpack each result and then append to results
        for result in random_sample:
            # Assert repo_parts is not None (for type checker)
            assert result.repo_parts is not None

            results.append(
                {
                    "source": source,
                    "paper_doi": result.paper_doi,
                    "repo_url": f"https://github.com/{result.repo_parts.owner}/{result.repo_parts.name}",
                    "label-match": None,
                    "label-software-type": None,
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
