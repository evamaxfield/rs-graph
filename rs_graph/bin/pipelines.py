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
from dotenv import load_dotenv
from gh_tokens_loader import GitHubTokensCycler
from prefect import Task, flow, task, unmapped
from tqdm import tqdm

from rs_graph import types
from rs_graph.bin.data import download as download_rs_graph_data_files
from rs_graph.bin.data import upload as upload_rs_graph_data_files
from rs_graph.db import utils as db_utils
from rs_graph.enrichment import article, entity_matching, github
from rs_graph.sources import joss, plos, proto, pwc, softwarex
from rs_graph.utils import code_host_parsing

###############################################################################

app = typer.Typer()

DEFAULT_RESULTS_DIR = Path("processing-results")
DEFAULT_GITHUB_TOKENS_FILE = ".github-tokens.yml"
DEFAULT_OPEN_ALEX_EMAILS_FILE = ".open-alex-emails.yml"
DEFAULT_ELSEVIER_API_KEYS_FILE = ".elsevier-api-keys.yml"

###############################################################################


SOURCE_MAP: dict[str, proto.DatasetRetrievalFunction] = {
    "joss": joss.get_dataset,
    "plos": plos.get_dataset,
    "softwarex": softwarex.get_dataset,
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
        timeout_seconds=600,  # 10 minutes
    )
    @coiled.function(
        **coiled_kwargs,
        name=func.__name__,
    )
    def wrapped_func(*args, **kwargs):  # type: ignore
        return func(*args, **kwargs)

    return wrapped_func


def _load_open_alex_emails(
    open_alex_emails_file: str,
) -> list[str]:
    # Load emails
    try:
        with open(open_alex_emails_file) as f:
            emails_file = yaml.safe_load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Open Alex emails file not found at path: {open_alex_emails_file}"
        ) from e

    # Get emails
    emails_list = emails_file["emails"].values()

    return emails_list


def _load_elsevier_api_keys(
    elsevier_api_keys_file: str,
) -> list[str]:
    # Load tokens
    try:
        with open(elsevier_api_keys_file) as f:
            tokens_file = yaml.safe_load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"GitHub tokens file not found at path: {elsevier_api_keys_file}"
        ) from e

    # Get tokens
    tokens_list = tokens_file["keys"].values()

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
    use_prod: bool,
    github_tokens_file: str,
    open_alex_emails: list[str],
    semantic_scholar_api_key: str,
    elsevier_api_keys: list[str],
    use_coiled: bool,
    batch_size: int,
    errored_store_path: Path,
) -> None:
    # Get an infinite cycle of github tokens
    cycled_github_tokens = GitHubTokensCycler(gh_tokens_file=github_tokens_file)

    # Workers is the number of github tokens
    n_github_tokens = len(cycled_github_tokens)

    # Get an infinite cycle of open alex emails
    cycled_open_alex_emails = itertools.cycle(open_alex_emails)

    # Get the number of open alex emails
    n_open_alex_emails = len(open_alex_emails)

    # Print dataset and coiled status
    print("-" * 80)
    print("Pipeline Options:")
    print(f"Source: {source}")
    print(f"Use Prod Database: {use_prod}")
    print(f"Use Coiled: {use_coiled}")
    print(f"Batch Size: {batch_size}")
    print(f"GitHub Token Count: {n_github_tokens}")
    print(f"Open Alex Email Count: {n_open_alex_emails}")
    print(f"Elsevier API Key Count: {len(elsevier_api_keys)}")
    print("-" * 80)

    # Get dataset
    source_func = SOURCE_MAP[source]
    print("Getting dataset...")
    source_results = source_func(
        github_tokens=cycled_github_tokens._gh_tokens,
        elsevier_api_keys=elsevier_api_keys,
        semantic_scholar_api_key=semantic_scholar_api_key,
        open_alex_emails=open_alex_emails,
    )

    # Construct different cluster parameters
    article_processing_cluster_config = {
        "keepalive": "15m",
        "vm_type": "t4g.small",
        "n_workers": 1,
        "threads_per_worker": 6,
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
    }
    github_cluster_config = {
        "keepalive": "15m",
        "vm_type": "t4g.small",
        "n_workers": 1,
        "threads_per_worker": n_github_tokens,
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
    }
    gpu_cluster_config = {
        "keepalive": "15m",
        "vm_type": "t4g.medium",
        "n_workers": [3, 4],
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
    }

    # Filter dataset
    code_filtered_results = code_host_parsing.filter_repo_paper_pairs(
        source_results.successful_results,
    )

    # Filter out already processed pairs
    stored_filtered_results = db_utils.filter_stored_pairs(
        code_filtered_results.successful_results,
        use_prod=use_prod,
    )

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
            process_article_wrapped_task = _wrap_func_with_coiled_prefect_task(
                article.process_article_task,
                coiled_kwargs=article_processing_cluster_config,
            )
            article_processing_futures = process_article_wrapped_task.map(
                pair=chunk,
                open_alex_email=[
                    next(cycled_open_alex_emails) for _ in range(len(chunk))
                ],
                open_alex_email_count=n_open_alex_emails,
                semantic_scholar_api_key=unmapped(semantic_scholar_api_key),
            )

            # Process github
            process_github_wrapped_task = _wrap_func_with_coiled_prefect_task(
                github.process_github_repo_task,
                coiled_kwargs=github_cluster_config,
            )
            github_futures = process_github_wrapped_task.map(
                pair=article_processing_futures,
                github_api_key=[
                    next(cycled_github_tokens)
                    for _ in range(len(article_processing_futures))
                ],
            )

            # Store everything
            stored_futures = db_utils.store_full_details_task.map(
                pair=github_futures,
                use_prod=unmapped(use_prod),
            )

            # Match devs and researchers
            match_devs_and_researchers_wrapped_task = (
                _wrap_func_with_coiled_prefect_task(
                    entity_matching.match_devs_and_researchers,
                    coiled_kwargs=gpu_cluster_config,
                )
            )
            dev_researcher_futures = match_devs_and_researchers_wrapped_task.map(
                pair=stored_futures,
            )

            # Store the dev-researcher links
            stored_dev_researcher_futures = (
                db_utils.store_dev_researcher_em_links_task.map(
                    pair=dev_researcher_futures,
                    use_prod=unmapped(use_prod),
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
        _upload_db(use_prod)

        # Sleep for a second before next chunk
        time.sleep(1)

    # Cooldown
    time.sleep(3)


@app.command()
def prelinked_dataset_ingestion(
    source: str,
    use_prod: bool = False,
    use_coiled: bool = False,
    github_tokens_file: str = DEFAULT_GITHUB_TOKENS_FILE,
    open_alex_emails_file: str = DEFAULT_OPEN_ALEX_EMAILS_FILE,
    elsevier_api_keys_file: str = DEFAULT_ELSEVIER_API_KEYS_FILE,
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
    if use_prod:
        print("Downloading latest data files...")
        download_rs_graph_data_files(force=True)

    # Keep track of duration
    start_dt = datetime.now()
    start_dt = start_dt.replace(microsecond=0)

    # Load environment variables
    load_dotenv()

    # Get semantic scholar API key
    try:
        semantic_scholar_api_key = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    except KeyError as e:
        raise KeyError(
            "Please set the SEMANTIC_SCHOLAR_API_KEY environment variable."
        ) from e

    # Ignore prefect task introspection warnings
    os.environ["PREFECT_TASK_INTROSPECTION_WARN_THRESHOLD"] = "0"

    # Load Open Alex emails
    open_alex_emails = _load_open_alex_emails(open_alex_emails_file)

    # Load Elsevier API keys
    elsevier_api_keys = _load_elsevier_api_keys(elsevier_api_keys_file)

    # Start the flow
    _prelinked_dataset_ingestion_flow(
        source=source,
        use_prod=use_prod,
        github_tokens_file=github_tokens_file,
        open_alex_emails=open_alex_emails,
        semantic_scholar_api_key=semantic_scholar_api_key,
        elsevier_api_keys=elsevier_api_keys,
        use_coiled=use_coiled,
        batch_size=batch_size,
        errored_store_path=errored_store_path,
    )

    # End duration
    end_dt = datetime.now()
    end_dt = end_dt.replace(microsecond=0)

    # Upload latest if prod
    if use_prod:
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
