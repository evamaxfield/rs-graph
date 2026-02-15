#!/usr/bin/env python

import itertools
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
import typer
from dotenv import load_dotenv
from gh_tokens_loader import GitHubTokensCycler
from prefect import flow, unmapped
from tqdm import tqdm

from rs_graph import types
from rs_graph.bin.data import download as download_rs_graph_data_files
from rs_graph.bin.data import upload as upload_rs_graph_data_files
from rs_graph.bin.pipeline_utils import (
    DEFAULT_ELSEVIER_API_KEYS_FILE,
    DEFAULT_ERRORS_CACHE_FILE,
    DEFAULT_GITHUB_TOKENS_FILE,
    DEFAULT_OPEN_ALEX_TOKENS_FILE,
    DEFAULT_RESULTS_DIR,
    _get_basic_gpu_cluster_config,
    _get_small_cpu_api_cluster,
    _load_elsevier_api_keys,
    _load_open_alex_tokens,
    _wrap_func_with_coiled_prefect_task,
)
from rs_graph.db import utils as db_utils
from rs_graph.enrichment import article, entity_matching, github
from rs_graph.sources import joss, plos, proto, pwc, softcite_2025, softwarex
from rs_graph.utils import code_host_parsing

###############################################################################

app = typer.Typer(rich_markup_mode=None, pretty_exceptions_enable=False)

PRELINKED_INGESTION_SOURCE_MAP: dict[str, proto.DatasetRetrievalFunction] = {
    "joss": joss.get_dataset,
    "plos": plos.get_dataset,
    "softwarex": softwarex.get_dataset,
    "pwc": pwc.get_dataset,
    "softcite-2025": softcite_2025.get_dataset,
}

###############################################################################


def _parse_error_identifier(identifier: str) -> tuple[str, str | tuple[str, str, str]] | None:
    """
    Parse an error identifier into its type and components.

    Returns:
        - ("doi", doi_string) for DOI identifiers
        - ("repo", (code_host, owner, name)) for repository identifiers
        - None if identifier doesn't match either pattern
    """
    # Check if DOI pattern (contains "10." prefix typical for DOIs)
    if identifier.startswith("10.") or "/10." in identifier:
        return ("doi", identifier)

    # Check if repo pattern: {code_host}:{owner}/{name}
    if ":" in identifier and "/" in identifier:
        parts = identifier.split(":", 1)
        if len(parts) == 2:
            code_host = parts[0]
            owner_name = parts[1].split("/", 1)
            if len(owner_name) == 2:
                return ("repo", (code_host, owner_name[0], owner_name[1]))

    return None


def _load_errors_cache(errors_cache_file: Path) -> set[str]:
    """Load error identifiers from cache file and log unrecognized formats."""
    if not errors_cache_file.exists():
        return set()

    df = pl.read_parquet(errors_cache_file)
    identifiers = set(df["identifier"].to_list())

    # Log unrecognized identifiers
    for identifier in identifiers:
        parsed = _parse_error_identifier(identifier)
        if parsed is None:
            print(f"Unrecognized error identifier format: {identifier}")

    return identifiers


def _filter_prior_errored_pairs(
    pairs: list[types.ExpandedRepositoryDocumentPair],
    errored_identifiers: set[str],
) -> list[types.ExpandedRepositoryDocumentPair]:
    """Filter out pairs whose DOI or repository identifier is in the error cache."""
    filtered = []

    for pair in pairs:
        # Check if DOI is in error cache
        if pair.paper_doi in errored_identifiers:
            continue

        # Check if repo identifier is in error cache
        if pair.repo_parts is not None:
            repo_id = f"{pair.repo_parts.host}:{pair.repo_parts.owner}/{pair.repo_parts.name}"
            if repo_id in errored_identifiers:
                continue

        filtered.append(pair)

    skipped_count = len(pairs) - len(filtered)
    print(f"Filtered out {skipped_count} pairs with prior errors")
    print(f"Remaining pairs after error filtering: {len(filtered)}")

    return filtered


def _append_errors_to_cache(
    errors: list[types.ErrorResult],
    errors_cache_file: Path,
) -> None:
    """Append new error identifiers to the cache file."""
    if not errors:
        return

    new_identifiers = {e.identifier for e in errors}

    # Load existing
    if errors_cache_file.exists():
        existing_df = pl.read_parquet(errors_cache_file)
        existing_identifiers = set(existing_df["identifier"].to_list())
        new_identifiers = new_identifiers | existing_identifiers

    # Write back
    pl.DataFrame({"identifier": list(new_identifiers)}).write_parquet(errors_cache_file)


###############################################################################


@flow(
    log_prints=True,
)
def _prelinked_dataset_ingestion_flow(
    source: str,
    use_prod: bool,
    github_tokens_file: str,
    open_alex_tokens: list[str],
    semantic_scholar_api_key: str,
    elsevier_api_keys: list[str],
    use_coiled: bool,
    coiled_region: str,
    batch_size: int,
    errored_store_path: Path,
    filter_prior_errors: bool,
    errors_cache_file: Path,
) -> None:
    # Get an infinite cycle of github tokens
    cycled_github_tokens = GitHubTokensCycler(gh_tokens_file=github_tokens_file)

    # Workers is the number of github tokens
    n_github_tokens = len(cycled_github_tokens)

    # Get an infinite cycle of open alex tokens
    cycled_open_alex_tokens = itertools.cycle(open_alex_tokens)

    # Get the number of open alex tokens
    n_open_alex_tokens = len(open_alex_tokens)

    # Print dataset and coiled status
    print("-" * 80)
    print("Pipeline Options:")
    print(f"Source: {source}")
    print(f"Use Prod Database: {use_prod}")
    print(f"Use Coiled: {use_coiled}")
    print(f"Coiled Region: {coiled_region}")
    print(f"Batch Size: {batch_size}")
    print(f"GitHub Token Count: {n_github_tokens}")
    print(f"Open Alex Token Count: {n_open_alex_tokens}")
    print(f"Elsevier API Key Count: {len(elsevier_api_keys)}")
    print("-" * 80)

    # Get dataset
    source_func = PRELINKED_INGESTION_SOURCE_MAP[source]
    print("Getting dataset...")
    source_results = source_func(
        github_tokens=cycled_github_tokens._gh_tokens,
        elsevier_api_keys=elsevier_api_keys,
        semantic_scholar_api_key=semantic_scholar_api_key,
        open_alex_tokens=open_alex_tokens,
    )

    # Filter dataset
    code_filtered_results = code_host_parsing.filter_repo_paper_pairs(
        source_results.successful_results,
    )

    # Filter out already processed pairs
    stored_filtered_results = db_utils.filter_stored_pairs(
        code_filtered_results.successful_results,
        use_prod=use_prod,
    )

    # Filter out prior errored pairs if enabled
    if filter_prior_errors:
        print("Loading errors cache...")
        errored_identifiers = _load_errors_cache(errors_cache_file)
        print(f"Loaded {len(errored_identifiers)} prior error identifiers")

        stored_filtered_results = _filter_prior_errored_pairs(
            stored_filtered_results,
            errored_identifiers,
        )

    # Keep track of processing times
    processing_times = ProcessingTimes(
        open_alex_processing_times=deque(maxlen=1024),
        github_processing_times=deque(maxlen=1024),
        store_article_and_repository_times=deque(maxlen=1024),
        author_developer_matching_times=deque(maxlen=1024),
        store_author_developer_links_times=deque(maxlen=1024),
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
                coiled_kwargs=_get_small_cpu_api_cluster(
                    # TODO:
                    # Hardcoded to 10 workers for now since I know it can handle that
                    # In the future, should be based on number of tokens
                    n_workers=10,
                    use_coiled=use_coiled,
                    coiled_region=coiled_region,
                ),
            )
            article_processing_futures = process_article_wrapped_task.map(
                pair=chunk,
                open_alex_token=[next(cycled_open_alex_tokens) for _ in range(len(chunk))],
                semantic_scholar_api_key=unmapped(semantic_scholar_api_key),
            )

            # Process github
            process_github_wrapped_task = _wrap_func_with_coiled_prefect_task(
                github.process_github_repo_task,
                coiled_kwargs=_get_small_cpu_api_cluster(
                    n_workers=n_github_tokens,
                    use_coiled=use_coiled,
                    coiled_region=coiled_region,
                ),
            )
            github_futures = process_github_wrapped_task.map(
                pair=article_processing_futures,
                github_api_key=[
                    next(cycled_github_tokens) for _ in range(len(article_processing_futures))
                ],
            )

            # Store everything
            stored_futures = db_utils.store_full_details_task.map(
                pair=github_futures,
                use_prod=unmapped(use_prod),
            )

            # Match devs and researchers
            match_devs_and_researchers_wrapped_task = _wrap_func_with_coiled_prefect_task(
                entity_matching.match_devs_and_researchers,
                coiled_kwargs=_get_basic_gpu_cluster_config(
                    use_coiled=use_coiled,
                    coiled_region=coiled_region,
                ),
            )
            dev_researcher_futures = match_devs_and_researchers_wrapped_task.map(
                pair=stored_futures,
            )

            # Store the dev-researcher links
            stored_dev_researcher_futures = db_utils.store_dev_researcher_em_links_task.map(
                pair=dev_researcher_futures,
                use_prod=unmapped(use_prod),
            )

            # Store this batch's errored results
            # Update errored store path with batch index
            this_batch_store_path = errored_store_path.with_name(
                errored_store_path.stem + f"-{i // batch_size}.parquet"
            )
            processing_times = _store_batch_results(
                results=[f.result() for f in stored_dev_researcher_futures],
                store_path=this_batch_store_path,
                processing_times=processing_times,
                errors_cache_file=errors_cache_file,
            )

            # Log "{median} ({mean} +- {std})" for each processing time
            open_alex_processing_times_described = pd.Series(
                processing_times.open_alex_processing_times
            ).describe()
            github_processing_times_described = pd.Series(
                processing_times.github_processing_times
            ).describe()
            store_article_and_repository_times_described = pd.Series(
                processing_times.store_article_and_repository_times
            ).describe()
            author_developer_matching_times_described = pd.Series(
                processing_times.author_developer_matching_times
            ).describe()
            store_author_developer_links_times_described = pd.Series(
                processing_times.store_author_developer_links_times
            ).describe()

            # Log with two decimal places
            print("Processing Times (ignoring retries):")
            print(
                f"Open Alex: {open_alex_processing_times_described['50%']:.2f} "
                f"({open_alex_processing_times_described['mean']:.2f} "
                f"+- {open_alex_processing_times_described['std']:.2f})"
            )
            print(
                f"GitHub: {github_processing_times_described['50%']:.2f} "
                f"({github_processing_times_described['mean']:.2f} "
                f"+- {github_processing_times_described['std']:.2f})"
            )
            print(
                f"Store Article and Repository: "
                f"{store_article_and_repository_times_described['50%']:.2f} "
                f"({store_article_and_repository_times_described['mean']:.2f} "
                f"+- {store_article_and_repository_times_described['std']:.2f})"
            )
            print(
                f"Author Developer Matching: "
                f"{author_developer_matching_times_described['50%']:.2f} "
                f"({author_developer_matching_times_described['mean']:.2f} "
                f"+- {author_developer_matching_times_described['std']:.2f})"
            )
            print(
                f"Store Author Developer Links: "
                f"{store_author_developer_links_times_described['50%']:.2f} "
                f"({store_author_developer_links_times_described['mean']:.2f} "
                f"+- {store_author_developer_links_times_described['std']:.2f})"
            )

        except Exception as e:
            print("Error processing chunk, skipping storage of errors...")
            print(f"Error: {e}")

        # Sleep for a second before next chunk
        time.sleep(1)

    # Cooldown
    time.sleep(3)


@dataclass
class ProcessingTimes:
    open_alex_processing_times: deque[float]
    github_processing_times: deque[float]
    store_article_and_repository_times: deque[float]
    author_developer_matching_times: deque[float]
    store_author_developer_links_times: deque[float]


def _store_batch_results(
    results: list[types.StoredRepositoryDocumentPair | types.ErrorResult],
    store_path: Path,
    processing_times: ProcessingTimes,
    errors_cache_file: Path,
) -> ProcessingTimes:
    print("Storing batch results...")

    # Get only errors
    errored_results = [result for result in results if isinstance(result, types.ErrorResult)]

    # Log this batch counts
    print(f"This Batch Success: {len(results) - len(errored_results)}")
    print(f"This Batch Errored: {len(errored_results)}")

    # Store errored results
    errored_df = pd.DataFrame(errored_results)
    errored_df.to_parquet(store_path)

    # Append errors to cache
    _append_errors_to_cache(errored_results, errors_cache_file)

    # Update processing times
    for result in results:
        if isinstance(result, types.StoredRepositoryDocumentPair):
            if result.open_alex_processing_time_seconds is not None:
                processing_times.open_alex_processing_times.append(
                    result.open_alex_processing_time_seconds
                )
            if result.github_processing_time_seconds is not None:
                processing_times.github_processing_times.append(
                    result.github_processing_time_seconds
                )
            if result.store_article_and_repository_time_seconds is not None:
                processing_times.store_article_and_repository_times.append(
                    result.store_article_and_repository_time_seconds
                )
            if result.author_developer_matching_time_seconds is not None:
                processing_times.author_developer_matching_times.append(
                    result.author_developer_matching_time_seconds
                )
            if result.store_author_developer_links_time_seconds is not None:
                processing_times.store_author_developer_links_times.append(
                    result.store_author_developer_links_time_seconds
                )

    return processing_times


@app.command()
def prelinked_dataset_ingestion(
    source: str,
    use_prod: bool = False,
    use_coiled: bool = False,
    coiled_region: str = "us-west-2",
    github_tokens_file: str = DEFAULT_GITHUB_TOKENS_FILE,
    open_alex_tokens_file: str = DEFAULT_OPEN_ALEX_TOKENS_FILE,
    elsevier_api_keys_file: str = DEFAULT_ELSEVIER_API_KEYS_FILE,
    batch_size: int = 50,
    filter_prior_errors: bool = True,
    errors_cache_file: Path = DEFAULT_ERRORS_CACHE_FILE,
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
    errored_store_path = current_datetime_dir / f"process-results-{source}-errored.parquet"

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
        raise KeyError("Please set the SEMANTIC_SCHOLAR_API_KEY environment variable.") from e

    # Ignore prefect task introspection warnings
    os.environ["PREFECT_TASK_INTROSPECTION_WARN_THRESHOLD"] = "0"

    # Load Open Alex tokens
    open_alex_tokens = _load_open_alex_tokens(open_alex_tokens_file)

    # Load Elsevier API keys
    elsevier_api_keys = _load_elsevier_api_keys(elsevier_api_keys_file)

    # Start the flow
    _prelinked_dataset_ingestion_flow(
        source=source,
        use_prod=use_prod,
        github_tokens_file=github_tokens_file,
        open_alex_tokens=open_alex_tokens,
        semantic_scholar_api_key=semantic_scholar_api_key,
        elsevier_api_keys=elsevier_api_keys,
        use_coiled=use_coiled,
        coiled_region=coiled_region,
        batch_size=batch_size,
        errored_store_path=errored_store_path,
        filter_prior_errors=filter_prior_errors,
        errors_cache_file=errors_cache_file,
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


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
