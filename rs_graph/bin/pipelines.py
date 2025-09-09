#!/usr/bin/env python

import itertools
import math
import os
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import coiled
import pandas as pd
import typer
import yaml
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from gh_tokens_loader import GitHubTokensCycler
from prefect import Task, flow, task, unmapped
from tqdm import tqdm

from rs_graph import types
from rs_graph.bin.data import download as download_rs_graph_data_files
from rs_graph.bin.data import upload as upload_rs_graph_data_files
from rs_graph.db import utils as db_utils
from rs_graph.enrichment import article, entity_matching, github
from rs_graph.sources import joss, plos, proto, pwc, softcite_2025, softwarex
from rs_graph.utils import code_host_parsing
from rs_graph.utils.dt_and_td import parse_timedelta

###############################################################################

app = typer.Typer(rich_markup_mode=None, pretty_exceptions_enable=False)

DEFAULT_RESULTS_DIR = Path("processing-results")
DEFAULT_GITHUB_TOKENS_FILE = ".github-tokens.yml"
DEFAULT_OPEN_ALEX_EMAILS_FILE = ".open-alex-emails.yml"
DEFAULT_ELSEVIER_API_KEYS_FILE = ".elsevier-api-keys.yml"


def _get_open_alex_cluster_config(
    n_open_alex_emails: int,
    use_coiled: bool,
    coiled_region: str,
) -> dict:
    return {
        "keepalive": "15m",
        "vm_type": "t4g.small",
        "n_workers": 2,
        "threads_per_worker": n_open_alex_emails,
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
        "region": coiled_region,
    }


def _get_github_cluster_config(
    n_github_tokens: int,
    use_coiled: bool,
    coiled_region: str,
) -> dict:
    return {
        "keepalive": "15m",
        "vm_type": "t4g.small",
        # One worker per token to avoid rate limiting
        # This isn't deterministic, that is,
        # a single token might be used by multiple workers,
        # but this does spread the load out a bit
        # and should help avoid rate limiting
        "n_workers": n_github_tokens,
        "threads_per_worker": 1,
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
        "region": coiled_region,
    }


def _get_basic_gpu_cluster_config(
    use_coiled: bool,
    coiled_region: str,
) -> dict:
    return {
        "keepalive": "15m",
        "vm_type": "g4dn.xlarge",
        "n_workers": [2, 3],
        "spot_policy": "spot_with_fallback",
        "local": not use_coiled,
        "region": coiled_region,
    }


PRELINKED_INGESTION_SOURCE_MAP: dict[str, proto.DatasetRetrievalFunction] = {
    "joss": joss.get_dataset,
    "plos": plos.get_dataset,
    "softwarex": softwarex.get_dataset,
    "pwc": pwc.get_dataset,
    "softcite-2025": softcite_2025.get_dataset,
}

###############################################################################


def _wrap_func_with_coiled_prefect_task(
    func: Callable,
    prefect_kwargs: dict[str, Any] | None = None,
    coiled_kwargs: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
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
        environ=environ,
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
            f"Elsevier tokens file not found at path: {elsevier_api_keys_file}"
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
    coiled_region: str,
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
    print(f"Coiled Region: {coiled_region}")
    print(f"Batch Size: {batch_size}")
    print(f"GitHub Token Count: {n_github_tokens}")
    print(f"Open Alex Email Count: {n_open_alex_emails}")
    print(f"Elsevier API Key Count: {len(elsevier_api_keys)}")
    print("-" * 80)

    # Get dataset
    source_func = PRELINKED_INGESTION_SOURCE_MAP[source]
    print("Getting dataset...")
    source_results = source_func(
        github_tokens=cycled_github_tokens._gh_tokens,
        elsevier_api_keys=elsevier_api_keys,
        semantic_scholar_api_key=semantic_scholar_api_key,
        open_alex_emails=open_alex_emails,
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
                coiled_kwargs=_get_open_alex_cluster_config(
                    n_open_alex_emails=n_open_alex_emails,
                    use_coiled=use_coiled,
                    coiled_region=coiled_region,
                ),
            )
            article_processing_futures = process_article_wrapped_task.map(
                pair=chunk,
                open_alex_email=[next(cycled_open_alex_emails) for _ in range(len(chunk))],
                open_alex_email_count=n_open_alex_emails,
                semantic_scholar_api_key=unmapped(semantic_scholar_api_key),
            )

            # Process github
            process_github_wrapped_task = _wrap_func_with_coiled_prefect_task(
                github.process_github_repo_task,
                coiled_kwargs=_get_github_cluster_config(
                    n_github_tokens=n_github_tokens,
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
    coiled_region: str = "us-west-2",
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
        coiled_region=coiled_region,
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
    for source, source_func in PRELINKED_INGESTION_SOURCE_MAP.items():
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


@dataclass
class AuthorDeveloperPossiblePairs(DataClassJsonMixin):
    researcher_open_alex_id: str
    developer_account_username: str
    article_repository_allowed_datetime_difference: str
    total_combinations_count: int
    possible_pairs_count: int
    possible_pairs: list[entity_matching.SimplePossibleArticleRepositoryPair]


def _get_possible_article_repo_pairs_for_author_developer_pair(
    researcher_open_alex_id: str,
    developer_account_username: str,
    article_repository_allowed_datetime_difference: str,
    github_token: str,
    open_alex_email: str,
    open_alex_email_count: int,
) -> AuthorDeveloperPossiblePairs | types.ErrorResult:
    # Get all papers from OpenAlex for the researcher
    author_works = article.get_articles_for_researcher(
        researcher_open_alex_id=researcher_open_alex_id,
        open_alex_email=open_alex_email,
        open_alex_email_count=open_alex_email_count,
    )

    # Handle error
    if isinstance(author_works, types.ErrorResult):
        return author_works

    # Get all repositories for the developer account
    developer_repos = github.get_github_repos_for_developer(
        username=developer_account_username,
        github_api_key=github_token,
    )

    # Handle error
    if isinstance(developer_repos, types.ErrorResult):
        return developer_repos

    # Calculate total possible pairs
    total_possible_pairs = len(author_works) * len(developer_repos)

    # Get possible article-repository pairs for matching
    possible_pairs = entity_matching.get_possible_article_repository_pairs_for_matching(
        works=author_works,
        repos=developer_repos,
        max_datetime_difference=parse_timedelta(article_repository_allowed_datetime_difference),
    )

    return AuthorDeveloperPossiblePairs(
        researcher_open_alex_id=researcher_open_alex_id,
        developer_account_username=developer_account_username,
        article_repository_allowed_datetime_difference=article_repository_allowed_datetime_difference,
        total_combinations_count=total_possible_pairs,
        possible_pairs_count=len(possible_pairs),
        possible_pairs=list(possible_pairs),
    )


@flow(
    log_prints=True,
)
def _author_developer_article_repository_discovery_flow(  # noqa: C901
    process_n_author_developer_pairs: int,
    article_repository_allowed_datetime_difference: str,
    author_developer_links_filter_confidence_threshold: float,
    author_developer_links_filter_datetime_difference: str,
    one_to_one_only: bool,
    author_developer_batch_size: int,
    github_extended_details_batch_size: int,
    article_repository_matching_batch_size: int,
    use_prod: bool,
    use_coiled: bool,
    coiled_region: str,
    github_tokens_file: str,
    open_alex_emails: list[str],
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
    print(f"Process N Author-Developer Pairs: {process_n_author_developer_pairs}")
    print(
        f"Article Repository Allowed Datetime Difference: "
        f"{article_repository_allowed_datetime_difference}"
    )
    print(
        f"Author Developer Links Filter Confidence Threshold: "
        f"{author_developer_links_filter_confidence_threshold}"
    )
    print(
        f"Skip Author Developer Links Processed Within Last: "
        f"{author_developer_links_filter_datetime_difference}"
    )
    print(f"Use Prod Database: {use_prod}")
    print(f"Use Coiled: {use_coiled}")
    print(f"Coiled Region: {coiled_region}")
    print(f"GitHub Token Count: {n_github_tokens}")
    print(f"Open Alex Email Count: {n_open_alex_emails}")
    print("-" * 80)

    # Get author-developer-account links from the database
    print("Getting author-developer-account links from the database...")
    hydrated_author_developer_links = db_utils.get_hydrated_author_developer_links(
        use_prod=use_prod,
        filter_datetime_difference=author_developer_links_filter_datetime_difference,
        filter_confidence_threshold=author_developer_links_filter_confidence_threshold,
        n=process_n_author_developer_pairs,
    )

    # TODO: consider splitting out the open alex calls and github calls
    # into separate coiled tasks to better parallelize
    # that is a pretty minor optimization though

    # Wrap the possible pairs function with coiled
    get_possible_pairs_wrapped_task = _wrap_func_with_coiled_prefect_task(
        _get_possible_article_repo_pairs_for_author_developer_pair,
        coiled_kwargs=_get_github_cluster_config(
            n_github_tokens=n_github_tokens,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
    )

    # Process pairs for each author-developer link in batches
    batch_start_time = time.time()
    total_possible_combinations = 0
    total_possible_pairs = 0
    total_possible_pairs_after_filtering = 0
    total_possible_pairs_for_inference = 0
    total_matches_found = 0
    possible_results = []
    for author_developer_index in tqdm(
        range(0, len(hydrated_author_developer_links), author_developer_batch_size),
        desc="Getting Possible Pairs",
        total=math.ceil(len(hydrated_author_developer_links) / author_developer_batch_size),
        leave=False,
    ):
        batch = hydrated_author_developer_links[
            author_developer_index : author_developer_index + author_developer_batch_size
        ]
        simple_possible_pairs_futures = get_possible_pairs_wrapped_task.map(
            researcher_open_alex_id=[link.researcher_open_alex_id for link in batch],
            developer_account_username=[link.developer_account_username for link in batch],
            article_repository_allowed_datetime_difference=[
                article_repository_allowed_datetime_difference for _ in batch
            ],
            github_token=[next(cycled_github_tokens) for _ in batch],
            open_alex_email=[next(cycled_open_alex_emails) for _ in batch],
            open_alex_email_count=[n_open_alex_emails for _ in batch],
        )

        # Wait for all futures to complete
        this_batch_results: list[AuthorDeveloperPossiblePairs | types.ErrorResult] = [
            f.result() for f in simple_possible_pairs_futures
        ]

        # Filter to only those with possible pairs
        this_batch_possible_pairs = [
            result
            for result in this_batch_results
            if (
                isinstance(result, AuthorDeveloperPossiblePairs)
                and len(result.possible_pairs) > 0
            )
        ]

        # Add to total possible pairs and combinations
        total_possible_combinations += sum(
            r.total_combinations_count for r in this_batch_possible_pairs
        )
        total_possible_pairs += sum(r.possible_pairs_count for r in this_batch_possible_pairs)

        # Convert all possible pairs to a flat list in the form of
        # PossibleArticleRepositoryPair
        flat_possible_pairs: list[entity_matching.PossibleArticleRepositoryPair] = []
        for author_developer_possible_pairs in this_batch_possible_pairs:
            for simple_possible_pair in author_developer_possible_pairs.possible_pairs:
                flat_possible_pairs.append(
                    entity_matching.PossibleArticleRepositoryPair(
                        source_researcher_open_alex_id=author_developer_possible_pairs.researcher_open_alex_id,
                        source_developer_account_username=author_developer_possible_pairs.developer_account_username,
                        article_doi=simple_possible_pair.work["doi"]
                        .lower()
                        .replace("https://doi.org/", "")
                        .strip(),
                        article_title=simple_possible_pair.work["title"],
                        open_alex_work=simple_possible_pair.work,
                        repo_owner=simple_possible_pair.repository["owner"]["login"],
                        repo_name=simple_possible_pair.repository["name"],
                        github_repository=simple_possible_pair.repository,
                    )
                )

        # Drop any possible pairs that are already in the db
        print("Filtering out already stored article-repository pairs...")
        simple_filtered_flat_possible_pairs: list[
            entity_matching.PossibleArticleRepositoryPair
        ] = []
        known_articles = []
        known_repos = []
        for expanded_possible_article_repo_pair in tqdm(
            flat_possible_pairs,
            desc="Filtering Stored Pairs",
            leave=False,
        ):
            if not db_utils.check_article_repository_pair_already_in_db(
                article_doi=expanded_possible_article_repo_pair.article_doi,
                article_title=expanded_possible_article_repo_pair.article_title,
                code_host="github",
                repo_owner=expanded_possible_article_repo_pair.repo_owner,
                repo_name=expanded_possible_article_repo_pair.repo_name,
                use_prod=use_prod,
            ):
                simple_filtered_flat_possible_pairs.append(expanded_possible_article_repo_pair)
            else:
                known_articles.append(expanded_possible_article_repo_pair.article_doi)
                known_repos.append(
                    (
                        expanded_possible_article_repo_pair.repo_owner,
                        expanded_possible_article_repo_pair.repo_name,
                    )
                )

        # Further filter pairs to ensure one-to-one mapping if specified
        if one_to_one_only:
            print("Applying one-to-one filtering...")
            filtered_flat_possible_pairs = []
            for expanded_possible_article_repo_pair in simple_filtered_flat_possible_pairs:
                article_key = expanded_possible_article_repo_pair.article_doi
                repo_key = (
                    expanded_possible_article_repo_pair.repo_owner,
                    expanded_possible_article_repo_pair.repo_name,
                )
                if article_key not in known_articles and repo_key not in known_repos:
                    filtered_flat_possible_pairs.append(expanded_possible_article_repo_pair)

        else:
            filtered_flat_possible_pairs = simple_filtered_flat_possible_pairs

        # Add to total possible pairs after filtering
        total_possible_pairs_after_filtering += len(filtered_flat_possible_pairs)

        # For each unique repository, get the README and the contributor details
        # First get the unique repos
        unique_repos: list[tuple[str, str]] = []
        for expanded_possible_article_repo_pair in filtered_flat_possible_pairs:
            repo_key = (
                expanded_possible_article_repo_pair.repo_owner,
                expanded_possible_article_repo_pair.repo_name,
            )
            if repo_key not in unique_repos:
                unique_repos.append(repo_key)

        print(f"Unique repositories to process: {len(unique_repos)}")

        # Get README and contributors for each unique repo
        print("Getting README and contributors for each unique repository...")
        get_github_repo_readme_and_contribs_task = _wrap_func_with_coiled_prefect_task(
            github.get_repo_readme_and_contributor_details,
            coiled_kwargs=_get_github_cluster_config(
                n_github_tokens=n_github_tokens,
                use_coiled=use_coiled,
                coiled_region=coiled_region,
            ),
        )

        # Process in batches to avoid overloading
        repo_readme_and_contributor_results: list[
            github.RepoReadmeAndContributorInfo | types.ErrorResult
        ] = []
        for repo_index in tqdm(
            range(0, len(unique_repos), github_extended_details_batch_size),
            desc="Getting README and Contributors",
            total=math.ceil(len(unique_repos) / github_extended_details_batch_size),
            leave=False,
        ):
            repo_batch = unique_repos[
                repo_index : repo_index + github_extended_details_batch_size
            ]
            readme_and_contributor_futures = get_github_repo_readme_and_contribs_task.map(
                repo_owner=[owner for owner, _ in repo_batch],
                repo_name=[name for _, name in repo_batch],
                github_api_key=[next(cycled_github_tokens) for _ in repo_batch],
            )
            repo_readme_and_contributor_results.extend(
                [f.result() for f in readme_and_contributor_futures]
            )

        # Filter out errors
        repo_readme_and_contributor_details = [
            result
            for result in repo_readme_and_contributor_results
            if isinstance(result, github.RepoReadmeAndContributorInfo)
        ]

        print(
            f"Successfully retrieved README and contributors for "
            f"{len(repo_readme_and_contributor_details)} repositories."
        )

        # Create a map of (owner, name) to readme and contributors
        repo_to_readme_and_contributors = {
            (result.repo_owner, result.repo_name): result
            for result in repo_readme_and_contributor_details
        }

        # Create InferenceReadyPossibleArticleRepositoryPair for each possible pair
        entity_matching_ready_possible_pairs: list[
            entity_matching.binary_article_repo_em.InferenceReadyArticleRepositoryPair
        ] = []
        for expanded_possible_article_repo_pair in filtered_flat_possible_pairs:
            repo_key = (
                expanded_possible_article_repo_pair.repo_owner,
                expanded_possible_article_repo_pair.repo_name,
            )
            readme_and_contributors = repo_to_readme_and_contributors.get(repo_key)
            if readme_and_contributors:
                entity_matching_ready_possible_pairs.append(
                    entity_matching._prep_for_article_repository_matching(
                        possible_pair=expanded_possible_article_repo_pair,
                        repository_readme_and_contributors=readme_and_contributors,
                    )
                )

        # Filter out errored prep results
        entity_matching_ready_possible_pairs = [
            pair
            for pair in entity_matching_ready_possible_pairs
            if isinstance(
                pair, entity_matching.binary_article_repo_em.InferenceReadyArticleRepositoryPair
            )
        ]

        print(
            f"Prepared {len(entity_matching_ready_possible_pairs)} pairs for entity matching."
        )
        total_possible_pairs_for_inference += len(entity_matching_ready_possible_pairs)

        # Get a wrapped function for article-repository matching
        match_articles_and_repositories_wrapped_task = _wrap_func_with_coiled_prefect_task(
            entity_matching.match_articles_and_repositories,
            coiled_kwargs=_get_basic_gpu_cluster_config(
                use_coiled=use_coiled,
                coiled_region=coiled_region,
            ),
            environ={
                # TODO: this model should be public soon
                "HF_TOKEN": os.environ["HF_TOKEN"],
            },
        )

        # Pass to inference
        print("Running entity matching inference...")

        # Create batches of prepped pairs to map across
        inference_batches = [
            entity_matching_ready_possible_pairs[i : i + article_repository_matching_batch_size]
            for i in range(
                0,
                len(entity_matching_ready_possible_pairs),
                article_repository_matching_batch_size,
            )
        ]

        # Map across batches
        inference_futures = match_articles_and_repositories_wrapped_task.map(
            inference_ready_article_repository_pairs=inference_batches,
        )

        # Collect results
        mapped_inference_results = [f.result() for f in inference_futures]

        # These can be either lists of predictions or ErrorResult
        predicted_matches: list[
            entity_matching.binary_article_repo_em.MatchedArticleRepository
        ] = []
        for result in mapped_inference_results:
            print(result)
            if isinstance(result, types.ErrorResult):
                print(f"Error during inference: {result}")
            else:
                predicted_matches.extend(result)

        # Double check that these predicted matches aren't in the database
        print("Filtering out already stored predicted matches...")
        print(f"Predicted Matches Before Filtering: {len(predicted_matches)}")
        predicted_matches = [
            match
            for match in predicted_matches
            if not db_utils.check_article_repository_pair_already_in_db(
                article_doi=match.article_details.doi,
                article_title=match.article_details.title,
                code_host="github",
                repo_owner=match.repository_details.owner,
                repo_name=match.repository_details.name,
                use_prod=use_prod,
            )
        ]
        print(f"Predicted Matches After Filtering: {len(predicted_matches)}")

        # Add to total matches found
        total_matches_found += len(predicted_matches)

        # Create dataframe of results to annotate / evaluate later
        possible_results.extend(
            [
                {
                    "document_doi": match.article_details.doi,
                    "document_url": f"https://doi.org/{match.article_details.doi}",
                    "repository_full_name": (
                        f"{match.repository_details.owner}/{match.repository_details.name}"
                    ),
                    "repository_url": (
                        f"https://github.com/"
                        f"{match.repository_details.owner}/{match.repository_details.name}"
                    ),
                    "model_confidence": match.confidence,
                }
                for match in predicted_matches
            ]
        )
        pd.DataFrame(possible_results).to_csv(
            "snowball-sampling-predicted-article-repo-pairs.csv",
            index=False,
        )

        # Print throughput stats
        print("Throughput Stats So Far:")
        print(f"Total Possible Combinations: {total_possible_combinations}")
        print(f"Total Possible Pairs: {total_possible_pairs}")
        print(f"Total Possible Pairs After Filtering: {total_possible_pairs_after_filtering}")
        print(f"Total Possible Pairs For Inference: {total_possible_pairs_for_inference}")
        print(f"Total Matches Found: {total_matches_found}")
        print()

        # Now, normalized by total author-developer links processed
        n_author_developer_links_processed = author_developer_index + len(batch)
        print("Normalized By Author-Developer Links Processed:")
        print(f"Author-Developer Links Processed: {n_author_developer_links_processed}")
        print(
            f"Possible Combinations Per Link: "
            f"{total_possible_combinations / n_author_developer_links_processed:.2f}"
        )
        print(
            f"Possible Pairs Per Link: "
            f"{total_possible_pairs / n_author_developer_links_processed:.2f}"
        )
        print(
            f"Possible Pairs After Filtering Per Link: "
            f"{total_possible_pairs_after_filtering / n_author_developer_links_processed:.2f}"
        )
        print(
            f"Possible Pairs For Inference Per Link: "
            f"{total_possible_pairs_for_inference / n_author_developer_links_processed:.2f}"
        )
        print(
            f"Matches Found Per Link: "
            f"{total_matches_found / n_author_developer_links_processed:.2f}"
        )
        print()

        # Get batch end time and normalize by duration
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        print(f"Batch Duration: {batch_duration:.2f} seconds")
        print(
            f"Author-Developer Links Processed Per Second: "
            f"{n_author_developer_links_processed / batch_duration:.2f}"
        )
        print(
            f"Possible Pairs For Inference Per Second: "
            f"{total_possible_pairs_for_inference / batch_duration:.2f}"
        )
        print(f"Matches Found Per Second: " f"{total_matches_found / batch_duration:.2f}")
        print()

        # Reset batch start time
        batch_start_time = time.time()
        print("-" * 80)
        print()


@app.command()
def snowball_sampling_discovery(
    process_n_author_developer_pairs: int = 200,
    article_repository_allowed_datetime_difference: str = "3 years",
    author_developer_links_filter_confidence_threshold: float = 0.97,
    author_developer_links_filter_datetime_difference: str = "2 years",
    one_to_one_only: bool = True,
    author_developer_batch_size: int = 1,
    github_extended_details_batch_size: int = 20,
    article_respository_matching_batch_size: int = 16,
    use_prod: bool = False,
    use_coiled: bool = False,
    coiled_region: str = "us-west-2",
    github_tokens_file: str = DEFAULT_GITHUB_TOKENS_FILE,
    open_alex_emails_file: str = DEFAULT_OPEN_ALEX_EMAILS_FILE,
) -> None:
    """
    Discover new article-repository pairs via snowball sampling.

    This will use the existing database starting from stored
    researcher-developer-account links, then lookup each author's articles
    and their repositories, use our article-repository matching model
    to predict new pairs, and then conduct standard processing.
    """
    # Load environment variables
    load_dotenv()

    # Get open alex emails
    _load_open_alex_emails(open_alex_emails_file)

    # Ignore prefect task introspection warnings
    os.environ["PREFECT_TASK_INTROSPECTION_WARN_THRESHOLD"] = "0"

    # Start the flow
    _author_developer_article_repository_discovery_flow(
        process_n_author_developer_pairs=process_n_author_developer_pairs,
        article_repository_allowed_datetime_difference=article_repository_allowed_datetime_difference,
        author_developer_links_filter_confidence_threshold=author_developer_links_filter_confidence_threshold,
        author_developer_links_filter_datetime_difference=author_developer_links_filter_datetime_difference,
        one_to_one_only=one_to_one_only,
        author_developer_batch_size=author_developer_batch_size,
        github_extended_details_batch_size=github_extended_details_batch_size,
        article_repository_matching_batch_size=article_respository_matching_batch_size,
        use_prod=use_prod,
        use_coiled=use_coiled,
        coiled_region=coiled_region,
        github_tokens_file=github_tokens_file,
        open_alex_emails=_load_open_alex_emails(open_alex_emails_file),
    )


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
