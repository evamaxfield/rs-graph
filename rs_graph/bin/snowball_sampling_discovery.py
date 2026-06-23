#!/usr/bin/env python

import itertools
import os
import signal
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta

import polars as pl
import typer
from dotenv import load_dotenv
from gh_tokens_loader import GitHubTokensCycler
from prefect import flow, unmapped
from tqdm import tqdm

from rs_graph import __version__ as rs_graph_version
from rs_graph import types
from rs_graph.bin.pipeline_utils import (
    DEFAULT_GITHUB_TOKENS_FILE,
    DEFAULT_OPEN_ALEX_TOKENS_FILE,
    _get_basic_gpu_cluster_config,
    _get_small_cpu_api_cluster,
    _load_open_alex_tokens,
    _wrap_func_with_coiled_prefect_task,
)
from rs_graph.data import DATA_FILES_DIR
from rs_graph.db import models as db_models
from rs_graph.db import utils as db_utils
from rs_graph.enrichment import article, entity_matching, github
from rs_graph.utils.dt_and_td import parse_timedelta

###############################################################################

app = typer.Typer()

# Event used to signal a graceful shutdown on keyboard interrupt.
# When set, the pipeline will finish any in-progress critical storage
# and exit at the next safe point.
_shutdown_requested = threading.Event()


def _handle_sigint(signum: int, frame: object) -> None:
    if _shutdown_requested.is_set():
        # Second interrupt — restore default handler and re-raise to force quit
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
    _shutdown_requested.set()
    print(
        "\nInterrupt received. Will exit at next safe point "
        "(before article-repo storage or after author-developer storage). "
        "Press Ctrl+C again to force quit."
    )


###############################################################################


def _summarize_errors(errors: list[types.ErrorResult], label: str) -> None:
    """Print a structured summary of errors grouped by step and error type."""
    print(f"WARNING: {len(errors)} errored {label} results.")

    # Group by step
    step_groups: dict[str, list[types.ErrorResult]] = {}
    for err in errors:
        step_groups.setdefault(err.step, []).append(err)

    for step, step_errors in step_groups.items():
        print(f"  step='{step}': {len(step_errors)} errors")
        # Count by error type (first line of error string, before ':')
        error_types = Counter(
            err.error.split("\n")[0].split(":")[0].strip() for err in step_errors
        )
        for error_type, count in error_types.most_common():
            print(f"    {error_type}: {count}")


def _get_author_articles_for_researcher(
    author_developer_link_id: int,
    researcher_open_alex_id: str,
    open_alex_token: str,
    semantic_scholar_api_key: str,
) -> list[types.AuthorArticleDetails | types.ErrorResult] | types.ErrorResult:
    # Get the articles for the researcher
    researcher_articles = article.get_articles_for_researcher(
        researcher_open_alex_id=researcher_open_alex_id,
        open_alex_token=open_alex_token,
        semantic_scholar_api_key=semantic_scholar_api_key,
    )

    if isinstance(researcher_articles, types.ErrorResult):
        return researcher_articles

    author_article_details_list: list[types.AuthorArticleDetails | types.ErrorResult] = []
    for work_and_oa_results in researcher_articles:
        if isinstance(work_and_oa_results, types.ErrorResult):
            author_article_details_list.append(work_and_oa_results)
        else:
            author_article_details = types.AuthorArticleDetails(
                author_developer_link_id=author_developer_link_id,
                researcher_open_alex_id=researcher_open_alex_id,
                pyalex_work=work_and_oa_results.pyalex_work,
                open_alex_results_models=work_and_oa_results.open_alex_results,
            )
            author_article_details_list.append(author_article_details)

    return author_article_details_list


def _get_developer_repositories_for_developer(
    author_developer_link_id: int,
    developer_account_username: str,
    github_api_key: str,
) -> list[types.DeveloperRepositoryDetails | types.ErrorResult] | types.ErrorResult:
    # Get the repositories for the developer
    developer_repositories = github.get_github_repos_for_developer(
        username=developer_account_username,
        github_api_key=github_api_key,
    )

    if isinstance(developer_repositories, types.ErrorResult):
        return developer_repositories

    developer_repository_details_list: list[
        types.DeveloperRepositoryDetails | types.ErrorResult
    ] = []
    for github_result in developer_repositories:
        if isinstance(github_result, types.ErrorResult):
            developer_repository_details_list.append(github_result)
        else:
            developer_repository_details = types.DeveloperRepositoryDetails(
                author_developer_link_id=author_developer_link_id,
                developer_account_username=developer_account_username,
                github_result_models=github_result,
            )
            developer_repository_details_list.append(developer_repository_details)

    return developer_repository_details_list


def _flatten_and_check_articles_in_db(
    all_author_articles_and_errors: list[
        list[types.AuthorArticleDetails | types.ErrorResult] | types.ErrorResult
    ],
    database_path: str,
    ignorable_doi_spans: list[str],
    extended_processing: bool,
) -> list[types.AuthorArticleDetails | types.FilteredResult | types.ErrorResult]:
    possible_count = 0
    to_process_count = 0
    no_doi_count = 0
    ignorable_doi_count = 0
    already_in_db_count = 0
    extended_in_db_count = 0
    flattened_results: list[
        types.AuthorArticleDetails | types.FilteredResult | types.ErrorResult
    ] = []
    for author_articles_and_errors in all_author_articles_and_errors:
        if isinstance(author_articles_and_errors, types.ErrorResult):
            flattened_results.append(author_articles_and_errors)
        else:
            for item in author_articles_and_errors:
                if isinstance(item, types.ErrorResult):
                    flattened_results.append(item)
                else:
                    possible_count += 1
                    if item.open_alex_results_models.document_model.doi is None:
                        no_doi_count += 1
                        filtered_result = types.FilteredResult(
                            source=item.researcher_open_alex_id,
                            identifier=item.open_alex_results_models.document_model.title,
                            reason="No DOI available",
                        )
                        flattened_results.append(filtered_result)

                    elif any(
                        span in item.open_alex_results_models.document_model.doi.lower()
                        for span in ignorable_doi_spans
                    ):
                        ignorable_doi_count += 1
                        filtered_result = types.FilteredResult(
                            source=item.researcher_open_alex_id,
                            identifier=item.open_alex_results_models.document_model.doi,
                            reason=f"DOI in ignorable spans ({', '.join(ignorable_doi_spans)})",
                        )
                        flattened_results.append(filtered_result)

                    else:
                        link_quality = db_utils.check_article_link_quality_in_db(
                            article_doi=item.open_alex_results_models.document_model.doi,
                            article_title=item.open_alex_results_models.document_model.title,
                            database_path=database_path,
                        )
                        if link_quality == "has_good_link":
                            already_in_db_count += 1
                            filtered_result = types.FilteredResult(
                                source=item.researcher_open_alex_id,
                                identifier=item.open_alex_results_models.document_model.doi,
                                reason="Already in database",
                            )
                            flattened_results.append(filtered_result)
                        elif link_quality == "has_only_poor_links" and extended_processing:
                            extended_in_db_count += 1
                            to_process_count += 1
                            item.is_extended = True
                            flattened_results.append(item)
                        elif link_quality == "has_only_poor_links" and not extended_processing:
                            already_in_db_count += 1
                            filtered_result = types.FilteredResult(
                                source=item.researcher_open_alex_id,
                                identifier=item.open_alex_results_models.document_model.doi,
                                reason="Already in database",
                            )
                            flattened_results.append(filtered_result)
                        else:
                            to_process_count += 1
                            flattened_results.append(item)

    total_filtered = no_doi_count + ignorable_doi_count + already_in_db_count
    print(
        f"Filtered {total_filtered} of {possible_count} author-articles: "
        f"{no_doi_count} no DOI, {ignorable_doi_count} ignorable DOI, "
        f"{already_in_db_count} already in DB. "
        f"{extended_in_db_count} extended (in DB with only poor links). "
        f"{to_process_count} to process."
    )

    return flattened_results


def _flatten_and_check_repositories_in_db(
    all_developer_repositories_and_errors: list[
        list[types.DeveloperRepositoryDetails | types.ErrorResult] | types.ErrorResult
    ],
    database_path: str,
    ignore_forks: bool,
    extended_processing: bool,
) -> list[types.DeveloperRepositoryDetails | types.FilteredResult | types.ErrorResult]:
    possible_count = 0
    to_process_count = 0
    extended_in_db_count = 0
    flattened_results: list[
        types.DeveloperRepositoryDetails | types.FilteredResult | types.ErrorResult
    ] = []
    for developer_repositories_and_errors in all_developer_repositories_and_errors:
        if isinstance(developer_repositories_and_errors, types.ErrorResult):
            flattened_results.append(developer_repositories_and_errors)
        else:
            for item in developer_repositories_and_errors:
                if isinstance(item, types.ErrorResult):
                    flattened_results.append(item)
                else:
                    possible_count += 1
                    link_quality = db_utils.check_repository_link_quality_in_db(
                        code_host=item.github_result_models.code_host_model.name,
                        repo_owner=item.github_result_models.repository_model.owner,
                        repo_name=item.github_result_models.repository_model.name,
                        database_path=database_path,
                    )
                    if link_quality == "has_good_link":
                        filtered_result = types.FilteredResult(
                            source=item.developer_account_username,
                            identifier=(
                                f"{item.github_result_models.repository_model.owner}/"
                                f"{item.github_result_models.repository_model.name}"
                            ),
                            reason="Already in database",
                        )
                        flattened_results.append(filtered_result)

                    elif link_quality == "has_only_poor_links" and extended_processing:
                        extended_in_db_count += 1
                        to_process_count += 1
                        item.is_extended = True
                        flattened_results.append(item)

                    elif link_quality == "has_only_poor_links" and not extended_processing:
                        filtered_result = types.FilteredResult(
                            source=item.developer_account_username,
                            identifier=(
                                f"{item.github_result_models.repository_model.owner}/"
                                f"{item.github_result_models.repository_model.name}"
                            ),
                            reason="Already in database",
                        )
                        flattened_results.append(filtered_result)

                    elif ignore_forks and item.github_result_models.repository_model.is_fork:
                        filtered_result = types.FilteredResult(
                            source=item.developer_account_username,
                            identifier=(
                                f"{item.github_result_models.repository_model.owner}/"
                                f"{item.github_result_models.repository_model.name}"
                            ),
                            reason="Repository is a fork and forks are being ignored",
                        )
                        flattened_results.append(filtered_result)

                    else:
                        to_process_count += 1
                        flattened_results.append(item)

    already_in_db_count = possible_count - to_process_count - extended_in_db_count
    print(
        f"Filtered out {already_in_db_count} developer-repositories already in DB. "
        f"{extended_in_db_count} extended (in DB with only poor links). "
        f"{to_process_count} to process."
    )

    return flattened_results


def _combine_to_possible_pairs(  # noqa: C901
    author_articles: list[
        types.AuthorArticleDetails | types.FilteredResult | types.ErrorResult
    ],
    developer_repositories: list[
        types.DeveloperRepositoryDetails | types.FilteredResult | types.ErrorResult
    ],
    negative_td: timedelta,
    positive_td: timedelta,
) -> list[types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair]:
    # Create a lookup of author_developer_link_id to list of author_articles
    author_articles_lut: dict[int, list[types.AuthorArticleDetails]] = {}
    filtered_author_articles: list[types.FilteredResult] = []
    errored_author_articles: list[types.ErrorResult] = []
    for item in author_articles:
        if isinstance(item, types.AuthorArticleDetails):
            if item.author_developer_link_id not in author_articles_lut:
                author_articles_lut[item.author_developer_link_id] = []
            author_articles_lut[item.author_developer_link_id].append(item)
        elif isinstance(item, types.FilteredResult):
            filtered_author_articles.append(item)
        elif isinstance(item, types.ErrorResult):
            errored_author_articles.append(item)

    # Create a lookup of author_developer_link_id to list of developer_repositories
    developer_repositories_lut: dict[int, list[types.DeveloperRepositoryDetails]] = {}
    filtered_developer_repositories: list[types.FilteredResult] = []
    errored_developer_repositories: list[types.ErrorResult] = []
    for item in developer_repositories:
        if isinstance(item, types.DeveloperRepositoryDetails):
            if item.author_developer_link_id not in developer_repositories_lut:
                developer_repositories_lut[item.author_developer_link_id] = []
            developer_repositories_lut[item.author_developer_link_id].append(item)
        elif isinstance(item, types.FilteredResult):
            filtered_developer_repositories.append(item)
        elif isinstance(item, types.ErrorResult):
            errored_developer_repositories.append(item)

    # Log observability info
    print(f"Author-developer links with articles: {len(author_articles_lut)}")
    print(f"Author-developer links with repos: {len(developer_repositories_lut)}")
    if errored_author_articles:
        _summarize_errors(errored_author_articles, "author-article")
    if errored_developer_repositories:
        _summarize_errors(errored_developer_repositories, "developer-repo")

    # Find common author_developer_link_ids
    common_author_developer_link_ids = set(author_articles_lut.keys()).intersection(
        set(developer_repositories_lut.keys())
    )
    print(
        f"Author-developer links with both articles AND repos: "
        f"{len(common_author_developer_link_ids)}"
    )

    # Combine author articles and developer repositories for common author_developer_link_ids
    unchecked_possible_combinations: list[
        types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair
    ] = []
    for author_developer_link_id in common_author_developer_link_ids:
        for author_article in author_articles_lut[author_developer_link_id]:
            for developer_repository in developer_repositories_lut[author_developer_link_id]:
                # Convert to datetimes
                article_published_date = (
                    author_article.open_alex_results_models.document_model.publication_date
                )
                article_published_dt = datetime(
                    year=article_published_date.year,
                    month=article_published_date.month,
                    day=article_published_date.day,
                )
                repo_created_dt = (
                    developer_repository.github_result_models.repository_model.creation_datetime
                )
                if repo_created_dt is None:
                    continue

                # Remove timezone info for comparison
                if article_published_dt.tzinfo is not None:
                    article_published_dt = article_published_dt.replace(tzinfo=None)
                if repo_created_dt.tzinfo is not None:
                    repo_created_dt = repo_created_dt.replace(tzinfo=None)

                # Must be within the allowed datetime difference
                datetime_difference = article_published_dt - repo_created_dt
                if negative_td < datetime_difference < positive_td:
                    unchecked_possible_combinations.append(
                        types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair(
                            author_developer_link_id=author_developer_link_id,
                            article_doi=author_article.open_alex_results_models.document_model.doi,
                            repository_identifier=(
                                f"{developer_repository.github_result_models.repository_model.owner}/"
                                f"{developer_repository.github_result_models.repository_model.name}"
                            ),
                            author_article=author_article,
                            developer_repository=developer_repository,
                        )
                    )

    return unchecked_possible_combinations


def _get_unique_repositories(
    unchecked_possible_combinations: list[
        types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair
    ],
) -> list[types.DeveloperRepositoryDetails]:
    unique_repos_set = set()
    unique_developer_repositories: list[types.DeveloperRepositoryDetails] = []
    for item in unchecked_possible_combinations:
        repo_identifier = item.repository_identifier
        if repo_identifier not in unique_repos_set:
            unique_repos_set.add(repo_identifier)
            unique_developer_repositories.append(item.developer_repository)

    return unique_developer_repositories


def _enrich_repository_with_data_required_for_matching(
    developer_repository: types.DeveloperRepositoryDetails,
    github_api_key: str,
) -> types.DeveloperRepositoryDetails | types.ErrorResult:
    result = github.process_github_repo(
        source="snowball-sampling-discovery",
        repo_parts=types.RepoParts(
            host=developer_repository.github_result_models.code_host_model.name,
            owner=developer_repository.github_result_models.repository_model.owner,
            name=developer_repository.github_result_models.repository_model.name,
        ),
        github_api_key=github_api_key,
        fetch_repo_data=False,
        fetch_repo_languages=False,
        fetch_repo_readme=True,
        fetch_repo_contributors=False,
        fetch_repo_commits_count=True,
        fetch_repo_files=False,
        existing_github_results=developer_repository.github_result_models,
    )

    if isinstance(result, types.ErrorResult):
        return result

    # Enrich the repository
    return types.DeveloperRepositoryDetails(
        author_developer_link_id=developer_repository.author_developer_link_id,
        developer_account_username=developer_repository.developer_account_username,
        github_result_models=result,
    )


def _replace_enriched_repositories_in_possible_combinations(
    unchecked_possible_combinations: list[
        types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair
    ],
    enriched_repositories: list[types.DeveloperRepositoryDetails | types.ErrorResult],
) -> list[types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair]:
    # Create a lookup of repository_identifier to enriched repository
    enriched_repos_lut: dict[str, types.DeveloperRepositoryDetails] = {}
    enrichment_error_count = 0
    for item in enriched_repositories:
        if isinstance(item, types.DeveloperRepositoryDetails):
            repo_identifier = (
                f"{item.github_result_models.repository_model.owner}/"
                f"{item.github_result_models.repository_model.name}"
            )
            enriched_repos_lut[repo_identifier] = item
        elif isinstance(item, types.ErrorResult):
            enrichment_error_count += 1

    if enrichment_error_count > 0:
        print(f"WARNING: {enrichment_error_count} repository enrichments failed.")

    # Replace developer repositories in unchecked_possible_combinations with enriched ones
    updated_combinations: list[
        types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair
    ] = []
    for item in unchecked_possible_combinations:
        repo_identifier = item.repository_identifier
        if repo_identifier in enriched_repos_lut:
            enriched_repo = enriched_repos_lut[repo_identifier]
            updated_combinations.append(
                types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair(
                    author_developer_link_id=item.author_developer_link_id,
                    article_doi=item.article_doi,
                    repository_identifier=item.repository_identifier,
                    author_article=item.author_article,
                    developer_repository=enriched_repo,
                )
            )

    dropped_count = len(unchecked_possible_combinations) - len(updated_combinations)
    if dropped_count > 0:
        print(
            f"WARNING: {dropped_count} combinations dropped "
            f"due to failed repository enrichment."
        )

    print(f"{len(updated_combinations)} combinations to match.")

    return updated_combinations


def _create_batches_for_matching(
    prepped_combinations: list[
        types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching | types.ErrorResult
    ],
    batch_size: int,
) -> list[list[types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching]]:
    error_count = sum(1 for item in prepped_combinations if isinstance(item, types.ErrorResult))
    if error_count > 0:
        print(
            f"WARNING: {error_count} prepped combinations were errors "
            f"and will be skipped for matching."
        )

    batches: list[list[types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching]] = []
    current_batch: list[types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching] = []
    for item in prepped_combinations:
        if isinstance(item, types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching):
            current_batch.append(item)
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

    # Add possible last batch
    if len(current_batch) > 0:
        batches.append(current_batch)

    return batches


def _filter_to_only_success_predictions(
    matched_results: list[
        list[types.MatchedAuthorArticleAndDeveloperRepositoryPair] | types.ErrorResult
    ],
) -> list[types.MatchedAuthorArticleAndDeveloperRepositoryPair]:
    # Flatten results
    flattened_results: list[types.MatchedAuthorArticleAndDeveloperRepositoryPair] = []
    errors: list[types.ErrorResult] = []
    for batch_result in matched_results:
        if isinstance(batch_result, types.ErrorResult):
            errors.append(batch_result)
        else:
            for item in batch_result:
                flattened_results.append(item)

    if errors:
        print(f"WARNING: {len(errors)} batch prediction errors:")
        for err in errors:
            print(f"  - {err.identifier}: {err.error}")

    return flattened_results


def _get_unique_and_highest_confidence_prediction_results(
    prediction_results: list[types.MatchedAuthorArticleAndDeveloperRepositoryPair],
) -> list[types.MatchedAuthorArticleAndDeveloperRepositoryPair]:
    # Create LUT of unique key to prediction result, keeping highest confidence
    results_lut: dict[
        tuple[str, str], types.MatchedAuthorArticleAndDeveloperRepositoryPair
    ] = {}
    for result in prediction_results:
        key = (result.article_doi, result.repository_identifier)
        if (
            key not in results_lut
            or result.matched_details.confidence > results_lut[key].matched_details.confidence
        ):
            results_lut[key] = result

    # Convert all results to polars dataframe with
    # article_doi, repository_identifier, confidence
    results_df = pl.DataFrame(
        [
            {
                "article_doi": result.article_doi,
                "repository_identifier": result.repository_identifier,
                "confidence": result.matched_details.confidence,
            }
            for result in prediction_results
        ]
    )

    if len(results_df) == 0:
        return []

    # Sort and get unique
    results_df = (
        results_df.sort("confidence", descending=True)
        .unique(
            "article_doi",
            maintain_order=True,
        )
        .unique(
            "repository_identifier",
            maintain_order=True,
        )
    )

    # Get final unique results by looking up in LUT
    return [
        results_lut[(row["article_doi"], row["repository_identifier"])]
        for row in results_df.iter_rows(named=True)
    ]


def _store_prediction_results(
    prediction_results: list[types.MatchedAuthorArticleAndDeveloperRepositoryPair],
    iteration: int,
) -> None:
    storage_file = DATA_FILES_DIR / "snowball-sampling-discovery-predictions.parquet"

    # Check if exists
    existing_results_df: pl.DataFrame | None
    if storage_file.exists():
        existing_results_df = pl.read_parquet(storage_file)
    else:
        existing_results_df = None

    # Prepare new results for storage
    new_results = pl.DataFrame(
        [
            {
                "article_doi": result.article_doi,
                "repository_identifier": f"https://github.com/{result.repository_identifier}",
                "confidence": result.matched_details.confidence,
                "author_developer_link_id": result.author_developer_link_id,
                "iteration": iteration,
                "model_name": result.matched_details.model_name,
                "model_version": result.matched_details.model_version,
            }
            for result in prediction_results
        ]
    )

    # Combine existing and new results
    if existing_results_df is None:
        combined_results_df = new_results
    else:
        combined_results_df = pl.concat([existing_results_df, new_results])

    # Save to parquet
    combined_results_df.write_parquet(storage_file)


def _update_processed_links_cache(
    iteration: int,
    link_counts: dict[int, int],
) -> None:
    cache_file = (
        DATA_FILES_DIR / f"snowball-sampling-processed-links-iteration-{iteration}.parquet"
    )

    # Check if exists
    if cache_file.exists():
        existing_df = pl.read_parquet(cache_file)
        existing_rows = existing_df.to_dicts()
    else:
        existing_rows = []

    # Prepare new rows
    new_rows = [
        {
            "researcher_developer_account_link_id": link_id,
            "document_repository_links_created": count,
        }
        for link_id, count in link_counts.items()
    ]

    # Combine and save
    combined_rows = existing_rows + new_rows
    combined_df = pl.DataFrame(combined_rows)
    combined_df.write_parquet(cache_file)


def _process_matched_article(
    matched_pair: types.MatchedAuthorArticleAndDeveloperRepositoryPair,
    open_alex_token: str,
    semantic_scholar_api_key: str,
) -> types.MatchedAuthorArticleAndDeveloperRepositoryPair | types.ErrorResult:
    # Try getting the rest of the data
    # Note: pyalex_work is typed as dict for JSON serialization, but is actually a pyalex.Work
    updated_open_alex_results = article.process_article(
        paper_doi=matched_pair.article_doi,
        source=f"snowball-sampling-discovery-v{rs_graph_version}",
        open_alex_token=open_alex_token,
        semantic_scholar_api_key=semantic_scholar_api_key,
        existing_pyalex_work=matched_pair.author_article.pyalex_work,  # type: ignore[arg-type]
        existing_open_alex_results=matched_pair.author_article.open_alex_results_models,
    )

    if isinstance(updated_open_alex_results, types.ErrorResult):
        return updated_open_alex_results

    # Return updated matched pair
    return types.MatchedAuthorArticleAndDeveloperRepositoryPair(
        author_developer_link_id=matched_pair.author_developer_link_id,
        article_doi=updated_open_alex_results.document_model.doi,
        repository_identifier=matched_pair.repository_identifier,
        author_article=types.AuthorArticleDetails(
            author_developer_link_id=matched_pair.author_article.author_developer_link_id,
            researcher_open_alex_id=matched_pair.author_article.researcher_open_alex_id,
            pyalex_work=matched_pair.author_article.pyalex_work,
            open_alex_results_models=updated_open_alex_results,
        ),
        developer_repository=matched_pair.developer_repository,
        matched_details=matched_pair.matched_details,
    )


def _process_matched_repository(
    matched_pair: types.MatchedAuthorArticleAndDeveloperRepositoryPair,
    github_api_key: str,
) -> types.MatchedAuthorArticleAndDeveloperRepositoryPair | types.ErrorResult:
    # Try getting the rest of the data
    updated_github_results = github.process_github_repo(
        source=f"snowball-sampling-discovery-v{rs_graph_version}",
        repo_parts=types.RepoParts(
            host=matched_pair.developer_repository.github_result_models.code_host_model.name,
            owner=matched_pair.developer_repository.github_result_models.repository_model.owner,
            name=matched_pair.developer_repository.github_result_models.repository_model.name,
        ),
        github_api_key=github_api_key,
        # Already have some of these
        fetch_repo_data=False,
        fetch_repo_languages=True,
        fetch_repo_readme=False,
        fetch_repo_contributors=True,
        fetch_repo_commits_count=False,
        fetch_repo_files=True,
        existing_github_results=matched_pair.developer_repository.github_result_models,
    )

    if isinstance(updated_github_results, types.ErrorResult):
        return updated_github_results

    # Return updated matched pair
    return types.MatchedAuthorArticleAndDeveloperRepositoryPair(
        author_developer_link_id=matched_pair.author_developer_link_id,
        article_doi=matched_pair.article_doi,
        repository_identifier=matched_pair.repository_identifier,
        author_article=matched_pair.author_article,
        developer_repository=types.DeveloperRepositoryDetails(
            author_developer_link_id=matched_pair.developer_repository.author_developer_link_id,
            developer_account_username=matched_pair.developer_repository.developer_account_username,
            github_result_models=updated_github_results,
        ),
        matched_details=matched_pair.matched_details,
    )


def _prep_updated_article_repository_details_for_storage_type(
    matched_pair: types.MatchedAuthorArticleAndDeveloperRepositoryPair,
    iteration: int,
    extended_processing: bool,
) -> types.ExpandedRepositoryDocumentPair:
    is_extended = extended_processing and (
        matched_pair.author_article.is_extended or matched_pair.developer_repository.is_extended
    )
    source = (
        "snowball-sampling-discovery-extended"
        if is_extended
        else f"snowball-sampling-discovery-v{rs_graph_version}"
    )
    open_alex_results = matched_pair.author_article.open_alex_results_models
    if is_extended:
        open_alex_results.dataset_source_model = db_models.DatasetSource(
            name="snowball-sampling-discovery-extended"
        )
    return types.ExpandedRepositoryDocumentPair(
        source=source,
        paper_doi=matched_pair.article_doi,
        paper_extra_data={
            "model_name": matched_pair.matched_details.model_name,
            "model_version": matched_pair.matched_details.model_version,
            "confidence": matched_pair.matched_details.confidence,
        },
        repo_parts=types.RepoParts(
            host=matched_pair.developer_repository.github_result_models.code_host_model.name,
            owner=matched_pair.developer_repository.github_result_models.repository_model.owner,
            name=matched_pair.developer_repository.github_result_models.repository_model.name,
        ),
        open_alex_results=open_alex_results,
        github_results=matched_pair.developer_repository.github_result_models,
        snowball_sampling_discovery_source_author_developer_link_id=(
            matched_pair.author_developer_link_id
        ),
        document_repository_link_metadata=types.DocumentRepositoryLinkMetadata(
            model_name=matched_pair.matched_details.model_name,
            model_version=matched_pair.matched_details.model_version,
            model_confidence=matched_pair.matched_details.confidence,
        ),
        iteration=iteration,
    )


@flow(
    log_prints=True,
)
def _snowball_sampling_discovery_flow(  # noqa: C901
    author_developer_links: list[db_utils.HydratedAuthorDeveloperLink],
    iteration: int,
    article_respository_allowed_datetime_difference_negative_td: timedelta,
    article_respository_allowed_datetime_difference_positive_td: timedelta,
    article_repository_matching_batch_size: int,
    ignore_forks: bool,
    ignorable_doi_spans: list[str],
    database_path: str,
    use_coiled: bool,
    coiled_region: str,
    cycled_github_tokens: GitHubTokensCycler,
    open_alex_tokens: list[str],
    semantic_scholar_api_key: str | None,
    extended_processing: bool = False,
) -> dict[int, int] | None:
    # Workers is the number of github tokens
    n_github_tokens = len(cycled_github_tokens)

    # Get an infinite cycle of open alex tokens
    cycled_open_alex_tokens = itertools.cycle(open_alex_tokens)

    # Get the number of open alex tokens
    n_open_alex_tokens = len(open_alex_tokens)

    # Preconstruct all wrapped tasks
    wrapped_get_articles_for_researcher = _wrap_func_with_coiled_prefect_task(
        _get_author_articles_for_researcher,
        coiled_func_name="open_alex_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=7,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
    )
    wrapped_get_repositories_for_developer = _wrap_func_with_coiled_prefect_task(
        _get_developer_repositories_for_developer,
        coiled_func_name="github_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_github_tokens,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
    )
    wrapped_enrich_repository = _wrap_func_with_coiled_prefect_task(
        _enrich_repository_with_data_required_for_matching,
        coiled_func_name="github_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_github_tokens,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
    )
    wrapped_match_prepped_pair = _wrap_func_with_coiled_prefect_task(
        entity_matching.match_articles_and_repositories,
        coiled_func_name="gpu_cluster",
        coiled_kwargs=_get_basic_gpu_cluster_config(
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
        environ={
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "HF_AUTH_TOKEN": os.environ["HF_AUTH_TOKEN"],
        },
    )
    process_article_wrapped_task = _wrap_func_with_coiled_prefect_task(
        _process_matched_article,
        coiled_func_name="open_alex_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_open_alex_tokens,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
    )
    process_github_wrapped_task = _wrap_func_with_coiled_prefect_task(
        _process_matched_repository,
        coiled_func_name="github_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_github_tokens,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
    )
    match_devs_and_researchers_wrapped_task = _wrap_func_with_coiled_prefect_task(
        entity_matching.match_devs_and_researchers,
        coiled_func_name="gpu_cluster",
        coiled_kwargs=_get_basic_gpu_cluster_config(
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
    )

    # Submit both article and repo fetches concurrently
    # (they use independent APIs and clusters)
    print("Getting each author's articles from Open Alex...")
    author_articles = wrapped_get_articles_for_researcher.map(
        author_developer_link_id=[
            link.author_developer_link_id for link in author_developer_links
        ],
        researcher_open_alex_id=[
            link.researcher_open_alex_id for link in author_developer_links
        ],
        open_alex_token=[
            next(cycled_open_alex_tokens) for _ in range(len(author_developer_links))
        ],
        semantic_scholar_api_key=unmapped(semantic_scholar_api_key),
    )

    print("Getting each developer's repositories from GitHub...")
    developer_repositories = wrapped_get_repositories_for_developer.map(
        author_developer_link_id=[
            link.author_developer_link_id for link in author_developer_links
        ],
        developer_account_username=[
            link.developer_account_username for link in author_developer_links
        ],
        github_api_key=[next(cycled_github_tokens) for _ in range(len(author_developer_links))],
    )

    # Now collect results from both (they've been running in parallel)
    if _shutdown_requested.is_set():
        print("Shutdown requested — exiting before filtering.")
        return None

    print("Filtering out articles already in the database...")
    flattened_author_articles = _flatten_and_check_articles_in_db(
        all_author_articles_and_errors=[aa.result() for aa in author_articles],
        database_path=database_path,
        ignorable_doi_spans=ignorable_doi_spans,
        extended_processing=extended_processing,
    )

    print("Filtering out repositories already in the database...")
    flattened_developer_repositories = _flatten_and_check_repositories_in_db(
        all_developer_repositories_and_errors=[dr.result() for dr in developer_repositories],
        database_path=database_path,
        ignore_forks=ignore_forks,
        extended_processing=extended_processing,
    )

    # Combine back together to possible article-repository pairs
    print("Combining to unchecked but possible article-repository pairs...")
    unchecked_possible_combinations = _combine_to_possible_pairs(
        author_articles=flattened_author_articles,
        developer_repositories=flattened_developer_repositories,
        negative_td=article_respository_allowed_datetime_difference_negative_td,
        positive_td=article_respository_allowed_datetime_difference_positive_td,
    )

    # Get the set of unique repositories for enrichment
    print("Getting unique repositories for enrichment...")
    unique_developer_repositories = _get_unique_repositories(
        unchecked_possible_combinations=unchecked_possible_combinations,
    )

    if _shutdown_requested.is_set():
        print("Shutdown requested — exiting before enrichment.")
        return None

    # Enrich unique repositories
    print("Enriching unique repositories...")
    enriched_repositories = wrapped_enrich_repository.map(
        developer_repository=unique_developer_repositories,
        github_api_key=[
            next(cycled_github_tokens) for _ in range(len(unique_developer_repositories))
        ],
    )

    # Replace enriched repositories back into possible combinations
    print("Replacing enriched repositories back into possible combinations...")
    possible_combinations_with_enriched_repos = (
        _replace_enriched_repositories_in_possible_combinations(
            unchecked_possible_combinations=unchecked_possible_combinations,
            enriched_repositories=[er.result() for er in enriched_repositories],
        )
    )

    if _shutdown_requested.is_set():
        print("Shutdown requested — exiting before matching.")
        return None

    # Prep for matching
    print("Preparing for article-repository matching...")
    prepped_combinations_for_matching = [
        entity_matching._prep_for_article_repository_matching(
            unchecked_possible_pair=unchecked_possible_pair,
        )
        for unchecked_possible_pair in possible_combinations_with_enriched_repos
    ]

    # Match in batches
    print("Creating batches for article-repository matching...")
    prepped_batches = _create_batches_for_matching(
        prepped_combinations=prepped_combinations_for_matching,
        batch_size=article_repository_matching_batch_size,
    )

    # Map and get results
    print("Matching article-repository pairs...")
    batched_matching_results = wrapped_match_prepped_pair.map(
        inference_ready_article_repository_pairs=prepped_batches,
    )

    # Get successful predictions
    print("Filtering to only successful predictions...")
    prediction_results = _filter_to_only_success_predictions(
        matched_results=[bmr.result() for bmr in batched_matching_results],
    )

    # Get unique prediction results
    print("Getting unique and highest confidence prediction results...")
    prediction_results = _get_unique_and_highest_confidence_prediction_results(
        prediction_results=prediction_results,
    )

    if _shutdown_requested.is_set():
        print("Shutdown requested — exiting before extended processing.")
        return None

    # Process all articles
    # Submit both article and repo processing in parallel
    # (they use independent APIs and clusters)
    print("Getting extended article data for predicted pairs...")
    updated_article_processing_futures = process_article_wrapped_task.map(
        matched_pair=prediction_results,
        open_alex_token=[next(cycled_open_alex_tokens) for _ in range(len(prediction_results))],
        semantic_scholar_api_key=unmapped(semantic_scholar_api_key),
    )

    print("Getting extended repository data for predicted pairs...")
    updated_github_futures = process_github_wrapped_task.map(
        matched_pair=prediction_results,
        github_api_key=[next(cycled_github_tokens) for _ in range(len(prediction_results))],
    )

    # Collect both sets of results (they've been running in parallel)
    print("Collecting and merging article + repository processing results...")
    gathered_article_results = [uapf.result() for uapf in updated_article_processing_futures]
    gathered_repo_results = [ugf.result() for ugf in updated_github_futures]

    # Merge: take article data from article results, repo data from repo results
    ready_for_storage = []
    for article_result, repo_result in zip(
        gathered_article_results, gathered_repo_results, strict=True
    ):
        if isinstance(article_result, types.ErrorResult):
            print(
                f"Error processing article for "
                f"{article_result.identifier}: {article_result.error}"
            )
            continue
        if isinstance(repo_result, types.ErrorResult):
            print(f"Error processing repo for {repo_result.identifier}: {repo_result.error}")
            continue

        merged_pair = types.MatchedAuthorArticleAndDeveloperRepositoryPair(
            author_developer_link_id=article_result.author_developer_link_id,
            article_doi=article_result.article_doi,
            repository_identifier=article_result.repository_identifier,
            author_article=article_result.author_article,
            developer_repository=repo_result.developer_repository,
            matched_details=article_result.matched_details,
        )
        ready_for_storage.append(
            _prep_updated_article_repository_details_for_storage_type(
                matched_pair=merged_pair,
                iteration=iteration,
                extended_processing=extended_processing,
            )
        )

    # Check for graceful shutdown before entering the critical storage section
    if _shutdown_requested.is_set():
        print("Shutdown requested — skipping storage for this batch.")
        return None

    # Store / extend results
    print("Storing prediction results...")
    _store_prediction_results(
        prediction_results=prediction_results,
        iteration=iteration,
    )

    # Store everything
    print("Storing full details of article-repository pairs...")
    stored_pairs = []
    for rfs in ready_for_storage:
        if isinstance(rfs, types.ErrorResult):
            print(
                f"Error preparing article-repository pair for storage for "
                f"author-developer link ID "
                f"{rfs.source}: {rfs.error}"
            )

        else:
            stored_pair = db_utils.store_full_details(
                pair=rfs,
                database_path=database_path,
            )
            if isinstance(stored_pair, types.ErrorResult):
                print(
                    f"Error storing full details for "
                    f"author-developer link ID "
                    f"{stored_pair.source}: {stored_pair.error}"
                )
            else:
                stored_pairs.append(stored_pair)
            time.sleep(0.05)

    # Match devs and researchers
    print("Matching developers and researchers...")
    dev_researcher_futures = match_devs_and_researchers_wrapped_task.map(
        pair=stored_pairs,
    )

    # Store the dev-researcher links
    print("Storing developer-researcher links...")
    gathered_dev_researcher_futures = [drf.result() for drf in dev_researcher_futures]
    stored_dev_researchers = []
    for drf in gathered_dev_researcher_futures:
        if isinstance(drf, types.ErrorResult):
            print(
                f"Error matching dev-researcher link for author-developer link ID "
                f"{drf.source}: {drf.error}"
            )
        else:
            stored_dev_researcher = db_utils.store_dev_researcher_em_links(
                pair=drf,
                database_path=database_path,
            )
            stored_dev_researchers.append(stored_dev_researcher)
            time.sleep(0.05)

    # Note that we have now processed this author-developer link
    print("Marking snowball source author-developer links as processed...")
    for adl in author_developer_links:
        db_utils.update_researcher_developer_account_link_with_new_process_dt(
            link_id=adl.author_developer_link_id,
            database_path=database_path,
        )
        time.sleep(0.05)

    # Count document-repository links created per source author-developer link
    doc_repo_link_counts: dict[int, int] = {
        adl.author_developer_link_id: 0 for adl in author_developer_links
    }
    for sp in stored_pairs:
        source_id = sp.snowball_sampling_discovery_source_author_developer_link_id
        if source_id is not None and source_id in doc_repo_link_counts:
            doc_repo_link_counts[source_id] += 1

    return doc_repo_link_counts


#######################################################################################

ignorable_doi_spans_default = typer.Option(
    default=["zenodo", "figshare"],
    help=(
        "List of DOI spans to ignore when checking if an article is already in the database."
    ),
)


@app.command()
def snowball_sampling_discovery(
    researcher_developer_account_links_parquet_file: str = typer.Argument(
        help=(
            "Path to a parquet file with a 'researcher_developer_account_link_id' "
            "column containing the IDs of ResearcherDeveloperAccountLink records "
            "to process."
        ),
    ),
    database_path: str = typer.Argument(
        help="Path to the SQLite database file to use.",
    ),
    article_repository_allowed_datetime_difference_positive: str = "556 days",
    article_repository_allowed_datetime_difference_negative: str = "73 days",
    author_developer_links_batch_size: int = 24,
    article_repository_matching_batch_size: int = 32,
    ignore_forks: bool = True,
    ignorable_doi_spans: list[str] = ignorable_doi_spans_default,
    use_coiled: bool = False,
    coiled_region: str = "us-west-2",
    github_tokens_file: str = DEFAULT_GITHUB_TOKENS_FILE,
    open_alex_tokens_file: str = DEFAULT_OPEN_ALEX_TOKENS_FILE,
    extended_processing: bool = False,
    shuffle_researcher_developer_links: bool = False,
) -> None:
    """
    Discover new article-repository pairs via snowball sampling.

    Reads a parquet file containing 'researcher_developer_account_link_id' values,
    hydrates them from the database, then looks up each author's articles
    and their repositories, uses our article-repository matching model
    to predict new pairs, and then conducts standard processing.
    """
    # Install graceful shutdown handler
    signal.signal(signal.SIGINT, _handle_sigint)

    # Load environment variables
    load_dotenv()

    # Get open alex tokens
    open_alex_tokens = _load_open_alex_tokens(open_alex_tokens_file)

    # Get semantic scholar API key
    try:
        semantic_scholar_api_key = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    except KeyError as e:
        raise KeyError("Please set the SEMANTIC_SCHOLAR_API_KEY environment variable.") from e

    # Ignore prefect task introspection warnings
    os.environ["PREFECT_TASK_INTROSPECTION_WARN_THRESHOLD"] = "0"

    # Get an infinite cycle of github tokens
    cycled_github_tokens = GitHubTokensCycler(gh_tokens_file=github_tokens_file)

    # Workers is the number of github tokens
    n_github_tokens = len(cycled_github_tokens)

    # Get the number of open alex tokens
    n_open_alex_tokens = len(open_alex_tokens)

    # Parse timedeltas
    article_respository_allowed_datetime_difference_negative_td = (
        parse_timedelta(article_repository_allowed_datetime_difference_negative) * -1
    )
    article_respository_allowed_datetime_difference_positive_td = parse_timedelta(
        article_repository_allowed_datetime_difference_positive
    )

    # Print dataset and coiled status
    print("-" * 80)
    print("Pipeline Options:")
    print(
        f"Researcher Developer Account Links Parquet File: "
        f"{researcher_developer_account_links_parquet_file}"
    )
    print(
        f"Article Repository Allowed Datetime Difference: "
        f"{article_respository_allowed_datetime_difference_negative_td.days} to "
        f"{article_respository_allowed_datetime_difference_positive_td.days}"
    )
    print(f"Database Path: {database_path}")
    print(f"Use Coiled: {use_coiled}")
    print(f"Coiled Region: {coiled_region}")
    print(f"GitHub Token Count: {n_github_tokens}")
    print(f"Open Alex Token Count: {n_open_alex_tokens}")
    print(f"Extended Processing: {extended_processing}")
    print("-" * 80)

    # Read the parquet file to get the link IDs
    print(
        f"Reading researcher-developer-account link IDs from: "
        f"{researcher_developer_account_links_parquet_file}"
    )
    link_ids_df = pl.read_parquet(researcher_developer_account_links_parquet_file)

    # Shuffle if desired
    if shuffle_researcher_developer_links:
        link_ids_df = link_ids_df.sample(fraction=1.0, shuffle=True)

    link_ids: list[int] = link_ids_df["researcher_developer_account_link_id"].to_list()
    iteration: int = link_ids_df["iteration"][0]
    print(f"Found {len(link_ids)} researcher-developer-account link IDs in parquet file.")
    print(f"Iteration: {iteration}")

    # Hydrate the links from the database
    print("Hydrating researcher-developer-account links from the database...")
    hydrated_author_developer_links = db_utils.get_hydrated_author_developer_links_by_ids(
        link_ids=link_ids,
        database_path=database_path,
    )

    # Check for already-processed links from a prior run of this iteration
    cache_file = (
        DATA_FILES_DIR / f"snowball-sampling-processed-links-iteration-{iteration}.parquet"
    )
    if cache_file.exists():
        already_processed_df = pl.read_parquet(cache_file)
        already_processed_ids: set[int] = set(
            already_processed_df["researcher_developer_account_link_id"].to_list()
        )
        original_count = len(hydrated_author_developer_links)
        hydrated_author_developer_links = [
            link
            for link in hydrated_author_developer_links
            if link.author_developer_link_id not in already_processed_ids
        ]
        skipped_count = original_count - len(hydrated_author_developer_links)
        print(f"Skipped {skipped_count} already-processed links (from cache: {cache_file}).")

    # Iter over author-developer links in batches
    print(f"Processing {len(hydrated_author_developer_links)} author-developer links...")
    add_remainder_batch = (
        1 if len(hydrated_author_developer_links) % author_developer_links_batch_size > 0 else 0
    )
    total_n_batches = (
        len(hydrated_author_developer_links) // author_developer_links_batch_size
        + add_remainder_batch
    )
    for author_developer_index in tqdm(
        range(
            0,
            len(hydrated_author_developer_links),
            author_developer_links_batch_size,
        ),
        desc="Author-Developer Link Batches",
        total=total_n_batches,
    ):
        author_developer_link_batch = hydrated_author_developer_links[
            author_developer_index : author_developer_index + author_developer_links_batch_size
        ]
        batch_link_counts: dict[int, int] | None = {
            link.author_developer_link_id: 0 for link in author_developer_link_batch
        }

        try:
            # Start the flow
            batch_link_counts = _snowball_sampling_discovery_flow(
                author_developer_links=author_developer_link_batch,
                iteration=iteration,
                article_respository_allowed_datetime_difference_negative_td=(
                    article_respository_allowed_datetime_difference_negative_td
                ),
                article_respository_allowed_datetime_difference_positive_td=(
                    article_respository_allowed_datetime_difference_positive_td
                ),
                article_repository_matching_batch_size=article_repository_matching_batch_size,
                ignore_forks=ignore_forks,
                ignorable_doi_spans=ignorable_doi_spans,
                database_path=database_path,
                use_coiled=use_coiled,
                coiled_region=coiled_region,
                cycled_github_tokens=cycled_github_tokens,
                open_alex_tokens=open_alex_tokens,
                semantic_scholar_api_key=semantic_scholar_api_key,
                extended_processing=extended_processing,
            )

        except Exception as e:
            print(
                f"Something went wrong processing author-developer link batch "
                f"starting at index {author_developer_index}."
            )
            print("Error:", str(e))
            print(traceback.format_exc())

        finally:
            if batch_link_counts is not None:
                _update_processed_links_cache(
                    iteration=iteration,
                    link_counts=batch_link_counts,
                )

        time.sleep(1)

        if _shutdown_requested.is_set():
            print("Shutdown requested — exiting after completing current batch.")
            break


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
