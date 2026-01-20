#!/usr/bin/env python

import itertools
import os
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

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
    DEFAULT_OPEN_ALEX_EMAILS_FILE,
    _get_basic_gpu_cluster_config,
    _get_small_cpu_api_cluster,
    _load_coiled_software_envs,
    _load_open_alex_emails,
    _wrap_func_with_coiled_prefect_task,
)
from rs_graph.db import utils as db_utils
from rs_graph.enrichment import article, entity_matching, github
from rs_graph.utils.dt_and_td import parse_timedelta

###############################################################################

app = typer.Typer()

###############################################################################


def _get_author_articles_for_researcher(
    author_developer_link_id: int,
    researcher_open_alex_id: str,
    open_alex_email: str,
    open_alex_email_count: int,
    semantic_scholar_api_key: str,
) -> list[types.AuthorArticleDetails | types.ErrorResult] | types.ErrorResult:
    # Get the articles for the researcher
    researcher_articles = article.get_articles_for_researcher(
        researcher_open_alex_id=researcher_open_alex_id,
        open_alex_email=open_alex_email,
        open_alex_email_count=open_alex_email_count,
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
    use_prod: bool,
    ignorable_doi_spans: list[str],
) -> list[types.AuthorArticleDetails | types.FilteredResult | types.ErrorResult]:
    possible_count = 0
    to_process_count = 0
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
                        filtered_result = types.FilteredResult(
                            source=item.researcher_open_alex_id,
                            identifier=item.open_alex_results_models.document_model.doi,
                            reason=f"DOI in ignorable spans ({', '.join(ignorable_doi_spans)})",
                        )
                        flattened_results.append(filtered_result)

                    else:
                        # Check if article is already in DB
                        exists_in_db = db_utils.check_article_in_db(
                            article_doi=item.open_alex_results_models.document_model.doi,
                            article_title=item.open_alex_results_models.document_model.title,
                            use_prod=use_prod,
                        )
                        if exists_in_db:
                            filtered_result = types.FilteredResult(
                                source=item.researcher_open_alex_id,
                                identifier=item.open_alex_results_models.document_model.doi,
                                reason="Already in database",
                            )
                            flattened_results.append(filtered_result)
                        else:
                            to_process_count += 1
                            flattened_results.append(item)

    print(f"Filtered out {possible_count - to_process_count} author-articles already in DB.")

    return flattened_results


def _flatten_and_check_repositories_in_db(
    all_developer_repositories_and_errors: list[
        list[types.DeveloperRepositoryDetails | types.ErrorResult] | types.ErrorResult
    ],
    use_prod: bool,
    ignore_forks: bool,
) -> list[types.DeveloperRepositoryDetails | types.FilteredResult | types.ErrorResult]:
    possible_count = 0
    to_process_count = 0
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
                    # Check if repository is already in DB
                    exists_in_db = db_utils.check_repository_in_db(
                        code_host=item.github_result_models.code_host_model.name,
                        repo_owner=item.github_result_models.repository_model.owner,
                        repo_name=item.github_result_models.repository_model.name,
                        use_prod=use_prod,
                    )
                    if exists_in_db:
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

    print(
        f"Filtered out {possible_count - to_process_count} "
        f"developer-repositories already in DB."
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

    # Find common author_developer_link_ids
    common_author_developer_link_ids = set(author_articles_lut.keys()).intersection(
        set(developer_repositories_lut.keys())
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
                repo_created_dt: datetime = (
                    developer_repository.github_result_models.repository_model.creation_datetime
                )

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
    for item in enriched_repositories:
        if isinstance(item, types.DeveloperRepositoryDetails):
            repo_identifier = (
                f"{item.github_result_models.repository_model.owner}/"
                f"{item.github_result_models.repository_model.name}"
            )
            enriched_repos_lut[repo_identifier] = item

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

    assert all(
        isinstance(combination, types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair)
        for combination in updated_combinations
    ), "Not all combinations were successfully updated."

    print(f"{len(updated_combinations)} combinations to match.")

    return updated_combinations


def _create_batches_for_matching(
    prepped_combinations: list[
        types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching | types.ErrorResult
    ],
    batch_size: int,
) -> list[list[types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching]]:
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
    errors = []
    for batch_result in matched_results:
        if isinstance(batch_result, types.ErrorResult):
            errors.append(batch_result)
        else:
            for item in batch_result:
                flattened_results.append(item)

    return flattened_results


def _get_unique_and_highest_confidence_prediction_results(
    prediction_results: list[types.MatchedAuthorArticleAndDeveloperRepositoryPair],
) -> list[types.MatchedAuthorArticleAndDeveloperRepositoryPair]:
    # Create LUT of unique key to prediction result
    results_lut = {
        (
            result.article_doi,
            result.repository_identifier,
        ): result
        for result in prediction_results
    }

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
) -> None:
    storage_file = Path("snowball-sampling-discovery-predictions.csv")

    # Check if exists
    if storage_file.exists():
        existing_results_df = pl.read_csv(storage_file)
        existing_results = existing_results_df.to_dicts()
    else:
        existing_results = []

    # Prepare new results for storage
    new_results = [
        {
            "article_doi": result.article_doi,
            "repository_identifier": f"https://github.com/{result.repository_identifier}",
            "confidence": result.matched_details.confidence,
        }
        for result in prediction_results
    ]

    # Combine existing and new results
    combined_results = existing_results + new_results

    # Save to CSV
    combined_results_df = pl.DataFrame(combined_results)
    combined_results_df.write_csv(storage_file)


def _process_matched_article(
    matched_pair: types.MatchedAuthorArticleAndDeveloperRepositoryPair,
    open_alex_email: str,
    open_alex_email_count: int,
    semantic_scholar_api_key: str,
) -> types.MatchedAuthorArticleAndDeveloperRepositoryPair | types.ErrorResult:
    # Try getting the rest of the data
    # Note: pyalex_work is typed as dict for JSON serialization, but is actually a pyalex.Work
    updated_open_alex_results = article.process_article(
        paper_doi=matched_pair.article_doi,
        source=f"snowball-sampling-discovery-v{rs_graph_version}",
        open_alex_email=open_alex_email,
        open_alex_email_count=open_alex_email_count,
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
    matched_pair: types.MatchedAuthorArticleAndDeveloperRepositoryPair | types.ErrorResult,
    github_api_key: str,
) -> types.MatchedAuthorArticleAndDeveloperRepositoryPair | types.ErrorResult:
    if isinstance(matched_pair, types.ErrorResult):
        return matched_pair

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
) -> types.ExpandedRepositoryDocumentPair:
    return types.ExpandedRepositoryDocumentPair(
        source=f"snowball-sampling-discovery-v{rs_graph_version}",
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
        open_alex_results=matched_pair.author_article.open_alex_results_models,
        github_results=matched_pair.developer_repository.github_result_models,
        snowball_sampling_discovery_source_author_developer_link_id=(
            matched_pair.author_developer_link_id
        ),
        document_repository_link_metadata=types.DocumentRepositoryLinkMetadata(
            model_name=matched_pair.matched_details.model_name,
            model_version=matched_pair.matched_details.model_version,
            model_confidence=matched_pair.matched_details.confidence,
        ),
    )


@flow(
    log_prints=True,
)
def _snowball_sampling_discovery_flow(
    author_developer_links: list[db_utils.HydratedAuthorDeveloperLink],
    article_respository_allowed_datetime_difference_negative_td: timedelta,
    article_respository_allowed_datetime_difference_positive_td: timedelta,
    article_repository_matching_batch_size: int,
    ignore_forks: bool,
    ignorable_doi_spans: list[str],
    use_prod: bool,
    use_coiled: bool,
    coiled_region: str,
    cycled_github_tokens: GitHubTokensCycler,
    open_alex_emails: list[str],
    semantic_scholar_api_key: str | None,
    coiled_software_envs: dict[str, str] | None,
    cache_coiled_software_envs: bool,
    coiled_software_envs_file: str,
) -> None:
    # Workers is the number of github tokens
    n_github_tokens = len(cycled_github_tokens)

    # Get an infinite cycle of open alex emails
    cycled_open_alex_emails = itertools.cycle(open_alex_emails)

    # Get the number of open alex emails
    n_open_alex_emails = len(open_alex_emails)

    # Preconstruct all wrapped tasks
    wrapped_get_articles_for_researcher = _wrap_func_with_coiled_prefect_task(
        _get_author_articles_for_researcher,
        coiled_func_name="open_alex_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_open_alex_emails,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
            software_env_name="package-sync-49802be2a88402f01f48cec991bd301a",
        ),
    )
    wrapped_get_repositories_for_developer = _wrap_func_with_coiled_prefect_task(
        _get_developer_repositories_for_developer,
        coiled_func_name="github_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_github_tokens,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
            software_env_name="package-sync-49802be2a88402f01f48cec991bd301a",
        ),
    )
    wrapped_enrich_repository = _wrap_func_with_coiled_prefect_task(
        _enrich_repository_with_data_required_for_matching,
        coiled_func_name="github_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_github_tokens,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
            software_env_name="package-sync-49802be2a88402f01f48cec991bd301a",
        ),
    )
    wrapped_match_prepped_pair = _wrap_func_with_coiled_prefect_task(
        entity_matching.match_articles_and_repositories,
        coiled_func_name="gpu_cluster",
        coiled_kwargs=_get_basic_gpu_cluster_config(
            use_coiled=use_coiled,
            coiled_region=coiled_region,
            software_env_name="package-sync-03ee32789d90159bf532a979364b3917",
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
            n_workers=n_open_alex_emails,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
            software_env_name="package-sync-49802be2a88402f01f48cec991bd301a",
        ),
    )
    process_github_wrapped_task = _wrap_func_with_coiled_prefect_task(
        _process_matched_repository,
        coiled_func_name="github_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_github_tokens,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
            software_env_name="package-sync-49802be2a88402f01f48cec991bd301a",
        ),
    )
    match_devs_and_researchers_wrapped_task = _wrap_func_with_coiled_prefect_task(
        entity_matching.match_devs_and_researchers,
        coiled_func_name="gpu_cluster",
        coiled_kwargs=_get_basic_gpu_cluster_config(
            use_coiled=use_coiled,
            coiled_region=coiled_region,
            software_env_name="package-sync-03ee32789d90159bf532a979364b3917",
        ),
    )

    # Get each authors articles
    print("Getting each author's articles from Open Alex...")
    author_articles = wrapped_get_articles_for_researcher.map(
        author_developer_link_id=[
            link.author_developer_link_id for link in author_developer_links
        ],
        researcher_open_alex_id=[
            link.researcher_open_alex_id for link in author_developer_links
        ],
        open_alex_email=[
            next(cycled_open_alex_emails) for _ in range(len(author_developer_links))
        ],
        open_alex_email_count=unmapped(n_open_alex_emails),
        semantic_scholar_api_key=unmapped(semantic_scholar_api_key),
    )

    # Drop existing articles already stored in the database
    print("Filtering out articles already in the database...")
    flattened_author_articles = _flatten_and_check_articles_in_db(
        all_author_articles_and_errors=[aa.result() for aa in author_articles],
        use_prod=use_prod,
        ignorable_doi_spans=ignorable_doi_spans,
    )

    # Get each developer's repositories
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

    # Drop existing repositories already stored in the database
    print("Filtering out repositories already in the database...")
    flattened_developer_repositories = _flatten_and_check_repositories_in_db(
        all_developer_repositories_and_errors=[dr.result() for dr in developer_repositories],
        use_prod=use_prod,
        ignore_forks=ignore_forks,
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

    # Store / extend results
    print("Storing prediction results...")
    _store_prediction_results(
        prediction_results=prediction_results,
    )

    # Process all articles
    print("Getting extended article data for predicted pairs...")
    updated_article_processing_futures = process_article_wrapped_task.map(
        matched_pair=prediction_results,
        open_alex_email=[next(cycled_open_alex_emails) for _ in range(len(prediction_results))],
        open_alex_email_count=unmapped(n_open_alex_emails),
        semantic_scholar_api_key=unmapped(semantic_scholar_api_key),
    )

    # Process github
    print("Getting extended repository data for predicted pairs...")
    updated_github_futures = process_github_wrapped_task.map(
        matched_pair=[uapf.result() for uapf in updated_article_processing_futures],
        github_api_key=[
            next(cycled_github_tokens) for _ in range(len(updated_article_processing_futures))
        ],
    )

    # Convert to expanded pair type
    print("Preparing updated article-repository details for storage...")
    gathered_github_futures = [ugf.result() for ugf in updated_github_futures]
    ready_for_storage = [
        _prep_updated_article_repository_details_for_storage_type(
            matched_pair=ggf,
        )
        for ggf in gathered_github_futures
        if not isinstance(ggf, types.ErrorResult)
    ]

    # Store everything
    print("Storing full details of article-repository pairs...")
    stored_pairs = []
    for rfs in ready_for_storage:
        if isinstance(rfs, types.ErrorResult):
            print(
                f"Error preparing article-repository pair for storage for "
                f"author-developer link ID "
                f"{rfs.source}: {rfs.traceback}"
            )

        else:
            stored_pair = db_utils.store_full_details(
                pair=rfs,
                use_prod=use_prod,
            )
            stored_pairs.append(stored_pair)
            time.sleep(0.25)

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
                f"{drf.source}: {drf.traceback}"
            )
        else:
            stored_dev_researcher = db_utils.store_dev_researcher_em_links(
                pair=drf,
                use_prod=use_prod,
            )
            stored_dev_researchers.append(stored_dev_researcher)
            time.sleep(0.25)

    # Note that we have now processed this author-developer link
    print("Marking snowball source author-developer links as processed...")
    for adl in author_developer_links:
        db_utils.update_researcher_developer_account_link_with_new_process_dt(
            link_id=adl.author_developer_link_id,
            use_prod=use_prod,
        )
        time.sleep(0.25)


#######################################################################################

ignorable_doi_spans_default = typer.Option(
    default=["zenodo", "figshare"],
    help=(
        "List of DOI spans to ignore when checking if an article is already in the database."
    ),
)


@app.command()
def snowball_sampling_discovery(
    process_n_author_developer_pairs: int = 10,
    article_repository_allowed_datetime_difference_positive: str = "374 days",
    article_repository_allowed_datetime_difference_negative: str = "326 days",
    author_developer_links_filter_confidence_threshold: float = 0.97,
    author_developer_links_duration_since_last_process: str = "2 years",
    author_developer_links_batch_size: int = 4,
    article_repository_matching_batch_size: int = 24,
    ignore_forks: bool = True,
    ignorable_doi_spans: list[str] = ignorable_doi_spans_default,
    use_prod: bool = False,
    use_coiled: bool = False,
    coiled_region: str = "us-west-2",
    github_tokens_file: str = DEFAULT_GITHUB_TOKENS_FILE,
    open_alex_emails_file: str = DEFAULT_OPEN_ALEX_EMAILS_FILE,
    coiled_software_envs_file: str = ".coiled-software-envs.yml",
    use_cached_coiled_software_envs: bool = False,
    cache_coiled_software_envs: bool = False,
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
    open_alex_emails = _load_open_alex_emails(open_alex_emails_file)

    # Load coiled software envs
    if use_cached_coiled_software_envs:
        coiled_software_envs = _load_coiled_software_envs(coiled_software_envs_file)
    else:
        coiled_software_envs = None

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

    # Get the number of open alex emails
    n_open_alex_emails = len(open_alex_emails)

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
    print(f"Process N Author-Developer Pairs: {process_n_author_developer_pairs}")
    print(
        f"Article Repository Allowed Datetime Difference: "
        f"{article_respository_allowed_datetime_difference_negative_td.days} to "
        f"{article_respository_allowed_datetime_difference_positive_td.days}"
    )
    print(
        f"Author Developer Links Filter Confidence Threshold: "
        f"{author_developer_links_filter_confidence_threshold}"
    )
    print(
        f"Skip Author Developer Links Processed Within Last: "
        f"{author_developer_links_duration_since_last_process}"
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
        filter_datetime_difference=author_developer_links_duration_since_last_process,
        filter_confidence_threshold=author_developer_links_filter_confidence_threshold,
        n=process_n_author_developer_pairs,
    )

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
        try:
            author_developer_link_batch = hydrated_author_developer_links[
                author_developer_index : author_developer_index
                + author_developer_links_batch_size
            ]

            # Start the flow
            _snowball_sampling_discovery_flow(
                author_developer_links=author_developer_link_batch,
                article_respository_allowed_datetime_difference_negative_td=(
                    article_respository_allowed_datetime_difference_negative_td
                ),
                article_respository_allowed_datetime_difference_positive_td=(
                    article_respository_allowed_datetime_difference_positive_td
                ),
                article_repository_matching_batch_size=article_repository_matching_batch_size,
                ignore_forks=ignore_forks,
                ignorable_doi_spans=ignorable_doi_spans,
                use_prod=use_prod,
                use_coiled=use_coiled,
                coiled_region=coiled_region,
                cycled_github_tokens=cycled_github_tokens,
                open_alex_emails=open_alex_emails,
                semantic_scholar_api_key=semantic_scholar_api_key,
                coiled_software_envs=coiled_software_envs,
                cache_coiled_software_envs=cache_coiled_software_envs,
                coiled_software_envs_file=coiled_software_envs_file,
            )

        except Exception as e:
            print(
                f"Something went wrong processing author-developer link batch "
                f"starting at index {author_developer_index}."
            )
            print("Error:", str(e))
            print(traceback.format_exc())
            continue

        time.sleep(10)


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
