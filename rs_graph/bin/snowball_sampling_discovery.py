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
    for open_alex_result in researcher_articles:
        if isinstance(open_alex_result, types.ErrorResult):
            author_article_details_list.append(open_alex_result)
        else:
            author_article_details = types.AuthorArticleDetails(
                author_developer_link_id=author_developer_link_id,
                researcher_open_alex_id=researcher_open_alex_id,
                open_alex_results_models=open_alex_result,
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
    max_datetime_difference: timedelta,
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
                repo_created_dt = (
                    developer_repository.github_result_models.repository_model.creation_datetime
                )

                # Remove timezone info for comparison
                if article_published_dt.tzinfo is not None:
                    article_published_dt = article_published_dt.replace(tzinfo=None)
                if repo_created_dt.tzinfo is not None:
                    repo_created_dt = repo_created_dt.replace(tzinfo=None)

                if abs(article_published_dt - repo_created_dt) <= max_datetime_difference:
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


def _wrapped_prep_for_matching(
    unchecked_possible_pair: types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair,
) -> types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching | types.ErrorResult:
    return entity_matching._prep_for_article_repository_matching(
        unchecked_possible_pair=unchecked_possible_pair,
    )


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

    # Store each result
    # storage_results = []
    # for item in flattened_results:
    #     storage_results.append(
    #         {
    #             "author_developer_link_id": item.author_developer_link_id,
    #             "article_doi": item.article_doi,
    #             "repository_identifier": f"https://github.com/{item.repository_identifier}",
    #             "model_name": item.matched_details.model_name,
    #             "model_version": item.matched_details.model_version,
    #             "confidence": item.matched_details.confidence,
    #         }
    #     )

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


@flow(
    log_prints=True,
)
def _snowball_sampling_discovery_flow(
    author_developer_links: list[db_utils.HydratedAuthorDeveloperLink],
    article_repository_allowed_datetime_difference: str,
    article_repository_matching_batch_size: int,
    ignore_forks: bool,
    ignorable_doi_spans: list[str],
    use_prod: bool,
    use_coiled: bool,
    coiled_region: str,
    github_tokens_file: str,
    open_alex_emails: list[str],
    semantic_scholar_api_key: str | None,
    coiled_software_envs: dict[str, str] | None,
    cache_coiled_software_envs: bool,
    coiled_software_envs_file: str,
) -> None:
    # Get an infinite cycle of github tokens
    cycled_github_tokens = GitHubTokensCycler(gh_tokens_file=github_tokens_file)

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
        coiled_func_name="entity_matching_cluster",
        coiled_kwargs=_get_basic_gpu_cluster_config(
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
        environ={
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "HF_AUTH_TOKEN": os.environ["HF_AUTH_TOKEN"],
        },
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

    # TODO: handle storage of errors and filters

    # Combine back together to possible article-repository pairs
    print("Combining to unchecked but possible article-repository pairs...")
    unchecked_possible_combinations = _combine_to_possible_pairs(
        author_articles=flattened_author_articles,
        developer_repositories=flattened_developer_repositories,
        max_datetime_difference=parse_timedelta(article_repository_allowed_datetime_difference),
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

    # Get unique set of repositories
    # print("Getting unique repositories for enrichment...")
    # unique_developer_repositories = _get_unique_repositories(
    #     developer_repositories_filtered=developer_repositories_filtered,
    # )

    # # Print throughput stats
    # print("Throughput Stats So Far:")
    # print(f"Total Possible Combinations: {total_possible_combinations}")
    # print(f"Total Possible Pairs: {total_possible_pairs}")
    # print(f"Total Possible Pairs After Filtering: {total_possible_pairs_after_filtering}")
    # print(f"Total Possible Pairs For Inference: {total_possible_pairs_for_inference}")
    # print(f"Total Matches Found: {total_matches_found}")
    # print()

    # # Now, normalized by total author-developer links processed
    # n_author_developer_links_processed = author_developer_index + len(batch)
    # print("Normalized By Author-Developer Links Processed:")
    # print(f"Author-Developer Links Processed: {n_author_developer_links_processed}")
    # print(
    #     f"Possible Combinations Per Author-Developer Link: "
    #     f"{total_possible_combinations / n_author_developer_links_processed:.2f}"
    # )
    # print(
    #     f"Possible Pairs Per Author-Developer Link: "
    #     f"{total_possible_pairs / n_author_developer_links_processed:.2f}"
    # )
    # print(
    #     f"Possible Pairs After Filtering Per Author-Developer Link: "
    #     f"{total_possible_pairs_after_filtering / n_author_developer_links_processed:.2f}"
    # )
    # print(
    #     f"Possible Pairs For Inference Per Author-Developer Link: "
    #     f"{total_possible_pairs_for_inference / n_author_developer_links_processed:.2f}"
    # )
    # print(
    #     f"Matches Found Per Author-Developer Link: "
    #     f"{total_matches_found / n_author_developer_links_processed:.2f}"
    # )
    # print()

    # # Get batch end time and normalize by duration
    # batch_end_time = time.time()
    # total_duration = batch_end_time - start_time
    # print("Normalized By Processing Time:")
    # print(f"Total Duration: {total_duration:.2f} seconds")
    # print(
    #     f"Author-Developer Links Processed Per Second: "
    #     f"{n_author_developer_links_processed / total_duration:.2f}"
    # )
    # print(
    #     f"Seconds per Author-Developer Link: "
    #     f"{total_duration / n_author_developer_links_processed:.2f}"
    # )
    # print(f"Matches Found Per Second: {total_matches_found / total_duration:.2f}")
    # print(f"Seconds per Match Found: {total_duration / total_matches_found:.2f}")
    # print()

    # print("-" * 80)
    # print()


ignorable_doi_spans_default = typer.Option(
    default=["zenodo", "figshare"],
    help=(
        "List of DOI spans to ignore when checking if an article is already in the database."
    ),
)


@app.command()
def snowball_sampling_discovery(
    process_n_author_developer_pairs: int = 10,
    article_repository_allowed_datetime_difference: str = "3 years",
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
                article_repository_allowed_datetime_difference=article_repository_allowed_datetime_difference,
                article_repository_matching_batch_size=article_repository_matching_batch_size,
                ignore_forks=ignore_forks,
                ignorable_doi_spans=ignorable_doi_spans,
                use_prod=use_prod,
                use_coiled=use_coiled,
                coiled_region=coiled_region,
                github_tokens_file=github_tokens_file,
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
