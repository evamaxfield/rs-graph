#!/usr/bin/env python

import itertools
import os
from dataclasses import dataclass

import coiled
import typer
import yaml
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from gh_tokens_loader import GitHubTokensCycler
from prefect import flow, task, unmapped
from datetime import datetime, timedelta

from rs_graph import types
from rs_graph.db import utils as db_utils
from rs_graph.enrichment import article, entity_matching, github
from rs_graph.utils.dt_and_td import parse_timedelta
from rs_graph.bin.pipeline_utils import (
    _get_small_cpu_api_cluster,
    _get_basic_gpu_cluster_config,
    _wrap_func_with_coiled_prefect_task,
    _load_coiled_software_envs,
    _load_open_alex_emails,
    DEFAULT_ELSEVIER_API_KEYS_FILE,
    DEFAULT_GITHUB_TOKENS_FILE,
    DEFAULT_OPEN_ALEX_EMAILS_FILE,
    DEFAULT_RESULTS_DIR,
)

###############################################################################

app = typer.Typer()

###############################################################################

@task
def _wrapped_get_hydrated_author_dev_links(
    use_prod: bool,
    author_developer_links_filter_datetime_difference: str,
    author_developer_links_filter_confidence_threshold: float,
    process_n_author_developer_pairs: int,
) -> list[db_utils.HydratedAuthorDeveloperLink]:
    return db_utils.get_hydrated_author_developer_links(
        use_prod=use_prod,
        filter_datetime_difference=author_developer_links_filter_datetime_difference,
        filter_confidence_threshold=author_developer_links_filter_confidence_threshold,
        n=process_n_author_developer_pairs,
    )


@dataclass
class AuthorArticleDetails(DataClassJsonMixin):
    author_developer_link_id: int
    researcher_open_alex_id: str
    open_alex_results_models: types.OpenAlexResultModels


def _get_author_articles_for_researcher(
    author_developer_link_id: int,
    researcher_open_alex_id: str,
    open_alex_email: str,
    open_alex_email_count: int,
    semantic_scholar_api_key: str | None,
) -> list[AuthorArticleDetails | types.ErrorResult] | types.ErrorResult: 
    # Get the articles for the researcher
    researcher_articles = article.get_articles_for_researcher(
        researcher_open_alex_id=researcher_open_alex_id,
        open_alex_email=open_alex_email,
        open_alex_email_count=open_alex_email_count,
        semantic_scholar_api_key=semantic_scholar_api_key,
    )

    if isinstance(researcher_articles, types.ErrorResult):
        return researcher_articles
    
    author_article_details_list: list[AuthorArticleDetails | types.ErrorResult] = []
    for open_alex_result in researcher_articles:
        if isinstance(open_alex_result, types.ErrorResult):
            author_article_details_list.append(open_alex_result)
        else:
            author_article_details = AuthorArticleDetails(
                author_developer_link_id=author_developer_link_id,
                researcher_open_alex_id=researcher_open_alex_id,
                open_alex_results_models=open_alex_result,
            )
            author_article_details_list.append(author_article_details)

    return author_article_details_list

@dataclass
class DeveloperRepositoryDetails(DataClassJsonMixin):
    author_developer_link_id: int
    developer_account_username: str
    github_result_models: types.GitHubResultModels


def _get_developer_repositories_for_developer(
    author_developer_link_id: int,
    developer_account_username: str,
    github_api_key: str,
) -> list[DeveloperRepositoryDetails | types.ErrorResult] | types.ErrorResult:
    # Get the repositories for the developer
    developer_repositories = github.get_github_repos_for_developer(
        username=developer_account_username,
        github_api_key=github_api_key,
    )

    if isinstance(developer_repositories, types.ErrorResult):
        return developer_repositories
    
    developer_repository_details_list: list[DeveloperRepositoryDetails | types.ErrorResult] = []
    for github_result in developer_repositories:
        if isinstance(github_result, types.ErrorResult):
            developer_repository_details_list.append(github_result)
        else:
            developer_repository_details = DeveloperRepositoryDetails(
                author_developer_link_id=author_developer_link_id,
                developer_account_username=developer_account_username,
                github_result_models=github_result,
            )
            developer_repository_details_list.append(developer_repository_details)

    return developer_repository_details_list


@dataclass
class FilteredResult(DataClassJsonMixin):
    source: str
    identifier: str
    reason: str


@task
def _flatten_and_check_articles_in_db(
    all_author_articles_and_errors: list[list[AuthorArticleDetails | types.ErrorResult] | types.ErrorResult],
    use_prod: bool,
    ignorable_doi_spans: list[str],
) -> list[AuthorArticleDetails | FilteredResult | types.ErrorResult]:
    possible_count = 0
    to_process_count = 0
    flattened_results: list[AuthorArticleDetails | FilteredResult | types.ErrorResult] = []
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
                        filtered_result = FilteredResult(
                            source=item.researcher_open_alex_id,
                            identifier=item.open_alex_results_models.document_model.title,
                            reason="No DOI available",
                        )
                        flattened_results.append(filtered_result)

                    elif any(
                        span in item.open_alex_results_models.document_model.doi.lower()
                        for span in ignorable_doi_spans
                    ):
                        filtered_result = FilteredResult(
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
                            filtered_result = FilteredResult(
                                source=item.researcher_open_alex_id,
                                identifier=item.open_alex_results_models.document_model.doi,
                                reason="Already in database",
                            )
                            flattened_results.append(filtered_result)
                        else:
                            to_process_count += 1
                            flattened_results.append(item)

    print(f"Filtered out {possible_count - to_process_count} articles already in DB.")

    return flattened_results


@task
def _flatten_and_check_repositories_in_db(
    all_developer_repositories_and_errors: list[list[DeveloperRepositoryDetails | types.ErrorResult] | types.ErrorResult],
    use_prod: bool,
    ignore_forks: bool,
) -> list[DeveloperRepositoryDetails | FilteredResult | types.ErrorResult]:
    possible_count = 0
    to_process_count = 0
    flattened_results: list[DeveloperRepositoryDetails | FilteredResult | types.ErrorResult] = []
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
                        filtered_result = FilteredResult(
                            source=item.developer_account_username,
                            identifier=(
                                f"{item.github_result_models.repository_model.owner}/"
                                f"{item.github_result_models.repository_model.name}"
                            ),
                            reason="Already in database",
                        )
                        flattened_results.append(filtered_result)

                    elif ignore_forks and item.github_result_models.repository_model.is_fork:
                        filtered_result = FilteredResult(
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

    print(f"Filtered out {possible_count - to_process_count} repositories already in DB.")

    return flattened_results


@dataclass
class UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair(DataClassJsonMixin):
    author_developer_link_id: int
    article_doi: str
    repository_identifier: str
    author_article: AuthorArticleDetails
    developer_repository: DeveloperRepositoryDetails


@task
def _combine_to_possible_pairs(
    author_articles: list[AuthorArticleDetails | FilteredResult | types.ErrorResult],
    developer_repositories: list[DeveloperRepositoryDetails | FilteredResult | types.ErrorResult],
    max_datetime_difference: timedelta,
) -> list[UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair]:
    # Create a lookup of author_developer_link_id to list of author_articles
    author_articles_lut: dict[int, list[AuthorArticleDetails]] = {}
    for item in author_articles:
        if isinstance(item, AuthorArticleDetails):
            if item.author_developer_link_id not in author_articles_lut:
                author_articles_lut[item.author_developer_link_id] = []
            author_articles_lut[item.author_developer_link_id].append(item)

    # Create a lookup of author_developer_link_id to list of developer_repositories
    developer_repositories_lut: dict[int, list[DeveloperRepositoryDetails]] = {}
    for item in developer_repositories:
        if isinstance(item, DeveloperRepositoryDetails):
            if item.author_developer_link_id not in developer_repositories_lut:
                developer_repositories_lut[item.author_developer_link_id] = []
            developer_repositories_lut[item.author_developer_link_id].append(item)

    # Find common author_developer_link_ids
    common_author_developer_link_ids = set(author_articles_lut.keys()).intersection(
        set(developer_repositories_lut.keys())
    )

    # Combine author articles and developer repositories for common author_developer_link_ids
    unchecked_possible_combinations: list[UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair] = []
    for author_developer_link_id in common_author_developer_link_ids:
        for author_article in author_articles_lut[author_developer_link_id]:
            for developer_repository in developer_repositories_lut[author_developer_link_id]:
                # Convert to datetimes
                article_published_date = author_article.open_alex_results_models.document_model.publication_date
                article_published_dt = datetime(
                    year=article_published_date.year,
                    month=article_published_date.month,
                    day=article_published_date.day,
                )
                repo_created_dt = developer_repository.github_result_models.repository_model.creation_datetime

                # Remove timezone info for comparison
                if article_published_dt.tzinfo is not None:
                    article_published_dt = article_published_dt.replace(tzinfo=None)
                if repo_created_dt.tzinfo is not None:
                    repo_created_dt = repo_created_dt.replace(tzinfo=None)

                if abs(article_published_dt - repo_created_dt) <= max_datetime_difference:
                    unchecked_possible_combinations.append(
                        UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair(
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


@task
def _get_unique_repositories(
    unchecked_possible_combinations: list[UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair],
) -> list[DeveloperRepositoryDetails]:
    unique_repos_set = set()
    unique_developer_repositories: list[DeveloperRepositoryDetails] = []
    for item in unchecked_possible_combinations:
        repo_identifier = item.repository_identifier
        if repo_identifier not in unique_repos_set:
            unique_repos_set.add(repo_identifier)
            unique_developer_repositories.append(item.developer_repository)

    return unique_developer_repositories



def _enrich_repository_with_data_required_for_matching(
    developer_repository: DeveloperRepositoryDetails,
    github_api_key: str,
) -> DeveloperRepositoryDetails | types.ErrorResult:
    # Enrich the repository
    return github.process_github_repo(
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


@task
def _replace_enriched_repositories_in_possible_combinations(
    unchecked_possible_combinations: list[UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair],
    enriched_repositories: list[DeveloperRepositoryDetails | types.ErrorResult],
) -> list[UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair | types.ErrorResult]:
    # Create a lookup of repository_identifier to enriched repository
    enriched_repos_lut: dict[str, DeveloperRepositoryDetails] = {}
    for item in enriched_repositories:
        if isinstance(item, DeveloperRepositoryDetails):
            repo_identifier = (
                f"{item.github_result_models.repository_model.owner}/"
                f"{item.github_result_models.repository_model.name}"
            )
            enriched_repos_lut[repo_identifier] = item

    # Replace developer repositories in unchecked_possible_combinations with enriched ones
    updated_combinations: list[UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair] = []
    for item in unchecked_possible_combinations:
        repo_identifier = item.repository_identifier
        if repo_identifier in enriched_repos_lut:
            enriched_repo = enriched_repos_lut[repo_identifier]
            updated_combinations.append(
                UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair(
                    author_developer_link_id=item.author_developer_link_id,
                    article_doi=item.article_doi,
                    repository_identifier=item.repository_identifier,
                    author_article=item.author_article,
                    developer_repository=enriched_repo,
                )
            )
        else:
            updated_combinations.append(
                types.ErrorResult(
                    source="snowball-sampling-discovery",
                    step="snowball-sampling-replace-enriched-repositories",
                    identifier=repo_identifier,
                    error="Repository enrichment failed; enriched repository not found during merge.",
                    traceback="",
                )
            )

    print(f"{len(updated_combinations)} combinations to match.")

    print(updated_combinations[0].article_doi)
    print(updated_combinations[0].repository_identifier)
    print(updated_combinations[0].developer_repository.github_result_models.repository_model.commits_count)
    print(updated_combinations[0].developer_repository.github_result_models.repository_readme_model.content)

    return updated_combinations


@flow(
    log_prints=True,
)
def _snowball_sampling_discovery_flow(  # noqa: C901
    process_n_author_developer_pairs: int,
    article_repository_allowed_datetime_difference: str,
    author_developer_links_filter_confidence_threshold: float,
    author_developer_links_filter_datetime_difference: str,
    article_repository_matching_filter_confidence_threshold: float,
    batch_size: int,
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

    # cluster_type_vm_types_lut = {
    #     "CPU": "t4g.small",
    #     "GPU": "g4dn.xlarge",
    # }

    # if coiled_software_envs is None:
    #     cluster_type_software_env_lut: dict[str, str | None] = {}
    #     for cluster_type, vm_type in cluster_type_vm_types_lut.items():
    #         if use_coiled:
    #             import coiled

    #             print(f"Creating {cluster_type} coiled cluster to push software environment...")
    #             cluster = coiled.Cluster(
    #                 scheduler_vm_types=[vm_type],
    #                 worker_vm_types=[vm_type],
    #                 n_workers=0,
    #                 region=coiled_region,
    #             )
    #             print("Coiled cluster created.")
    #             time.sleep(5)
    #             cluster_type_software_env_lut[cluster_type] = cluster._software_environment_name
    #             cluster.close()
    #         else:
    #             cluster_type_software_env_lut[cluster_type] = None
    # else:
    #     cluster_type_software_env_lut = coiled_software_envs

    # # Cache coiled software envs if specified
    # if cache_coiled_software_envs and use_coiled:
    #     with open(coiled_software_envs_file, "w") as f:
    #         yaml.dump(cluster_type_software_env_lut, f)

    # Preconstruct all wrapped tasks
    wrapped_get_articles_for_researcher = _wrap_func_with_coiled_prefect_task(
        _get_author_articles_for_researcher,
        coiled_func_name="open_alex_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_open_alex_emails,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
            # software_env_name=cluster_type_software_env_lut["CPU"],
        ),
    )
    wrapped_get_repositories_for_developer = _wrap_func_with_coiled_prefect_task(
        _get_developer_repositories_for_developer,
        coiled_func_name="github_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_github_tokens,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
            # software_env_name=cluster_type_software_env_lut["CPU"],
        ),
    )
    wrapped_enrich_repository = _wrap_func_with_coiled_prefect_task(
        _enrich_repository_with_data_required_for_matching,
        coiled_func_name="github_cluster",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=n_github_tokens,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
            # software_env_name=cluster_type_software_env_lut["CPU"],
        ),
    )

    # Get author-developer-account links from the database
    print("Getting author-developer-account links from the database...")
    hydrated_author_developer_links = _wrapped_get_hydrated_author_dev_links(
        use_prod=use_prod,
        author_developer_links_filter_datetime_difference=author_developer_links_filter_datetime_difference,
        author_developer_links_filter_confidence_threshold=author_developer_links_filter_confidence_threshold,
        process_n_author_developer_pairs=process_n_author_developer_pairs,
    )

    # Get each authors articles
    print("Getting each author's articles from Open Alex...")
    author_articles = wrapped_get_articles_for_researcher.map(
        author_developer_link_id=[
            link.author_developer_link_id for link in hydrated_author_developer_links
        ],
        researcher_open_alex_id=[
            link.researcher_open_alex_id for link in hydrated_author_developer_links
        ],
        open_alex_email=[
            next(cycled_open_alex_emails) for _ in range(len(hydrated_author_developer_links))
        ],
        open_alex_email_count=unmapped(n_open_alex_emails),
        semantic_scholar_api_key=unmapped(semantic_scholar_api_key),
    )

    # Drop existing articles already stored in the database
    print("Filtering out articles already in the database...")
    flattened_author_articles = _flatten_and_check_articles_in_db(
        all_author_articles_and_errors=author_articles,
        use_prod=use_prod,
        ignorable_doi_spans=ignorable_doi_spans,
    )

    # Get each developer's repositories
    print("Getting each developer's repositories from GitHub...")
    developer_repositories = wrapped_get_repositories_for_developer.map(
        author_developer_link_id=[
            link.author_developer_link_id for link in hydrated_author_developer_links
        ],
        developer_account_username=[
            link.developer_account_username for link in hydrated_author_developer_links
        ],
        github_api_key=[next(cycled_github_tokens) for _ in range(len(hydrated_author_developer_links))],
    )

    # Drop existing repositories already stored in the database
    print("Filtering out repositories already in the database...")
    developer_repositories_filtered = _flatten_and_check_repositories_in_db(
        all_developer_repositories_and_errors=developer_repositories,
        use_prod=use_prod,
        ignore_forks=ignore_forks,
    )

    # Combine back together to possible article-repository pairs
    print("Combining to unchecked but possible article-repository pairs...")
    unchecked_possible_combinations = _combine_to_possible_pairs(
        author_articles=flattened_author_articles,
        developer_repositories=developer_repositories_filtered,
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
        github_api_key=[next(cycled_github_tokens) for _ in range(len(unique_developer_repositories))],
    )

    # Replace enriched repositories back into possible combinations
    print("Replacing enriched repositories back into possible combinations...")
    possible_combinations_with_enriched_repos = _replace_enriched_repositories_in_possible_combinations(
        unchecked_possible_combinations=unchecked_possible_combinations,
        enriched_repositories=enriched_repositories,
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


@app.command()
def snowball_sampling_discovery(
    process_n_author_developer_pairs: int = 10,
    article_repository_allowed_datetime_difference: str = "3 years",
    author_developer_links_filter_confidence_threshold: float = 0.97,
    author_developer_links_filter_datetime_difference: str = "2 years",
    article_repository_matching_filter_confidence_threshold: float = 0.97,
    batch_size: int = 16,
    ignore_forks: bool = True,
    ignorable_doi_spans: list[str] = typer.Option(
        default=["zenodo", "figshare"],
        help="List of DOI spans to ignore when checking if an article is already in the database.",
    ),
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
    _load_open_alex_emails(open_alex_emails_file)

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

    # Start the flow
    _snowball_sampling_discovery_flow(
        process_n_author_developer_pairs=process_n_author_developer_pairs,
        article_repository_allowed_datetime_difference=article_repository_allowed_datetime_difference,
        author_developer_links_filter_confidence_threshold=author_developer_links_filter_confidence_threshold,
        author_developer_links_filter_datetime_difference=author_developer_links_filter_datetime_difference,
        article_repository_matching_filter_confidence_threshold=article_repository_matching_filter_confidence_threshold,
        batch_size=batch_size,
        ignore_forks=ignore_forks,
        ignorable_doi_spans=ignorable_doi_spans,
        use_prod=use_prod,
        use_coiled=use_coiled,
        coiled_region=coiled_region,
        github_tokens_file=github_tokens_file,
        open_alex_emails=_load_open_alex_emails(open_alex_emails_file),
        semantic_scholar_api_key=semantic_scholar_api_key,
        coiled_software_envs=coiled_software_envs,
        cache_coiled_software_envs=cache_coiled_software_envs,
        coiled_software_envs_file=coiled_software_envs_file,
    )


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
