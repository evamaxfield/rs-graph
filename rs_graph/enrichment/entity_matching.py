#!/usr/bin/env python

from __future__ import annotations

import time
import traceback

from sci_soft_models import binary_article_repo_em, dev_author_em

from .. import types
from ..db import models as db_models

###############################################################################


def match_devs_and_researchers(
    pair: types.StoredRepositoryDocumentPair | types.ErrorResult,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    try:
        # Get start time
        start_time = time.perf_counter()

        # Get the model details
        model_details = dev_author_em.get_model_details()

        # Convert db types to ml ready types and create LUTs
        dev_username_to_dev_details = {
            dev_account_model.username: (
                dev_author_em.DeveloperDetails(
                    username=dev_account_model.username,
                    name=dev_account_model.name,
                    email=dev_account_model.email,
                )
            )
            for dev_account_model in pair.developer_account_models
        }
        dev_username_to_dev_model = {
            dev_account_model.username: dev_account_model
            for dev_account_model in pair.developer_account_models
        }
        researcher_name_to_researcher_model = {
            researcher.name: researcher for researcher in pair.researcher_models
        }

        # Predict matches
        matches = dev_author_em.match_devs_and_authors(
            devs=list(dev_username_to_dev_details.values()),
            authors=list(researcher_name_to_researcher_model.keys()),
        )

        # Create the linked pairs
        linked_dev_researcher_pairs: list[db_models.ResearcherDeveloperAccountLink] = []
        for matched_dev_author in matches:
            # Find the matching db models
            dev_username = matched_dev_author.dev.username
            researcher_name = matched_dev_author.author

            # Create formal link
            linked_researcher_dev_account = db_models.ResearcherDeveloperAccountLink(
                researcher_id=researcher_name_to_researcher_model[researcher_name].id,
                developer_account_id=dev_username_to_dev_model[dev_username].id,
                predictive_model_name=model_details.name,
                predictive_model_version=model_details.version,
                predictive_model_confidence=matched_dev_author.confidence,
            )
            linked_dev_researcher_pairs.append(linked_researcher_dev_account)

        # Attach
        pair.researcher_developer_links = linked_dev_researcher_pairs

        # Get end time
        end_time = time.perf_counter()

        # Attach processing time
        pair.author_developer_matching_time_seconds = end_time - start_time

        return pair

    except Exception as e:
        return types.ErrorResult(
            source=pair.dataset_source_model.name,
            step="developer-researcher-linking",
            identifier=pair.document_model.doi,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def _prep_for_article_repository_matching(
    unchecked_possible_pair: types.UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair,
) -> types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching | types.ErrorResult:
    try:
        # Create article details
        article_details = binary_article_repo_em.ArticleDetails(
            title=unchecked_possible_pair.author_article.open_alex_results_models.document_model.title,
            doi=unchecked_possible_pair.author_article.open_alex_results_models.document_model.doi,
            publication_date=unchecked_possible_pair.author_article.open_alex_results_models.document_model.publication_date,
            abstract_content=unchecked_possible_pair.author_article.open_alex_results_models.document_abstract_model.content,
        )

        # Create repository details
        repository_details = binary_article_repo_em.RepositoryDetails(
            owner=unchecked_possible_pair.developer_repository.github_result_models.repository_model.owner,
            name=unchecked_possible_pair.developer_repository.github_result_models.repository_model.name,
            description=unchecked_possible_pair.developer_repository.github_result_models.repository_model.description,
            primary_language=unchecked_possible_pair.developer_repository.github_result_models.repository_model.primary_language,
            is_fork=unchecked_possible_pair.developer_repository.github_result_models.repository_model.is_fork,
            creation_datetime=unchecked_possible_pair.developer_repository.github_result_models.repository_model.creation_datetime,
            last_pushed_datetime=unchecked_possible_pair.developer_repository.github_result_models.repository_model.last_pushed_datetime,
            stargazers_count=unchecked_possible_pair.developer_repository.github_result_models.repository_model.stargazers_count,
            forks_count=unchecked_possible_pair.developer_repository.github_result_models.repository_model.forks_count,
            open_issues_count=unchecked_possible_pair.developer_repository.github_result_models.repository_model.open_issues_count,
            commits_count=unchecked_possible_pair.developer_repository.github_result_models.repository_model.commits_count,
            size_kb=unchecked_possible_pair.developer_repository.github_result_models.repository_model.size_kb,
            topics=(
                unchecked_possible_pair.developer_repository.github_result_models.repository_model.topics.split(
                    ";"
                )
                if unchecked_possible_pair.developer_repository.github_result_models.repository_model.topics  # noqa: E501
                else []
            ),
            readme_content=(
                unchecked_possible_pair.developer_repository.github_result_models.repository_readme_model.content
                if unchecked_possible_pair.developer_repository.github_result_models.repository_readme_model  # noqa: E501
                else None
            ),
        )

        return types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching(
            author_developer_link_id=unchecked_possible_pair.author_developer_link_id,
            article_doi=unchecked_possible_pair.article_doi,
            repository_identifier=unchecked_possible_pair.repository_identifier,
            author_article=unchecked_possible_pair.author_article,
            developer_repository=unchecked_possible_pair.developer_repository,
            inference_ready_pair=binary_article_repo_em.InferenceReadyArticleRepositoryPair(
                article_details=article_details,
                repository_details=repository_details,
            ),
        )

    except Exception as e:
        return types.ErrorResult(
            source="author-developer-article-repo-discovery",
            step="prep-article-repo-matching",
            identifier=(
                f"{unchecked_possible_pair.author_developer_link_id}-"
                f"{unchecked_possible_pair.article_doi}-"
                f"{unchecked_possible_pair.repository_identifier}"
            ),
            error=str(e),
            traceback=traceback.format_exc(),
        )


def match_articles_and_repositories(
    inference_ready_article_repository_pairs: list[
        types.AuthorArticleAndDeveloperRepositoryPairPreppedForMatching
    ],
) -> list[types.MatchedAuthorArticleAndDeveloperRepositoryPair] | types.ErrorResult:
    try:
        # Create LUT to match after prediction
        inputs_lut = {
            (pair.article_doi, pair.repository_identifier): pair
            for pair in inference_ready_article_repository_pairs
        }

        # Predict match
        results = binary_article_repo_em.match_articles_and_repositories(
            inference_ready_article_repository_pairs=[
                pair.inference_ready_pair for pair in inference_ready_article_repository_pairs
            ],
            model_choice="optimized",
        )

        # Results LUT
        results_lut = {
            (
                result.article_details.doi,
                f"{result.repository_details.owner}/{result.repository_details.name}",
            ): result
            for result in results
        }

        # Convert to typed results
        return [
            types.MatchedAuthorArticleAndDeveloperRepositoryPair(
                author_developer_link_id=inputs_lut[key].author_developer_link_id,
                article_doi=inputs_lut[key].article_doi,
                repository_identifier=inputs_lut[key].repository_identifier,
                author_article=inputs_lut[key].author_article,
                developer_repository=inputs_lut[key].developer_repository,
                matched_details=results_lut[key],
            )
            for key in results_lut.keys()
        ]

    except Exception as e:
        return types.ErrorResult(
            source="snowball-sampling-discovery",
            step="article-repository-matching",
            identifier="",
            error=str(e),
            traceback=traceback.format_exc(),
        )
