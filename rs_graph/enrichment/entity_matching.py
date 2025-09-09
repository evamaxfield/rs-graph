#!/usr/bin/env python

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import sci_soft_models.binary_article_repo_em as binary_article_repo_em
from dataclasses_json import DataClassJsonMixin
from sci_soft_models import dev_author_em

from .. import types
from ..db import models as db_models
from . import article

if TYPE_CHECKING:
    import pyalex

    from . import github

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


@dataclass
class SimplePossibleArticleRepositoryPair:
    """A possible article-repository pair for matching."""

    work: pyalex.Work
    repository: dict


def get_possible_article_repository_pairs_for_matching(
    works: list[pyalex.Work],
    repos: list[dict],
    max_datetime_difference: timedelta,
) -> list[SimplePossibleArticleRepositoryPair]:
    """Get possible article-repository pairs for matching."""
    # Create possible article-repository pairs by filtering pairs so that each pair
    # has an article publication date and the repository creation date
    # are within the allowed datetime difference
    pairs = []
    for work in works:
        if work["doi"] is not None:
            # Skip if "zenodo" or "figshare" in DOI
            if any(x in work["doi"].lower() for x in ["zenodo", "figshare"]):
                continue

            for repo in repos:
                try:
                    # Convert to datetimes
                    work_published_date = datetime.fromisoformat(work["publication_date"])
                    repo_created_at = datetime.fromisoformat(repo["created_at"])

                    # Remove timezone info for comparison
                    if work_published_date.tzinfo is not None:
                        work_published_date = work_published_date.replace(tzinfo=None)
                    if repo_created_at.tzinfo is not None:
                        repo_created_at = repo_created_at.replace(tzinfo=None)

                    if abs(work_published_date - repo_created_at) <= max_datetime_difference:
                        pairs.append(
                            SimplePossibleArticleRepositoryPair(work=work, repository=repo)
                        )

                except Exception as e:
                    print("error", e)
                    print(work)
                    print()
                    print(repo)

    return pairs


@dataclass
class PossibleArticleRepositoryPair(DataClassJsonMixin):
    source_researcher_open_alex_id: str
    source_developer_account_username: str
    article_title: str
    article_doi: str
    open_alex_work: pyalex.Work
    repo_owner: str
    repo_name: str
    github_repository: dict


def _prep_for_article_repository_matching(
    possible_pair: PossibleArticleRepositoryPair,
    repository_readme_and_contributors: github.RepoReadmeAndContributorInfo,
) -> binary_article_repo_em.InferenceReadyArticleRepositoryPair | types.ErrorResult:
    try:
        # Create article details
        article_details = binary_article_repo_em.ArticleDetails(
            title=possible_pair.article_title.strip(),
            doi=possible_pair.article_doi.lower().strip(),
            publication_date=possible_pair.open_alex_work["publication_date"],
            topic_unique_domains=list(
                {
                    topic["domain"]["display_name"].strip()
                    for topic in possible_pair.open_alex_work["topics"]
                }
            ),
            topic_unique_fields=list(
                {
                    topic["field"]["display_name"].strip()
                    for topic in possible_pair.open_alex_work["topics"]
                }
            ),
            document_type=possible_pair.open_alex_work["type"].strip(),
            authors_list=[
                authorship["author"]["display_name"].strip()
                for authorship in possible_pair.open_alex_work["authorships"]
            ],
            abstract_content=article.convert_from_inverted_index_abstract(
                abstract=possible_pair.open_alex_work["abstract_inverted_index"]
            ),
        )

        # Create repository details
        repository_details = binary_article_repo_em.RepositoryDetails(
            owner=possible_pair.repo_owner.strip(),
            name=possible_pair.repo_name.strip(),
            description=possible_pair.github_repository["description"],
            primary_language=possible_pair.github_repository["language"],
            is_fork=possible_pair.github_repository["fork"],
            creation_datetime=possible_pair.github_repository["created_at"],
            last_pushed_datetime=possible_pair.github_repository["pushed_at"],
            stargazers_count=possible_pair.github_repository["stargazers_count"],
            forks_count=possible_pair.github_repository["forks_count"],
            open_issues_count=possible_pair.github_repository["open_issues_count"],
            commits_count=None,
            size_kb=possible_pair.github_repository["size"],
            topics=list(possible_pair.github_repository["topics"]),
            contributors_list=[
                binary_article_repo_em.RepositoryContributorDetails(
                    username=contributor_info.username,
                    name=contributor_info.name,
                    email=contributor_info.email,
                )
                for contributor_info in repository_readme_and_contributors.contributor_infos
            ],
            tree=None,
            readme_content=(
                repository_readme_and_contributors.readme.strip()
                if repository_readme_and_contributors.readme
                else None
            ),
        )

        return binary_article_repo_em.InferenceReadyArticleRepositoryPair(
            article_details=article_details,
            repository_details=repository_details,
        )

    except Exception as e:
        return types.ErrorResult(
            source="author-developer-article-repo-discovery",
            step="prep-article-repo-matching",
            identifier=(
                f"{possible_pair.source_researcher_open_alex_id}-"
                f"{possible_pair.source_developer_account_username}-"
                f"{possible_pair.article_doi}-"
                f"{possible_pair.repo_owner}/{possible_pair.repo_name}"
            ),
            error=str(e),
            traceback=traceback.format_exc(),
        )


def match_articles_and_repositories(
    inference_ready_article_repository_pairs: list[
        binary_article_repo_em.InferenceReadyArticleRepositoryPair
    ],
) -> list[binary_article_repo_em.MatchedArticleRepository] | types.ErrorResult:
    try:
        # Predict match
        return binary_article_repo_em.match_articles_and_repositories(
            inference_ready_article_repository_pairs=inference_ready_article_repository_pairs,
            model_choice="optimized",
        )

    except Exception as e:
        return types.ErrorResult(
            source="snowball-sampling-discovery",
            step="article-repository-matching",
            identifier="",
            error=str(e),
            traceback=traceback.format_exc(),
        )
