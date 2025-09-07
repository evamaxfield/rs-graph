#!/usr/bin/env python

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from sci_soft_models import dev_author_em
import sci_soft_models.binary_article_repo_em.main as binary_article_repo_em

from .. import types
from ..db import models as db_models

if TYPE_CHECKING:
    import pyalex

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
class InferredArticleRepositoryMatch:
    source_researcher_open_alex_id: str
    source_developer_account_username: str
    article_doi: str
    repo_owner: str
    repo_name: str
    model_details: binary_article_repo_em.ModelDetails
    predicted_label: bool
    prediction_confidence: float

# TODO: Change the input type to already be a prepped article and repository details
# with accompanying tracking metadata
# We want to process multiple at a time so we can batch the model load/inference calls

def match_article_and_repository(
    source_researcher_open_alex_id: str,
    source_developer_account_username: str,
    article_doi: str,
    repo_owner: str,
    repo_name: str,
    article_details: binary_article_repo_em.ArticleDetails,
    repository_details: binary_article_repo_em.RepositoryDetails,
) -> InferredArticleRepositoryMatch | types.ErrorResult:
    try:
        # Get the model details
        model_details = binary_article_repo_em.get_model_details()

        # Predict match
        matches = binary_article_repo_em.match_articles_and_repositories(
            article=article_details,
            repository=repository_details,
        )

        return matches[0]

    except Exception as e:
        return types.ErrorResult(
            source="snowball-sampling-discovery",
            step="article-repository-matching",
            identifier=f"{article_doi} <-> {repo_owner}/{repo_name}",
            error=str(e),
            traceback=traceback.format_exc(),
        )
    