#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import random
import time
import traceback
from dataclasses import dataclass
from functools import partial

import backoff
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from fastcore.net import HTTP403ForbiddenError
from ghapi.all import GhApi
from prefect import task

from .. import types
from ..db import models as db_models

###############################################################################

GITHUB_TOKENS_FILE = ".github-tokens.json"

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@dataclass
class RepoContributorInfo(DataClassJsonMixin):
    pair: types.ExpandedRepositoryDocumentPair
    username: str
    name: str | None
    email: str | None


def _setup_gh_api(github_api_key: str | None = None) -> GhApi:
    """Create a GitHub API object."""
    # Setup API
    if github_api_key:
        api = GhApi(token=github_api_key)
    else:
        # Load env
        load_dotenv()
        api = GhApi()

    return api


@backoff.on_exception(
    backoff.expo,
    (HTTP403ForbiddenError),
    max_time=16,
)
def _get_user_info_from_login(
    login: str,
    pair: types.ExpandedRepositoryDocumentPair,
    github_api_key: str | None = None,
) -> list[RepoContributorInfo]:
    # Setup API
    api = _setup_gh_api(github_api_key)

    # Get user info
    user_info = api.users.get_by_username(username=login)

    # Sleep to avoid API limits
    time.sleep(0.85)

    # Store info
    return RepoContributorInfo(
        pair=pair,
        username=login,
        name=user_info["name"],
        email=user_info["email"],
    )


@backoff.on_exception(
    backoff.expo,
    (HTTP403ForbiddenError),
    max_time=16,
)
def get_repo_contributors(
    pair: types.ExpandedRepositoryDocumentPair,
    github_api_key: str | None = None,
    top_n: int = 30,
) -> list[RepoContributorInfo]:
    # Setup API
    api = _setup_gh_api(github_api_key)

    # Assert RepoParts is not None
    assert pair.repo_parts is not None

    # Get contributors
    contributors = api.repos.list_contributors(
        owner=pair.repo_parts.owner,
        repo=pair.repo_parts.name,
        per_page=top_n,
    )

    # Sleep to avoid API limits
    time.sleep(0.85)

    # Get user infos
    _get_user_partial = partial(
        _get_user_info_from_login,
        pair=pair,
        github_api_key=github_api_key,
    )

    return [_get_user_partial(login=contrib["login"]) for contrib in contributors]


def process_github_repo(
    pair: types.ExpandedRepositoryDocumentPair,
    github_api_key: str | None = None,
    top_n: int = 30,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    """Get repo contributors and add them to database."""
    # Assert repo parts is not None
    assert pair.repo_parts is not None

    try:
        # Setup API
        api = _setup_gh_api(github_api_key)

        # Get repo info
        repo_info = api.repos.get(
            owner=pair.repo_parts.owner,
            repo=pair.repo_parts.name,
        )

        # Sleep to avoid API limits
        time.sleep(0.85)

        # Get repo contributors
        repo_contributors = get_repo_contributors(
            pair=pair,
            github_api_key=github_api_key,
            top_n=top_n,
        )

        # Create the code host
        code_host = db_models.CodeHost(
            name=pair.repo_parts.host,
        )

        # Create the repository
        repo = db_models.Repository(
            code_host_id=code_host.id,
            owner=pair.repo_parts.owner,
            name=pair.repo_parts.name,
            description=repo_info["description"],
            is_fork=repo_info["fork"],
            forks_count=repo_info["forks_count"],
            stargazers_count=repo_info["stargazers_count"],
            watchers_count=repo_info["watchers_count"],
            open_issues_count=repo_info["open_issues_count"],
            size_kb=repo_info["size"],
            creation_datetime=repo_info["created_at"],
            last_push_datetime=repo_info["pushed_at"],
        )

        # For each contributor, create a developer account
        # and then link the dev account to the repo
        # as a repository contributor
        all_repo_contributors = []
        for contributor in repo_contributors:
            # Create the developer account
            dev_account = db_models.DeveloperAccount(
                code_host_id=code_host.id,
                username=contributor.username,
                name=contributor.name,
                email=contributor.email,
            )

            # Create the repository contributor
            repo_contributor = db_models.RepositoryContributor(
                repository_id=repo.id,
                developer_account_id=dev_account.id,
            )

            # Store the pair
            all_repo_contributors.append(
                types.RepositoryContributorDetails(
                    developer_account_model=dev_account,
                    repository_contributor_model=repo_contributor,
                )
            )

        # Attach the results to the pair
        pair.github_results = types.GitHubResultModels(
            code_host_model=code_host,
            repository_model=repo,
            repository_contributor_details=all_repo_contributors,
        )

        return pair

    except Exception as e:
        return types.ErrorResult(
            source=pair.source,
            step="github-repo-contributors",
            identifier=(
                f"{pair.repo_parts.host}/"
                f"{pair.repo_parts.owner}/"
                f"{pair.repo_parts.name}"
            ),
            error=str(e),
            traceback=traceback.format_exc(),
        )


@task(timeout_seconds=180)
def process_github_repo_task(
    pair: types.ExpandedRepositoryDocumentPair | types.ErrorResult,
    top_n: int = 30,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    # Load tokens
    try:
        with open(GITHUB_TOKENS_FILE) as f:
            tokens = json.load(f)

    except FileNotFoundError as e:
        return types.ErrorResult(
            source="github",
            step="github-repo-contributors",
            identifier="",  # No identifier, as this is a general error
            error="No .github-tokens.json file found",
            traceback=str(e),
        )

    return process_github_repo(
        pair=pair,
        github_api_key=random.choice(tokens),
        top_n=top_n,
    )
