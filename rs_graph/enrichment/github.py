#!/usr/bin/env python

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import backoff
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from fastcore.net import HTTP403ForbiddenError, HTTP404NotFoundError
from ghapi.all import GhApi
from tqdm import tqdm

from ..types import (
    ErrorResult,
    ExpandedRepositoryDocumentPair,
    RepoParts,
    SuccessAndErroredResultsLists,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@dataclass
class RepoContributorInfo(DataClassJsonMixin):
    repo_parts: RepoParts
    username: str
    name: str | None
    company: str | None
    email: str | None
    location: str | None
    bio: str | None
    co_contributors: tuple[str, ...]


@backoff.on_exception(
    backoff.expo,
    (HTTP403ForbiddenError),
    max_time=60,
)
def _get_user_info_from_login(
    login: str,
    api: GhApi,
    repo_parts: RepoParts,
    co_contributors: tuple[str, ...],
) -> list[RepoContributorInfo] | None:
    # Get contributor info
    try:
        user_info = api.users.get_by_username(username=login)

        # Remove login from co-contributors
        co_contributor_logins = tuple(
            co_contrib for co_contrib in co_contributors if co_contrib != login
        )

        # Sleep before return to avoid rate limit
        time.sleep(2)

        # Store info
        return RepoContributorInfo(
            repo_parts=repo_parts,
            username=login,
            name=user_info["name"],
            email=user_info["email"],
            co_contributors=co_contributor_logins,
        )

    except HTTP404NotFoundError:
        return None


@backoff.on_exception(
    backoff.expo,
    (HTTP403ForbiddenError),
    max_time=60,
)
def get_repo_contributors(
    repo_parts: RepoParts,
    github_api_key: str | None = None,
    top_n: int = 30,
) -> list[RepoContributorInfo] | ErrorResult:
    # Setup API
    if github_api_key:
        api = GhApi(token=github_api_key)
    else:
        # Load env
        load_dotenv()
        api = GhApi()

    # Get contributors
    try:
        contributors = api.repos.list_contributors(
            owner=repo_parts.owner,
            repo=repo_parts.repo,
            per_page=top_n,
        )
    except HTTP404NotFoundError:
        return ErrorResult(
            source="get_repo_contributors",
            identifier=f"{repo_parts.host}/{repo_parts.owner}/{repo_parts.name}",
            error="Repo not found.",
            traceback="",
        )

    # Get user infos
    contributor_infos = []
    for contrib in tqdm(contributors, desc="Getting user info", leave=False):
        # Get user info
        contributor_infos.append(
            _get_user_info_from_login(
                contrib["login"],
                api=api,
                repo_parts=repo_parts,
                co_contributors=tuple(contrib["login"] for contrib in contributors),
            )
        )

    # Filter out none
    contributor_infos = [info for info in contributor_infos if info is not None]

    return contributor_infos


def process_pairs(
    pairs: list[ExpandedRepositoryDocumentPair],
    prod: bool = False,
) -> SuccessAndErroredResultsLists:
    return SuccessAndErroredResultsLists(successes=[], errored=[])
