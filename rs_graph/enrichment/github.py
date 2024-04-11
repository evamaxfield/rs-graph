#!/usr/bin/env python

from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import dataclass
from functools import partial

import backoff
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from fastcore.net import HTTP403ForbiddenError
from ghapi.all import GhApi
from sqlmodel import Session

from ..db import models as db_models
from ..db.utils import get_engine, get_or_add
from ..types import (
    ErrorResult,
    ExpandedRepositoryDocumentPair,
    SuccessAndErroredResultsLists,
)
from ..utils.dask_functions import process_func

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@dataclass
class RepoContributorInfo(DataClassJsonMixin):
    repo_doc_pair: ExpandedRepositoryDocumentPair
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
    repo_doc_pair: ExpandedRepositoryDocumentPair,
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
        repo_doc_pair=repo_doc_pair,
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
    repo_doc_pair: ExpandedRepositoryDocumentPair,
    github_api_key: str | None = None,
    top_n: int = 30,
) -> list[RepoContributorInfo]:
    # Setup API
    api = _setup_gh_api(github_api_key)

    # Get contributors
    contributors = api.repos.list_contributors(
        owner=repo_doc_pair.repo_owner,
        repo=repo_doc_pair.repo_name,
        per_page=top_n,
    )

    # Sleep to avoid API limits
    time.sleep(0.85)

    # Get user infos
    _get_user_partial = partial(
        _get_user_info_from_login,
        repo_doc_pair=repo_doc_pair,
        github_api_key=github_api_key,
    )
    contributor_infos = process_func(
        name="github-user-info-from-login",
        func=_get_user_partial,
        func_iterables=[
            [contrib["login"] for contrib in contributors],
        ],
        use_dask=False,  # Force synchronous processing
        display_tqdm=False,
    )

    return contributor_infos


def _wrapped_get_contributors(
    repo_doc_pair: ExpandedRepositoryDocumentPair,
    github_api_key: str | None = None,
    prod: bool = False,
    top_n: int = 30,
) -> ExpandedRepositoryDocumentPair | ErrorResult:
    """Get repo contributors and add them to database."""
    # Get engine
    engine = get_engine(prod=prod)
    with Session(engine) as session:
        try:
            # Get repo contributors
            repo_contributors = get_repo_contributors(
                repo_doc_pair=repo_doc_pair,
                github_api_key=github_api_key,
                top_n=top_n,
            )

            # Create the code host
            code_host = db_models.CodeHost(
                name=repo_doc_pair.repo_host,
            )
            code_host = get_or_add(session=session, model=code_host)

            # Create the repository
            repo = db_models.Repository(
                code_host_id=code_host.id,
                owner=repo_doc_pair.repo_owner,
                name=repo_doc_pair.repo_name,
            )
            repo = get_or_add(session=session, model=repo)

            # For each contributor, create a developer account
            # and then link the dev account to the repo
            # as a repository contributor
            for contributor in repo_contributors:
                # Create the developer account
                dev_account = db_models.DeveloperAccount(
                    code_host_id=code_host.id,
                    username=contributor.username,
                    name=contributor.name,
                    email=contributor.email,
                )
                dev_account = get_or_add(session=session, model=dev_account)

                # Create the repository contributor
                repo_contributor = db_models.RepositoryContributor(
                    repository_id=repo.id,
                    developer_account_id=dev_account.id,
                )
                get_or_add(session=session, model=repo_contributor)

            return repo_doc_pair

        except Exception as e:
            session.rollback()
            return ErrorResult(
                source=repo_doc_pair.source,
                step="github-repo-contributors",
                identifier=(
                    f"{repo_doc_pair.repo_host}/"
                    f"{repo_doc_pair.repo_owner}/"
                    f"{repo_doc_pair.repo_name}"
                ),
                error=str(e),
                traceback=traceback.format_exc(),
            )


def process_repos_for_contributors(
    pairs: list[ExpandedRepositoryDocumentPair],
    prod: bool = False,
    **kwargs: dict,
) -> SuccessAndErroredResultsLists:
    """Process a list of repositories."""
    # Load dotenv
    load_dotenv()
    github_api_key = os.getenv("GITHUB_TOKEN")

    # Check for / init worker client
    get_contribs_partial = partial(
        _wrapped_get_contributors,
        prod=prod,
        github_api_key=github_api_key,
    )
    results = process_func(
        name="github-repo-contributors",
        func=get_contribs_partial,
        func_iterables=[pairs],
        use_dask=False,  # Force synchronous processing
    )

    # Split results
    split_results = SuccessAndErroredResultsLists(
        successful_results=[
            r for r in results if isinstance(r, ExpandedRepositoryDocumentPair)
        ],
        errored_results=[r for r in results if isinstance(r, ErrorResult)],
    )

    # Log stats
    log.info(f"Total succeeded: {len(split_results.successful_results)}")
    log.info(f"Total errored: {len(split_results.errored_results)}")

    return split_results
