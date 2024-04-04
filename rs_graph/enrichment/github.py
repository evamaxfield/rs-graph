#!/usr/bin/env python

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path

import backoff
import pandas as pd
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from fastcore.net import HTTP403ForbiddenError, HTTP404NotFoundError
from ghapi.all import GhApi
from tqdm import tqdm

from ..types import (
    CodeHostResult,
    ErrorResult,
    ExpandedRepositoryDocumentPair,
    RepoParts,
)
from ..utils import code_host_parsing

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

GITHUB_API_SBOM_URL_TEMPLATE = "/repos/{owner}/{repo}/dependency-graph/sbom"

###############################################################################


class RegistryEnum(Enum):
    CARGO = "rust"
    COMPOSER = "composer"
    NUGET = "nuget"
    ACTIONS = "actions"
    GO = "go"
    MAVEN = "maven"
    NPM = "npm"
    PIP = "pip"
    PNPM = "pnpm"
    PUB = "pub"
    POETRY = "poetry"
    RUBYGEMS = "rubygems"
    SWIFT = "swift"
    YARN = "yarn"


@dataclass
class DependencyDetails:
    registry: RegistryEnum
    name: str
    version: str


@backoff.on_exception(
    backoff.expo,
    (HTTP403ForbiddenError),
    max_time=300,
)
def get_repo_upstream_dependency_list(
    repo_parts: RepoParts,
    github_api_key: str | None = None,
    allowed_registries: list[str] | None = None,
) -> list[DependencyDetails] | None:
    """
    Get the upstream dependency list for a repo.

    Parameters
    ----------
    repo_parts: RepoParts
        The repo parts to get the upstream dependencies for.
    github_api_key: str, optional
        The GitHub API key to use for the request.
        If not provided, will use the GITHUB_TOKEN env variable.
    allowed_registries: list[RegistryEnum], optional
        The list of allowed registries to return.
        If not provided, will return all registries.

    Returns
    -------
    list[DependencyDetails] | None
        The list of upstream dependencies for the repo.
        If the repo is not found or partitioning the repo parts fails, returns None.

    Notes
    -----
    This function uses the GitHub API to get the upstream dependency list.
    See https://docs.github.com/en/rest/dependency-graph/sboms for more info.

    Supported package registries: https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/about-the-dependency-graph#supported-package-ecosystems
    """
    # Setup API
    if github_api_key:
        api = GhApi(token=github_api_key)
    else:
        # Load env
        load_dotenv()
        api = GhApi()

    # Set allowed registries
    if not allowed_registries:
        allowed_registries = list(RegistryEnum)

    # Get SBOM full data
    result = api(
        GITHUB_API_SBOM_URL_TEMPLATE.format(
            owner=repo_parts.owner,
            repo=repo_parts.name,
        ),
    )

    # Process deps
    deps = []
    for package in result["sbom"]["packages"][1:]:
        # Unpack package details
        registry_and_name = package["name"].split(":")
        registry = registry_and_name[0]
        name = ":".join(registry_and_name[1:])  # Join back in case name has ":"
        version = package["versionInfo"]

        # Handle not allowed registry
        if RegistryEnum(registry) not in allowed_registries:
            continue

        # Store to deps
        deps.append(
            DependencyDetails(
                registry=RegistryEnum(registry),
                name=name,
                version=version,
            ),
        )

    # Return deps
    return deps


@dataclass
class RepoDependency(DataClassJsonMixin):
    repo: RepoParts
    upstream_dep_registry: str
    upstream_dep_name: str
    upstream_dep_version: str


def _get_upstream_dependencies_for_repo(
    repo_parts: RepoParts,
    github_api_key: str | None = None,
    allowed_registries: list[RegistryEnum] | None = None,
) -> list[RepoDependency] | ErrorResult:
    try:
        # Get deps
        repo_deps = get_repo_upstream_dependency_list(
            repo_parts=repo_parts,
            github_api_key=github_api_key,
            allowed_registries=allowed_registries,
        )

        # Handle failure
        if not repo_deps:
            return ErrorResult(
                source="get_upstream_dependencies_for_repo",
                identifier=f"{repo_parts.host}/{repo_parts.owner}/{repo_parts.name}",
                error="Failed to get upstream dependencies for repo.",
                traceback="",
            )

    except Exception as e:
        # Handle failure
        return ErrorResult(
            source="get_upstream_dependencies_for_repo",
            identifier=f"{repo_parts.host}/{repo_parts.owner}/{repo_parts.name}",
            error=str(e),
            traceback=traceback.format_exc(),
        )

    # Store deps
    this_repo_deps = []
    for dep in repo_deps:
        this_repo_deps.append(
            RepoDependency(
                repo=repo_parts,
                upstream_dep_registry=dep.registry.value,
                upstream_dep_name=dep.name,
                upstream_dep_version=dep.version,
            ),
        )

    return this_repo_deps


def get_upstream_dependencies_for_repos(
    repos: list[str],
    github_api_key: str | None = None,
    allowed_registries: list[RegistryEnum] | None = None,
) -> tuple[list[RepoDependency], list[ErrorResult]]:
    """
    Get the upstream dependencies for a list of repos.

    Parameters
    ----------
    repos: list[str]
        The list of repos to get the upstream dependencies for.
    github_api_key: str, optional
        The GitHub API key to use for the request.
        If not provided, will use the GITHUB_TOKEN env variable.
    allowed_registries: list[RegistryEnum], optional
        The list of allowed registries to return.
        If not provided, will return all registries.

    Returns
    -------
    tuple[list[RepoDependency], list[ErrorResult]]
        The list of upstream dependencies for the repos.
        The list of failed repos.
    """
    # Filter out duplicates
    repos = list(set(repos))

    # Log diff
    log.info(
        f"Filtered out {len(repos) - len(set(repos))} duplicate repos.",
    )

    # Filter out non-github repos
    github_repos = [repo for repo in repos if "github.com" in repo]

    # Log diff
    log.info(
        f"Filtered out {len(repos) - len(github_repos)} non-GitHub repos.",
    )

    # Create partial with args
    partial_get_upstream_dependencies_for_repo = partial(
        _get_upstream_dependencies_for_repo,
        github_api_key=github_api_key,
        allowed_registries=allowed_registries,
    )

    # Get deps
    all_deps: list[RepoDependency] = []
    failed: list[ErrorResult] = []
    total_succeeded = 0
    for repo in tqdm(
        github_repos,
        desc="Getting upstream dependencies",
    ):
        # Get repo results
        repo_results = partial_get_upstream_dependencies_for_repo(repo)

        # Handle failure
        if isinstance(repo_results, ErrorResult):
            failed.append(repo_results)
            continue

        # Store deps
        all_deps.extend(repo_results)

        # Update success
        total_succeeded += 1

    # Log state
    log.info(f"Total succeeded: {total_succeeded}")
    log.info(f"Total found upstream dependencies: {len(all_deps)}")
    log.info(f"Total failed: {len(failed)}")

    # Return deps
    return all_deps, failed


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


def get_repo_contributors_for_repos(
    repo_urls: list[str],
    github_api_key: str | None = None,
    top_n: int = 30,
    cache_file: str | Path = "repo_contributors.parquet",
    cache_every: int = 10,
) -> list[RepoContributorInfo]:
    # Parse every URL
    code_host_results: list[CodeHostResult] = []
    for repo_url in repo_urls:
        # Parse the URL
        try:
            code_host_results.append(code_host_parsing.parse_code_host_url(repo_url))
        except ValueError:
            pass

    # Filter out non-GitHub non-repo items
    repo_parts = [
        RepoParts(
            host=r.host,
            owner=r.owner,
            name=r.name,
        )
        for r in code_host_results
        if r.host == "github" and r.owner is not None and r.name is not None
    ]

    # Filter out duplicates
    repo_parts = list(set(repo_parts))

    # Get contributors
    all_contributors = []
    for i, repo in tqdm(
        enumerate(repo_parts),
        desc="Getting contributors",
        total=len(repo_parts),
    ):
        # Get contributors
        repo_contributors = get_repo_contributors(
            repo_url=repo,
            github_api_key=github_api_key,
            top_n=top_n,
        )

        # Handle failure
        if not repo_contributors:
            continue

        # Store contributors
        all_contributors.extend(repo_contributors)

        # Store to cache
        if i % cache_every == 0:
            log.debug(f"Storing to cache: '{cache_file}'")
            # Convert to dataframe and store to parquet
            pd.DataFrame(
                [d.to_dict() for d in all_contributors],
            ).to_parquet(cache_file)

    return all_contributors


def process_pairs(
    pairs: list[ExpandedRepositoryDocumentPair],
    prod: bool = False,
    use_dask: bool = False,
) -> None:
    pass
