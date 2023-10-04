#!/usr/bin/env python

import logging
from dataclasses import dataclass
from functools import partial

import backoff
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from fastcore.net import HTTP403ForbiddenError
from ghapi.all import GhApi
from parse import search
from tqdm import tqdm

from .registries import RegistryEnum

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

GITHUB_API_SBOM_URL_TEMPLATE = "/repos/{owner}/{repo}/dependency-graph/sbom"

###############################################################################


@dataclass
class RepoParts:
    owner: str
    repo: str


def get_repo_parts_from_url(url: str) -> RepoParts | None:
    """
    Best effort to get the repo parts from a URL.

    Parameters
    ----------
    url: str
        The URL to the repo.

    Returns
    -------
    RepoParts | None
        The repo parts.
        If the parts cannot be parsed, returns None.
    """
    # Handle no http
    if not url.startswith("http"):
        url = f"https://{url}"

    # Add trailing slash
    if not url.endswith("/"):
        url = f"{url}/"

    # Remove www
    url = url.replace("www.", "")

    # Parse the URL
    url_format = "https://github.com/{owner}/{repo}/"
    result = search(url_format, url)

    # Handle result
    if result:
        return RepoParts(owner=result["owner"], repo=result["repo"])

    # Make mypy happy
    return None


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
    url: str,
    github_api_key: str | None = None,
    allowed_registries: list[str] | None = None,
) -> list[DependencyDetails] | None:
    """
    Get the upstream dependency list for a repo.

    Parameters
    ----------
    url: str
        The URL to the repo.
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

    # Get repo parts
    repo_parts = get_repo_parts_from_url(url)

    # Handle failed parse
    if not repo_parts:
        log.debug(f"Failed to parse repo parts from url: '{url}'")
        return None

    # Get SBOM full data
    result = api(
        GITHUB_API_SBOM_URL_TEMPLATE.format(
            owner=repo_parts.owner,
            repo=repo_parts.repo,
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
    repo: str
    upstream_dep_registry: str
    upstream_dep_name: str
    upstream_dep_version: str


@dataclass
class RepoFailure(DataClassJsonMixin):
    repo: str
    exception: str


def _get_upstream_dependencies_for_repo(
    repo: str,
    github_api_key: str | None = None,
    allowed_registries: list[RegistryEnum] | None = None,
) -> list[RepoDependency] | RepoFailure:
    try:
        # Get deps
        repo_deps = get_repo_upstream_dependency_list(
            url=repo,
            github_api_key=github_api_key,
            allowed_registries=allowed_registries,
        )

        # Handle failure
        if not repo_deps:
            return RepoFailure(
                repo=repo,
                exception="Failed to parse repo owner and name.",
            )

    except Exception as e:
        # Handle failure
        return RepoFailure(
            repo=repo,
            exception=str(e),
        )

    # Store deps
    this_repo_deps = []
    for dep in repo_deps:
        this_repo_deps.append(
            RepoDependency(
                repo=repo,
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
) -> tuple[list[RepoDependency], list[RepoFailure]]:
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
    tuple[list[RepoDependency], list[RepoFailure]]
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
    failed: list[RepoFailure] = []
    total_succeeded = 0
    for repo in tqdm(
        github_repos,
        desc="Getting upstream dependencies",
    ):
        # Get repo results
        repo_results = partial_get_upstream_dependencies_for_repo(repo)

        # Handle failure
        if isinstance(repo_results, RepoFailure):
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
