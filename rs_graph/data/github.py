#!/usr/bin/env python

import logging
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from ghapi.all import GhApi
from parse import search

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


# TODO: add backoff for rate limit
def get_repo_upstream_dependency_list(
    url: str,
    github_api_key: str | None = None,
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


# TODO:
# handle value errors for unknown registry
# handle non-github-urls
# handle HTTP404NotFoundError
