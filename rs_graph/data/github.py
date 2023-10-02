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
    PYPI = "pip"
    NPMJS = "npm"
    ACTIONS = "actions"
    RUBYGEMS = "rubygems"
    MAVEN = "maven"
    GO = "go"
    CARGO = "rust"
    NUGET = "nuget"


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
