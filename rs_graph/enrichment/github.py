#!/usr/bin/env python

from __future__ import annotations

import base64
import socket
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import backoff
import requests
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from fastcore.net import HTTP403ForbiddenError  # type: ignore[import-not-found,attr-defined]
from ghapi.all import GhApi, paged

from .. import types
from ..db import models as db_models

###############################################################################

# ghapi issues requests through urllib (via fastcore), which uses no socket
# timeout by default -- a stalled GitHub connection therefore blocks forever and
# can hang an entire enrichment task (and the flow waiting on its result). urllib
# falls back to socket.getdefaulttimeout() when no explicit timeout is given, so
# we set a generous process-wide default here. It is a per-read idle timeout, so
# slow-but-streaming responses (e.g. large git trees) are unaffected; only true
# stalls (no bytes for this long) are cut off and surfaced as a timeout error.
GITHUB_SOCKET_TIMEOUT_SECONDS = 60

# Throttle between GitHub API calls within a single task. These sleeps guard
# against GitHub's *secondary* rate limits (<=100 concurrent, <=900 points/min
# where a GET costs 1 point). With tokens cycled across the task map, a single
# token sees only a few requests/second even at this small value -- ~10x under
# the 900/min ceiling -- so this is intentionally low. The *binding* limit is the
# primary 5,000 requests/hour per GitHub user, which is managed by the number of
# (distinct-account) tokens, not by this sleep.
GITHUB_API_SLEEP_SECONDS = 0.1


@dataclass
class RepoContributorInfoSimple(DataClassJsonMixin):
    username: str
    name: str | None
    email: str | None


@dataclass
class RepoContributorInfo(DataClassJsonMixin):
    repo_parts: types.RepoParts
    username: str
    name: str | None
    email: str | None


def _setup_gh_api(github_api_key: str | None = None) -> GhApi:
    """Create a GitHub API object."""
    # Bound every subsequent (blocking, urllib-backed) GitHub request so a
    # stalled connection raises a timeout instead of hanging the task forever.
    socket.setdefaulttimeout(GITHUB_SOCKET_TIMEOUT_SECONDS)

    # Setup API
    if github_api_key:
        api = GhApi(token=github_api_key)
    else:
        # Load env
        load_dotenv()
        api = GhApi()

    return api


@backoff.on_exception(  # type: ignore[misc]
    backoff.expo,
    (HTTP403ForbiddenError),
    max_time=16,
)
@cached(  # type: ignore[misc]
    cache=LRUCache(maxsize=2**12),
    key=lambda login, **kwargs: hashkey(login),  # Only cache by login
    lock=threading.Lock(),  # LRUCache is not thread-safe; workers run multiple threads
)
def _get_user_info_from_login(
    login: str,
    github_api_key: str | None = None,
) -> RepoContributorInfoSimple:
    # Setup API
    api = _setup_gh_api(github_api_key)

    # Get user info
    user_info = api.users.get_by_username(username=login)  # type: ignore[attr-defined]

    # Sleep to avoid API limits
    time.sleep(GITHUB_API_SLEEP_SECONDS)

    # Store info
    return RepoContributorInfoSimple(
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
    repo_parts: types.RepoParts | None,
    github_api_key: str | None = None,
    top_n: int = 30,
) -> list[RepoContributorInfo]:
    # Setup API
    api = _setup_gh_api(github_api_key)
    assert repo_parts is not None

    # Get contributors
    contributors = api.repos.list_contributors(  # type: ignore[attr-defined]
        owner=repo_parts.owner,
        repo=repo_parts.name,
        per_page=top_n,
    )

    # Sleep to avoid API limits
    time.sleep(GITHUB_API_SLEEP_SECONDS)

    # Get user infos
    _get_user_partial = partial(
        _get_user_info_from_login,
        github_api_key=github_api_key,
    )

    # Get expanding contributor info
    user_expanded_infos = [
        _get_user_partial(login=contrib["login"]) for contrib in contributors
    ]

    # Store with pair details
    return [
        RepoContributorInfo(
            repo_parts=repo_parts,
            username=user_info.username,
            name=user_info.name,
            email=user_info.email,
        )
        for user_info in user_expanded_infos
    ]


def process_github_repo(  # noqa: C901
    source: str,
    repo_parts: types.RepoParts | None,
    github_api_key: str | None = None,
    top_n: int = 30,
    fetch_repo_data: bool = True,
    fetch_repo_languages: bool = True,
    fetch_repo_readme: bool = True,
    fetch_repo_contributors: bool = True,
    fetch_repo_commits_count: bool = True,
    fetch_repo_files: bool = True,
    existing_repo_data: dict | None = None,
    existing_github_results: types.GitHubResultModels | None = None,
) -> types.GitHubResultModels | types.ErrorResult:
    """Get repo contributors and add them to database."""
    assert repo_parts is not None

    try:
        # Setup API
        api = _setup_gh_api(github_api_key)

        # Create the code host
        code_host = db_models.CodeHost(
            name=repo_parts.host,
        )

        # Get repo info
        if fetch_repo_data:
            repo_info = api.repos.get(  # type: ignore[attr-defined]
                owner=repo_parts.owner,
                repo=repo_parts.name,
            )

            # Sleep to avoid API limits
            time.sleep(GITHUB_API_SLEEP_SECONDS)

            if existing_repo_data:
                # Update existing repo data with new info
                repo_info = {**existing_repo_data, **repo_info}

        else:
            if existing_repo_data:
                repo_info = existing_repo_data
            else:
                repo_info = None

        # Get repo languages
        if fetch_repo_languages:
            repo_languages = api.repos.list_languages(  # type: ignore[attr-defined]
                owner=repo_parts.owner,
                repo=repo_parts.name,
            )

            # Sleep to avoid API limits
            time.sleep(GITHUB_API_SLEEP_SECONDS)

            # For each language, create a repository language
            repo_language_models = []
            for language, bytes_of_code in repo_languages.items():
                repo_language_models.append(
                    db_models.RepositoryLanguage(
                        # Placeholder, to be updated after repo model is created
                        repository_id=None,
                        language=language,
                        bytes_of_code=bytes_of_code,
                    )
                )
        else:
            repo_language_models = None

        # Get repo README
        if fetch_repo_readme:
            try:
                repo_readme_response = api.repos.get_readme(  # type: ignore[attr-defined]
                    owner=repo_parts.owner,
                    repo=repo_parts.name,
                )

                # Decode the content
                repo_readme = base64.b64decode(repo_readme_response["content"]).decode("utf-8")

            except Exception:
                repo_readme = None

            finally:
                # Sleep to avoid API limits
                time.sleep(GITHUB_API_SLEEP_SECONDS)

                # Create model
                repo_readme_model = db_models.RepositoryReadme(
                    # Placeholder, to be updated after repo model is created
                    repository_id=None,
                    content=repo_readme,
                )
        else:
            repo_readme_model = None

        # Get repo contributors
        if fetch_repo_contributors:
            repo_contributors = get_repo_contributors(
                repo_parts=repo_parts,
                github_api_key=github_api_key,
                top_n=top_n,
            )
        else:
            repo_contributors = None

        # Get default branch from new or existing repo model
        if existing_github_results and existing_github_results.repository_model:
            default_branch = existing_github_results.repository_model.default_branch
        elif repo_info:
            default_branch = repo_info.get("default_branch", None)
        else:
            default_branch = None

        # Get the default branch as it is needed for commit count and file lists
        if fetch_repo_commits_count and default_branch is not None:
            # Get the commit count for the default branch via checking
            # header response for the page count after rel="last"
            # https://stackoverflow.com/a/70610670
            try:
                params: dict[str, str | int | None] = {
                    "sha": default_branch,
                    "per_page": 1,
                    "page": 1,
                }

                # Use raw requests lib rather than GhApi
                # As we need the headers
                response = requests.get(
                    f"https://api.github.com/repos/{repo_parts.owner}/{repo_parts.name}/commits",
                    params=params,
                    headers=(
                        {"Authorization": f"Bearer {github_api_key}"} if github_api_key else {}
                    ),
                )

                # Raise for status
                response.raise_for_status()

                # Get the sha
                processed_at_sha = response.json()[0]["sha"]

                # Parse the Link header
                # We want the page right before rel="last"
                link_header = response.headers.get("Link", "")
                if link_header:
                    # Split by comma to get each part
                    _, last_page = link_header.split(", ")
                    # Extract the page number from the last page
                    last_page_url = last_page.split(";")[0].strip("<>")
                    commits_count = int(last_page_url.split("page=")[-1])
                else:
                    commits_count = None

            except Exception:
                processed_at_sha = None
                commits_count = None

            finally:
                # Sleep to avoid API limits
                time.sleep(GITHUB_API_SLEEP_SECONDS)
        else:
            processed_at_sha = None
            commits_count = None

        # Create the repository
        if repo_info:
            # Create repo model
            repo_model = db_models.Repository(
                code_host_id=code_host.id,
                owner=repo_parts.owner,
                name=repo_parts.name,
                description=repo_info["description"],
                is_fork=repo_info["fork"],
                forks_count=repo_info["forks_count"],
                stargazers_count=repo_info["stargazers_count"],
                watchers_count=repo_info["watchers_count"],
                open_issues_count=repo_info["open_issues_count"],
                commits_count=commits_count,
                size_kb=repo_info["size"],
                topics=";".join(repo_info["topics"]) if repo_info["topics"] else None,
                primary_language=repo_info["language"],
                default_branch=repo_info["default_branch"],
                license=repo_info["license"]["name"] if repo_info["license"] else None,
                processed_at_sha=processed_at_sha,
                creation_datetime=datetime.fromisoformat(repo_info["created_at"]),
                last_pushed_datetime=datetime.fromisoformat(repo_info["pushed_at"]),
            )

        else:
            repo_model = db_models.Repository(
                code_host_id=code_host.id,
                owner=repo_parts.owner,
                name=repo_parts.name,
                size_kb=0,
                commits_count=commits_count,
                default_branch=default_branch,
                processed_at_sha=processed_at_sha,
            )

        # Get repository files
        repo_file_models: list[db_models.RepositoryFile] | None
        if fetch_repo_files and default_branch:
            repo_file_models = []
            try:
                # Use the git trees API to get the files in the repository
                tree_results = api.git.get_tree(  # type: ignore[attr-defined]
                    owner=repo_parts.owner,
                    repo=repo_parts.name,
                    tree_sha=default_branch,
                    recursive=True,
                )

                # Iterate through the tree and create RepositoryFile models
                for file in tree_results["tree"]:
                    repo_file_models.append(
                        db_models.RepositoryFile(
                            # Placeholder, to be updated after repo model is created
                            repository_id=None,
                            path=file["path"],
                            tree_type=file["type"],
                            bytes_of_code=file.get("size", 0),
                        )
                    )

            except Exception as e:
                # Ignore the error if the tree cannot be fetched
                print(f"Error fetching repository files: {e}")

            finally:
                # Sleep to avoid API limits
                time.sleep(GITHUB_API_SLEEP_SECONDS)
        else:
            repo_file_models = None

        # For each contributor, create a developer account
        # and then link the dev account to the repo
        # as a repository contributor
        repo_contributor_models: list[types.RepositoryContributorDetails] | None
        if repo_contributors:
            repo_contributor_models = []
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
                    # Placeholder, to be updated after repo and dev account models are created
                    repository_id=repo_model.id,
                    developer_account_id=dev_account.id,
                )

                # Store the pair
                repo_contributor_models.append(
                    types.RepositoryContributorDetails(
                        developer_account_model=dev_account,
                        repository_contributor_model=repo_contributor,
                    )
                )

        else:
            repo_contributor_models = None

        # Attach the results to the pair
        github_results = types.GitHubResultModels(
            code_host_model=(
                existing_github_results.code_host_model
                if existing_github_results
                else code_host
            ),
            repository_model=(
                existing_github_results.repository_model
                if existing_github_results
                else repo_model
            ),
            repository_readme_model=(
                existing_github_results.repository_readme_model
                if existing_github_results and existing_github_results.repository_readme_model
                else repo_readme_model
            ),
            repository_language_models=(
                existing_github_results.repository_language_models
                if existing_github_results
                and existing_github_results.repository_language_models
                else repo_language_models
            ),
            repository_contributor_details=(
                existing_github_results.repository_contributor_details
                if existing_github_results
                and existing_github_results.repository_contributor_details
                else repo_contributor_models
            ),
            repository_file_models=(
                existing_github_results.repository_file_models
                if existing_github_results and existing_github_results.repository_file_models
                else repo_file_models
            ),
        )

        # Final cases where we want to update existing model fields
        # Commits Count
        if github_results.repository_model.commits_count is None and commits_count is not None:
            github_results.repository_model.commits_count = commits_count

        # Default Branch
        if (
            github_results.repository_model.default_branch is None
            and default_branch is not None
        ):
            github_results.repository_model.default_branch = default_branch

        # Processed At SHA
        if (
            github_results.repository_model.processed_at_sha is None
            and processed_at_sha is not None
        ):
            github_results.repository_model.processed_at_sha = processed_at_sha

        return github_results

    except Exception as e:
        if "Bad credentials" in str(e):
            print(
                f"!!! GitHub API Error: Bad credentials. "
                f"Please check your API key. '{github_api_key}'!!!"
            )
        return types.ErrorResult(
            source=source,
            step="process-github-repo",
            identifier=f"{repo_parts.host}:{repo_parts.owner}/{repo_parts.name}",
            error=str(e),
            traceback=traceback.format_exc(),
        )


def process_github_repo_task(
    pair: types.ExpandedRepositoryDocumentPair | types.ErrorResult,
    github_api_key: str,
    top_n: int = 30,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    # Get start time
    start_time = time.perf_counter()

    # Process the GitHub repo
    result = process_github_repo(
        source=pair.source,
        repo_parts=pair.repo_parts,
        github_api_key=github_api_key,
        top_n=top_n,
    )

    # Get end time
    end_time = time.perf_counter()

    if isinstance(result, types.ErrorResult):
        return result

    # Attach processing time if successful
    pair.github_results = result
    pair.github_processing_time_seconds = end_time - start_time

    return pair


def get_github_repos_for_developer(
    username: str,
    github_api_key: str | None = None,
) -> list[types.GitHubResultModels | types.ErrorResult] | types.ErrorResult:
    """Get all GitHub repositories for a developer account."""
    # Setup API
    api = _setup_gh_api(github_api_key)

    try:
        # Page and get repos
        repo_pager = paged(
            api.repos.list_for_user,  # type: ignore[attr-defined]
            username=username,
            type="all",
            per_page=100,
        )

        developer_repos = []
        for page in repo_pager:
            time.sleep(GITHUB_API_SLEEP_SECONDS)  # Sleep to avoid API limits
            if page is None:
                continue
            developer_repos.extend(page)

        # Get the GitHub response object
        return [
            process_github_repo(
                source="snowball-sampling-discovery",
                repo_parts=types.RepoParts(
                    host="github",
                    owner=repo["owner"]["login"].lower(),
                    name=repo["name"].lower(),
                ),
                github_api_key=github_api_key,
                fetch_repo_data=False,
                fetch_repo_languages=False,
                fetch_repo_readme=False,
                fetch_repo_contributors=False,
                fetch_repo_commits_count=False,
                fetch_repo_files=False,
                existing_repo_data=repo,
            )
            for repo in developer_repos
        ]

    except Exception as e:
        if "Bad credentials" in str(e):
            print(
                f"!!! GitHub API Error: Bad credentials. "
                f"Please check your API key. '{github_api_key}'!!!"
            )
        return types.ErrorResult(
            source="snowball-sampling-discovery",
            step="github-repos-for-developer",
            identifier=username,
            error=str(e),
            traceback=traceback.format_exc(),
        )
