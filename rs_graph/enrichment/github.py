#!/usr/bin/env python

from __future__ import annotations

import base64
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import backoff
import requests
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from fastcore.net import HTTP403ForbiddenError
from ghapi.all import GhApi, paged

from .. import types
from ..db import models as db_models

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


def process_github_repo(  # noqa: C901
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

        # Get repo languages
        repo_languages = api.repos.list_languages(
            owner=pair.repo_parts.owner,
            repo=pair.repo_parts.name,
        )

        # Sleep to avoid API limits
        time.sleep(0.85)

        # Get repo README
        try:
            repo_readme_response = api.repos.get_readme(
                owner=pair.repo_parts.owner,
                repo=pair.repo_parts.name,
            )

            # Decode the content
            repo_readme = base64.b64decode(repo_readme_response["content"]).decode("utf-8")

        except Exception:
            repo_readme = None

        finally:
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

        # Get the default branch as it is needed for commit count and file lists
        default_branch = repo_info.get("default_branch", None)
        if default_branch is not None:
            # Get the commit count for the default branch via checking
            # header response for the page count after rel="last"
            # https://stackoverflow.com/a/70610670
            try:
                # Use raw requests lib rather than GhApi
                # As we need the headers
                response = requests.get(
                    f"https://api.github.com/repos/{pair.repo_parts.owner}/{pair.repo_parts.name}/commits",
                    params={"sha": default_branch, "per_page": 1, "page": 1},
                    headers={"Authorization": f"Bearer {github_api_key}"}
                    if github_api_key
                    else {},
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
                time.sleep(0.85)
        else:
            commits_count = None

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
            commits_count=commits_count,
            size_kb=repo_info["size"],
            topics=";".join(repo_info["topics"]) if repo_info["topics"] else None,
            primary_language=repo_info["language"],
            default_branch=default_branch,
            license=repo_info["license"]["name"] if repo_info["license"] else None,
            processed_at_sha=processed_at_sha,
            creation_datetime=datetime.fromisoformat(repo_info["created_at"]),
            last_pushed_datetime=datetime.fromisoformat(repo_info["pushed_at"]),
        )

        # For each language, create a repository language
        all_repo_languages = []
        for language, bytes_of_code in repo_languages.items():
            all_repo_languages.append(
                db_models.RepositoryLanguage(
                    repository_id=repo.id,
                    language=language,
                    bytes_of_code=bytes_of_code,
                )
            )

        # Get repository files
        all_repo_files = []
        if default_branch:
            try:
                # Use the git trees API to get the files in the repository
                tree_results = api.git.get_tree(
                    owner=pair.repo_parts.owner,
                    repo=pair.repo_parts.name,
                    tree_sha=default_branch,
                    recursive=True,
                )

                # Iterate through the tree and create RepositoryFile models
                for file in tree_results["tree"]:
                    all_repo_files.append(
                        db_models.RepositoryFile(
                            repository_id=repo.id,
                            path=file["path"],
                            tree_type=file["type"],
                            bytes_of_code=file.get("size", 0),
                        )
                    )

            except Exception as e:
                # Ignore the error if the tree cannot be fetched
                all_repo_files = []
                print(f"Error fetching repository files: {e}")

            finally:
                # Sleep to avoid API limits
                time.sleep(0.85)

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
            repository_readme_model=db_models.RepositoryReadme(
                repository_id=repo.id,
                content=repo_readme,
            ),
            repository_language_models=all_repo_languages,
            repository_contributor_details=all_repo_contributors,
            repository_file_models=all_repo_files,
        )

        return pair

    except Exception as e:
        return types.ErrorResult(
            source=pair.source,
            step="github-repo-contributors",
            identifier=pair.paper_doi,
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
        pair=pair,
        github_api_key=github_api_key,
        top_n=top_n,
    )

    # Get end time
    end_time = time.perf_counter()

    # Attach processing time if successful
    if isinstance(result, types.ExpandedRepositoryDocumentPair):
        result.github_processing_time_seconds = end_time - start_time

    return result


def get_github_repos_for_developer(
    username: str,
    github_api_key: str | None = None,
) -> list[dict] | types.ErrorResult:
    """Get all GitHub repositories for a developer account."""
    # Setup API
    api = _setup_gh_api(github_api_key)

    try:
        # Page and get repos
        repo_pager = paged(
            api.repos.list_for_user,
            username=username,
            type="all",
            per_page=100,
        )

        developer_repos = []
        for page in repo_pager:
            time.sleep(0.85)  # Sleep to avoid API limits
            developer_repos.extend(page)

        return developer_repos

    except Exception as e:
        return types.ErrorResult(
            source="snowball-sampling-discovery",
            step="github-repos-for-developer",
            identifier=username,
            error=str(e),
            traceback=traceback.format_exc(),
        )
