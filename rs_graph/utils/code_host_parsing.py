#!/usr/bin/env python

import logging
import traceback

from parse import Parser

from ..types import (
    CodeHostResult,
    ErrorResult,
    ExpandedRepositoryDocumentPair,
    RepositoryDocumentPair,
    SuccessAndErroredResultsLists,
)
from .dask_functions import process_func

#######################################################################################

log = logging.getLogger(__name__)

#######################################################################################

# GitHub Parsers
GITHUB_COM_REPO_PARSER = Parser(
    "github.com/{owner}/{repo}/",
)
GITHUB_IO_REPO_PARSER = Parser(
    "{owner}.github.io/{repo}/",
)
GITHUB_IO_REPO_PARSER_TYPO = Parser(
    "{owner}.github.com/{repo}",
)
GITHUB_COM_OWNER_PARSER = Parser(
    "github.com/{owner}/",
)
GITHUB_IO_OWNER_PARSER = Parser(
    "{owner}.github.io/",
)
GITHUB_IO_OWNER_PARSER_TYPO = Parser(
    "{owner}.github.com/",
)

# GitLab Parsers
GITLAB_COM_REPO_PARSER = Parser(
    "gitlab.com/{owner}/{repo}/",
)
GITLAB_IO_REPO_PARSER = Parser(
    "{owner}.gitlab.io/{repo}/",
)
GITLAB_IO_REPO_PARSER_TYPO = Parser(
    "{owner}.gitlab.com/{repo}/",
)
GITLAB_COM_OWNER_PARSER = Parser(
    "gitlab.com/{owner}/",
)
GITLAB_IO_OWNER_PARSER = Parser(
    "{owner}.gitlab.io/",
)
GITLAB_IO_OWNER_PARSER_TYPO = Parser(
    "{owner}.gitlab.com/",
)

# Bitbucket Parsers
BITBUCKET_ORG_REPO_PARSER = Parser(
    "bitbucket.org/{owner}/{repo}/",
)

# SourceForge Parsers
SOURCEFORGE_REPO_PARSER = Parser(
    "sourceforge.net/projects/{repo}/",
)

#######################################################################################


def _parse_github_urls(url: str) -> CodeHostResult | None:
    if result := GITHUB_COM_REPO_PARSER.parse(url):
        return CodeHostResult("github", result["owner"], result["repo"])
    if result := GITHUB_IO_REPO_PARSER.parse(url):
        return CodeHostResult("github", result["owner"], result["repo"])
    if result := GITHUB_IO_REPO_PARSER_TYPO.parse(url):
        return CodeHostResult("github", result["owner"], result["repo"])
    if result := GITHUB_COM_OWNER_PARSER.parse(url):
        return CodeHostResult("github", result["owner"], None)
    if result := GITHUB_IO_OWNER_PARSER.parse(url):
        return CodeHostResult("github", result["owner"], None)
    if result := GITHUB_IO_OWNER_PARSER_TYPO.parse(url):
        return CodeHostResult("github", result["owner"], None)

    return None


def _parse_gitlab_urls(url: str) -> CodeHostResult | None:
    if result := GITLAB_COM_REPO_PARSER.parse(url):
        return CodeHostResult("gitlab", result["owner"], result["repo"])
    if result := GITLAB_IO_REPO_PARSER.parse(url):
        return CodeHostResult("gitlab", result["owner"], result["repo"])
    if result := GITLAB_IO_REPO_PARSER_TYPO.parse(url):
        return CodeHostResult("gitlab", result["owner"], result["repo"])
    if result := GITLAB_COM_OWNER_PARSER.parse(url):
        return CodeHostResult("gitlab", result["owner"], None)
    if result := GITLAB_IO_OWNER_PARSER.parse(url):
        return CodeHostResult("gitlab", result["owner"], None)
    if result := GITLAB_IO_OWNER_PARSER_TYPO.parse(url):
        return CodeHostResult("gitlab", result["owner"], None)

    return None


def _parse_bitbucket_urls(url: str) -> CodeHostResult | None:
    if result := BITBUCKET_ORG_REPO_PARSER.parse(url):
        return CodeHostResult("bitbucket", result["owner"], result["repo"])

    return None


def _parse_sourceforge_urls(url: str) -> CodeHostResult | None:
    if result := SOURCEFORGE_REPO_PARSER.parse(url):
        return CodeHostResult("sourceforge", None, result["repo"])

    return None


def _clean_repo_name(repo: str) -> str:
    name = repo.split("/")[0]
    name = name.replace(".git", "")

    return name


def parse_code_host_url(url: str) -> CodeHostResult:
    # Standardize the URL
    url = url.strip().lower()

    # Get rid of HTTPS, then HTTP, the www.
    url = url.replace("https://", "")
    url = url.replace("http://", "")
    url = url.replace("www.", "")

    # If no trailing slash always add one
    if url[-1] != "/":
        url += "/"

    # Run the parsers
    result = None
    if gh_result := _parse_github_urls(url):
        result = gh_result
    if gl_result := _parse_gitlab_urls(url):
        result = gl_result
    if bb_result := _parse_bitbucket_urls(url):
        result = bb_result
    if sf_result := _parse_sourceforge_urls(url):
        result = sf_result

    # If no result, raise an error
    if result is None:
        raise ValueError("Could not parse code host URL")

    # Clean the repo name
    if result.name:
        result.name = _clean_repo_name(result.name)

    # For all hosts except SourceForge, the owner is required
    if result.host != "sourceforge" and result.owner is None:
        raise ValueError("Could not parse code host URL")

    # For SourceForge, the name is required
    if result.host == "sourceforge" and result.name is None:
        raise ValueError("Could not parse code host URL")

    return result


def _wrapped_parse_code_host_url(
    repo_paper_pair: RepositoryDocumentPair,
) -> ExpandedRepositoryDocumentPair | ErrorResult:
    try:
        code_host_res = parse_code_host_url(repo_paper_pair.repo_url)

        # Check if github
        if (
            code_host_res.host != "github"
            or code_host_res.owner is None
            or code_host_res.name is None
        ):
            return ErrorResult(
                source=repo_paper_pair.source,
                step="code-host-parsing",
                identifier=repo_paper_pair.repo_url,
                error="Not a GitHub repository",
                traceback="",
            )

        return ExpandedRepositoryDocumentPair(
            source=repo_paper_pair.source,
            repo_host=code_host_res.host,
            repo_owner=code_host_res.owner,
            repo_name=code_host_res.name,
            paper_doi=repo_paper_pair.paper_doi,
            paper_extra_data=repo_paper_pair.paper_extra_data,
        )

    except ValueError:
        return ErrorResult(
            source=repo_paper_pair.source,
            step="code-host-parsing",
            identifier=repo_paper_pair.repo_url,
            error="Could not parse code host URL",
            traceback=traceback.format_exc(),
        )


def filter_repo_paper_pairs(
    pairs: list[RepositoryDocumentPair],
    use_dask: bool = False,
    **kwargs: dict,
) -> SuccessAndErroredResultsLists:
    # Filter each repo pair
    # Accepting only GitHub full repository results
    raw_results = process_func(
        name="code-host-parsing",
        func=_wrapped_parse_code_host_url,
        func_iterables=[pairs],
        use_dask=use_dask,
        cluster_kwargs={
            "processes": True,
            "n_workers": 4,
            "threads_per_worker": 1,
        },
    )

    # Split results
    split_results = SuccessAndErroredResultsLists(
        successful_results=[
            result
            for result in raw_results
            if isinstance(result, ExpandedRepositoryDocumentPair)
        ],
        errored_results=[
            result for result in raw_results if isinstance(result, ErrorResult)
        ],
    )

    # Log total succeeded and errored
    log.info(f"Total succeeded: {len(split_results.successful_results)}")
    log.info(f"Total errored: {len(split_results.errored_results)}")

    return split_results
