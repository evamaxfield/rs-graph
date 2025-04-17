#!/usr/bin/env python

import traceback

from parse import Parser
from tqdm import tqdm

from .. import types

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


def _parse_github_urls(url: str) -> types.CodeHostResult | None:
    if result := GITHUB_COM_REPO_PARSER.parse(url):
        return types.CodeHostResult("github", result["owner"], result["repo"])
    if result := GITHUB_IO_REPO_PARSER.parse(url):
        return types.CodeHostResult("github", result["owner"], result["repo"])
    if result := GITHUB_IO_REPO_PARSER_TYPO.parse(url):
        return types.CodeHostResult("github", result["owner"], result["repo"])
    if result := GITHUB_COM_OWNER_PARSER.parse(url):
        return types.CodeHostResult("github", result["owner"], None)
    if result := GITHUB_IO_OWNER_PARSER.parse(url):
        return types.CodeHostResult("github", result["owner"], None)
    if result := GITHUB_IO_OWNER_PARSER_TYPO.parse(url):
        return types.CodeHostResult("github", result["owner"], None)

    return None


def _parse_gitlab_urls(url: str) -> types.CodeHostResult | None:
    if result := GITLAB_COM_REPO_PARSER.parse(url):
        return types.CodeHostResult("gitlab", result["owner"], result["repo"])
    if result := GITLAB_IO_REPO_PARSER.parse(url):
        return types.CodeHostResult("gitlab", result["owner"], result["repo"])
    if result := GITLAB_IO_REPO_PARSER_TYPO.parse(url):
        return types.CodeHostResult("gitlab", result["owner"], result["repo"])
    if result := GITLAB_COM_OWNER_PARSER.parse(url):
        return types.CodeHostResult("gitlab", result["owner"], None)
    if result := GITLAB_IO_OWNER_PARSER.parse(url):
        return types.CodeHostResult("gitlab", result["owner"], None)
    if result := GITLAB_IO_OWNER_PARSER_TYPO.parse(url):
        return types.CodeHostResult("gitlab", result["owner"], None)

    return None


def _parse_bitbucket_urls(url: str) -> types.CodeHostResult | None:
    if result := BITBUCKET_ORG_REPO_PARSER.parse(url):
        return types.CodeHostResult("bitbucket", result["owner"], result["repo"])

    return None


def _parse_sourceforge_urls(url: str) -> types.CodeHostResult | None:
    if result := SOURCEFORGE_REPO_PARSER.parse(url):
        return types.CodeHostResult("sourceforge", None, result["repo"])

    return None


def _clean_repo_name(repo: str) -> str:
    name = repo.split("/")[0]
    name = name.replace(".git", "")

    return name


def _clean_code_host_url(url: str) -> tuple[str, str]:
    # Standardize the URL
    cleaned_url = url.strip().lower()

    # Get rid of HTTPS, then HTTP, the www.
    cleaned_url = cleaned_url.replace("https://", "")
    cleaned_url = cleaned_url.replace("http://", "")
    cleaned_url = cleaned_url.replace("www.", "")

    # Handle SSH Git URLs (git@github.com:owner/repo.git)
    if cleaned_url.startswith("git@"):
        cleaned_url = cleaned_url.replace("git@", "")
        cleaned_url = cleaned_url.replace(":", "/")

    # Remove .git suffix if present
    if cleaned_url.endswith(".git"):
        cleaned_url = cleaned_url[:-4]

    # Remove query parameters and fragments
    cleaned_url = cleaned_url.split("?")[0]
    cleaned_url = cleaned_url.split("#")[0]

    # Extract base repository URL without subdirectory paths
    # Match patterns like github.com/owner/repo/additional/path
    parts = cleaned_url.split("/")
    if len(parts) >= 3:
        base_url = "/".join(parts[:3])
        # Add trailing slash for parser matching
        base_url = base_url + "/"
    else:
        # Add trailing slash if needed for parser matching
        if cleaned_url[-1] != "/":
            cleaned_url += "/"
        base_url = cleaned_url

    return cleaned_url, base_url


def parse_code_host_url(url: str) -> types.CodeHostResult:
    # Clean the URL
    cleaned_url, base_url = _clean_code_host_url(url)

    # Run the parsers on the base URL
    if gh_result := _parse_github_urls(base_url):
        result = gh_result
    elif gl_result := _parse_gitlab_urls(base_url):
        result = gl_result
    elif bb_result := _parse_bitbucket_urls(base_url):
        result = bb_result
    elif sf_result := _parse_sourceforge_urls(base_url):
        result = sf_result
    else:
        # If no result, raise an error
        raise ValueError(
            f"Could not parse code host URL: {url} "
            f"(after cleaning: {cleaned_url}, and found base: {base_url})"
        )

    # Clean the repo name
    if result.name:
        result.name = _clean_repo_name(result.name)

    # For all hosts except SourceForge, the owner is required
    if result.host != "sourceforge" and result.owner is None:
        raise ValueError(f"Could not parse code host URL (missing owner): {url}")

    # For SourceForge, the name is required
    if result.host == "sourceforge" and result.name is None:
        raise ValueError(f"Could not parse code host URL (missing repo name): {url}")

    return result


def _wrapped_parse_code_host_url(
    pair: types.BasicRepositoryDocumentPair,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    try:
        code_host_res = parse_code_host_url(pair.repo_url)

        # Check if we have valid repository information
        # For GitHub, GitLab, and Bitbucket, we need both owner and name
        if code_host_res.host in ["github", "gitlab", "bitbucket"] and (
            code_host_res.owner is None or code_host_res.name is None
        ):
            return types.ErrorResult(
                source=pair.source,
                step="code-host-parsing",
                identifier=pair.paper_doi,
                error=f"Invalid {code_host_res.host} repository: missing owner or name",
                traceback="",
            )

        # For SourceForge, we only need the name
        if code_host_res.host == "sourceforge" and code_host_res.name is None:
            return types.ErrorResult(
                source=pair.source,
                step="code-host-parsing",
                identifier=pair.paper_doi,
                error="Invalid SourceForge repository: missing name",
                traceback="",
            )

        return types.ExpandedRepositoryDocumentPair(
            source=pair.source,
            repo_parts=types.RepoParts(
                host=code_host_res.host,
                owner=code_host_res.owner if code_host_res.owner else "",
                name=code_host_res.name if code_host_res.name else "",
            ),
            paper_doi=pair.paper_doi,
            paper_extra_data=pair.paper_extra_data,
        )

    except ValueError as e:
        return types.ErrorResult(
            source=pair.source,
            step="code-host-parsing",
            identifier=pair.paper_doi,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def filter_repo_paper_pairs(
    pairs: list[types.BasicRepositoryDocumentPair],
    **kwargs: dict,
) -> types.SuccessAndErroredResultsLists:
    # Filter each repo pair
    # Accepting only GitHub full repository results
    results = [
        _wrapped_parse_code_host_url(pair)
        for pair in tqdm(
            pairs,
            desc="Filtering Repository Pairs",
        )
    ]

    # Split results
    split_results = types.SuccessAndErroredResultsLists(
        successful_results=[
            result
            for result in results
            if isinstance(result, types.ExpandedRepositoryDocumentPair)
        ],
        errored_results=[
            result for result in results if isinstance(result, types.ErrorResult)
        ],
    )

    # Log total succeeded and errored
    print(f"Total succeeded: {len(split_results.successful_results)}")
    print(f"Total errored: {len(split_results.errored_results)}")

    return split_results
