#!/usr/bin/env python

from dataclasses import dataclass

from parse import Parser

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


@dataclass
class CodeHostResult:
    host: str
    owner: str | None
    name: str | None


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


def _clean_extra_slashes_from_repo(repo: str) -> str:
    return repo.split("/")[0]


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
        result.name = _clean_extra_slashes_from_repo(result.name)

    # For all hosts except SourceForge, the owner is required
    if result.host != "sourceforge" and result.owner is None:
        raise ValueError("Could not parse code host URL")

    # For SourceForge, the name is required
    if result.host == "sourceforge" and result.name is None:
        raise ValueError("Could not parse code host URL")

    return result
