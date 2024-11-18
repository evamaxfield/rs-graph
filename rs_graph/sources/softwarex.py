#!/usr/bin/env python

from __future__ import annotations

import itertools
import time
import traceback

import backoff
import pybliometrics
from ghapi.all import GhApi, paged
from pybliometrics.sciencedirect import ScienceDirectSearch
from tqdm import tqdm

from .. import types

###############################################################################


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_time=120,
)
def _process_elsevier_repo(
    repo_name: str,
    github_api_token: str,
    elsevier_api_keys: str,
) -> types.BasicRepositoryDocumentPair | types.ErrorResult:
    # Be nice to APIs
    time.sleep(0.85)

    # Get Elsevier repo details
    github_api = GhApi(token=github_api_token)
    repo_details = github_api.repos.get(
        owner="ElsevierSoftwareX",
        repo=repo_name,
    )

    # Get parent repo url
    if "parent" in repo_details:
        repo_url = repo_details["parent"]["html_url"]
    else:
        repo_url = f"https://github.com/ElsevierSoftwareX/{repo_name}"

    # Init API
    pybliometrics.sciencedirect.init(keys=elsevier_api_keys)

    # Search
    try:
        # Search SciDirect
        s = ScienceDirectSearch(
            query=f'"{repo_name}"',
            subscriber=False,
        )

        # Get results
        results = s.results

        # Handle single
        if len(results) == 0:
            raise ValueError(f"No papers found for {repo_name}")
        if len(results) == 1:
            doi = results[0].doi

        # Handle multiple
        else:
            # Don't know how to handle right now
            raise ValueError(f"Multiple papers found for {repo_name}")

        # Return the result
        return types.BasicRepositoryDocumentPair(
            source="softwarex",
            repo_url=repo_url,
            paper_doi=doi,
        )

    except Exception as e:
        return types.ErrorResult(
            source="softwarex",
            step="elsevier-repo-processing",
            identifier=repo_url,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def get_dataset(
    github_tokens: list[str],
    elsevier_api_keys: list[str],
    **kwargs: dict[str, str],
) -> types.SuccessAndErroredResultsLists:
    """Download the SoftwareX dataset."""
    # Cycle github tokens
    github_tokens_cycle = itertools.cycle(github_tokens)

    # Setup API
    github_api = GhApi(token=next(github_tokens_cycle))

    # Get softwareX repos names
    paged_repos = paged(
        github_api.repos.list_for_org,
        org="ElsevierSoftwareX",
    )
    all_softwarex_repos = []
    for page in tqdm(
        paged_repos,
        desc="Getting ElsevierSoftwareX repos",
    ):
        # Be nice to APIs
        time.sleep(0.85)

        # Get repo name, created at, and description
        software_x_repo_details = [
            {
                "name": repo["name"],
                "created_at": repo["created_at"],
                "description": repo["description"],
            }
            for repo in page
        ]
        all_softwarex_repos.extend(software_x_repo_details)

    # Get original parent repo for each elsevier repo and get paper details
    successful_results = []
    errored_results = []
    for repo_details in tqdm(
        all_softwarex_repos[::-1],
        desc="Getting SoftwareX paper details",
    ):
        # Get paper details
        result = _process_elsevier_repo(
            repo_name=repo_details["name"],
            github_api_token=next(github_tokens_cycle),
            elsevier_api_keys=elsevier_api_keys,
        )

        # Add to results
        if isinstance(result, types.BasicRepositoryDocumentPair):
            successful_results.append(result)
        else:
            errored_results.append(result)

    # Log total succeeded and errored
    print(f"Total succeeded: {len(successful_results)}")
    print(f"Total errored: {len(errored_results)}")

    # Return filepath
    return types.SuccessAndErroredResultsLists(
        successful_results=successful_results,
        errored_results=errored_results,
    )
