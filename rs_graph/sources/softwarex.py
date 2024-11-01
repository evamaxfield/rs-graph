#!/usr/bin/env python

from __future__ import annotations

import os
import time
import traceback

import backoff
import requests
from dotenv import load_dotenv
from fastcore.net import HTTP403ForbiddenError
from ghapi.all import GhApi, paged
from tqdm import tqdm

from .. import types

###############################################################################

ELSEVIER_PAPER_SEARCH_URL_TEMPLATE = (
    "https://api.elsevier.com/content/search/"
    "sciencedirect?query={query}&apiKey={apiKey}"
)

###############################################################################

# TODO: use pybliometrics and "ScienceDirectSearch" as follows
# to search for the softwarex papers associated to a github repo
# pybliometrics.sciencedirect.init(keys=LIST_OF_API_KEYS)
# s = ScienceDirectSearch(
#     query="SOFTX-D-24-00113",  # this is the repo name
#     subscriber=False,
# )
# s.results
#
# In some cases this will give us a single item list
# and in others it will give back multiple items.
# As a first pass we can try with just single items
# There may be a way to find the correct item in the multi-item case
# by comparing repo creation date with paper publication date


class RateLimitError(Exception):
    pass


@backoff.on_exception(
    backoff.expo,
    (HTTP403ForbiddenError, RateLimitError),
    max_time=300,
)
def _process_elsevier_repo(
    repo_name: str,
    github_api: GhApi,
    elsevier_api_key: str,
) -> types.BasicRepositoryDocumentPair | types.ErrorResult:
    # Be nice to APIs
    time.sleep(0.85)

    # Get Elsevier repo details
    repo_details = github_api.repos.get(
        owner="ElsevierSoftwareX",
        repo=repo_name,
    )

    # Get parent repo url
    if "parent" in repo_details:
        repo_url = repo_details["parent"]["html_url"]
    else:
        repo_url = f"https://github.com/ElsevierSoftwareX/{repo_name}"

    # Get paper details
    response = requests.get(
        ELSEVIER_PAPER_SEARCH_URL_TEMPLATE.format(
            query=repo_name,
            apiKey=elsevier_api_key,
        )
    )

    # Ensure response is successful
    if response.status_code == 429:
        raise RateLimitError("Rate limit exceeded for Elsevier API")
    else:
        try:
            response.raise_for_status()
        except Exception as e:
            print(f"Error getting response from Elsevier API for repo: {repo_name}")
            return types.ErrorResult(
                source="softwarex",
                step="elsevier-repo-processing",
                identifier=repo_url,
                error=str(e),
                traceback=traceback.format_exc(),
            )

    # Parse response to json
    response_json = response.json()

    # Ensure the schema is returned properly
    try:
        # Check number of results
        num_results = len(response_json["search-results"]["entry"])
        if num_results != 1:
            print(f"Unexpected number of results for repo: {repo_name} ({num_results})")
            return types.ErrorResult(
                source="softwarex",
                step="elsevier-repo-processing",
                identifier=repo_url,
                error="Unexpected number of results",
                traceback="",
            )

        # Get the first result
        first_result = response_json["search-results"]["entry"][0]

        # Get the title, doi, and published date
        doi = first_result["prism:doi"]
    except Exception as e:
        print(f"Error parsing response json from Elsevier API for repo: {repo_name}")
        return types.ErrorResult(
            source="softwarex",
            step="elsevier-repo-processing",
            identifier=repo_url,
            error=str(e),
            traceback=traceback.format_exc(),
        )

    # Return the result
    return types.BasicRepositoryDocumentPair(
        source="softwarex",
        repo_url=repo_url,
        paper_doi=doi,
    )


def get_dataset(
    **kwargs: dict[str, str],
) -> types.SuccessAndErroredResultsLists:
    """Download the SoftwareX dataset."""
    # Load env
    load_dotenv()

    # Setup API
    github_api = GhApi()

    # Get elsevier api key
    elsevier_api_key = os.getenv("ELSEVIER_API_KEY")

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

        # Only extract the repo name from each repo detail
        only_repo_names = [repo["name"] for repo in page]
        all_softwarex_repos.extend(only_repo_names)

    # Get original parent repo for each elsevier repo and get paper details
    successful_results = []
    errored_results = []
    for repo_name in tqdm(
        all_softwarex_repos,
        desc="Getting SoftwareX paper details",
    ):
        # Get paper details
        result = _process_elsevier_repo(
            repo_name=repo_name,
            github_api=github_api,
            elsevier_api_key=elsevier_api_key,
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
