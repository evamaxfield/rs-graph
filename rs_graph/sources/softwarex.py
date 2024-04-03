#!/usr/bin/env python

from __future__ import annotations

import logging
import os
import time
import traceback

import backoff
import requests
from dotenv import load_dotenv
from fastcore.net import HTTP403ForbiddenError
from ghapi.all import GhApi, paged
from tqdm import tqdm

from ..types import ErrorResult, RepositoryDocumentPair, SuccessAndErroredResultsLists
from .proto import DataSource

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

ELSEVIER_PAPER_SEARCH_URL_TEMPLATE = (
    "https://api.elsevier.com/content/search/"
    "sciencedirect?query={query}&apiKey={apiKey}"
)

###############################################################################


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
) -> RepositoryDocumentPair | ErrorResult:
    # Be nice to APIs
    time.sleep(0.8)

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
        except Exception:
            log.debug(f"Error getting response from Elsevier API for repo: {repo_name}")
            return ErrorResult(
                source="softwarex",
                identifier=repo_name,
                error="Error getting response",
                traceback=traceback.format_exc(),
            )

    # Parse response to json
    response_json = response.json()

    # Ensure the schema is returned properly
    try:
        # Check number of results
        num_results = len(response_json["search-results"]["entry"])
        if num_results != 1:
            log.debug(
                f"Unexpected number of results for repo: {repo_name} ({num_results})"
            )
            return ErrorResult(
                source="softwarex",
                identifier=repo_name,
                error="Unexpected number of results",
                traceback="",
            )

        # Get the first result
        first_result = response_json["search-results"]["entry"][0]

        # Get the title, doi, and published date
        doi = first_result["prism:doi"]
    except Exception:
        log.debug(
            f"Error parsing response json from Elsevier API for repo: {repo_name}"
        )
        return ErrorResult(
            source="softwarex",
            identifier=repo_name,
            error="Error parsing response json",
            traceback=traceback.format_exc(),
        )

    # Return the result
    return RepositoryDocumentPair(
        source="softwarex",
        repo_url=repo_url,
        paper_doi=doi,
    )


class SoftwareXDataSource(DataSource):
    @staticmethod
    def get_dataset(
        use_dask: bool = False,
        **kwargs: dict[str, str],
    ) -> SuccessAndErroredResultsLists:
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
            time.sleep(1)

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
            if isinstance(result, RepositoryDocumentPair):
                successful_results.append(result)
            else:
                errored_results.append(result)

        # Log total succeeded and errored
        log.info(f"Total succeeded: {len(successful_results)}")
        log.info(f"Total errored: {len(errored_results)}")

        # Return filepath
        return SuccessAndErroredResultsLists(
            successful_results=successful_results,
            errored_results=errored_results,
        )
