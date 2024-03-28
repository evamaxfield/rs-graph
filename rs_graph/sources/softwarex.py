#!/usr/bin/env python

from __future__ import annotations

import logging
import os
import time

import backoff
import requests
from dotenv import load_dotenv
from fastcore.net import HTTP403ForbiddenError
from ghapi.all import GhApi, paged
from tqdm import tqdm

from .proto import DataSource, RepositoryDocumentPair

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
) -> RepositoryDocumentPair | None:
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
        response.raise_for_status()

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
            return None

        # Get the first result
        first_result = response_json["search-results"]["entry"][0]

        # Get the title, doi, and published date
        doi = first_result["prism:doi"]
    except Exception:
        log.debug(
            f"Error parsing response json from Elsevier API for repo: {repo_name}"
        )
        return None

    # Return the result
    return RepositoryDocumentPair(
        source="softwarex",
        repo_url=repo_url,
        paper_doi=doi,
    )


class SoftwareXDataSource(DataSource):
    @staticmethod
    def get_dataset(
        **kwargs: dict[str, str],
    ) -> list[RepositoryDocumentPair]:
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
        all_paper_details = []
        for repo_name in tqdm(
            all_softwarex_repos,
            desc="Getting SoftwareX paper details",
        ):
            # Get paper details
            paper_details = _process_elsevier_repo(
                repo_name=repo_name,
                github_api=github_api,
                elsevier_api_key=elsevier_api_key,
            )
            all_paper_details.append(paper_details)

        # Get count of total processed and errored
        total_processed = len(all_paper_details)
        processed_correctly = [
            paper for paper in all_paper_details if paper is not None
        ]
        total_errored = total_processed - len(processed_correctly)

        # Log total processed and errored
        log.info(f"Total succeeded: {len(processed_correctly)}")
        log.info(f"Total errored: {total_errored}")

        # Return filepath
        return processed_correctly
