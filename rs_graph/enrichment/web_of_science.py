#!/usr/bin/env python

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass

import requests
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv

from .. import types

###############################################################################

WEB_OF_SCIENCE_API_URL = "https://wos-api.clarivate.com/api/wos/"
DEFAULT_API_PARAMS = {
    "databaseId": "WOS",
    "optionView": "FR",
    "sortField": "LD+D",
}
DEFAULT_API_HEADERS = {
    "accept": "application/json",
}


@dataclass
class NullGrantArticleSearchResult(DataClassJsonMixin):
    grant_id: str
    funder: str


###############################################################################


def _get_articles_from_grant_id_and_funder(  # noqa: C901
    grant_id: str,
    funder: str,
    wos_api_key: str,
) -> (
    list[types.DocumentWithGrantInformation]
    | NullGrantArticleSearchResult
    | types.ErrorResult
):
    try:
        # Create a copy of params
        api_params = DEFAULT_API_PARAMS.copy()

        # Add user quer to params
        api_params["usrQuery"] = f"fo=({funder}) AND fg=({grant_id})"

        # Add API key to headers
        api_headers = DEFAULT_API_HEADERS.copy()
        api_headers["X-ApiKey"] = wos_api_key

        # Request and handle response
        response = requests.get(
            WEB_OF_SCIENCE_API_URL, params=api_params, headers=api_headers
        )
        response.raise_for_status()

        # Parse response
        data = response.json()

        # Check for any records
        if (
            "Data" not in data
            or "Records" not in data["Data"]
            or "records" not in data["Data"]["Records"]
        ):
            return types.ErrorResult(
                source="Web of Science",
                step="get_articles_from_grant_id_and_funder",
                identifier=f"{funder};{grant_id}",
                error="Unexpected response structure",
                traceback="",
            )

        # Check if we have any records
        records = data["Data"]["Records"]["records"]
        if isinstance(records, str):
            # No records found
            return []

        # Check if we have any articles
        if "REC" not in records or len(records["REC"]) == 0:
            return []

        # Iter over all records
        article_grant_pairs = []
        for record in records["REC"]:
            # Check if we have a DOI
            if (
                "dynamic_data" not in record
                or "cluster_related" not in record["dynamic_data"]
                or "identifiers" not in record["dynamic_data"]["cluster_related"]
                or "identifier"
                not in record["dynamic_data"]["cluster_related"]["identifiers"]
            ):
                continue

            # Get the DOI from the identifiers
            article_identifiers = record["dynamic_data"]["cluster_related"][
                "identifiers"
            ]["identifier"]

            # Find the DOI
            paper_doi_or_none: str | None = None
            for identifier in article_identifiers:
                if identifier["type"] == "doi":
                    paper_doi_or_none = identifier["value"]
                    break

            # If we found a DOI, add to results
            if paper_doi_or_none is not None:
                article_grant_pairs.append(
                    types.DocumentWithGrantInformation(
                        grant_id=grant_id,
                        funder=funder,
                        paper_doi=paper_doi_or_none,
                    )
                )

        if len(article_grant_pairs) == 0:
            # No articles found for this grant and funder
            return NullGrantArticleSearchResult(grant_id=grant_id, funder=funder)

        return article_grant_pairs

    except Exception as e:
        return types.ErrorResult(
            source="Web of Science",
            step="get_articles_from_grant_id_and_funder",
            identifier=f"{funder};{grant_id}",
            error=str(e),
            traceback=traceback.format_exc(),
        )


def get_articles_from_grant_id_and_funder(
    grant_id: str,
    funder: str,
    wos_api_key: str | None = None,
) -> (
    list[types.DocumentWithGrantInformation]
    | NullGrantArticleSearchResult
    | types.ErrorResult
):
    # Get the API key from environment variables if not provided
    if wos_api_key is None:
        load_dotenv()
        wos_api_key = os.getenv("WOS_API_KEY")
        if wos_api_key is None:
            raise ValueError("WOS_API_KEY environment variable not set")

    # Sleep for a bit to avoid hitting the API rate limit
    time.sleep(0.65)

    # Call the function with backoff for retries
    return _get_articles_from_grant_id_and_funder(
        grant_id=grant_id,
        funder=funder,
        wos_api_key=wos_api_key,
    )
