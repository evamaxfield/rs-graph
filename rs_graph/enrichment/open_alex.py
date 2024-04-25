#!/usr/bin/env python

from __future__ import annotations

import logging
import random
import time
import traceback
from datetime import date
from pathlib import Path

import backoff
import msgspec
import pyalex
from prefect import task

from .. import types
from ..db import models as db_models

#######################################################################################

log = logging.getLogger(__name__)

#######################################################################################

DEFAULT_ALEX_EMAIL = "evamxb@uw.edu"


class OpenAlexAPICallStatus(msgspec.Struct):
    call_count: int
    current_date: date


API_CALL_STATUS_FILEPATH = (
    Path("~/.rs-graph/open_alex_api_call_status.msgpack").expanduser().resolve()
)

MSGSPEC_ENCODER = msgspec.msgpack.Encoder()
MSGSPEC_DECODER = msgspec.msgpack.Decoder(type=OpenAlexAPICallStatus)

#######################################################################################
# Setup API


@backoff.on_exception(backoff.expo, Exception, max_time=5)
def _write_api_call_status(current_status: OpenAlexAPICallStatus) -> None:
    """Write the API call status to disk."""
    # Make dirs if needed
    API_CALL_STATUS_FILEPATH.parent.mkdir(parents=True, exist_ok=True)

    with open(API_CALL_STATUS_FILEPATH, "wb") as f:
        f.write(MSGSPEC_ENCODER.encode(current_status))


@backoff.on_exception(backoff.expo, Exception, max_time=5)
def _read_api_call_status() -> OpenAlexAPICallStatus:
    """Read the API call status from disk."""
    with open(API_CALL_STATUS_FILEPATH, "rb") as f:
        current_status = MSGSPEC_DECODER.decode(f.read())

    # Check if current API status is out of date
    if current_status.current_date != date.today():
        current_status = OpenAlexAPICallStatus(
            call_count=0,
            current_date=date.today(),
        )
        _write_api_call_status(current_status=current_status)

    return current_status


def _create_api_call_status_file() -> None:
    """Reset the API call status."""
    # Check and setup call status
    if not API_CALL_STATUS_FILEPATH.exists():
        _write_api_call_status(
            OpenAlexAPICallStatus(
                call_count=0,
                current_date=date.today(),
            )
        )


def _setup() -> None:
    """Set up a pool of polite workers for OpenAlex."""
    # Add email for polite pool
    pyalex.config.email = DEFAULT_ALEX_EMAIL

    # Add retries
    pyalex.config.max_retries = 3
    pyalex.config.retry_backoff_factor = 0.5
    pyalex.config.retry_http_codes = [429, 500, 503]

    _create_api_call_status_file()


def _increment_call_count_and_check() -> None:
    """Increment the API call count and check if we need to sleep."""
    # Read latest
    current_status = _read_api_call_status()

    # Temporary log
    if current_status.call_count % 1000 == 0:
        print(f"OpenAlex Daily API call count: {current_status.call_count}")

    # If we've made 80,000 calls in a single day
    # pause all processing until tomorrow
    if current_status.call_count >= 80_000:
        print("Sleeping until tomorrow to avoid OpenAlex API limit.")
        while date.today() == current_status.current_date:
            time.sleep(600)

        # Reset the call count
        current_status = OpenAlexAPICallStatus(
            call_count=0,
            current_date=date.today(),
        )

    # Increment count
    current_status.call_count += 1
    _write_api_call_status(current_status)

    # Sleep for a random amount of time
    time.sleep(random.uniform(0, 2))


_setup()

# Create APIs
OPENALEX_WORKS = pyalex.Works()
OPENALEX_AUTHORS = pyalex.Authors()

#######################################################################################


def get_open_alex_work_from_doi(doi: str) -> pyalex.Work:
    """Get work from a DOI."""
    # Lowercase DOI
    doi = doi.lower()

    # Handle DOI to doi.org
    if "doi.org" not in doi:
        doi = f"https://doi.org/{doi}"

    _increment_call_count_and_check()
    return OPENALEX_WORKS[doi]


def convert_from_inverted_index_abstract(abstract: dict) -> str:
    # Inverted index looks like:
    # {
    #     "Despite": [0],
    #     "growing": [1],
    #     "interest": [2],
    #     "in": [3, 57, 73, 110, 122],
    #     "Open": [4, 201],
    #     "Access": [5],
    #     ...
    # }
    # Convert to:
    # "Despite growing interest in Open Access ..."
    abstract_as_list: list[str | None] = [None] * 10000
    for word, indices in abstract.items():
        for index in indices:
            abstract_as_list[index] = word

    # Remove all extra Nones
    abstract_as_list_of_str = [word for word in abstract_as_list if word is not None]
    return " ".join(abstract_as_list_of_str)


def get_open_alex_author_from_id(author_id: str) -> pyalex.Author:
    """Get author from an ID."""
    _increment_call_count_and_check()
    return OPENALEX_AUTHORS[author_id]


def process_open_alex_work(
    pair: types.ExpandedRepositoryDocumentPair,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    try:
        # Get the OpenAlex work
        open_alex_work = get_open_alex_work_from_doi(pair.paper_doi)

        # Convert inverted index abstract to string
        if open_alex_work["abstract_inverted_index"] is None:
            abstract_text = None
        else:
            abstract_text = convert_from_inverted_index_abstract(
                open_alex_work["abstract_inverted_index"]
            )

        # Create DatasetSource
        dataset_source = db_models.DatasetSource(name=pair.source)

        # Create the Document
        document = db_models.Document(
            doi=pair.paper_doi,
            open_alex_id=open_alex_work["id"],
            title=open_alex_work["title"],
            publication_date=date.fromisoformat(open_alex_work["publication_date"]),
            cited_by_count=open_alex_work["cited_by_count"],
            cited_by_percentile_year_min=open_alex_work["cited_by_percentile_year"][
                "min"
            ],
            cited_by_percentile_year_max=open_alex_work["cited_by_percentile_year"][
                "max"
            ],
        )

        # Create the abstract
        abstract_model = db_models.DocumentAbstract(
            document_id=document.id,
            content=abstract_text,
        )

        # For each Topic, create the Topic
        all_topic_details = []
        for topic_details in open_alex_work["topics"]:
            # Create the topic
            topic = db_models.Topic(
                open_alex_id=topic_details["id"],
                name=topic_details["display_name"],
                field_name=topic_details["field"]["display_name"],
                field_open_alex_id=topic_details["field"]["id"],
                subfield_name=topic_details["subfield"]["display_name"],
                subfield_open_alex_id=topic_details["subfield"]["id"],
                domain_name=topic_details["domain"]["display_name"],
                domain_open_alex_id=topic_details["domain"]["id"],
            )

            # Create the connection between topic and document
            document_topic = db_models.DocumentTopic(
                document_id=document.id,
                topic_id=topic.id,
                score=topic_details["score"],
            )

            # Add to list
            all_topic_details.append(
                types.TopicDetails(
                    topic_model=topic,
                    document_topic_model=document_topic,
                )
            )

        # For each author, create the Researcher
        all_researcher_details = []
        for author_details in open_alex_work["authorships"]:
            # Fetch extra author details
            open_alex_author = get_open_alex_author_from_id(
                author_details["author"]["id"]
            )

            # Create the Researcher
            researcher = db_models.Researcher(
                open_alex_id=open_alex_author["id"],
                name=open_alex_author["display_name"],
                works_count=open_alex_author["works_count"],
                cited_by_count=open_alex_author["cited_by_count"],
                h_index=open_alex_author["summary_stats"]["h_index"],
                i10_index=open_alex_author["summary_stats"]["i10_index"],
                two_year_mean_citedness=open_alex_author["summary_stats"][
                    "2yr_mean_citedness"
                ],
            )

            # Create the connection between researcher and document
            document_contributor = db_models.DocumentContributor(
                researcher_id=researcher.id,
                document_id=document.id,
                position=author_details["author_position"],
                is_corresponding=author_details["is_corresponding"],
            )

            # Create the Institutions
            institution_models = []

            # Create the Institution
            for institution_details in author_details["institutions"]:
                institution = db_models.Institution(
                    open_alex_id=institution_details["id"],
                    name=institution_details["display_name"],
                    ror=institution_details["ror"],
                )
                institution_models.append(institution)

            # Add to list
            all_researcher_details.append(
                types.ResearcherDetails(
                    researcher_model=researcher,
                    document_contributor_model=document_contributor,
                    institution_models=institution_models,
                )
            )

        # For each grant, create the
        # Funder, FundingInstance, and DocumentFundingInstance
        all_funding_instance_details = []
        for grant_details in open_alex_work["grants"]:
            # Create the Funder
            funder = db_models.Funder(
                open_alex_id=grant_details["funder"],
                name=grant_details["funder_display_name"],
            )

            # Create the FundingInstance
            # TODO: Handle None award_id
            # planned: add a DocumentFunder model that stores these types of connections
            if grant_details["award_id"] is None:
                continue
            funding_instance = db_models.FundingInstance(
                funder_id=funder.id,
                award_id=grant_details["award_id"],
            )

            # Add to list
            all_funding_instance_details.append(
                types.FundingInstanceDetails(
                    funder_model=funder,
                    funding_instance_model=funding_instance,
                )
            )

        # Attach everything back to the pair
        pair.open_alex_results = types.OpenAlexResultModels(
            source_model=dataset_source,
            document_model=document,
            document_abstract_model=abstract_model,
            topic_details=all_topic_details,
            researcher_details=all_researcher_details,
            funding_instance_details=all_funding_instance_details,
        )

        return pair

    except Exception as e:
        return types.ErrorResult(
            source=pair.source,
            step="open-alex-processing",
            identifier=pair.paper_doi,
            error=str(e),
            traceback=traceback.format_exc(),
        )


@task(timeout_seconds=120)
def process_open_alex_work_task(
    pair: types.ExpandedRepositoryDocumentPair | types.ErrorResult,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    # Pass through
    if isinstance(pair, types.ErrorResult):
        return pair

    return process_open_alex_work(pair=pair)
