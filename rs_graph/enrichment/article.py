#!/usr/bin/env python

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import backoff
import msgspec
import pyalex
import requests

from .. import types
from ..db import models as db_models

#######################################################################################

log = logging.getLogger(__name__)

#######################################################################################


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


def _setup_open_alex(open_alex_email: str) -> None:
    """Set up a pool of polite workers for OpenAlex."""
    # Add email for polite pool
    pyalex.config.email = open_alex_email

    # Add retries
    pyalex.config.max_retries = 3
    pyalex.config.retry_backoff_factor = 0.5
    pyalex.config.retry_http_codes = [429, 500, 503]

    _create_api_call_status_file()


def _increment_call_count_and_check(open_alex_email_count: int) -> None:
    """Increment the API call count and check if we need to sleep."""
    # Read latest
    current_status = _read_api_call_status()

    # Temporary log
    if current_status.call_count % 1000 == 0:
        print(f"OpenAlex Daily API call count: {current_status.call_count}")

    # If we've made 90,000 calls in a single day (per OpenAlex email)
    # pause all processing until tomorrow
    if current_status.call_count >= (90_000 * open_alex_email_count):
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

    # Sleep for 0.6 seconds to avoid rate limiting
    time.sleep(0.6)


#######################################################################################


def get_updated_doi_from_semantic_scholar(
    doi: str,
    semantic_scholar_api_key: str,
) -> str:
    try:
        # Handle searchable ID
        if "arxiv" in doi.lower():
            search_id = doi.lower().split("arxiv.")[-1]
            search_string = f"ARXIV:{search_id}"
        else:
            search_string = f"DOI:{doi}"

        # Build API request
        url = f"https://api.semanticscholar.org/graph/v1/paper/{search_string}"
        headers = {
            "x-api-key": semantic_scholar_api_key,
        }
        params = {"fields": "externalIds"}

        # Make request
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        # Parse response
        paper_details = response.json()

        # Return updated DOI if available
        if "externalIds" in paper_details and "DOI" in paper_details["externalIds"]:
            return paper_details["externalIds"]["DOI"]
        else:
            return doi

    except Exception:
        # Return original DOI on any error
        return doi


def get_open_alex_work_from_doi(
    open_alex_email: str,
    open_alex_email_count: int,
    doi: str,
) -> pyalex.Work:
    """Get work from a DOI."""
    # Lowercase DOI
    doi = doi.lower()

    # Handle DOI to doi.org
    if "doi.org" not in doi:
        doi = f"https://doi.org/{doi}"

    # Setup OpenAlex API
    _setup_open_alex(open_alex_email=open_alex_email)

    # Create works api
    open_alex_works = pyalex.Works()

    # Increment call count and then actually request
    _increment_call_count_and_check(open_alex_email_count=open_alex_email_count)
    return open_alex_works[doi]


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
    abstract_as_list: list[str | None] = [None] * 20000
    for word, indices in abstract.items():
        for index in indices:
            abstract_as_list[index] = word

    # Remove all extra Nones
    abstract_as_list_of_str = [word for word in abstract_as_list if word is not None]
    return " ".join(abstract_as_list_of_str)


def get_open_alex_author_from_id(
    open_alex_email: str,
    open_alex_email_count: int,
    author_id: str,
) -> pyalex.Author:
    """Get author from an ID."""
    # Create OpenAlex API
    _setup_open_alex(open_alex_email=open_alex_email)

    # Create authors api
    open_alex_authors = pyalex.Authors()

    # Increment call count and then actually request
    _increment_call_count_and_check(open_alex_email_count=open_alex_email_count)
    return open_alex_authors[author_id]


def process_article(  # noqa: C901
    paper_doi: str,
    source: str,
    open_alex_email: str,
    open_alex_email_count: int,
    semantic_scholar_api_key: str,
    fetch_author_details: bool = True,
    fetch_grant_details: bool = True,
    existing_pyalex_work: pyalex.Work | None = None,
    existing_open_alex_results: types.OpenAlexResultModels | None = None,
) -> types.OpenAlexResultModels | types.ErrorResult:
    try:
        if existing_open_alex_results is None:
            # Check for updated DOI
            updated_doi = get_updated_doi_from_semantic_scholar(
                doi=paper_doi,
                semantic_scholar_api_key=semantic_scholar_api_key,
            )

            # Handle "Alternate DOI" case
            alternate_dois: list[str] = []
            if updated_doi != paper_doi:
                # Store the original DOI as an alternate DOI
                # as we have resolved a more recent version
                alternate_dois.append(
                    paper_doi,
                )

                # Get the OpenAlex work
                open_alex_work = get_open_alex_work_from_doi(
                    open_alex_email=open_alex_email,
                    open_alex_email_count=open_alex_email_count,
                    doi=updated_doi,
                )

            else:
                if existing_pyalex_work is not None:
                    open_alex_work = existing_pyalex_work
                else:
                    open_alex_work = get_open_alex_work_from_doi(
                        open_alex_email=open_alex_email,
                        open_alex_email_count=open_alex_email_count,
                        doi=paper_doi,
                    )

            # Convert inverted index abstract to string
            if open_alex_work["abstract_inverted_index"] is None:
                abstract_text = None
            else:
                abstract_text = convert_from_inverted_index_abstract(
                    open_alex_work["abstract_inverted_index"]
                )

            # Create Primary Document Source and Primary Location
            if open_alex_work["primary_location"] is not None:
                if open_alex_work["primary_location"]["source"] is not None:
                    primary_document_source = db_models.Source(
                        name=open_alex_work["primary_location"]["source"]["display_name"],
                        open_alex_id=open_alex_work["primary_location"]["source"]["id"],
                        source_type=open_alex_work["primary_location"]["source"]["type"],
                        host_organization_name=open_alex_work["primary_location"]["source"][
                            "host_organization_name"
                        ],
                        host_organization_open_alex_id=open_alex_work["primary_location"][
                            "source"
                        ]["host_organization"],
                    )

                    primary_location = db_models.Location(
                        is_open_access=open_alex_work["primary_location"]["is_oa"],
                        landing_page_url=open_alex_work["primary_location"]["landing_page_url"],
                        pdf_url=open_alex_work["primary_location"]["pdf_url"],
                        license=open_alex_work["primary_location"]["license"],
                        version=open_alex_work["primary_location"]["version"],
                        source_id=primary_document_source.id,
                    )
                else:
                    primary_document_source = None
                    primary_location = db_models.Location(
                        is_open_access=open_alex_work["primary_location"]["is_oa"],
                        landing_page_url=open_alex_work["primary_location"]["landing_page_url"],
                        pdf_url=open_alex_work["primary_location"]["pdf_url"],
                        license=open_alex_work["primary_location"]["license"],
                        version=open_alex_work["primary_location"]["version"],
                        source_id=None,
                    )

            else:
                # If no primary location, set to None
                primary_document_source = None
                primary_location = None

            # Create Best OA Document Source and Best OA Location
            if open_alex_work["best_oa_location"] is not None:
                if open_alex_work["best_oa_location"]["source"] is not None:
                    best_oa_document_source = db_models.Source(
                        name=open_alex_work["best_oa_location"]["source"]["display_name"],
                        open_alex_id=open_alex_work["best_oa_location"]["source"]["id"],
                        source_type=open_alex_work["best_oa_location"]["source"]["type"],
                        host_organization_name=open_alex_work["best_oa_location"]["source"][
                            "host_organization_name"
                        ],
                        host_organization_open_alex_id=open_alex_work["best_oa_location"][
                            "source"
                        ]["host_organization"],
                    )
                    best_oa_location = db_models.Location(
                        is_open_access=open_alex_work["best_oa_location"]["is_oa"],
                        landing_page_url=open_alex_work["best_oa_location"]["landing_page_url"],
                        pdf_url=open_alex_work["best_oa_location"]["pdf_url"],
                        license=open_alex_work["best_oa_location"]["license"],
                        version=open_alex_work["best_oa_location"]["version"],
                        source_id=best_oa_document_source.id,
                    )

                else:
                    best_oa_document_source = None
                    best_oa_location = db_models.Location(
                        is_open_access=open_alex_work["best_oa_location"]["is_oa"],
                        landing_page_url=open_alex_work["best_oa_location"]["landing_page_url"],
                        pdf_url=open_alex_work["best_oa_location"]["pdf_url"],
                        license=open_alex_work["best_oa_location"]["license"],
                        version=open_alex_work["best_oa_location"]["version"],
                        source_id=None,
                    )
            else:
                # If no best OA location, set to None
                best_oa_document_source = None
                best_oa_location = None

            # Create DatasetSource
            dataset_source = db_models.DatasetSource(name=source)

            # Check for citation_normalized_percentile
            if open_alex_work["citation_normalized_percentile"] is not None:
                citation_normalized_percentile = open_alex_work[
                    "citation_normalized_percentile"
                ]["value"]
            else:
                citation_normalized_percentile = None

            # Create the Document
            document = db_models.Document(
                doi=updated_doi,
                open_alex_id=open_alex_work["id"],
                title=open_alex_work["title"],
                publication_date=date.fromisoformat(open_alex_work["publication_date"]),
                cited_by_count=open_alex_work["cited_by_count"],
                fwci=open_alex_work["fwci"],
                citation_normalized_percentile=citation_normalized_percentile,
                document_type=open_alex_work["type"],
                is_open_access=open_alex_work["open_access"]["is_oa"],
                open_access_status=open_alex_work["open_access"]["oa_status"],
                primary_location_id=primary_location.id if primary_location else None,
                best_oa_location_id=best_oa_location.id if best_oa_location else None,
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

        else:
            # Always shortcut the document
            document = existing_open_alex_results.document_model

            if existing_pyalex_work is not None:
                open_alex_work = existing_pyalex_work
            else:
                # Not provided but needed, get a new pyalex work
                # TODO: This could be optimized to avoid double fetching
                # Specifically for the snowball sampling discovery pipeline
                open_alex_work = get_open_alex_work_from_doi(
                    open_alex_email=open_alex_email,
                    open_alex_email_count=open_alex_email_count,
                    doi=paper_doi,
                )

        # For each author, create the Researcher
        if fetch_author_details:
            all_researcher_details = []
            for author_details in open_alex_work["authorships"]:
                # Fetch extra author details
                open_alex_author = get_open_alex_author_from_id(
                    open_alex_email=open_alex_email,
                    open_alex_email_count=open_alex_email_count,
                    author_id=author_details["author"]["id"],
                )

                # Create the Researcher
                researcher = db_models.Researcher(
                    open_alex_id=open_alex_author["id"],
                    orcid=open_alex_author["orcid"],
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
                        country_code=institution_details["country_code"],
                        institution_type=institution_details["type"],
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
        else:
            all_researcher_details = None

        # For each grant, create the
        # Funder, FundingInstance, and DocumentFundingInstance
        if fetch_grant_details:
            all_funding_instance_details = []
            for grant_details in open_alex_work["awards"]:
                # Create the Funder
                funder = db_models.Funder(
                    open_alex_id=grant_details["funder_id"],
                    name=grant_details["funder_display_name"],
                )

                # Create the FundingInstance
                # TODO: Handle None funder_award_id
                # planned: add a DocumentFunder model that stores these types of connections
                if grant_details["funder_award_id"] is None:
                    continue

                funding_instance = db_models.FundingInstance(
                    funder_id=funder.id,
                    award_id=grant_details["funder_award_id"],
                )

                # Add to list
                all_funding_instance_details.append(
                    types.FundingInstanceDetails(
                        funder_model=funder,
                        funding_instance_model=funding_instance,
                    )
                )
        else:
            all_funding_instance_details = None

        # Combine existing and new
        if existing_open_alex_results is None:
            return types.OpenAlexResultModels(
                dataset_source_model=dataset_source,
                primary_document_source_model=primary_document_source,
                primary_location_model=primary_location,
                best_oa_document_source_model=best_oa_document_source,
                best_oa_location_model=best_oa_location,
                document_model=document,
                document_abstract_model=abstract_model,
                document_alternate_dois=alternate_dois,
                topic_details=all_topic_details,
                researcher_details=all_researcher_details,
                funding_instance_details=all_funding_instance_details,
            )

        return types.OpenAlexResultModels(
            # Take existing
            dataset_source_model=existing_open_alex_results.dataset_source_model,
            primary_document_source_model=existing_open_alex_results.primary_document_source_model,
            primary_location_model=existing_open_alex_results.primary_location_model,
            best_oa_document_source_model=existing_open_alex_results.best_oa_document_source_model,
            best_oa_location_model=existing_open_alex_results.best_oa_location_model,
            document_model=existing_open_alex_results.document_model,
            document_abstract_model=existing_open_alex_results.document_abstract_model,
            document_alternate_dois=existing_open_alex_results.document_alternate_dois,
            topic_details=existing_open_alex_results.topic_details,
            # Add possible new researcher and funding details
            researcher_details=all_researcher_details,
            funding_instance_details=all_funding_instance_details,
        )

    except Exception as e:
        return types.ErrorResult(
            source=source,
            step="open-alex-processing",
            identifier=paper_doi,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def process_article_task(
    pair: types.ExpandedRepositoryDocumentPair | types.ErrorResult,
    open_alex_email: str,
    open_alex_email_count: int,
    semantic_scholar_api_key: str,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    # Pass through
    if isinstance(pair, types.ErrorResult):
        return pair

    # Get start time
    start_time = time.perf_counter()

    # Process article
    open_alex_results = process_article(
        paper_doi=pair.paper_doi,
        source=pair.source,
        open_alex_email=open_alex_email,
        open_alex_email_count=open_alex_email_count,
        semantic_scholar_api_key=semantic_scholar_api_key,
    )

    # Get end time
    end_time = time.perf_counter()

    if isinstance(open_alex_results, types.ErrorResult):
        return open_alex_results

    # Attach processing time if successful
    pair.open_alex_results = open_alex_results
    pair.open_alex_processing_time_seconds = end_time - start_time

    return pair


@dataclass
class WorkAndOAResultModels:
    pyalex_work: pyalex.Work
    open_alex_results: types.OpenAlexResultModels


def get_articles_for_researcher(
    researcher_open_alex_id: str,
    open_alex_email: str,
    open_alex_email_count: int,
    semantic_scholar_api_key: str,
) -> list[WorkAndOAResultModels | types.ErrorResult] | types.ErrorResult:
    """Get articles for a researcher."""
    try:
        # Setup OpenAlex API
        _setup_open_alex(open_alex_email=open_alex_email)

        # Strip the open alex url from the id if needed
        if "https://openalex.org/" in researcher_open_alex_id:
            researcher_open_alex_id = researcher_open_alex_id.split("https://openalex.org/")[-1]

        # Get works api
        open_alex_works = pyalex.Works()

        # Increment call count and then actually request
        author_works = []
        pager = open_alex_works.filter(author={"id": researcher_open_alex_id}).paginate(
            per_page=200
        )
        for page in pager:
            _increment_call_count_and_check(open_alex_email_count=open_alex_email_count)
            author_works.extend(page)

        # Convert author works to list of OpenAlexResultModels
        all_results: list[WorkAndOAResultModels | types.ErrorResult] = []
        for work in author_works:
            process_result = process_article(
                paper_doi=work["doi"],
                source="snowball-sampling-discovery",
                open_alex_email=open_alex_email,
                open_alex_email_count=open_alex_email_count,
                semantic_scholar_api_key=semantic_scholar_api_key,
                fetch_author_details=False,
                fetch_grant_details=False,
                existing_pyalex_work=work,
            )

            if isinstance(process_result, types.ErrorResult):
                all_results.append(process_result)
            else:
                all_results.append(
                    WorkAndOAResultModels(
                        pyalex_work=work,
                        open_alex_results=process_result,
                    )
                )

        # Deduplicate based on DOI
        # This can happen if open alex has multiple versions of the same paper
        # listed separately and SemanticScholar correctly points to the same DOI
        known_dois = []
        deduplicated_results = []
        for result in all_results:
            if isinstance(result, WorkAndOAResultModels):
                if result.open_alex_results.document_model.doi not in known_dois:
                    known_dois.append(result.open_alex_results.document_model.doi)
                    deduplicated_results.append(result)
            else:
                deduplicated_results.append(result)

        return deduplicated_results

    except Exception as e:
        return types.ErrorResult(
            source="snowball-sampling-discovery",
            step="get-articles-for-researcher",
            identifier=researcher_open_alex_id,
            error=str(e),
            traceback=traceback.format_exc(),
        )
