#!/usr/bin/env python

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pyalex
import requests
from cachetools import LRUCache, cached
from cachetools.keys import hashkey

from .. import types
from ..db import models as db_models
from ..utils.identifier_normalization import normalize_doi

#######################################################################################

log = logging.getLogger(__name__)


#######################################################################################
# Safe dictionary access helpers


def _safe_get(data: dict | None, *keys: str, default: Any = None) -> Any:
    """Safely access nested dictionary keys, returning default if any key is missing."""
    if data is None:
        return default
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


class MissingRequiredFieldError(ValueError):
    """Raised when a required field is missing from API response."""

    pass


def _require_field(data: dict, field: str, context: str) -> Any:
    """Get a required field from a dictionary, raising a descriptive error if missing."""
    value = data.get(field)
    if value is None:
        raise MissingRequiredFieldError(
            f"Required field '{field}' is missing or None in {context}"
        )
    return value


#######################################################################################
# Setup API


OPEN_ALEX_API_CALL_COUNT = 0


def _setup_open_alex(open_alex_token: str) -> None:
    """Set up the OpenAlex API."""
    # Add token for polite pool
    pyalex.config.api_key = open_alex_token

    # Add retries
    pyalex.config.max_retries = 3
    pyalex.config.retry_backoff_factor = 0.5
    pyalex.config.retry_http_codes = [429, 500, 503]


def _increment_call_count_and_check() -> None:
    """Increment the API call count and check if we need to sleep."""
    global OPEN_ALEX_API_CALL_COUNT

    # Temporary log
    if OPEN_ALEX_API_CALL_COUNT >= 1000:
        # Check rate limit API
        try:
            response = requests.get(
                f"https://api.openalex.org/rate-limit?api_key={pyalex.config.api_key}",
            )
            response.raise_for_status()

            # Parse and check remaining
            rate_limit_data = response.json()

            # Response data looks like this:
            # {'api_key': '...',
            # 'is_grandfathered': False,
            # 'rate_limit': {'credit_costs': {'content': 100,
            #                                 'list': 1,
            #                                 'search': 10,
            #                                 'semantic': 100,
            #                                 'singleton': 0,
            #                                 'text': 100},
            #                 'credits_limit': 10000,
            #                 'credits_remaining': 363,
            #                 'credits_used': 9637,
            #                 'daily_budget_usd': 1,
            #                 'daily_remaining_usd': 0.0363,
            #                 'daily_used_usd': 0.9637,
            #                 'endpoint_costs_usd': {'content': 0.01,
            #                                     'list': 0.0001,
            #                                     'search': 0.001,
            #                                     'semantic': 0.01,
            #                                     'singleton': 0,
            #                                     'text': 0.01},
            #                 'onetime_credits_balance': 1000000,
            #                 'onetime_credits_expires_at': 'Wed May 20 2026 00:24:52 '
            #                                             'GMT+0000 (Coordinated Universal '
            #                                             'Time)',
            #                 'onetime_credits_remaining': 1000000,
            #                 'prepaid_balance_usd': 100,
            #                 'prepaid_expires_at': 'Wed May 20 2026 00:24:52 GMT+0000 '
            #                                     '(Coordinated Universal Time)',
            #                 'prepaid_remaining_usd': 100,
            #                 'resets_at': '2026-02-20T00:00:00.000Z',
            #                 'resets_in_seconds': 75513}}
            if (
                "rate_limit" in rate_limit_data
                and "credits_remaining" in rate_limit_data["rate_limit"]
            ) or "onetime_credits_remaining" in rate_limit_data["rate_limit"]:
                total_credits_remaining = (
                    rate_limit_data["rate_limit"]["credits_remaining"]
                    + rate_limit_data["rate_limit"]["onetime_credits_remaining"]
                )
                print(f"OpenAlex API credits remaining: {total_credits_remaining}")

                if total_credits_remaining <= 500:
                    reset_datetime = datetime.fromisoformat(
                        rate_limit_data["rate_limit"]["resets_at"]
                    )
                    reset_timedelta = reset_datetime - datetime.now(tz=reset_datetime.tzinfo)
                    log.warning(
                        f"Sleeping until OpenAlex API reset at {reset_datetime} "
                        f"(about {reset_timedelta} from now) "
                        f"to avoid hitting rate limit."
                    )

                    # Sleep until reset
                    while datetime.now(tz=reset_datetime.tzinfo) < reset_datetime:
                        time.sleep(60)

                # Reset the call count after every successful check
                OPEN_ALEX_API_CALL_COUNT = 0

        # Else just log and continue with local sleeping as backup
        except Exception as e:
            raise RuntimeError(f"Error checking OpenAlex API rate limit: {e}") from e

    # Increment count
    OPEN_ALEX_API_CALL_COUNT += 1

    # Sleep for 0.05 seconds to stay within rate limit of 100 requests per second
    time.sleep(0.05)


#######################################################################################


@cached(  # type: ignore[misc]
    cache=LRUCache(maxsize=2**12),
    key=lambda doi, semantic_scholar_api_key: hashkey(normalize_doi(doi)),
)
def get_updated_doi_from_semantic_scholar(
    doi: str,
    semantic_scholar_api_key: str,
) -> str:
    try:
        # Normalize DOI (strip https://doi.org/ prefix if present)
        doi = normalize_doi(doi)

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
        if response.status_code == 404:
            return doi

        response.raise_for_status()

        # Parse response
        paper_details = response.json()

        # Return updated DOI if available
        if "externalIds" in paper_details and "DOI" in paper_details["externalIds"]:
            return paper_details["externalIds"]["DOI"]
        else:
            return doi

    except Exception as e:
        raise RuntimeError(
            f"Error fetching updated DOI from Semantic Scholar for DOI '{doi}': {e}. "
            f"Be sure to check that your Semantic Scholar API key is valid."
        ) from e


@cached(  # type: ignore[misc]
    cache=LRUCache(maxsize=2**12),
    key=lambda open_alex_token, doi: hashkey(doi.lower()),
)
def get_open_alex_work_from_doi(
    open_alex_token: str,
    doi: str,
) -> pyalex.Work:
    """Get work from a DOI."""
    # Lowercase DOI
    doi = doi.lower()

    # Handle DOI to doi.org
    if "doi.org" not in doi:
        doi = f"https://doi.org/{doi}"

    # Setup OpenAlex API
    _setup_open_alex(open_alex_token=open_alex_token)

    # Create works api
    open_alex_works = pyalex.Works()

    # Increment call count and then actually request
    _increment_call_count_and_check()
    return open_alex_works[doi]  # type: ignore[return-value]


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


@cached(  # type: ignore[misc]
    cache=LRUCache(maxsize=2**12),
    key=lambda open_alex_token, author_id: hashkey(author_id),
)
def get_open_alex_author_from_id(
    open_alex_token: str,
    author_id: str,
) -> pyalex.Author:
    """Get author from an ID."""
    # Create OpenAlex API
    _setup_open_alex(open_alex_token=open_alex_token)

    # Create authors api
    open_alex_authors = pyalex.Authors()

    # Increment call count and then actually request
    _increment_call_count_and_check()
    return open_alex_authors[author_id]  # type: ignore[return-value]


def process_article(  # noqa: C901
    paper_doi: str,
    source: str,
    open_alex_token: str,
    semantic_scholar_api_key: str,
    fetch_author_details: bool = True,
    fetch_grant_details: bool = True,
    existing_pyalex_work: pyalex.Work | None = None,
    existing_open_alex_results: types.OpenAlexResultModels | None = None,
    skip_semantic_scholar: bool = False,
) -> types.OpenAlexResultModels | types.ErrorResult:
    try:
        if existing_open_alex_results is None:
            if skip_semantic_scholar:
                # Use DOI as-is from the caller (e.g. OpenAlex)
                # Semantic Scholar DOI resolution will happen later
                # for matched articles only
                updated_doi = paper_doi
                alternate_dois: list[str] = []
            else:
                # Check for updated DOI
                updated_doi = get_updated_doi_from_semantic_scholar(
                    doi=paper_doi,
                    semantic_scholar_api_key=semantic_scholar_api_key,
                )

                # Handle "Alternate DOI" case
                alternate_dois = []
                if updated_doi != paper_doi:
                    # Store the original DOI as an alternate DOI
                    # as we have resolved a more recent version
                    alternate_dois.append(
                        paper_doi,
                    )

            if not skip_semantic_scholar and updated_doi != paper_doi:
                # Get the OpenAlex work for the resolved DOI
                open_alex_work = get_open_alex_work_from_doi(
                    open_alex_token=open_alex_token,
                    doi=updated_doi,
                )

            else:
                if existing_pyalex_work is not None:
                    open_alex_work = existing_pyalex_work
                else:
                    open_alex_work = get_open_alex_work_from_doi(
                        open_alex_token=open_alex_token,
                        doi=paper_doi,
                    )

            # Cast to dict for type checker (pyalex.Work is an untyped dict subclass)
            work: dict[str, Any] = open_alex_work  # type: ignore[assignment]

            # Convert inverted index abstract to string
            if work["abstract_inverted_index"] is None:
                abstract_text = None
            else:
                abstract_text = convert_from_inverted_index_abstract(
                    work["abstract_inverted_index"]
                )

            # Create Primary Document Source and Primary Location
            if work["primary_location"] is not None:
                if work["primary_location"]["source"] is not None:
                    primary_document_source = db_models.Source(
                        name=work["primary_location"]["source"]["display_name"],
                        open_alex_id=work["primary_location"]["source"]["id"],
                        source_type=work["primary_location"]["source"]["type"],
                        host_organization_name=work["primary_location"]["source"][
                            "host_organization_name"
                        ],
                        host_organization_open_alex_id=work["primary_location"]["source"][
                            "host_organization"
                        ],
                    )

                    primary_location = db_models.Location(
                        is_open_access=work["primary_location"]["is_oa"],
                        landing_page_url=work["primary_location"]["landing_page_url"],
                        pdf_url=work["primary_location"]["pdf_url"],
                        license=work["primary_location"]["license"],
                        version=work["primary_location"]["version"],
                        source_id=primary_document_source.id,
                    )
                else:
                    primary_document_source = None
                    primary_location = db_models.Location(
                        is_open_access=work["primary_location"]["is_oa"],
                        landing_page_url=work["primary_location"]["landing_page_url"],
                        pdf_url=work["primary_location"]["pdf_url"],
                        license=work["primary_location"]["license"],
                        version=work["primary_location"]["version"],
                        source_id=None,
                    )

            else:
                # If no primary location, set to None
                primary_document_source = None
                primary_location = None

            # Create Best OA Document Source and Best OA Location
            if work["best_oa_location"] is not None:
                if work["best_oa_location"]["source"] is not None:
                    best_oa_document_source = db_models.Source(
                        name=work["best_oa_location"]["source"]["display_name"],
                        open_alex_id=work["best_oa_location"]["source"]["id"],
                        source_type=work["best_oa_location"]["source"]["type"],
                        host_organization_name=work["best_oa_location"]["source"][
                            "host_organization_name"
                        ],
                        host_organization_open_alex_id=work["best_oa_location"]["source"][
                            "host_organization"
                        ],
                    )
                    best_oa_location = db_models.Location(
                        is_open_access=work["best_oa_location"]["is_oa"],
                        landing_page_url=work["best_oa_location"]["landing_page_url"],
                        pdf_url=work["best_oa_location"]["pdf_url"],
                        license=work["best_oa_location"]["license"],
                        version=work["best_oa_location"]["version"],
                        source_id=best_oa_document_source.id,
                    )

                else:
                    best_oa_document_source = None
                    best_oa_location = db_models.Location(
                        is_open_access=work["best_oa_location"]["is_oa"],
                        landing_page_url=work["best_oa_location"]["landing_page_url"],
                        pdf_url=work["best_oa_location"]["pdf_url"],
                        license=work["best_oa_location"]["license"],
                        version=work["best_oa_location"]["version"],
                        source_id=None,
                    )
            else:
                # If no best OA location, set to None
                best_oa_document_source = None
                best_oa_location = None

            # Create DatasetSource
            dataset_source = db_models.DatasetSource(name=source)

            # Check for citation_normalized_percentile
            citation_normalized_percentile = _safe_get(
                work, "citation_normalized_percentile", "value"
            )

            # Validate required fields for Document
            open_alex_id = _require_field(work, "id", "OpenAlex work response")
            title = _require_field(work, "title", "OpenAlex work response")
            publication_date_str = _require_field(
                work, "publication_date", "OpenAlex work response"
            )
            try:
                publication_date_parsed = date.fromisoformat(publication_date_str)
            except ValueError as e:
                raise MissingRequiredFieldError(
                    f"Invalid publication_date format '{publication_date_str}': {e}"
                ) from e

            # Create the Document
            document = db_models.Document(
                doi=updated_doi,
                open_alex_id=open_alex_id,
                title=title,
                publication_date=publication_date_parsed,
                cited_by_count=work.get("cited_by_count", 0),
                fwci=work.get("fwci"),
                citation_normalized_percentile=citation_normalized_percentile,
                document_type=work.get("type", "unknown"),
                is_open_access=_safe_get(work, "open_access", "is_oa", default=False),
                open_access_status=_safe_get(
                    work, "open_access", "oa_status", default="unknown"
                ),
                primary_location_id=primary_location.id if primary_location else None,
                best_open_access_location_id=best_oa_location.id if best_oa_location else None,
            )

            # Create the abstract
            abstract_model = db_models.DocumentAbstract(
                document_id=document.id,
                content=abstract_text,
            )

            # For each Topic, create the Topic
            all_topic_details = []
            for topic_details in work["topics"]:
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
                    open_alex_token=open_alex_token,
                    doi=paper_doi,
                )

            # Cast to dict for type checker (pyalex.Work is an untyped dict subclass)
            work = open_alex_work  # type: ignore[assignment]

        # For each author, create the Researcher
        if fetch_author_details:
            all_researcher_details = []
            for author_details in work["authorships"]:
                # Fetch extra author details
                open_alex_author = get_open_alex_author_from_id(
                    open_alex_token=open_alex_token,
                    author_id=author_details["author"]["id"],
                )
                # Cast to dict for type checker
                author: dict[str, Any] = open_alex_author  # type: ignore[assignment]

                # Create the Researcher
                researcher = db_models.Researcher(
                    open_alex_id=author["id"],
                    orcid=author["orcid"],
                    name=author["display_name"],
                    works_count=author["works_count"],
                    cited_by_count=author["cited_by_count"],
                    h_index=author["summary_stats"]["h_index"],
                    i10_index=author["summary_stats"]["i10_index"],
                    two_year_mean_citedness=author["summary_stats"]["2yr_mean_citedness"],
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
            for grant_details in work["awards"]:
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
    open_alex_token: str,
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
        open_alex_token=open_alex_token,
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
    open_alex_token: str,
    semantic_scholar_api_key: str,
) -> list[WorkAndOAResultModels | types.ErrorResult] | types.ErrorResult:
    """Get articles for a researcher."""
    try:
        # Setup OpenAlex API
        _setup_open_alex(open_alex_token=open_alex_token)

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
            _increment_call_count_and_check()
            author_works.extend(page)

        # Convert author works to list of OpenAlexResultModels
        all_results: list[WorkAndOAResultModels | types.ErrorResult] = []
        for work in author_works:
            # Skip articles with no DOI early -- they get filtered out
            # downstream in _flatten_and_check_articles_in_db anyway
            if work["doi"] is None:
                continue

            process_result = process_article(
                paper_doi=work["doi"],
                source="snowball-sampling-discovery",
                open_alex_token=open_alex_token,
                semantic_scholar_api_key=semantic_scholar_api_key,
                fetch_author_details=False,
                fetch_grant_details=False,
                existing_pyalex_work=work,
                skip_semantic_scholar=True,
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
        known_dois: list[str] = []
        deduplicated_results: list[WorkAndOAResultModels | types.ErrorResult] = []
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
