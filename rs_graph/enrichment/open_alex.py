#!/usr/bin/env python

from __future__ import annotations

import time
from datetime import date

import pyalex
import requests
from sqlmodel import Session
from tqdm import tqdm
from sqlalchemy import exc as db_exceptions

from ..db import models
from ..db import utils as db_utils
from ..sources.proto import RepositoryDocumentPair

#######################################################################################

DEFAULT_ALEX_EMAIL = "evamxb@uw.edu"
API_CALL_COUNT: int = 0
API_CURRENT_DATE: date = date.today()

#######################################################################################


def _setup() -> None:
    """Set up a pool of polite workers for OpenAlex."""
    # Add email for polite pool
    pyalex.config.email = DEFAULT_ALEX_EMAIL

    # Add retries
    pyalex.config.max_retries = 1
    pyalex.config.retry_backoff_factor = 0.1
    pyalex.config.retry_http_codes = [429, 500, 503]


class OpenAlexAPILimitError(Exception):
    """Error for when we hit the OpenAlex API limit."""

    pass


def _increment_call_count_and_check() -> None:
    """Increment the API call count and check if we need to sleep."""
    global API_CALL_COUNT, API_CURRENT_DATE

    # If date has changed, reset count
    if API_CURRENT_DATE != date.today():
        API_CURRENT_DATE = date.today()
        API_CALL_COUNT = 0

    # Increment count
    API_CALL_COUNT += 1

    # Sleep to also help this
    time.sleep(1)

    # If we've made 90,000 calls in a single day raise an error
    if API_CALL_COUNT >= 90_000:
        raise OpenAlexAPILimitError()


def get_open_alex_work_from_doi(doi: str) -> pyalex.Work:
    """Get work from a DOI."""
    _setup()

    # Lowercase DOI
    doi = doi.lower()

    # Handle DOI to doi.org
    if "doi.org" not in doi:
        doi = f"https://doi.org/{doi}"

    _increment_call_count_and_check()
    return pyalex.Works()[doi]


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
    abstract_as_list: list[str] = [None] * 5000
    for word, indices in abstract.items():
        for index in indices:
            abstract_as_list[index] = word

    # Remove all extra Nones
    abstract_as_list = [word for word in abstract_as_list if word is not None]
    return " ".join(abstract_as_list)


def get_open_alex_author_from_id(author_id: str) -> pyalex.Author:
    """Get author from an ID."""
    _setup()

    _increment_call_count_and_check()
    return pyalex.Authors()[author_id]


def process_pair(
    doi: str,
    dataset_source_name: str,
    session: Session,
) -> None:
    """Process a DOI."""
    # Get the OpenAlex work
    open_alex_work = get_open_alex_work_from_doi(doi)

    # Convert inverted index abstract to string
    if open_alex_work["abstract_inverted_index"] is None:
        abstract_text = None
    else:
        abstract_text = convert_from_inverted_index_abstract(
            open_alex_work["abstract_inverted_index"]
        )

    # Create DatasetSource
    dataset_source = models.DatasetSource(name=dataset_source_name)
    dataset_source = db_utils.get_or_add(dataset_source, session)

    # Create the Document
    document = models.Document(
        doi=doi,
        open_alex_id=open_alex_work["id"],
        title=open_alex_work["title"],
        publication_date=date.fromisoformat(open_alex_work["publication_date"]),
        cited_by_count=open_alex_work["cited_by_count"],
        cited_by_percentile_year_min=open_alex_work["cited_by_percentile_year"]["min"],
        cited_by_percentile_year_max=open_alex_work["cited_by_percentile_year"]["max"],
        abstract=abstract_text,
        dataset_source_id=dataset_source.id,
    )
    document = db_utils.get_or_add(document, session)

    # For each Topic, create the Topic
    for topic_details in open_alex_work["topics"]:
        # Create the topic
        topic = models.Topic(
            open_alex_id=topic_details["id"],
            name=topic_details["display_name"],
            field_name=topic_details["field"]["display_name"],
            field_open_alex_id=topic_details["field"]["id"],
            subfield_name=topic_details["subfield"]["display_name"],
            subfield_open_alex_id=topic_details["subfield"]["id"],
            domain_name=topic_details["domain"]["display_name"],
            domain_open_alex_id=topic_details["domain"]["id"],
        )
        topic = db_utils.get_or_add(topic, session)

        # Create the connection between topic and document
        document_topic = models.DocumentTopic(
            document_id=document.id,
            topic_id=topic.id,
            score=topic_details["score"],
        )
        document_topic = db_utils.get_or_add(document_topic, session)

    # For each author, create the Researcher
    for author_details in open_alex_work["authorships"]:
        # Fetch extra author details
        open_alex_author = get_open_alex_author_from_id(author_details["author"]["id"])

        # Create the Researcher
        researcher = models.Researcher(
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
        researcher = db_utils.get_or_add(researcher, session)

        # Create the connection between researcher and document
        researcher_document = models.ResearcherDocument(
            researcher_id=researcher.id,
            document_id=document.id,
            position=author_details["author_position"],
            is_corresponding=author_details["is_corresponding"],
        )
        researcher_document = db_utils.get_or_add(researcher_document, session)

        # Create the Institution
        for institution_details in author_details["institutions"]:
            institution = models.Institution(
                open_alex_id=institution_details["id"],
                name=institution_details["display_name"],
                ror=institution_details["ror"],
            )
            institution = db_utils.get_or_add(institution, session)

            # Create the connection between institution, researcher, and the document
            researcher_document_institution = models.ResearcherDocumentInstitution(
                researcher_document_id=researcher_document.id,
                institution_id=institution.id,
            )
            db_utils.get_or_add(researcher_document_institution, session)

    # For each grant, create the Funder, FundingInstance, and DocumentFundingInstance
    for grant_details in open_alex_work["grants"]:
        # Create the Funder
        funder = models.Funder(
            open_alex_id=grant_details["funder"],
            name=grant_details["funder_display_name"],
        )
        funder = db_utils.get_or_add(funder, session)

        # Create the FundingInstance
        funding_instance = models.FundingInstance(
            funder_id=funder.id,
            award_id=grant_details["award_id"],
        )
        funding_instance = db_utils.get_or_add(funding_instance, session)

        # Create the connection between document and funding instance
        document_funding_instance = models.DocumentFundingInstance(
            document_id=document.id,
            funding_instance_id=funding_instance.id,
        )
        db_utils.get_or_add(document_funding_instance, session)


def process_pairs(
    pairs: list[RepositoryDocumentPair],
    prod: bool = False,
) -> None:
    """Process a list of DOIs."""
    # Create db engine
    engine = db_utils.get_engine(prod=prod)

    # Init session
    for pair in tqdm(
        pairs,
        desc="Processing DOIs",
    ):
        with Session(engine) as session:
            try:
                process_pair(
                    doi=pair.paper_doi,
                    dataset_source_name=pair.source,
                    session=session,
                )
            except requests.exceptions.HTTPError as e:
                if "404 Client Error" in str(e):
                    print(f"DOI {pair.paper_doi} not found in OpenAlex")
            except db_exceptions.IntegrityError as e:
                print(f"Something wrong with paper-repo, doi: '{pair.paper_doi}', repo: '{pair.repo_url}'")
                print(e)
            except Exception as e:
                print("Something went wrong with the following pair:")
                print(pair)
                print(e)
