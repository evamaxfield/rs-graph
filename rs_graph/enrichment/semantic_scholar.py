#!/usr/bin/env python

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import partial
from uuid import uuid4

import backoff
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from semanticscholar import SemanticScholar
from semanticscholar.ApiRequester import ObjectNotFoundException
from tqdm.contrib.concurrent import thread_map

###############################################################################

log = logging.getLogger(__name__)

# Stop "httpx" from logging
logging.getLogger("httpx").setLevel(logging.WARNING)

###############################################################################


class AuthorOrder:
    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"
    ONLY = "only"


@dataclass
class EmbeddingDetails(DataClassJsonMixin):
    model: str
    vector: list[float]


@dataclass
class AuthorDetails(DataClassJsonMixin):
    author_id: str
    name: str
    affiliations: list[str] | None
    h_index: int
    author_position: str


@dataclass
class ExtendedPaperDetails(DataClassJsonMixin):
    corpus_id: str
    url: str
    doi: str
    title: str
    authors: list[AuthorDetails]
    abstract: str | None
    tldr: str
    topics: list[str]
    primary_topic: str | None
    secondary_topic: str | None
    citation_count: int


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_time=10,
)
def _get_single_paper_details(
    doi: str, api: SemanticScholar
) -> ExtendedPaperDetails | None:
    try:
        # Get full paper details
        paper = api.get_paper(
            doi,
            fields=[
                "corpusId",
                "title",
                "authors",
                "url",
                "abstract",
                "citationCount",
                "tldr",
                "venue",
                "publicationVenue",
                "externalIds",
                "publicationTypes",
                "journal",
                "s2FieldsOfStudy",
            ],
        )

        return paper

        # Construct in parts
        # Get author details
        authors = []
        for i, author in enumerate(paper.authors):
            # Get uuid4 if author has no author id
            if author.authorId is None:
                processed_author_id = str(uuid4())
            else:
                processed_author_id = author.authorId

            if i == 0:
                if len(paper.authors) > 1:
                    author_position = AuthorOrder.FIRST
                else:
                    author_position = AuthorOrder.ONLY
            elif i == len(paper.authors) - 1:
                author_position = AuthorOrder.LAST
            else:
                author_position = AuthorOrder.MIDDLE

            authors.append(
                AuthorDetails(
                    author_id=processed_author_id,
                    name=author.name,
                    affiliations=author.affiliations,
                    h_index=author.hIndex,
                    author_position=author_position,
                )
            )

        # Process topics
        topics = []
        for topic in paper.s2FieldsOfStudy:
            topic_cat = topic["category"]
            if topic_cat not in topics:
                topics.append(topic_cat)

        # Return parsed object
        return ExtendedPaperDetails(
            corpus_id=paper.corpusId,
            url=paper.url,
            doi=doi,
            title=paper.title,
            authors=authors,
            abstract=paper.abstract,
            citation_count=paper.citationCount,
            tldr=paper.tldr["text"] if paper.tldr is not None else None,
            topics=topics,
            primary_topic=topics[0] if len(topics) > 0 else None,
            secondary_topic=topics[1] if len(topics) > 1 else None,
        )

    # Handle any error and return None
    except ObjectNotFoundException:
        return None


def get_extended_paper_details(
    paper_dois: list[str] | str,
    filter_out_nones: bool = True,
) -> list[ExtendedPaperDetails]:
    # Handle single paper
    if isinstance(paper_dois, str):
        paper_dois = [paper_dois]

    # Load env and create semantic scholar api
    load_dotenv()
    api = SemanticScholar(api_key=os.environ["SEMANTIC_SCHOLAR_API_KEY"])

    # Create partial function with api
    partial_get_single_paper_details = partial(_get_single_paper_details, api=api)

    # Get all paper details
    results = thread_map(
        partial_get_single_paper_details,
        paper_dois,
    )

    # Filter out Nones
    if filter_out_nones:
        return [result for result in results if result is not None]

    # Return results
    return results
