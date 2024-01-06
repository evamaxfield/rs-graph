#!/usr/bin/env python

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import partial
from uuid import uuid4

import backoff
from dataclasses_json import DataClassJsonMixin
from distributed import worker_client
from dotenv import load_dotenv
from semanticscholar import SemanticScholar
from semanticscholar.ApiRequester import ObjectNotFoundException
from tqdm.contrib.concurrent import thread_map

###############################################################################

log = logging.getLogger(__name__)

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
    embedding: EmbeddingDetails | None
    citation_count: int


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_time=16,
)
def _get_single_paper_details(
    doi: str, semantic_scholar_api: SemanticScholar | str
) -> ExtendedPaperDetails | None:
    try:
        # Handle api
        if isinstance(semantic_scholar_api, str):
            semantic_scholar_api = SemanticScholar(semantic_scholar_api)

        # Get full paper details
        paper = semantic_scholar_api.get_paper(doi)

        # Construct in parts
        # Get author details
        authors = []
        for i, author in enumerate(paper.authors):
            # Get uuid4 if author has no author id
            if author.authorId is None:
                author.authorId = str(uuid4())

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
                    author_id=author.authorId,
                    name=author.name,
                    affiliations=author.affiliations,
                    h_index=author.hIndex,
                    author_position=author_position,
                )
            )

        # Get embedding details
        if paper.embedding is not None:
            embedding_details = EmbeddingDetails(
                model=paper.embedding["model"],
                vector=paper.embedding["vector"],
            )
        else:
            embedding_details = None

        # Return parsed object
        return ExtendedPaperDetails(
            corpus_id=paper.corpusId,
            url=paper.url,
            doi=doi,
            title=paper.title,
            authors=authors,
            embedding=embedding_details,
            citation_count=paper.citationCount,
        )

    # Handle any error and return None
    except ObjectNotFoundException:
        return None


def get_extended_paper_details(
    paper_dois: list[str] | str,
    semantic_scholar_api_key: str | None = None,
) -> list[ExtendedPaperDetails]:
    # Handle single paper
    if isinstance(paper_dois, str):
        paper_dois = [paper_dois]

    # Load env and create semantic scholar api
    if not semantic_scholar_api_key:
        load_dotenv()
        semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    # Get all paper details
    try:
        with worker_client() as client:
            # Create partial function with api
            partial_get_single_paper_details = partial(
                _get_single_paper_details,
                semantic_scholar_api=semantic_scholar_api_key,
            )

            futures = client.map(
                partial_get_single_paper_details,
                paper_dois,
            )
            results = client.gather(futures)

    # Run locally
    except ValueError:
        # Create single api to pass to thread_map
        semantic_scholar_api = SemanticScholar(semantic_scholar_api_key)

        # Create partial function with api
        partial_get_single_paper_details = partial(
            _get_single_paper_details,
            semantic_scholar_api=semantic_scholar_api,
        )

        results = thread_map(
            partial_get_single_paper_details,
            paper_dois,
        )

    # Filter out Nones
    return [result for result in results if result is not None]
