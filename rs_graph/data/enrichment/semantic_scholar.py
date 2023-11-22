#!/usr/bin/env python

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import partial

import backoff
from dataclasses_json import DataClassJsonMixin
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
    aliases: list[str]
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
    max_time=10,
)
def _get_single_paper_details(
    doi: str, api: SemanticScholar
) -> ExtendedPaperDetails | None:
    try:
        # Get full paper details
        paper = api.get_paper(doi)

        # Construct in parts
        # Get author details
        authors = []
        for i, author in enumerate(paper.authors):
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
                    aliases=author.aliases,
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
) -> list[ExtendedPaperDetails]:
    # Handle single paper
    if isinstance(paper_dois, str):
        paper_dois = [paper_dois]

    # Load env and create semantic scholar api
    load_dotenv()
    api = SemanticScholar(api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"))

    # Create partial function with api
    partial_get_single_paper_details = partial(_get_single_paper_details, api=api)

    # Get all paper details
    results = thread_map(
        partial_get_single_paper_details,
        paper_dois,
    )

    # Filter out Nones
    return [result for result in results if result is not None]
