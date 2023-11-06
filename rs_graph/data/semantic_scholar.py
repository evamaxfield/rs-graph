#!/usr/bin/env python

import os
import backoff
import logging
from dotenv import load_dotenv
from semanticscholar import SemanticScholar
from semanticscholar.ApiRequester import ObjectNotFoundException
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from tqdm.contrib.concurrent import thread_map
from functools import partial

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

@dataclass
class EmbeddingDetails(DataClassJsonMixin):
    model: str
    vector: list[float]


@dataclass
class AuthorDetails(DataClassJsonMixin):
    author_id: str
    name: str
    aliases: list[str]
    h_index: int
    author_order_index: int


@dataclass
class ExtendedPaperDetails(DataClassJsonMixin):
    corpus_id: str
    url: str
    doi: str
    title: str
    authors: list[AuthorDetails]
    embedding: EmbeddingDetails | None
    citation_count: int



def _get_single_paper_details(doi: str, api: SemanticScholar) -> ExtendedPaperDetails | None:
    try:
        # Get full paper details
        paper = api.get_paper(doi)

        # Construct in parts
        # Get author details
        authors=[
            AuthorDetails(
                author_id=author.authorId,
                name=author.name,
                aliases=author.aliases,
                h_index=author.hIndex,
                # author order is typically 1-indexed in spoken language
                author_order_index=i + 1,
            )
            for i, author in enumerate(paper.authors)
        ]

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
    partial_get_single_paper_details = partial(
        _get_single_paper_details, api=api
    )

    # Get all paper details
    results = thread_map(
        partial_get_single_paper_details,
        paper_dois,
    )

    # Filter out Nones
    return [result for result in results if result is not None]