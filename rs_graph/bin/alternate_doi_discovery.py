#!/usr/bin/env python

"""
Pipeline to discover and store alternate DOIs for existing documents.

This pipeline iterates through documents in the database, queries OpenAlex
and Semantic Scholar for alternate DOI versions (preprints, published versions, etc.),
and stores any discovered alternates in the document_alternate_doi table.

Both APIs are queried in batch mode for efficiency:
- Semantic Scholar: POST /paper/batch (up to 500 IDs per request)
- OpenAlex: pipe-separated DOI filter (up to 50 DOIs per request)
"""

from __future__ import annotations

import math
import os
import signal
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import typer
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from sqlmodel import Session, select
from tqdm import tqdm

from rs_graph.bin import pipeline_utils
from rs_graph.db import models as db_models
from rs_graph.db.utils import get_engine
from rs_graph.enrichment.article import (
    _increment_call_count_and_check,
    _setup_open_alex,
)
from rs_graph.utils.identifier_normalization import normalize_doi

###############################################################################

app = typer.Typer()

DEFAULT_OPEN_ALEX_TOKENS_FILE = ".open-alex-tokens.yml"

# Max IDs per request for each API
SS_BATCH_SIZE = 500
OA_BATCH_SIZE = 50

# Event used to signal a graceful shutdown on keyboard interrupt.
# When set, the pipeline will finish any in-progress critical storage
# and exit at the next safe point.
_shutdown_requested = threading.Event()


def _handle_sigint(signum: int, frame: object) -> None:
    if _shutdown_requested.is_set():
        # Second interrupt — restore default handler and re-raise to force quit
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
    _shutdown_requested.set()
    print(
        "\nInterrupt received. Will exit at next safe point "
        "(after current batch storage completes). Press Ctrl+C again to force quit."
    )


###############################################################################


@dataclass
class DocumentDOIInfo(DataClassJsonMixin):
    """Basic info needed to query for alternate DOIs."""

    document_id: int
    doi: str


@dataclass
class AlternateDOIResult(DataClassJsonMixin):
    """Result of alternate DOI discovery for a single document."""

    document_id: int
    original_doi: str
    alternate_dois: list[str] = field(default_factory=list)
    openalex_doi: str | None = None
    semantic_scholar_doi: str | None = None
    processing_time_seconds: float | None = None


@dataclass
class ErrorResult(DataClassJsonMixin):
    """Error result for failed processing."""

    source: str
    step: str
    identifier: str
    error: str
    traceback_str: str


@dataclass
class ProcessingTimes:
    """Tracks processing times for reporting."""

    discovery_times: deque[float] = field(default_factory=deque)
    storage_times: deque[float] = field(default_factory=deque)


###############################################################################


def _doi_to_ss_id(doi: str) -> str:
    """Convert a DOI to a Semantic Scholar paper ID string."""
    normalized = normalize_doi(doi)
    if "arxiv" in normalized:
        search_id = normalized.split("arxiv.")[-1]
        return f"ARXIV:{search_id}"
    return f"DOI:{normalized}"


def _get_dois_from_semantic_scholar_batch(
    dois: list[str],
    api_key: str | None = None,
) -> dict[str, str | None]:
    """
    Batch query Semantic Scholar for paper DOIs.

    Uses POST /paper/batch to fetch up to 500 papers at once.
    Returns a mapping from input DOI -> resolved DOI (or None if not found).
    """
    results: dict[str, str | None] = {}

    for chunk_start in range(0, len(dois), SS_BATCH_SIZE):
        chunk = dois[chunk_start : chunk_start + SS_BATCH_SIZE]
        ss_ids = [_doi_to_ss_id(doi) for doi in chunk]

        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key
        params = {"fields": "externalIds"}

        try:
            response = requests.post(
                url,
                headers=headers,
                params=params,
                json={"ids": ss_ids},
                timeout=30,
            )
            # Rate limit: one sleep per batch request
            time.sleep(3.05 if not api_key else 1.05)

            response.raise_for_status()
            papers = response.json()

            # Response is a list in the same order as input IDs.
            # Entries may be None if a paper was not found.
            for doi, paper in zip(chunk, papers, strict=False):
                if paper is None:
                    results[doi] = None
                elif (
                    "externalIds" in paper
                    and paper["externalIds"] is not None
                    and "DOI" in paper["externalIds"]
                ):
                    results[doi] = paper["externalIds"]["DOI"]
                else:
                    results[doi] = None

        except Exception as e:
            # On batch failure, mark all DOIs in chunk as None
            print(f"Semantic Scholar batch request failed: {e}")
            for doi in chunk:
                results[doi] = None

    return results


def _get_dois_from_openalex_batch(
    dois: list[str],
    open_alex_token: str,
) -> dict[str, list[str]]:
    """
    Batch query OpenAlex for DOI variants of multiple works.

    Uses pipe-separated DOI filter to query up to 50 DOIs per request.
    Returns a mapping from input DOI -> list of DOIs found in OpenAlex.
    """
    import pyalex

    _setup_open_alex(open_alex_token=open_alex_token)

    results: dict[str, list[str]] = {doi: [] for doi in dois}

    for chunk_start in range(0, len(dois), OA_BATCH_SIZE):
        chunk = dois[chunk_start : chunk_start + OA_BATCH_SIZE]
        _increment_call_count_and_check()

        # Build pipe-separated DOI filter
        query_dois = [f"https://doi.org/{normalize_doi(doi)}" for doi in chunk]
        doi_filter = "|".join(query_dois)

        try:
            works = pyalex.Works().filter(doi=doi_filter).get(per_page=OA_BATCH_SIZE)

            # Build a lookup from normalized DOI -> work
            for work_item in works:
                work: dict[str, Any] = work_item  # type: ignore[assignment]
                work_dois: list[str] = []

                if work.get("doi"):
                    work_dois.append(normalize_doi(work["doi"]))

                if work.get("ids") and work["ids"].get("doi"):
                    doi_from_ids = normalize_doi(work["ids"]["doi"])
                    if doi_from_ids not in work_dois:
                        work_dois.append(doi_from_ids)

                # Match this work back to the input DOI(s) it corresponds to
                for input_doi in chunk:
                    input_normalized = normalize_doi(input_doi)
                    if input_normalized in work_dois:
                        results[input_doi] = work_dois
                        break

        except Exception as e:
            print(f"OpenAlex batch request failed: {e}")
            # Results for this chunk remain as empty lists

    return results


def discover_alternate_dois_batch(
    doc_infos: list[DocumentDOIInfo],
    open_alex_token: str,
    semantic_scholar_api_key: str | None = None,
) -> list[AlternateDOIResult | ErrorResult]:
    """
    Discover alternate DOIs for a batch of documents.

    Queries both Semantic Scholar and OpenAlex in batch mode,
    then merges results per document.
    """
    start_time = time.time()
    dois = [doc.doi for doc in doc_infos]

    # Batch query both APIs
    try:
        ss_results = _get_dois_from_semantic_scholar_batch(
            dois, api_key=semantic_scholar_api_key
        )
    except Exception as e:
        print(f"Semantic Scholar batch failed entirely: {e}")
        ss_results = dict.fromkeys(dois)

    try:
        oa_results = _get_dois_from_openalex_batch(dois, open_alex_token=open_alex_token)
    except Exception as e:
        print(f"OpenAlex batch failed entirely: {e}")
        oa_results = {doi: [] for doi in dois}

    batch_time = time.time() - start_time
    per_doc_time = batch_time / len(doc_infos) if doc_infos else 0

    # Build per-document results
    results: list[AlternateDOIResult | ErrorResult] = []
    for doc_info in doc_infos:
        try:
            original_normalized = normalize_doi(doc_info.doi)
            alternate_dois: set[str] = set()

            # Semantic Scholar result
            ss_doi = ss_results.get(doc_info.doi)
            ss_doi_normalized = normalize_doi(ss_doi) if ss_doi else None
            if ss_doi_normalized and ss_doi_normalized != original_normalized:
                alternate_dois.add(ss_doi_normalized)

            # OpenAlex results
            oa_dois = oa_results.get(doc_info.doi, [])
            for oa_doi in oa_dois:
                oa_doi_normalized = normalize_doi(oa_doi)
                if oa_doi_normalized != original_normalized:
                    alternate_dois.add(oa_doi_normalized)

            results.append(
                AlternateDOIResult(
                    document_id=doc_info.document_id,
                    original_doi=doc_info.doi,
                    alternate_dois=list(alternate_dois),
                    openalex_doi=oa_dois[0] if oa_dois else None,
                    semantic_scholar_doi=ss_doi,
                    processing_time_seconds=per_doc_time,
                )
            )
        except Exception as e:
            results.append(
                ErrorResult(
                    source="alternate_doi_discovery",
                    step="discover_alternate_dois",
                    identifier=f"doc_id={doc_info.document_id}, doi={doc_info.doi}",
                    error=str(e),
                    traceback_str=traceback.format_exc(),
                )
            )

    return results


###############################################################################


def get_documents_without_alternates(
    database_path: str,
    limit: int | None = None,
) -> list[DocumentDOIInfo]:
    """
    Get all documents that don't yet have alternate DOIs discovered.

    Returns documents where there's no entry in document_alternate_doi table.
    """
    engine = get_engine(database_path=database_path)

    with Session(engine) as session:
        # Get document IDs that already have alternates
        existing_alternates_query = select(
            db_models.DocumentAlternateDOI.document_id
        ).distinct()
        existing_doc_ids = set(session.exec(existing_alternates_query).all())

        # Get all documents
        # Sort by publication datetime (oldest first)
        docs_query = (
            select(db_models.Document.id, db_models.Document.doi)
            .where(
                db_models.Document.doi != None  # noqa: E711
            )
            .order_by("publication_date")
        )

        if limit:
            docs_query = docs_query.limit(limit)

        results = session.exec(docs_query).all()

        # Filter out documents that already have alternates processed
        doc_infos = [
            DocumentDOIInfo(document_id=int(doc_id), doi=str(doi))
            for doc_id, doi in results
            if doc_id not in existing_doc_ids
        ]

        return doc_infos


def store_alternate_dois_batch(
    results: list[AlternateDOIResult | ErrorResult],
    database_path: str,
) -> list[AlternateDOIResult | ErrorResult]:
    """Store discovered alternate DOIs in the database for a batch of results."""
    engine = get_engine(database_path=database_path)

    stored_results: list[AlternateDOIResult | ErrorResult] = []
    try:
        with Session(engine) as session:
            for result in results:
                if isinstance(result, ErrorResult) or not result.alternate_dois:
                    stored_results.append(result)
                    continue

                try:
                    for alt_doi in result.alternate_dois:
                        existing = session.exec(
                            select(db_models.DocumentAlternateDOI).where(
                                db_models.DocumentAlternateDOI.doi == alt_doi
                            )
                        ).first()

                        if existing is None:
                            alternate_model = db_models.DocumentAlternateDOI(
                                document_id=result.document_id,
                                doi=alt_doi,
                            )
                            session.add(alternate_model)

                    stored_results.append(result)
                except Exception as e:
                    stored_results.append(
                        ErrorResult(
                            source="alternate_doi_discovery",
                            step="store_alternate_dois",
                            identifier=f"doc_id={result.document_id}",
                            error=str(e),
                            traceback_str=traceback.format_exc(),
                        )
                    )

            session.commit()

    except Exception as e:
        # If the entire commit fails, convert remaining to errors
        print(f"Batch commit failed: {e}")
        for result in results:
            if isinstance(result, AlternateDOIResult) and result.alternate_dois:
                stored_results.append(
                    ErrorResult(
                        source="alternate_doi_discovery",
                        step="store_alternate_dois",
                        identifier=f"doc_id={result.document_id}",
                        error=str(e),
                        traceback_str=traceback.format_exc(),
                    )
                )

    return stored_results


###############################################################################


def _report_statistics(
    processing_times: ProcessingTimes,
    total_processed: int,
    total_alternates_found: int,
    total_errors: int,
) -> None:
    """Report processing statistics."""
    print("\n" + "=" * 60)
    print("Processing Statistics")
    print("=" * 60)
    print(f"Total documents processed: {total_processed}")
    print(f"Total alternate DOIs found: {total_alternates_found}")
    print(f"Total errors: {total_errors}")

    if processing_times.discovery_times:
        discovery_series = pd.Series(list(processing_times.discovery_times))
        described = discovery_series.describe()
        print(
            f"Discovery time (seconds): median={described['50%']:.2f}, "
            f"mean={described['mean']:.2f} +/- {described['std']:.2f}"
        )


def _process_batches(
    doc_infos: list[DocumentDOIInfo],
    open_alex_token: str,
    semantic_scholar_api_key: str | None,
    database_path: str,
    batch_size: int,
) -> tuple[list[AlternateDOIResult | ErrorResult], ProcessingTimes, int, int]:
    """Process documents in batches and return results with statistics."""
    processing_times = ProcessingTimes()
    total_alternates_found = 0
    total_errors = 0
    all_results: list[AlternateDOIResult | ErrorResult] = []

    n_batches = math.ceil(len(doc_infos) / batch_size)
    print(f"\nProcessing {len(doc_infos)} documents in {n_batches} batches...")

    for i in tqdm(range(0, len(doc_infos), batch_size), total=n_batches, desc="Batches"):
        batch = doc_infos[i : i + batch_size]

        # Batch discover alternate DOIs
        batch_results = discover_alternate_dois_batch(
            doc_infos=batch,
            open_alex_token=open_alex_token,
            semantic_scholar_api_key=semantic_scholar_api_key,
        )

        # Batch store results
        batch_results = store_alternate_dois_batch(
            results=batch_results,
            database_path=database_path,
        )

        all_results.extend(batch_results)

        # Track statistics
        for result in batch_results:
            if isinstance(result, ErrorResult):
                total_errors += 1
            else:
                if result.processing_time_seconds:
                    processing_times.discovery_times.append(result.processing_time_seconds)
                total_alternates_found += len(result.alternate_dois)

        # Report progress
        if (i // batch_size + 1) % 10 == 0:
            print(
                f"\nProgress: {i + len(batch)}/{len(doc_infos)} documents, "
                f"{total_alternates_found} alternates found, {total_errors} errors"
            )

        if _shutdown_requested.is_set():
            print("Shutdown requested — exiting after completing current batch.")
            break

    return all_results, processing_times, total_alternates_found, total_errors


def _run_alternate_doi_discovery(
    database_path: str,
    open_alex_tokens_file: str,
    semantic_scholar_api_key: str | None,
    batch_size: int,
    limit: int | None,
) -> None:
    """Run the alternate DOI discovery process."""
    # Load credentials
    open_alex_tokens = pipeline_utils._load_open_alex_tokens(open_alex_tokens_file)
    n_open_alex_tokens = len(open_alex_tokens)

    print(f"Loaded {n_open_alex_tokens} OpenAlex tokens")
    ss_status = "loaded" if semantic_scholar_api_key else "not configured"
    print(f"Semantic Scholar API key: {ss_status}")

    # Get documents to process
    print("\nFetching documents without alternate DOIs...")
    doc_infos = get_documents_without_alternates(database_path=database_path, limit=limit)
    print(f"Found {len(doc_infos)} documents to process")

    if not doc_infos:
        print("No documents to process. Exiting.")
        return

    # Use first token for batch queries (token cycling not needed with batch)
    open_alex_token = open_alex_tokens[0]

    # Process in batches
    all_results, processing_times, total_alternates_found, total_errors = _process_batches(
        doc_infos=doc_infos,
        open_alex_token=open_alex_token,
        semantic_scholar_api_key=semantic_scholar_api_key,
        database_path=database_path,
        batch_size=batch_size,
    )

    # Final report
    _report_statistics(
        processing_times=processing_times,
        total_processed=len(doc_infos),
        total_alternates_found=total_alternates_found,
        total_errors=total_errors,
    )

    # Store error results to file
    error_results = [r for r in all_results if isinstance(r, ErrorResult)]
    if error_results:
        error_path = Path("alternate_doi_discovery_errors.parquet")
        error_df = pd.DataFrame([r.to_dict() for r in error_results])
        error_df.to_parquet(error_path)
        print(f"\nErrors saved to {error_path}")


###############################################################################


@app.command()
def alternate_doi_discovery(
    database_path: str = typer.Argument(
        help="Path to the SQLite database file to use.",
    ),
    open_alex_tokens_file: str = DEFAULT_OPEN_ALEX_TOKENS_FILE,
    batch_size: int = 500,
    limit: int | None = None,
) -> None:
    """
    Discover and store alternate DOIs for existing documents.

    This pipeline queries OpenAlex and Semantic Scholar to find alternate
    DOI versions (preprints, published versions, etc.) for documents
    already in the database.

    Both APIs are queried in batch mode for efficiency.
    """
    # Install graceful shutdown handler
    signal.signal(signal.SIGINT, _handle_sigint)

    # Load env for semantic scholar API key
    load_dotenv()
    semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    _run_alternate_doi_discovery(
        database_path=database_path,
        open_alex_tokens_file=open_alex_tokens_file,
        semantic_scholar_api_key=semantic_scholar_api_key,
        batch_size=batch_size,
        limit=limit,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    app()
