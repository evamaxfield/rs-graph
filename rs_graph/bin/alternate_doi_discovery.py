#!/usr/bin/env python

"""
Pipeline to discover and store alternate DOIs for existing documents.

This pipeline iterates through documents in the database, queries OpenAlex
and Semantic Scholar for alternate DOI versions (preprints, published versions, etc.),
and stores any discovered alternates in the document_alternate_doi table.
"""

from __future__ import annotations

import itertools
import math
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests
import typer
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from prefect import Task, flow, task, unmapped
from sqlmodel import Session, select
from tqdm import tqdm

from rs_graph.bin import pipeline_utils
from rs_graph.db import models as db_models
from rs_graph.db.utils import get_engine
from rs_graph.enrichment.article import (
    _increment_call_count_and_check,
    _setup_open_alex,
)

###############################################################################

app = typer.Typer()

DEFAULT_OPEN_ALEX_EMAILS_FILE = ".open-alex-emails.yml"

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


def _normalize_doi(doi: str) -> str:
    """Normalize a DOI for comparison."""
    return (
        doi.lower()
        .strip()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .replace("https://dx.doi.org/", "")
        .replace("doi:", "")
        .strip()
    )


def _get_doi_from_semantic_scholar(
    doi: str,
    api_key: str | None = None,
) -> str | None:
    """
    Query Semantic Scholar for a paper's DOI.

    Returns the DOI that Semantic Scholar has on file, which may differ
    from the input DOI if Semantic Scholar has resolved it to a different version.
    """
    # Handle arXiv IDs
    if "arxiv" in doi.lower():
        search_id = doi.lower().split("arxiv.")[-1]
        search_string = f"ARXIV:{search_id}"
    else:
        search_string = f"DOI:{doi}"

    url = f"https://api.semanticscholar.org/graph/v1/paper/{search_string}"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    params = {"fields": "externalIds"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        # Rate limit: ~100 requests per 5 minutes without key, more with key
        time.sleep(3.0 if not api_key else 1.0)

        if response.status_code == 404:
            return None
        response.raise_for_status()

        paper_details = response.json()
        if "externalIds" in paper_details and "DOI" in paper_details["externalIds"]:
            return paper_details["externalIds"]["DOI"]
        return None
    except Exception:
        return None


def _get_dois_from_openalex(
    doi: str,
    open_alex_email: str,
    open_alex_email_count: int,
) -> list[str]:
    """
    Query OpenAlex for all DOI variants of a work.

    OpenAlex may have the canonical DOI plus knowledge of alternate versions
    through its "ids" field and related works.
    """
    import pyalex

    _setup_open_alex(open_alex_email=open_alex_email)
    _increment_call_count_and_check(open_alex_email_count=open_alex_email_count)

    # Normalize DOI for query
    query_doi = doi.lower()
    if "doi.org" not in query_doi:
        query_doi = f"https://doi.org/{query_doi}"

    try:
        work = pyalex.Works()[query_doi]
        if work is None:
            return []

        dois = []

        # Get the primary DOI
        if work.get("doi"):
            dois.append(work["doi"].replace("https://doi.org/", ""))

        # Check for DOI in ids field
        if work.get("ids"):
            if work["ids"].get("doi"):
                doi_from_ids = work["ids"]["doi"].replace("https://doi.org/", "")
                if doi_from_ids not in dois:
                    dois.append(doi_from_ids)

        return dois
    except Exception:
        return []


def discover_alternate_dois(
    doc_info: DocumentDOIInfo,
    open_alex_email: str,
    open_alex_email_count: int,
    semantic_scholar_api_key: str | None = None,
) -> AlternateDOIResult | ErrorResult:
    """
    Discover alternate DOIs for a single document.

    Queries both OpenAlex and Semantic Scholar to find any DOI variants.
    """
    start_time = time.time()

    try:
        original_normalized = _normalize_doi(doc_info.doi)
        alternate_dois: set[str] = set()

        # Query Semantic Scholar
        ss_doi = _get_doi_from_semantic_scholar(
            doc_info.doi,
            api_key=semantic_scholar_api_key,
        )
        ss_doi_normalized = _normalize_doi(ss_doi) if ss_doi else None

        # Query OpenAlex
        oa_dois = _get_dois_from_openalex(
            doc_info.doi,
            open_alex_email=open_alex_email,
            open_alex_email_count=open_alex_email_count,
        )

        # Collect alternates (any DOI that differs from the original)
        if ss_doi_normalized and ss_doi_normalized != original_normalized:
            alternate_dois.add(ss_doi_normalized)

        for oa_doi in oa_dois:
            oa_doi_normalized = _normalize_doi(oa_doi)
            if oa_doi_normalized != original_normalized:
                alternate_dois.add(oa_doi_normalized)

        end_time = time.time()

        return AlternateDOIResult(
            document_id=doc_info.document_id,
            original_doi=doc_info.doi,
            alternate_dois=list(alternate_dois),
            openalex_doi=oa_dois[0] if oa_dois else None,
            semantic_scholar_doi=ss_doi,
            processing_time_seconds=end_time - start_time,
        )

    except Exception as e:
        return ErrorResult(
            source="alternate_doi_discovery",
            step="discover_alternate_dois",
            identifier=f"doc_id={doc_info.document_id}, doi={doc_info.doi}",
            error=str(e),
            traceback_str=traceback.format_exc(),
        )


###############################################################################


def get_documents_without_alternates(
    use_prod: bool = False,
    limit: int | None = None,
) -> list[DocumentDOIInfo]:
    """
    Get all documents that don't yet have alternate DOIs discovered.

    Returns documents where there's no entry in document_alternate_doi table.
    """
    engine = get_engine(use_prod=use_prod)

    with Session(engine) as session:
        # Get document IDs that already have alternates
        existing_alternates_query = select(
            db_models.DocumentAlternateDOI.document_id
        ).distinct()
        existing_doc_ids = set(session.exec(existing_alternates_query).all())

        # Get all documents
        docs_query = select(db_models.Document.id, db_models.Document.doi).where(
            db_models.Document.doi != None  # noqa: E711
        )

        if limit:
            docs_query = docs_query.limit(limit)

        results = session.exec(docs_query).all()

        # Filter out documents that already have alternates processed
        # (We'll track processed docs separately to avoid re-querying)
        doc_infos = [
            DocumentDOIInfo(document_id=doc_id, doi=doi)
            for doc_id, doi in results
            if doc_id not in existing_doc_ids
        ]

        return doc_infos


def store_alternate_dois(
    result: AlternateDOIResult | ErrorResult,
    use_prod: bool = False,
) -> AlternateDOIResult | ErrorResult:
    """Store discovered alternate DOIs in the database."""
    if isinstance(result, ErrorResult):
        return result

    if not result.alternate_dois:
        # No alternates to store
        return result

    engine = get_engine(use_prod=use_prod)

    try:
        with Session(engine) as session:
            for alt_doi in result.alternate_dois:
                # Check if this alternate DOI already exists
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

            session.commit()

        return result

    except Exception as e:
        return ErrorResult(
            source="alternate_doi_discovery",
            step="store_alternate_dois",
            identifier=f"doc_id={result.document_id}",
            error=str(e),
            traceback_str=traceback.format_exc(),
        )


###############################################################################


@task(
    log_prints=True,
    retries=2,
    retry_delay_seconds=5,
    retry_jitter_factor=0.5,
)
def discover_alternate_dois_task(
    doc_info: DocumentDOIInfo,
    open_alex_email: str,
    open_alex_email_count: int,
    semantic_scholar_api_key: str | None = None,
) -> AlternateDOIResult | ErrorResult:
    """Prefect task wrapper for discover_alternate_dois."""
    return discover_alternate_dois(
        doc_info=doc_info,
        open_alex_email=open_alex_email,
        open_alex_email_count=open_alex_email_count,
        semantic_scholar_api_key=semantic_scholar_api_key,
    )


@task(
    log_prints=True,
    retries=3,
    retry_delay_seconds=3,
    retry_jitter_factor=0.5,
)
def store_alternate_dois_task(
    result: AlternateDOIResult | ErrorResult,
    use_prod: bool = False,
) -> AlternateDOIResult | ErrorResult:
    """Prefect task wrapper for store_alternate_dois."""
    return store_alternate_dois(result=result, use_prod=use_prod)


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


@flow(log_prints=True)
def alternate_doi_discovery_flow(
    use_prod: bool = False,
    use_coiled: bool = False,
    coiled_region: str = "us-west-2",
    open_alex_emails_file: str = DEFAULT_OPEN_ALEX_EMAILS_FILE,
    semantic_scholar_api_key: str | None = None,
    batch_size: int = 50,
    limit: int | None = None,
) -> None:
    """
    Discover alternate DOIs for documents in the database.

    Args:
        use_prod: Whether to use production database
        use_coiled: Whether to use Coiled for distributed execution
        coiled_region: AWS region for Coiled cluster
        open_alex_emails_file: Path to OpenAlex emails YAML file
        semantic_scholar_api_key: Semantic Scholar API key
        batch_size: Number of documents to process per batch
        limit: Maximum number of documents to process (None for all)
    """
    # Load credentials and process documents
    _run_alternate_doi_discovery(
        use_prod=use_prod,
        use_coiled=use_coiled,
        coiled_region=coiled_region,
        open_alex_emails_file=open_alex_emails_file,
        semantic_scholar_api_key=semantic_scholar_api_key,
        batch_size=batch_size,
        limit=limit,
    )


def _process_batches(
    doc_infos: list[DocumentDOIInfo],
    discover_task: Task,
    cycled_emails: itertools.cycle,
    n_open_alex_emails: int,
    semantic_scholar_api_key: str | None,
    use_prod: bool,
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

        # Discover alternate DOIs
        discovery_futures = discover_task.map(
            doc_info=batch,
            open_alex_email=[next(cycled_emails) for _ in range(len(batch))],
            open_alex_email_count=unmapped(n_open_alex_emails),
            semantic_scholar_api_key=unmapped(semantic_scholar_api_key),
        )

        # Store results
        store_futures = store_alternate_dois_task.map(
            result=discovery_futures,
            use_prod=unmapped(use_prod),
        )

        # Collect results
        batch_results = [f.result() for f in store_futures]
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

    return all_results, processing_times, total_alternates_found, total_errors


def _run_alternate_doi_discovery(
    use_prod: bool,
    use_coiled: bool,
    coiled_region: str,
    open_alex_emails_file: str,
    semantic_scholar_api_key: str | None,
    batch_size: int,
    limit: int | None,
) -> None:
    """Run the alternate DOI discovery process."""
    # Load credentials
    open_alex_emails = pipeline_utils._load_open_alex_emails(open_alex_emails_file)
    n_open_alex_emails = len(open_alex_emails)
    cycled_emails = itertools.cycle(open_alex_emails)

    print(f"Loaded {n_open_alex_emails} OpenAlex emails")
    ss_status = "loaded" if semantic_scholar_api_key else "not configured"
    print(f"Semantic Scholar API key: {ss_status}")

    # Get documents to process
    print("\nFetching documents without alternate DOIs...")
    doc_infos = get_documents_without_alternates(use_prod=use_prod, limit=limit)
    print(f"Found {len(doc_infos)} documents to process")

    if not doc_infos:
        print("No documents to process. Exiting.")
        return

    # Setup task wrapper with coiled if enabled
    if use_coiled:
        discover_task = pipeline_utils._wrap_func_with_coiled_prefect_task(
            discover_alternate_dois,
            coiled_kwargs=pipeline_utils._get_small_cpu_api_cluster(
                n_workers=n_open_alex_emails,
                use_coiled=use_coiled,
                coiled_region=coiled_region,
            ),
            timeout_seconds=300,
        )
    else:
        discover_task = discover_alternate_dois_task

    # Process in batches
    all_results, processing_times, total_alternates_found, total_errors = _process_batches(
        doc_infos=doc_infos,
        discover_task=discover_task,
        cycled_emails=cycled_emails,
        n_open_alex_emails=n_open_alex_emails,
        semantic_scholar_api_key=semantic_scholar_api_key,
        use_prod=use_prod,
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
    use_prod: bool = False,
    use_coiled: bool = False,
    coiled_region: str = "us-west-2",
    open_alex_emails_file: str = DEFAULT_OPEN_ALEX_EMAILS_FILE,
    batch_size: int = 50,
    limit: int | None = None,
) -> None:
    """
    Discover and store alternate DOIs for existing documents.

    This pipeline queries OpenAlex and Semantic Scholar to find alternate
    DOI versions (preprints, published versions, etc.) for documents
    already in the database.
    """
    # Load env for semantic scholar API key
    load_dotenv()
    semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    alternate_doi_discovery_flow(
        use_prod=use_prod,
        use_coiled=use_coiled,
        coiled_region=coiled_region,
        open_alex_emails_file=open_alex_emails_file,
        semantic_scholar_api_key=semantic_scholar_api_key,
        batch_size=batch_size,
        limit=limit,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    app()
