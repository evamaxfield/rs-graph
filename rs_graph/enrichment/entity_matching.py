#!/usr/bin/env python

from __future__ import annotations

import logging
import traceback

from sqlmodel import Session

from ..db import models as db_models
from ..db.utils import add_model, get_engine, get_unique_first_model
from ..types import (
    ErrorResult,
    ExpandedRepositoryDocumentPair,
    LinkedRepositoryDocumentPair,
    SuccessAndErroredResultsLists,
)
from ..utils.dask_functions import process_func

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _link_repo_and_document(
    pair: ExpandedRepositoryDocumentPair,
) -> LinkedRepositoryDocumentPair | ErrorResult:
    # Get engine and init session
    engine = get_engine()
    with Session(engine) as session:
        try:
            # Get the dataset source by querying the database
            source = get_unique_first_model(
                model=db_models.DatasetSource(
                    name=pair.source,
                ),
                session=session,
            )

            # Get code host by querying the database
            code_host = get_unique_first_model(
                model=db_models.CodeHost(
                    name=pair.repo_host,
                ),
                session=session,
            )

            # Get repository by querying the database
            repository = get_unique_first_model(
                model=db_models.Repository(
                    code_host_id=code_host.id,
                    owner=pair.repo_owner,
                    name=pair.repo_name,
                ),
                session=session,
            )

            # Get document by querying the database
            document = get_unique_first_model(
                model=db_models.Document(
                    url=pair.paper_doi,
                ),
                session=session,
            )

            # Link the repository and document
            linked_repo_doc = add_model(
                model=db_models.DocumentRepositoryLink(
                    repository_id=repository.id,
                    document_id=document.id,
                    source=pair.source,
                    em_model_name=pair.em_model_name,
                    em_model_version=pair.em_model_version,
                ),
                session=session,
            )

            return LinkedRepositoryDocumentPair(
                source_id=source.id,
                code_host_id=code_host.id,
                repository_id=repository.id,
                document_id=document.id,
                document_repository_link_id=linked_repo_doc.id,
            )

        except Exception as e:
            session.rollback()
            return ErrorResult(
                source=pair.source,
                step="document-repository-db-linking",
                identifier=(
                    f"{pair.paper_doi} -- "
                    f"{pair.repo_host}/{pair.repo_owner}/{pair.repo_name}"
                ),
                error=str(e),
                traceback=traceback.format_exc(),
            )


def link_repo_and_documents(
    pairs: list[ExpandedRepositoryDocumentPair],
) -> SuccessAndErroredResultsLists:
    # Process the pairs
    results = process_func(
        name="document-repository-db-linking",
        func=_link_repo_and_document,
        func_iterables=[pairs],
        use_dask=False,  # Force synchronous processing
    )

    # Separate the successful and errored results
    successful_results = []
    errored_results = []
    for result in results:
        if isinstance(result, LinkedRepositoryDocumentPair):
            successful_results.append(result)
        elif isinstance(result, ErrorResult):
            errored_results.append(result)
        else:
            log.error(f"Unexpected result: {result}")

    return SuccessAndErroredResultsLists(
        successful_results=successful_results,
        errored_results=errored_results,
    )
