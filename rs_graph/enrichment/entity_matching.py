#!/usr/bin/env python

from __future__ import annotations

import logging
import traceback

from sqlmodel import Session, select

from ..db import models as db_models
from ..db.utils import get_engine, get_or_add, get_unique_first_model
from ..ml import dev_author_em_clf
from ..types import (
    ErrorResult,
    ExpandedRepositoryDocumentPair,
    LinkedDeveloperResearcherPair,
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

            # Check code host found
            if code_host is None or code_host.id is None:
                raise ValueError("Could not find code host")

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
                    doi=pair.paper_doi,
                ),
                session=session,
            )

            # Check that all models are found
            if (
                source is None
                or source.id is None
                or repository is None
                or repository.id is None
                or document is None
                or document.id is None
            ):
                raise ValueError("Could not find all required models")

            # Link the repository and document
            linked_repo_doc = get_or_add(
                model=db_models.DocumentRepositoryLink(
                    repository_id=repository.id,
                    document_id=document.id,
                    source=pair.source,  # TODO: use source.id
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
    return SuccessAndErroredResultsLists(
        successful_results=[
            r for r in results if isinstance(r, LinkedRepositoryDocumentPair)
        ],
        errored_results=[r for r in results if isinstance(r, ErrorResult)],
    )


def _link_devs_and_researchers_for_repo_paper_pair(
    linked_repo_doc: LinkedRepositoryDocumentPair,
) -> list[LinkedDeveloperResearcherPair] | ErrorResult:
    # Get engine and init session
    engine = get_engine()
    with Session(engine) as session:
        try:
            # Get list of dev acounts which have
            # contributed to this repo
            statement = (
                select(
                    db_models.DeveloperAccount,
                    db_models.RepositoryContributor,
                )
                .join(
                    db_models.RepositoryContributor,
                )
                .where(
                    db_models.RepositoryContributor.repository_id
                    == linked_repo_doc.repository_id,
                )
            )
            devs = [dev for dev, _ in session.exec(statement)]

            # Get list of researchers who have
            # contributed to this document
            statement = (
                select(
                    db_models.Researcher,
                    db_models.DocumentContributor,
                )
                .join(
                    db_models.DocumentContributor,
                )
                .where(
                    db_models.DocumentContributor.document_id
                    == linked_repo_doc.document_id,
                )
            )
            researchers = [researcher for researcher, _ in session.exec(statement)]

            # Convert these to ml ready types and luts
            dev_username_to_dev_details = {
                dev.username: dev_author_em_clf.DeveloperDetails(
                    username=dev.username,
                    name=dev.name,
                    email=dev.email,
                )
                for dev in devs
            }
            dev_username_to_dev_model = {dev.username: dev for dev in devs}
            researcher_name_to_researcher_model = {
                researcher.name: researcher for researcher in researchers
            }

            # Predict matches
            matches = dev_author_em_clf.match_devs_and_authors(
                devs=list(dev_username_to_dev_details.values()),
                authors=[researcher.name for researcher in researchers],
            )

            # Create the linked pairs
            created_models = []
            for dev_username, author_name in matches.items():
                linked_researcher_dev_acount = get_or_add(
                    model=db_models.ResearcherDeveloperAccountLink(
                        researcher_id=researcher_name_to_researcher_model[
                            author_name
                        ].id,
                        developer_id=dev_username_to_dev_model[dev_username].id,
                    ),
                    session=session,
                )
                created_models.append(linked_researcher_dev_acount)

            return [
                LinkedDeveloperResearcherPair(
                    **linked_repo_doc.to_dict(),
                    researcher_id=researcher_name_to_researcher_model[author_name].id,
                    developer_id=dev_username_to_dev_model[dev_username].id,
                    researcher_developer_link_id=linked_researcher_dev_acount.id,
                )
            ]

        except Exception:
            session.rollback()
            return ErrorResult()


def link_devs_and_researchers(
    pairs: list[LinkedRepositoryDocumentPair],
    use_dask: bool = False,
) -> SuccessAndErroredResultsLists:
    # Process the pairs
    results = process_func(
        name="developer-researcher-db-linking",
        func=_link_devs_and_researchers_for_repo_paper_pair,
        func_iterables=[pairs],
        use_dask=use_dask,
        cluster_kwargs={
            "processes": True,
            "n_workers": 4,
            "threads_per_worker": 1,
        },
    )

    # Separate the successful and errored results
    # Further, the successful results are lists of LinkedDeveloperResearcherPair
    # Flatten the list of lists
    return SuccessAndErroredResultsLists(
        # flatten the list of lists
        successful_results=[
            item for sublist in results for item in sublist if isinstance(sublist, list)
        ],
        errored_results=[r for r in results if isinstance(r, ErrorResult)],
    )
