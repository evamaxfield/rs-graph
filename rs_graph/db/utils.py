#!/usr/bin/env python

import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from prefect import task
from sqlalchemy.engine import Engine
from sqlalchemy.sql.operators import is_
from sqlmodel import (
    Session,
    SQLModel,
    UniqueConstraint,
    col,
    create_engine,
    or_,
    select,
)
from tqdm import tqdm

from .. import types
from ..utils.dt_and_td import parse_timedelta
from . import models as db_models
from .constants import V2_DATABASE_PATHS

###############################################################################


def get_engine(use_prod: bool = False) -> Engine:
    db_path = Path(__file__).parent.parent / "data" / "files"

    if use_prod:
        db_path = db_path / V2_DATABASE_PATHS.prod.name

        return create_engine(f"sqlite:///{db_path}")
    else:
        db_path = db_path / V2_DATABASE_PATHS.dev.name

        return create_engine(f"sqlite:///{db_path}")


def get_unique_first_model(model: SQLModel, session: Session) -> SQLModel | None:
    # Get model class
    model_cls = model.__class__

    # Get constrained fields
    all_table_contraints = model_cls.__table__.constraints
    unique_constraints = [
        constraint
        for constraint in all_table_contraints
        if isinstance(constraint, UniqueConstraint)
    ]

    # Unpack constraints to just a list (converted from set) of all of the field names
    # in each constraint
    required_fields = list(
        {field.name for constraint in unique_constraints for field in constraint.columns}
    )

    # Try and select a matching model
    query = select(model_cls)
    for field in required_fields:
        query = query.where(getattr(model_cls, field) == getattr(model, field))

    # Execute the query
    return session.exec(query).first()


def _get_or_add_and_flush(
    model: SQLModel,
    session: Session,
) -> SQLModel:
    # Try getting matching model
    existing_model = get_unique_first_model(model=model, session=session)

    # If model exists, return it
    if existing_model is not None:
        return existing_model

    # Otherwise, add and flush
    session.add(model)
    session.flush()

    return model


def store_full_details(  # noqa: C901
    pair: types.ExpandedRepositoryDocumentPair,
    use_prod: bool = False,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    # Get the engine
    engine = get_engine(use_prod=use_prod)

    # Create a session
    with Session(engine) as session:
        try:
            assert pair.open_alex_results is not None
            assert pair.github_results is not None
            assert pair.repo_parts is not None

            # Dataset source
            pair.open_alex_results.dataset_source_model = _get_or_add_and_flush(
                model=pair.open_alex_results.dataset_source_model, session=session
            )
            assert pair.open_alex_results.dataset_source_model.id is not None

            # Store the primary document source
            if pair.open_alex_results.primary_document_source_model is not None:
                pair.open_alex_results.primary_document_source_model = _get_or_add_and_flush(
                    model=pair.open_alex_results.primary_document_source_model,
                    session=session,
                )
                assert (
                    pair.open_alex_results.primary_document_source_model.id  # type:ignore
                    is not None
                )

            # Store the primary location
            if pair.open_alex_results.primary_location_model is not None:
                # Update with the document source id
                if pair.open_alex_results.primary_document_source_model is not None:
                    pair.open_alex_results.primary_location_model.source_id = (
                        pair.open_alex_results.primary_document_source_model.id  # type: ignore
                    )

                # Add and flush
                pair.open_alex_results.primary_location_model = _get_or_add_and_flush(
                    model=pair.open_alex_results.primary_location_model, session=session
                )
                assert (
                    pair.open_alex_results.primary_location_model.id is not None  # type: ignore
                )

                # Update the document model with the primary location id
                pair.open_alex_results.document_model.primary_location_id = (
                    pair.open_alex_results.primary_location_model.id  # type: ignore
                )

            # Store the best oa source
            if pair.open_alex_results.best_oa_document_source_model is not None:
                pair.open_alex_results.best_oa_document_source_model = _get_or_add_and_flush(
                    model=pair.open_alex_results.best_oa_document_source_model,
                    session=session,
                )
                assert (
                    pair.open_alex_results.best_oa_document_source_model.id  # type: ignore
                    is not None
                )

            # Store the best oa location
            if pair.open_alex_results.best_oa_location_model is not None:
                # Update with the document source id
                if pair.open_alex_results.best_oa_document_source_model is not None:
                    pair.open_alex_results.best_oa_location_model.source_id = (
                        pair.open_alex_results.best_oa_document_source_model.id  # type: ignore
                    )

                # Add and flush
                pair.open_alex_results.best_oa_location_model = _get_or_add_and_flush(
                    model=pair.open_alex_results.best_oa_location_model, session=session
                )
                assert (
                    pair.open_alex_results.best_oa_location_model.id is not None  # type: ignore
                )

                # Update the document model with the best oa location id
                pair.open_alex_results.document_model.best_open_access_location_id = (
                    pair.open_alex_results.best_oa_location_model.id  # type: ignore
                )

            # Document
            pair.open_alex_results.document_model = _get_or_add_and_flush(
                model=pair.open_alex_results.document_model, session=session
            )
            assert pair.open_alex_results.document_model.id is not None

            # Store the abstract
            pair.open_alex_results.document_abstract_model.document_id = (
                pair.open_alex_results.document_model.id
            )
            pair.open_alex_results.document_abstract_model = _get_or_add_and_flush(
                model=pair.open_alex_results.document_abstract_model,
                session=session,
            )

            # Store alternate DOIs
            for alternate_doi in pair.open_alex_results.document_alternate_dois:
                # Create new model
                alternate_doi_model = db_models.DocumentAlternateDOI(
                    document_id=pair.open_alex_results.document_model.id,
                    doi=alternate_doi,
                )

                # Persist
                alternate_doi_model = _get_or_add_and_flush(
                    model=alternate_doi_model,
                    session=session,
                )

            # Topic details
            for topic_detail in pair.open_alex_results.topic_details:
                # Topic model
                topic_detail.topic_model = _get_or_add_and_flush(
                    model=topic_detail.topic_model, session=session
                )
                assert topic_detail.topic_model.id is not None

                # Document topic model
                topic_detail.document_topic_model.document_id = (
                    pair.open_alex_results.document_model.id
                )
                topic_detail.document_topic_model.topic_id = topic_detail.topic_model.id
                topic_detail.document_topic_model = _get_or_add_and_flush(
                    model=topic_detail.document_topic_model, session=session
                )
                assert topic_detail.document_topic_model.id is not None

            # Researcher details
            for researcher_detail in pair.open_alex_results.researcher_details:
                # Researcher model
                researcher_detail.researcher_model = _get_or_add_and_flush(
                    model=researcher_detail.researcher_model, session=session
                )
                assert researcher_detail.researcher_model.id is not None

                # Document contributor model
                researcher_detail.document_contributor_model.document_id = (
                    pair.open_alex_results.document_model.id
                )
                researcher_detail.document_contributor_model.researcher_id = (
                    researcher_detail.researcher_model.id
                )
                researcher_detail.document_contributor_model = _get_or_add_and_flush(
                    model=researcher_detail.document_contributor_model, session=session
                )
                assert researcher_detail.document_contributor_model.id is not None

                # Institution models
                for institution_model in researcher_detail.institution_models:
                    institution_model = _get_or_add_and_flush(
                        model=institution_model, session=session
                    )
                    assert institution_model.id is not None

                    # Create all of the researcher document institution models
                    r_d_i = db_models.DocumentContributorInstitution(
                        document_contributor_id=researcher_detail.document_contributor_model.id,
                        institution_id=institution_model.id,
                    )
                    _get_or_add_and_flush(model=r_d_i, session=session)

            # Funding instance details
            for funding_instance_detail in pair.open_alex_results.funding_instance_details:
                # Funder model
                funding_instance_detail.funder_model = _get_or_add_and_flush(
                    model=funding_instance_detail.funder_model, session=session
                )
                assert funding_instance_detail.funder_model.id is not None

                # Funding instance model
                funding_instance_detail.funding_instance_model.funder_id = (
                    funding_instance_detail.funder_model.id
                )
                funding_instance_detail.funding_instance_model = _get_or_add_and_flush(
                    model=funding_instance_detail.funding_instance_model,
                    session=session,
                )
                assert funding_instance_detail.funding_instance_model.id is not None

                # Create the document funding instance models
                d_f_i = db_models.DocumentFundingInstance(
                    document_id=pair.open_alex_results.document_model.id,
                    funding_instance_id=funding_instance_detail.funding_instance_model.id,
                )
                _get_or_add_and_flush(model=d_f_i, session=session)

            # Code host
            pair.github_results.code_host_model = _get_or_add_and_flush(
                model=pair.github_results.code_host_model, session=session
            )
            assert pair.github_results.code_host_model.id is not None

            # Repository
            pair.github_results.repository_model.code_host_id = (
                pair.github_results.code_host_model.id
            )
            pair.github_results.repository_model = _get_or_add_and_flush(
                model=pair.github_results.repository_model, session=session
            )
            assert pair.github_results.repository_model.id is not None

            # Store the README
            pair.github_results.repository_readme_model.repository_id = (
                pair.github_results.repository_model.id
            )
            pair.github_results.repository_readme_model = _get_or_add_and_flush(
                model=pair.github_results.repository_readme_model, session=session
            )

            # Repository languages
            for repo_language in pair.github_results.repository_language_models:
                repo_language.repository_id = pair.github_results.repository_model.id
                repo_language = _get_or_add_and_flush(model=repo_language, session=session)

            # Store the repository file models
            for repo_file in pair.github_results.repository_file_models:
                repo_file.repository_id = pair.github_results.repository_model.id
                repo_file = _get_or_add_and_flush(model=repo_file, session=session)

            # Repository contributor details
            for repo_contributor_detail in pair.github_results.repository_contributor_details:
                # Developer account
                repo_contributor_detail.developer_account_model.code_host_id = (
                    pair.github_results.code_host_model.id
                )
                repo_contributor_detail.developer_account_model = _get_or_add_and_flush(
                    model=repo_contributor_detail.developer_account_model,
                    session=session,
                )
                assert repo_contributor_detail.developer_account_model.id is not None

                # Repository contributor
                repo_contributor_detail.repository_contributor_model.repository_id = (
                    pair.github_results.repository_model.id
                )
                repo_contributor_detail.repository_contributor_model.developer_account_id = (
                    repo_contributor_detail.developer_account_model.id
                )
                repo_contributor_detail.repository_contributor_model = _get_or_add_and_flush(
                    model=repo_contributor_detail.repository_contributor_model,
                    session=session,
                )

            # Create a connection between the document and the repository
            d_r = db_models.DocumentRepositoryLink(
                document_id=pair.open_alex_results.document_model.id,
                repository_id=pair.github_results.repository_model.id,
                dataset_source_id=pair.open_alex_results.dataset_source_model.id,
            )
            _get_or_add_and_flush(model=d_r, session=session)

            # Get all the info needed to refresh after commit
            dataset_source_id = pair.open_alex_results.dataset_source_model.id
            document_model_id = pair.open_alex_results.document_model.id
            repository_model_id = pair.github_results.repository_model.id
            developer_account_model_ids = [
                repo_contributor_detail.developer_account_model.id
                for repo_contributor_detail in (
                    pair.github_results.repository_contributor_details
                )
            ]
            researcher_model_ids = [
                researcher_detail.researcher_model.id
                for researcher_detail in pair.open_alex_results.researcher_details
            ]

            session.commit()

            # Get all the models after they have been added
            dataset_source_stmt = select(db_models.DatasetSource).where(
                db_models.DatasetSource.id == dataset_source_id
            )
            dataset_source_model = session.exec(dataset_source_stmt).first()
            document_stmt = select(db_models.Document).where(
                db_models.Document.id == document_model_id
            )
            document_model = session.exec(document_stmt).first()
            assert document_model is not None
            repository_stmt = select(db_models.Repository).where(
                db_models.Repository.id == repository_model_id
            )
            repository_model = session.exec(repository_stmt).first()
            assert repository_model is not None
            developer_account_stmt = select(db_models.DeveloperAccount).where(
                db_models.DeveloperAccount.id.in_(developer_account_model_ids)  # type: ignore
            )
            developer_account_models = session.exec(developer_account_stmt).all()
            researcher_stmt = select(db_models.Researcher).where(
                db_models.Researcher.id.in_(researcher_model_ids)  # type: ignore
            )
            researcher_models = session.exec(researcher_stmt).all()

            # Create the stored pair
            stored_pair = types.StoredRepositoryDocumentPair(
                dataset_source_model=dataset_source_model,
                document_model=document_model,
                repository_model=repository_model,
                developer_account_models=developer_account_models,
                researcher_models=researcher_models,
            )

            return stored_pair

        except Exception as e:
            session.rollback()
            return types.ErrorResult(
                source=pair.source,
                step="store-full-details",
                identifier=pair.paper_doi,
                error=str(e),
                traceback=traceback.format_exc(),
            )


@task(
    log_prints=True,
    retries=3,
    retry_delay_seconds=3,
    retry_jitter_factor=0.5,
)
def store_full_details_task(
    pair: types.ExpandedRepositoryDocumentPair | types.ErrorResult,
    use_prod: bool = False,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    # Get start time
    start_time = time.time()

    # Store the full details
    result = store_full_details(pair=pair, use_prod=use_prod)

    # Get end time and calculate processing time
    end_time = time.time()

    # Add processing time to the pair
    if isinstance(result, types.StoredRepositoryDocumentPair):
        result.open_alex_processing_time_seconds = end_time - start_time
        result.github_processing_time_seconds = end_time - start_time
        result.store_article_and_repository_time_seconds = end_time - start_time

    return result


def store_dev_researcher_em_links(
    pair: types.StoredRepositoryDocumentPair,
    use_prod: bool = False,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    # Get the engine
    engine = get_engine(use_prod=use_prod)

    # Create a session
    with Session(engine) as session:
        try:
            assert pair.researcher_developer_links is not None

            # Iter over each linked dev researcher link and add to the database
            link_ids = []
            for linked_dev_researcher_pair in pair.researcher_developer_links:
                linked_dev_researcher_pair = _get_or_add_and_flush(
                    model=linked_dev_researcher_pair, session=session
                )
                link_ids.append(linked_dev_researcher_pair.id)

            # Commit
            session.commit()

            # Get all the models after they have been added
            linked_dev_researcher_stmt = select(db_models.ResearcherDeveloperAccountLink).where(
                db_models.ResearcherDeveloperAccountLink.id.in_(link_ids)  # type: ignore
            )
            linked_dev_researcher_models = session.exec(linked_dev_researcher_stmt).all()

            # Attach to model
            pair.researcher_developer_links = linked_dev_researcher_models

            return pair

        except Exception as e:
            session.rollback()
            return types.ErrorResult(
                source=pair.dataset_source_model.name,
                step="store-dev-research-em-links",
                identifier=pair.document_model.doi,
                error=str(e),
                traceback=traceback.format_exc(),
            )


@task(
    log_prints=True,
    retries=3,
    retry_delay_seconds=3,
    retry_jitter_factor=0.5,
)
def store_dev_researcher_em_links_task(
    pair: types.StoredRepositoryDocumentPair | types.ErrorResult,
    use_prod: bool = False,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    # Get start time
    start_time = time.perf_counter()

    # Store the developer-researcher links
    pair = store_dev_researcher_em_links(
        pair=pair,
        use_prod=use_prod,
    )

    # Get end time
    end_time = time.perf_counter()

    # Attach processing time
    if isinstance(pair, types.StoredRepositoryDocumentPair):
        pair.store_author_developer_links_time_seconds = end_time - start_time

    return pair


def check_pair_exists(
    pair: types.ExpandedRepositoryDocumentPair,
    use_prod: bool = False,
) -> bool:
    # Get the engine
    engine = get_engine(use_prod=use_prod)

    # Check we have already processed to repo parts
    assert pair.repo_parts is not None

    # Create a session
    with Session(engine) as session:
        # Five statements
        # One to check and get the document
        # One to check for backup alternate dois
        # One to get the repo host id
        # One to check and get the repository
        # And a last to check and get the link

        # Document
        document_stmt = select(db_models.Document).where(
            db_models.Document.doi == pair.paper_doi
        )
        document_model = session.exec(document_stmt).first()

        # Check for document alternate doi
        if document_model is None:
            document_stmt = select(db_models.DocumentAlternateDOI).where(
                db_models.DocumentAlternateDOI.doi == pair.paper_doi
            )
            document_alternate_doi_model = session.exec(document_stmt).first()

            # Fast fail
            if document_alternate_doi_model is None:
                return False

            # Else get updated document model
            else:
                document_stmt = select(db_models.Document).where(
                    db_models.Document.id == document_alternate_doi_model.document_id
                )
                document_model = session.exec(document_stmt).first()

        # Code Host
        code_host_stmt = select(db_models.CodeHost).where(
            db_models.CodeHost.name == pair.repo_parts.host
        )
        code_host_model = session.exec(code_host_stmt).first()

        # Fast fail
        if code_host_model is None:
            return False

        # Repository
        repository_stmt = select(db_models.Repository).where(
            db_models.Repository.code_host_id == code_host_model.id,
            db_models.Repository.owner == pair.repo_parts.owner,
            db_models.Repository.name == pair.repo_parts.name,
        )
        repository_model = session.exec(repository_stmt).first()

        # Fast fail
        if repository_model is None:
            return False

        # Link
        link_stmt = select(db_models.DocumentRepositoryLink).where(
            db_models.DocumentRepositoryLink.document_id == document_model.id,
            db_models.DocumentRepositoryLink.repository_id == repository_model.id,
        )
        link_model = session.exec(link_stmt).first()

        # Final check
        return link_model is not None


def filter_stored_pairs(
    pairs: list[types.ExpandedRepositoryDocumentPair],
    use_prod: bool = False,
) -> list[types.ExpandedRepositoryDocumentPair]:
    # For each pair, check if it exists in the database
    unprocessed_pairs = [
        pair
        for pair in tqdm(
            pairs,
            desc="Filtering already stored pairs",
        )
        if not check_pair_exists(pair=pair, use_prod=use_prod)
    ]

    # Log remaining pairs and filtered out pairs
    print(f"Filtered out pairs: {len(pairs) - len(unprocessed_pairs)}")
    print(f"Remaining pairs: {len(unprocessed_pairs)}")

    return unprocessed_pairs


@dataclass
class HydratedAuthorDeveloperLink:
    id: int
    researcher_open_alex_id: str
    developer_account_username: str
    last_snowball_processed_datetime: datetime | None


def get_hydrated_author_developer_links(
    use_prod: bool = False,
    filter_datetime_difference: str | None = None,
    filter_confidence_threshold: float = 0.97,
    n: int | None = None,
) -> list[HydratedAuthorDeveloperLink]:
    """
    Get all researcher-developer account links with full model details.

    Parameters
    ----------
    use_prod: bool
        Whether to use production database
    filter_datetime_difference: str | None
        Optional time string (e.g. "365 days") to filter
        links that haven't been processed in this timeframe
    filter_confidence_threshold: float
        Confidence threshold to filter links (default 0.97)
    n: int | None
        Optional limit on the number of links to return
    """
    # Get the engine
    engine = get_engine(use_prod=use_prod)

    # Create a session
    with Session(engine) as session:
        # Get all the researcher-developer account links with joins
        stmt = (
            select(
                db_models.ResearcherDeveloperAccountLink,
                db_models.Researcher,
                db_models.DeveloperAccount,
            )
            .join(db_models.Researcher)
            .join(db_models.DeveloperAccount)
        )

        # Apply datetime filtering if specified
        if filter_datetime_difference is not None:
            cutoff_datetime = datetime.now() - parse_timedelta(filter_datetime_difference)
            stmt = stmt.where(
                or_(
                    is_(
                        db_models.ResearcherDeveloperAccountLink.last_snowball_processed_datetime,
                        None,
                    ),
                    (
                        col(
                            db_models.ResearcherDeveloperAccountLink.last_snowball_processed_datetime
                        )
                        < cutoff_datetime
                    ),
                ),
                (
                    col(db_models.ResearcherDeveloperAccountLink.predictive_model_confidence)
                    >= filter_confidence_threshold
                ),
            )

        # Limit results if specified
        if n is not None:
            stmt = stmt.limit(n)

        # Execute the query
        results = session.exec(stmt).all()

        # Convert to hydrated objects
        hydrated_links = []
        for link, researcher, developer_account in results:
            hydrated_links.append(
                HydratedAuthorDeveloperLink(
                    id=link.id,
                    researcher_open_alex_id=researcher.open_alex_id,
                    developer_account_username=developer_account.username,
                    last_snowball_processed_datetime=link.last_snowball_processed_datetime,
                )
            )

        return hydrated_links


def check_article_repository_pair_already_in_db(
    article_doi: str,
    article_title: str,
    code_host: str,
    repo_owner: str,
    repo_name: str,
    use_prod: bool = False,
) -> bool:
    # Get the engine
    engine = get_engine(use_prod=use_prod)

    # Lowercase and strip all inputs
    article_doi = article_doi.lower().strip()
    code_host = code_host.lower().strip()
    repo_owner = repo_owner.lower().strip()
    repo_name = repo_name.lower().strip()

    # Create a session
    with Session(engine) as session:
        # Document
        document_stmt = select(db_models.Document).where(db_models.Document.doi == article_doi)
        document_model = session.exec(document_stmt).first()

        # Check for document alternate doi
        if document_model is None:
            document_stmt = select(db_models.DocumentAlternateDOI).where(
                db_models.DocumentAlternateDOI.doi == article_doi
            )
            document_alternate_doi_model = session.exec(document_stmt).first()

            # Fast fail
            if document_alternate_doi_model is None:
                return False

            # Else get updated document model
            else:
                document_stmt = select(db_models.Document).where(
                    db_models.Document.id == document_alternate_doi_model.document_id
                )
                document_model = session.exec(document_stmt).first()

        # Now try and match on title if we still don't have a document model
        if document_model is None:
            document_stmt = select(db_models.Document).where(
                db_models.Document.title == article_title.strip()
            )
            document_model = session.exec(document_stmt).first()

        # Fast fail
        if document_model is None:
            return False

        # Code Host (assume GitHub for now)
        code_host_stmt = select(db_models.CodeHost).where(db_models.CodeHost.name == code_host)
        code_host_model = session.exec(code_host_stmt).first()

        # Fast fail
        if code_host_model is None:
            return False

        # Repository
        repository_stmt = select(db_models.Repository).where(
            db_models.Repository.code_host_id == code_host_model.id,
            db_models.Repository.owner == repo_owner,
            db_models.Repository.name == repo_name,
        )
        repository_model = session.exec(repository_stmt).first()

        # Fast fail
        if repository_model is None:
            return False

        # Link
        link_stmt = select(db_models.DocumentRepositoryLink).where(
            db_models.DocumentRepositoryLink.document_id == document_model.id,
            db_models.DocumentRepositoryLink.repository_id == repository_model.id,
        )
        link_model = session.exec(link_stmt).first()

        # Final check
        return link_model is not None
