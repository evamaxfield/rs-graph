#!/usr/bin/env python

import traceback

from prefect import task
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, UniqueConstraint, create_engine, select

from .. import types
from . import constants
from . import models as db_models

###############################################################################


def get_engine(prod: bool = False) -> Engine:
    if prod:
        return create_engine(f"sqlite:///{constants.PROD_DATABASE_FILEPATH}")
    else:
        return create_engine(f"sqlite:///{constants.DEV_DATABASE_FILEPATH}")


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
        {
            field.name
            for constraint in unique_constraints
            for field in constraint.columns
        }
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


def store_full_details(
    pair: types.ExpandedRepositoryDocumentPair,
    prod: bool = False,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    # Get the engine
    engine = get_engine(prod=prod)

    # Create a session
    with Session(engine) as session:
        try:
            assert pair.open_alex_results is not None
            assert pair.github_results is not None

            # Dataset source
            pair.open_alex_results.source_model = _get_or_add_and_flush(
                model=pair.open_alex_results.source_model, session=session
            )
            assert pair.open_alex_results.source_model.id is not None

            # Document (update dataset source)
            pair.open_alex_results.document_model.dataset_source_id = (
                pair.open_alex_results.source_model.id
            )
            pair.open_alex_results.document_model = _get_or_add_and_flush(
                model=pair.open_alex_results.document_model, session=session
            )
            assert pair.open_alex_results.document_model.id is not None

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
            for (
                funding_instance_detail
            ) in pair.open_alex_results.funding_instance_details:
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

            # Repository contributor details
            for (
                repo_contributor_detail
            ) in pair.github_results.repository_contributor_details:
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
                repo_contributor_detail.repository_contributor_model.developer_account_id = (  # noqa: E501
                    repo_contributor_detail.developer_account_model.id
                )
                repo_contributor_detail.repository_contributor_model = (
                    _get_or_add_and_flush(
                        model=repo_contributor_detail.repository_contributor_model,
                        session=session,
                    )
                )

            session.commit()
        except Exception as e:
            session.rollback()
            return types.ErrorResult(
                source=pair.source,
                step="store-full-details",
                identifier=(pair.paper_doi),
                error=str(e),
                traceback=traceback.format_exc(),
            )

    return pair


@task(
    retries=2,
    retry_delay_seconds=2,
)
def store_full_details_task(
    pair: types.ExpandedRepositoryDocumentPair | types.ErrorResult,
    prod: bool = False,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    return store_full_details(pair=pair, prod=prod)
