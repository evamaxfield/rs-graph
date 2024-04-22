#!/usr/bin/env python

from prefect import task
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, UniqueConstraint, create_engine, select

from .. import types
from . import constants

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


def store_full_details(
    pair: types.ExpandedRepositoryDocumentPair,
    prod: bool = False,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    # Get the engine
    engine = get_engine(prod=prod)

    # Create a session
    with Session(engine) as session:
        try:
            # Assert that everything is in the correct state
            assert pair.source_model is not None
            assert pair.document_model is not None

            # Work through document results
            # try:
            # Dataset source
            session.add(pair.source_model)
            session.flush()
            print(pair.source_model)

            # Document (update dataset source)
            assert pair.source_model.id is not None
            pair.document_model.dataset_source_id = pair.source_model.id
            print(pair.document_model)
            session.add(pair.document_model)
            session.flush()
            print(pair.document_model)

            session.commit()
        except Exception:
            session.rollback()

    return pair


@task
def store_full_details_task(
    pair: types.ExpandedRepositoryDocumentPair | types.ErrorResult,
    prod: bool = False,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    return store_full_details(pair=pair, prod=prod)
