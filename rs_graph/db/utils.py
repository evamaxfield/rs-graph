#!/usr/bin/env python

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


def trans_add_model(model: SQLModel, session: Session) -> SQLModel:
    session.add(model)
    session.flush()

    return model


def get_or_trans_add(model: SQLModel, session: Session) -> SQLModel:
    # Check if the model exists
    result = get_unique_first_model(model=model, session=session)

    # If the model exists, return it
    if result:
        return result

    # Otherwise, add the model
    return trans_add_model(model=model, session=session)


def store_full_details(
    pair: types.ExpandedRepositoryDocumentPair,
    prod: bool = False,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    # Get the engine
    engine = get_engine(prod=prod)

    # Create a session
    with Session(engine) as session:
        # Work through document results
        # try:
        # Dataset source
        pair.source_model = get_or_trans_add(model=pair.source_model, session=session)

        # Document (update dataset source)
        pair.document_model.source_id = pair.source_model.id
        print(pair.document_model)
        pair.document_model = get_or_trans_add(model=pair.document_model, session=session)

        print(pair.document_model)

    return pair
