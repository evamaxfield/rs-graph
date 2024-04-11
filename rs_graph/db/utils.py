#!/usr/bin/env python

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine, select

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

    # Get required fields for model
    required_fields = model.model_json_schema()["required"]

    # Remove created_datetime and updated_datetime
    required_fields = [
        field
        for field in required_fields
        if field not in ["created_datetime", "updated_datetime"]
    ]

    # Try and select a matching model
    query = select(model_cls)
    for field in required_fields:
        query = query.where(getattr(model_cls, field) == getattr(model, field))

    # Execute the query
    result = session.exec(query).first()

    return result


def add_model(model: SQLModel, session: Session) -> SQLModel:
    session.add(model)
    session.commit()
    session.refresh(model)

    return model


def get_or_add(model: SQLModel, session: Session) -> SQLModel:
    # Check if the model exists
    result = get_unique_first_model(model=model, session=session)

    # If the model exists, return it
    if result:
        return result

    # Otherwise, add the model
    return add_model(model=model, session=session)
