#!/usr/bin/env python

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Column, DateTime, func
from sqlmodel import Field, SQLModel, UniqueConstraint

###############################################################################


class DatasetSource(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a dataset source."""

    __tablename__ = "dataset_source"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Document(SQLModel, table=True):  # type: ignore
    """Stores paper, report, or other academic document details."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    doi: str = Field(unique=True)

    # Data
    open_alex_id: str
    title: str
    publication_date: date
    cited_by_count: int
    cited_by_percentile_year_min: int
    cited_by_percentile_year_max: int
    dataset_source_id: int = Field(foreign_key="dataset_source.id")
    abstract: str | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Topic(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a topic."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)

    # Data
    name: str
    field_name: str
    field_open_alex_id: str
    subfield_name: str
    subfield_open_alex_id: str
    domain_name: str
    domain_open_alex_id: str

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DocumentTopic(SQLModel, table=True):  # type: ignore
    """Stores the connection between a document and a topic."""

    __tablename__ = "document_topic"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id")
    topic_id: int = Field(foreign_key="topic.id")

    __table_args__ = (UniqueConstraint("document_id", "topic_id"),)

    # Data
    score: float

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Researcher(SQLModel, table=True):  # type: ignore
    """Stores researcher details."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)

    # Data
    name: str
    works_count: int
    cited_by_count: int
    h_index: int
    i10_index: int
    two_year_mean_citedness: float

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class ResearcherDocument(SQLModel, table=True):  # type: ignore
    """Stores the connection between a researcher and a document."""

    __tablename__ = "researcher_document"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    researcher_id: int = Field(foreign_key="researcher.id")
    document_id: int = Field(foreign_key="document.id")

    __table_args__ = (UniqueConstraint("researcher_id", "document_id"),)

    # Data
    position: str
    is_corresponding: bool

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Institution(SQLModel, table=True):  # type: ignore
    """Stores institution details."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)

    # Data
    name: str
    ror: str

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class ResearcherDocumentInstitution(SQLModel, table=True):  # type: ignore
    """Stores the connection between a researcher, document, and institution."""

    __tablename__ = "researcher_document_institution"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    researcher_document_id: int = Field(foreign_key="researcher_document.id")
    institution_id: int = Field(foreign_key="institution.id")

    __table_args__ = (UniqueConstraint("researcher_document_id", "institution_id"),)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Funder(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a funding source (e.g. NSF, NIH)."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)

    # Data
    name: str

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class FundingInstance(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a single funding instance (e.g. grant)."""

    __tablename__ = "funding_instance"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    funder_id: int = Field(foreign_key="funder.id")
    award_id: str

    __table_args__ = (UniqueConstraint("funder_id", "award_id"),)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DocumentFundingInstance(SQLModel, table=True):  # type: ignore
    """Stores the connection between a document and a funding instance."""

    __tablename__ = "document_funding_instance"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id")
    funding_instance_id: int = Field(foreign_key="funding_instance.id")

    __table_args__ = (UniqueConstraint("document_id", "funding_instance_id"),)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )