#!/usr/bin/env python

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Column, DateTime, func
from sqlmodel import Field, SQLModel, UniqueConstraint

###############################################################################


class CodeHost(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a source code repository host."""

    __tablename__ = "code_host"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    name: str
    url: str

    __table_args__ = (UniqueConstraint("name", "url"),)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Repository(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a source code repository."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    host_id: int = Field(foreign_key="code_host.id")
    name: str

    __table_args__ = (UniqueConstraint("host_id", "name"),)

    # Data
    readme_uri: str | None = None
    repository_created_datetime: datetime | None = None
    description: str | None = None
    license: str | None = None
    source_code_embedding_uri: str | None = None
    source_code_embedding_model_name: str | None = None
    source_code_embedding_model_version: str | None = None
    source_code_embedding_updated_datetime: datetime | None = None

    # External
    ecosystems_uuid: str | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Developer(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a developer account."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    host: int = Field(foreign_key="code_host.id")
    username: str

    __table_args__ = (UniqueConstraint("name", "username"),)

    # Data
    name: str | None = None
    email: str | None = None

    # External
    ecosystems_uuid: str | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Commit(SQLModel, table=True):  # type: ignore
    """Stores commit details connected to a repository and developer."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id")
    developer_id: int = Field(foreign_key="developer.id")
    hash: str

    __table_args__ = (UniqueConstraint("repository_id", "developer_id", "hash"),)

    # Data
    message: str
    commit_datetime: datetime
    committer_username: str
    committer_email: str
    code_positive_change: int
    code_negative_change: int
    code_abs_change: int
    filenames_changed: str

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class RepositoryContributor(SQLModel, table=True):  # type: ignore
    """Stores the connection between a repository and a developer."""

    __tablename__ = "repository_contributor"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id", unique=True)
    developer_id: int = Field(foreign_key="developer.id", unique=True)

    __table_args__ = (UniqueConstraint("repository_id", "developer_id"),)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class FieldOfStudy(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a field of study."""

    __tablename__ = "field_of_study"

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
    """
    Stores paper, report, or other academic document details with connections
    to Web of Science (WoS) and Semantic Scholar (S2) identifiers.
    """

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    doi: str = Field(unique=True)
    # wos_uuid: str = Field(unique=True)

    # Data
    title: str
    published_datetime: date
    # citation_count: int
    s2_uuid: str | None = None
    abstract: str | None = None
    keywords: str | None = None
    primary_field_id: int | None = Field(foreign_key="field_of_study.id")
    secondary_field_id: int | None = Field(foreign_key="field_of_study.id")
    field_weighted_citation_impact: float | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Researcher(SQLModel, table=True):  # type: ignore
    """
    Stores researcher details with connections to Web of Science (WoS) and
    Semantic Scholar (S2) identifiers.
    """

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    wos_uuid: str = Field(unique=True)

    # Data
    name: str
    email: str | None = None
    h_index: int | None = None
    s2_uuid: str | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Position(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a research position position."""

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


class CreditRole(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a credit role (e.g. author, acknowledgee)."""

    __tablename__ = "credit_role"

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


class ResearcherDocument(SQLModel, table=True):  # type: ignore
    """
    Stores the connection between a researcher and a document.

    Notes
    -----
    This class additionally stores position in the document
    (e.g. author, acknowledgee, etc.). To extract acknowledgees, we will need
    an NER model to extract names from the document text and thus we also store
    the model name and version.
    """

    __tablename__ = "researcher_document"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    researcher_id: int = Field(foreign_key="researcher.id")
    document_id: int = Field(foreign_key="document.id")
    position_id: int = Field(foreign_key="position.id")

    __table_args__ = (UniqueConstraint("researcher_id", "document_id", "position_id"),)

    # Data
    acknowledgement_extraction_model_name: str | None = None
    acknowledgement_extraction_model_version: str | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class ResearcherDocumentCreditRole(SQLModel, table=True):  # type: ignore
    """Stores the connection between a researcher, document, and credit role."""

    __tablename__ = "researcher_document_credit_role"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    researcher_id: int = Field(foreign_key="researcher.id")
    document_id: int = Field(foreign_key="document.id")
    credit_role_id: int = Field(foreign_key="credit_role.id")

    __table_args__ = (
        UniqueConstraint("researcher_id", "document_id", "credit_role_id"),
    )

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


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


class RepositoryDocument(SQLModel, table=True):  # type: ignore
    """
    Stores the connection between a repository and a document.

    Notes
    -----
    This class additionally stores the source of the connection and the model
    name and version used to extract the connection (if needed).
    """

    __tablename__ = "repository_document"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id")
    document_id: int = Field(foreign_key="document.id")
    source_id: int = Field(foreign_key="dataset_source.id")

    __table_args__ = (UniqueConstraint("repository_id", "document_id", "source_id"),)

    # Data
    repository_document_match_model_name: str | None = None
    repository_document_match_model_version: str | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DeveloperResearcher(SQLModel, table=True):  # type: ignore
    """
    Stores the connection between a developer and a researcher.

    Notes
    -----
    This class additionally stores the model name and version used to extract
    the connection (if needed).
    """

    __tablename__ = "developer_researcher"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    developer_id: int = Field(foreign_key="developer.id")
    researcher_id: int = Field(foreign_key="researcher.id")

    __table_args__ = (UniqueConstraint("developer_id", "researcher_id"),)

    # Data
    developer_researcher_match_model_name: str | None = None
    developer_researcher_match_model_version: str | None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class FundingSource(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a funding source (e.g. NSF, NIH)."""

    __tablename__ = "funding_source"

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


class FundingInstance(SQLModel, table=True):  # type: ignore
    """Stores the basic information for a single funding instance (e.g. grant)."""

    __tablename__ = "funding_instance"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    funding_source_id: int = Field(foreign_key="funding_source.id")
    title: str

    __table_args__ = (UniqueConstraint("funding_source_id", "title"),)

    # Data
    abstract: str | None
    amount: float
    funding_start_datetime: datetime
    funding_end_datetime: datetime

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DocumentFundingInstance(SQLModel, table=True):  # type: ignore
    """
    Stores the connection between a document and a funding instance.

    Notes
    -----
    This class additionally stores the source of the connection and the model
    name and version used to extract the connection (if needed).
    """

    __tablename__ = "document_funding_instance"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id")
    funding_instance_id: int = Field(foreign_key="funding_instance.id")
    source_id: int = Field(foreign_key="dataset_source.id")

    __table_args__ = (
        UniqueConstraint("document_id", "funding_instance_id", "source_id"),
    )

    # Data
    document_funding_match_model_name: str | None = None
    document_funding_match_model_version: str | None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class RepositoryFundingInstance(SQLModel, table=True):  # type: ignore
    """
    Stores the connection between a repository and a funding instance.

    Notes
    -----
    This class additionally stores the source of the connection and the model
    name and version used to extract the connection (if needed).
    """

    __tablename__ = "repository_funding_instance"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id")
    funding_instance_id: int = Field(foreign_key="funding_instance.id")
    source_id: int = Field(foreign_key="dataset_source.id")

    __table_args__ = (
        UniqueConstraint("repository_id", "funding_instance_id", "source_id"),
    )

    # Data
    repository_funding_match_model_name: str | None = None
    repository_funding_match_model_version: str | None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )
