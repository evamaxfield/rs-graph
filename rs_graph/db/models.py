#!/usr/bin/env python

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import Column, DateTime, func
from sqlmodel import Field, SQLModel, UniqueConstraint


class StripMixin:
    def __init__(self, **data: Any):
        for field, value in data.items():
            if isinstance(value, str):
                data[field] = value.strip()
        super().__init__(**data)


class StrippedSQLModel(StripMixin, SQLModel):
    pass


###############################################################################


class DatasetSource(StrippedSQLModel, table=True):  # type: ignore
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


class Document(StrippedSQLModel, table=True):  # type: ignore
    """Stores paper, report, or other academic document details."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    doi: str = Field(unique=True)

    # Data
    open_alex_id: str
    title: str
    publication_date: date = Field(index=True)
    cited_by_count: int = Field(index=True)
    cited_by_percentile_year_min: int = Field(index=True)
    cited_by_percentile_year_max: int = Field(index=True)
    fwci: float | None = Field(index=True, nullable=True)
    document_type: str = Field(index=True)
    is_open_access: bool = Field(index=True)
    open_access_status: str = Field(index=True)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DocumentAlternateDOI(StrippedSQLModel, table=True):  # type: ignore
    """
    Stores alternate DOIs for a document.

    "Alternate" DOIs are DOIs that were previously associated with a document
    which we have resolved to a more recent version.
    """

    __tablename__ = "document_alternate_doi"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    doi: str = Field(unique=True)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DocumentAbstract(StrippedSQLModel, table=True):  # type: ignore
    """Stores the abstract for a document."""

    __tablename__ = "document_abstract"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)

    __table_args__ = (UniqueConstraint("document_id"),)

    # Data
    content: str | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Topic(StrippedSQLModel, table=True):  # type: ignore
    """Stores the basic information for a topic."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)

    # Data
    name: str = Field(index=True)
    field_name: str = Field(index=True)
    field_open_alex_id: str
    subfield_name: str = Field(index=True)
    subfield_open_alex_id: str
    domain_name: str = Field(index=True)
    domain_open_alex_id: str

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DocumentTopic(StrippedSQLModel, table=True):  # type: ignore
    """Stores the connection between a document and a topic."""

    __tablename__ = "document_topic"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    topic_id: int = Field(foreign_key="topic.id", index=True)

    __table_args__ = (UniqueConstraint("document_id", "topic_id"),)

    # Data
    score: float = Field(index=True)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Researcher(StrippedSQLModel, table=True):  # type: ignore
    """Stores researcher details."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)

    # Data
    name: str
    works_count: int = Field(index=True)
    cited_by_count: int = Field(index=True)
    h_index: int = Field(index=True)
    i10_index: int = Field(index=True)
    two_year_mean_citedness: float = Field(index=True)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DocumentContributor(StrippedSQLModel, table=True):  # type: ignore
    """Stores the connection between a researcher and a document."""

    __tablename__ = "document_contributor"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    researcher_id: int = Field(foreign_key="researcher.id", index=True)
    document_id: int = Field(foreign_key="document.id", index=True)

    __table_args__ = (UniqueConstraint("researcher_id", "document_id"),)

    # Data
    position: str = Field(index=True)
    is_corresponding: bool = Field(index=True)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Institution(StrippedSQLModel, table=True):  # type: ignore
    """Stores institution details."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)

    # Data
    name: str = Field(index=True)
    ror: str

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DocumentContributorInstitution(StrippedSQLModel, table=True):  # type: ignore
    """Stores the connection between a researcher, document, and institution."""

    __tablename__ = "document_contributor_institution"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_contributor_id: int = Field(
        foreign_key="document_contributor.id", index=True
    )
    institution_id: int = Field(foreign_key="institution.id", index=True)

    __table_args__ = (UniqueConstraint("document_contributor_id", "institution_id"),)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Funder(StrippedSQLModel, table=True):  # type: ignore
    """Stores the basic information for a funding source (e.g. NSF, NIH)."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)

    # Data
    name: str = Field(index=True)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class FundingInstance(StrippedSQLModel, table=True):  # type: ignore
    """Stores the basic information for a single funding instance (e.g. grant)."""

    __tablename__ = "funding_instance"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    funder_id: int = Field(foreign_key="funder.id", index=True)
    award_id: str = Field(index=True)

    __table_args__ = (UniqueConstraint("funder_id", "award_id"),)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DocumentFundingInstance(StrippedSQLModel, table=True):  # type: ignore
    """Stores the connection between a document and a funding instance."""

    __tablename__ = "document_funding_instance"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    funding_instance_id: int = Field(foreign_key="funding_instance.id", index=True)

    __table_args__ = (UniqueConstraint("document_id", "funding_instance_id"),)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class CodeHost(StrippedSQLModel, table=True):  # type: ignore
    """Stores the basic information for a code host (e.g. GitHub, GitLab)."""

    __tablename__ = "code_host"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)

    # Data
    name: str = Field(unique=True, index=True)

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class Repository(StrippedSQLModel, table=True):  # type: ignore
    """Stores the basic information for a repository."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    code_host_id: int = Field(foreign_key="code_host.id", index=True)
    owner: str
    name: str

    __table_args__ = (UniqueConstraint("code_host_id", "owner", "name"),)

    # Data
    description: str | None = None
    is_fork: bool = Field(index=True)
    forks_count: int = Field(index=True)
    stargazers_count: int = Field(index=True)
    watchers_count: int = Field(index=True)
    open_issues_count: int = Field(index=True)
    size_kb: int
    creation_datetime: datetime = Field(index=True)
    last_pushed_datetime: datetime = Field(index=True)

    # TODO: add total commits field

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class RepositoryReadme(StrippedSQLModel, table=True):  # type: ignore
    """Stores the readme for a repository."""

    __tablename__ = "repository_readme"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id", index=True)

    __table_args__ = (UniqueConstraint("repository_id"),)

    # Data
    content: str | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class RepositoryLanguage(StrippedSQLModel, table=True):  # type: ignore
    """Stores the connection between a repository and a language."""

    __tablename__ = "repository_language"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id", index=True)
    language: str = Field(index=True)

    __table_args__ = (UniqueConstraint("repository_id", "language"),)

    # Data
    bytes_of_code: int

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class DeveloperAccount(StrippedSQLModel, table=True):  # type: ignore
    """Stores the basic information for a developer account (e.g. GitHub, GitLab)."""

    __tablename__ = "developer_account"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    code_host_id: int = Field(foreign_key="code_host.id", index=True)
    username: str

    __table_args__ = (UniqueConstraint("code_host_id", "username"),)

    # Data
    name: str | None = None
    email: str | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class RepositoryContributor(StrippedSQLModel, table=True):  # type: ignore
    """Stores the connection between a repository and a contributor."""

    __tablename__ = "repository_contributor"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id", index=True)
    developer_account_id: int = Field(foreign_key="developer_account.id", index=True)

    __table_args__ = (UniqueConstraint("repository_id", "developer_account_id"),)

    # TODO: add "total commits" field

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


# TODO: Store basic repository commit details
# https://docs.github.com/en/rest/metrics/statistics?apiVersion=2022-11-28#get-all-contributor-commit-activity
# I.e. WeeklyRepositoryCommitActivity
# includes reference to repository contributor
# includes date and additions, deletions, and number of commits for that week


class DocumentRepositoryLink(StrippedSQLModel, table=True):  # type: ignore
    """Stores the connection between a document and a repository."""

    __tablename__ = "document_repository_link"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    repository_id: int = Field(foreign_key="repository.id", index=True)

    __table_args__ = (UniqueConstraint("document_id", "repository_id"),)

    # Data
    dataset_source_id: int = Field(foreign_key="dataset_source.id", index=True)
    predictive_model_name: str | None = None
    predictive_model_version: str | None = None
    predictive_model_confidence: float | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )


class ResearcherDeveloperAccountLink(StrippedSQLModel, table=True):  # type: ignore
    """Stores the connection between a researcher and a developer account."""

    __tablename__ = "researcher_developer_account_link"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    researcher_id: int = Field(foreign_key="researcher.id", index=True)
    developer_account_id: int = Field(foreign_key="developer_account.id", index=True)

    __table_args__ = (UniqueConstraint("researcher_id", "developer_account_id"),)

    # Data
    predictive_model_name: str | None = None
    predictive_model_version: str | None = None
    predictive_model_confidence: float | None = None

    # Updates
    created_datetime: datetime = Field(
        sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_datetime: datetime = Field(
        sa_column=Column(DateTime(), onupdate=func.now())
    )
