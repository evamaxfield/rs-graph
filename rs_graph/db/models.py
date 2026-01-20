#!/usr/bin/env python

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import Column, DateTime, func
from sqlmodel import Field, SQLModel, UniqueConstraint

# Define naming convention for all constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Apply the naming convention to SQLModel's metadata
SQLModel.metadata.naming_convention = convention


class StripMixin:
    def __init__(self, **data: Any):
        for field, value in data.items():
            # Handle AttrDict and empty dicts
            if isinstance(value, dict) and len(value) == 0:
                data[field] = None
            elif hasattr(value, "__class__") and value.__class__.__name__ == "AttrDict":
                data[field] = None
            elif isinstance(value, str):
                data[field] = value.strip()
        super().__init__(**data)


class StrippedSQLModel(StripMixin, SQLModel):
    pass


###############################################################################


class DatasetSource(StrippedSQLModel, table=True):
    """Stores the basic information for a dataset source."""

    __tablename__ = "dataset_source"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class Source(StrippedSQLModel, table=True):
    """Stores the basic information about a possible document source."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    open_alex_id: str = Field(unique=True)

    # Data
    source_type: str = Field(index=True)
    host_organization_name: str | None = Field(index=True)
    host_organization_open_alex_id: str | None = None

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class Location(StrippedSQLModel, table=True):
    """Stores the basic information about a document's possible location."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    landing_page_url: str | None = Field(index=True)
    pdf_url: str | None = Field(index=True)
    source_id: int | None = Field(
        foreign_key="source.id",
        index=True,
        nullable=True,
        ondelete="CASCADE",
    )

    __table_args__ = (UniqueConstraint("landing_page_url", "pdf_url", "source_id"),)

    # Data
    is_open_access: bool = Field(index=True)
    license: str | None = Field(index=True)
    version: str | None = Field(index=True)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class Document(StrippedSQLModel, table=True):
    """Stores paper, report, or other academic document details."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    doi: str = Field(unique=True)

    # Data
    open_alex_id: str
    title: str
    publication_date: date = Field(index=True)
    cited_by_count: int = Field(index=True)
    fwci: float | None = Field(index=True, nullable=True)
    citation_normalized_percentile: float | None = Field(index=True, nullable=True)
    document_type: str = Field(index=True)
    is_open_access: bool = Field(index=True)
    open_access_status: str = Field(index=True)

    # Foreign Keys
    primary_location_id: int | None = Field(
        foreign_key="location.id",
        index=True,
        nullable=True,
        ondelete="CASCADE",
    )
    best_open_access_location_id: int | None = Field(
        foreign_key="location.id",
        index=True,
        nullable=True,
        ondelete="CASCADE",
    )

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class DocumentAlternateDOI(StrippedSQLModel, table=True):
    """
    Stores alternate DOIs for a document.

    "Alternate" DOIs are DOIs that were previously associated with a document
    which we have resolved to a more recent version.
    """

    __tablename__ = "document_alternate_doi"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(
        foreign_key="document.id",
        index=True,
        ondelete="CASCADE",
    )
    doi: str = Field(unique=True)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class DocumentAbstract(StrippedSQLModel, table=True):
    """Stores the abstract for a document."""

    __tablename__ = "document_abstract"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(
        foreign_key="document.id",
        index=True,
        ondelete="CASCADE",
    )

    __table_args__ = (UniqueConstraint("document_id"),)

    # Data
    content: str | None = None

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class Topic(StrippedSQLModel, table=True):
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
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class DocumentTopic(StrippedSQLModel, table=True):
    """Stores the connection between a document and a topic."""

    __tablename__ = "document_topic"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(
        foreign_key="document.id",
        index=True,
        ondelete="CASCADE",
    )
    topic_id: int = Field(
        foreign_key="topic.id",
        index=True,
        ondelete="CASCADE",
    )

    __table_args__ = (UniqueConstraint("document_id", "topic_id"),)

    # Data
    score: float = Field(index=True)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class Researcher(StrippedSQLModel, table=True):
    """Stores researcher details."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)
    orcid: str | None = Field(index=True, nullable=True)

    # Data
    name: str
    works_count: int = Field(index=True)
    cited_by_count: int = Field(index=True)
    h_index: int = Field(index=True)
    i10_index: int = Field(index=True)
    two_year_mean_citedness: float = Field(index=True)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class DocumentContributor(StrippedSQLModel, table=True):
    """Stores the connection between a researcher and a document."""

    __tablename__ = "document_contributor"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    researcher_id: int = Field(
        foreign_key="researcher.id",
        index=True,
        ondelete="CASCADE",
    )
    document_id: int = Field(
        foreign_key="document.id",
        index=True,
        ondelete="CASCADE",
    )

    __table_args__ = (UniqueConstraint("researcher_id", "document_id"),)

    # Data
    position: str = Field(index=True)
    is_corresponding: bool = Field(index=True)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class Institution(StrippedSQLModel, table=True):
    """Stores institution details."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)

    # Data
    name: str = Field(index=True)
    country_code: str | None = Field(index=True, nullable=True)
    institution_type: str | None = Field(index=True, nullable=True)
    ror: str | None = Field(index=True, nullable=True)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class DocumentContributorInstitution(StrippedSQLModel, table=True):
    """Stores the connection between a researcher, document, and institution."""

    __tablename__ = "document_contributor_institution"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_contributor_id: int = Field(
        foreign_key="document_contributor.id",
        index=True,
        ondelete="CASCADE",
    )
    institution_id: int = Field(
        foreign_key="institution.id",
        index=True,
        ondelete="CASCADE",
    )

    __table_args__ = (UniqueConstraint("document_contributor_id", "institution_id"),)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class Funder(StrippedSQLModel, table=True):
    """Stores the basic information for a funding source (e.g. NSF, NIH)."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    open_alex_id: str = Field(unique=True)

    # Data
    name: str = Field(index=True)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class FundingInstance(StrippedSQLModel, table=True):
    """Stores the basic information for a single funding instance (e.g. grant)."""

    __tablename__ = "funding_instance"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    funder_id: int = Field(
        foreign_key="funder.id",
        index=True,
        ondelete="CASCADE",
    )
    award_id: str = Field(index=True)

    __table_args__ = (UniqueConstraint("funder_id", "award_id"),)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class DocumentFundingInstance(StrippedSQLModel, table=True):
    """Stores the connection between a document and a funding instance."""

    __tablename__ = "document_funding_instance"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(
        foreign_key="document.id",
        index=True,
        ondelete="CASCADE",
    )
    funding_instance_id: int = Field(
        foreign_key="funding_instance.id",
        index=True,
        ondelete="CASCADE",
    )

    __table_args__ = (UniqueConstraint("document_id", "funding_instance_id"),)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class CodeHost(StrippedSQLModel, table=True):
    """Stores the basic information for a code host (e.g. GitHub, GitLab)."""

    __tablename__ = "code_host"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)

    # Data
    name: str = Field(unique=True, index=True)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class Repository(StrippedSQLModel, table=True):
    """Stores the basic information for a repository."""

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    code_host_id: int = Field(
        foreign_key="code_host.id",
        index=True,
        ondelete="CASCADE",
    )
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
    commits_count: int | None = Field(index=True)
    size_kb: int
    topics: str | None = Field(nullable=True)
    primary_language: str | None = Field(index=True, nullable=True)
    default_branch: str | None = Field(index=True, nullable=True)
    license: str | None = Field(index=True, nullable=True)
    processed_at_sha: str | None = Field(index=True, nullable=True)
    creation_datetime: datetime = Field(index=True)
    last_pushed_datetime: datetime = Field(index=True)

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class RepositoryReadme(StrippedSQLModel, table=True):
    """Stores the readme for a repository."""

    __tablename__ = "repository_readme"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(
        foreign_key="repository.id",
        index=True,
        ondelete="CASCADE",
    )

    __table_args__ = (UniqueConstraint("repository_id"),)

    # Data
    content: str | None = None

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class RepositoryLanguage(StrippedSQLModel, table=True):
    """Stores the connection between a repository and a language."""

    __tablename__ = "repository_language"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(
        foreign_key="repository.id",
        index=True,
        ondelete="CASCADE",
    )
    language: str = Field(index=True)

    __table_args__ = (UniqueConstraint("repository_id", "language"),)

    # Data
    bytes_of_code: int

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class RepositoryFile(StrippedSQLModel, table=True):
    """Stores the connection between a repository and a file."""

    __tablename__ = "repository_file"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(
        foreign_key="repository.id",
        index=True,
        ondelete="CASCADE",
    )
    path: str = Field(index=True)

    __table_args__ = (UniqueConstraint("repository_id", "path"),)

    # Data
    tree_type: str  # e.g. "blob" for files, "tree" for directories
    bytes_of_code: int

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class DeveloperAccount(StrippedSQLModel, table=True):
    """Stores the basic information for a developer account (e.g. GitHub, GitLab)."""

    __tablename__ = "developer_account"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    code_host_id: int = Field(
        foreign_key="code_host.id",
        index=True,
        ondelete="CASCADE",
    )
    username: str

    __table_args__ = (UniqueConstraint("code_host_id", "username"),)

    # Data
    name: str | None = None
    email: str | None = None

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class RepositoryContributor(StrippedSQLModel, table=True):
    """Stores the connection between a repository and a contributor."""

    __tablename__ = "repository_contributor"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    repository_id: int = Field(
        foreign_key="repository.id",
        index=True,
        ondelete="CASCADE",
    )
    developer_account_id: int = Field(
        foreign_key="developer_account.id",
        index=True,
        ondelete="CASCADE",
    )

    __table_args__ = (UniqueConstraint("repository_id", "developer_account_id"),)

    # TODO: add "total commits" field

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


# TODO: Store basic repository commit details
# https://docs.github.com/en/rest/metrics/statistics?apiVersion=2022-11-28#get-all-contributor-commit-activity
# I.e. WeeklyRepositoryCommitActivity
# includes reference to repository contributor
# includes date and additions, deletions, and number of commits for that week


class DocumentRepositoryLink(StrippedSQLModel, table=True):
    """Stores the connection between a document and a repository."""

    __tablename__ = "document_repository_link"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(
        foreign_key="document.id",
        index=True,
        ondelete="CASCADE",
    )
    repository_id: int = Field(
        foreign_key="repository.id",
        index=True,
        ondelete="CASCADE",
    )

    __table_args__ = (UniqueConstraint("document_id", "repository_id"),)

    # Data
    dataset_source_id: int = Field(
        foreign_key="dataset_source.id",
        index=True,
        ondelete="CASCADE",
    )
    predictive_model_name: str | None = None
    predictive_model_version: str | None = None
    predictive_model_confidence: float | None = None

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))


class ResearcherDeveloperAccountLink(StrippedSQLModel, table=True):
    """Stores the connection between a researcher and a developer account."""

    __tablename__ = "researcher_developer_account_link"

    # Primary Keys / Uniqueness
    id: int | None = Field(default=None, primary_key=True)
    researcher_id: int = Field(
        foreign_key="researcher.id",
        index=True,
        ondelete="CASCADE",
    )
    developer_account_id: int = Field(
        foreign_key="developer_account.id",
        index=True,
        ondelete="CASCADE",
    )

    __table_args__ = (UniqueConstraint("researcher_id", "developer_account_id"),)

    # Data
    predictive_model_name: str | None = None
    predictive_model_version: str | None = None
    predictive_model_confidence: float | None = None
    last_snowball_processed_datetime: datetime | None = Field(
        default=None,
        index=True,
        nullable=True,
    )

    # Updates
    created_datetime: datetime = Field(sa_column=Column(DateTime(), server_default=func.now()))
    updated_datetime: datetime = Field(sa_column=Column(DateTime(), onupdate=func.now()))
