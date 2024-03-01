#!/usr/bin/env python

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

###############################################################################

class Host(BaseModel):
    """Stores the basic information for a source code repository host."""

    uuid: str
    name: str

class Repository(BaseModel):
    """Stores the basic information for a source code repository."""

    uuid: str
    ecosystems_uuid: str
    url: str
    host: Host
    name: str
    description: str | None
    readme_uri: str
    license: str | None
    source_code_embedding_uri: str
    source_code_embedding_model_name: str
    source_code_embedding_model_version: str
    repository_created_datetime: datetime
    cache_datetime: datetime

class RepositoryMetrics(BaseModel):
    """
    Store the metrics for a repository.

    Notes
    -----
    This is stored as it's own table for simpler querying and because many
    (if not most) of these metrics come from AUGUR / CHAOSS.

    See Also
    --------
    AUGUR: https://github.com/chaoss/augur
    CHAOSS: https://chaoss.community/
    """

    repository_uuid: Repository

    # Popularity
    total_star_count: int
    mean_start_count_per_month: float
    median_star_count_per_month: float
    total_fork_count: int
    mean_fork_count_per_month: float
    median_fork_count_per_month: float

    # Scores
    elephant_factor: float
    """https://chaoss.community/kb/metric-elephant-factor/"""
    bus_factor: float
    """https://chaoss.community/kb/metric-bus-factor/"""
    license_coverage: float
    """https://chaoss.community/kb/metric-license-coverage/"""
    burstiness: float
    """https://chaoss.community/kb/metric-burstiness/"""
    development_distribution_score: float
    """https://report.opensustain.tech/chapters/development-distribution-score.html"""

    # Issues
    total_issue_count: int
    mean_new_issue_count_per_month: float
    """https://chaoss.community/kb/metric-issues-new/"""
    median_new_issue_count_per_month: float
    """https://chaoss.community/kb/metric-issues-new/"""
    mean_issues_closed_count_per_month: float
    """https://chaoss.community/kb/metric-issues-closed/"""
    median_issues_closed_count_per_month: float
    """https://chaoss.community/kb/metric-issues-closed/"""
    mean_issues_active_count_per_month: float
    """https://chaoss.community/kb/metric-issues-active/"""
    median_issues_active_count_per_month: float
    """https://chaoss.community/kb/metric-issues-active/"""
    mean_issue_response_time: float
    """https://chaoss.community/kb/metric-issue-response-time/"""
    median_issue_response_time: float
    """https://chaoss.community/kb/metric-issue-response-time/"""
    mean_issue_resolution_duration: float
    """https://chaoss.community/kb/metric-issue-resolution-duration/"""
    median_issue_resolution_duration: float
    """https://chaoss.community/kb/metric-issue-resolution-duration/"""
    mean_issue_age: float
    """https://chaoss.community/kb/metric-issue-age/"""
    median_issue_age: float
    """https://chaoss.community/kb/metric-issue-age/"""

    # Change Requests
    total_change_request_count: int
    """https://chaoss.community/kb/metric-change-requests/"""
    mean_new_change_requests_count_per_month: float
    """https://chaoss.community/kb/metric-change-requests-new/"""
    median_new_change_requests_count_per_month: float
    """https://chaoss.community/kb/metric-change-requests-new/"""
    mean_change_request_closed_count_per_month: float
    """https://chaoss.community/kb/metric-change-requests-closed/"""
    median_change_request_closed_count_per_month: float
    """https://chaoss.community/kb/metric-change-requests-closed/"""
    mean_change_request_reviews: float
    """https://chaoss.community/kb/metric-change-request-reviews/"""
    median_change_request_reviews: float
    """https://chaoss.community/kb/metric-change-request-reviews/"""
    mean_change_request_duration: float
    """https://chaoss.community/kb/metric-change-requests-duration/"""
    median_change_request_duration: float
    """https://chaoss.community/kb/metric-change-requests-duration/"""
    mean_change_request_review_duration: float
    """https://chaoss.community/kb/metric-change-request-review-duration/"""
    median_change_request_review_duration: float
    """https://chaoss.community/kb/metric-change-request-review-duration/"""
    mean_change_request_accepted: float
    """https://chaoss.community/kb/metric-change-requests-accepted/"""
    median_change_request_accepted: float
    """https://chaoss.community/kb/metric-change-requests-accepted/"""
    mean_change_request_declined: float
    """https://chaoss.community/kb/metric-change-requests-declined/"""
    median_change_request_declined: float
    """https://chaoss.community/kb/metric-change-requests-declined/"""
    change_request_closure_ratio: float
    """https://chaoss.community/kb/metric-change-request-closure-ratio/"""
    change_request_acceptance_ratio: float
    """https://chaoss.community/kb/metric-change-request-acceptance-ratio/"""
    mean_change_request_commits: float
    """https://chaoss.community/kb/metric-change-request-commits/"""
    median_change_request_commits: float
    """https://chaoss.community/kb/metric-change-request-commits/"""

    # Releases
    total_release_count: int
    release_frequency: float
    """https://chaoss.community/kb/metric-release-frequency/"""

    # Code Change
    mean_code_change_lines: float
    """https://chaoss.community/kb/metric-code-changes-lines/"""
    median_code_change_lines: float
    """https://chaoss.community/kb/metric-code-changes-lines/"""

    cache_datetime: datetime


class Developer(BaseModel):
    """Stores the basic information for a developer account."""

    uuid: str
    ecosystems_uuid: str
    host: Host
    username: str
    name: str | None
    email: str | None
    cache_datetime: datetime


class Commit(BaseModel):
    """Stores commit details connected to a repository and developer."""

    uuid: str
    repository_uuid: Repository
    developer_uuid: Developer
    hash: str
    message: str
    datetime: datetime
    committer_username: str | None
    committer_email: str | None
    code_positive_change: int
    code_negative_change: int
    code_abs_change: int
    filenames_changed: str


class RepositoryContributor(BaseModel):
    """Stores the connection between a repository and a developer."""

    repository_uuid: Repository
    developer_uuid: Developer
    cache_datetime: datetime

class Field(BaseModel):
    """Stores the basic information for a field of study."""

    uuid: str
    name: str

class Document(BaseModel):
    """
    Stores paper, report, or other academic document details with connections
    to Web of Science (WoS) and Semantic Scholar (S2) identifiers.
    """

    uuid: str
    wos_uuid: str
    s2_uuid: str
    title: str
    abstract: str
    keywords: str | None
    primary_field: Field
    secondary_field: Field | None
    published_datetime: datetime
    citation_count: int
    field_weighted_citation_impact: float | None
    cache_datetime: datetime


class Researcher(BaseModel):
    """
    Stores researcher details with connections to Web of Science (WoS) and
    Semantic Scholar (S2) identifiers.
    """

    uuid: str
    wos_uuid: str
    s2_uuid: str
    name: str
    email: str | None
    h_index: int
    cache_datetime: datetime

class Position(BaseModel):
    """Stores the basic information for a position (e.g. professor, postdoc)."""

    uuid: str
    name: str

class CreditRole(BaseModel):
    """Stores the basic information for a credit role (e.g. author, acknowledgee)."""

    uuid: str
    name: str

class ResearcherDocument(BaseModel):
    """
    Stores the connection between a researcher and a document.

    Notes
    -----
    This class additionally stores position in the document
    (e.g. author, acknowledgee, etc.). To extract acknowledgees, we will need
    an NER model to extract names from the document text and thus we also store
    the model name and version.
    """

    researcher_uuid: Researcher
    document_uuid: Document
    position: Position
    acknowledgement_extraction_model_name: str | None
    acknowledgement_extraction_model_version: str | None


class ResearcherDocumentCreditRole(BaseModel):
    """Stores the connection between a researcher, document, and credit role."""

    researcher_uuid: Researcher
    document_uuid: Document
    credit_role_uuid: CreditRole

class DatasetSource(BaseModel):
    """Stores the basic information for a dataset source."""

    uuid: str
    name: str

class RepositoryDocument(BaseModel):
    """
    Stores the connection between a repository and a document.

    Notes
    -----
    This class additionally stores the source of the connection and the model
    name and version used to extract the connection (if needed).
    """

    repository_uuid: Repository
    document_uuid: Document
    source: DatasetSource
    repository_document_match_model_name: str | None
    repository_document_match_model_version: str | None


class DeveloperResearcher(BaseModel):
    """
    Stores the connection between a developer and a researcher.

    Notes
    -----
    This class additionally stores the model name and version used to extract
    the connection (if needed).
    """

    developer_uuid: Developer
    researcher_uuid: Researcher
    developer_researcher_match_model_name: str | None
    developer_researcher_match_model_version: str | None


class FundingSource(BaseModel):
    """Stores the basic information for a funding source (e.g. NSF, NIH)."""

    uuid: str
    name: str


class FundingInstance(BaseModel):
    """Stores the basic information for a single funding instance (e.g. grant)."""

    uuid: str
    funding_source: FundingSource
    title: str
    abstract: str
    amount: float
    funding_start_datetime: datetime
    funding_end_datetime: datetime


class DocumentFundingInstance(BaseModel):
    """
    Stores the connection between a document and a funding instance.

    Notes
    -----
    This class additionally stores the source of the connection and the model
    name and version used to extract the connection (if needed).
    """

    document_uuid: Document
    funding_instance_uuid: FundingInstance
    source: DatasetSource
    funding_extraction_model_name: str | None
    funding_extraction_model_version: str | None


class RepositoryFundingInstance(BaseModel):
    """
    Stores the connection between a repository and a funding instance.

    Notes
    -----
    This class additionally stores the source of the connection and the model
    name and version used to extract the connection (if needed).
    """

    repository_uuid: Repository
    funding_instance_uuid: FundingInstance
    source: DatasetSource
    funding_extraction_model_name: str | None
    funding_extraction_model_version: str | None


###############################################################################

if __name__ == "__main__":
    import erdantic as erd

    # Get list of models
    models = [
        obj for obj in locals().values()
        # check if it is a class
        if isinstance(obj, type)
        and obj not in [datetime, BaseModel]
    ]
    
    # Call model rebuild on all models
    for model in models:
        model.model_rebuild()

    # Create diagram
    diagram = erd.create(*models)
    diagram.draw("rs-graph-schema.png")
