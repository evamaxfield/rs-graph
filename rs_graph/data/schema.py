#!/usr/bin/env python

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pydantic import BaseModel

###############################################################################

class Repository(BaseModel):
    """
    Stores the basic information for a source code repository.
    """
    uuid: str
    ecosystems_uuid: str
    url: str
    host: str
    name: str
    description: str | None
    star_count: int
    fork_count: int
    watcher_count: int
    issue_count: int
    pull_request_count: int
    code_quality: float
    development_distribution_score: float
    readme_uri: str
    license: str | None
    source_code_embedding_uri: str
    source_code_embedding_model_name: str
    source_code_embedding_model_version: str
    cache_datetime: datetime

class Developer(BaseModel):
    """
    Stores the basic information for a developer account.
    """
    uuid: str
    ecosystems_uuid: str
    host: str
    username: str
    name: str | None
    email: str | None
    cache_datetime: datetime

class Commit(BaseModel):
    """
    Stores commit details connected to a repository and developer.
    """
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
    """
    Stores the connection between a repository and a developer.
    """
    repository_uuid: Repository
    developer_uuid: Developer
    cache_datetime: datetime

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
    primary_field: str
    secondary_field: str | None
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
    position: str
    model_name: str | None
    model_version: str | None
    credit_roles: str

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
    source: str
    model_name: str | None
    model_version: str | None

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
    model_name: str | None
    model_version: str | None

class FundingSource(BaseModel):
    """
    Stores the basic information for a funding source (e.g. NSF, NIH).
    """
    uuid: str
    name: str

class FundingInstance(BaseModel):
    """
    Stores the basic information for a single funding instance (e.g. grant).
    """
    uuid: str
    funding_source_uuid: str
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
    source: str
    model_name: str | None
    model_version: str | None

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
    source: str
    model_name: str | None
    model_version: str | None

###############################################################################
    
if __name__ == "__main__":
    import erdantic as erd

    # Get all model base classes dynamically
    models = [
        model for model in locals().values()
        if isinstance(model, type) and issubclass(model, BaseModel)
    ]

    # Remove BaseModel from the list
    models.remove(BaseModel)

    # Call model_rebuild for all models
    for model in models:
        model.update_forward_refs(**locals())

    # Create diagram
    diagram = erd.create(Repository)
    diagram.draw("rs-graph-schema.png")