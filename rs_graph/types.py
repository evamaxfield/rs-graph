#!/usr/bin/env python

from dataclasses import dataclass

import pandas as pd
from dataclasses_json import DataClassJsonMixin
from sci_soft_models import binary_article_repo_em

from .db import models as db_models

###############################################################################
# General Types


@dataclass
class ErrorResult(DataClassJsonMixin):
    source: str
    step: str
    identifier: str
    error: str
    traceback: str


@dataclass
class SuccessAndErroredResults:
    successful_results: pd.DataFrame
    errored_results: pd.DataFrame


###############################################################################
# Prelinked Ingestion Pipeline Types


@dataclass
class BasicRepositoryDocumentPair(DataClassJsonMixin):
    source: str
    repo_url: str
    paper_doi: str
    paper_extra_data: dict | None = None


@dataclass
class CodeHostResult:
    host: str
    owner: str | None
    name: str | None


@dataclass
class RepoParts(DataClassJsonMixin):
    host: str
    owner: str
    name: str


@dataclass
class TopicDetails(DataClassJsonMixin):
    topic_model: db_models.Topic
    document_topic_model: db_models.DocumentTopic


@dataclass
class ResearcherDetails(DataClassJsonMixin):
    researcher_model: db_models.Researcher
    document_contributor_model: db_models.DocumentContributor
    institution_models: list[db_models.Institution]


@dataclass
class FundingInstanceDetails(DataClassJsonMixin):
    funder_model: db_models.Funder
    funding_instance_model: db_models.FundingInstance


@dataclass
class OpenAlexResultModels(DataClassJsonMixin):
    dataset_source_model: db_models.DatasetSource
    primary_document_source_model: db_models.Source | None
    primary_location_model: db_models.Location | None
    best_oa_document_source_model: db_models.Source | None
    best_oa_location_model: db_models.Location | None
    document_model: db_models.Document
    document_abstract_model: db_models.DocumentAbstract
    document_alternate_dois: list[str]
    topic_details: list[TopicDetails]
    researcher_details: list[ResearcherDetails] | None
    funding_instance_details: list[FundingInstanceDetails] | None


@dataclass
class RepositoryContributorDetails(DataClassJsonMixin):
    developer_account_model: db_models.DeveloperAccount
    repository_contributor_model: db_models.RepositoryContributor


@dataclass
class GitHubResultModels(DataClassJsonMixin):
    code_host_model: db_models.CodeHost
    repository_model: db_models.Repository
    repository_readme_model: db_models.RepositoryReadme | None
    repository_language_models: list[db_models.RepositoryLanguage] | None
    repository_contributor_details: list[RepositoryContributorDetails] | None
    repository_file_models: list[db_models.RepositoryFile] | None


@dataclass
class DocumentRepositoryLinkMetadata(DataClassJsonMixin):
    model_name: str
    model_version: str
    model_confidence: float


@dataclass
class ExpandedRepositoryDocumentPair(DataClassJsonMixin):
    # Required / Basic
    source: str
    paper_doi: str
    paper_extra_data: dict | None = None

    # Added after processing
    repo_parts: RepoParts | None = None
    open_alex_results: OpenAlexResultModels | None = None
    github_results: GitHubResultModels | None = None

    # Processing metadata
    snowball_sampling_discovery_source_author_developer_link_id: int | None = None
    document_repository_link_metadata: DocumentRepositoryLinkMetadata | None = None
    open_alex_processing_time_seconds: float | None = None
    github_processing_time_seconds: float | None = None


@dataclass
class StoredRepositoryDocumentPair(DataClassJsonMixin):
    # Required / Basic
    dataset_source_model: db_models.DatasetSource
    document_model: db_models.Document
    repository_model: db_models.Repository
    developer_account_models: list[db_models.DeveloperAccount]
    researcher_models: list[db_models.Researcher]

    # Added after processing
    researcher_developer_links: list[db_models.ResearcherDeveloperAccountLink] | None = None

    # Processing metadata
    snowball_sampling_discovery_source_author_developer_link_id: int | None = None
    document_repository_link_metadata: DocumentRepositoryLinkMetadata | None = None
    open_alex_processing_time_seconds: float | None = None
    github_processing_time_seconds: float | None = None
    store_article_and_repository_time_seconds: float | None = None
    author_developer_matching_time_seconds: float | None = None
    store_author_developer_links_time_seconds: float | None = None


@dataclass
class SuccessAndErroredResultsLists(DataClassJsonMixin):
    successful_results: list[BasicRepositoryDocumentPair | ExpandedRepositoryDocumentPair]
    errored_results: list[ErrorResult]


###############################################################################
# Types for article-repository discovery


@dataclass
class AuthorArticleDetails(DataClassJsonMixin):
    author_developer_link_id: int
    researcher_open_alex_id: str
    open_alex_results_models: OpenAlexResultModels


@dataclass
class DeveloperRepositoryDetails(DataClassJsonMixin):
    author_developer_link_id: int
    developer_account_username: str
    github_result_models: GitHubResultModels


@dataclass
class FilteredResult(DataClassJsonMixin):
    source: str
    identifier: str
    reason: str


@dataclass
class UncheckedPossibleAuthorArticleAndDeveloperRepositoryPair(DataClassJsonMixin):
    author_developer_link_id: int
    article_doi: str
    repository_identifier: str
    author_article: AuthorArticleDetails
    developer_repository: DeveloperRepositoryDetails


@dataclass
class AuthorArticleAndDeveloperRepositoryPairPreppedForMatching(DataClassJsonMixin):
    author_developer_link_id: int
    article_doi: str
    repository_identifier: str
    author_article: AuthorArticleDetails
    developer_repository: DeveloperRepositoryDetails
    inference_ready_pair: binary_article_repo_em.InferenceReadyArticleRepositoryPair


@dataclass
class MatchedAuthorArticleAndDeveloperRepositoryPair(DataClassJsonMixin):
    author_developer_link_id: int
    article_doi: str
    repository_identifier: str
    author_article: AuthorArticleDetails
    developer_repository: DeveloperRepositoryDetails
    matched_details: binary_article_repo_em.MatchedArticleRepository
