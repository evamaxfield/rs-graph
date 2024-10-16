#!/usr/bin/env python

from dataclasses import dataclass

import pandas as pd
from dataclasses_json import DataClassJsonMixin

from .db import models as db_models

###############################################################################


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
    source_model: db_models.DatasetSource
    document_model: db_models.Document
    document_abstract_model: db_models.DocumentAbstract
    document_alternate_dois: list[str]
    topic_details: list[TopicDetails]
    researcher_details: list[ResearcherDetails]
    funding_instance_details: list[FundingInstanceDetails]


@dataclass
class RepositoryContributorDetails(DataClassJsonMixin):
    developer_account_model: db_models.DeveloperAccount
    repository_contributor_model: db_models.RepositoryContributor


@dataclass
class GitHubResultModels(DataClassJsonMixin):
    code_host_model: db_models.CodeHost
    repository_model: db_models.Repository
    repository_readme_model: db_models.RepositoryReadme
    repository_language_models: list[db_models.RepositoryLanguage]
    repository_contributor_details: list[RepositoryContributorDetails]


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


@dataclass
class StoredRepositoryDocumentPair(DataClassJsonMixin):
    # Required / Basic
    dataset_source_model: db_models.DatasetSource
    document_model: db_models.Document
    repository_model: db_models.Repository
    developer_account_models: list[db_models.DeveloperAccount]
    researcher_models: list[db_models.Researcher]

    # Added after processing
    researcher_developer_links: list[
        db_models.ResearcherDeveloperAccountLink
    ] | None = None


@dataclass
class SuccessAndErroredResultsLists(DataClassJsonMixin):
    successful_results: list[
        BasicRepositoryDocumentPair | ExpandedRepositoryDocumentPair
    ]
    errored_results: list[ErrorResult]
