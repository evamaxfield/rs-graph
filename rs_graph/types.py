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
class DeveloperDetails:
    username: str
    name: str | None
    email: str | None


@dataclass
class BasicRepositoryDocumentPair(DataClassJsonMixin):
    source: str
    repo_url: str
    paper_doi: str
    paper_extra_data: dict | None = None
    em_model_name: str | None = None
    em_model_version: str | None = None


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
    topic_model: db_models.Topic | None = None
    document_topic_model: db_models.DocumentTopic | None = None


@dataclass
class ResearcherDetails(DataClassJsonMixin):
    researcher_model: db_models.Researcher | None = None
    document_contributor_model: db_models.DocumentContributor | None = None
    institution_models: list[db_models.Institution] | None = None


@dataclass
class FundingInstanceDetails(DataClassJsonMixin):
    funder_model: db_models.Funder | None = None
    funding_instance_model: db_models.FundingInstance | None = None


@dataclass
class RepositoryContributorDetails(DataClassJsonMixin):
    developer_account_model: db_models.DeveloperAccount | None = None
    repository_contributor_model: db_models.RepositoryContributor | None = None


@dataclass
class ExpandedRepositoryDocumentPair(DataClassJsonMixin):
    # Required / Basic
    source: str
    paper_doi: str
    paper_extra_data: dict | None = None
    em_model_name: str | None = None
    em_model_version: str | None = None

    # Added after code host parsing
    repo_parts: RepoParts | None = None

    # Added after OpenAlex processing
    source_model: db_models.DatasetSource | None = None
    document_model: db_models.Document | None = None
    topic_details: list[TopicDetails] | None = None
    researcher_details: list[ResearcherDetails] | None = None
    funding_instance_details: list[FundingInstanceDetails] | None = None

    # Added after GitHub processing
    code_host_model: db_models.CodeHost | None = None
    repository_model: db_models.Repository | None = None
    repository_contributor_details: list[RepositoryContributorDetails] | None = None

    # Added after matching devs and researchers
    # linked_devs_and_researcher_details:


@dataclass
class SuccessAndErroredResultsLists(DataClassJsonMixin):
    successful_results: list[
        BasicRepositoryDocumentPair | ExpandedRepositoryDocumentPair
    ]
    errored_results: list[ErrorResult]
