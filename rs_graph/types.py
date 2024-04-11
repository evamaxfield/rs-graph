#!/usr/bin/env python

from dataclasses import dataclass

import pandas as pd
from dataclasses_json import DataClassJsonMixin

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
class RepositoryDocumentPair(DataClassJsonMixin):
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
class ExpandedRepositoryDocumentPair(DataClassJsonMixin):
    source: str
    repo_host: str
    repo_owner: str
    repo_name: str
    paper_doi: str
    paper_extra_data: dict | None = None
    em_model_name: str | None = None
    em_model_version: str | None = None


@dataclass
class LinkedRepositoryDocumentPair(DataClassJsonMixin):
    source_id: int
    code_host_id: int
    repository_id: int
    document_id: int
    document_repository_link_id: int


@dataclass
class SuccessAndErroredResultsLists(DataClassJsonMixin):
    successful_results: list[
        RepositoryDocumentPair
        | ExpandedRepositoryDocumentPair
        | LinkedRepositoryDocumentPair
    ]
    errored_results: list[ErrorResult]
