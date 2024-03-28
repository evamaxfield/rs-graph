#!/usr/bin/env python

from __future__ import annotations

from typing import Protocol
from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin

###############################################################################


@dataclass
class RepositoryDocumentPair(DataClassJsonMixin):
    source: str
    repo_url: str
    paper_doi: str
    paper_extra_data: dict | None = None


class DataSource(Protocol):
    
    @staticmethod
    def get_dataset(
        raise_on_error: bool = True,
        **kwargs,
    ) -> list[RepositoryDocumentPair]:
        raise NotImplementedError()