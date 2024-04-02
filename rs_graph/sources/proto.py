#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
        use_dask: bool = False,
        **kwargs: dict[str, str],
    ) -> list[RepositoryDocumentPair]:
        """Download the dataset."""
        raise NotImplementedError()
