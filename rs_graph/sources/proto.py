#!/usr/bin/env python

from __future__ import annotations

from typing import Protocol

from ..types import RepositoryDocumentPair

###############################################################################


class DataSource(Protocol):
    @staticmethod
    def get_dataset(
        use_dask: bool = False,
        **kwargs: dict[str, str],
    ) -> list[RepositoryDocumentPair]:
        """Download the dataset."""
        raise NotImplementedError()
