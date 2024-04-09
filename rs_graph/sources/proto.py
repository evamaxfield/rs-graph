#!/usr/bin/env python

from __future__ import annotations

from typing import Protocol

from ..types import SuccessAndErroredResultsLists

###############################################################################


class DataSource(Protocol):
    @staticmethod
    def get_dataset(
        **kwargs: dict[str, str],
    ) -> SuccessAndErroredResultsLists:
        """Download the dataset."""
        raise NotImplementedError()
