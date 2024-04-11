#!/usr/bin/env python

from __future__ import annotations

from collections.abc import Callable

from ..types import SuccessAndErroredResultsLists

###############################################################################


DatasetRetrievalFunction = Callable[..., SuccessAndErroredResultsLists]
