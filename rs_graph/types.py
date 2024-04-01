#!/usr/bin/env python

from dataclasses import dataclass

import pandas as pd
from dataclasses_json import DataClassJsonMixin

###############################################################################


@dataclass
class ErrorResult(DataClassJsonMixin):
    source: str
    identifier: str
    step: str
    error: str


@dataclass
class SuccessAndErroredResults:
    successful_results: pd.DataFrame
    errored_results: pd.DataFrame


@dataclass
class DeveloperDetails:
    username: str
    name: str | None
    email: str | None
