#!/usr/bin/env python

from __future__ import annotations

import traceback

from sci_soft_models import soft_search

from .. import types
from ..db import models as db_models

###############################################################################


def predict_software_production_from_grant(
    grant_details: types.GrantDetails,
) -> types.GrantDetailsWithSoftwareProductionPrediction | types.ErrorResult:
    try:
        # TODO:
        # lots of soft search implementation stuff here

        # Get the model details
        model_details = soft_search.get_model_details()

        # Create the input types and predict
        # ...

    except Exception as e:
        return types.ErrorResult(
            source=grant_details.funder,
            step="software-production-prediction",
            identifier=grant_details.grant_id,
            error=str(e),
            traceback=traceback.format_exc(),
        )
