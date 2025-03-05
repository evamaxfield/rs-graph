#!/usr/bin/env python

from __future__ import annotations

import traceback

from sci_soft_models import soft_search

from .. import types

###############################################################################


def predict_software_production_from_grant(
    grant_details: types.GrantDetails,
) -> types.GrantDetailsWithSoftwareProductionPrediction | types.ErrorResult:
    try:
        # Get the model details
        model_details = soft_search.get_model_details()

        # Predict
        prediction = soft_search.predict_software_production_from_awards(
            awards=[
                soft_search.AwardDetails(
                    id_=grant_details.grant_id,
                    title=grant_details.title,
                    abstract=grant_details.abstract,
                    outcomes_report=grant_details.outcomes_report,
                ),
            ],
        )

        # Return
        return types.GrantDetailsWithSoftwareProductionPrediction(
            funder=grant_details.funder,
            grant_id=grant_details.grant_id,
            title=grant_details.title,
            abstract=grant_details.abstract,
            outcomes_report=grant_details.outcomes_report,
            directorate=grant_details.directorate,
            start_date=grant_details.start_date,
            end_date=grant_details.end_date,
            funded_amount=grant_details.funded_amount,
            primary_investigator=grant_details.primary_investigator,
            institution=grant_details.institution,
            software_production_prediction=prediction[0].software_production_prediction,
            predictive_model_name=model_details.name,
            predictive_model_version=model_details.version,
            predictive_model_confidence=prediction[0].confidence,
        )

    except Exception as e:
        return types.ErrorResult(
            source=grant_details.funder,
            step="software-production-prediction",
            identifier=grant_details.grant_id,
            error=str(e),
            traceback=traceback.format_exc(),
        )
