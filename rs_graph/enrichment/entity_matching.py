#!/usr/bin/env python

from __future__ import annotations

import traceback

from sci_soft_models import dev_author_em

from .. import types
from ..db import models as db_models

###############################################################################


def match_devs_and_researchers(
    pair: types.StoredRepositoryDocumentPair | types.ErrorResult,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    try:
        # Get the model details
        model_details = dev_author_em.get_model_details()

        # Convert db types to ml ready types and create LUTs
        dev_username_to_dev_details = {
            dev_account_model.username: (
                dev_author_em.DeveloperDetails(
                    username=dev_account_model.username,
                    name=dev_account_model.name,
                    email=dev_account_model.email,
                )
            )
            for dev_account_model in pair.developer_account_models
        }
        dev_username_to_dev_model = {
            dev_account_model.username: dev_account_model
            for dev_account_model in pair.developer_account_models
        }
        researcher_name_to_researcher_model = {
            researcher.name: researcher for researcher in pair.researcher_models
        }

        # Predict matches
        matches = dev_author_em.match_devs_and_authors(
            devs=list(dev_username_to_dev_details.values()),
            authors=list(researcher_name_to_researcher_model.keys()),
        )

        # Create the linked pairs
        linked_dev_researcher_pairs: list[db_models.ResearcherDeveloperAccountLink] = []
        for matched_dev_author in matches:
            # Find the matching db models
            dev_username = matched_dev_author.dev.username
            researcher_name = matched_dev_author.author

            # Create formal link
            linked_researcher_dev_account = db_models.ResearcherDeveloperAccountLink(
                researcher_id=researcher_name_to_researcher_model[researcher_name].id,
                developer_account_id=dev_username_to_dev_model[dev_username].id,
                predictive_model_name=model_details.name,
                predictive_model_version=model_details.version,
                predictive_model_confidence=matched_dev_author.confidence,
            )
            linked_dev_researcher_pairs.append(linked_researcher_dev_account)

        # Attach
        pair.researcher_developer_links = linked_dev_researcher_pairs

        return pair

    except Exception as e:
        return types.ErrorResult(
            source=pair.dataset_source_model.name,
            step="developer-researcher-linking",
            identifier=pair.document_model.doi,
            error=str(e),
            traceback=traceback.format_exc(),
        )
