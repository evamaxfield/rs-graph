#!/usr/bin/env python

from __future__ import annotations

import logging
import traceback

from prefect import task

from .. import types
from ..db import models as db_models
from ..ml import dev_author_em_clf

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@task
def match_devs_and_researchers(
    pair: types.StoredRepositoryDocumentPair | types.ErrorResult,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    try:
        # Convert these to ml ready types and luts
        dev_username_to_dev_details = {
            dev_account_model.username: (
                dev_author_em_clf.DeveloperDetails(
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
        matches = dev_author_em_clf.match_devs_and_authors(
            devs=list(dev_username_to_dev_details.values()),
            authors=[researcher.name for researcher in pair.researcher_models],
        )

        # Create the linked pairs
        linked_dev_researcher_pairs: list[db_models.ResearcherDeveloperAccountLink] = []
        for dev_username, researcher_name in matches.items():
            linked_researcher_dev_account = db_models.ResearcherDeveloperAccountLink(
                researcher_id=researcher_name_to_researcher_model[researcher_name].id,
                developer_account_id=dev_username_to_dev_model[dev_username].id,
            )
            linked_dev_researcher_pairs.append(linked_researcher_dev_account)

        # Attach
        pair.researcher_developer_links = linked_dev_researcher_pairs

        return pair

    except Exception as e:
        return types.ErrorResult(
            source=pair.dataset_source_model.name,
            step="developer-researcher-linking",
            identifier=pair.document_model.paper_doi,
            error=str(e),
            traceback=traceback.format_exc(),
        )
