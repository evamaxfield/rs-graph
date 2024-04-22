#!/usr/bin/env python

from __future__ import annotations

import logging
import traceback

from .. import types
from ..db import models as db_models
from ..ml import dev_author_em_clf

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def match_devs_and_researchers_for_pair(
    pair: types.ExpandedRepositoryDocumentPair,
) -> types.ExpandedRepositoryDocumentPair | types.ErrorResult:
    try:
        # Assert that the pair has the necessary details
        assert pair.repository_contributor_details is not None
        assert pair.researcher_details is not None

        # Convert these to ml ready types and luts
        dev_username_to_dev_details = {
            repo_contrib.developer_account_model.username: (
                dev_author_em_clf.DeveloperDetails(
                    username=repo_contrib.developer_account_model.username,
                    name=repo_contrib.developer_account_model.name,
                    email=repo_contrib.developer_account_model.email,
                )
            )
            for repo_contrib in pair.repository_contributor_details
            if repo_contrib.developer_account_model is not None
        }
        dev_username_to_dev_model = {
            repo_contrib.developer_account_model.username: repo_contrib
            for repo_contrib in pair.repository_contributor_details
            if repo_contrib.developer_account_model is not None
        }
        researcher_name_to_researcher_model = {
            researcher.researcher_model.name: researcher
            for researcher in pair.researcher_details
            if researcher.researcher_model is not None
        }

        # Predict matches
        matches = dev_author_em_clf.match_devs_and_authors(
            devs=list(dev_username_to_dev_details.values()),
            authors=[
                researcher.researcher_model.name
                for researcher in pair.researcher_details
                if researcher.researcher_model is not None
            ],
        )

        # Create the linked pairs
        linked_dev_researcher_pairs = []
        for dev_username, author_name in matches.items():
            linked_researcher_dev_account = (
                db_models.ResearcherDeveloperAccountLink(
                    researcher_id=researcher_name_to_researcher_model[author_name].id,
                    developer_account_id=dev_username_to_dev_model[dev_username].id,
                ),
            )
            linked_dev_researcher_pairs.append(linked_researcher_dev_account)

        # Add to model
        # pair.linked_devs_and_researcher_details = linked_dev_researcher_pairs

        return pair

    except Exception:
        return types.ErrorResult(
            source=str(pair.source_id),
            step="developer-researcher-linking",
            identifier=(f"{pair.document_id} -- {pair.repository_id}"),
            error="Error linking developers and researchers",
            traceback=traceback.format_exc(),
        )
