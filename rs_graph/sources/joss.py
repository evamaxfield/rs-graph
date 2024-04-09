#!/usr/bin/env python

from __future__ import annotations

import logging

import requests

from ..types import RepositoryDocumentPair, SuccessAndErroredResultsLists
from .proto import DataSource

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

JOSS_PUBLISHED_PAPERS_URL_TEMPLATE = (
    "https://joss.theoj.org/papers/published.json?page={page}"
)

###############################################################################


class JOSSDataSource(DataSource):
    @staticmethod
    def _process_joss_results_page(
        results: list[dict],
    ) -> tuple[list[RepositoryDocumentPair | None], bool]:
        # Store "continuation" flag
        continue_next = len(results) == 10

        # Store processed results
        processed_results: list[RepositoryDocumentPair | None] = []

        # Parse each result
        for paper in results:
            # Ensure "state" == "accepted"
            if paper["state"] != "accepted":
                processed_results.append(None)
                continue

            # Parse paper information
            processed_results.append(
                RepositoryDocumentPair(
                    source="joss",
                    repo_url=paper["software_repository"],
                    paper_doi=paper["doi"],
                )
            )

        return processed_results, continue_next

    @staticmethod
    def get_dataset(
        cluster_address: str | None = None,
        **kwargs: dict[str, str],
    ) -> SuccessAndErroredResultsLists:
        """Download the JOSS dataset."""
        # Get all processed results
        processed_results = []

        # Set initial continue_next flag
        current_page = 1
        continue_next = True

        # State for storing valid metrics
        total_processed = 0
        total_errored = 0

        # While continue_next flag is True
        # Process each page
        while continue_next:
            # Get response
            response = requests.get(
                JOSS_PUBLISHED_PAPERS_URL_TEMPLATE.format(page=current_page)
            )

            # Raise error
            try:
                response.raise_for_status()
            except Exception as e:
                log.error(f"Error getting JOSS page {current_page}: {e}")
                log.error(f"Full response data: {response}")
                continue

            # Process page results
            (
                original_page_results,
                continue_next,
            ) = JOSSDataSource._process_joss_results_page(response.json())

            # Increment total processed
            total_processed += len(original_page_results)

            # Filter out None values
            cleaned_page_results = [
                result for result in original_page_results if result is not None
            ]

            # Increment total errored
            total_errored += len(original_page_results) - len(cleaned_page_results)

            # Store page results
            processed_results.extend(cleaned_page_results)

            # Increment page
            current_page += 1

            # Update progress
            if total_processed % 500 == 0:
                log.info(f"Processed {total_processed} papers")

        # Log final metrics
        log.info(f"Total processed: {total_processed}")
        log.info(f"Total errored: {total_errored}")

        return SuccessAndErroredResultsLists(
            successful_results=processed_results,
            errored_results=[],
        )
