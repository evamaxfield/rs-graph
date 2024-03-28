#!/usr/bin/env python

from __future__ import annotations

import logging
from xml.etree import ElementTree as ET  # noqa: N817

from pathlib import Path
from allofplos.corpus.plos_corpus import get_corpus_dir
from allofplos.update import main as get_latest_plos_corpus

from .proto import DataSource, RepositoryDocumentPair

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class PLOSDataSource(DataSource):

    @staticmethod
    def _get_plos_xmls() -> list[Path]:
        # Download all plos files
        get_latest_plos_corpus()

        # Get the corpus dir
        corpus_dir = get_corpus_dir()

        # Get all files
        return list(Path(corpus_dir).resolve(strict=True).glob("*.xml"))

    @staticmethod
    def get_dataset(
        **kwargs,
    ) -> list[RepositoryDocumentPair]:
        """
        Download the PLOS dataset.
        """
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
            original_page_results, continue_next = _process_joss_results_page(
                response.json()
            )

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
            log.info(f"Processed {total_processed} papers")

        # Log final metrics
        log.info(f"Total processed: {total_processed}")
        log.info(f"Total errored: {total_errored}")

        return processed_results