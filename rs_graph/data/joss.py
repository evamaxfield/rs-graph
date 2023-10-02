#!/usr/bin/env python

import logging
from dataclasses import dataclass

import pandas as pd
import requests
from dataclasses_json import DataClassJsonMixin
from tqdm import tqdm

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

JOSS_PUBLISHED_PAPERS_URL_TEMPLATE = (
    "https://joss.theoj.org/papers/published.json?page={page}"
)

JOSS_PUBLISHED_PAPERS_ESTIMATE = 2176  # As of 2023-10-02

###############################################################################


@dataclass
class JOSSPaperResults(DataClassJsonMixin):
    repo: str
    title: str
    doi: str
    published_date: str


def _process_joss_results_page(
    results: list[dict],
) -> tuple[list[JOSSPaperResults | None], bool]:
    # Store "continuation" flag
    continue_next = len(results) == 10

    # Store processed results
    processed_results: list[JOSSPaperResults | None] = []

    # Parse each result
    for paper in results:
        # Ensure "state" == "accepted"
        if paper["state"] != "accepted":
            processed_results.append(None)
            continue

        # Parse paper information
        processed_results.append(
            JOSSPaperResults(
                repo=paper["software_repository"],
                title=paper["title"],
                doi=paper["doi"],
                published_date=paper["published_at"],
            )
        )

    return processed_results, continue_next


def get_joss_dataset(
    output_filepath: str = "joss-short-paper-details.parquet",
    start_page: int = 1,
) -> str:
    """
    Download the JOSS dataset.

    Parameters
    ----------
    output_filepath: str
        Output filepath for the JOSS dataset.
    start_page: int
        Page to start downloading from.
    """
    # Get all processed results
    processed_results = []

    # Set initial continue_next flag
    current_page = start_page
    continue_next = True

    # State for storing valid metrics
    total_processed = 0
    total_errored = 0

    # Get progress bar
    progress_bar = tqdm(
        desc="Getting papers from JOSS API",
        total=JOSS_PUBLISHED_PAPERS_ESTIMATE,
    )

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
            raise e

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

        # Store to file
        pd.DataFrame(processed_results).to_parquet(output_filepath)

        # Increment page
        current_page += 1

        # Update progress bar
        progress_bar.update(len(original_page_results))

    # Log final metrics
    log.info(f"Total processed: {total_processed}")
    log.info(f"Total errored: {total_errored}")

    # Return final full results path
    return output_filepath
