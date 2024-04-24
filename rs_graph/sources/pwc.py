#!/usr/bin/env python

from __future__ import annotations

import gzip
import json
import shutil
import traceback
from pathlib import Path

import requests
from tqdm import tqdm

from .. import types

###############################################################################

DOWNLOAD_URL = (
    "https://production-media.paperswithcode.com/"
    "about/links-between-papers-and-code.json.gz"
)

###############################################################################


def _get_raw_data() -> list[dict]:
    try:
        # Remove any existing files
        Path("pwc.json.gz").unlink(missing_ok=True)

        # Download the JSON and store to gz file
        with open("pwc.json.gz", "wb") as f:
            with requests.get(DOWNLOAD_URL, stream=True) as r:
                shutil.copyfileobj(r.raw, f)

        # Load the JSON
        with gzip.open("pwc.json.gz") as f:
            return json.load(f)

    except Exception as e:
        return types.ErrorResult(
            source="pwc",
            step="pwc-json-download",
            identifier="n/a",
            error=str(e),
            traceback=traceback.format_exc(),
        )

    finally:
        # Remove archive and JSON
        Path("pwc.json.gz").unlink(missing_ok=True)


def _process_item(
    item: dict,
) -> types.BasicRepositoryDocumentPair | types.ErrorResult:
    try:
        # Example
        # {
        #     "paper_url": "https://paperswithcode.com/paper/fast-disparity-estimation-using-dense",
        #     "paper_title": "Fast Disparity Estimation using Dense Networks",
        #     "paper_arxiv_id": "1805.07499",
        #     "paper_url_abs": "http://arxiv.org/abs/1805.07499v1",
        #     "paper_url_pdf": "http://arxiv.org/pdf/1805.07499v1.pdf",
        #     "repo_url": "https://github.com/roatienza/densemapnet",
        #     "is_official": true,
        #     "mentioned_in_paper": true,
        #     "mentioned_in_github": true,
        #     "framework": "tf"
        # },

        # If it is not "official" reject
        if not item["is_official"]:
            return types.ErrorResult(
                source="pwc",
                step="pwc-json-processing",
                identifier=item["paper_arxiv_id"],
                error="Not an official paper",
                traceback="",
            )

        # Construct the doi
        doi = f"10.48550/arxiv.{item['paper_arxiv_id']}"

        # Create the pair
        return types.BasicRepositoryDocumentPair(
            source="pwc",
            repo_url=item["repo_url"],
            paper_doi=doi,
        )

    except Exception as e:
        return types.ErrorResult(
            source="pwc",
            step="pwc-json-processing",
            identifier=item["paper_url"],
            error=str(e),
            traceback=traceback.format_exc(),
        )


def get_dataset(
    **kwargs: dict[str, str],
) -> types.SuccessAndErroredResultsLists:
    """Download the PLOS dataset."""
    # Get all PLOS XMLs
    import random

    # Get all data
    data = random.sample(_get_raw_data(), 50)

    # Process each item
    results = [
        _process_item(item)
        for item in tqdm(
            data,
            desc="Processing Papers with Code JSONs",
        )
    ]

    # Create successful results and errored results lists
    successful_results = [
        r for r in results if isinstance(r, types.BasicRepositoryDocumentPair)
    ]
    errored_results = [r for r in results if isinstance(r, types.ErrorResult)]

    # Log total processed and errored
    print(f"Total succeeded: {len(successful_results)}")
    print(f"Total errored: {len(errored_results)}")

    # Return filepath
    return types.SuccessAndErroredResultsLists(
        successful_results=successful_results,
        errored_results=errored_results,
    )
