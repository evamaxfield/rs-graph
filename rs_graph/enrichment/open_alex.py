#!/usr/bin/env python

from __future__ import annotations

import pyalex

#######################################################################################

DEFAULT_ALEX_EMAIL = "evamxb@uw.edu"

#######################################################################################


def _setup() -> None:
    """Set up a pool of polite workers for OpenAlex."""
    # Add email for polite pool
    pyalex.config.email = DEFAULT_ALEX_EMAIL

    # Add retries
    pyalex.config.max_retries = 2
    pyalex.config.retry_backoff_factor = 0.1
    pyalex.config.retry_http_codes = [429, 500, 503]


def get_work_from_doi(doi: str) -> pyalex.Work:
    """Get work from a DOI."""
    _setup()

    # Lowercase DOI
    doi = doi.lower()

    # Handle DOI to doi.org
    if "doi.org" not in doi:
        doi = f"https://doi.org/{doi}"

    return pyalex.Works()[doi]
