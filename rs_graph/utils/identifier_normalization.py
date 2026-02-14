"""Utilities for normalizing academic identifiers (DOIs, etc.)."""

import re

DOI_PREFIX_PATTERN = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:)",
    flags=re.IGNORECASE,
)


def normalize_doi(value: str) -> str:
    """Normalize DOI-like values to their canonical DOI-only form."""
    return DOI_PREFIX_PATTERN.sub("", value.strip(), count=1)
