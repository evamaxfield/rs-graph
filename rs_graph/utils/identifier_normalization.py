"""Utilities for normalizing software names and academic identifiers (DOIs, etc.)."""

import re

import polars as pl

DOI_PREFIX_PATTERN = r"^(?:https?://(?:dx\.)?doi\.org/|doi:)"
COMPILED_DOI_PREFIX_PATTERN = re.compile(
    DOI_PREFIX_PATTERN,
    flags=re.IGNORECASE,
)


def normalize_doi(value: str) -> str:
    """Normalize DOI-like values to their canonical, lowercase DOI-only form."""
    return COMPILED_DOI_PREFIX_PATTERN.sub("", value.strip(), count=1).lower().strip()


def normalize_doi_col(col_name: str) -> pl.Expr:
    """
    Create a Polars expression to normalize a DOI column.

    Normalization:
    - Strip whitespace
    - Lowercase
    - Remove URL prefixes (https://doi.org/, http://dx.doi.org/, etc.)
    - Remove 'doi:' prefix
    """
    return (
        pl.col(col_name)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace(DOI_PREFIX_PATTERN, "")
        .str.strip_chars()
    )


def normalize_name(name: str) -> str:
    """
    Normalize a software name for comparison.

    - Converts to lowercase
    - Removes hyphens, underscores, and spaces
    - Preserves alphanumeric characters and dots
    - Removes newlines and other whitespace
    """
    return (
        name.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace("\n", "")
        .replace("\r", "")
        .strip()
    )


def prep_name_for_printing(name: str) -> str:
    """
    Normalize a name for printing.

    - Removes newlines, carriage returns, and extra spaces
    """
    return name.replace("\n", "").replace("\r", "").strip()
