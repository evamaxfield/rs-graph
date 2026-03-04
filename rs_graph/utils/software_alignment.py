"""Utilities for aligning software names across multiple sources using fuzzy matching."""

from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from rs_graph.utils.identifier_normalization import normalize_name


@dataclass
class PairwiseAlignmentResult:
    item_one_source: str
    item_one: str
    normalized_item_one: str
    item_two_source: str
    item_two: str
    normalized_item_two: str
    score: float


def align_software_names(
    items_a: list[str],
    items_b: list[str],
    source_a: str,
    source_b: str,
    cutoff: float = 75.0,
) -> list[PairwiseAlignmentResult]:
    """
    Align two lists of software names using fuzzy matching and the Hungarian algorithm.

    Finds the globally optimal one-to-one assignment between `items_a` and `items_b`
    that maximizes total fuzzy similarity, then filters out pairs below `cutoff`.

    Args:
        items_a: Software names from the first source.
        items_b: Software names from the second source.
        source_a: Label for the first source (e.g. "import").
        source_b: Label for the second source (e.g. "mention").
        cutoff: Minimum similarity score (0-100) to accept a match.

    Returns:
        One `PairwiseAlignmentResult` per accepted match. Unmatched items are not
        returned; callers can find them by diffing input lists against results.
    """
    if not items_a or not items_b:
        return []

    # Build lookup tables: original -> normalized
    lut_a = {orig: normalize_name(orig) for orig in items_a}
    lut_b = {orig: normalize_name(orig) for orig in items_b}

    norm_a = [lut_a[x] for x in items_a]
    norm_b = [lut_b[x] for x in items_b]

    n_a = len(items_a)
    n_b = len(items_b)
    max_size = max(n_a, n_b)

    # Square cost matrix padded with -cutoff (negated for minimization)
    cost_matrix = np.full((max_size, max_size), -cutoff)

    # Fill with negated fuzz.ratio scores on normalized names
    for i, nb in enumerate(norm_b):
        for j, na in enumerate(norm_a):
            cost_matrix[i, j] = -fuzz.ratio(nb, na)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    results: list[PairwiseAlignmentResult] = []
    for i, j in zip(row_ind, col_ind, strict=False):
        if i >= n_b or j >= n_a:
            continue
        score = -cost_matrix[i, j]
        if score >= cutoff:
            results.append(
                PairwiseAlignmentResult(
                    item_one_source=source_a,
                    item_one=items_a[j],
                    normalized_item_one=lut_a[items_a[j]],
                    item_two_source=source_b,
                    item_two=items_b[i],
                    normalized_item_two=lut_b[items_b[i]],
                    score=score,
                )
            )

    return results
