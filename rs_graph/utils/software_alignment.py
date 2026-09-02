"""Utilities for aligning software names across multiple sources using fuzzy matching."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from .identifier_normalization import normalize_name
from .software_alternates import are_alternates, are_known_distinct

AlignmentMethod = Literal["global_min_diff", "greedy_max_first"]


@dataclass
class PairwiseAlignmentResult:
    item_one_source: str
    item_one: str
    normalized_item_one: str
    item_two_source: str
    item_two: str
    normalized_item_two: str
    score: float


def _solve_global_min_diff(
    sim_matrix: np.ndarray,
    cutoff: float,
) -> list[tuple[int, int, float]]:
    """Hungarian algorithm for globally optimal one-to-one assignment."""
    n_b, n_a = sim_matrix.shape
    max_size = max(n_a, n_b)
    cost_matrix = np.full((max_size, max_size), -cutoff)
    # Gate sub-cutoff similarities to 0 before solving: they can never become accepted
    # matches, so they must not influence which above-cutoff assignment wins (otherwise a
    # junk item can "absorb" a column at sub-cutoff similarity and steal an exact match's
    # partner whenever a near-tie alternative exists).
    cost_matrix[:n_b, :n_a] = -np.where(sim_matrix >= cutoff, sim_matrix, 0.0)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pairs: list[tuple[int, int, float]] = []
    for i, j in zip(row_ind, col_ind, strict=False):
        if i >= n_b or j >= n_a:
            continue
        score = sim_matrix[i, j]
        if score >= cutoff:
            pairs.append((i, j, float(score)))
    return pairs


def _solve_greedy_max_first(
    sim_matrix: np.ndarray,
    cutoff: float,
) -> list[tuple[int, int, float]]:
    """Greedily pick the highest-scoring pair, remove both items, repeat."""
    n_a = sim_matrix.shape[1]
    greedy_sim = sim_matrix.copy()
    pairs: list[tuple[int, int, float]] = []
    while True:
        flat_idx = int(np.argmax(greedy_sim))
        i, j = divmod(flat_idx, n_a)
        score = greedy_sim[i, j]
        if score < cutoff:
            break
        pairs.append((i, j, float(score)))
        greedy_sim[i, :] = -1.0
        greedy_sim[:, j] = -1.0
    return pairs


def align_software_names(
    items_a: list[str],
    items_b: list[str],
    source_a: str,
    source_b: str,
    cutoff: float = 75.0,
    method: AlignmentMethod = "global_min_diff",
    use_alternates: bool = True,
) -> list[PairwiseAlignmentResult]:
    """
    Align two lists of software names using fuzzy matching.

    Finds a one-to-one assignment between `items_a` and `items_b`
    that maximizes fuzzy similarity, then filters out pairs below `cutoff`.

    Args:
        items_a: Software names from the first source.
        items_b: Software names from the second source.
        source_a: Label for the first source (e.g. "import").
        source_b: Label for the second source (e.g. "mention").
        cutoff: Minimum similarity score (0-100) to accept a match.
        method: Assignment strategy.
            "global_min_diff" — Hungarian algorithm for globally optimal assignment.
            "greedy_max_first" — Greedily pick the highest-scoring pair, remove both
            items, and repeat.
        use_alternates: When True, names that belong to the same alternate group
            (from software-name-alternates.yaml) are scored as 100.0.

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

    # Compute similarity matrix (rows=items_b, cols=items_a)
    sim_matrix = np.zeros((len(norm_b), len(norm_a)))
    for i, nb in enumerate(norm_b):
        for j, na in enumerate(norm_a):
            if na == nb:
                sim_matrix[i, j] = 100.0
            elif use_alternates and are_alternates(na, nb):
                # Slightly below an exact-name match so one-to-one assignment prefers
                # identical names over alias-group ties (still far above any cutoff).
                sim_matrix[i, j] = 99.9
            elif use_alternates and are_known_distinct(na, nb):
                # Registry veto: both names known, in different groups -- never fuzzy-match.
                sim_matrix[i, j] = 0.0
            else:
                sim_matrix[i, j] = fuzz.ratio(nb, na)

    # Solve assignment
    if method == "global_min_diff":
        pairs = _solve_global_min_diff(sim_matrix, cutoff)
    else:
        pairs = _solve_greedy_max_first(sim_matrix, cutoff)

    # Convert pairs to results
    return [
        PairwiseAlignmentResult(
            item_one_source=source_a,
            item_one=items_a[j],
            normalized_item_one=lut_a[items_a[j]],
            item_two_source=source_b,
            item_two=items_b[i],
            normalized_item_two=lut_b[items_b[i]],
            score=score,
        )
        for i, j, score in pairs
    ]
