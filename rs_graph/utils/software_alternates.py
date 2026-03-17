"""Loader and lookup utilities for software name alternate groups."""

from functools import lru_cache
from pathlib import Path

import yaml

from rs_graph.utils.identifier_normalization import normalize_name

_ALTERNATES_PATH = (
    Path(__file__).parent.parent / "data" / "files" / "software-name-alternates.yaml"
)


@lru_cache(maxsize=1)
def load_alternate_groups(
    path: Path = _ALTERNATES_PATH,
) -> dict[str, frozenset[str]]:
    """
    Load alternate groups and return a mapping where every normalized variant
    maps to the full set of all normalized variants in its group.

    E.g. looking up any of "cv2", "opencv", "opencvpython" returns
    frozenset({"opencvpython", "cv2", "opencv", "opencvcontribpython"}).
    """
    with open(path) as f:
        raw: dict[str, list[str]] = yaml.safe_load(f) or {}

    # First pass: build each group as a set of normalized names
    groups: list[frozenset[str]] = []
    seen: dict[str, str] = {}  # norm_name -> install_key (for error messages)

    for install_name, alternates in raw.items():
        all_names = [install_name, *alternates]
        norm_set: set[str] = set()
        for name in all_names:
            norm = normalize_name(name)
            if norm in seen and seen[norm] != install_name:
                msg = (
                    f"Alternate '{name}' (normalized: '{norm}') appears in "
                    f"multiple groups: '{seen[norm]}' and '{install_name}'"
                )
                raise ValueError(msg)
            seen[norm] = install_name
            norm_set.add(norm)
        groups.append(frozenset(norm_set))

    # Second pass: map every member to its full group
    mapping: dict[str, frozenset[str]] = {}
    for group in groups:
        for member in group:
            mapping[member] = group

    return mapping


def are_alternates(norm_a: str, norm_b: str) -> bool:
    """Check whether two normalized names belong to the same alternate group."""
    mapping = load_alternate_groups()
    group = mapping.get(norm_a)
    if group is None:
        return False
    return norm_b in group
