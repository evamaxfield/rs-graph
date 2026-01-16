from collections.abc import Callable

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment


def normalize_name(name: str) -> str:
    """
    Normalize a software name for comparison.

    - Converts to lowercase
    - Removes hyphens, underscores, and spaces
    - Preserves alphanumeric characters and dots

    Examples:
        "Scikit-Learn" -> "scikitlearn"
        "scikit_learn" -> "scikitlearn"
        "OpenCV-Python" -> "opencvpython"
    """
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def align_dependencies(
    imported: set[str],
    mentioned: set[str],
    score_cutoff: float = 70.0,
    scorer: Callable = fuzz.ratio,
) -> tuple[dict[str, tuple[str, float]], list[str], dict[str, str], dict[str, str]]:
    """
    Align imported dependencies with mentioned software names using optimal assignment.

    Uses the Hungarian algorithm to find the globally optimal matching that
    maximizes the total similarity score across all pairs.

    Normalizes strings before comparison but tracks original names.

    Args:
        imported: Set of imported dependency names (e.g., from code analysis)
        mentioned: Set of mentioned software names (e.g., from papers)
        score_cutoff: Minimum similarity score (0-100) to consider a match
        scorer: Scoring function from rapidfuzz.fuzz (default: fuzz.ratio)
                Other options: fuzz.token_sort_ratio, fuzz.token_set_ratio

    Returns:
        Tuple of:
        - matched: Mapping mentioned software (original) -> (imported dep original, score)
        - unmatched: List of imported dependencies (original) that weren't matched
        - imported_lut: Dict mapping original imported name -> normalized name
        - mentioned_lut: Dict mapping original mentioned name -> normalized name

    Example:
        imported = {"sklearn", "pandas", "numpy", "rapidfuzz"}
        mentioned = {"Scikit-Learn", "Pandas", "NumPy"}

        matched, unmatched, imp_lut, men_lut = align_dependencies(imported, mentioned)
        # matched = {
        #     "Scikit-Learn": ("sklearn", 85.7),
        #     "Pandas": ("pandas", 100.0),
        #     "NumPy": ("numpy", 90.0)
        # }
        # unmatched = ["rapidfuzz"]
    """
    if not imported or not mentioned:
        # Handle empty sets
        imported_lut = {orig: normalize_name(orig) for orig in imported}
        mentioned_lut = {orig: normalize_name(orig) for orig in mentioned}
        return {}, list(imported), imported_lut, mentioned_lut

    # Create lookup tables: original -> normalized
    imported_lut = {orig: normalize_name(orig) for orig in imported}
    mentioned_lut = {orig: normalize_name(orig) for orig in mentioned}

    # Convert to lists to maintain consistent ordering
    imported_list = list(imported)
    mentioned_list = list(mentioned)

    # Get normalized versions
    imported_norm = [imported_lut[imp] for imp in imported_list]
    mentioned_norm = [mentioned_lut[men] for men in mentioned_list]

    # Build cost matrix (we'll use negative scores since linear_sum_assignment minimizes)
    n_mentioned = len(mentioned_list)
    n_imported = len(imported_list)

    # Create a square cost matrix by padding with dummy entries if needed
    max_size = max(n_mentioned, n_imported)
    cost_matrix = np.full((max_size, max_size), -score_cutoff)  # Fill with minimum score

    # Fill in actual similarity scores (negated for minimization)
    for i, men_norm in enumerate(mentioned_norm):
        for j, imp_norm in enumerate(imported_norm):
            score = scorer(men_norm, imp_norm)
            cost_matrix[i, j] = -score  # Negate because we want to maximize

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract matches above threshold
    matched = {}
    used_imported_indices = set()

    for i, j in zip(row_ind, col_ind, strict=False):
        # Skip if this is a padding entry
        if i >= n_mentioned or j >= n_imported:
            continue

        score = -cost_matrix[i, j]  # Convert back to positive score

        if score >= score_cutoff:
            mention_orig = mentioned_list[i]
            import_orig = imported_list[j]
            matched[mention_orig] = (import_orig, score)
            used_imported_indices.add(j)

    # Find unmatched imports
    unmatched = [imported_list[j] for j in range(n_imported) if j not in used_imported_indices]

    return matched, unmatched, imported_lut, mentioned_lut


# Example usage and testing
if __name__ == "__main__":
    # Realistic example from scientific software
    # Some imported deps won't be in mentioned (like rapidfuzz - internal tooling)
    # Some mentioned software won't be in imported (like FIJI - external tools)
    imported_deps = {
        "sklearn",
        "pandas",
        "seaborn",
        "numpy",
        "matplotlib",
        "rapidfuzz",
        "scipy",
        "requests",
    }

    mentioned_software = {
        "Scikit-Learn",
        "Pandas",
        "Seaborn",
        "NumPy",
        "SciPy",
        "Matplotlib",
        "FIJI",
        "ImageJ",
        "GraphPad Prism",  # Hardware/external tools not in code
    }

    print("Example 1: Realistic paper scenario")
    print("=" * 60)
    print(f"Imported deps: {sorted(imported_deps)}")
    print(f"Mentioned software: {sorted(mentioned_software)}")

    matched, unmatched, imp_lut, men_lut = align_dependencies(
        imported_deps, mentioned_software, score_cutoff=70.0
    )

    print("\nNormalization lookup tables:")
    print("Imported (original -> normalized):")
    for orig, norm in sorted(imp_lut.items()):
        if orig != norm:  # Only show if different
            print(f"  {orig:20} -> {norm}")
    print("Mentioned (original -> normalized):")
    for orig, norm in sorted(men_lut.items()):
        if orig != norm:  # Only show if different
            print(f"  {orig:20} -> {norm}")

    print("\nMatched (mentioned software -> imported dependency):")
    for mention, (imp, score) in sorted(matched.items()):
        print(f"  {mention:20} <- {imp:15} (score: {score:.1f})")

    print(f"\nTotal match score: {sum(score for _, score in matched.values()):.1f}")

    print("\nUnmatched imported (likely internal/utility libraries):")
    for imp in sorted(unmatched):
        print(f"  - {imp}")

    # Show which mentioned software had no matches
    unmatched_mentioned = [m for m in mentioned_software if m not in matched]
    print("\nUnmatched mentioned (likely hardware/external tools):")
    for mention in sorted(unmatched_mentioned):
        print(f"  - {mention}")

    # Example with sklearn and pandas extensions
    print("\n\nExample 2: sklearn and pandas extensions")
    print("=" * 60)

    imported_deps2 = {
        "sklearn",
        "sktime",
        "imblearn",
        "skimage",
        "pandas",
        "geopandas",
        "dask",
        "modin",
        "boto3",
        "requests",
    }
    mentioned_software2 = {
        "Scikit-Learn",
        "scikit-time",
        "imbalanced-learn",
        "scikit-image",
        "Pandas",
        "GeoPandas",
        "Dask",
        "R",
        "SPSS",
        "ImageJ",
    }

    print(f"Imported deps: {sorted(imported_deps2)}")
    print(f"Mentioned software: {sorted(mentioned_software2)}")

    matched2, unmatched2, imp_lut2, men_lut2 = align_dependencies(
        imported_deps2, mentioned_software2, score_cutoff=70.0
    )

    print("\nMatched:")
    for mention, (imp, score) in sorted(matched2.items()):
        print(f"  {mention:20} <- {imp:15} (score: {score:.1f})")

    print(f"\nTotal match score: {sum(score for _, score in matched2.values()):.1f}")
    print(f"\nUnmatched imported: {sorted(unmatched2)}")

    unmatched_mentioned2 = [m for m in mentioned_software2 if m not in matched2]
    print(f"Unmatched mentioned: {sorted(unmatched_mentioned2)}")

    # Example with ambiguous/competing matches
    print("\n\nExample 3: Handling competing matches with optimal assignment")
    print("=" * 60)

    imported_deps3 = {"sklearn", "sklearn_pandas", "pandas"}
    mentioned_software3 = {"Scikit-Learn", "sklearn-pandas", "Pandas"}

    print(f"Imported deps: {sorted(imported_deps3)}")
    print(f"Mentioned software: {sorted(mentioned_software3)}")

    matched3, unmatched3, _, _ = align_dependencies(
        imported_deps3, mentioned_software3, score_cutoff=70.0
    )

    print("\nMatched (optimal global assignment):")
    for mention, (imp, score) in sorted(matched3.items()):
        print(f"  {mention:20} <- {imp:15} (score: {score:.1f})")
    print(f"\nTotal match score: {sum(score for _, score in matched3.values()):.1f}")
    print(f"\nUnmatched imported: {sorted(unmatched3)}")

    unmatched_mentioned3 = [m for m in mentioned_software3 if m not in matched3]
    if unmatched_mentioned3:
        print(f"Unmatched mentioned: {sorted(unmatched_mentioned3)}")
