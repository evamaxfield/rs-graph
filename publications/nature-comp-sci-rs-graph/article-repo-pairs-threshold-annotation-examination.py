#!/usr/bin/env python

from pathlib import Path

import polars as pl
import typer

###############################################################################

app = typer.Typer()

THIS_FILE_PATH = Path(__file__).resolve()
THIS_DIR = THIS_FILE_PATH.parent

ANNOTATION_CSV_PATH = THIS_DIR / "article-repo-pairs-threshold-annotation.csv"
CONFIDENCE_THRESHOLD = 0.9994

###############################################################################


def _print_stats(label: str, count: int, total: int) -> None:
    pct = 100 * count / total if total else 0
    prop = count / total if total else 0
    print(f"  {label}: n={count}  ({pct:.2f}%  /  {prop:.4f})")


@app.command()
def main() -> None:
    """Report match rates and link-directionality stats for the threshold annotation set."""
    df = pl.read_csv(ANNOTATION_CSV_PATH)

    # Keep labeled, non-unclear rows at or above the pair-confidence threshold

    df = df.filter(
        pl.col("label").is_not_null()
        & (pl.col("label") != "unclear")
        & (pl.col("predictive_model_confidence") >= CONFIDENCE_THRESHOLD)
    )

    total = len(df)
    matches = df.filter(pl.col("label") == "match")
    n_matches = len(matches)

    print(f"\nPost-filter rows: {total}")
    print("\n--- Overall match rate ---")
    _print_stats("match", n_matches, total)
    _print_stats("non-match", total - n_matches, total)

    print("\n--- Matches: repo linked in paper ---")
    _print_stats(
        "repo NOT linked in paper",
        len(matches.filter(pl.col("this-repo-linked-in-paper") == "no")),
        n_matches,
    )
    _print_stats(
        "repo linked in paper",
        len(matches.filter(pl.col("this-repo-linked-in-paper") == "yes")),
        n_matches,
    )

    print("\n--- Matches: paper linked in repo ---")
    _print_stats(
        "paper NOT linked in repo",
        len(matches.filter(pl.col("this-paper-linked-in-repo") == "no")),
        n_matches,
    )
    _print_stats(
        "paper linked in repo",
        len(matches.filter(pl.col("this-paper-linked-in-repo") == "yes")),
        n_matches,
    )

    print("\n--- Matches: bi-directionality ---")
    _print_stats(
        "bi-directional linkage (both linked)",
        len(
            matches.filter(
                (pl.col("this-repo-linked-in-paper") == "yes")
                & (pl.col("this-paper-linked-in-repo") == "yes")
            )
        ),
        n_matches,
    )
    _print_stats(
        "no linkage at all (neither linked)",
        len(
            matches.filter(
                (pl.col("this-repo-linked-in-paper") == "no")
                & (pl.col("this-paper-linked-in-repo") == "no")
            )
        ),
        n_matches,
    )
    _print_stats(
        "one-way linkage (only one linked)",
        len(
            matches.filter(
                (
                    (pl.col("this-repo-linked-in-paper") == "yes")
                    & (pl.col("this-paper-linked-in-repo") == "no")
                )
                | (
                    (pl.col("this-repo-linked-in-paper") == "no")
                    & (pl.col("this-paper-linked-in-repo") == "yes")
                )
            )
        ),
        n_matches,
    )


###############################################################################

if __name__ == "__main__":
    app()
