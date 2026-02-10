#!/usr/bin/env python3
"""Post-process Quarto-generated TeX for Overleaf/arXiv cleanup.

Usage examples:
  python cleanup-tex.py
  python cleanup-tex.py --input qss-code-authors.tex --in-place
  python cleanup-tex.py --input qss-code-authors.tex --output qss-code-authors-cleaned.tex
  python cleanup-tex.py --input qss-code-authors.tex --supplementary
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _apply_regex(text: str, pattern: str, repl: str, flags: int = 0) -> tuple[str, int]:
    return re.subn(pattern, repl, text, flags=flags)


def cleanup_tex(text: str) -> tuple[str, list[tuple[str, int]]]:
    """Apply a fixed set of text cleanups.

    Returns:
      cleaned_text, list of (rule_name, replacement_count)
    """
    changes: list[tuple[str, int]] = []

    # 1) Replace literal tilde escapes with math-mode approximately symbol.
    text, n = _apply_regex(text, r"\\textasciitilde", r"$\\sim$")
    changes.append(("textasciitilde_to_sim", n))

    # 2) Replace Unicode plus/minus with LaTeX math symbol.
    n = text.count("±")
    if n:
        text = text.replace("±", r"$\pm$")
    changes.append(("unicode_pm_to_latex_pm", n))

    # 3) Normalize textual Chi-square to LaTeX chi-squared symbol.
    # Examples:
    #   "Chi-square tests" -> "$\\chi^2$ tests"
    #   "chi square test"  -> "$\\chi^2$ test"
    text, n = _apply_regex(text, r"\b[Cc]hi[- ]square\b", r"$\\chi^2$")
    changes.append(("chi_square_to_chi2_symbol", n))

    # 4) Replace text-mode escaped inequalities with math-mode symbols.
    text, n1 = _apply_regex(text, r"\\textless\{\}", r"$<$")
    text, n2 = _apply_regex(text, r"\\textgreater\{\}", r"$>$")
    changes.append(("textless_to_math_lt", n1))
    changes.append(("textgreater_to_math_gt", n2))

    # 5) Normalize dash spacing to compact em-dash style in prose.
    # "word --- word" -> "word---word"
    text, n = _apply_regex(text, r"(?<=\S)\s+---\s+(?=\S)", r"---")
    changes.append(("trim_spaces_around_em_dash", n))

    # 6) Convert any Unicode dash variants to LaTeX-safe dash forms.
    # En dash -> --
    # Em dash/minus variants -> ---
    n_en = text.count("–")
    n_em = text.count("—")
    n_minus = text.count("−")
    n_nb_hyphen = text.count("‑")
    if n_en:
        text = text.replace("–", "--")
    if n_em:
        text = text.replace("—", "---")
    if n_minus:
        text = text.replace("−", "---")
    if n_nb_hyphen:
        text = text.replace("‑", "-")
    changes.append(("unicode_en_dash_to_double_dash", n_en))
    changes.append(("unicode_em_dash_to_triple_dash", n_em))
    changes.append(("unicode_minus_to_triple_dash", n_minus))
    changes.append(("unicode_nonbreaking_hyphen_to_hyphen", n_nb_hyphen))

    # 7) Swap image paths in \includegraphics from .pdf to .png.
    text, n = _apply_regex(
        text,
        r"(\\includegraphics(?:\[[^\]]*\])?\{[^{}]+?)\.pdf(\})",
        r"\1.png\2",
    )
    changes.append(("includegraphics_pdf_to_png", n))

    return text, changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean generated TeX for Overleaf/arXiv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("qss-code-authors.tex"),
        help="Path to input TeX file (default: qss-code-authors.tex)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output TeX file (default: <input-stem>-cleaned.tex)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file in place.",
    )
    parser.add_argument(
        "--supplementary",
        action="store_true",
        help=(
            "Also clean supplementary-material.tex located in the same directory as --input. "
            "If --output is provided, it applies only to --input; supplementary output uses "
            "the default '<stem>-cleaned.tex' naming."
        ),
    )
    return parser.parse_args()


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}-cleaned{input_path.suffix}")


def _process_file(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input TeX file not found: {input_path}")

    original = input_path.read_text(encoding="utf-8")
    cleaned, changes = cleanup_tex(original)
    output_path.write_text(cleaned, encoding="utf-8")

    total_changes = sum(count for _, count in changes)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Total replacements: {total_changes}")
    for rule_name, count in changes:
        if count:
            print(f"  - {rule_name}: {count}")
    print("")


def main() -> int:
    args = parse_args()
    input_paths = [args.input]

    if args.supplementary:
        supplementary_path = args.input.parent / "supplementary-material.tex"
        if supplementary_path not in input_paths:
            input_paths.append(supplementary_path)

    for idx, input_path in enumerate(input_paths):
        if args.in_place:
            output_path = input_path
        elif idx == 0 and args.output is not None:
            output_path = args.output
        else:
            output_path = _default_output_path(input_path)

        _process_file(input_path, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
