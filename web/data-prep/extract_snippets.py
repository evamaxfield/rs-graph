"""Extracts the site-snippet block from each queries/*.py file.

Run as part of regenerating web/data-prep/output/, not by the Astro build --
keeps web/site/ dependency-free of Python. Writes plain-text fragments to
web/site/src/generated/, which pages import directly. This is what makes the
on-site snippet literally the same code that produced the number.
"""

import os

START_MARKER = "# --- site-snippet:start ---"
END_MARKER = "# --- site-snippet:end ---"

QUERIES_DIR = os.path.join(os.path.dirname(__file__), "queries")
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "..", "site", "src", "generated")


def extract_snippet(path: str) -> str | None:
    with open(path) as f:
        lines = f.readlines()
    in_block = False
    block_lines = []
    for line in lines:
        if START_MARKER in line:
            in_block = True
            continue
        if END_MARKER in line:
            break
        if in_block:
            block_lines.append(line)
    if not block_lines:
        return None
    # dedent to the block's own minimum indentation
    indents = [len(line) - len(line.lstrip()) for line in block_lines if line.strip()]
    min_indent = min(indents) if indents else 0
    return "".join(line[min_indent:] if line.strip() else line for line in block_lines)


def run() -> None:
    os.makedirs(GENERATED_DIR, exist_ok=True)
    for filename in sorted(os.listdir(QUERIES_DIR)):
        if not filename.endswith(".py") or filename == "__init__.py":
            continue
        snippet = extract_snippet(os.path.join(QUERIES_DIR, filename))
        if snippet is None:
            continue
        out_name = filename.replace(".py", ".py.txt")
        with open(os.path.join(GENERATED_DIR, out_name), "w") as f:
            f.write(snippet)
        print(f"extracted {out_name}")


if __name__ == "__main__":
    run()
