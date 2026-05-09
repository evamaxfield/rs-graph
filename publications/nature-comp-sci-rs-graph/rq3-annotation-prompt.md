I am a researcher studying how scientific software is attributed in academic papers. I have an Excel file (`mentions-imports-deps-annotation.xlsx`) with 100 rows, each representing a scientific paper paired with a code repository. I need you to partially pre-populate an annotation for me.

## What the data represents

Each row has three sets of software names associated with the paper-repository pair:
- **Mentions** (`mentioned_software_normalized`): software names extracted from the text of the paper itself using the SoftCite dataset
- **Imports** (`imported_software_normalized`): software packages actually imported in the repository's source code
- **Dependencies** (`dependencies_software_normalized`): software packages declared in the repository's dependency manifests (requirements.txt, DESCRIPTION, package.json, etc.)

The six set-difference columns show what appears in one set but not another (all normalized, semicolon-separated):
- `mentions_not_in_imports_normalized`
- `imports_not_in_mentions_normalized`
- `mentions_not_in_dependencies_normalized`
- `dependencies_not_in_mentions_normalized`
- `imports_not_in_dependencies_normalized`
- `dependencies_not_in_imports_normalized`

## Your task

For each row, populate the three notes columns with semicolon-separated codes from the codebook below. Each notes column covers two diff columns:
- `mentions_imports_differences_notes` → explains `mentions_not_in_imports_normalized` and `imports_not_in_mentions_normalized`
- `mentions_dependencies_differences_notes` → explains `mentions_not_in_dependencies_normalized` and `dependencies_not_in_mentions_normalized`
- `imports_dependencies_differences_notes` → explains `imports_not_in_dependencies_normalized` and `dependencies_not_in_imports_normalized`

Apply whichever codes from the codebook best explain the dominant reasons items appear in each diff. A cell might get one code or several (e.g. `INFRA_CI;INFRA_DOC;BASELINE_REQUIRED_LIB`).

## Codebook

| Code | Meaning |
|------|---------|
| `DATASET_BENCHMARK` | Item refers to a dataset, benchmark, or evaluation corpus, not a software tool |
| `COMPARISON_METHOD` | A compared-against method/system not actually installed in the repo |
| `PLATFORM_SERVICE` | External platform or cloud service (AWS, GEE, Azure) — conceptual, not an importable package |
| `NAME_VARIANT` | Same package under a different name (import alias vs. package name, e.g. `pil`/`pillow`, `sklearn`/`scikitlearn`, `torch`/`pytorch`) |
| `INFRA_CI` | CI/CD tooling or GitHub Actions (e.g. `actions/checkout`, `actions/setuppython`, `jamesives/githubpagesdeployaction`) |
| `INFRA_DOC` | Documentation or notebook tooling (`knitr`, `rmarkdown`, `jupyter`, `nbconvert`, `nbformat`) |
| `INFRA_TEST` | Testing framework (`pytest`, `testthat`, `unittest`) |
| `UNDECLARED_DEP` | A package imported in code but absent from the dependency manifest — likely an oversight by the authors |
| `BASELINE_REQUIRED_LIB` | A library so fundamental to contemporary computational science that authors use it without citing it (e.g. `numpy`, `pandas`, `matplotlib`, `ggplot2`, `dplyr`, `scipy`, `torch`) |
| `CONCEPTUAL_REF` | Software discussed or cited as prior work or context in the paper but not directly used or installed |

## Which codes you can apply confidently vs. which to leave blank

**Apply these based on the item names alone — no paper reading needed:**
- `NAME_VARIANT` — you can detect these by checking if two items across sets are known aliases (e.g. pil/pillow, sklearn/scikitlearn, torch/pytorch, ignite/pytorchignite, flake8/pyflakes, cv2/opencv)
- `INFRA_CI` — any item matching a GitHub Actions pattern (`owner/action-name`) or known CI tools
- `INFRA_DOC` — knitr, rmarkdown, jupyter, nbconvert, nbformat, sphinx, pkgdown, roxygen2, quarto
- `INFRA_TEST` — pytest, testthat, unittest, nose, coverage, mock, tox
- `UNDECLARED_DEP` — when `imports_not_in_dependencies_normalized` is non-empty and the items are not infra/test/doc tools
- `BASELINE_REQUIRED_LIB` — numpy, pandas, matplotlib, scipy, torch, tensorflow, sklearn/scikitlearn, ggplot2, dplyr, tidyverse, tidyr, stringr, purrr, data.table, xarray, statsmodels, seaborn, plotly
- `PLATFORM_SERVICE` — aws, azure, gcp, google, googleearthengine, s3, dynamodb, cloudwatch, firebase, heroku, docker

**Leave these blank — they require reading the paper:**
- `DATASET_BENCHMARK`
- `COMPARISON_METHOD`
- `CONCEPTUAL_REF`

## Output

Write the codes into the three notes columns in the Excel file and save it. Do not modify any other columns. If a notes column already has content, append to it rather than overwriting. If none of the applicable codes apply to a given row's diff, leave that cell empty.
