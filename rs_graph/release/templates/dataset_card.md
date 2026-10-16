---
license: cc-by-4.0
pretty_name: $title
language:
- en
tags:
- research-software
- scientometrics
- github
- openalex
configs:
$configs
$gated
---

# $title

$summary

## Access

$access

## What is in this release

$contents

Tables: $tables

## Recommended filters

The release itself applies no filter: every row is shipped regardless of model confidence.
These are the filters used in the accompanying paper:

- **Article-repository pair:** `document_repository_link.predictive_model_confidence`
  >= 0.9994, or NULL (a curated link).
- **Researcher-developer identity:**
  `researcher_developer_account_link.predictive_model_confidence` >= 0.97.
- **Publication year:** 2008 or later.

## Loading the data

```bash
pip install datasets polars
```

```python
$load_snippet
```

## Licence and third-party terms

The dataset is released under CC BY 4.0. Repository READMEs, repository descriptions,
$texts and software-mention snippets are third-party text and keep their original terms.
Rows sourced from Papers with Code are redistributed under CC BY-SA.

## Takedown requests

To request removal of a record, open an issue on the rs-graph repository:
https://github.com/evamaxfield/rs-graph/issues

## Citation

Please cite the Quantitative Science Studies paper that introduced the data
(https://doi.org/10.1162/QSS.a.465). A citation for the Nature Computational Science paper
will be added once it exists.

## Other variant

$other
