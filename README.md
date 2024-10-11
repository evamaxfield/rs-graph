# Research Software Graph

Collecting the dependencies (and dependents) of research software and their contributors.

## Setup

* Install [Just](https://github.com/casey/just#packages)
* Create a new Conda / Mamba Env (Python 3.11+ preferred)
* Install lib: `just install` (or `pip install .`)

## Data Download

There is a lot of pre-processed data stored on a GCS bucket. To download it, run:

```bash
rs-graph-data download
```

This will always download the latest version of the data (as pushed by @evamaxfield).

## Data Processing

To process a prelinked-dataset (i.e. JOSS, PLOS, Papers with Code, etc.), run:

```bash
rs-graph-pipelines prelinked-dataset-ingestion {dataset_name}
```

Currently supported datasets can be found in the
[lookup table in the source code](https://github.com/evamaxfield/rs-graph/blob/main/rs_graph/bin/pipelines.py#L38).

There are some optional arguments as well:

- `--use-coiled` will use [coiled.io](https://coiled.io) to run the 
non-database writing portions of the pipeline in parallel on 
multiple Dask clusters. Database writes still occur locally on your 
machine so you can use the data while processing. 
Defaults to False (process everything locally).

- `--use-prod` will use the "production" database for storage rather than 
the "development" database. Defaults to False (use development database).

- `--github-tokens-file` allows you to provide the path to the YAML file 
which contains the GitHub tokens to use for the GitHub API. 
Defaults to `.github-tokens.yml`.

### GitHub Tokens

The pipeline makes lots of requests to the GitHub API. 
To provide GitHub Personal Access Tokens (PATs), 
create a YAML file with the following format:

```yaml
tokens:
  - name1: token_value1
  - name2: token_value2
```

It is recommeded to label the tokens with a name,
for example the name of the GitHub username
associated with the token.

Each GitHub PAT does not need to have any special permissions,
we simply need them to increase the rate limit for the API.

In general, you should never be rate limited as the 
pipeline scales the number of workers based on 
the number of tokens you provide.