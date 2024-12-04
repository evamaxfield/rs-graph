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

## Publications

### Code Contribution and Scientific Authorship

From start to finish, to serve the web version of the
manuscript and enable hot-reloading, run:

The manuscript is located at: [./publications/qss-code-authors/qss-code-authors.qmd](./publications/qss-code-authors/qss-code-authors.qmd)

```bash
micromamba create --name rs-graph python=3.12 -y
micromamba activate rs-graph
micromamba install -c conda-forge just -y
micromamba install -c conda-forge quarto -y
just install
rs-graph-data download
just quarto-serve
```

Notes:
- We use `micromamba` for environment management, feel free to use whichever environment manage you prefer, however you will still need to install `just` and `quarto` via Homebrew / Conda / from source.
- The `rs-graph-data download` data pulls down the latest version of the production database. This is a ~1.5GB SQLite file, so it may take a while to download.
- The `just quarto-serve` command will start a local server and open the manuscript in your default browser. You can make changes to the manuscript and see them reflected in real-time in your browser.
  - If the hot-reload fails to render after a while, kill the process and try again.
- **Important:** by default, the manuscript will be rendered using a very small sample (2%) of the data to speed up the rendering process and iterative writing. If you want to render the full dataset, change the `USE_SAMPLE` flag in the manuscript file to `False`.