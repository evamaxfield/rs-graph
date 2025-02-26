# Research Software Graph

Collecting the dependencies (and dependents) of research software and their contributors.

## Data

### rs-graph-v1

There is a lot of pre-processed data stored in Harvard Dataverse:
[doi:10.7910/DVN/KPYVI1](https://doi.org/10.7910/DVN/KPYVI1)

To access the dataset, please create an account on Harvard Dataverse and download the
`rs-graph-v1-redacted.db` and/or `rs-graph-v1-prod.db` SQLite database file(s).

If you simply want access to the article-repository pairs and their associated metadata, the `rs-graph-v1-redacted.db`
SQLite database file is sufficient. If you want the full dataset, including the full list of repository contributors and 
their predicted author associations, you will need to request access to the `rs-graph-v1-prod.db` file.

#### Usage

Once downloaded, the database can be used like any other SQLite database.

The database schema/models are available at: [./rs_graph/db/models.py](./rs_graph/db/models.py)

With [SQLModel](https://sqlmodel.tiangolo.com/#select-from-the-database):

```python
from rs_graph.db import models as db_models
from sqlmodel import Session, create_engine, select

engine = create_engine("sqlite:///rs-graph-v1-redacted.db")
with Session(engine) as session:
  # Get the first 10 articles in the database
  statement = select(db_models.Document).limit(10)
  articles = session.exec(statement).all()
  for article in articles:
    print(article)
    print("---")

  print("-" * 80)

  # Get the first 10 articles, article-repository-links, and repositories
  statement = select(
    db_models.Document,
    db_models.DocumentRepositoryLink,
    db_models.Repository,
  ).join(
    db_models.DocumentRepositoryLink,
    db_models.Document.id == db_models.DocumentRepositoryLink.document_id,
  ).join(
    db_models.Repository,
    db_models.DocumentRepositoryLink.repository_id == db_models.Repository.id,
  ).limit(10)
  results = session.exec(statement).all()
  for article, link, repo in results:
    print(article, link, repo)
    print("---")
```

With [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html):

```python
import pandas as pd
from rs_graph.db import models as db_models
from sqlalchemy import text, create_engine

# Create a connection to the database
engine = create_engine("sqlite:///rs-graph-v1-redacted.db")

# Helper function to make this easy
def read_table(table: str) -> pd.DataFrame:
    return pd.read_sql(text(f"SELECT * FROM {table}"), engine)

# Load all article info
articles = read_table(db_models.Document.__tablename__)
print(articles.head(10))

print("-" * 80)

# Load all article-repository links and repository info
article_repo_links = read_table(db_models.DocumentRepositoryLink.__tablename__)
repositories = read_table(db_models.Repository.__tablename__)

# Merge
merged = article_repo_links.merge(
  articles.rename(
    columns={
      "id": "document_id",
      "title": "article_title",
    }
  ),
  on="document_id",
).merge(
  repositories.rename(
    columns={
      "id": "repository_id",
      "name": "repository_name",
      "description": "repository_description",
    }
  ),
  on="repository_id",
)

# Only display a few columns
merged[[
  "article_title",
  "repository_name",
  "repository_description",
  "cited_by_count",
]].head(10)
```

## Data Processing

To process a prelinked-dataset (i.e. JOSS, PLOS, Papers with Code, etc.), run:

```bash
rs-graph-pipelines prelinked-dataset-ingestion {dataset_name}
```

Currently supported datasets can be found in the
[lookup table in the source code](https://github.com/evamaxfield/rs-graph/blob/main/rs_graph/bin/pipelines.py#L42).

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
Defaults to `.github-tokens.yml`. See [gh-tokens-loader](https://github.com/evamaxfield/gh-tokens-loader)
for details about the GitHub Tokens file specification.

## Publications

### Code Contribution and Scientific Authorship

**Note:** You will need to request access to the `rs-graph-v1-prod.db` SQLite database file from
Harvard Dataverse ([doi:10.7910/DVN/KPYVI1](https://doi.org/10.7910/DVN/KPYVI1)) to regenerate the manuscript.
The `rs-graph-v1-redacted.db` file is not sufficient to regenerate the manuscript.

From start to finish, to serve the web version of the
manuscript and enable hot-reloading, run:

The manuscript is located at: [./publications/qss-code-authors/qss-code-authors.qmd](./publications/qss-code-authors/qss-code-authors.qmd)

```bash
micromamba create --name rs-graph python=3.12 -y
micromamba activate rs-graph
micromamba install -c conda-forge just -y
micromamba install -c conda-forge quarto -y
just install
rs-graph-data download --dataverse-token {{ YOUR_DATAVERSE_API_TOKEN }}
just quarto-serve
```

Notes:
- We use `micromamba` for environment management, feel free to use whichever environment manage you prefer, however you will still need to install `just` and `quarto` via Homebrew / Conda / from source.
- The `rs-graph-data download` data pulls down
- The `just quarto-serve` command will start a local server and open the manuscript in your default browser. You can make changes to the manuscript and see them reflected in real-time in your browser.
  - If the hot-reload fails to render after a while, kill the process and try again.
- **Important:** by default, the manuscript will be rendered using a very small sample (2%) of the data to speed up the rendering process and iterative writing. If you want to render the full dataset, change the `USE_SAMPLE` flag in the Quarto Markdown file to `False`.