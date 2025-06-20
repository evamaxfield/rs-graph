# Research Software Graph

Research Software Graph (rs-graph) is a software repository which contains the following:

1. A Python library for collecting, processing, and standardizing scientific publication and associated research software/code from multiple sources.

2. Select publications, analysis scripts, and presentations which utilize the created dataset(s). Notably, this includes our preprint [Code Contribution and Credit in Science](https://evamaxfield.github.io/rs-graph/qss-code-contribution.html) which investigates how code contributions are recognized and rewarded in scientific publications.

If you are interested in:

- the created dataset(s), see the [Data](#data) section
- the code for specific publications, see the [Publications](#publications) section
- the Python library (either for use or for contributing), see the [Data Processing](#data-processing) section

## Data

In short, the current version of the library produces a dataset which contains the following details for the following entities:

- Documents (e.g., peer-reviewed articles, preprints), with:
  - DOI
  - title
  - publication date
  - cited-by count
  - topics/fields/domains
- Researchers (e.g., scientists), with:
  - name
  - works count
  - h-index
- Document Contributors (e.g., authors), with:
  - position (e.g., first author, last author)
  - corresponding author status
- Repositories (e.g., GitHub repositories), with:
  - owner
  - name
  - description
  - stargazers count
  - created datetime
  - README content
- Developer Accounts (e.g., GitHub accounts), with:
  - username
  - name
  - email
- Repository Contributors (e.g., code contributors), with:
  - link between a repository and a developer account
- Document-Repository Links (e.g., links between a document and a repository), with:
  - link between a document and a repository (i.e., a paper and its associated code repository)
- Researcher-Developer Account Links (e.g., links between a researcher and a developer account), with:
  - link between a researcher and a developer account (i.e., a scientist and their GitHub account)
  - entity matching provenance and confidence details

While there is more information available in the dataset, these are the primary entities and their associated details.

### rs-graph-v1

The initial release of data from our processing pipeline is stored in Harvard Dataverse: [https://doi.org/10.7910/DVN/KPYVI1](https://doi.org/10.7910/DVN/KPYVI1)

This dataset was used to create the preprint manuscript [Code Contribution and Credit in Science](https://evamaxfield.github.io/rs-graph/qss-code-contribution.html).

To access the dataset, please create an account on Harvard Dataverse and download the
`rs-graph-v1-redacted.db` and/or `rs-graph-v1-prod.db` SQLite database file(s).

If you simply want access to the article-repository pairs and their associated metadata, the `rs-graph-v1-redacted.db`
SQLite database file is sufficient. If you want the full dataset, including the full list of repository contributors and 
their predicted author associations, you will need to request access to the `rs-graph-v1-prod.db` file.

#### Usage

Once downloaded, the database can be used like any other SQLite database.

The database schema/models are available at: [./rs_graph/db/models.py](./rs_graph/db/models.py)

With [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html):

```python
import pandas as pd
from sqlalchemy import text, create_engine

# Create a connection to the database
engine = create_engine("sqlite:///rs-graph-v1-redacted.db")

# Helper function to make this easy
def read_table(table: str) -> pd.DataFrame:
    return pd.read_sql(text(f"SELECT * FROM {table}"), engine)

# Load all article info
articles = read_table("document")
print(articles.head(10))

print("-" * 80)

# Load all article-repository links and repository info
article_repo_links = read_table("document_repository_link")
repositories = read_table("repository")

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

With [SQLModel](https://sqlmodel.tiangolo.com/#select-from-the-database) after installing the `rs-graph` package:

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

## Publications

### Code Contribution and Credit in Science

**Note:** You will need to request access to the `rs-graph-v1-prod.db` SQLite database file from
Harvard Dataverse [https://doi.org/10.7910/DVN/KPYVI1](https://doi.org/10.7910/DVN/KPYVI1) to regenerate the manuscript.
The `rs-graph-v1-redacted.db` file is not sufficient to regenerate the manuscript.

The manuscript is located at: [./publications/qss-code-authors/qss-code-authors.qmd](./publications/qss-code-authors/qss-code-authors.qmd)

This is a [Quarto Markdown](https://quarto.org/docs/get-started/hello/vscode.html) file which contains not only the text of the manuscript, but in addition, the code to load the data from the `rs-graph-v1-prod.db` SQLite database, and conduct all analyses and create all visualizations.

From start to finish, to serve the web version of the
manuscript and enable hot-reloading, run:

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
- The `rs-graph-data download` data pulls down all required files for the research. Including the rs-graph-v1 database and supporting data for the author-developer account matching model.
- The `just quarto-serve` command will start a local server and open the manuscript in your default browser. You can make changes to the manuscript and see them reflected in real-time in your browser.
  - If the hot-reload fails to render after a while, kill the process and try again.
- **Important:** by default, the manuscript will be rendered using a very small sample (2%) of the data to speed up the rendering process and iterative writing. If you want to render the full dataset, change the `USE_SAMPLE` flag in the Quarto Markdown file to `False`.

## Data Processing

First, clone and set up your environment:

```bash
git clone https://github.com/evamaxfield/rs-graph.git
cd rs-graph
micromamba create --name rs-graph python=3.12 -y
micromamba activate rs-graph
micromamba install -c conda-forge just -y
just install
rs-graph-data download --dataverse-token {{ YOUR_DATAVERSE_API_TOKEN }}
```

Notes:
- We use `micromamba` for environment management, feel free to use whichever environment manage you prefer, however you will still need to install `just` via Homebrew / Conda / from source.
- The `rs-graph-data download` data pulls down all required files for the research. Including the rs-graph-v1 database and supporting data for the author-developer account matching model.

After setting up your environment, you can process a prelinked-dataset (i.e. JOSS, PLOS, Papers with Code), with:

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

- `--github-tokens-file` allows you to provide the path to a YAML file 
which contains the GitHub tokens to use for the GitHub API. 
Defaults to `.github-tokens.yml`. See [gh-tokens-loader](https://github.com/evamaxfield/gh-tokens-loader)
for details about the GitHub Tokens file specification.

### TL;DR: How Pre-Linked Datasets are Processed

The current version of the library processes pre-linked datasets. That is, datasets which have known links between scientific articles (preprints, peer-reviewed publications, software articles) and their associated code repositories (e.g., GitHub repositories which store the analysis scripts, the created tool described by the publication). We are interested in adding other dataset "sources" in the future, but for now, we are focused on these pre-linked datasets.

The processing pipeline in short looks something like the following:

First, gather article-repository pairs from the specified dataset source (e.g., JOSS, PLOS, Papers with Code). Then, for each article-repository pair:

1. Parse the software repository "code host" (e.g., GitHub, GitLab), filter out any non-GitHub hosted repositories (for now)
2. Filter out any pairs which already exist in the database (done via checks to article DOI and repository owner and name)
3. Process the article: first checking for an updated DOI via the Semantic Scholar API, then retrieving article metadata (title, publication date, author information, funding) via the OpenAlex API
4. Process the repository: retrieving repository metadata (README content, description, stargazers count, contributor information) via the GitHub API
5. Match the retrieved article authors to the retrieved repository developer accounts
6. Store all retrieved and processed information in the database, deduplicating where possible

### Contributing

If you are interested in contributing to the data processing library, please reach out via email at evamxb@uw.edu or open an issue on the GitHub repository. We would love to collaborate with you and help you get started. We may already be working on something you are interested in and would love to have you join us!