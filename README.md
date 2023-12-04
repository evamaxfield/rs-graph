# Research Software Graph

🚀 [Rendered Interactive Graph](https://evamaxfield.github.io/rs-graph/) 🚀

Graphing the dependencies (and dependents) of research software and their contributors.

## Setup

* Install [Just](https://github.com/casey/just#packages)
* Create a new Conda / Mamba Env (Python 3.11 preferred)
* Install [pylbfgs](https://github.com/larsmans/pylbfgs): `conda install -c fgregg pylbfgs`
* Install lib: `just install` (or `pip install .`)

### Data Download

There is a lot of pre-processed data stored on a GCS bucket. To download it, run:

```bash
rs-graph-data download
```

This will always download the latest version of the data (as pushed by @evamaxfield).

## Order of Data Operations

If this project was starting from scratch, the order of operations to recreate our work
would be:

**Note:** at each step of the process, the data or model is cached
to the installed package directory (`rs_graph.data.files`).

1. Get all data sources using the command: `rs-graph-sources get-all`
2. Options
    1. Get all recursive upstream dependencies for the repos:
        `rs-graph-enrichment get-upstreams`
    2. Get all of the extended paper details for the papers:
        `rs-graph-enrichment get-extended-paper-details`
    3. Get top 30 contributors to each repository:
        `rs-graph-enrichment get-repo-contributors`
    4.  Train github users dedupe model:
        `rs-graph-modeling train-developer-deduper`
    5. Train author to github user linker model:
        `rs-graph-modeling train-author-developer-linker`


## GitHub User and Author Entity Matching Model

### Annotation

In order to construct an annotation set, we generated an embedding
for each GitHub user and author using the basic information we had available
for them (for GitHub users this includes their 
username, the repos of interest, their provided name, their co-contributors, etc.; 
for authors this included their name, the repos of interest, and their co-authors).
We then created an annotation dataset by getting the top 5 most similar authors for each
GitHub user in our set. We then manually annotated each of these pairs as either a 
match or not a match.

In general, our annotation criteria stated that we needed at least two points of 
agreement between the two entities to consider them a match. For example, if the 
GitHub user and the author had the same name and had a repo in common, we would 
consider them a match; if they had the same name and a co-contributor's GitHub username 
seemed similar to a co-author's name, we would consider them a match.
