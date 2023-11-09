# Research Software Graph

🚀 [Rendered Interactive Graph](https://evamaxfield.github.io/rs-graph/) 🚀

Graphing the dependencies (and dependents) of research software and their contributors.

## Setup

* Install [Just](https://github.com/casey/just#packages)
* Create a new Conda / Mamba Env (Python 3.11 preferred)
* Install [pylbfgs](https://github.com/larsmans/pylbfgs): `conda install -c fgregg pylbfgs`
* Install lib: `just install` (or `pip install .`)

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
    3. **TODO:** Train author name dedupe model:
        `rs-graph-modeling train-author-name-dedupe-model`
    4. **TODO:** Train github users dedupe model:
        `rs-graph-modeling train-github-users-dedupe-model`
    5. **TODO:** Train author to github user linker model:
        `rs-graph-modeling train-author-to-github-user-linker-model`