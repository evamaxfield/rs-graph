# Research Software Graph

Graphing the dependencies (and dependents) of research software and their contributors.

## Current Status

Work in progress.

Completed:

- [x] Created dataset from JOSS API (~2100 papers with their repository URL)
- [x] Implemented functions to get the upstream dependencies from any GitHub Repository using the GitHub SBOM
- [x] Cached the results of the above function to the repository.
  - NOTE: R / CRAN is not supported by GitHub SBOM
- [ ] Visualize the dependency graph for intermediate results
- [ ] Write wrappers around different registry APIs to return the source code repository URL for a given package name
- [ ] Write recursive logic to get the entire dependency graph for a given repository

### Update 2023-10-03

Nic and I got the SoftwareX journal articles added to the dataset. We now have ~2,600 papers with their repository URL.

Updated short results:
- ~6,800 unique npm packages (~47,000 total dependency relations)
- ~3,200 unique PyPI packages (~25,000 total dependency relations)
- ~700 unique cargo packages (~2,000 total dependency relations)

Full data can be loaded using: `rs_graph.data.load_rs_graph_upstream_deps_dataset`