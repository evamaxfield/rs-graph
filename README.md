# Research Software Graph

Graphing the dependencies (and dependents) of research software and their contributors.

## Current Status

Work in progress.

Completed:

- [x] Created dataset from JOSS API (~2100 papers with their repository URL)
- [x] Implemented functions to get the upstream dependencies from any GitHub Repository using the GitHub SBOM
- [x] Cached the results of the above function to the repository. Can be loaded with: `rs_graph.data.load_one_hop_joss_deps`
  - Short results for ONE HOP dependencies (the dependencies of the repository itself, not the recursive dependencies)
    - 5,719 unique npm packages (~30,000 total dependency relations)
    - 2,853 unique PyPI packages (~21,000 total dependency relations)
    - 704 unique cargo packages (~2,000 total dependency relations)
    - 489 unique maven packages (~800 total dependency relations)
    - 452 unique actions packages (~7,000 total dependency relations)
    - 358 unique rubygems packages (~800 total dependency relations)
    - 138 unique composer packages (~150 total dependency relations)
    - 118 unique go packages (~150 total dependency relations)
    - 82 unique nuget packages (~100 total dependency relations)
  - NOTE: R / CRAN is not supported by GitHub SBOM
  - I think we should filter down to npm, PyPI, and cargo for now
- [ ] Visualize the dependency graph for intermediate results
- [ ] Write wrappers around different registry APIs to return the source code repository URL for a given package name
- [ ] Write recursive logic to get the entire dependency graph for a given repository
