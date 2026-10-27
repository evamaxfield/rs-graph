# rs-graph web

Docs/marketing site for the rs-graph dataset. See
`experiments-comms/memory/2026-08-20-rs-graph-docs-site-plan.md` for the full design doc.

## Layout

- `data-prep/` -- Python. Queries the gated `sci-soft-collections/rs-graph-v2-full` on HuggingFace
  (the public release is `sci-soft-collections/rs-graph-v2`; the site's loading snippet and
  viewer links point there) and writes small, committed JSON files to `data-prep/output/`.
  Nothing heavy runs in the browser or in CI.
- `site/` -- Astro app. Imports `data-prep/output/*.json` (staged into `site/public/data/`) and
  renders charts with Observable Plot. Zero JS shipped per page except the chart islands.

## Regenerating data

```bash
cd web/data-prep
python3 -m venv .venv
.venv/bin/pip install -e .
# requires HF_TOKEN in rs-graph/.env
# PYTHONPATH makes `import load_rs_graph` and `lib.*` resolve
export PYTHONPATH=.:../../rs_graph/release/templates
# optional: pin the dataset to a commit sha or tag (default: main)
export RS_GRAPH_REVISION=main
# optional: repo the load_rs_graph-based queries read (default: the -full dataset)
export RS_GRAPH_REPO_ID=sci-soft-collections/rs-graph-v2-full
.venv/bin/python queries/non_author_contributors.py
.venv/bin/python queries/top_libraries.py
.venv/bin/python queries/ml_tooling_adoption.py
.venv/bin/python queries/home_stats.py
.venv/bin/python queries/home_samples.py
.venv/bin/python queries/license_distribution.py   # read from output/ by Docs, not staged
.venv/bin/python queries/search_index.py           # staged by `npm run build-search-index`
.venv/bin/python extract_snippets.py   # regenerates site/src/generated/*.py.txt (also copies rs_graph/release/templates/load_rs_graph.py)

# stage the JSON the site imports from public/data/
cp output/{home_stats,home_samples,non_author_contributors,top_libraries,ml_tooling_adoption}.json ../site/public/data/
```

## Running the site locally

Requires Node 22+ (this repo's dev machine defaults to an older system Node -- use nvm to select
a newer one if `astro dev` complains about engine support).

```bash
cd web/site
npm ci
npm run dev
npm run build   # outputs to site/dist/
```

## Deploying

`.github/workflows/deploy-web.yml` builds `site/` and publishes to `gh-pages`, but ships
`workflow_dispatch`-only -- no automatic trigger yet. See the workflow file for how to wire up a
`push` trigger when ready to go live.
