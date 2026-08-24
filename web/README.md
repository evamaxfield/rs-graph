# rs-graph web

Docs/marketing site for the rs-graph dataset. See
`experiments-comms/memory/2026-08-20-rs-graph-docs-site-plan.md` for the full design doc.

## Layout

- `data-prep/` -- Python. Queries `sci-soft-collections/rs-graph-v2-full` on HuggingFace and writes small,
  committed JSON files to `data-prep/output/`. Nothing heavy runs in the browser or in CI.
- `site/` -- Astro app. Imports `data-prep/output/*.json` (staged into `site/public/data/`) and
  renders charts with Observable Plot. Zero JS shipped per page except the chart islands.

## Regenerating data

```bash
cd web/data-prep
python3 -m venv .venv
.venv/bin/pip install -e .
# requires HF_TOKEN in rs-graph/.env
PYTHONPATH=. .venv/bin/python queries/dependency_manifest_growth.py
PYTHONPATH=. .venv/bin/python queries/non_author_contributors.py
PYTHONPATH=. .venv/bin/python queries/top_libraries.py
PYTHONPATH=. .venv/bin/python queries/library_cross_view.py
PYTHONPATH=. .venv/bin/python queries/ml_tooling_adoption.py
PYTHONPATH=. .venv/bin/python queries/coauthorship_network.py
PYTHONPATH=. .venv/bin/python queries/embedding_clusters.py   # slow -- embeds + UMAPs a sample; NOT rendered on the site (see below)
PYTHONPATH=. .venv/bin/python queries/home_stats.py
.venv/bin/python extract_snippets.py   # regenerates site/src/generated/*.py.txt

# stage the new output for the site
cp output/*.json ../site/public/data/
```

`embedding_clusters.py`'s `SAMPLE_SIZE` constant is currently 3,200 -- the plan doc recommends up
to 5,000 for a more statistically robust production run. Its output is **not currently wired into
the site** -- the "code vs. text clusters" question it backed was cut (2026-08-23, see
`experiments-comms/memory/2026-08-23-rs-graph-docs-site-q3-rework-q4-cut-coauthorship.md`); the
script is left in place in case that question comes back, but running it is optional for a normal
site regen.

## Running the site locally

Requires Node 20+ (this repo's dev machine defaults to an older system Node -- use nvm to select
a newer one if `astro dev` complains about engine support).

```bash
cd web/site
npm install
npm run dev
npm run build   # outputs to site/dist/
```

## Deploying

`.github/workflows/deploy-web.yml` builds `site/` and publishes to `gh-pages`, but ships
`workflow_dispatch`-only -- no automatic trigger yet. See the workflow file for how to wire up a
`push` trigger when ready to go live.
