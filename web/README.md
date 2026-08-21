# rs-graph web

Docs/marketing site for the rs-graph dataset. See
`experiments-comms/memory/2026-08-20-rs-graph-docs-site-plan.md` for the full design doc.

## Layout

- `data-prep/` -- Python. Queries `evamxb/rs-graph-v2-full` on HuggingFace and writes small,
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
PYTHONPATH=. .venv/bin/python queries/embedding_clusters.py   # slow -- embeds + UMAPs a sample
PYTHONPATH=. .venv/bin/python queries/repo_size_vs_impact.py  # AI4Science bonus card
PYTHONPATH=. .venv/bin/python queries/home_stats.py
.venv/bin/python extract_snippets.py   # regenerates site/src/generated/*.py.txt

# stage the new output for the site
cp output/*.json ../site/public/data/
```

`embedding_clusters.py`'s `SAMPLE_SIZE` constant is currently 3,200 -- the plan doc recommends up
to 5,000 for a more statistically robust production run.

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
