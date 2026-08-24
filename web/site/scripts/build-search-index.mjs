// Stages the raw pairs payload data-prep writes to search-data/ (from
// web/data-prep/output/search_articles_repos.json -- see web/README.md) into
// public/search-index/ for the "Explore the Data" page to fetch.
//
// No index-build step: an earlier version built a MiniSearch inverted index
// here, but that added real size on top of the raw records (145MB built vs.
// 65.9MB raw for the old 544,128-entity-record shape -- the tokenizing/trie
// structure itself is the overhead, not the text). Eva asked (2026-08-23) to
// drop fuzzy/prefix search in favor of plain substring matching directly
// against the raw records instead. The substring-match logic lives
// client-side in explore/index.astro; this script just copies the file into
// place.
//
// Payload shape (see search_index.py, rewritten 2026-08-23 for the
// Papers-with-Code-style pairs redesign):
//   { fields: string[], records: [title, doi, cited_by_count, publication_date, field_idx|null, repo_full_name, stars][] }
// records are one row per high-precision document<->repository pair (not
// per-entity like the old shape), pre-sorted by publication_date descending.
import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SEARCH_DATA_DIR = join(__dirname, '..', 'search-data');
const OUTPUT_DIR = join(__dirname, '..', 'public', 'search-index');

mkdirSync(OUTPUT_DIR, { recursive: true });

const srcPath = join(SEARCH_DATA_DIR, 'search_articles_repos.json');
const destPath = join(OUTPUT_DIR, 'articles-repos.json');
copyFileSync(srcPath, destPath);

console.log(`Staged search records: ${destPath}`);
