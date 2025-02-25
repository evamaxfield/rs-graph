#!/usr/bin/env python

from ..data import DATA_FILES_DIR

PROD_DATABASE_FILEPATH = (DATA_FILES_DIR / "rs-graph-v1-prod.db").resolve()
DEV_DATABASE_FILEPATH = (DATA_FILES_DIR / "rs-graph-v1-dev.db").resolve()
