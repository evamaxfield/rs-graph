#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path

from ..data import DATA_FILES_DIR


@dataclass(frozen=True)
class DatabasePaths:
    version: str
    prod: Path
    dev: Path


V2_DATABASE_PATHS = DatabasePaths(
    version="v2",
    prod=(DATA_FILES_DIR / "rs-graph-v2-prod.db").resolve(),
    dev=(DATA_FILES_DIR / "rs-graph-v2-dev.db").resolve(),
)

V1_DATABASE_PATHS = DatabasePaths(
    version="v1",
    prod=(DATA_FILES_DIR / "rs-graph-v1-prod.db").resolve(),
    dev=(DATA_FILES_DIR / "rs-graph-v1-dev.db").resolve(),
)
