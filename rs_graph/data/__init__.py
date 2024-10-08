"""Stored dataset loaders."""

from __future__ import annotations

from pathlib import Path

###############################################################################
# Remote storage paths

GCS_PROJECT_ID = "sci-software-graph"
REMOTE_STORAGE_BUCKET = "gs://sci-software-graph-data-store"

###############################################################################
# Local storage paths

DATA_FILES_DIR = Path(__file__).parent / "files"
