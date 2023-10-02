"""Preprocessing and availability of different datasets."""

from pathlib import Path

import pandas as pd

###############################################################################

DATA_FILES_DIR = Path(__file__).parent / "files"
JOSS_DATASET_PATH = DATA_FILES_DIR / "joss-2023-10-02.parquet"

###############################################################################

def load_joss() -> pd.DataFrame:
    """Load the JOSS dataset."""
    return pd.read_parquet(JOSS_DATASET_PATH)