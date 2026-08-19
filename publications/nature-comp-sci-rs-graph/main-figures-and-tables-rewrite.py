#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import evaplot
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import typer
from data_utils import load_base_dataset, load_table

###############################################################################

app = typer.Typer()

###############################################################################

@app.command()