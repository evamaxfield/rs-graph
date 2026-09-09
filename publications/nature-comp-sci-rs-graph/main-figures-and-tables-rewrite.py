#!/usr/bin/env python3

"""
Nature Computational Science `rs-graph` manuscript -- figures and tables.

Thin CLI shell: every command lives in a topic module in this directory and is registered
here. Every command loads directly from HuggingFace
(`sci-soft-collections/rs-graph-v2-full`), applies the standard filters itself, and is
runnable standalone -- no dependency on a local SQLite checkout. Builds Figures 1-4,
Table 1, the mention-predictors logistic regression, and the supporting statistics/tables
cited in the manuscript text.
"""

from __future__ import annotations

import typer
from alignment_iou_trends import (
    import_dependency_iou_over_time,
    mentions_alignment_over_time,
)
from armm_verification import (
    armm_model_diagnostics,
    classification_models_table_verification,
)
from figure1_quadpartite_network import figure_1_quadpartite_network
from figure2_dataset_coverage import figure_2_dataset_coverage
from figure3_development_characteristics import (
    figure_3_software_development_characteristics,
    fwsi_fwci_comparison_table,
    supplemental_manifest_file_adoption,
    supplemental_package_vs_script_diagnostics,
)
from mention_analysis import (
    figure_4_mention_rate_by_field_and_year,
    mentions_coverage_by_year,
    predictors_of_software_mentioning,
)
from network_statistics import (
    coauthorship_network,
    date_delta_figure,
    network_entity_edge_counts,
)
from tables_and_counts import (
    data_coverage_counts,
    median_repository_contributor_count,
    mining_rounds_table,
    table1_top_software_by_usage,
)

###############################################################################

app = typer.Typer()

app.command()(figure_1_quadpartite_network)
app.command()(figure_2_dataset_coverage)
app.command()(figure_3_software_development_characteristics)
app.command()(fwsi_fwci_comparison_table)
app.command()(figure_4_mention_rate_by_field_and_year)
app.command()(predictors_of_software_mentioning)
app.command()(table1_top_software_by_usage)
app.command()(median_repository_contributor_count)
app.command()(classification_models_table_verification)
app.command()(armm_model_diagnostics)
app.command()(date_delta_figure)
app.command()(mining_rounds_table)
app.command()(coauthorship_network)
app.command()(network_entity_edge_counts)
app.command()(import_dependency_iou_over_time)
app.command()(mentions_coverage_by_year)
app.command()(mentions_alignment_over_time)
app.command()(supplemental_manifest_file_adoption)
app.command()(supplemental_package_vs_script_diagnostics)
app.command()(data_coverage_counts)

###############################################################################

if __name__ == "__main__":
    app()
