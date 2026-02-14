#!/usr/bin/env python

"""
Generate the input parquet file for the snowball sampling discovery pipeline.

Queries the database for researcher-developer account links matching the given
filters and writes a parquet file that can be passed directly to the
snowball_sampling_discovery command.
"""

from datetime import datetime

import polars as pl
import typer
from sqlmodel import Session, col, or_, select

from rs_graph.data import DATA_FILES_DIR
from rs_graph.db import models as db_models
from rs_graph.db.utils import get_engine
from rs_graph.utils.dt_and_td import parse_timedelta

###############################################################################

app = typer.Typer()

###############################################################################


@app.command()
def generate_snowball_sampling_input(
    iteration: int = typer.Argument(
        help=(
            "Iteration / batch ID for this snowball sampling run. "
            "Stored in the output parquet and later in DocumentRepositoryLink.iteration."
        ),
    ),
    researcher_developer_links_filter_confidence_threshold: float = 0.97,
    researcher_developer_links_duration_since_last_process_filter: str | None = None,
    top_unique: bool = True,
    use_prod: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Generate a parquet file of researcher-developer account link IDs
    for use as input to the snowball sampling discovery pipeline.
    """
    output_path = DATA_FILES_DIR / f"snowball-sampling-batch-iteration-{iteration}.parquet"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    engine = get_engine(use_prod=use_prod)

    with Session(engine) as session:
        # 3-way join: Link -> Researcher -> DeveloperAccount
        stmt = (
            select(
                db_models.ResearcherDeveloperAccountLink,
                db_models.Researcher,
                db_models.DeveloperAccount,
            )
            .join(db_models.Researcher)
            .join(db_models.DeveloperAccount)
            .where(
                col(db_models.ResearcherDeveloperAccountLink.predictive_model_confidence)
                >= researcher_developer_links_filter_confidence_threshold
            )
        )

        # Apply datetime filtering if specified
        if researcher_developer_links_duration_since_last_process_filter is not None:
            cutoff_datetime = datetime.now() - parse_timedelta(
                researcher_developer_links_duration_since_last_process_filter
            )
            stmt = stmt.where(
                or_(
                    col(
                        db_models.ResearcherDeveloperAccountLink.last_snowball_processed_datetime
                    ).is_(None),
                    col(
                        db_models.ResearcherDeveloperAccountLink.last_snowball_processed_datetime
                    )
                    < cutoff_datetime,
                )
            )

        results = session.exec(stmt).all()

    # Build the output dataframe
    rows = []
    for link, researcher, developer_account in results:
        assert link.id is not None
        rows.append(
            {
                "researcher_developer_account_link_id": link.id,
                "iteration": iteration,
                "researcher_name": researcher.name,
                "researcher_orcid": researcher.orcid,
                "developer_account_username": developer_account.username,
                "predictive_model_confidence": link.predictive_model_confidence,
                "researcher_id": link.researcher_id,
                "developer_account_id": link.developer_account_id,
            }
        )

    df = pl.DataFrame(rows)

    # Deduplicate to one link per researcher and one link per developer,
    # keeping the highest-confidence link in each case.
    if top_unique:
        df = (
            df.sort("predictive_model_confidence", descending=True)
            .unique(subset=["researcher_id"], keep="first")
            .unique(subset=["developer_account_id"], keep="first")
        )

    df = df.drop("predictive_model_confidence", "researcher_id", "developer_account_id")

    # Write to parquet
    df.write_parquet(output_path)

    print(f"Wrote {len(df)} researcher-developer links to {output_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    app()
