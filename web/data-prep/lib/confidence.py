"""Confidence thresholds and the publication-year floor for rs-graph.

Two different confidence fields, two different tables, two different
thresholds -- never conflate them. See web/site's /methodology page for the
full rationale.
"""

import polars as pl

# document_repository_link.predictive_model_confidence:
# NULL = curated/pre-linked (no model involved), else >= this gives ~0.99 precision.
DOCUMENT_REPOSITORY_LINK_THRESHOLD = 0.9994

# researcher_developer_account_link.predictive_model_confidence:
# every row is model-predicted; this retains ~97% of links while dropping the uncertain tail.
RESEARCHER_DEVELOPER_ACCOUNT_LINK_THRESHOLD = 0.9

# GitHub launched February 2008 -- any document/repository data implying an
# earlier date for linked code is a data-encoding artifact (likely OpenAlex-side),
# not a real pre-GitHub repository. Applied wherever a publication or repo
# creation year is used as an axis or filter.
PUBLICATION_YEAR_FLOOR = 2008


def filter_high_precision_document_repository_links(df: pl.DataFrame) -> pl.DataFrame:
    """Restrict a document_repository_link frame to the high-precision subset."""
    return df.filter(
        pl.col("predictive_model_confidence").is_null()
        | (pl.col("predictive_model_confidence") >= DOCUMENT_REPOSITORY_LINK_THRESHOLD)
    )


def filter_high_confidence_researcher_developer_links(df: pl.DataFrame) -> pl.DataFrame:
    """Restrict a researcher_developer_account_link frame to the high-confidence subset."""
    return df.filter(
        pl.col("predictive_model_confidence") >= RESEARCHER_DEVELOPER_ACCOUNT_LINK_THRESHOLD
    )
