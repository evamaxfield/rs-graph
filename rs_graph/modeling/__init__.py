"""Stored model loaders."""

from collections.abc import Iterable
from functools import partial

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from skops import io as sk_io

from ..data import DATA_FILES_DIR

###############################################################################

# Trained models
AUTHOR_DEV_EM_MODEL_PATH = DATA_FILES_DIR / "author-dev-em-model.skops"

# Default global values
DEFAULT_AUTHOR_DEV_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

###############################################################################


def load_base_author_dev_em_model() -> LogisticRegressionCV:
    """Load the author dev em model."""
    return sk_io.load(AUTHOR_DEV_EM_MODEL_PATH)


def predict_author_dev_match(
    author_details: pd.Series,
    dev_details: pd.Series,
    embedding_model: SentenceTransformer,
    classifier: LogisticRegressionCV,
) -> bool:
    """Predict if author and dev match."""
    # Get dev value embeddings
    dev_embedding_values = {}
    for key in [
        "username",
        "name",
        "email",
        "co_contributors",
    ]:
        # Get value
        dev_value = dev_details[key]

        # If value is iterable, join it
        if isinstance(dev_value, Iterable):
            dev_value = ", ".join(dev_value)

        # If value is None, stringify
        if dev_value is None:
            dev_value = "None"

        # Create embedding
        dev_embedding_values[key] = embedding_model.encode(
            dev_value,
            show_progress_bar=False,
        ).tolist()

    # Get author value embeddings
    author_embedding_values = {}
    for key in [
        "name",
        "co_authors",
    ]:
        # Get value
        author_value = author_details[key]

        # If value is iterable, join it
        if isinstance(author_value, Iterable):
            author_value = ", ".join(author_value)

        # If value is None, stringify
        if author_value is None:
            author_value = "None"

        # Create embedding
        author_embedding_values[key] = embedding_model.encode(
            author_value,
            show_progress_bar=False,
        ).tolist()

    # Create interactions by creating pairwise combinations
    this_x = {}
    for dev_key, dev_key_embed_dim_values in dev_embedding_values.items():
        for (
            author_key,
            author_key_embed_dim_values,
        ) in author_embedding_values.items():
            pairwise_key = f"dev-{dev_key}---author-{author_key}"
            for (
                embed_dim_index,
                dev_key_embed_dim_value,
            ) in enumerate(dev_key_embed_dim_values):
                this_x[f"{pairwise_key}---dim-{embed_dim_index}"] = (
                    dev_key_embed_dim_value
                    * author_key_embed_dim_values[embed_dim_index]
                )

    # Create pd series
    return classifier.predict(pd.DataFrame([this_x]))[0]


def load_ready_to_use_author_dev_em_clf() -> partial:
    """Load the author dev em model."""
    # Load classifier
    clf = sk_io.load(AUTHOR_DEV_EM_MODEL_PATH)

    # Load embedding model
    embedding_model = SentenceTransformer(DEFAULT_AUTHOR_DEV_EMBEDDING_MODEL_NAME)

    # Create partial function
    return partial(
        predict_author_dev_match,
        embedding_model=embedding_model,
        classifier=clf,
    )
