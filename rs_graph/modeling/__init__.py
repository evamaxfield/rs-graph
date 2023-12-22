"""Stored model loaders."""

from functools import partial

from skops import io as sk_io
from sklearn.linear_model import LogisticRegressionCV
from sentence_transformers import SentenceTransformer
import pandas as pd

from ..data import DATA_FILES_DIR

###############################################################################

# Trained models
AUTHOR_DEV_EM_MODEL_PATH = DATA_FILES_DIR / "author-dev-em-model.skops"

# Default global values
DEFAULT_AUTHOR_DEV_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

###############################################################################

def load_author_dev_em_model() -> LogisticRegressionCV:
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
        dev_embedding_values[key] = embedding_model.encode(
            dev_details[key],
            show_progress_bar=False,
        ).tolist()
    
    # Get author value embeddings
    author_embedding_values = {}
    for key in [
        "name",
        "co_authors",
    ]:
        author_embedding_values[key] = embedding_model.encode(
            author_details[key],
            show_progress_bar=False,
        ).tolist()
    
    # Create interactions by creating pairwise combinations
    this_x = {}
    for dev_key, dev_value in dev_embedding_values.items():
        for author_key, author_value in author_embedding_values.items():
            pairwise_key = f"dev-{dev_key}---author-{author_key}"
            for i, dev_i_v in enumerate(dev_value):
                this_x[f"{pairwise_key}---dim-{i}"] = dev_i_v * author_value[i]

    return classifier.predict(list(this_x.values()))[0]

def load_ready_to_use_em_model() -> partial:
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