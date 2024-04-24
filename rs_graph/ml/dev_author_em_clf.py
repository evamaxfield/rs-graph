#!/usr/bin/env python

import logging

import xarray as xr
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from skops import io as skio

from ..data import DATA_FILES_DIR
from ..types import DeveloperDetails

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEV_AUTHOR_EM_CLASSIFIER_PATH = DATA_FILES_DIR / "dev-author-em-clf.skops"
DEV_AUTHOR_EM_EMBEDDING_MODEL = "microsoft/deberta-v3-base"
DEV_AUTHOR_EM_TEMPLATE = """
### Developer Details

username: {dev_username}
name: {dev_name}
email: {dev_email}

---

### Author Details

name: {author_name}
""".strip()

###############################################################################


def load_dev_author_em_model() -> LogisticRegressionCV:
    """
    Load the author EM model.

    Returns
    -------
    LogisticRegressionCV
        The author EM model
    """
    import warnings

    warnings.filterwarnings("ignore", module="sklearn")
    return skio.load(DEV_AUTHOR_EM_CLASSIFIER_PATH, trusted=True)


def load_dev_author_em_embedding_model() -> SentenceTransformer:
    """
    Load the author EM embedding model.

    Returns
    -------
    SentenceTransformer
        The author EM embedding model
    """
    import warnings

    warnings.filterwarnings("ignore", module="transformers")

    sentence_tfs_logger = logging.getLogger("sentence_transformers")
    sentence_tfs_logger.setLevel(logging.ERROR)

    # Return model
    return SentenceTransformer(DEV_AUTHOR_EM_EMBEDDING_MODEL)


def match_devs_and_authors(
    devs: list[DeveloperDetails],
    authors: list[str],
    loaded_dev_author_em_model: LogisticRegressionCV | None = None,
    loaded_embedding_model: SentenceTransformer | None = None,
) -> dict[str, str]:
    """
    Embed developers and authors and predict matches.

    Only use this on a set of authors and developers
    from the same repo-document pair. This will not work globally.

    Parameters
    ----------
    devs : list[DeveloperDetails]
        The developers to embed.
    authors : list[str]
        The authors to embed.
    loaded_dev_author_em_model : LogisticRegressionCV, optional
        The loaded author EM model, by default None
    loaded_embedding_model : SentenceTransformer, optional
        The loaded embedding model, by default None

    Returns
    -------
    dict[str, str]
        The predicted matches
    """
    # If no loaded classifer, load the model
    if loaded_dev_author_em_model is None:
        clf = load_dev_author_em_model()
    else:
        clf = loaded_dev_author_em_model

    # If no loaded embedding model, load the model
    if loaded_embedding_model is None:
        embed_model = load_dev_author_em_embedding_model()
    else:
        embed_model = loaded_embedding_model

    # Create xarray for filled templates
    data = []
    for dev in devs:
        this_dev_data = []
        for author in authors:
            this_dev_data.append(
                DEV_AUTHOR_EM_TEMPLATE.format(
                    dev_username=dev.username,
                    dev_name=dev.name,
                    dev_email=dev.email,
                    author_name=author,
                )
            )

        data.append(this_dev_data)

    # Construct the array
    arr = xr.DataArray(
        data,
        dims=("dev", "author"),
        coords={"dev": [dev.username for dev in devs], "author": authors},
    )

    # Create embeddings for all of the pairs
    log.debug("Creating embeddings for all dev-author pairs")
    embeddings = embed_model.encode(arr.values.flatten(), show_progress_bar=False)

    # Reshape the embeddings to fit the array
    embeddings = embeddings.reshape(len(devs), len(authors), -1)

    # Make a new xarray with the embeddings
    arr = xr.DataArray(
        embeddings,
        dims=("dev", "author", "embedding_dim"),
        coords={"dev": [dev.username for dev in devs], "author": authors},
    )

    # Predict the matches
    log.debug("Predicting matches")
    preds = clf.predict(arr.values.reshape(-1, arr.sizes["embedding_dim"]))

    # Reshape the predictions to fit the array
    preds = preds.reshape(len(devs), len(authors))

    # Make a new xarray with the predictions
    arr = xr.DataArray(
        preds,
        dims=("dev", "author"),
        coords={"dev": [dev.username for dev in devs], "author": authors},
    )

    # Create a dict of matches where the dev is the key and the author is the value
    # Only add those pairs to the dict if the prediction is "match"
    matches = {}
    for dev in devs:
        for author in authors:
            if arr.sel(dev=dev.username, author=author).values == "match":
                matches[dev.username] = author

    return matches
