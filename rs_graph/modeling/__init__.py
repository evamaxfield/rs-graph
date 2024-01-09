"""Stored model loaders."""

import logging
from collections.abc import Sequence

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from skops import io as sk_io

from ..data import DATA_FILES_DIR

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# Trained models
DEV_AUTHOR_EM_MODEL_PATH = DATA_FILES_DIR / "dev-author-em-model.skops"

# Default global values
DEFAULT_DEV_AUTHOR_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

###############################################################################


class GitHubUserModelingColumns:
    username = "username"
    name = "name"
    email = "email"


ALL_GITHUB_USER_MODELING_COLUMNS = [
    v for k, v in GitHubUserModelingColumns.__dict__.items() if "__" not in k
]


class AuthorModelingColumns:
    name = "name"


ALL_AUTHOR_MODELING_COLUMNS = [
    v for k, v in AuthorModelingColumns.__dict__.items() if "__" not in k
]

###############################################################################


def load_base_dev_author_em_model() -> LogisticRegressionCV:
    """Load the author dev em model."""
    return sk_io.load(DEV_AUTHOR_EM_MODEL_PATH)


def _check_and_init_dev_author_interaction_embedding(
    devs: Sequence[pd.Series]
    | Sequence[dict[str, str | None]]
    | pd.Series
    | dict[str, str | None],
    authors: Sequence[pd.Series]
    | Sequence[dict[str, str | None]]
    | pd.Series
    | dict[str, str | None],
    embedding_model: str
    | SentenceTransformer = DEFAULT_DEV_AUTHOR_EMBEDDING_MODEL_NAME,
) -> tuple[
    list[dict[str, str | None]],
    list[dict[str, str | None]],
    SentenceTransformer,
]:
    # Wrap single dev and author in list
    if isinstance(devs, pd.Series | dict):
        devs = [devs]
    if isinstance(authors, pd.Series | dict):
        authors = [authors]

    # Load embedding model
    if isinstance(embedding_model, str):
        embed_model = SentenceTransformer(embedding_model)
    else:
        embed_model = embedding_model

    # Convert to series if dict
    if isinstance(devs[0], pd.Series):
        devs = [d.to_dict() for d in devs]  # type: ignore
    if isinstance(authors[0], pd.Series):
        authors = [a.to_dict() for a in authors]  # type: ignore

    # Assert that all required modeling keys are present
    # Find any missing keys and report them back
    if any(set(ALL_GITHUB_USER_MODELING_COLUMNS) - set(dev.keys()) for dev in devs):
        raise ValueError(
            "Dev is missing required dev keys. Be sure to include: "
            f"{', '.join(ALL_GITHUB_USER_MODELING_COLUMNS)}"
        )
    if any(set(ALL_AUTHOR_MODELING_COLUMNS) - set(author.keys()) for author in authors):
        raise ValueError(
            "Author is missing required author keys. Be sure to include: "
            f"{', '.join(ALL_AUTHOR_MODELING_COLUMNS)}"
        )

    # Assert that devs and authors are the same size
    if len(devs) != len(authors):
        raise ValueError(
            f"Devs and authors must be the same size. "
            f"Got {len(devs)} devs and {len(authors)} authors."
        )

    return devs, authors, embed_model  # type: ignore


def get_dev_author_interaction_embeddings(
    devs: Sequence[pd.Series]
    | Sequence[dict[str, str | None]]
    | pd.Series
    | dict[str, str | None],
    authors: Sequence[pd.Series]
    | Sequence[dict[str, str | None]]
    | pd.Series
    | dict[str, str | None],
    embedding_model: str
    | SentenceTransformer = DEFAULT_DEV_AUTHOR_EMBEDDING_MODEL_NAME,
) -> list[list[float]]:
    # Check and initialize
    devs, authors, embed_model = _check_and_init_dev_author_interaction_embedding(
        devs=devs,
        authors=authors,
        embedding_model=embedding_model,
    )

    # Create pairwise string pairs to encode
    dev_author_str_pairs: dict[str, dict[str, list[str]]] = {}
    for dev, author in zip(devs, authors, strict=True):
        for dev_key, dev_value in dev.items():
            # Ignore fields that aren't found in training fields
            if dev_key not in ALL_GITHUB_USER_MODELING_COLUMNS:
                continue

            for author_key, author_value in author.items():
                # Ignore fields that aren't found in training fields
                if author_key not in ALL_AUTHOR_MODELING_COLUMNS:
                    continue

                # Create str pair
                interaction_embed_name = f"dev-{dev_key}--author-{author_key}"

                # Create list if not already
                if interaction_embed_name not in dev_author_str_pairs:
                    dev_author_str_pairs[interaction_embed_name] = {
                        "dev": [],
                        "author": [],
                    }

                # Add to list
                dev_author_str_pairs[interaction_embed_name]["dev"].append(
                    str(dev_value)
                )
                dev_author_str_pairs[interaction_embed_name]["author"].append(
                    str(author_value)
                )

    # Encode all pairs
    dev_author_interaction_embeddings = {}
    for interaction_embed_name, interaction_embed_pairs in dev_author_str_pairs.items():
        log.info(f"Encoding: '{interaction_embed_name}'")

        # Encode
        encode_dev_values = embed_model.encode(
            interaction_embed_pairs["dev"],
            show_progress_bar=True,
        )
        encode_author_values = embed_model.encode(
            interaction_embed_pairs["author"],
            show_progress_bar=True,
        )

        # Multiply
        interaction_embeddings = encode_dev_values * encode_author_values

        # Store
        dev_author_interaction_embeddings[
            interaction_embed_name
        ] = interaction_embeddings.tolist()

    # Concat all embeddings into a single list
    joint_interaction_embeddings = []
    for i in range(len(devs)):
        this_dev_author_interaction_embeddings = []
        for interaction_embed_name in dev_author_interaction_embeddings.keys():
            this_dev_author_interaction_embeddings.extend(
                dev_author_interaction_embeddings[interaction_embed_name][i]
            )

        joint_interaction_embeddings.append(this_dev_author_interaction_embeddings)

    return joint_interaction_embeddings


def predict_dev_author_matches(
    devs: Sequence[pd.Series]
    | Sequence[dict[str, str | None]]
    | pd.Series
    | dict[str, str | None],
    authors: Sequence[pd.Series]
    | Sequence[dict[str, str | None]]
    | pd.Series
    | dict[str, str | None],
    embedding_model: str
    | SentenceTransformer = DEFAULT_DEV_AUTHOR_EMBEDDING_MODEL_NAME,
    clf_model: LogisticRegressionCV | None = None,
) -> list[str] | str:
    # Check and initialize
    devs, authors, embed_model = _check_and_init_dev_author_interaction_embedding(
        devs=devs,
        authors=authors,
        embedding_model=embedding_model,
    )

    # Get interaction embeddings
    joint_interaction_embeddings = get_dev_author_interaction_embeddings(
        devs=devs,
        authors=authors,
        embedding_model=embed_model,
    )

    # Load model if not provided
    if clf_model is None:
        clf_model = load_base_dev_author_em_model()

    # Predict
    predictions = clf_model.predict(joint_interaction_embeddings)

    # If only one prediction, return as string
    if len(predictions) == 1:
        return predictions[0]

    return predictions.tolist()
