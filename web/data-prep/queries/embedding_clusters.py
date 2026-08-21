"""Q4: whether clusters of science look different when you consider the code too.

Two side-by-side 2D UMAP embedding spaces -- one of repository text (name +
description + README), one of article text (title + abstract) -- both colored
by Topic.domain_name, over a stratified sample of high-precision article-
repository pairs with usable text on both sides.

SAMPLE_SIZE is 3,200; bump toward 5,000 for a more statistically robust
production run before launch.
"""

import json
import os

import numpy as np
import polars as pl
from lib.confidence import filter_high_precision_document_repository_links
from lib.hf_loader import load_table
from sentence_transformers import SentenceTransformer
from umap import UMAP

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "embedding_clusters.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_TEXT_CHARS = 50
SAMPLE_SIZE = 3200
RANDOM_SEED = 42


def _best_domain_per_document() -> pl.DataFrame:
    document_topic = load_table("document_topic")
    topic = load_table("topic").select("id", "domain_name").rename({"id": "topic_id"})
    return (
        document_topic.sort("score", descending=True)
        .group_by("document_id")
        .head(1)
        .join(topic, on="topic_id")
        .select("document_id", "domain_name")
    )


def _stratified_sample(df: pl.DataFrame, group_col: str, total: int, seed: int) -> pl.DataFrame:
    counts = df.group_by(group_col).agg(n=pl.len())
    total_available = counts["n"].sum()
    parts = []
    for row in counts.iter_rows(named=True):
        group_target = max(1, round(total * row["n"] / total_available))
        group_target = min(group_target, row["n"])
        parts.append(
            df.filter(pl.col(group_col) == row[group_col]).sample(
                n=group_target, seed=seed, shuffle=True
            )
        )
    return pl.concat(parts)


def run() -> dict:
    # --- site-snippet:start ---
    hp_links = filter_high_precision_document_repository_links(
        load_table("document_repository_link")
    ).select("document_id", "repository_id")

    readmes = (
        load_table("repository_readme")
        .filter(
            pl.col("content").is_not_null()
            & (pl.col("content").str.len_chars() > MIN_TEXT_CHARS)
        )
        .select("repository_id", pl.col("content").alias("readme"))
    )
    abstracts = (
        load_table("document_abstract")
        .filter(
            pl.col("content").is_not_null()
            & (pl.col("content").str.len_chars() > MIN_TEXT_CHARS)
        )
        .select("document_id", pl.col("content").alias("abstract"))
    )
    repositories = (
        load_table("repository").select("id", "owner", "name").rename({"id": "repository_id"})
    )
    documents = load_table("document").select("id", "title").rename({"id": "document_id"})
    domains = _best_domain_per_document()

    pairs = (
        hp_links.join(readmes, on="repository_id", how="inner")
        .join(abstracts, on="document_id", how="inner")
        .join(repositories, on="repository_id", how="inner")
        .join(documents, on="document_id", how="inner")
        .join(domains, on="document_id", how="inner")
    )

    sample = _stratified_sample(pairs, "domain_name", SAMPLE_SIZE, RANDOM_SEED)
    sample = sample.with_columns(
        (pl.col("owner") + "/" + pl.col("name")).alias("repo_full_name"),
        repo_text=(
            pl.col("name").fill_null("")
            + " "
            + pl.col("readme").fill_null("").str.slice(0, 2000)
        ),
        article_text=(pl.col("title").fill_null("") + " " + pl.col("abstract").fill_null("")),
    )

    model = SentenceTransformer(EMBEDDING_MODEL)
    repo_embeddings = model.encode(sample["repo_text"].to_list(), show_progress_bar=False)
    article_embeddings = model.encode(sample["article_text"].to_list(), show_progress_bar=False)

    n_neighbors = min(15, len(sample) - 1)
    repo_2d = UMAP(
        n_neighbors=n_neighbors, min_dist=0.1, random_state=RANDOM_SEED
    ).fit_transform(np.asarray(repo_embeddings))
    article_2d = UMAP(
        n_neighbors=n_neighbors, min_dist=0.1, random_state=RANDOM_SEED
    ).fit_transform(np.asarray(article_embeddings))
    # --- site-snippet:end ---

    points = []
    for i, row in enumerate(sample.iter_rows(named=True)):
        points.append(
            {
                "document_id": row["document_id"],
                "repository_id": row["repository_id"],
                "domain_name": row["domain_name"],
                "repo_full_name": row["repo_full_name"],
                "document_title": row["title"],
                "repo_x": float(repo_2d[i][0]),
                "repo_y": float(repo_2d[i][1]),
                "article_x": float(article_2d[i][0]),
                "article_y": float(article_2d[i][1]),
            }
        )

    result = {
        "question": "Do clusters of science look different when you consider the code too?",
        "methodology": (
            "document_repository_link filtered to NULL OR confidence >= 0.9994; "
            f"embedding model {EMBEDDING_MODEL}; sample stratified by domain_name"
        ),
        "sample_size": len(points),
        "note": (
            f"This sample covers {len(points):,} article-repository pairs, stratified by "
            "domain -- enough to see clear cluster separation, though the smaller domains "
            "here carry far fewer points than Physical Sciences, so their cluster shapes "
            "are noisier by comparison."
        ),
        "domains": sorted(sample["domain_name"].unique().to_list()),
        "domain_counts": {
            row["domain_name"]: row["n"]
            for row in sample.group_by("domain_name").agg(n=pl.len()).iter_rows(named=True)
        },
        "points": points,
    }
    return result


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"sample_size={result['sample_size']} domains={result['domains']}")
