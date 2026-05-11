# =========================
# topic_modelling.py
# Multilingual BERTopic pipeline — year → cluster → topics.
#
# Pipeline:
#   1. For each year, run a coarse HDBSCAN pre-clustering on the year's
#      full embedding matrix (UMAP-reduced for speed).
#   2. For each cluster (plus an optional noise-doc group), run BERTopic
#      independently to discover fine-grained topics.
#
# Output structure:
#   {
#     "generated_at": "...",
#     "model_type":   "bertopic",
#     "parameters":   { ... },
#     "years": {
#       "<year>": {
#         "n_docs":      int,
#         "n_clusters":  int,
#         "noise_ratio": float,   ← share of docs unclustered at the year level
#         "clusters": [
#           {
#             "cluster_id": int,          ← -1 = noise bucket
#             "n_docs":     int,
#             "n_topics":   int,
#             "coherence":  float,        ← c_v across this cluster's topics
#             "topics": [
#               {
#                 "topic_id":        int,
#                 "label":           str,
#                 "count":           int,
#                 "top_words":       [str, ...],
#                 "top_word_scores": [float, ...]
#               }, ...
#             ],
#             "error": str              ← only on failure
#           }
#         ],
#         "error": str                  ← only on full-year failure
#       }
#     }
#   }
# =========================

import numpy as np
import pandas as pd

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import config
from utils import save_json, timestamp, get_logger

logger = get_logger("topic_modelling")


# =========================================================
# COHERENCE  (c_v via gensim CoherenceModel)
# =========================================================

def compute_coherence(docs: list, topic_words: list) -> float:
    """
    Compute corpus-level c_v coherence across all topics.
    Returns 0.0 if there are no topics or on any failure.
    """
    if not topic_words:
        return 0.0
    try:
        tokenized  = [doc.split() for doc in docs]
        dictionary = Dictionary(tokenized)
        model = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        return model.get_coherence()
    except Exception as e:
        logger.warning(f"Coherence computation failed: {e}")
        return 0.0


def generate_topic_label(top_words: list) -> str:
    return " / ".join(top_words[:3]).title()


# =========================================================
# PRE-CLUSTERING  (coarse HDBSCAN on the year's UMAP embedding)
# =========================================================

def pre_cluster_year(embeddings: np.ndarray, n_docs: int) -> np.ndarray:
    """
    Run a coarse UMAP + HDBSCAN pass to split a year's docs into
    broad thematic clusters before per-cluster topic modelling.

    Returns an array of integer cluster labels (length == n_docs).
    -1 labels are noise docs.
    """
    # UMAP reduction — keep more neighbours than BERTopic's internal pass
    # so the geometry is coarser / more spread out.
    n_neighbors = min(config.PRE_CLUSTER_MIN_CLUSTER_SIZE, n_docs - 1)
    n_components = min(config.UMAP_COMPONENTS, n_docs - 2)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.1,        # slight spread keeps clusters separable
        metric="cosine",
        random_state=42,
    )
    reduced = umap_model.fit_transform(embeddings)

    hdbscan_model = HDBSCAN(
        min_cluster_size=config.PRE_CLUSTER_MIN_CLUSTER_SIZE,
        min_samples=config.PRE_CLUSTER_MIN_SAMPLES,
        metric="euclidean",
        prediction_data=False,
    )
    labels = hdbscan_model.fit_predict(reduced)
    return labels


# =========================================================
# PER-CLUSTER BERTOPIC
# =========================================================

def run_bertopic_on_cluster(
    docs: list,
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    vectorizer_model: CountVectorizer,
    mts: int,
    nr: int,
) -> list:
    """
    Fit a BERTopic model on a single cluster's docs and embeddings.
    Returns a list of topic dicts (topic_id, label, count, top_words, …).
    """
    n = len(docs)
    if n < config.MIN_DOCS_PER_CLUSTER:
        logger.info(f"    cluster too small ({n} docs) — skipping BERTopic")
        return []

    umap_model = UMAP(
        n_neighbors=min(config.UMAP_NEIGHBORS, n - 1),
        n_components=min(config.UMAP_COMPONENTS, n - 2),
        min_dist=config.UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.MIN_CLUSTER_SIZE,
        min_samples=config.MIN_SAMPLES,
        metric="euclidean",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=KeyBERTInspired(),
        min_topic_size=mts,
        nr_topics=nr,
        calculate_probabilities=True,
        verbose=False,
    )

    topic_model.fit_transform(docs, embeddings)
    topic_info = topic_model.get_topic_info()

    topic_results = []
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue  # BERTopic's internal noise — skip

        words      = topic_model.get_topic(tid)
        top_words  = [w for w, _ in words[: config.TOP_N_WORDS]]
        top_scores = [round(float(s), 6) for _, s in words[: config.TOP_N_WORDS]]

        topic_results.append({
            "topic_id":        tid,
            "label":           generate_topic_label(top_words),
            "count":           int(row["Count"]),
            "top_words":       top_words,
            "top_word_scores": top_scores,
        })

    return topic_results


# =========================================================
# MAIN
# =========================================================

def main(min_topic_size: int = None, nr_topics: int = None) -> dict:
    """
    Run the year → cluster → BERTopic pipeline.

    Parameters
    ----------
    min_topic_size : overrides config.BERTOPIC_MIN_TOPIC_SIZE
    nr_topics      : overrides config.BERTOPIC_NR_TOPICS
    """
    mts = min_topic_size if min_topic_size is not None else config.BERTOPIC_MIN_TOPIC_SIZE
    nr  = nr_topics      if nr_topics      is not None else config.BERTOPIC_NR_TOPICS

    logger.info("Loading embeddings from preprocessed pickle...")
    df = pd.read_pickle(config.CLUSTERED_FILE)
    df = df.dropna(subset=["processed"])

    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

    year_results: dict[str, dict] = {}

    for year in sorted(df["year"].unique()):
        year_df    = df[df["year"] == year].copy()
        docs       = year_df["processed"].tolist()
        embeddings = np.array(year_df["embedding"].tolist())
        n_docs     = len(docs)

        logger.info(f"\n══ Year {year}: {n_docs} docs ══")

        if n_docs < config.PRE_CLUSTER_MIN_CLUSTER_SIZE * 2:
            logger.info(f"  too few docs for pre-clustering — skipping year")
            year_results[str(year)] = {
                "n_docs":      n_docs,
                "n_clusters":  0,
                "noise_ratio": 1.0,
                "clusters":    [],
                "error":       f"Only {n_docs} docs — below minimum for clustering",
            }
            continue

        # ── Step 1: coarse pre-clustering ───────────────────────────────
        try:
            cluster_labels = pre_cluster_year(embeddings, n_docs)
        except Exception as e:
            logger.error(f"  Pre-clustering failed for {year}: {e}")
            year_results[str(year)] = {
                "n_docs":      n_docs,
                "n_clusters":  0,
                "noise_ratio": 1.0,
                "clusters":    [],
                "error":       str(e),
            }
            continue

        unique_cluster_ids = sorted(set(cluster_labels))
        noise_count = int(np.sum(cluster_labels == -1))
        noise_ratio = round(noise_count / n_docs, 4)

        real_cluster_ids = [c for c in unique_cluster_ids if c != -1]
        logger.info(
            f"  Pre-clustering → {len(real_cluster_ids)} clusters, "
            f"noise={noise_ratio:.2%}"
        )

        # ── Step 2: BERTopic inside each cluster ─────────────────────────
        clusters_out = []

        # Process each real cluster
        for cid in real_cluster_ids:
            mask        = cluster_labels == cid
            c_docs      = [d for d, m in zip(docs, mask) if m]
            c_embeddings = embeddings[mask]

            logger.info(f"  ── Cluster {cid}: {len(c_docs)} docs ──")

            # Fresh vectorizer per cluster (vocab differs between clusters)
            vectorizer = CountVectorizer(
                ngram_range=config.NGRAM_RANGE,
                stop_words=list(),
                min_df=config.MIN_DF,
                max_df=config.MAX_DF,
            )

            try:
                topics = run_bertopic_on_cluster(
                    docs=c_docs,
                    embeddings=c_embeddings,
                    embedding_model=embedding_model,
                    vectorizer_model=vectorizer,
                    mts=mts,
                    nr=nr,
                )
                coherence_words = [t["top_words"] for t in topics]
                coherence = compute_coherence(c_docs, coherence_words)

                logger.info(
                    f"    topics={len(topics)}  coherence={coherence:.4f}"
                )
                clusters_out.append({
                    "cluster_id": int(cid),
                    "n_docs":     len(c_docs),
                    "n_topics":   len(topics),
                    "coherence":  round(coherence, 4),
                    "topics":     topics,
                })
            except Exception as e:
                logger.error(f"    BERTopic failed for cluster {cid}: {e}")
                clusters_out.append({
                    "cluster_id": int(cid),
                    "n_docs":     len(c_docs),
                    "n_topics":   0,
                    "coherence":  0.0,
                    "topics":     [],
                    "error":      str(e),
                })

        # Optionally process noise docs as a single pseudo-cluster
        if config.INCLUDE_NOISE_CLUSTER and noise_count >= config.MIN_DOCS_PER_CLUSTER:
            noise_mask        = cluster_labels == -1
            noise_docs        = [d for d, m in zip(docs, noise_mask) if m]
            noise_embeddings  = embeddings[noise_mask]

            logger.info(f"  ── Noise bucket: {noise_count} docs ──")
            vectorizer = CountVectorizer(
                ngram_range=config.NGRAM_RANGE,
                stop_words=list(),
                min_df=config.MIN_DF,
                max_df=config.MAX_DF,
            )
            try:
                topics = run_bertopic_on_cluster(
                    docs=noise_docs,
                    embeddings=noise_embeddings,
                    embedding_model=embedding_model,
                    vectorizer_model=vectorizer,
                    mts=mts,
                    nr=nr,
                )
                coherence_words = [t["top_words"] for t in topics]
                coherence = compute_coherence(noise_docs, coherence_words)

                clusters_out.append({
                    "cluster_id": -1,
                    "n_docs":     noise_count,
                    "n_topics":   len(topics),
                    "coherence":  round(coherence, 4),
                    "topics":     topics,
                })
            except Exception as e:
                logger.error(f"    BERTopic on noise bucket failed: {e}")
                clusters_out.append({
                    "cluster_id": -1,
                    "n_docs":     noise_count,
                    "n_topics":   0,
                    "coherence":  0.0,
                    "topics":     [],
                    "error":      str(e),
                })

        year_results[str(year)] = {
            "n_docs":      n_docs,
            "n_clusters":  len(real_cluster_ids),
            "noise_ratio": noise_ratio,
            "clusters":    clusters_out,
        }

    output = {
        "generated_at": timestamp(),
        "model_type":   config.TOPIC_MODEL_TYPE,
        "parameters": {
            "pre_cluster_min_size": config.PRE_CLUSTER_MIN_CLUSTER_SIZE,
            "pre_cluster_min_samples": config.PRE_CLUSTER_MIN_SAMPLES,
            "min_topic_size": mts,
            "nr_topics":      nr,
            "top_n_words":    config.TOP_N_WORDS,
            "ngram_range":    list(config.NGRAM_RANGE),
        },
        "years": year_results,
    }

    save_json(output, config.TOPIC_OUTPUT_FILE)
    logger.info(f"\nTopic results saved → {config.TOPIC_OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    main()
