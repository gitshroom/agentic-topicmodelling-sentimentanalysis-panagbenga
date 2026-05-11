# =========================
# topic_modelling.py
# Multilingual BERTopic pipeline — one model per year.
#
# Output structure:
#   {
#     "generated_at": "...",
#     "model_type":   "bertopic",
#     "parameters":   { ... },
#     "years": {
#       "<year>": {
#         "n_docs":          int,
#         "n_topics":        int,
#         "noise_ratio":     float,
#         "avg_coherence":   float,   ← mean of per-topic c_v scores
#         "topics": [
#           {
#             "topic_id":        int,
#             "label":           str,
#             "count":           int,
#             "top_words":       [str, ...],
#             "top_word_scores": [float, ...],
#             "coherence":       float   ← per-topic c_v score
#           }, ...
#         ],
#         "error": str                  ← only present on failure
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

def compute_coherence_per_topic(docs: list, topic_words: list) -> list:
    """
    Compute per-topic c_v coherence scores.
    Returns a list of floats aligned with topic_words.
    Returns 0.0 for any topic that fails.
    """
    if not topic_words:
        return []
    try:
        tokenized  = [doc.split() for doc in docs]
        dictionary = Dictionary(tokenized)
        scores = []
        for words in topic_words:
            try:
                model = CoherenceModel(
                    topics=[words],
                    texts=tokenized,
                    dictionary=dictionary,
                    coherence="c_v",
                )
                scores.append(round(model.get_coherence(), 4))
            except Exception as e:
                logger.warning(f"Per-topic coherence failed: {e}")
                scores.append(0.0)
        return scores
    except Exception as e:
        logger.warning(f"Coherence computation failed: {e}")
        return [0.0] * len(topic_words)


def generate_topic_label(top_words: list) -> str:
    return " / ".join(top_words[:3]).title()


# =========================================================
# MAIN
# =========================================================

def main(min_topic_size: int = None, nr_topics: int = None) -> dict:
    """
    Run one BERTopic model per year.

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
    df["year"] = df["year"].astype(int)

    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

    vectorizer_model = CountVectorizer(
        ngram_range=config.NGRAM_RANGE,
        stop_words=list(),
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
    )

    representation_model = KeyBERTInspired()

    year_results: dict[str, dict] = {}

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year]
        docs    = year_df["processed"].tolist()

        if len(docs) < config.MIN_DOCS_PER_YEAR:
            logger.info(f"[{year}] only {len(docs)} docs — skipping (< MIN_DOCS_PER_YEAR={config.MIN_DOCS_PER_YEAR})")
            continue

        logger.info(f"\n── Year {year}: {len(docs)} docs ──")
        embeddings = np.array(year_df["embedding"].tolist())

        umap_model = UMAP(
            n_neighbors=min(config.UMAP_NEIGHBORS, len(docs) - 1),
            n_components=min(config.UMAP_COMPONENTS, len(docs) - 2),
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

        try:
            topic_model = BERTopic(
                embedding_model=embedding_model,
                vectorizer_model=vectorizer_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                representation_model=representation_model,
                min_topic_size=mts,
                nr_topics=nr,
                calculate_probabilities=True,
                verbose=False,
            )

            topics_assigned, _ = topic_model.fit_transform(docs, embeddings)
            topic_info         = topic_model.get_topic_info()

            # Collect top words per topic first (needed for coherence batch)
            raw_topics = []
            for _, row in topic_info.iterrows():
                tid = int(row["Topic"])
                if tid == -1:
                    continue
                words      = topic_model.get_topic(tid)
                top_words  = [w for w, _ in words[: config.TOP_N_WORDS]]
                top_scores = [round(float(s), 6) for _, s in words[: config.TOP_N_WORDS]]
                raw_topics.append({
                    "topic_id":        tid,
                    "count":           int(row["Count"]),
                    "top_words":       top_words,
                    "top_word_scores": top_scores,
                })

            # Compute per-topic coherence in one pass
            all_top_words    = [t["top_words"] for t in raw_topics]
            per_topic_scores = compute_coherence_per_topic(docs, all_top_words)

            topic_results = []
            for t, coh in zip(raw_topics, per_topic_scores):
                topic_results.append({
                    "topic_id":        t["topic_id"],
                    "label":           generate_topic_label(t["top_words"]),
                    "count":           t["count"],
                    "top_words":       t["top_words"],
                    "top_word_scores": t["top_word_scores"],
                    "coherence":       coh,
                })

            # Year-level average coherence across all topics
            valid_scores  = [t["coherence"] for t in topic_results if t["coherence"] > 0]
            avg_coherence = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0

            noise_ratio = round(
                sum(1 for t in topics_assigned if t == -1) / len(topics_assigned), 4
            )

            logger.info(
                f"  topics={len(topic_results)}  "
                f"avg_coherence={avg_coherence:.4f}  "
                f"noise={noise_ratio:.4f}"
            )

            year_results[str(year)] = {
                "n_docs":        len(docs),
                "n_topics":      len(topic_results),
                "noise_ratio":   noise_ratio,
                "avg_coherence": avg_coherence,
                "topics":        topic_results,
            }

        except Exception as e:
            logger.error(f"  BERTopic failed for year {year}: {e}")
            year_results[str(year)] = {
                "n_docs":        len(docs),
                "n_topics":      0,
                "noise_ratio":   1.0,
                "avg_coherence": 0.0,
                "topics":        [],
                "error":         str(e),
            }

    output = {
        "generated_at": timestamp(),
        "model_type":   config.TOPIC_MODEL_TYPE,
        "parameters": {
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