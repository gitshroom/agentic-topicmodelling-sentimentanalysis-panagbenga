# =========================
# topic_modelling.py
# Agent: discovers topics per cluster using BERTopic or LDA.
# Called by orchestrator.py with optional parameter overrides.
# =========================

import argparse
import ast
import pandas as pd
from utils import get_logger, save_json, timestamp, log_banner
import config

logger = get_logger("topic_modelling")


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------

def run_bertopic(df: pd.DataFrame, min_topic_size: int, nr_topics) -> dict:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    log_banner(logger, "BERTopic topic modelling")
    results = {}

    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue

        subset = df[df["cluster"] == cluster_id]
        docs = subset["processed"].tolist()

        if len(docs) < min_topic_size:
            logger.warning(f"Cluster {cluster_id}: too few docs ({len(docs)}), skipping.")
            continue

        logger.info(f"Cluster {cluster_id}: {len(docs)} docs")

        try:
            model = BERTopic(
                min_topic_size=min_topic_size,
                nr_topics=nr_topics,
                verbose=False,
            )
            topics, probs = model.fit_transform(docs)
            topic_info = model.get_topic_info()

            cluster_topics = []
            for _, row in topic_info.iterrows():
                tid = row["Topic"]
                if tid == -1:
                    continue
                words = model.get_topic(tid)
                cluster_topics.append({
                    "topic_id": int(tid),
                    "count": int(row["Count"]),
                    "top_words": [w for w, _ in words[:config.TOP_N_WORDS]],
                    "top_word_scores": [round(float(s), 4) for _, s in words[:config.TOP_N_WORDS]],
                    "label": row.get("Name", f"topic_{tid}"),
                })

            # coherence proxy: avg of top-word scores in the top topic
            coherence_proxy = 0.0
            if cluster_topics:
                coherence_proxy = round(
                    sum(cluster_topics[0]["top_word_scores"]) / len(cluster_topics[0]["top_word_scores"]), 4
                )

            noise_count = sum(1 for t in topics if t == -1)
            noise_ratio = round(noise_count / len(topics), 4) if topics else 1.0

            results[str(cluster_id)] = {
                "cluster_id": int(cluster_id),
                "n_docs": len(docs),
                "n_topics": len(cluster_topics),
                "noise_ratio": noise_ratio,
                "coherence_proxy": coherence_proxy,
                "topics": cluster_topics,
            }

        except Exception as e:
            logger.error(f"Cluster {cluster_id} BERTopic failed: {e}")
            results[str(cluster_id)] = {"cluster_id": int(cluster_id), "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------

def run_lda(df: pd.DataFrame, n_topics: int) -> dict:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import numpy as np

    log_banner(logger, "LDA topic modelling")
    results = {}

    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue

        subset = df[df["cluster"] == cluster_id]
        docs = subset["processed"].tolist()

        if len(docs) < 5:
            logger.warning(f"Cluster {cluster_id}: too few docs ({len(docs)}), skipping.")
            continue

        logger.info(f"Cluster {cluster_id}: {len(docs)} docs — {n_topics} LDA topics")

        try:
            vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=500)
            dtm = vectorizer.fit_transform(docs)
            feature_names = vectorizer.get_feature_names_out()

            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(docs) - 1),
                max_iter=config.LDA_MAX_ITER,
                learning_method="online",
                random_state=42,
            )
            lda.fit(dtm)

            cluster_topics = []
            for tid, component in enumerate(lda.components_):
                top_idx = component.argsort()[-config.TOP_N_WORDS:][::-1]
                top_words = [feature_names[i] for i in top_idx]
                top_scores = [round(float(component[i]), 4) for i in top_idx]
                cluster_topics.append({
                    "topic_id": tid,
                    "top_words": top_words,
                    "top_word_scores": top_scores,
                })

            doc_topics = lda.transform(dtm)
            dominant = doc_topics.argmax(axis=1)
            counts = {tid: int(np.sum(dominant == tid)) for tid in range(n_topics)}
            for t in cluster_topics:
                t["count"] = counts.get(t["topic_id"], 0)

            results[str(cluster_id)] = {
                "cluster_id": int(cluster_id),
                "n_docs": len(docs),
                "n_topics": len(cluster_topics),
                "noise_ratio": 0.0,       # LDA assigns all docs
                "coherence_proxy": 0.0,   # LDA doesn't expose per-topic coherence easily
                "topics": cluster_topics,
            }

        except Exception as e:
            logger.error(f"Cluster {cluster_id} LDA failed: {e}")
            results[str(cluster_id)] = {"cluster_id": int(cluster_id), "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(min_topic_size: int = None, nr_topics=None, n_lda_topics: int = None):
    logger.info("Loading clustered dataset...")
    df = pd.read_csv(config.CLUSTERED_FILE)
    df = df.dropna(subset=["processed"])
    logger.info(f"Loaded {len(df)} rows, {df['cluster'].nunique()} clusters")

    model_type = config.TOPIC_MODEL_TYPE

    if model_type == "bertopic":
        mts = min_topic_size if min_topic_size is not None else config.BERTOPIC_MIN_TOPIC_SIZE
        nrt = nr_topics if nr_topics is not None else config.BERTOPIC_NR_TOPICS
        results = run_bertopic(df, min_topic_size=mts, nr_topics=nrt)
    else:
        ntop = n_lda_topics if n_lda_topics is not None else config.LDA_N_TOPICS
        results = run_lda(df, n_topics=ntop)

    output = {
        "generated_at": timestamp(),
        "model_type": model_type,
        "parameters": {
            "min_topic_size": min_topic_size,
            "nr_topics": str(nr_topics),
            "n_lda_topics": n_lda_topics,
        },
        "clusters": results,
    }

    save_json(output, config.TOPIC_OUTPUT_FILE)
    logger.info(f"Topic results saved to {config.TOPIC_OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topic modelling agent")
    parser.add_argument("--min_topic_size", type=int, default=None)
    parser.add_argument("--nr_topics", default=None)
    parser.add_argument("--n_lda_topics", type=int, default=None)
    args = parser.parse_args()

    nr = args.nr_topics
    if nr is not None and nr.isdigit():
        nr = int(nr)

    main(min_topic_size=args.min_topic_size, nr_topics=nr, n_lda_topics=args.n_lda_topics)
