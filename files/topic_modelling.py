# =========================
# topic_modelling.py
# Agent: discovers topics per cluster, organised by year.
# Output structure:  { year: { cluster_id: { topics, coherence, … } } }
# Called by orchestrator.py with optional parameter overrides.
# =========================

import argparse
import pandas as pd
from utils import get_logger, save_json, timestamp, log_banner
import config

logger = get_logger("topic_modelling")


# ---------------------------------------------------------------------------
# BERTopic — single year-cluster slice
# ---------------------------------------------------------------------------

def run_bertopic_slice(
    docs: list[str],
    cluster_id: int,
    year: int,
    min_topic_size: int,
    nr_topics,
) -> dict:
    from bertopic import BERTopic

    if len(docs) < min_topic_size:
        logger.warning(f"  [{year}][cluster {cluster_id}] only {len(docs)} docs — skipping.")
        return {"cluster_id": cluster_id, "skipped": True, "n_docs": len(docs)}

    try:
        model = BERTopic(min_topic_size=min_topic_size, nr_topics=nr_topics, verbose=False)
        topics, _ = model.fit_transform(docs)
        topic_info = model.get_topic_info()

        cluster_topics = []
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                continue
            words = model.get_topic(tid)
            cluster_topics.append({
                "topic_id":        int(tid),
                "count":           int(row["Count"]),
                "top_words":       [w for w, _ in words[: config.TOP_N_WORDS]],
                "top_word_scores": [round(float(s), 4) for _, s in words[: config.TOP_N_WORDS]],
                "label":           row.get("Name", f"topic_{tid}"),
            })

        coherence_proxy = 0.0
        if cluster_topics:
            scores = cluster_topics[0]["top_word_scores"]
            coherence_proxy = round(sum(scores) / len(scores), 4)

        noise_count = sum(1 for t in topics if t == -1)
        noise_ratio = round(noise_count / len(topics), 4) if topics else 1.0

        return {
            "cluster_id":      cluster_id,
            "n_docs":          len(docs),
            "n_topics":        len(cluster_topics),
            "noise_ratio":     noise_ratio,
            "coherence_proxy": coherence_proxy,
            "topics":          cluster_topics,
        }

    except Exception as e:
        logger.error(f"  [{year}][cluster {cluster_id}] BERTopic failed: {e}")
        return {"cluster_id": cluster_id, "error": str(e), "n_docs": len(docs)}


# ---------------------------------------------------------------------------
# LDA — single year-cluster slice
# ---------------------------------------------------------------------------

def run_lda_slice(
    docs: list[str],
    cluster_id: int,
    year: int,
    n_topics: int,
) -> dict:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import numpy as np

    if len(docs) < 5:
        logger.warning(f"  [{year}][cluster {cluster_id}] only {len(docs)} docs — skipping.")
        return {"cluster_id": cluster_id, "skipped": True, "n_docs": len(docs)}

    try:
        vectorizer  = CountVectorizer(max_df=0.9, min_df=2, max_features=500)
        dtm         = vectorizer.fit_transform(docs)
        feat_names  = vectorizer.get_feature_names_out()

        n_comp = min(n_topics, len(docs) - 1)
        lda = LatentDirichletAllocation(
            n_components=n_comp,
            max_iter=config.LDA_MAX_ITER,
            learning_method="online",
            random_state=42,
        )
        lda.fit(dtm)

        cluster_topics = []
        for tid, component in enumerate(lda.components_):
            top_idx = component.argsort()[-config.TOP_N_WORDS :][::-1]
            cluster_topics.append({
                "topic_id":        tid,
                "top_words":       [feat_names[i] for i in top_idx],
                "top_word_scores": [round(float(component[i]), 4) for i in top_idx],
            })

        doc_topics = lda.transform(dtm)
        dominant   = doc_topics.argmax(axis=1)
        counts     = {tid: int(np.sum(dominant == tid)) for tid in range(n_comp)}
        for t in cluster_topics:
            t["count"] = counts.get(t["topic_id"], 0)

        return {
            "cluster_id":      cluster_id,
            "n_docs":          len(docs),
            "n_topics":        len(cluster_topics),
            "noise_ratio":     0.0,
            "coherence_proxy": 0.0,
            "topics":          cluster_topics,
        }

    except Exception as e:
        logger.error(f"  [{year}][cluster {cluster_id}] LDA failed: {e}")
        return {"cluster_id": cluster_id, "error": str(e), "n_docs": len(docs)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    min_topic_size: int = None,
    nr_topics=None,
    n_lda_topics: int = None,
) -> dict:
    log_banner(logger, "Topic modelling agent — per year")

    df = pd.read_csv(config.CLUSTERED_FILE)
    df = df.dropna(subset=["processed"])
    if "year" not in df.columns:
        raise ValueError("'year' column missing — run preprocessing.py first.")
    df["year"] = df["year"].astype(int)
    logger.info(f"Loaded {len(df)} rows | years: {sorted(df['year'].unique())}")

    model_type = config.TOPIC_MODEL_TYPE
    mts  = min_topic_size if min_topic_size is not None else config.BERTOPIC_MIN_TOPIC_SIZE
    nrt  = nr_topics      if nr_topics      is not None else config.BERTOPIC_NR_TOPICS
    ntop = n_lda_topics   if n_lda_topics   is not None else config.LDA_N_TOPICS

    all_years: dict[str, dict] = {}

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year]
        logger.info(f"\n── Year {year}: {len(year_df)} docs ──")

        year_results: dict[str, dict] = {}

        for cluster_id in sorted(year_df["cluster"].unique()):
            if cluster_id == -1:
                continue  # skip noise cluster

            subset = year_df[year_df["cluster"] == cluster_id]
            docs   = subset["processed"].tolist()
            logger.info(f"  cluster {cluster_id}: {len(docs)} docs")

            if model_type == "bertopic":
                result = run_bertopic_slice(docs, cluster_id, year, mts, nrt)
            else:
                result = run_lda_slice(docs, cluster_id, year, ntop)

            year_results[str(cluster_id)] = result

        all_years[str(year)] = year_results

    output = {
        "generated_at": timestamp(),
        "model_type":   model_type,
        "parameters": {
            "min_topic_size": mts,
            "nr_topics":      str(nrt),
            "n_lda_topics":   ntop,
        },
        "years": all_years,
    }

    save_json(output, config.TOPIC_OUTPUT_FILE)
    logger.info(f"\nTopic results saved → {config.TOPIC_OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topic modelling agent (per year)")
    parser.add_argument("--min_topic_size", type=int, default=None)
    parser.add_argument("--nr_topics",      default=None)
    parser.add_argument("--n_lda_topics",   type=int, default=None)
    args = parser.parse_args()

    nr = args.nr_topics
    if nr is not None and nr.isdigit():
        nr = int(nr)

    main(min_topic_size=args.min_topic_size, nr_topics=nr, n_lda_topics=args.n_lda_topics)
