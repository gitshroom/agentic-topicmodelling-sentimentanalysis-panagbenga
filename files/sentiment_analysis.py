# =========================
# sentiment_analysis.py
# Agent: runs sentiment analysis per TOPIC per YEAR.
#
# For each year → cluster → topic, we collect the docs assigned to that topic
# and score them.  The output mirrors the year→cluster→topic structure of
# topic_modelling.py so results_formatter can merge them easily.
#
# Called by orchestrator.py with optional parameter overrides.
# Run standalone:  python sentiment_analysis.py
# =========================

import argparse
import pandas as pd
from utils import get_logger, save_json, timestamp, log_banner
import config

logger = get_logger("sentiment_analysis")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_texts(texts: list, pipeline, batch_size: int) -> list:
    """Run the HuggingFace pipeline in batches. Returns list of {label, score}."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            out = pipeline(batch, truncation=True, max_length=512)
            results.extend(out)
        except Exception as e:
            logger.error(f"Batch {i}–{i + batch_size} failed: {e}")
            results.extend([{"label": "ERROR", "score": 0.0}] * len(batch))
    return results


def aggregate_sentiment(records: list) -> dict:
    """Compute label distribution and mean confidence."""
    label_counts: dict[str, int] = {}
    total_score = 0.0
    valid = 0

    for r in records:
        lbl = r.get("label", "UNKNOWN")
        sc  = r.get("score", 0.0)
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        if lbl not in ("ERROR", "UNKNOWN"):
            total_score += sc
            valid += 1

    n              = len(records)
    avg_confidence = round(total_score / valid, 4) if valid else 0.0
    coverage       = round(valid / n, 4)           if n     else 0.0
    dominant       = max(label_counts, key=label_counts.get) if label_counts else "UNKNOWN"

    return {
        "n_docs":             n,
        "coverage":           coverage,
        "avg_confidence":     avg_confidence,
        "dominant_sentiment": dominant,
        "label_distribution": label_counts,
        "label_ratios":       {k: round(v / n, 4) for k, v in label_counts.items()},
    }


# ---------------------------------------------------------------------------
# Topic-level sentiment
# ---------------------------------------------------------------------------

def score_topic(
    topic_entry: dict,
    year_df: pd.DataFrame,
    cluster_id: int,
    pipe,
    batch_size: int,
    text_col: str,
) -> dict:
    """
    Given a topic dict from topic_modelling output (with 'topic_id', 'count',
    'top_words'), find the matching docs in year_df and score them.

    Topic assignment: BERTopic assigns each doc a topic_id stored in a
    'topic' column that we add during this step.  For LDA we derive it from
    the topic with highest doc-topic probability.

    Because we don't persist per-doc topic assignments from topic_modelling,
    we re-run a lightweight topic assignment here using just the top-word
    vocabulary overlap — a simple but fast proxy.
    """
    topic_id   = topic_entry["topic_id"]
    top_words  = set(topic_entry.get("top_words", []))
    cluster_df = year_df[year_df["cluster"] == cluster_id].copy()

    if cluster_df.empty or not top_words:
        return {**aggregate_sentiment([]), "topic_id": topic_id, "per_doc": []}

    # Score each doc by how many top words appear in its processed text
    cluster_df["_overlap"] = cluster_df["processed"].apply(
        lambda t: sum(1 for w in top_words if w in str(t).split())
    )

    # Assign docs to the topic with the highest overlap — simplified assignment
    # when there is only one topic per cluster this just takes all docs.
    topic_docs = cluster_df[cluster_df["_overlap"] > 0]
    if topic_docs.empty:
        topic_docs = cluster_df  # fall back to all cluster docs

    texts  = topic_docs[text_col].tolist()
    scored = score_texts(texts, pipe, batch_size)

    per_doc = []
    for (_, row), s in zip(topic_docs.iterrows(), scored):
        per_doc.append({
            "id":    row.get("id"),
            "label": s["label"],
            "score": round(float(s["score"]), 4),
        })

    agg = aggregate_sentiment(scored)
    return {**agg, "topic_id": topic_id, "per_doc": per_doc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_name: str = None, batch_size: int = None) -> dict:
    from transformers import pipeline as hf_pipeline
    from utils import load_json

    log_banner(logger, "Sentiment analysis agent — per year/topic")

    mdl      = model_name or config.SENTIMENT_MODEL
    bs       = batch_size or config.SENTIMENT_BATCH_SIZE
    text_col = "processed" if config.SENTIMENT_USE_PROCESSED else "text"

    logger.info(f"Model: {mdl}  |  Batch size: {bs}  |  Column: {text_col}")

    # Load clustered data (has 'year' and 'cluster' columns)
    df = pd.read_csv(config.CLUSTERED_FILE)
    df = df.dropna(subset=[text_col])
    if "year" not in df.columns:
        raise ValueError("'year' column missing — run preprocessing.py first.")
    df["year"] = df["year"].astype(int)
    logger.info(f"Loaded {len(df)} rows")

    # Load topic results to know which topics exist per year/cluster
    topic_data = load_json(config.TOPIC_OUTPUT_FILE)
    if not topic_data:
        raise FileNotFoundError(
            f"Topic results not found at {config.TOPIC_OUTPUT_FILE}. "
            "Run topic_modelling.py first."
        )

    logger.info("Loading sentiment pipeline…")
    pipe = hf_pipeline(
        "sentiment-analysis",
        model=mdl,
        tokenizer=mdl,
        device=-1,   # CPU; set to 0 for first GPU
    )

    year_results: dict[str, dict] = {}

    for year_str, clusters in topic_data.get("years", {}).items():
        year = int(year_str)
        year_df = df[df["year"] == year]
        logger.info(f"\n── Year {year}: {len(year_df)} docs ──")

        cluster_results: dict[str, dict] = {}

        for cluster_str, cdata in clusters.items():
            cluster_id = int(cluster_str)
            topics_list = cdata.get("topics", [])

            if not topics_list:
                logger.info(f"  cluster {cluster_id}: no topics — skipping.")
                cluster_results[cluster_str] = {"cluster_id": cluster_id, "topics": []}
                continue

            logger.info(f"  cluster {cluster_id}: {len(topics_list)} topics")
            topic_sentiments = []

            for topic_entry in topics_list:
                tid = topic_entry["topic_id"]
                logger.info(f"    topic {tid}: scoring docs…")

                sent = score_topic(
                    topic_entry=topic_entry,
                    year_df=year_df,
                    cluster_id=cluster_id,
                    pipe=pipe,
                    batch_size=bs,
                    text_col=text_col,
                )

                logger.info(
                    f"      → dominant={sent['dominant_sentiment']}  "
                    f"coverage={sent['coverage']}  "
                    f"avg_conf={sent['avg_confidence']}"
                )
                topic_sentiments.append(sent)

            cluster_results[cluster_str] = {
                "cluster_id": cluster_id,
                "n_docs":     int(cdata.get("n_docs", 0)),
                "topics":     topic_sentiments,
            }

        year_results[year_str] = cluster_results

    output = {
        "generated_at": timestamp(),
        "model":        mdl,
        "text_column":  text_col,
        "years":        year_results,
    }

    save_json(output, config.SENTIMENT_OUTPUT_FILE)
    logger.info(f"\nSentiment results saved → {config.SENTIMENT_OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment analysis agent (per year/topic)")
    parser.add_argument("--model",      type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    main(model_name=args.model, batch_size=args.batch_size)
