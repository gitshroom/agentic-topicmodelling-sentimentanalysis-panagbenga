# =========================
# sentiment_analysis.py
# Agent: runs per-cluster sentiment analysis using a HuggingFace pipeline.
# Called by orchestrator.py with optional parameter overrides.
# =========================

import argparse
import pandas as pd
from utils import get_logger, save_json, timestamp, log_banner
import config

logger = get_logger("sentiment_analysis")


def score_texts(texts: list, pipeline, batch_size: int) -> list:
    """Run the HuggingFace pipeline in batches. Returns list of {label, score}."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            out = pipeline(batch, truncation=True, max_length=512)
            results.extend(out)
        except Exception as e:
            logger.error(f"Batch {i}–{i+batch_size} failed: {e}")
            results.extend([{"label": "ERROR", "score": 0.0}] * len(batch))
    return results


def aggregate_sentiment(records: list) -> dict:
    """Compute label distribution and mean confidence from a list of {label, score}."""
    label_counts = {}
    total_score = 0.0
    valid = 0

    for r in records:
        lbl = r.get("label", "UNKNOWN")
        sc = r.get("score", 0.0)
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        if lbl not in ("ERROR", "UNKNOWN"):
            total_score += sc
            valid += 1

    n = len(records)
    avg_confidence = round(total_score / valid, 4) if valid else 0.0
    coverage = round(valid / n, 4) if n else 0.0
    dominant = max(label_counts, key=label_counts.get) if label_counts else "UNKNOWN"

    return {
        "n_docs": n,
        "coverage": coverage,
        "avg_confidence": avg_confidence,
        "dominant_sentiment": dominant,
        "label_distribution": label_counts,
        "label_ratios": {k: round(v / n, 4) for k, v in label_counts.items()},
    }


def main(model_name: str = None, batch_size: int = None):
    from transformers import pipeline as hf_pipeline

    log_banner(logger, "Sentiment analysis agent")

    mdl = model_name or config.SENTIMENT_MODEL
    bs = batch_size or config.SENTIMENT_BATCH_SIZE
    text_col = "processed" if config.SENTIMENT_USE_PROCESSED else "text"

    logger.info(f"Model: {mdl}  |  Batch size: {bs}  |  Column: {text_col}")

    df = pd.read_csv(config.CLUSTERED_FILE)
    df = df.dropna(subset=[text_col])
    logger.info(f"Loaded {len(df)} rows")

    logger.info("Loading sentiment pipeline...")
    pipe = hf_pipeline(
        "sentiment-analysis",
        model=mdl,
        tokenizer=mdl,
        device=-1,   # CPU; set to 0 for first GPU
    )

    cluster_results = {}

    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        texts = subset[text_col].tolist()

        logger.info(f"Cluster {cluster_id}: scoring {len(texts)} docs...")

        scored = score_texts(texts, pipe, batch_size=bs)

        # Attach scores back to rows for per-doc output
        per_doc = []
        for (_, row), s in zip(subset.iterrows(), scored):
            per_doc.append({
                "id": row.get("id", None),
                "label": s["label"],
                "score": round(float(s["score"]), 4),
            })

        agg = aggregate_sentiment(scored)

        cluster_results[str(cluster_id)] = {
            "cluster_id": int(cluster_id),
            **agg,
            "per_doc": per_doc,
        }

        logger.info(
            f"  → dominant={agg['dominant_sentiment']}  "
            f"coverage={agg['coverage']}  "
            f"avg_conf={agg['avg_confidence']}"
        )

    output = {
        "generated_at": timestamp(),
        "model": mdl,
        "text_column": text_col,
        "clusters": cluster_results,
    }

    save_json(output, config.SENTIMENT_OUTPUT_FILE)
    logger.info(f"Sentiment results saved to {config.SENTIMENT_OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment analysis agent")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    main(model_name=args.model, batch_size=args.batch_size)
