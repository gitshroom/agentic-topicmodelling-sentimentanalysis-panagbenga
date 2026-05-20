# =========================
# sentiment_analysis.py (LLM Version - String Cluster IDs)
# =========================

import argparse
import pandas as pd
from openai import OpenAI
import os
import time
from utils import get_logger, save_json, timestamp, log_banner
import config

logger = get_logger("sentiment_analysis")

client = OpenAI(
    api_key=config.LLM_API_KEY,
    base_url="http://localhost:11434/v1"
)

def score_texts_with_llm(texts: list) -> list:
    results = []
    for text in texts:
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis engine. Reply ONLY with 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'. Do not add any other text."},
                    {"role": "user", "content": f"Analyze this text: {text}"}
                ],
                temperature=0.0
            )
            
            label = response.choices[0].message.content.strip().upper()
            
            if label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                results.append({"label": label, "score": 1.0})
            else:
                results.append({"label": "UNKNOWN", "score": 0.0})

        except Exception as e:
            logger.error(f"LLM failed on text: {e}")
            results.append({"label": "ERROR", "score": 0.0})
            
        time.sleep(0.1) 
            
    return results

def aggregate_sentiment(records: list) -> dict:
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
    log_banner(logger, "LLM Sentiment analysis agent")
    text_col = "processed" if config.SENTIMENT_USE_PROCESSED else "text"

    df = pd.read_csv(config.CLUSTERED_FILE)
    df = df.dropna(subset=[text_col])
    logger.info(f"Loaded {len(df)} rows")

    cluster_results = {}

    for cluster_id in sorted(df["cluster"].unique()):
        # Skip noise clusters for sentiment to save time
        if str(cluster_id).endswith("_-1"):
            continue

        subset = df[df["cluster"] == cluster_id]
        texts = subset[text_col].tolist()

        logger.info(f"Cluster {cluster_id}: scoring {len(texts)} docs using LLM...")

        scored = score_texts_with_llm(texts)

        per_doc = []
        for (_, row), s in zip(subset.iterrows(), scored):
            per_doc.append({
                "id": row.get("id", None),
                "label": s["label"],
                "score": s["score"],
            })

        agg = aggregate_sentiment(scored)

        cluster_results[str(cluster_id)] = {
            "cluster_id": str(cluster_id), # FIXED: Kept as string
            **agg,
            "per_doc": per_doc,
        }

    output = {
        "generated_at": timestamp(),
        "model": "LLM_Prompting",
        "text_column": text_col,
        "clusters": cluster_results,
    }

    save_json(output, config.SENTIMENT_OUTPUT_FILE)
    logger.info(f"Sentiment results saved to {config.SENTIMENT_OUTPUT_FILE}")
    return output

if __name__ == "__main__":
    main()