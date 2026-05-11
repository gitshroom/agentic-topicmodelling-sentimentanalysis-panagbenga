# =========================
# sentiment_analysis.py
# Agent: sentiment analysis per TOPIC per CLUSTER per YEAR.
#
# Reads topic_modelling output (year → clusters → topics) and scores
# every topic's docs using a HuggingFace sentiment pipeline.
#
# Doc assignment: docs in a cluster that overlap with a topic's top words
# are selected for that topic. Falls back to all cluster docs when no
# overlap is found (preserves full coverage).
#
# Output structure mirrors topic_modelling — adds a sentiment record to
# each topic:
#   {
#     "generated_at": "...",
#     "model":        str,
#     "text_column":  str,
#     "years": {
#       "<year>": {
#         "n_docs": int,
#         "clusters": [
#           {
#             "cluster_id": int,
#             "n_docs":     int,
#             "topics": [
#               {
#                 "topic_id":           int,
#                 "n_docs":             int,
#                 "coverage":           float,
#                 "avg_confidence":     float,
#                 "dominant_sentiment": str,
#                 "label_distribution": { POSITIVE: int, ... },
#                 "label_ratios":       { POSITIVE: float, ... }
#               }, ...
#             ]
#           }, ...
#         ]
#       }
#     }
#   }
#
# Standalone: python sentiment_analysis.py
# Called by orchestrator with optional overrides.
# =========================

import argparse
import numpy as np
import pandas as pd

from utils import get_logger, save_json, load_json, timestamp, log_banner
import config

logger = get_logger("sentiment_analysis")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_texts(texts: list, pipeline, batch_size: int) -> list:
    """Run HuggingFace pipeline in batches. Returns list of {label, score}."""
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
    """Compute label distribution and mean confidence from scored records."""
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
# Topic-level doc assignment + scoring
# ---------------------------------------------------------------------------

def score_topic(
    topic_entry: dict,
    cluster_df: pd.DataFrame,
    pipe,
    batch_size: int,
    text_col: str,
) -> dict:
    """
    Select docs in cluster_df that best match this topic via top-word
    overlap, then score them with the sentiment pipeline.
    Falls back to all cluster docs when no overlap is found.
    """
    topic_id  = topic_entry["topic_id"]
    top_words = set(topic_entry.get("top_words", []))

    if cluster_df.empty or not top_words:
        result = aggregate_sentiment([])
        result["topic_id"] = topic_id
        return result

    cluster_df = cluster_df.copy()
    cluster_df["_overlap"] = cluster_df["processed"].apply(
        lambda t: sum(1 for w in top_words if w in str(t).split())
    )
    topic_docs = cluster_df[cluster_df["_overlap"] > 0]
    if topic_docs.empty:
        topic_docs = cluster_df   # fall back: score the whole cluster

    texts  = topic_docs[text_col].tolist()
    scored = score_texts(texts, pipe, batch_size)

    agg = aggregate_sentiment(scored)
    agg["topic_id"] = topic_id
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_name: str = None, batch_size: int = None,
         use_processed: bool = None) -> dict:
    from transformers import pipeline as hf_pipeline

    log_banner(logger, "Sentiment analysis — per year / cluster / topic")

    mdl  = model_name    if model_name    is not None else config.SENTIMENT_MODEL
    bs   = batch_size    if batch_size    is not None else config.SENTIMENT_BATCH_SIZE
    up   = use_processed if use_processed is not None else config.SENTIMENT_USE_PROCESSED
    text_col = "processed" if up else "text"

    logger.info(f"Model: {mdl}  |  Batch size: {bs}  |  Column: {text_col}")

    # Load preprocessed data
    df = pd.read_pickle(config.CLUSTERED_FILE)
    df = df.dropna(subset=[text_col])

    if "year" not in df.columns:
        raise ValueError("'year' column missing — run preprocessing.py first.")

    df["year"] = df["year"].astype(int)
    logger.info(f"Loaded {len(df)} rows")

    # Load topic results — structure: years → year_str → { n_docs, clusters: [...] }
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

    for year_str, ydata in topic_data.get("years", {}).items():
        year    = int(year_str)
        year_df = df[df["year"] == year]
        clusters_in = ydata.get("clusters", [])

        logger.info(f"\n══ Year {year}: {len(year_df)} docs, {len(clusters_in)} clusters ══")

        if not clusters_in:
            year_results[year_str] = {
                "n_docs":   len(year_df),
                "clusters": [],
            }
            continue

        clusters_out = []

        for cluster_entry in clusters_in:
            cid         = cluster_entry["cluster_id"]
            topics_list = cluster_entry.get("topics", [])

            # ── Select the cluster's docs from the dataframe ──
            # The pre_cluster labels aren't saved to df by default, so we
            # use top-word matching against the full year_df and rely on the
            # topic-level scorer to narrow it down.  For a hard assignment
            # we'd need to store pre_cluster_label in the pickle — if that
            # column exists, use it; otherwise fall back to the full year_df.
            if "pre_cluster_label" in year_df.columns:
                cluster_df = year_df[year_df["pre_cluster_label"] == cid]
            else:
                # Approximate: all year docs (topic scorer filters by overlap)
                cluster_df = year_df

            logger.info(
                f"  ── Cluster {cid}: {cluster_entry['n_docs']} docs, "
                f"{len(topics_list)} topics ──"
            )

            if not topics_list:
                clusters_out.append({
                    "cluster_id": cid,
                    "n_docs":     cluster_entry["n_docs"],
                    "topics":     [],
                })
                continue

            topic_sentiments = []
            for topic_entry in topics_list:
                tid = topic_entry["topic_id"]
                logger.info(f"    topic {tid}: scoring docs…")

                sent = score_topic(
                    topic_entry=topic_entry,
                    cluster_df=cluster_df,
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

            clusters_out.append({
                "cluster_id": cid,
                "n_docs":     cluster_entry["n_docs"],
                "topics":     topic_sentiments,
            })

        year_results[year_str] = {
            "n_docs":   int(ydata.get("n_docs", len(year_df))),
            "clusters": clusters_out,
        }

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
    parser = argparse.ArgumentParser(description="Sentiment analysis (per year / cluster / topic)")
    parser.add_argument("--model",         type=str,  default=None)
    parser.add_argument("--batch_size",    type=int,  default=None)
    parser.add_argument("--use_processed", type=bool, default=None)
    args = parser.parse_args()
    main(model_name=args.model, batch_size=args.batch_size, use_processed=args.use_processed)
