# =========================
# sentiment_analysis.py
# Step 4 of Pipeline 2 (with_clustering).
#
# Sentiment analysis per TOPIC per CLUSTER per YEAR using
# `cardiffnlp/twitter-xlm-roberta-base-sentiment`. Confidence (model
# softmax score) is documented at three levels:
#   - per topic: `avg_confidence`, share of `high_confidence` predictions
#   - per cluster: `avg_confidence`, `dominant_sentiment`
#   - per year:   `avg_confidence`, `dominant_sentiment`
#
# Doc-to-topic assignment uses persisted `pre_cluster_label` from the
# embeddings step (hard cluster membership), then narrows to docs whose
# vocabulary overlaps the topic's top words. If no overlap is found the
# whole cluster is scored to preserve coverage.
#
# Output: outputs/sentiment_results.json
# =========================

import argparse
import os

import numpy as np
import pandas as pd

import config
from utils import get_logger, save_json, load_json, timestamp, log_banner, ensure_dir

logger = get_logger("sentiment_analysis")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_texts(texts: list, pipeline, batch_size: int) -> list:
    """Run HuggingFace pipeline in batches. Returns list of {label, score}.

    Labels are normalised to uppercase (POSITIVE / NEUTRAL / NEGATIVE)
    so downstream JSON and dashboard CSS are consistent regardless of
    the underlying model casing.
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            out = pipeline(batch, truncation=True, max_length=512)
            for r in out:
                lbl = r.get("label", "UNKNOWN")
                results.append({
                    "label": str(lbl).upper(),
                    "score": float(r.get("score", 0.0)),
                })
        except Exception as e:
            logger.error(f"Batch {i}–{i + batch_size} failed: {e}")
            results.extend([{"label": "ERROR", "score": 0.0}] * len(batch))
    return results


def aggregate_sentiment(records: list, high_thresh: float = None) -> dict:
    """Compute label distribution and confidence stats from scored records."""
    high_thresh = high_thresh if high_thresh is not None else config.SENTIMENT_HIGH_CONFIDENCE

    label_counts: dict[str, int] = {}
    confidences: list[float] = []
    high_conf_count = 0

    for r in records:
        lbl = r.get("label", "UNKNOWN")
        sc  = float(r.get("score", 0.0))
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        if lbl not in ("ERROR", "UNKNOWN"):
            confidences.append(sc)
            if sc >= high_thresh:
                high_conf_count += 1

    n        = len(records)
    valid    = len(confidences)
    avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
    std_conf = round(float(np.std(confidences)),  4) if confidences else 0.0
    coverage = round(valid / n, 4) if n else 0.0
    dominant = max(label_counts, key=label_counts.get) if label_counts else "UNKNOWN"
    high_conf_ratio = round(high_conf_count / valid, 4) if valid else 0.0

    return {
        "n_docs":               n,
        "coverage":             coverage,
        "avg_confidence":       avg_conf,
        "std_confidence":       std_conf,
        "high_confidence_ratio": high_conf_ratio,
        "dominant_sentiment":   dominant,
        "label_distribution":   label_counts,
        "label_ratios":         {k: round(v / n, 4) for k, v in label_counts.items()},
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
    """Score the docs that best match a topic inside its cluster."""
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
        topic_docs = cluster_df  # fall back: score the entire cluster

    texts  = topic_docs[text_col].tolist()
    scored = score_texts(texts, pipe, batch_size)

    agg = aggregate_sentiment(scored)
    agg["topic_id"] = topic_id
    return agg


def aggregate_topics(topic_sentiments: list) -> dict:
    """Aggregate per-topic results into cluster-level totals."""
    total_docs = 0
    weighted_conf = 0.0
    label_dist: dict[str, int] = {}
    high_conf_weighted = 0.0
    coverage_weighted  = 0.0

    for s in topic_sentiments:
        n = s.get("n_docs", 0)
        total_docs += n
        weighted_conf      += s.get("avg_confidence", 0.0)        * n
        coverage_weighted  += s.get("coverage", 0.0)              * n
        high_conf_weighted += s.get("high_confidence_ratio", 0.0) * n
        for lbl, cnt in s.get("label_distribution", {}).items():
            label_dist[lbl] = label_dist.get(lbl, 0) + cnt

    if total_docs:
        avg_conf      = round(weighted_conf      / total_docs, 4)
        avg_coverage  = round(coverage_weighted  / total_docs, 4)
        avg_high_conf = round(high_conf_weighted / total_docs, 4)
    else:
        avg_conf = avg_coverage = avg_high_conf = 0.0

    real_dist = {
        k.upper(): v for k, v in label_dist.items()
        if k.upper() not in ("ERROR", "UNKNOWN")
    }
    dominant = max(real_dist, key=real_dist.get) if real_dist else "UNKNOWN"

    return {
        "n_docs":                total_docs,
        "avg_confidence":        avg_conf,
        "avg_coverage":          avg_coverage,
        "high_confidence_ratio": avg_high_conf,
        "dominant_sentiment":    dominant,
        "label_distribution":    label_dist,
    }


# ---------------------------------------------------------------------------
# Sentiment visualization
# ---------------------------------------------------------------------------

def render_sentiment_chart(year_results: dict, out_path: str) -> None:
    """Save a per-year sentiment ratio + confidence chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for year_str, ydata in year_results.items():
        summary = ydata.get("summary", {})
        rows.append({
            "year":  int(year_str),
            "POSITIVE": summary.get("label_distribution", {}).get("POSITIVE", 0),
            "NEGATIVE": summary.get("label_distribution", {}).get("NEGATIVE", 0),
            "NEUTRAL":  summary.get("label_distribution", {}).get("NEUTRAL",  0),
            "avg_conf": summary.get("avg_confidence", 0.0),
        })
    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("year")
    totals = df[["POSITIVE", "NEGATIVE", "NEUTRAL"]].sum(axis=1).replace(0, 1)
    pos_pct = df["POSITIVE"] / totals * 100
    neg_pct = df["NEGATIVE"] / totals * 100
    neu_pct = df["NEUTRAL"]  / totals * 100

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    width = 0.6

    ax1.bar(x, pos_pct, width, label="POSITIVE", color="#4dffa0")
    ax1.bar(x, neu_pct, width, bottom=pos_pct,           label="NEUTRAL",  color="#ffd166")
    ax1.bar(x, neg_pct, width, bottom=pos_pct + neu_pct, label="NEGATIVE", color="#ff6b6b")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["year"].astype(str))
    ax1.set_ylabel("Sentiment share (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Sentiment distribution per year (with model confidence)")
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, df["avg_conf"], color="#222222", marker="o",
             linewidth=2, label="Avg confidence")
    ax2.set_ylabel("Avg confidence")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.VIZ_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_name: str = None, batch_size: int = None,
         use_processed: bool = None) -> dict:
    from transformers import pipeline as hf_pipeline

    log_banner(logger, "Sentiment analysis — year → cluster → topic")

    mdl  = model_name    if model_name    is not None else config.SENTIMENT_MODEL
    bs   = batch_size    if batch_size    is not None else config.SENTIMENT_BATCH_SIZE
    up   = use_processed if use_processed is not None else config.SENTIMENT_USE_PROCESSED
    text_col = "processed" if up else "text"

    logger.info(f"Model: {mdl}  |  Batch size: {bs}  |  Column: {text_col}")

    df = pd.read_pickle(config.CLUSTERED_FILE)
    df = df.dropna(subset=[text_col]).copy()
    if "year" not in df.columns:
        raise ValueError("'year' column missing — run preprocessing.py first.")
    df["year"] = df["year"].astype(int)

    has_cluster_col = "pre_cluster_label" in df.columns
    if has_cluster_col:
        logger.info("Using persisted `pre_cluster_label` for cluster membership.")
    else:
        logger.warning(
            "`pre_cluster_label` missing — sentiment will scan the full "
            "year for each cluster (topic-word overlap narrows results)."
        )

    logger.info(f"Loaded {len(df)} rows")

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
        device=-1,  # CPU; set to 0 for first GPU
    )

    year_results: dict[str, dict] = {}

    for year_str, ydata in topic_data.get("years", {}).items():
        year        = int(year_str)
        year_df     = df[df["year"] == year]
        clusters_in = ydata.get("clusters", [])

        logger.info(
            f"\n══ Year {year}: {len(year_df)} docs, {len(clusters_in)} clusters ══"
        )

        if not clusters_in:
            year_results[year_str] = {
                "n_docs":   len(year_df),
                "clusters": [],
                "summary":  aggregate_topics([]),
            }
            continue

        clusters_out = []

        for cluster_entry in clusters_in:
            cid         = cluster_entry["cluster_id"]
            topics_list = cluster_entry.get("topics", [])

            if has_cluster_col:
                cluster_df = year_df[year_df["pre_cluster_label"] == cid]
            else:
                cluster_df = year_df

            logger.info(
                f"  ── Cluster {cid}: {len(cluster_df)} docs, "
                f"{len(topics_list)} topics ──"
            )

            if not topics_list:
                clusters_out.append({
                    "cluster_id": cid,
                    "n_docs":     int(cluster_entry.get("n_docs", len(cluster_df))),
                    "topics":     [],
                    "summary":    aggregate_topics([]),
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
                    f"avg_conf={sent['avg_confidence']}  "
                    f"high_conf={sent['high_confidence_ratio']}"
                )
                topic_sentiments.append(sent)

            cluster_summary = aggregate_topics(topic_sentiments)
            clusters_out.append({
                "cluster_id": cid,
                "n_docs":     int(cluster_entry.get("n_docs", len(cluster_df))),
                "topics":     topic_sentiments,
                "summary":    cluster_summary,
            })

        year_summary = aggregate_topics(
            [t for c in clusters_out for t in c.get("topics", [])]
        )

        year_results[year_str] = {
            "n_docs":   int(ydata.get("n_docs", len(year_df))),
            "clusters": clusters_out,
            "summary":  year_summary,
        }
        logger.info(
            f"  Year {year} summary: dominant={year_summary['dominant_sentiment']}  "
            f"avg_conf={year_summary['avg_confidence']}  "
            f"high_conf={year_summary['high_confidence_ratio']}"
        )

    output = {
        "generated_at": timestamp(),
        "model":        mdl,
        "text_column":  text_col,
        "confidence_threshold": config.SENTIMENT_HIGH_CONFIDENCE,
        "years":        year_results,
    }

    save_json(output, config.SENTIMENT_OUTPUT_FILE)
    logger.info(f"\nSentiment results saved → {config.SENTIMENT_OUTPUT_FILE}")

    if config.GENERATE_VISUALIZATIONS:
        ensure_dir(config.VIZ_DIR)
        chart_path = os.path.join(config.VIZ_DIR, "sentiment_overview.png")
        try:
            render_sentiment_chart(year_results, chart_path)
            logger.info(f"Sentiment chart saved → {chart_path}")
        except Exception as e:
            logger.warning(f"Sentiment chart failed: {e}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentiment analysis (per year / cluster / topic)"
    )
    parser.add_argument("--model",         type=str,  default=None)
    parser.add_argument("--batch_size",    type=int,  default=None)
    parser.add_argument("--use_processed", type=bool, default=None)
    args = parser.parse_args()
    main(
        model_name=args.model,
        batch_size=args.batch_size,
        use_processed=args.use_processed,
    )
