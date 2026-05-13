# =========================
# results_formatter.py
# Merges topic, sentiment, and cluster metadata into the final JSON
# consumed by the dashboard and the explanation agent.
#
# Final structure:
#   {
#     "pipeline_version": "4.1-with-clustering",
#     "generated_at":     "...",
#     "summary": { year_range, total_years, total_docs,
#                  total_topics, total_clusters,
#                  avg_coherence, avg_confidence },
#     "models":  { topic_model_type, topic_parameters,
#                  sentiment_model, sentiment_confidence_threshold },
#     "visualizations": { ...paths relative to BASE_DIR... },
#     "years": [
#       {
#         "year", "total_docs", "n_clusters", "total_topics", "noise_ratio",
#         "avg_coherence",                  # average c_v across clusters
#         "sentiment_summary": { dominant, avg_confidence, ... },
#         "dominant_sentiment_dist": { POSITIVE: n, NEGATIVE: n, NEUTRAL: n },
#         "visualization": "<relative path or null>",
#         "clusters": [
#           {
#             "cluster_id", "n_docs", "n_topics", "coherence",
#             "sentiment_summary": {...},
#             "dominant_sentiment_dist": {...},
#             "topics": [
#               { topic_id, label, count, top_words, top_word_scores,
#                 sentiment: { dominant, avg_confidence,
#                              coverage, high_confidence_ratio,
#                              label_distribution, label_ratios } }
#             ]
#           }
#         ]
#       }
#     ]
#   }
# =========================

import os

import config
from utils import get_logger, save_json, load_json, timestamp, log_banner

logger = get_logger("results_formatter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentiment_dist(topics: list) -> dict[str, int]:
    """Count dominant-sentiment occurrences across topics."""
    dist: dict[str, int] = {}
    for t in topics:
        d = t.get("sentiment", {}).get("dominant")
        if d:
            dist[d] = dist.get(d, 0) + 1
    return dist


def _merge_cluster(t_cluster: dict, s_cluster: dict) -> dict:
    """Merge topic and sentiment records for one cluster."""
    t_topics    = t_cluster.get("topics", [])
    sent_by_tid = {t["topic_id"]: t for t in s_cluster.get("topics", [])}

    topics_out = []
    for t in t_topics:
        tid   = t["topic_id"]
        s_rec = sent_by_tid.get(tid, {})
        topics_out.append({
            "topic_id":        tid,
            "label":           t.get("label", f"topic_{tid}"),
            "count":           t.get("count", 0),
            "top_words":       t.get("top_words", []),
            "top_word_scores": t.get("top_word_scores", []),
            "sentiment": {
                "dominant":              s_rec.get("dominant_sentiment"),
                "avg_confidence":        s_rec.get("avg_confidence"),
                "std_confidence":        s_rec.get("std_confidence"),
                "high_confidence_ratio": s_rec.get("high_confidence_ratio"),
                "coverage":              s_rec.get("coverage"),
                "label_distribution":    s_rec.get("label_distribution", {}),
                "label_ratios":          s_rec.get("label_ratios", {}),
            },
        })

    merged = {
        "cluster_id":              t_cluster["cluster_id"],
        "n_docs":                  t_cluster.get("n_docs", s_cluster.get("n_docs", 0)),
        "n_topics":                t_cluster.get("n_topics", len(topics_out)),
        "coherence":               t_cluster.get("coherence"),
        "sentiment_summary":       s_cluster.get("summary", {}),
        "dominant_sentiment_dist": _sentiment_dist(topics_out),
        "topics":                  topics_out,
    }
    if t_cluster.get("error"):
        merged["error"] = t_cluster["error"]
    return merged


# ---------------------------------------------------------------------------
# Year-level merge
# ---------------------------------------------------------------------------

def merge_year_results(
    topic_data: dict,
    sentiment_data: dict,
    cluster_summary: dict,
) -> list:
    """Merge year → cluster → topic data into a list of year dicts."""
    topic_years     = topic_data.get("years", {})
    sentiment_years = sentiment_data.get("years", {})
    summary_years   = cluster_summary.get("years", {})

    all_year_keys = sorted(
        set(topic_years.keys())
        | set(sentiment_years.keys())
        | set(summary_years.keys()),
        key=int,
    )

    years_out = []

    for year_str in all_year_keys:
        year   = int(year_str)
        tdata  = topic_years.get(year_str, {})
        sdata  = sentiment_years.get(year_str, {})
        cdata  = summary_years.get(year_str, {})

        t_clusters = tdata.get("clusters", [])
        s_clusters = sdata.get("clusters", [])

        sent_cluster_by_id = {c["cluster_id"]: c for c in s_clusters}

        clusters_out = []
        for t_cluster in t_clusters:
            cid       = t_cluster["cluster_id"]
            s_cluster = sent_cluster_by_id.get(
                cid, {"cluster_id": cid, "n_docs": 0, "topics": [], "summary": {}}
            )
            clusters_out.append(_merge_cluster(t_cluster, s_cluster))

        all_topics    = [t for c in clusters_out for t in c["topics"]]
        total_topics  = sum(c["n_topics"] for c in clusters_out)
        n_clusters    = tdata.get(
            "n_clusters",
            len([c for c in clusters_out if c["cluster_id"] != -1]),
        )

        year_entry = {
            "year":                    year,
            "total_docs":              tdata.get("n_docs")
                                       or sdata.get("n_docs")
                                       or cdata.get("n_docs")
                                       or 0,
            "n_clusters":              n_clusters,
            "total_topics":            total_topics,
            "noise_ratio":             tdata.get("noise_ratio"),
            "avg_coherence":           tdata.get("avg_coherence"),
            "sentiment_summary":       sdata.get("summary", {}),
            "dominant_sentiment_dist": _sentiment_dist(all_topics),
            "visualization":           cdata.get("visualization"),
            "clusters":                clusters_out,
        }
        if tdata.get("error"):
            year_entry["error"] = tdata["error"]
        years_out.append(year_entry)

    return years_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    topic_data: dict = None,
    sentiment_data: dict = None,
    cluster_summary: dict = None,
) -> dict:
    log_banner(logger, "Results formatter (year → cluster → topic)")

    if topic_data is None:
        logger.info(f"Loading topic results from {config.TOPIC_OUTPUT_FILE}")
        topic_data = load_json(config.TOPIC_OUTPUT_FILE)

    if sentiment_data is None:
        logger.info(f"Loading sentiment results from {config.SENTIMENT_OUTPUT_FILE}")
        sentiment_data = load_json(config.SENTIMENT_OUTPUT_FILE)

    if cluster_summary is None:
        logger.info(f"Loading cluster summary from {config.CLUSTER_SUMMARY_FILE}")
        cluster_summary = load_json(config.CLUSTER_SUMMARY_FILE)

    years = merge_year_results(topic_data, sentiment_data, cluster_summary)

    total_docs     = sum(y["total_docs"]   for y in years)
    total_topics   = sum(y["total_topics"] for y in years)
    total_clusters = sum(y["n_clusters"]   for y in years)

    coherence_vals = [y["avg_coherence"] for y in years if y.get("avg_coherence")]
    avg_coherence = round(sum(coherence_vals) / len(coherence_vals), 4) if coherence_vals else 0.0

    confidence_vals = [
        y["sentiment_summary"].get("avg_confidence", 0.0)
        for y in years if y.get("sentiment_summary")
    ]
    avg_confidence = round(
        sum(confidence_vals) / len(confidence_vals), 4
    ) if confidence_vals else 0.0

    visualizations = {
        "cluster_summary": os.path.relpath(config.CLUSTER_SUMMARY_FILE, config.BASE_DIR),
    }
    if os.path.isdir(config.VIZ_DIR):
        for fname in sorted(os.listdir(config.VIZ_DIR)):
            full = os.path.join(config.VIZ_DIR, fname)
            visualizations[fname] = os.path.relpath(full, config.BASE_DIR)

    output = {
        "pipeline_version": "4.1-with-clustering",
        "generated_at":     timestamp(),
        "summary": {
            "year_range":     f"{config.YEAR_START}–{config.YEAR_END}",
            "total_years":    len(years),
            "total_docs":     total_docs,
            "total_topics":   total_topics,
            "total_clusters": total_clusters,
            "avg_coherence":  avg_coherence,
            "avg_confidence": avg_confidence,
        },
        "models": {
            "topic_model_type":              topic_data.get("model_type"),
            "topic_parameters":              topic_data.get("parameters"),
            "sentiment_model":               sentiment_data.get("model"),
            "sentiment_confidence_threshold": sentiment_data.get(
                "confidence_threshold", config.SENTIMENT_HIGH_CONFIDENCE
            ),
        },
        "visualizations": visualizations,
        "years": years,
    }

    save_json(output, config.FINAL_OUTPUT_FILE)
    logger.info(f"Final results written → {config.FINAL_OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    main()
