# =========================
# results_formatter.py
# Merges topic and sentiment results into the final JSON.
#
# Input structures (both year → clusters → topics):
#
#   topic_data["years"][year_str] = {
#       n_docs, n_clusters, noise_ratio,
#       clusters: [
#           { cluster_id, n_docs, n_topics, coherence,
#             topics: [{ topic_id, label, count, top_words, top_word_scores }] }
#       ]
#   }
#
#   sentiment_data["years"][year_str] = {
#       n_docs,
#       clusters: [
#           { cluster_id, n_docs,
#             topics: [{ topic_id, n_docs, coverage, avg_confidence,
#                        dominant_sentiment, label_distribution, label_ratios }] }
#       ]
#   }
#
# Final output structure:
#   years: [
#     {
#       year, total_docs, n_clusters, noise_ratio,
#       dominant_sentiment_dist,   ← aggregated across all clusters' topics
#       clusters: [
#         {
#           cluster_id, n_docs, n_topics, coherence,
#           dominant_sentiment_dist,   ← aggregated for this cluster
#           topics: [
#             {
#               topic_id, label, count, top_words, top_word_scores,
#               sentiment: {
#                 dominant, avg_confidence, coverage,
#                 label_distribution, label_ratios
#               }
#             }
#           ]
#         }
#       ]
#     }
#   ]
# =========================

from utils import get_logger, save_json, load_json, timestamp, log_banner
import config

logger = get_logger("results_formatter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentiment_dist(topics: list) -> dict[str, int]:
    """Count dominant-sentiment occurrences across a list of merged topics."""
    dist: dict[str, int] = {}
    for t in topics:
        d = t["sentiment"].get("dominant")
        if d:
            dist[d] = dist.get(d, 0) + 1
    return dist


def _merge_cluster(t_cluster: dict, s_cluster: dict) -> dict:
    """
    Merge a single cluster's topic list with its sentiment records.
    Returns the merged cluster dict ready for the final output.
    """
    t_topics = t_cluster.get("topics", [])
    # Index sentiment records by topic_id for O(1) lookup
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
                "dominant":           s_rec.get("dominant_sentiment"),
                "avg_confidence":     s_rec.get("avg_confidence"),
                "coverage":           s_rec.get("coverage"),
                "label_distribution": s_rec.get("label_distribution", {}),
                "label_ratios":       s_rec.get("label_ratios", {}),
            },
        })

    return {
        "cluster_id":              t_cluster["cluster_id"],
        "n_docs":                  t_cluster.get("n_docs", s_cluster.get("n_docs", 0)),
        "n_topics":                t_cluster.get("n_topics", len(topics_out)),
        "coherence":               t_cluster.get("coherence"),
        "dominant_sentiment_dist": _sentiment_dist(topics_out),
        "topics":                  topics_out,
        # Carry through any error message from topic modelling
        **({"error": t_cluster["error"]} if t_cluster.get("error") else {}),
    }


# ---------------------------------------------------------------------------
# Year-level merge
# ---------------------------------------------------------------------------

def merge_year_results(topic_data: dict, sentiment_data: dict) -> list:
    """
    Merge year → cluster → topic data from both sources into a list of
    year objects for the final JSON.
    """
    topic_years     = topic_data.get("years", {})
    sentiment_years = sentiment_data.get("years", {})

    all_year_keys = sorted(
        set(topic_years.keys()) | set(sentiment_years.keys()),
        key=int,
    )

    years_out = []

    for year_str in all_year_keys:
        year   = int(year_str)
        tdata  = topic_years.get(year_str, {})
        sdata  = sentiment_years.get(year_str, {})

        t_clusters = tdata.get("clusters", [])
        s_clusters = sdata.get("clusters", [])

        # Index sentiment clusters by cluster_id
        sent_cluster_by_id = {c["cluster_id"]: c for c in s_clusters}

        clusters_out = []
        for t_cluster in t_clusters:
            cid       = t_cluster["cluster_id"]
            s_cluster = sent_cluster_by_id.get(cid, {"cluster_id": cid, "n_docs": 0, "topics": []})
            clusters_out.append(_merge_cluster(t_cluster, s_cluster))

        # Year-level topic count and dominant-sentiment aggregation
        all_topics = [t for c in clusters_out for t in c["topics"]]
        total_topics = sum(c["n_topics"] for c in clusters_out)

        years_out.append({
            "year":                    year,
            "total_docs":              tdata.get("n_docs") or sdata.get("n_docs") or 0,
            "n_clusters":              tdata.get("n_clusters", len([c for c in clusters_out if c["cluster_id"] != -1])),
            "total_topics":            total_topics,
            "noise_ratio":             tdata.get("noise_ratio"),
            "dominant_sentiment_dist": _sentiment_dist(all_topics),
            "clusters":                clusters_out,
            **({"error": tdata["error"]} if tdata.get("error") else {}),
        })

    return years_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    topic_data: dict = None,
    sentiment_data: dict = None,
) -> dict:
    log_banner(logger, "Results formatter (year → cluster → topic)")

    if topic_data is None:
        logger.info(f"Loading topic results from {config.TOPIC_OUTPUT_FILE}")
        topic_data = load_json(config.TOPIC_OUTPUT_FILE)

    if sentiment_data is None:
        logger.info(f"Loading sentiment results from {config.SENTIMENT_OUTPUT_FILE}")
        sentiment_data = load_json(config.SENTIMENT_OUTPUT_FILE)

    years = merge_year_results(topic_data, sentiment_data)

    total_docs    = sum(y["total_docs"]   for y in years)
    total_topics  = sum(y["total_topics"] for y in years)
    total_clusters = sum(y["n_clusters"]  for y in years)

    output = {
        "pipeline_version": "4.0",
        "generated_at":     timestamp(),
        "summary": {
            "year_range":    f"{config.YEAR_START}–{config.YEAR_END}",
            "total_years":   len(years),
            "total_docs":    total_docs,
            "total_topics":  total_topics,
            "total_clusters": total_clusters,
        },
        "models": {
            "topic_model_type": topic_data.get("model_type"),
            "topic_parameters": topic_data.get("parameters"),
            "sentiment_model":  sentiment_data.get("model"),
        },
        "years": years,
    }

    save_json(output, config.FINAL_OUTPUT_FILE)
    logger.info(f"Final results written → {config.FINAL_OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    main()
