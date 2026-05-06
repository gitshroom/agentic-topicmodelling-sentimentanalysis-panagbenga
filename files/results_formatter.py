# =========================
# results_formatter.py
# Agent: merges topic and sentiment results into a clean final JSON.
# Output structure:
#   years[year][cluster_id] → {
#       cluster_id, n_docs,
#       topics[ { topic_id, top_words, count, sentiment: { … } } ]
#   }
# =========================

from utils import get_logger, save_json, load_json, timestamp, log_banner
import config

logger = get_logger("results_formatter")


def merge_year_results(topic_data: dict, sentiment_data: dict) -> list:
    """
    Merge topic and sentiment dicts (both keyed by year → cluster → …)
    into a list of year objects for the final JSON.
    """
    topic_years     = topic_data.get("years", {})
    sentiment_years = sentiment_data.get("years", {})

    all_year_keys = sorted(
        set(topic_years.keys()) | set(sentiment_years.keys()),
        key=int,
    )

    years_out = []

    for year_str in all_year_keys:
        year          = int(year_str)
        topic_clusters = topic_years.get(year_str, {})
        sent_clusters  = sentiment_years.get(year_str, {})

        all_cluster_keys = sorted(
            set(topic_clusters.keys()) | set(sent_clusters.keys()),
            key=int,
        )

        clusters_out = []

        for cid_str in all_cluster_keys:
            cid    = int(cid_str)
            tdata  = topic_clusters.get(cid_str, {})
            sdata  = sent_clusters.get(cid_str, {})

            # Build per-topic list, attaching sentiment to each topic
            topic_list   = tdata.get("topics", [])
            sent_topics  = {t["topic_id"]: t for t in sdata.get("topics", [])}

            topics_out = []
            for t in topic_list:
                tid   = t["topic_id"]
                s_rec = sent_topics.get(tid, {})

                topics_out.append({
                    "topic_id":        tid,
                    "label":           t.get("label", f"topic_{tid}"),
                    "count":           t.get("count", 0),
                    "top_words":       t.get("top_words", []),
                    "top_word_scores": t.get("top_word_scores", []),
                    "sentiment": {
                        "dominant":          s_rec.get("dominant_sentiment"),
                        "avg_confidence":    s_rec.get("avg_confidence"),
                        "coverage":          s_rec.get("coverage"),
                        "label_distribution":s_rec.get("label_distribution", {}),
                        "label_ratios":      s_rec.get("label_ratios", {}),
                    },
                })

            clusters_out.append({
                "cluster_id": cid,
                "n_docs":     tdata.get("n_docs") or sdata.get("n_docs"),
                "n_topics":   tdata.get("n_topics", len(topics_out)),
                "noise_ratio":     tdata.get("noise_ratio"),
                "coherence_proxy": tdata.get("coherence_proxy"),
                "topics":     topics_out,
                "error":      tdata.get("error"),
            })

        # Year-level sentiment summary
        all_dominant = []
        for c in clusters_out:
            for t in c["topics"]:
                d = t["sentiment"].get("dominant")
                if d:
                    all_dominant.append(d)

        sent_summary: dict[str, int] = {}
        for d in all_dominant:
            sent_summary[d] = sent_summary.get(d, 0) + 1

        years_out.append({
            "year":                    year,
            "total_docs":              sum(c["n_docs"] or 0 for c in clusters_out),
            "total_clusters":          len(clusters_out),
            "total_topics":            sum(c["n_topics"] for c in clusters_out),
            "dominant_sentiment_dist": sent_summary,
            "clusters":                clusters_out,
        })

    return years_out


def main(
    topic_data: dict = None,
    sentiment_data: dict = None,
    topic_issues: list = None,
    sentiment_issues: list = None,
    passed: bool = True,
) -> dict:
    log_banner(logger, "Results formatter")

    if topic_data is None:
        logger.info(f"Loading topic results from {config.TOPIC_OUTPUT_FILE}")
        topic_data = load_json(config.TOPIC_OUTPUT_FILE)

    if sentiment_data is None:
        logger.info(f"Loading sentiment results from {config.SENTIMENT_OUTPUT_FILE}")
        sentiment_data = load_json(config.SENTIMENT_OUTPUT_FILE)

    topic_issues    = topic_issues    or []
    sentiment_issues= sentiment_issues or []

    years = merge_year_results(topic_data, sentiment_data)

    # Global summary
    total_docs    = sum(y["total_docs"]    for y in years)
    total_topics  = sum(y["total_topics"]  for y in years)
    total_clusters= sum(y["total_clusters"]for y in years)

    output = {
        "pipeline_version": "2.0",
        "generated_at":     timestamp(),
        "quality_passed":   passed,
        "validation": {
            "topic_issues":     topic_issues,
            "sentiment_issues": sentiment_issues,
        },
        "summary": {
            "year_range":       f"{config.YEAR_START}–{config.YEAR_END}",
            "total_years":      len(years),
            "total_docs":       total_docs,
            "total_clusters":   total_clusters,
            "total_topics":     total_topics,
        },
        "models": {
            "topic_model_type":  topic_data.get("model_type"),
            "topic_parameters":  topic_data.get("parameters"),
            "sentiment_model":   sentiment_data.get("model"),
        },
        "years": years,
    }

    save_json(output, config.FINAL_OUTPUT_FILE)
    logger.info(f"Final results written → {config.FINAL_OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    main()
