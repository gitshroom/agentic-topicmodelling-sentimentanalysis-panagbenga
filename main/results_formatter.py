# =========================
# results_formatter.py
# Merges topic and sentiment results into the final JSON.
#
# Input structures (both flat year → topics, no cluster layer):
#
#   topic_data["years"][year_str] = {
#       n_docs, n_topics, noise_ratio, coherence,
#       topics: [{ topic_id, label, count, top_words, top_word_scores }]
#   }
#
#   sentiment_data["years"][year_str] = {
#       n_docs,
#       topics: [{ topic_id, n_docs, coverage, avg_confidence,
#                  dominant_sentiment, label_distribution, label_ratios }]
#   }
#
# Final output structure:
#   years: [
#     {
#       year, total_docs, total_topics, noise_ratio, coherence,
#       dominant_sentiment_dist,
#       topics: [
#         {
#           topic_id, label, count, top_words, top_word_scores,
#           sentiment: {
#             dominant, avg_confidence, coverage,
#             label_distribution, label_ratios
#           }
#         }
#       ]
#     }
#   ]
# =========================

from utils import get_logger, save_json, load_json, timestamp, log_banner
import config

logger = get_logger("results_formatter")


def merge_year_results(topic_data: dict, sentiment_data: dict) -> list:
    """
    Merge flat year→topics topic and sentiment dicts
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
        year   = int(year_str)
        tdata  = topic_years.get(year_str, {})
        sdata  = sentiment_years.get(year_str, {})

        topic_list  = tdata.get("topics", [])
        # Index sentiment records by topic_id for O(1) lookup
        sent_by_tid = {t["topic_id"]: t for t in sdata.get("topics", [])}

        topics_out = []
        for t in topic_list:
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

        # Year-level dominant sentiment distribution (count per label)
        sent_summary: dict[str, int] = {}
        for t in topics_out:
            d = t["sentiment"].get("dominant")
            if d:
                sent_summary[d] = sent_summary.get(d, 0) + 1

        years_out.append({
            "year":                    year,
            "total_docs":              tdata.get("n_docs") or sdata.get("n_docs") or 0,
            "total_topics":            tdata.get("n_topics", len(topics_out)),
            "noise_ratio":             tdata.get("noise_ratio"),
            "coherence":               tdata.get("coherence"),
            "dominant_sentiment_dist": sent_summary,
            "topics":                  topics_out,
            "error":                   tdata.get("error"),
        })

    return years_out


def main(
    topic_data: dict = None,
    sentiment_data: dict = None,
) -> dict:
    log_banner(logger, "Results formatter")

    if topic_data is None:
        logger.info(f"Loading topic results from {config.TOPIC_OUTPUT_FILE}")
        topic_data = load_json(config.TOPIC_OUTPUT_FILE)

    if sentiment_data is None:
        logger.info(f"Loading sentiment results from {config.SENTIMENT_OUTPUT_FILE}")
        sentiment_data = load_json(config.SENTIMENT_OUTPUT_FILE)

    years = merge_year_results(topic_data, sentiment_data)

    total_docs   = sum(y["total_docs"]   for y in years)
    total_topics = sum(y["total_topics"] for y in years)

    output = {
        "pipeline_version": "3.0",
        "generated_at":     timestamp(),
        "summary": {
            "year_range":   f"{config.YEAR_START}–{config.YEAR_END}",
            "total_years":  len(years),
            "total_docs":   total_docs,
            "total_topics": total_topics,
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