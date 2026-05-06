# =========================
# results_formatter.py
# Agent: merges topic and sentiment results into a clean final JSON.
# Called by orchestrator.py once quality checks pass (or retries exhaust).
# =========================

from utils import get_logger, save_json, load_json, timestamp, log_banner
import config

logger = get_logger("results_formatter")


def merge_cluster_results(topic_data: dict, sentiment_data: dict) -> list:
    """Combine topic and sentiment results keyed by cluster id."""
    topic_clusters = topic_data.get("clusters", {})
    sentiment_clusters = sentiment_data.get("clusters", {})

    all_ids = set(topic_clusters.keys()) | set(sentiment_clusters.keys())
    merged = []

    for cid in sorted(all_ids, key=lambda x: int(x)):
        tdata = topic_clusters.get(cid, {})
        sdata = sentiment_clusters.get(cid, {})

        entry = {
            "cluster_id": int(cid),
            "n_docs": tdata.get("n_docs") or sdata.get("n_docs"),
            "topics": {
                "n_topics": tdata.get("n_topics", 0),
                "noise_ratio": tdata.get("noise_ratio", None),
                "coherence_proxy": tdata.get("coherence_proxy", None),
                "top_topics": tdata.get("topics", []),
                "error": tdata.get("error", None),
            },
            "sentiment": {
                "dominant": sdata.get("dominant_sentiment", None),
                "avg_confidence": sdata.get("avg_confidence", None),
                "coverage": sdata.get("coverage", None),
                "label_distribution": sdata.get("label_distribution", {}),
                "label_ratios": sdata.get("label_ratios", {}),
            },
        }
        merged.append(entry)

    return merged


def main(
    topic_data: dict = None,
    sentiment_data: dict = None,
    topic_issues: list = None,
    sentiment_issues: list = None,
    passed: bool = True,
):
    log_banner(logger, "Results formatter")

    # Allow standalone usage by loading from files
    if topic_data is None:
        logger.info(f"Loading topic results from {config.TOPIC_OUTPUT_FILE}")
        topic_data = load_json(config.TOPIC_OUTPUT_FILE)

    if sentiment_data is None:
        logger.info(f"Loading sentiment results from {config.SENTIMENT_OUTPUT_FILE}")
        sentiment_data = load_json(config.SENTIMENT_OUTPUT_FILE)

    if topic_issues is None:
        topic_issues = []
    if sentiment_issues is None:
        sentiment_issues = []

    clusters = merge_cluster_results(topic_data, sentiment_data)

    # Summary stats
    total_docs = sum(c["n_docs"] or 0 for c in clusters if c["cluster_id"] != -1)
    total_topics = sum(c["topics"]["n_topics"] for c in clusters if c["cluster_id"] != -1)
    sentiments = [c["sentiment"]["dominant"] for c in clusters if c["sentiment"]["dominant"]]
    sentiment_summary = {}
    for s in sentiments:
        sentiment_summary[s] = sentiment_summary.get(s, 0) + 1

    output = {
        "pipeline_version": "1.0",
        "generated_at": timestamp(),
        "quality_passed": passed,
        "validation": {
            "topic_issues": topic_issues,
            "sentiment_issues": sentiment_issues,
        },
        "summary": {
            "total_clusters": len([c for c in clusters if c["cluster_id"] != -1]),
            "total_docs_clustered": total_docs,
            "total_topics_discovered": total_topics,
            "dominant_sentiment_by_cluster": sentiment_summary,
        },
        "models": {
            "topic_model_type": topic_data.get("model_type"),
            "topic_parameters": topic_data.get("parameters"),
            "sentiment_model": sentiment_data.get("model"),
        },
        "clusters": clusters,
    }

    save_json(output, config.FINAL_OUTPUT_FILE)
    logger.info(f"Final results written to {config.FINAL_OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    main()
