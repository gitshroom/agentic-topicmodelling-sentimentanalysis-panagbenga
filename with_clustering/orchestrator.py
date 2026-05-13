# =========================
# orchestrator.py
# Pipeline 2 (with_clustering) — agentic year → cluster → topic → sentiment.
#
# Data flow:
#   topic_modelling    → outputs/topic_results.json
#   sentiment_analysis → outputs/sentiment_results.json
#   results_formatter  → outputs/results.json
#   explainer_agent    → outputs/explained_results.json
#
# Note: embeddings.py persists `pre_cluster_label` in CLUSTERED_FILE which
# topic_modelling and sentiment_analysis use to keep clusters aligned.
# =========================

import os

import config
import topic_modelling
import sentiment_analysis
import results_formatter
import explainer_agent
from utils import get_logger, log_banner, load_json, timestamp

logger = get_logger("orchestrator")


def main():
    log_banner(logger, "Orchestrator — Pipeline 2 (with clustering)")
    logger.info(f"Started at {timestamp()}")

    if not os.path.exists(config.CLUSTERED_FILE):
        raise FileNotFoundError(
            f"Embedded dataset not found at {config.CLUSTERED_FILE}. "
            "Run embeddings.py first (or use run_pipeline.py)."
        )

    # ── Step A: Topic modelling ───────────────────────────────────────────
    log_banner(logger, "Step A: Topic Modelling (year → cluster → BERTopic)")
    topic_data = topic_modelling.main(
        min_topic_size=config.BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics=config.BERTOPIC_NR_TOPICS,
    )

    # ── Step B: Sentiment analysis ────────────────────────────────────────
    log_banner(logger, "Step B: Sentiment Analysis (per topic per cluster)")
    sentiment_data = sentiment_analysis.main(
        model_name=config.SENTIMENT_MODEL,
        batch_size=config.SENTIMENT_BATCH_SIZE,
        use_processed=config.SENTIMENT_USE_PROCESSED,
    )

    # ── Step C: Format results ────────────────────────────────────────────
    log_banner(logger, "Step C: Formatting Final Results")
    cluster_summary = load_json(config.CLUSTER_SUMMARY_FILE)
    results_formatter.main(
        topic_data=topic_data,
        sentiment_data=sentiment_data,
        cluster_summary=cluster_summary,
    )

    # ── Step D: Explainer agent (non-fatal) ───────────────────────────────
    log_banner(logger, "Step D: Explainer Agent — plain-language descriptions")
    try:
        explainer_agent.main()
        logger.info("Explainer agent complete.")
    except Exception as e:
        logger.error(f"Explainer agent failed (non-fatal): {e}")
        logger.info("Dashboard will fall back to raw topic labels.")

    logger.info(f"Pipeline complete → {config.FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
