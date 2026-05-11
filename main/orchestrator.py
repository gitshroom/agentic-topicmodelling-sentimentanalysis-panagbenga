# =========================
# orchestrator.py
# Agentic pipeline: topic modelling → sentiment (sequential, per year).
#
# Data flow:
#   topic_modelling  →  TOPIC_OUTPUT_FILE  (year → topics)
#   sentiment_analysis reads TOPIC_OUTPUT_FILE, outputs SENTIMENT_OUTPUT_FILE
#   results_formatter merges both into FINAL_OUTPUT_FILE
#   explainer_agent annotates FINAL_OUTPUT_FILE → explained_results.json
# =========================

from utils import get_logger, save_json, load_json, timestamp, log_banner
import config
import topic_modelling
import sentiment_analysis
import results_formatter
import explainer_agent

logger = get_logger("orchestrator")


def main():
    log_banner(logger, "Orchestrator — per-year topic → sentiment pipeline")
    logger.info(f"Started at {timestamp()}")

    # ── Step A: Topic modelling ───────────────────────────────────────────
    log_banner(logger, "Step A: Topic Modelling")
    logger.info("Running topic modelling…")
    topic_data = topic_modelling.main(
        min_topic_size=config.BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics=config.BERTOPIC_NR_TOPICS,
    )

    # ── Step B: Sentiment analysis ────────────────────────────────────────
    log_banner(logger, "Step B: Sentiment Analysis")
    logger.info("Running sentiment analysis…")
    sentiment_data = sentiment_analysis.main(
        model_name=config.SENTIMENT_MODEL,
        batch_size=config.SENTIMENT_BATCH_SIZE,
        use_processed=config.SENTIMENT_USE_PROCESSED,
    )

    # ── Step C: Format results ────────────────────────────────────────────
    log_banner(logger, "Step C: Formatting Final Results")
    results_formatter.main(
    topic_data=topic_data,
    sentiment_data=sentiment_data,
)

    # ── Step D: Explainer agent ───────────────────────────────────────────
    log_banner(logger, "Step D: Explainer Agent — plain-language topic descriptions")
    try:
        explainer_agent.main()
        logger.info("Explainer agent complete.")
    except Exception as e:
        logger.error(f"Explainer agent failed (non-fatal): {e}")
        logger.info("Dashboard will fall back to raw topic labels.")

    logger.info(f"Pipeline complete → {config.FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()