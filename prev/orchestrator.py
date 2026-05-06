# =========================
# orchestrator.py
# Master agent: runs topic_modelling.py and sentiment_analysis.py in parallel,
# validates results against quality thresholds, and retries with adjusted
# parameters if quality is insufficient.
# =========================

import concurrent.futures
import sys
from utils import get_logger, load_json, save_json, timestamp, log_banner
import config
import topic_modelling
import sentiment_analysis
import results_formatter

logger = get_logger("orchestrator")


# ---------------------------------------------------------------------------
# Quality validation
# ---------------------------------------------------------------------------

def validate_topics(topic_data: dict) -> tuple[bool, list[str]]:
    """Return (passed, list_of_issues) for topic modelling results."""
    issues = []
    clusters = topic_data.get("clusters", {})

    if not clusters:
        return False, ["No clusters in topic results."]

    all_noise_ratios = []
    all_coherences = []
    total_topics = 0
    errored = 0

    for cid, cdata in clusters.items():
        if "error" in cdata:
            errored += 1
            continue
        total_topics += cdata.get("n_topics", 0)
        all_noise_ratios.append(cdata.get("noise_ratio", 0))
        coherence = cdata.get("coherence_proxy", 0)
        if coherence > 0:
            all_coherences.append(coherence)

    if total_topics < config.MIN_TOPICS:
        issues.append(
            f"Too few topics discovered: {total_topics} < {config.MIN_TOPICS}"
        )

    if all_noise_ratios:
        avg_noise = sum(all_noise_ratios) / len(all_noise_ratios)
        if avg_noise > config.MAX_NOISE_RATIO:
            issues.append(
                f"Avg noise ratio too high: {avg_noise:.2f} > {config.MAX_NOISE_RATIO}"
            )

    if all_coherences:
        avg_coherence = sum(all_coherences) / len(all_coherences)
        if avg_coherence < config.MIN_TOPIC_COHERENCE:
            issues.append(
                f"Avg coherence too low: {avg_coherence:.3f} < {config.MIN_TOPIC_COHERENCE}"
            )

    if errored:
        issues.append(f"{errored} cluster(s) errored during topic modelling.")

    return len(issues) == 0, issues


def validate_sentiment(sentiment_data: dict) -> tuple[bool, list[str]]:
    """Return (passed, list_of_issues) for sentiment results."""
    issues = []
    clusters = sentiment_data.get("clusters", {})

    if not clusters:
        return False, ["No clusters in sentiment results."]

    coverages = []
    confidences = []

    for cid, cdata in clusters.items():
        coverages.append(cdata.get("coverage", 0))
        confidences.append(cdata.get("avg_confidence", 0))

    avg_coverage = sum(coverages) / len(coverages) if coverages else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    if avg_coverage < config.MIN_SENTIMENT_COVERAGE:
        issues.append(
            f"Avg sentiment coverage too low: {avg_coverage:.2f} < {config.MIN_SENTIMENT_COVERAGE}"
        )

    if avg_confidence < config.MIN_SENTIMENT_CONFIDENCE:
        issues.append(
            f"Avg sentiment confidence too low: {avg_confidence:.3f} < {config.MIN_SENTIMENT_CONFIDENCE}"
        )

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Parameter adjustment for retry
# ---------------------------------------------------------------------------

def adjust_topic_params(attempt: int, current_params: dict) -> dict:
    """Loosen topic modelling parameters each retry."""
    params = dict(current_params)

    if config.TOPIC_MODEL_TYPE == "bertopic":
        mts = params.get("min_topic_size", config.BERTOPIC_MIN_TOPIC_SIZE)
        mts = max(2, mts + config.BERTOPIC_MIN_TOPIC_SIZE_DELTA)
        params["min_topic_size"] = mts
        logger.info(f"[feedback] BERTopic min_topic_size → {mts}")
    else:
        ntop = params.get("n_lda_topics", config.LDA_N_TOPICS)
        ntop = ntop + config.LDA_N_TOPICS_DELTA
        params["n_lda_topics"] = ntop
        logger.info(f"[feedback] LDA n_topics → {ntop}")

    return params


def adjust_sentiment_params(attempt: int, current_params: dict) -> dict:
    """Currently sentiment has no dynamic params; placeholder for extension."""
    logger.info("[feedback] No sentiment param adjustments configured.")
    return dict(current_params)


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def run_agents_parallel(topic_params: dict, sentiment_params: dict):
    """Run topic modelling and sentiment analysis concurrently."""
    logger.info("Launching topic modelling and sentiment analysis in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        topic_future = executor.submit(
            topic_modelling.main,
            min_topic_size=topic_params.get("min_topic_size"),
            nr_topics=topic_params.get("nr_topics"),
            n_lda_topics=topic_params.get("n_lda_topics"),
        )
        sentiment_future = executor.submit(
            sentiment_analysis.main,
            model_name=sentiment_params.get("model_name"),
            batch_size=sentiment_params.get("batch_size"),
        )

        topic_data = topic_future.result()
        sentiment_data = sentiment_future.result()

    return topic_data, sentiment_data


# ---------------------------------------------------------------------------
# Main orchestration loop
# ---------------------------------------------------------------------------

def main():
    log_banner(logger, "Orchestrator — agentic pipeline")
    logger.info(f"Started at {timestamp()}")

    # Initial parameters (from config)
    topic_params = {
        "min_topic_size": config.BERTOPIC_MIN_TOPIC_SIZE,
        "nr_topics": config.BERTOPIC_NR_TOPICS,
        "n_lda_topics": config.LDA_N_TOPICS,
    }
    sentiment_params = {
        "model_name": config.SENTIMENT_MODEL,
        "batch_size": config.SENTIMENT_BATCH_SIZE,
    }

    topic_ok = False
    sentiment_ok = False
    final_topic_data = None
    final_sentiment_data = None
    all_topic_issues = []
    all_sentiment_issues = []

    for attempt in range(1, config.MAX_RETRIES + 1):
        log_banner(logger, f"Attempt {attempt} / {config.MAX_RETRIES}")

        # Only re-run agents that previously failed (or first attempt)
        if not topic_ok or not sentiment_ok:
            run_topic = not topic_ok
            run_sentiment = not sentiment_ok

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}

                if run_topic:
                    futures["topic"] = executor.submit(
                        topic_modelling.main,
                        min_topic_size=topic_params.get("min_topic_size"),
                        nr_topics=topic_params.get("nr_topics"),
                        n_lda_topics=topic_params.get("n_lda_topics"),
                    )
                if run_sentiment:
                    futures["sentiment"] = executor.submit(
                        sentiment_analysis.main,
                        model_name=sentiment_params.get("model_name"),
                        batch_size=sentiment_params.get("batch_size"),
                    )

                if "topic" in futures:
                    final_topic_data = futures["topic"].result()
                if "sentiment" in futures:
                    final_sentiment_data = futures["sentiment"].result()

        # Validate
        topic_ok, topic_issues = validate_topics(final_topic_data)
        sentiment_ok, sentiment_issues = validate_sentiment(final_sentiment_data)

        all_topic_issues = topic_issues
        all_sentiment_issues = sentiment_issues

        logger.info(f"Topic modelling validation: {'PASS' if topic_ok else 'FAIL'}")
        for iss in topic_issues:
            logger.warning(f"  - {iss}")

        logger.info(f"Sentiment validation: {'PASS' if sentiment_ok else 'FAIL'}")
        for iss in sentiment_issues:
            logger.warning(f"  - {iss}")

        if topic_ok and sentiment_ok:
            logger.info("All quality checks passed.")
            break

        if attempt < config.MAX_RETRIES:
            logger.info("Adjusting parameters for next attempt...")
            if not topic_ok:
                topic_params = adjust_topic_params(attempt, topic_params)
            if not sentiment_ok:
                sentiment_params = adjust_sentiment_params(attempt, sentiment_params)
        else:
            logger.warning(
                f"Max retries ({config.MAX_RETRIES}) reached. "
                "Proceeding with best available results."
            )

    # Format and save final results
    log_banner(logger, "Formatting final results")
    results_formatter.main(
        topic_data=final_topic_data,
        sentiment_data=final_sentiment_data,
        topic_issues=all_topic_issues,
        sentiment_issues=all_sentiment_issues,
        passed=(topic_ok and sentiment_ok),
    )

    logger.info(f"Pipeline complete. Output: {config.FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
