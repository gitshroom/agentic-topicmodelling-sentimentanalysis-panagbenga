# =========================
# orchestrator.py
# Master agent: runs topic_modelling and sentiment_analysis in parallel,
# validates results per year, and retries with adjusted parameters.
# Now operates on year → cluster → topic structure.
# =========================

import concurrent.futures
from utils import get_logger, save_json, timestamp, log_banner
import config
import topic_modelling
import sentiment_analysis
import results_formatter

logger = get_logger("orchestrator")


# ---------------------------------------------------------------------------
# Quality validation
# ---------------------------------------------------------------------------

def validate_topics(topic_data: dict) -> tuple[bool, list[str]]:
    """Return (passed, issues) by scanning all years."""
    issues: list[str] = []
    all_years = topic_data.get("years", {})

    if not all_years:
        return False, ["No year data in topic results."]

    total_topics    = 0
    noise_ratios    = []
    coherences      = []
    errored_slices  = 0

    for year_str, clusters in all_years.items():
        for cid, cdata in clusters.items():
            if "error" in cdata:
                errored_slices += 1
                continue
            total_topics += cdata.get("n_topics", 0)
            nr = cdata.get("noise_ratio")
            if nr is not None:
                noise_ratios.append(nr)
            cp = cdata.get("coherence_proxy")
            if cp and cp > 0:
                coherences.append(cp)

    if total_topics < config.MIN_TOPICS:
        issues.append(f"Too few topics total: {total_topics} < {config.MIN_TOPICS}")

    if noise_ratios:
        avg_noise = sum(noise_ratios) / len(noise_ratios)
        if avg_noise > config.MAX_NOISE_RATIO:
            issues.append(f"Avg noise ratio too high: {avg_noise:.2f} > {config.MAX_NOISE_RATIO}")

    if coherences:
        avg_coh = sum(coherences) / len(coherences)
        if avg_coh < config.MIN_TOPIC_COHERENCE:
            issues.append(f"Avg coherence too low: {avg_coh:.3f} < {config.MIN_TOPIC_COHERENCE}")

    if errored_slices:
        issues.append(f"{errored_slices} year-cluster slice(s) errored during topic modelling.")

    return len(issues) == 0, issues


def validate_sentiment(sentiment_data: dict) -> tuple[bool, list[str]]:
    """Return (passed, issues) by scanning all years → clusters → topics."""
    issues: list[str] = []
    all_years = sentiment_data.get("years", {})

    if not all_years:
        return False, ["No year data in sentiment results."]

    coverages:    list[float] = []
    confidences:  list[float] = []

    for year_str, clusters in all_years.items():
        for cid, cdata in clusters.items():
            for topic_sent in cdata.get("topics", []):
                coverages.append(topic_sent.get("coverage", 0))
                confidences.append(topic_sent.get("avg_confidence", 0))

    if not coverages:
        return False, ["No topic-level sentiment data found."]

    avg_cov  = sum(coverages)    / len(coverages)
    avg_conf = sum(confidences)  / len(confidences)

    if avg_cov < config.MIN_SENTIMENT_COVERAGE:
        issues.append(f"Avg sentiment coverage too low: {avg_cov:.2f} < {config.MIN_SENTIMENT_COVERAGE}")

    if avg_conf < config.MIN_SENTIMENT_CONFIDENCE:
        issues.append(f"Avg sentiment confidence too low: {avg_conf:.3f} < {config.MIN_SENTIMENT_CONFIDENCE}")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Parameter adjustment for retry
# ---------------------------------------------------------------------------

def adjust_topic_params(attempt: int, params: dict) -> dict:
    p = dict(params)
    if config.TOPIC_MODEL_TYPE == "bertopic":
        mts = max(2, p.get("min_topic_size", config.BERTOPIC_MIN_TOPIC_SIZE) + config.BERTOPIC_MIN_TOPIC_SIZE_DELTA)
        p["min_topic_size"] = mts
        logger.info(f"[retry] BERTopic min_topic_size → {mts}")
    else:
        ntop = p.get("n_lda_topics", config.LDA_N_TOPICS) + config.LDA_N_TOPICS_DELTA
        p["n_lda_topics"] = ntop
        logger.info(f"[retry] LDA n_topics → {ntop}")
    return p


def adjust_sentiment_params(attempt: int, params: dict) -> dict:
    logger.info("[retry] No sentiment param adjustments configured.")
    return dict(params)


# ---------------------------------------------------------------------------
# Main orchestration loop
# ---------------------------------------------------------------------------

def main():
    log_banner(logger, "Orchestrator — per-year agentic pipeline")
    logger.info(f"Started at {timestamp()}")

    topic_params = {
        "min_topic_size": config.BERTOPIC_MIN_TOPIC_SIZE,
        "nr_topics":      config.BERTOPIC_NR_TOPICS,
        "n_lda_topics":   config.LDA_N_TOPICS,
    }
    sentiment_params = {
        "model_name": config.SENTIMENT_MODEL,
        "batch_size": config.SENTIMENT_BATCH_SIZE,
    }

    topic_ok     = False
    sentiment_ok = False
    final_topic_data     = None
    final_sentiment_data = None
    all_topic_issues:     list[str] = []
    all_sentiment_issues: list[str] = []

    for attempt in range(1, config.MAX_RETRIES + 1):
        log_banner(logger, f"Attempt {attempt} / {config.MAX_RETRIES}")

        run_topic     = not topic_ok
        run_sentiment = not sentiment_ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures: dict[str, concurrent.futures.Future] = {}

            if run_topic:
                futures["topic"] = executor.submit(
                    topic_modelling.main,
                    min_topic_size=topic_params.get("min_topic_size"),
                    nr_topics=topic_params.get("nr_topics"),
                    n_lda_topics=topic_params.get("n_lda_topics"),
                )

            # Sentiment depends on topic results being present on disk,
            # so if topics also need re-running we wait for that first.
            if run_topic and run_sentiment:
                # Resolve topic first so sentiment can read the updated file
                final_topic_data = futures["topic"].result()
                futures.pop("topic")
                futures["sentiment"] = executor.submit(
                    sentiment_analysis.main,
                    model_name=sentiment_params.get("model_name"),
                    batch_size=sentiment_params.get("batch_size"),
                )
            elif run_sentiment and not run_topic:
                futures["sentiment"] = executor.submit(
                    sentiment_analysis.main,
                    model_name=sentiment_params.get("model_name"),
                    batch_size=sentiment_params.get("batch_size"),
                )

            if "topic"     in futures:
                final_topic_data     = futures["topic"].result()
            if "sentiment" in futures:
                final_sentiment_data = futures["sentiment"].result()

        # Validate
        topic_ok,     topic_issues     = validate_topics(final_topic_data)
        sentiment_ok, sentiment_issues = validate_sentiment(final_sentiment_data)

        all_topic_issues     = topic_issues
        all_sentiment_issues = sentiment_issues

        logger.info(f"Topic validation:     {'PASS' if topic_ok else 'FAIL'}")
        for iss in topic_issues:
            logger.warning(f"  - {iss}")

        logger.info(f"Sentiment validation: {'PASS' if sentiment_ok else 'FAIL'}")
        for iss in sentiment_issues:
            logger.warning(f"  - {iss}")

        if topic_ok and sentiment_ok:
            logger.info("All quality checks passed.")
            break

        if attempt < config.MAX_RETRIES:
            logger.info("Adjusting parameters for next attempt…")
            if not topic_ok:
                topic_params = adjust_topic_params(attempt, topic_params)
            if not sentiment_ok:
                sentiment_params = adjust_sentiment_params(attempt, sentiment_params)
        else:
            logger.warning(
                f"Max retries ({config.MAX_RETRIES}) reached. "
                "Proceeding with best available results."
            )

    # Format and save
    log_banner(logger, "Formatting final results")
    results_formatter.main(
        topic_data=final_topic_data,
        sentiment_data=final_sentiment_data,
        topic_issues=all_topic_issues,
        sentiment_issues=all_sentiment_issues,
        passed=(topic_ok and sentiment_ok),
    )

    logger.info(f"Pipeline complete. Output → {config.FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
