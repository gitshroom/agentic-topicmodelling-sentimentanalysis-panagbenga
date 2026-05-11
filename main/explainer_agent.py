# =========================
# explainer_agent.py
# Generates plain-language explanations for top topics per year via Ollama.
#
# Works with the flat year → topics structure (no cluster layer).
# Final output structure adds "explanation" and "top_10_rank" to each
# topic dict, and attaches a "top_10_topics" summary list to each year.
# =========================

import json
import time
import argparse
import requests
from utils import get_logger, save_json, load_json, timestamp, log_banner
import config

logger = get_logger("explainer_agent")

EXPLAINED_OUTPUT_FILE = "outputs/explained_results.json"
TOP_N_PER_YEAR        = 10
LOCAL_LLM_MODEL       = "qwen2.5:3b"


# ---------------------------------------------------------------------------
# LOCAL LLM CALL (OLLAMA)
# ---------------------------------------------------------------------------

def call_local_llm(prompt: str, model: str = LOCAL_LLM_MODEL) -> str:
    """Call a local Ollama model. Make sure `ollama serve` is running."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Local LLM call failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------

def build_prompt(year: int, topic_label: str, top_words: list,
                 count: int, dominant_sentiment: str) -> str:
    words_str = ", ".join(top_words[:10])
    sentiment_phrase = {
        "POSITIVE": "mostly positive reactions",
        "NEGATIVE": "mostly complaints or criticism",
        "NEUTRAL":  "mixed or neutral reactions",
    }.get(dominant_sentiment, "mixed reactions")

    return f"""
You are analyzing social media discussions about the Panagbenga Festival in Baguio City.

Topic keywords:
{words_str}

Topic label:
{topic_label}

Post count:
{count}

Audience reaction:
{sentiment_phrase}

Task:
Write a short natural explanation of what people are discussing.

Rules:
- Use simple language
- Maximum 2 sentences
- Maximum 50 words
- Do NOT repeat the keywords directly
- Do NOT say "the topic is about"
- Be specific and human-like

Explanation:
"""


# ---------------------------------------------------------------------------
# GET TOP TOPICS PER YEAR  (flat structure: years is a list of year objects)
# ---------------------------------------------------------------------------

def get_top_topics_per_year(results: dict, top_n: int = TOP_N_PER_YEAR) -> dict:
    """
    Returns { year(int): [ top_n topic dicts sorted by count desc ] }
    Works with the flat year → topics structure from results_formatter.
    """
    top_by_year: dict[int, list] = {}

    for year_entry in results.get("years", []):
        year       = year_entry["year"]
        all_topics = list(year_entry.get("topics", []))
        all_topics.sort(key=lambda t: t.get("count", 0), reverse=True)
        top_by_year[year] = all_topics[:top_n]

    return top_by_year


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(results_path: str = None, delay_seconds: float = 0.3):
    log_banner(logger, "Explainer Agent (Local LLM via Ollama)")

    src = results_path or config.FINAL_OUTPUT_FILE
    logger.info(f"Loading results from {src}…")

    results = load_json(src)
    if not results:
        raise FileNotFoundError(f"No results found at {src}")

    top_by_year = get_top_topics_per_year(results)
    logger.info(f"Years found: {sorted(top_by_year.keys())}")

    # Deep-copy so we can inject fields without touching the source dict
    explained = json.loads(json.dumps(results))
    explained["generated_at"]    = timestamp()
    explained["explainer_model"] = f"{LOCAL_LLM_MODEL} (local via Ollama)"

    explanations_added = 0

    # ── Pass 1: generate and inject explanations ──────────────────────────
    for year, top_topics in sorted(top_by_year.items()):
        logger.info(f"\n── Year {year}: {len(top_topics)} topics ──")

        for rank, topic in enumerate(top_topics, start=1):
            topic_id = topic["topic_id"]

            prompt = build_prompt(
                year=year,
                topic_label=topic.get("label", ""),
                top_words=topic.get("top_words", []),
                count=topic.get("count", 0),
                dominant_sentiment=topic.get("sentiment", {}).get("dominant", "NEUTRAL"),
            )

            logger.info(f"[{rank}] Topic {topic_id} → generating…")
            explanation = call_local_llm(prompt)

            if not explanation:
                explanation = (
                    f"In {year}, people were talking about "
                    + ", ".join(topic.get("top_words", [])[:4]) + "."
                )

            # Inject into the deep-copied structure (flat: year → topics)
            for year_entry in explained.get("years", []):
                if year_entry["year"] != year:
                    continue
                for t in year_entry.get("topics", []):
                    if t["topic_id"] == topic_id:
                        t["explanation"]  = explanation
                        t["top_10_rank"]  = rank
                        explanations_added += 1

            logger.info(f"✓ {explanation[:80]}")
            time.sleep(delay_seconds)

    # ── Pass 2: attach top-10 summary list to each year entry ────────────
    for year_entry in explained.get("years", []):
        year = year_entry["year"]
        top  = top_by_year.get(year, [])

        # Build a quick lookup of explanation already injected above
        topic_map = {t["topic_id"]: t for t in year_entry.get("topics", [])}

        year_entry["top_10_topics"] = []
        for rank, t in enumerate(top, start=1):
            tid         = t["topic_id"]
            full_topic  = topic_map.get(tid, t)
            year_entry["top_10_topics"].append({
                "rank":        rank,
                "topic_id":    tid,
                "label":       full_topic.get("label", ""),
                "count":       full_topic.get("count", 0),
                "top_words":   full_topic.get("top_words", [])[:6],
                "sentiment":   full_topic.get("sentiment", {}),
                "explanation": full_topic.get("explanation", ""),
            })

    save_json(explained, EXPLAINED_OUTPUT_FILE)
    logger.info(f"\n✓ Saved → {EXPLAINED_OUTPUT_FILE}")
    logger.info(f"Total explanations added: {explanations_added}")
    return explained


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--delay",   type=float, default=0.3)
    args = parser.parse_args()
    main(results_path=args.results, delay_seconds=args.delay)