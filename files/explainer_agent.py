# =========================
# explainer_agent.py
# Local LLM version using Ollama
# =========================

import json
import time
import argparse
import requests
from utils import get_logger, save_json, load_json, timestamp, log_banner
import config

logger = get_logger("explainer_agent")

EXPLAINED_OUTPUT_FILE = "outputs/explained_results.json"
TOP_N_PER_YEAR = 10

# 🔥 CHANGE THIS if you want another model
LOCAL_LLM_MODEL = "mistral"   # or "phi3", "llama3"


# ---------------------------------------------------------------------------
# LOCAL LLM CALL (OLLAMA)
# ---------------------------------------------------------------------------
def call_local_llm(prompt: str, model: str = LOCAL_LLM_MODEL) -> str:
    """
    Calls a local LLM using Ollama API.
    Make sure Ollama is running: `ollama serve`
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        result = response.json()
        return result.get("response", "").strip()

    except Exception as e:
        logger.error(f"Local LLM call failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------
def build_prompt(year: int, topic_label: str, top_words: list,
                 count: int, dominant_sentiment: str) -> str:

    words_str = ", ".join(top_words[:8])

    sentiment_phrase = {
        "POSITIVE": "mostly positive reactions",
        "NEGATIVE": "mostly negative or critical reactions",
        "NEUTRAL":  "mixed or neutral reactions",
    }.get(dominant_sentiment, "mixed reactions")

    return f"""
You are explaining social media discussions about the Panagbenga Festival in Baguio City.

Topic keywords: {words_str}
Number of posts: {count}
Audience reaction: {sentiment_phrase}

Write 1-2 simple sentences explaining what people are talking about.
Start with: "In {year}, people were talking about..."
Keep it clear, natural, and under 50 words.
"""


# ---------------------------------------------------------------------------
# GET TOP TOPICS
# ---------------------------------------------------------------------------
def get_top_topics_per_year(results: dict, top_n: int = TOP_N_PER_YEAR):
    top_by_year = {}

    for year_entry in results.get("years", []):
        year = year_entry["year"]
        all_topics = []

        for cluster in year_entry.get("clusters", []):
            cid = cluster["cluster_id"]
            if cid == -1:
                continue

            for topic in cluster.get("topics", []):
                all_topics.append({
                    **topic,
                    "_cluster_id": cid,
                    "_year": year,
                })

        all_topics.sort(key=lambda t: t.get("count", 0), reverse=True)
        top_by_year[year] = all_topics[:top_n]

    return top_by_year


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(results_path: str = None, delay_seconds: float = 0.3):

    log_banner(logger, "Explainer Agent (Local LLM)")

    src = results_path or config.FINAL_OUTPUT_FILE
    logger.info(f"Loading results from {src}…")

    results = load_json(src)
    if not results:
        raise FileNotFoundError(f"No results found at {src}")

    top_by_year = get_top_topics_per_year(results)
    logger.info(f"Years found: {sorted(top_by_year.keys())}")

    explained = json.loads(json.dumps(results))
    explained["generated_at"] = timestamp()
    explained["explainer_model"] = f"{LOCAL_LLM_MODEL} (local via Ollama)"

    explanations_added = 0

    for year, top_topics in sorted(top_by_year.items()):
        logger.info(f"\n── Year {year}: {len(top_topics)} topics ──")

        for rank, topic in enumerate(top_topics, start=1):

            prompt = build_prompt(
                year=year,
                topic_label=topic.get("label", ""),
                top_words=topic.get("top_words", []),
                count=topic.get("count", 0),
                dominant_sentiment=topic.get("sentiment", {}).get("dominant", "NEUTRAL"),
            )

            logger.info(f"[{rank}] Topic {topic['topic_id']} → generating...")

            explanation = call_local_llm(prompt)

            # fallback if model fails
            if not explanation:
                explanation = (
                    f"In {year}, people were talking about "
                    + ", ".join(topic.get("top_words", [])[:4]) + "."
                )

            # inject explanation
            cluster_id = topic["_cluster_id"]
            topic_id = topic["topic_id"]

            for year_entry in explained.get("years", []):
                if year_entry["year"] != year:
                    continue

                for cluster in year_entry.get("clusters", []):
                    if cluster["cluster_id"] != cluster_id:
                        continue

                    for t in cluster.get("topics", []):
                        if t["topic_id"] == topic_id:
                            t["explanation"] = explanation
                            t["top_10_rank"] = rank
                            explanations_added += 1

            logger.info(f"✓ {explanation[:80]}")

            time.sleep(delay_seconds)

    # attach top 10 summary
    for year_entry in explained.get("years", []):
        year = year_entry["year"]
        top = top_by_year.get(year, [])

        year_entry["top_10_topics"] = []

        for rank, t in enumerate(top, start=1):
            cluster_id = t["_cluster_id"]
            topic_id = t["topic_id"]

            explanation = ""

            for cluster in year_entry.get("clusters", []):
                if cluster["cluster_id"] != cluster_id:
                    continue

                for tt in cluster.get("topics", []):
                    if tt["topic_id"] == topic_id:
                        explanation = tt.get("explanation", "")

            year_entry["top_10_topics"].append({
                "rank": rank,
                "topic_id": topic_id,
                "cluster_id": cluster_id,
                "label": t.get("label", ""),
                "count": t.get("count", 0),
                "top_words": t.get("top_words", [])[:6],
                "sentiment": t.get("sentiment", {}),
                "explanation": explanation,
            })

    save_json(explained, EXPLAINED_OUTPUT_FILE)

    logger.info(f"\n✓ Saved → {EXPLAINED_OUTPUT_FILE}")
    logger.info(f"Total explanations: {explanations_added}")

    return explained


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    main(results_path=args.results, delay_seconds=args.delay)