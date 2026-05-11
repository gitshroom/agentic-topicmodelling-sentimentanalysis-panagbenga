# =========================
# explainer_agent.py
# Generates plain-language explanations for top topics per year via Ollama.
#
# Works with the nested year → cluster → topics structure produced by
# results_formatter v4.0.
#
# Algorithm:
#   1. Flatten all topics across all clusters for a year.
#   2. Sort by count desc, take top N.
#   3. Generate an explanation for each via a local LLM.
#   4. Inject "explanation" and "top_10_rank" back into the nested structure.
#   5. Attach a "top_10_topics" summary list to each year entry.
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
                 count: int, dominant_sentiment: str,
                 cluster_id: int = None) -> str:
    words_str = ", ".join(top_words[:10])
    sentiment_phrase = {
        "POSITIVE": "mostly positive reactions",
        "NEGATIVE": "mostly complaints or criticism",
        "NEUTRAL":  "mixed or neutral reactions",
    }.get(dominant_sentiment, "mixed reactions")

    cluster_hint = (
        f"\nThematic group: cluster {cluster_id}"
        if cluster_id is not None and cluster_id != -1
        else ""
    )

    return f"""
You are analyzing social media discussions about the Panagbenga Festival in Baguio City.

Topic keywords:
{words_str}

Topic label:
{topic_label}{cluster_hint}

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
# FLATTEN TOPICS ACROSS CLUSTERS  (with cluster provenance)
# ---------------------------------------------------------------------------

def flatten_year_topics(year_entry: dict) -> list:
    """
    Return a flat list of (cluster_id, topic_dict) tuples for a year,
    excluding the noise cluster (cluster_id == -1) from ranking but
    keeping it for injection.
    """
    flat = []
    for cluster in year_entry.get("clusters", []):
        cid = cluster["cluster_id"]
        for topic in cluster.get("topics", []):
            flat.append((cid, topic))
    return flat


def get_top_topics_per_year(results: dict, top_n: int = TOP_N_PER_YEAR) -> dict:
    """
    Returns { year(int): [ (cluster_id, topic_dict) … ] }
    sorted by count desc, top_n entries.
    Noise topics (cluster_id == -1) are eligible but deprioritised by
    appearing after real-cluster topics at the same count.
    """
    top_by_year: dict[int, list] = {}

    for year_entry in results.get("years", []):
        year = year_entry["year"]
        flat = flatten_year_topics(year_entry)
        # Sort: real clusters first (cluster_id >= 0), then by count desc
        flat.sort(key=lambda x: (x[0] == -1, -x[1].get("count", 0)))
        top_by_year[year] = flat[:top_n]

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

        for rank, (cid, topic) in enumerate(top_topics, start=1):
            topic_id = topic["topic_id"]

            prompt = build_prompt(
                year=year,
                topic_label=topic.get("label", ""),
                top_words=topic.get("top_words", []),
                count=topic.get("count", 0),
                dominant_sentiment=topic.get("sentiment", {}).get("dominant", "NEUTRAL"),
                cluster_id=cid,
            )

            logger.info(f"[{rank}] Cluster {cid} / Topic {topic_id} → generating…")
            explanation = call_local_llm(prompt)

            if not explanation:
                explanation = (
                    f"In {year}, people were talking about "
                    + ", ".join(topic.get("top_words", [])[:4]) + "."
                )

            # Inject into the deep-copied nested structure
            for year_entry in explained.get("years", []):
                if year_entry["year"] != year:
                    continue
                for cluster in year_entry.get("clusters", []):
                    if cluster["cluster_id"] != cid:
                        continue
                    for t in cluster.get("topics", []):
                        if t["topic_id"] == topic_id:
                            t["explanation"] = explanation
                            t["top_10_rank"] = rank
                            explanations_added += 1

            logger.info(f"✓ {explanation[:80]}")
            time.sleep(delay_seconds)

    # ── Pass 2: attach top-10 summary list to each year entry ────────────
    for year_entry in explained.get("years", []):
        year = year_entry["year"]
        top  = top_by_year.get(year, [])

        # Build a lookup: (cluster_id, topic_id) → enriched topic
        topic_map: dict[tuple, dict] = {}
        for cluster in year_entry.get("clusters", []):
            for t in cluster.get("topics", []):
                topic_map[(cluster["cluster_id"], t["topic_id"])] = t

        year_entry["top_10_topics"] = []
        for rank, (cid, orig_topic) in enumerate(top, start=1):
            tid        = orig_topic["topic_id"]
            full_topic = topic_map.get((cid, tid), orig_topic)

            year_entry["top_10_topics"].append({
                "rank":        rank,
                "cluster_id":  cid,
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
