# =========================
# topic_modelling.py (LLM Version - String Cluster IDs)
# =========================

import argparse
import pandas as pd
from openai import OpenAI
import os
import json
import time
from utils import get_logger, save_json, timestamp, log_banner
import config

logger = get_logger("topic_modelling")

client = OpenAI(
    api_key=config.LLM_API_KEY,
    base_url="http://localhost:11434/v1"
)

def run_llm_topic_modelling(df: pd.DataFrame, n_topics: int) -> dict:
    log_banner(logger, "LLM Topic Modelling")
    results = {}

    for cluster_id in sorted(df["cluster"].unique()):
        # Skip noise clusters for any year (e.g., "2022_-1", "2023_-1")
        if str(cluster_id).endswith("_-1"): 
            continue

        subset = df[df["cluster"] == cluster_id]
        docs = subset["processed"].tolist()

        if len(docs) < 5:
            logger.warning(f"Cluster {cluster_id}: too few docs ({len(docs)}), skipping.")
            continue

        logger.info(f"Cluster {cluster_id}: extracting topics for {len(docs)} docs via LLM")

        try:
            sample_size = min(len(docs), 15) 
            sampled_docs = "\n- ".join(docs[:sample_size])

            prompt = f"""
            I have a cluster of related text documents. Based on the following sample of documents, 
            identify the top {n_topics} main topics. 
            
            Documents:
            - {sampled_docs}

            You must respond ONLY with a valid JSON object. The object must contain a single key called "topics", which contains an array of objects.
            Each object in the array must have a "label" (a 2-3 word string) and a "top_words" array (5 keyword strings).
            Example: {{"topics": [{{"label": "Customer Support", "top_words": ["refund", "help", "agent", "wait", "call"]}}]}}
            """

            response = client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"} 
            )

            llm_output = response.choices[0].message.content
            
            clean_text = llm_output.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            topic_data = json.loads(clean_text)
            topics_list = topic_data.get("topics", [])

            cluster_topics = []
            for tid, t in enumerate(topics_list):
                if isinstance(t, dict):
                    cluster_topics.append({
                        "topic_id": tid,
                        "label": t.get("label", "Unknown"),
                        "top_words": t.get("top_words", []),
                        "count": len(docs) 
                    })

            results[str(cluster_id)] = {
                "cluster_id": str(cluster_id), # FIXED: Kept as string
                "n_docs": len(docs),
                "n_topics": len(cluster_topics),
                "topics": cluster_topics,
            }

        except Exception as e:
            logger.error(f"Cluster {cluster_id} LLM Topic failed: {e}")
            results[str(cluster_id)] = {"cluster_id": str(cluster_id), "error": str(e)} # FIXED

        time.sleep(0.1) 

    return results

def main(n_topics: int = 3):
    logger.info("Loading clustered dataset...")
    df = pd.read_csv(config.CLUSTERED_FILE)
    df = df.dropna(subset=["processed"])
    
    results = run_llm_topic_modelling(df, n_topics=n_topics)

    output = {
        "generated_at": timestamp(),
        "model_type": "LLM_Prompting",
        "clusters": results,
    }

    save_json(output, config.TOPIC_OUTPUT_FILE)
    logger.info(f"Topic results saved to {config.TOPIC_OUTPUT_FILE}")
    return output

if __name__ == "__main__":
    main()