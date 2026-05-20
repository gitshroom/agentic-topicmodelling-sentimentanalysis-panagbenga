# =========================
# topic_modelling.py (BERTopic)
# =========================

import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan
import os
import config
from utils import get_logger, save_json, timestamp, log_banner

logger = get_logger("topic_modelling_bertopic")

def main(**kwargs):
    log_banner(logger, "BERTopic Per-Cluster Modelling")
    
    # 1. Load the clustered dataset
    logger.info(f"Loading {config.CLUSTERED_FILE}...")
    df = pd.read_csv(config.CLUSTERED_FILE)
    df = df.dropna(subset=["processed"])

    # 2. Load the Embedding Model
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

    cluster_results = {}

    # 3. Loop through every unique cluster (e.g., "2023_1")
    for cluster_id in sorted(df["cluster"].unique()):
        # Skip HDBSCAN noise from the previous stage
        if str(cluster_id).endswith("_-1"):
            continue

        subset = df[df["cluster"] == cluster_id]
        docs = subset["processed"].tolist()

        if len(docs) < 10: 
            logger.warning(f"Cluster {cluster_id}: Only {len(docs)} docs. Skipping BERTopic.")
            continue

        logger.info(f"Cluster {cluster_id}: Running BERTopic on {len(docs)} docs")

        try:
            # Generate embeddings just for this specific cluster
            embeddings = embedding_model.encode(docs, show_progress_bar=False)

            # Iteration 5 Scaling Rules for small clusters
            local_min_cluster = max(3, min(10, len(docs) // 8))
            local_min_samples = max(1, min(5, local_min_cluster - 1))
            local_n_neighbors = min(15, len(docs) - 1)

            # Configure BERTopic components
            umap_model = umap.UMAP(n_neighbors=local_n_neighbors, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
            hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=local_min_cluster, min_samples=local_min_samples, metric='euclidean', cluster_selection_method='eom')
            vectorizer_model = CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0)
            representation_model = KeyBERTInspired()

            # Initialize BERTopic
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                min_topic_size=local_min_cluster,
                nr_topics=10 # Max 10 topics per cluster as per Iteration 5
            )

            # Fit the model
            topics, _ = topic_model.fit_transform(docs, embeddings)
            topic_info = topic_model.get_topic_info()

            # Format the output to match our JSON structure
            cluster_topics = []
            for _, row in topic_info.iterrows():
                t_id = row['Topic']
                if t_id == -1: 
                    continue # Skip BERTopic's internal noise bucket
                
                # Get top 5 keywords
                words = [word for word, _ in topic_model.get_topic(t_id)][:5]
                
                cluster_topics.append({
                    "topic_id": int(t_id),
                    "label": " / ".join(words[:3]).title(), # Create a label out of top 3 words
                    "top_words": words,
                    "count": int(row['Count'])
                })

            cluster_results[str(cluster_id)] = {
                "cluster_id": str(cluster_id),
                "n_docs": len(docs),
                "n_topics": len(cluster_topics),
                "topics": cluster_topics,
            }

        except Exception as e:
            logger.error(f"Cluster {cluster_id} BERTopic failed: {e}")
            cluster_results[str(cluster_id)] = {"cluster_id": str(cluster_id), "error": str(e)}

    # 4. Save the Output
    output = {
        "generated_at": timestamp(),
        "model_type": "BERTopic_Extractive",
        "clusters": cluster_results,
    }

    save_json(output, config.TOPIC_OUTPUT_FILE)
    logger.info(f"Topic results saved to {config.TOPIC_OUTPUT_FILE}")
    return output

if __name__ == "__main__":
    main()