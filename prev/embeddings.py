# =========================
# embeddings.py (Year-Level Clustering)
# =========================

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import os

import config

def load_data():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(config.PREPROCESSED_FILE)
    df = df.dropna(subset=["processed", "year"])
    print(f"[INFO] Dataset loaded: {df.shape}")
    return df

def generate_global_embeddings(texts):
    print(f"[INFO] Loading embedding model: {config.EMBEDDING_MODEL}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("[INFO] Generating 768-dimensional embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    print("[INFO] Normalizing embeddings (L2)...")
    embeddings = normalize(embeddings)
    return embeddings

def process_year_clusters(df):
    """Loops through each year, applies UMAP + HDBSCAN, and formats cluster IDs."""
    
    # Temporarily store embeddings in the dataframe for slicing
    texts = df["processed"].tolist()
    global_embeddings = generate_global_embeddings(texts)
    df["embedding_temp"] = list(global_embeddings)
    
    # Prepare column for explicit cluster IDs
    df["cluster"] = "-1" 
    
    years = sorted(df["year"].unique())
    print(f"\n[INFO] Starting explicit per-year clustering for: {years}")
    
    all_reduced_embeddings = [] # Save for 2D plot later
    
    for year in years:
        year_mask = df["year"] == year
        year_df = df[year_mask]
        
        if len(year_df) < config.MIN_CLUSTER_SIZE:
            print(f"[WARNING] Year {year} has too few docs ({len(year_df)}). Skipping.")
            continue
            
        print(f"\n--- Year {year}: {len(year_df)} documents ---")
        year_embeddings = np.stack(year_df["embedding_temp"].values)
        
        # 1. UMAP Dimensionality Reduction
        n_neighbors = min(config.UMAP_NEIGHBORS, len(year_df) - 1)
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=config.UMAP_COMPONENTS,
            min_dist=config.UMAP_MIN_DIST,
            metric="cosine",
            random_state=42
        )
        reduced = umap_model.fit_transform(year_embeddings)
        all_reduced_embeddings.extend(reduced)
        
        # 2. HDBSCAN Density Clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.MIN_CLUSTER_SIZE,
            min_samples=config.MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        labels = clusterer.fit_predict(reduced)
        
        # 3. Format Labels to Year_ClusterID (e.g., "2023_0", "2023_-1")
        formatted_labels = [f"{int(year)}_{lbl}" if lbl != -1 else f"{int(year)}_-1" for lbl in labels]
        df.loc[year_mask, "cluster"] = formatted_labels
        
        # Log Stats
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = list(labels).count(-1) / len(labels)
        print(f"[INFO] Year {year} -> Identified {n_clusters} clusters. Noise: {noise_ratio:.2%}")

    # Clean up the massive temporary embedding column
    df = df.drop(columns=["embedding_temp"])
    
    return df, np.array(all_reduced_embeddings)

def inspect_clusters(df):
    print("\n[INFO] Cluster Distribution:")
    print(df["cluster"].value_counts().head(10)) # Just show top 10 to save space

def visualize_clusters_2d(df, reduced_embeddings):
    print("\n[INFO] Generating 2D visualization (Global Map)...")
    
    # We need to further reduce the 5D UMAP down to 2D for plotting
    umap_2d = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = umap_2d.fit_transform(reduced_embeddings)

    plt.figure(figsize=(12, 8))
    labels = df["cluster"].values

    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        if str(label).endswith("_-1"): # It's a noise cluster
            plt.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                s=5, color='lightgrey', alpha=0.5, label="Noise" if label == unique_labels[0] else ""
            )
        else:
            plt.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                s=15, label=f"Cluster {label}"
            )

    plt.title("Year-Segmented UMAP Cluster Visualization")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    
    # Hide legend if there are too many clusters
    if len(unique_labels) < 20:
        plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/cluster_visualization.png")
    print("[INFO] Saved plot to outputs/cluster_visualization.png")

def save_results(df):
    df.to_csv(config.CLUSTERED_FILE, index=False)
    print(f"\n[INFO] Results saved to {config.CLUSTERED_FILE}")

# =========================
# MAIN PIPELINE
# =========================
def main():
    df = load_data()
    
    # Run the new Year-Level clustering logic
    df, reduced_embeddings = process_year_clusters(df)
    
    inspect_clusters(df)
    
    # Visualise and save
    if len(reduced_embeddings) > 0:
        visualize_clusters_2d(df, reduced_embeddings)
        
    save_results(df)

if __name__ == "__main__":
    main()