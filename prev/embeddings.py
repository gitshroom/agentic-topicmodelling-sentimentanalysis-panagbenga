# =========================
# 1. IMPORTS
# =========================
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 2. CONFIGURATION
# =========================
INPUT_FILE = "data/prep_dataset_v3.csv"
OUTPUT_FILE = "data/clustered_dataset.csv"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# UMAP parameters
UMAP_NEIGHBORS = 15
UMAP_COMPONENTS = 5
UMAP_MIN_DIST = 0.0

# HDBSCAN parameters
MIN_CLUSTER_SIZE = 13
MIN_SAMPLES = 9


# =========================
# 3. LOAD DATA
# =========================
def load_data():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=["processed"])
    print(f"[INFO] Dataset loaded: {df.shape}")

    return df

# =========================
# 4. EMBEDDINGS
# =========================
def generate_embeddings(texts):
    print("[INFO] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("[INFO] Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    print("[INFO] Normalizing embeddings...")
    embeddings = normalize(embeddings)

    return embeddings

# =========================
# 5. UMAP REDUCTION
# =========================
def reduce_dimensions(embeddings):
    print("[INFO] Running UMAP...")

    umap_model = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=UMAP_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42
    )

    reduced = umap_model.fit_transform(embeddings)

    print(f"[INFO] Reduced shape: {reduced.shape}")
    return reduced

# =========================
# 6. CLUSTERING
# =========================
def cluster_data(reduced_embeddings):
    print("[INFO] Running HDBSCAN clustering...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(reduced_embeddings)

    print("[INFO] Clustering complete.")
    return labels

# =========================
# 7. INSPECT CLUSTERS
# =========================
def inspect_clusters(df):
    print("\n[INFO] Cluster Distribution:")
    print(df["cluster"].value_counts())

    print("\n[INFO] Sample Clusters:\n")

    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue

        print("=" * 40)
        print(f"CLUSTER {cluster_id}")
        print("=" * 40)

        sample = df[df["cluster"] == cluster_id].head(5)

        for _, row in sample.iterrows():
            print("TEXT:", row["text"])
            print("PROCESSED:", row["processed"])
            print("-" * 20)

# =========================
# 8. SAVE OUTPUT
# =========================
def save_results(df):
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[INFO] Results saved to {OUTPUT_FILE}")

# visualizations

def visualize_clusters_2d(reduced_embeddings, labels):
    print("[INFO] Generating 2D visualization...")

    # Reduce to 2D for plotting
    import umap
    umap_2d = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = umap_2d.fit_transform(reduced_embeddings)

    plt.figure(figsize=(10, 7))

    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label

        if label == -1:
            # noise
            plt.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                s=10,
                label="Noise (-1)"
            )
        else:
            plt.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                s=10,
                label=f"Cluster {label}"
            )

    plt.title("UMAP 2D Cluster Visualization")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()

    plt.savefig("outputs/cluster_visualization.png")
    plt.show()

    print("[INFO] Saved as data/cluster_visualization.png")

def main():
    df = load_data()

    embeddings = generate_embeddings(df["processed"].tolist())

    reduced_embeddings = reduce_dimensions(embeddings)

    labels = cluster_data(reduced_embeddings)
    df["cluster"] = labels

    inspect_clusters(df)

    # 🔥 ADD THESE
    visualize_clusters_2d(reduced_embeddings, labels)
    plot_cluster_distribution(labels)

    save_results(df)
# =========================
# 9. MAIN PIPELINE
# =========================
def main():
    df = load_data()

    embeddings = generate_embeddings(df["processed"].tolist())

    reduced_embeddings = reduce_dimensions(embeddings)

    df["cluster"] = cluster_data(reduced_embeddings)

    inspect_clusters(df)

    save_results(df)

# =========================
# 10. RUN SCRIPT
# =========================
if __name__ == "__main__":
    main()