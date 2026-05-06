# =========================
# embeddings.py
# Generates sentence embeddings, reduces dimensions with UMAP, and clusters
# with HDBSCAN — independently for each year slice.
#
# Output adds columns: cluster, year  (year was already present from preprocessing)
# Run standalone:  python embeddings.py
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import umap
import hdbscan

import config

# =========================
# CONFIGURATION
# =========================
INPUT_FILE  = config.PREPROCESSED_FILE
OUTPUT_FILE = config.CLUSTERED_FILE

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

UMAP_NEIGHBORS  = 15
UMAP_COMPONENTS = 5
UMAP_MIN_DIST   = 0.0

MIN_CLUSTER_SIZE = 13
MIN_SAMPLES      = 9


# =========================
# LOAD DATA
# =========================
def load_data() -> pd.DataFrame:
    print("[embeddings] Loading dataset…")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["processed"])
    if "year" not in df.columns:
        raise ValueError("'year' column missing — run preprocessing.py first.")
    df["year"] = df["year"].astype(int)
    print(f"[embeddings] Loaded {df.shape[0]} rows across years {df['year'].min()}–{df['year'].max()}")
    return df


# =========================
# EMBEDDINGS
# =========================
def generate_embeddings(texts: list) -> np.ndarray:
    print("[embeddings] Loading embedding model…")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("[embeddings] Generating embeddings…")
    emb = model.encode(texts, show_progress_bar=True)
    print("[embeddings] Normalising embeddings…")
    return normalize(emb)


# =========================
# UMAP REDUCTION
# =========================
def reduce_dimensions(embeddings: np.ndarray, n_components: int = UMAP_COMPONENTS) -> np.ndarray:
    print(f"[embeddings] Running UMAP (n_components={n_components})…")
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=n_components,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"[embeddings] Reduced shape: {reduced.shape}")
    return reduced


# =========================
# CLUSTERING
# =========================
def cluster_data(reduced: np.ndarray) -> np.ndarray:
    print("[embeddings] Running HDBSCAN…")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)
    print("[embeddings] Clustering complete.")
    return labels


# =========================
# PER-YEAR CLUSTERING
# =========================
def cluster_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Embed and cluster each year slice independently so that cluster IDs are
    local to a year.  Adds a 'cluster' column (int, -1 = noise).
    Years with fewer than MIN_DOCS_PER_YEAR docs are assigned cluster -1.
    """
    all_frames = []
    years = sorted(df["year"].unique())

    # Load the model once
    print("[embeddings] Loading embedding model for per-year clustering…")
    model = SentenceTransformer(EMBEDDING_MODEL)

    for year in years:
        year_df = df[df["year"] == year].copy()
        n = len(year_df)
        print(f"\n[embeddings] ── Year {year}: {n} docs ──")

        if n < config.MIN_DOCS_PER_YEAR:
            print(f"[embeddings]   Too few docs ({n}), assigning cluster=-1.")
            year_df["cluster"] = -1
            all_frames.append(year_df)
            continue

        texts = year_df["processed"].tolist()

        emb = model.encode(texts, show_progress_bar=False)
        emb = normalize(emb)

        # UMAP needs n_neighbors < n_samples
        n_neighbors = min(UMAP_NEIGHBORS, n - 1)
        # n_components must be < n_samples as well
        n_components = min(UMAP_COMPONENTS, n - 1)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=42,
        )
        try:
            reduced = reducer.fit_transform(emb)
        except Exception as e:
            print(f"[embeddings]   UMAP failed for {year}: {e} — assigning cluster=-1.")
            year_df["cluster"] = -1
            all_frames.append(year_df)
            continue

        # HDBSCAN min_cluster_size must be <= n
        min_cs = min(MIN_CLUSTER_SIZE, max(2, n // 4))
        min_s  = min(MIN_SAMPLES, min_cs)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=min_s,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced)
        year_df["cluster"] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise      = int((labels == -1).sum())
        print(f"[embeddings]   → {n_clusters} clusters, {noise} noise docs")

        all_frames.append(year_df)

    return pd.concat(all_frames, ignore_index=True)


# =========================
# INSPECT CLUSTERS
# =========================
def inspect_clusters(df: pd.DataFrame) -> None:
    print("\n[embeddings] Cluster distribution per year:")
    for year in sorted(df["year"].unique()):
        ydf = df[df["year"] == year]
        dist = ydf["cluster"].value_counts().sort_index()
        print(f"  {year}: {dict(dist)}")


# =========================
# VISUALISATION
# =========================
def visualize_clusters_2d(df: pd.DataFrame) -> None:
    """One scatter plot per year, each saved to outputs/."""
    print("[embeddings] Generating 2D cluster visualisations…")
    model = SentenceTransformer(EMBEDDING_MODEL)

    for year in sorted(df["year"].unique()):
        ydf   = df[df["year"] == year]
        texts = ydf["processed"].tolist()
        if len(texts) < 4:
            continue

        emb = normalize(model.encode(texts, show_progress_bar=False))
        umap_2d = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(texts) - 1),
            random_state=42,
        )
        try:
            coords = umap_2d.fit_transform(emb)
        except Exception:
            continue

        labels = ydf["cluster"].values
        plt.figure(figsize=(10, 7))
        for lbl in np.unique(labels):
            mask  = labels == lbl
            label = "Noise" if lbl == -1 else f"Cluster {lbl}"
            plt.scatter(coords[mask, 0], coords[mask, 1], s=10, label=label)

        plt.title(f"UMAP 2D — {year}")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend(markerscale=2, fontsize=7, loc="best")
        plt.tight_layout()
        path = f"outputs/cluster_viz_{year}.png"
        plt.savefig(path)
        plt.close()
        print(f"[embeddings]   Saved {path}")


# =========================
# SAVE OUTPUT
# =========================
def save_results(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[embeddings] Results saved → {OUTPUT_FILE}")


# =========================
# MAIN
# =========================
def main() -> pd.DataFrame:
    df = load_data()
    df = cluster_per_year(df)
    inspect_clusters(df)
    visualize_clusters_2d(df)
    save_results(df)
    return df


if __name__ == "__main__":
    main()
