# =========================
# embeddings.py
# Step 3 of Pipeline 2 (with_clustering):
#   Embedding -> UMAP -> HDBSCAN clustering -> persisted labels + visuals.
#
# Output:
#   - data/clustered_dataset.pkl  (pickle preserves embedding list dtype)
#       columns added: `embedding`, `pre_cluster_label`, `umap_x`, `umap_y`
#   - outputs/cluster_summary.json
#       per-year cluster counts, noise ratio, doc totals, params
#   - outputs/visualizations/cluster_viz_<year>.png
#       2-D UMAP scatter coloured by HDBSCAN cluster label
# =========================

import os
import warnings

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

import config
from utils import get_logger, log_banner, save_json, timestamp, ensure_dir

logger = get_logger("embeddings")

# Suppress noisy warnings from UMAP / numba on small year subsets.
warnings.filterwarnings("ignore", category=UserWarning, module="umap")


# =========================================================
# CORE STEPS
# =========================================================

def encode_texts(model: SentenceTransformer, texts: list) -> np.ndarray:
    """Encode docs to L2-normalised multilingual embeddings."""
    logger.info(f"Encoding {len(texts)} docs with {config.EMBEDDING_MODEL}…")
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH,
        show_progress_bar=True,
        normalize_embeddings=config.EMBEDDING_NORMALIZE,
    )
    return np.asarray(embeddings, dtype=np.float32)


def cluster_year_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce a year's embeddings with UMAP (5-D for clustering) and cluster
    with HDBSCAN. Returns (cluster_labels, umap_5d).

    HDBSCAN labels: -1 == noise, >=0 == cluster id.
    """
    n_docs = embeddings.shape[0]
    n_neighbors  = max(2, min(config.UMAP_NEIGHBORS, n_docs - 1))
    n_components = max(2, min(config.UMAP_COMPONENTS, n_docs - 2))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=config.UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    reduced = umap_model.fit_transform(embeddings)

    hdbscan_model = HDBSCAN(
        min_cluster_size=config.PRE_CLUSTER_MIN_CLUSTER_SIZE,
        min_samples=config.PRE_CLUSTER_MIN_SAMPLES,
        metric="euclidean",
        prediction_data=False,
    )
    labels = hdbscan_model.fit_predict(reduced)
    return labels.astype(int), reduced


def project_2d(embeddings: np.ndarray) -> np.ndarray:
    """Lightweight 2-D UMAP projection for visualization only."""
    n_docs = embeddings.shape[0]
    n_neighbors = max(2, min(config.UMAP_NEIGHBORS, n_docs - 1))
    umap_2d = UMAP(
        n_neighbors=n_neighbors,
        n_components=config.UMAP_VIZ_COMPONENTS,
        min_dist=config.UMAP_VIZ_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    return umap_2d.fit_transform(embeddings)


def render_cluster_viz(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    year: int,
    out_path: str,
) -> None:
    """Save a 2-D scatter plot of clusters for one year."""
    import matplotlib
    matplotlib.use("Agg")  # headless backend so this works on servers
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.5))
    unique_labels = sorted(np.unique(labels).tolist())

    cmap = plt.colormaps.get_cmap("tab20")
    real_clusters = [lab for lab in unique_labels if lab != -1]

    for idx, lab in enumerate(unique_labels):
        mask = labels == lab
        if lab == -1:
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                s=8, alpha=0.35, c="#aaaaaa", label=f"Noise ({mask.sum()})",
            )
        else:
            colour = cmap(real_clusters.index(lab) % cmap.N)
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                s=12, alpha=0.85, color=colour,
                label=f"Cluster {lab} ({mask.sum()})",
            )

    ax.set_title(f"Year {year} — UMAP + HDBSCAN clusters", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.VIZ_DPI)
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main() -> pd.DataFrame:
    log_banner(logger, "Embeddings + per-year UMAP/HDBSCAN clustering")

    logger.info(f"Loading preprocessed CSV → {config.PREPROCESSED_FILE}")
    df = pd.read_csv(config.PREPROCESSED_FILE)
    df = df.dropna(subset=["processed"]).copy()

    if "year" not in df.columns:
        raise ValueError("'year' column missing — run preprocessing.py first.")

    df["year"] = df["year"].astype(int)
    df = df.reset_index(drop=True)

    # ---------- Embeddings (single model, all docs) ------------------------
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = encode_texts(model, df["processed"].tolist())
    df["embedding"] = list(embeddings)

    # ---------- Per-year clustering + 2-D projection -----------------------
    ensure_dir(config.VIZ_DIR)

    cluster_labels_full = np.full(len(df), -1, dtype=int)
    umap_x_full         = np.full(len(df), np.nan, dtype=np.float32)
    umap_y_full         = np.full(len(df), np.nan, dtype=np.float32)

    per_year_summary: dict[str, dict] = {}

    for year in sorted(df["year"].unique()):
        year_mask  = (df["year"] == year).values
        year_idx   = np.where(year_mask)[0]
        year_embs  = embeddings[year_idx]
        n_docs     = year_embs.shape[0]

        logger.info(f"\n── Year {year}: {n_docs} docs ──")

        if n_docs < config.PRE_CLUSTER_MIN_CLUSTER_SIZE * 2:
            logger.info(
                f"  too few docs ({n_docs}) for clustering — labelling as noise"
            )
            cluster_labels_full[year_idx] = -1
            per_year_summary[str(year)] = {
                "n_docs":      n_docs,
                "n_clusters":  0,
                "noise_docs":  n_docs,
                "noise_ratio": 1.0,
                "skipped":     True,
                "reason":      "too few docs for HDBSCAN",
            }
            continue

        try:
            labels, _ = cluster_year_embeddings(year_embs)
            coords_2d = project_2d(year_embs)
        except Exception as e:
            logger.error(f"  Clustering failed for {year}: {e}")
            cluster_labels_full[year_idx] = -1
            per_year_summary[str(year)] = {
                "n_docs":      n_docs,
                "n_clusters":  0,
                "noise_docs":  n_docs,
                "noise_ratio": 1.0,
                "error":       str(e),
            }
            continue

        cluster_labels_full[year_idx] = labels
        umap_x_full[year_idx]         = coords_2d[:, 0]
        umap_y_full[year_idx]         = coords_2d[:, 1]

        unique = sorted(set(labels.tolist()))
        real_clusters = [c for c in unique if c != -1]
        noise_count = int((labels == -1).sum())
        noise_ratio = round(noise_count / n_docs, 4) if n_docs else 0.0

        logger.info(
            f"  Clusters: {len(real_clusters)}  |  noise: {noise_count} "
            f"({noise_ratio:.2%})"
        )

        # Per-cluster doc counts for documentation
        size_by_cluster = {
            int(cid): int((labels == cid).sum()) for cid in real_clusters
        }

        per_year_summary[str(year)] = {
            "n_docs":          n_docs,
            "n_clusters":      len(real_clusters),
            "noise_docs":      noise_count,
            "noise_ratio":     noise_ratio,
            "cluster_sizes":   size_by_cluster,
        }

        # Visualization
        if config.GENERATE_VISUALIZATIONS:
            viz_path = os.path.join(config.VIZ_DIR, f"cluster_viz_{year}.png")
            try:
                render_cluster_viz(coords_2d, labels, int(year), viz_path)
                per_year_summary[str(year)]["visualization"] = os.path.relpath(
                    viz_path, config.BASE_DIR
                )
                logger.info(f"  Visualization saved → {viz_path}")
            except Exception as e:
                logger.warning(f"  Visualization failed for {year}: {e}")

    # ---------- Persist enriched dataframe ---------------------------------
    df["pre_cluster_label"] = cluster_labels_full
    df["umap_x"]            = umap_x_full
    df["umap_y"]            = umap_y_full

    ensure_dir(config.DATA_DIR)
    df.to_pickle(config.CLUSTERED_FILE)
    logger.info(
        f"\nSaved {len(df)} rows with embeddings + cluster labels → "
        f"{config.CLUSTERED_FILE}"
    )

    # ---------- Cluster summary --------------------------------------------
    summary = {
        "generated_at": timestamp(),
        "embedding_model": config.EMBEDDING_MODEL,
        "params": {
            "umap_neighbors":              config.UMAP_NEIGHBORS,
            "umap_components":             config.UMAP_COMPONENTS,
            "umap_min_dist":               config.UMAP_MIN_DIST,
            "pre_cluster_min_cluster_size": config.PRE_CLUSTER_MIN_CLUSTER_SIZE,
            "pre_cluster_min_samples":     config.PRE_CLUSTER_MIN_SAMPLES,
        },
        "totals": {
            "documents": int(len(df)),
            "years":     int(df["year"].nunique()),
        },
        "years": per_year_summary,
    }
    save_json(summary, config.CLUSTER_SUMMARY_FILE)
    logger.info(f"Cluster summary saved → {config.CLUSTER_SUMMARY_FILE}")

    return df


if __name__ == "__main__":
    main()
