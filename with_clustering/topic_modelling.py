# =========================
# topic_modelling.py
# Multilingual BERTopic pipeline — year → cluster → topics.
#
# Pipeline 2 (with_clustering):
#   1. Read pickled dataframe written by embeddings.py
#      (must contain `embedding` and `pre_cluster_label` columns).
#   2. For each year, iterate over its persisted clusters and run
#      a BERTopic model independently inside each cluster.
#   3. Compute c_v coherence per cluster, write per-year/cluster summaries.
#
# If `pre_cluster_label` is missing (legacy pickle), fall back to running
# a coarse pre-cluster pass in memory so the pipeline still completes.
#
# Output: outputs/topic_results.json
#   {
#     "generated_at": "...",
#     "model_type":   "bertopic",
#     "parameters":   { ... },
#     "years": {
#       "<year>": {
#         "n_docs":      int,
#         "n_clusters":  int,
#         "noise_ratio": float,
#         "avg_coherence": float,
#         "clusters": [
#           {
#             "cluster_id": int,          # -1 == noise pseudo-cluster
#             "n_docs":     int,
#             "n_topics":   int,
#             "coherence":  float,        # c_v across this cluster's topics
#             "topics": [
#               {
#                 "topic_id":        int,
#                 "label":           str,
#                 "count":           int,
#                 "top_words":       [str, ...],
#                 "top_word_scores": [float, ...]
#               }, ...
#             ],
#             "error": str               # only on failure
#           }
#         ],
#         "error": str                   # only on full-year failure
#       }
#     }
#   }
# =========================

import os
import warnings

import numpy as np
import pandas as pd

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import config
from utils import save_json, timestamp, get_logger, log_banner, ensure_dir

logger = get_logger("topic_modelling")

warnings.filterwarnings("ignore", category=UserWarning, module="umap")


# =========================================================
# COHERENCE  (c_v via gensim CoherenceModel)
# =========================================================

def compute_coherence(docs: list, topic_words: list) -> float:
    """Compute corpus-level c_v coherence across all topics. 0.0 on failure."""
    if not topic_words:
        return 0.0
    # Filter out topics whose top words are absent from the docs to avoid
    # gensim returning NaN.
    try:
        tokenized  = [doc.split() for doc in docs]
        dictionary = Dictionary(tokenized)
        token_set  = set(dictionary.token2id.keys())
        kept_topics = []
        for words in topic_words:
            present = [w for w in words if w in token_set]
            if len(present) >= 2:
                kept_topics.append(present)
        if not kept_topics:
            return 0.0
        model = CoherenceModel(
            topics=kept_topics,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,  # avoid per-process import overhead on Windows
        )
        score = float(model.get_coherence())
        # gensim sometimes returns NaN with very small corpora — treat as 0.
        if score != score:  # NaN check
            return 0.0
        return score
    except Exception as e:
        logger.warning(f"Coherence computation failed: {e}")
        return 0.0


def generate_topic_label(top_words: list) -> str:
    return " / ".join(top_words[:3]).title()


# =========================================================
# FALLBACK PRE-CLUSTERING  (only used if pre_cluster_label is missing)
# =========================================================

def pre_cluster_year(embeddings: np.ndarray, n_docs: int) -> np.ndarray:
    n_neighbors  = max(2, min(config.PRE_CLUSTER_MIN_CLUSTER_SIZE, n_docs - 1))
    n_components = max(2, min(config.UMAP_COMPONENTS, n_docs - 2))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.1,
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
    return hdbscan_model.fit_predict(reduced).astype(int)


# =========================================================
# PER-CLUSTER BERTOPIC
# =========================================================

def build_vectorizer(n_docs: int) -> CountVectorizer:
    """Build a CountVectorizer with parameters safe for the cluster size.

    BERTopic re-runs this vectorizer on per-topic doc subsets while
    computing the c-TF-IDF representation; if any topic ends up with a
    single doc, a fractional max_df rounds below min_df and sklearn
    raises "max_df corresponds to < documents than min_df". To avoid
    that we keep min_df=1 and a high absolute max_df.
    """
    return CountVectorizer(
        ngram_range=config.NGRAM_RANGE,
        stop_words=list(),  # already filtered in preprocessing
        min_df=1,
        max_df=1.0,
    )


def run_bertopic_on_cluster(
    docs: list,
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    vectorizer_model: CountVectorizer,
    mts: int,
    nr: int,
) -> list:
    """Fit BERTopic on a single cluster's docs and return topic dicts."""
    n = len(docs)
    if n < config.MIN_DOCS_PER_CLUSTER:
        logger.info(f"    cluster too small ({n} docs) — skipping BERTopic")
        return []

    # Scale BERTopic params down for small clusters so HDBSCAN can still
    # find topics (otherwise everything becomes noise).
    local_mts        = max(3, min(mts, max(3, n // 8)))
    local_min_cluster = max(3, min(config.MIN_CLUSTER_SIZE, max(3, n // 10)))
    local_min_samples = max(1, min(config.MIN_SAMPLES, local_min_cluster - 1))

    umap_model = UMAP(
        n_neighbors=max(2, min(config.UMAP_NEIGHBORS, n - 1)),
        n_components=max(2, min(config.UMAP_COMPONENTS, n - 2)),
        min_dist=config.UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=local_min_cluster,
        min_samples=local_min_samples,
        metric="euclidean",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=KeyBERTInspired(),
        min_topic_size=local_mts,
        nr_topics=nr,
        calculate_probabilities=False,
        verbose=False,
    )

    topic_model.fit_transform(docs, embeddings)
    topic_info = topic_model.get_topic_info()

    topic_results = []
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue  # BERTopic internal noise — skip
        words      = topic_model.get_topic(tid) or []
        top_words  = [w for w, _ in words[: config.TOP_N_WORDS]]
        top_scores = [round(float(s), 6) for _, s in words[: config.TOP_N_WORDS]]
        topic_results.append({
            "topic_id":        tid,
            "label":           generate_topic_label(top_words),
            "count":           int(row["Count"]),
            "top_words":       top_words,
            "top_word_scores": top_scores,
        })

    return topic_results


# =========================================================
# COHERENCE VISUALIZATION (per-year bar chart)
# =========================================================

def render_coherence_chart(year_results: dict, out_path: str) -> None:
    """Save a per-year, per-cluster coherence bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for year_str, ydata in year_results.items():
        for c in ydata.get("clusters", []):
            rows.append({
                "year":       int(year_str),
                "cluster_id": c["cluster_id"],
                "coherence":  c.get("coherence", 0.0),
                "n_topics":   c.get("n_topics", 0),
            })
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    years = sorted(df["year"].unique())
    width = 0.8 / max(1, len(years))

    cmap = plt.colormaps.get_cmap("viridis")
    for i, yr in enumerate(years):
        sub = df[df["year"] == yr].sort_values("cluster_id")
        positions = np.arange(len(sub)) + i * width
        ax.bar(
            positions, sub["coherence"], width=width,
            label=str(yr), color=cmap(i / max(1, len(years) - 1)),
        )

    ax.set_xlabel("Cluster (grouped by year)")
    ax.set_ylabel("c_v coherence")
    ax.set_title("BERTopic c_v coherence per cluster")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(title="Year", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.VIZ_DPI)
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main(min_topic_size: int = None, nr_topics: int = None) -> dict:
    log_banner(logger, "Topic Modelling — year → cluster → BERTopic")

    mts = min_topic_size if min_topic_size is not None else config.BERTOPIC_MIN_TOPIC_SIZE
    nr  = nr_topics      if nr_topics      is not None else config.BERTOPIC_NR_TOPICS

    logger.info(f"Loading embedded dataset → {config.CLUSTERED_FILE}")
    df = pd.read_pickle(config.CLUSTERED_FILE)
    df = df.dropna(subset=["processed"]).copy()
    df["year"] = df["year"].astype(int)

    use_persisted_clusters = "pre_cluster_label" in df.columns
    if use_persisted_clusters:
        logger.info("Using persisted `pre_cluster_label` from embeddings step.")
    else:
        logger.warning(
            "`pre_cluster_label` missing — falling back to in-memory clustering."
        )

    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    year_results: dict[str, dict] = {}

    for year in sorted(df["year"].unique()):
        year_df    = df[df["year"] == year].reset_index(drop=True)
        docs       = year_df["processed"].tolist()
        embeddings = np.array(year_df["embedding"].tolist())
        n_docs     = len(docs)

        logger.info(f"\n══ Year {year}: {n_docs} docs ══")

        if n_docs < config.PRE_CLUSTER_MIN_CLUSTER_SIZE * 2:
            logger.info("  too few docs for clustering — skipping year")
            year_results[str(year)] = {
                "n_docs":        n_docs,
                "n_clusters":    0,
                "noise_ratio":   1.0,
                "avg_coherence": 0.0,
                "clusters":      [],
                "error":         f"Only {n_docs} docs — below clustering minimum",
            }
            continue

        # ── Resolve cluster labels ────────────────────────────────────────
        if use_persisted_clusters:
            cluster_labels = year_df["pre_cluster_label"].astype(int).values
        else:
            try:
                cluster_labels = pre_cluster_year(embeddings, n_docs)
            except Exception as e:
                logger.error(f"  Pre-clustering failed for {year}: {e}")
                year_results[str(year)] = {
                    "n_docs":        n_docs,
                    "n_clusters":    0,
                    "noise_ratio":   1.0,
                    "avg_coherence": 0.0,
                    "clusters":      [],
                    "error":         str(e),
                }
                continue

        unique_cluster_ids = sorted(set(cluster_labels.tolist()))
        noise_count = int((cluster_labels == -1).sum())
        noise_ratio = round(noise_count / n_docs, 4) if n_docs else 0.0
        real_cluster_ids = [c for c in unique_cluster_ids if c != -1]

        logger.info(
            f"  Pre-clustering → {len(real_cluster_ids)} clusters, "
            f"noise={noise_ratio:.2%}"
        )

        # ── Run BERTopic per cluster ──────────────────────────────────────
        clusters_out = []
        cluster_targets = list(real_cluster_ids)
        if config.INCLUDE_NOISE_CLUSTER and noise_count >= config.MIN_DOCS_PER_CLUSTER:
            cluster_targets.append(-1)

        for cid in cluster_targets:
            mask         = cluster_labels == cid
            c_docs       = [d for d, m in zip(docs, mask) if m]
            c_embeddings = embeddings[mask]
            label_tag    = "Noise bucket" if cid == -1 else f"Cluster {cid}"

            logger.info(f"  ── {label_tag}: {len(c_docs)} docs ──")

            vectorizer = build_vectorizer(len(c_docs))

            try:
                topics = run_bertopic_on_cluster(
                    docs=c_docs,
                    embeddings=c_embeddings,
                    embedding_model=embedding_model,
                    vectorizer_model=vectorizer,
                    mts=mts,
                    nr=nr,
                )
                coherence_words = [t["top_words"] for t in topics]
                coherence = compute_coherence(c_docs, coherence_words)

                logger.info(
                    f"    topics={len(topics)}  coherence={coherence:.4f}"
                )
                clusters_out.append({
                    "cluster_id": int(cid),
                    "n_docs":     len(c_docs),
                    "n_topics":   len(topics),
                    "coherence":  round(coherence, 4),
                    "topics":     topics,
                })
            except Exception as e:
                logger.error(f"    BERTopic failed for cluster {cid}: {e}")
                clusters_out.append({
                    "cluster_id": int(cid),
                    "n_docs":     len(c_docs),
                    "n_topics":   0,
                    "coherence":  0.0,
                    "topics":     [],
                    "error":      str(e),
                })

        # ── Year-level aggregates ─────────────────────────────────────────
        coh_values = [c["coherence"] for c in clusters_out if c.get("n_topics", 0) > 0]
        avg_coherence = round(float(np.mean(coh_values)), 4) if coh_values else 0.0

        year_results[str(year)] = {
            "n_docs":        n_docs,
            "n_clusters":    len(real_cluster_ids),
            "noise_ratio":   noise_ratio,
            "avg_coherence": avg_coherence,
            "clusters":      clusters_out,
        }

    output = {
        "generated_at": timestamp(),
        "model_type":   config.TOPIC_MODEL_TYPE,
        "parameters": {
            "pre_cluster_min_size":     config.PRE_CLUSTER_MIN_CLUSTER_SIZE,
            "pre_cluster_min_samples":  config.PRE_CLUSTER_MIN_SAMPLES,
            "min_topic_size":           mts,
            "nr_topics":                nr,
            "top_n_words":              config.TOP_N_WORDS,
            "ngram_range":              list(config.NGRAM_RANGE),
            "embedding_model":          config.EMBEDDING_MODEL,
            "used_persisted_clusters":  use_persisted_clusters,
        },
        "years": year_results,
    }

    save_json(output, config.TOPIC_OUTPUT_FILE)
    logger.info(f"\nTopic results saved → {config.TOPIC_OUTPUT_FILE}")

    # ── Coherence visualization ───────────────────────────────────────────
    if config.GENERATE_VISUALIZATIONS:
        ensure_dir(config.VIZ_DIR)
        chart_path = os.path.join(config.VIZ_DIR, "coherence_overview.png")
        try:
            render_coherence_chart(year_results, chart_path)
            logger.info(f"Coherence chart saved → {chart_path}")
        except Exception as e:
            logger.warning(f"Coherence chart failed: {e}")

    return output


if __name__ == "__main__":
    main()
