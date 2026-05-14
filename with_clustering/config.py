# =========================
# config.py
# Optimized multilingual topic modelling configuration.
# Pipeline 2 (with_clustering):
#   year -> cluster -> topic modelling -> sentiment analysis
#         -> explanation agent -> dashboard
# =========================

import os

# --- Paths --------------------------------------------------------------
# BASE_DIR is the with_clustering/ directory so the pipeline can be run
# from any working directory without path surprises.
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
VIZ_DIR   = os.path.join(OUT_DIR, "visualizations")

RAW_FILE              = os.path.join(DATA_DIR, "panagbenga2013-2026_cleaned=9013.csv")
PREPROCESSED_FILE     = os.path.join(DATA_DIR, "prep_dataset_v4.csv")
# Output of preprocessing_second.py (stricter v2 rules + blocked-account filter)
PREPROCESSED_FILE_SECOND = os.path.join(DATA_DIR, "prep_dataset_second.csv")
# CSV read by embeddings.py — set via set_preprocessing_variant() from run_pipeline.py
PREPROCESSED_INPUT_FILE = PREPROCESSED_FILE
PREPROCESSING_VARIANT = "default"
# CLUSTERED_FILE is a pickle (preserves embedding + pre_cluster_label columns)
CLUSTERED_FILE        = os.path.join(DATA_DIR, "clustered_dataset.pkl")
CLUSTER_SUMMARY_FILE  = os.path.join(OUT_DIR, "cluster_summary.json")
TOPIC_OUTPUT_FILE     = os.path.join(OUT_DIR, "topic_results.json")
SENTIMENT_OUTPUT_FILE = os.path.join(OUT_DIR, "sentiment_results.json")
FINAL_OUTPUT_FILE     = os.path.join(OUT_DIR, "results.json")
EXPLAINED_OUTPUT_FILE = os.path.join(OUT_DIR, "explained_results.json")

# Fallback source for raw data when DATA_DIR is empty (Pipeline 1's data dir).
FALLBACK_RAW_FILE     = os.path.join(BASE_DIR, "..", "main", "data",
                                     "panagbenga2013-2026_cleaned=9013.csv")


def set_preprocessing_variant(variant: str) -> None:
    """Which preprocessed CSV embeddings and clustering consume.

    ``default`` → ``preprocessing.py`` output (``prep_dataset_v4.csv``).
    ``second`` → ``preprocessing_second.py`` output (``prep_dataset_second.csv``).
    """
    global PREPROCESSED_INPUT_FILE, PREPROCESSING_VARIANT
    PREPROCESSING_VARIANT = variant
    if variant == "second":
        PREPROCESSED_INPUT_FILE = PREPROCESSED_FILE_SECOND
    else:
        PREPROCESSED_INPUT_FILE = PREPROCESSED_FILE


# --- Year range ---------------------------------------------------------
YEAR_START = 2022
YEAR_END   = 2026

# =========================================================
# EMBEDDINGS
# =========================================================

# Best multilingual option for code-switching (EN + TL + ILO)
EMBEDDING_MODEL    = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_BATCH    = 32
EMBEDDING_NORMALIZE = True

# =========================================================
# UMAP  (used both for pre-clustering and inside BERTopic)
# =========================================================

UMAP_NEIGHBORS  = 15
UMAP_COMPONENTS = 5
UMAP_MIN_DIST   = 0.0

# Lightweight 2-D projection used only for cluster visualizations.
UMAP_VIZ_COMPONENTS = 2
UMAP_VIZ_MIN_DIST   = 0.1

# =========================================================
# PRE-CLUSTERING  (year -> cluster layer, runs before BERTopic)
# =========================================================

# HDBSCAN params for the coarse pre-cluster pass on the year embedding.
# Intentionally looser than BERTopic's internal HDBSCAN so each cluster
# is large enough for a meaningful second-pass BERTopic model.
PRE_CLUSTER_MIN_CLUSTER_SIZE = 20
PRE_CLUSTER_MIN_SAMPLES      = 5

# Noise docs (cluster_id == -1) are collected into a single "noise"
# pseudo-cluster and passed through topic modelling as one unit.
INCLUDE_NOISE_CLUSTER        = True

# =========================================================
# TOPIC MODELLING  (runs inside each cluster)
# =========================================================

TOPIC_MODEL_TYPE        = "bertopic"
BERTOPIC_MIN_TOPIC_SIZE = 10
BERTOPIC_NR_TOPICS      = 10
TOP_N_WORDS             = 10
MIN_DOCS_PER_CLUSTER    = 20

# =========================================================
# VECTORIZATION
# =========================================================

NGRAM_RANGE = (1, 2)
MIN_DF      = 2
MAX_DF      = 0.85

# =========================================================
# SENTIMENT ANALYSIS  (per topic inside each cluster)
# =========================================================

SENTIMENT_MODEL         = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SENTIMENT_BATCH_SIZE    = 32
SENTIMENT_USE_PROCESSED = True   # True = 'processed' column; False = 'text'
# Threshold above which a prediction is considered high-confidence.
SENTIMENT_HIGH_CONFIDENCE = 0.75

# =========================================================
# HDBSCAN  (BERTopic's internal pass — kept separate from PRE_CLUSTER)
# =========================================================

MIN_CLUSTER_SIZE = 10
MIN_SAMPLES      = 5

# =========================================================
# VISUALIZATIONS
# =========================================================

GENERATE_VISUALIZATIONS = True
VIZ_DPI = 130

# =========================================================
# DASHBOARD
# =========================================================

DASHBOARD_PORT = 5050   # different from Pipeline 1 to avoid port clashes

# =========================================================
# LOGGING
# =========================================================

LOG_LEVEL = "INFO"
