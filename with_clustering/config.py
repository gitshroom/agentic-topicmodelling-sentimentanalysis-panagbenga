# =========================
# config.py
# Optimized multilingual topic modelling configuration
# Pipeline: year → cluster → topic modelling → sentiment per topic
# =========================

# --- File paths ---
RAW_FILE              = "data/panagbenga2013-2026_cleaned=9013.csv"
PREPROCESSED_FILE     = "data/prep_dataset_v4.csv"
CLUSTERED_FILE        = "data/clustered_dataset.csv"
TOPIC_OUTPUT_FILE     = "outputs/topic_results.json"
SENTIMENT_OUTPUT_FILE = "outputs/sentiment_results.json"
FINAL_OUTPUT_FILE     = "outputs/results.json"

# --- Year range ---
YEAR_START = 2022
YEAR_END   = 2026

# =========================================================
# EMBEDDINGS
# =========================================================

# BEST multilingual option for code-switching
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Alternative:
# EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# =========================================================
# UMAP
# =========================================================

UMAP_NEIGHBORS  = 15
UMAP_COMPONENTS = 5
UMAP_MIN_DIST   = 0.0

# =========================================================
# PRE-CLUSTERING  (year → cluster layer, runs before BERTopic)
# =========================================================

# HDBSCAN params for the coarse pre-cluster pass on the full year embedding.
# These are intentionally looser than BERTopic's internal HDBSCAN so that
# each cluster is large enough for a second-pass BERTopic model.
PRE_CLUSTER_MIN_CLUSTER_SIZE = 20   # minimum docs to form a cluster
PRE_CLUSTER_MIN_SAMPLES      = 5    # controls outlier sensitivity
# Noise docs (cluster_id == -1) are collected into a single "noise" cluster
# and passed through topic modelling as one unit.
INCLUDE_NOISE_CLUSTER        = True

# =========================================================
# TOPIC MODELLING  (runs inside each cluster)
# =========================================================

TOPIC_MODEL_TYPE        = "bertopic"

# More stable for noisy multilingual social data
BERTOPIC_MIN_TOPIC_SIZE = 10   # lower than pre-cluster size — clusters are smaller

# Avoid unstable auto-merging
BERTOPIC_NR_TOPICS      = 10   # fewer topics per cluster; aggregate view emerges naturally

TOP_N_WORDS             = 10

MIN_DOCS_PER_CLUSTER    = 20   # skip BERTopic for clusters smaller than this

# =========================================================
# VECTORIZATION
# =========================================================

NGRAM_RANGE = (1, 2)
MIN_DF      = 2        # lower than before — cluster corpora are smaller
MAX_DF      = 0.85

# =========================================================
# SENTIMENT ANALYSIS  (per topic inside each cluster)
# =========================================================

SENTIMENT_MODEL         = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SENTIMENT_BATCH_SIZE    = 32
SENTIMENT_USE_PROCESSED = True  # True = 'processed' col; False = 'text'

# =========================================================
# HDBSCAN  (BERTopic's internal pass — kept separate from PRE_CLUSTER)
# =========================================================

MIN_CLUSTER_SIZE = 10
MIN_SAMPLES      = 5

# =========================================================
# LOGGING
# =========================================================

LOG_LEVEL = "INFO"
