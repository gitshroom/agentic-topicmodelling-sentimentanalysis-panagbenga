# =========================
# config.py
# Optimized multilingual topic modelling configuration
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
# TOPIC MODELLING
# =========================================================

TOPIC_MODEL_TYPE        = "bertopic"

# More stable for noisy multilingual social data
BERTOPIC_MIN_TOPIC_SIZE = 15

# Avoid unstable auto-merging
BERTOPIC_NR_TOPICS      = 20

TOP_N_WORDS             = 10

MIN_DOCS_PER_YEAR       = 30

# =========================================================
# VECTORIZATION
# =========================================================

NGRAM_RANGE = (1, 2)
MIN_DF      = 3
MAX_DF      = 0.85

# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

SENTIMENT_MODEL         = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SENTIMENT_BATCH_SIZE    = 32
SENTIMENT_USE_PROCESSED = True  # True = 'processed' col; False = 'text'

# =========================================================
# HDBSCAN
# =========================================================

MIN_CLUSTER_SIZE = 10
MIN_SAMPLES      = 5

# =========================================================
# LOGGING
# =========================================================

LOG_LEVEL = "INFO"