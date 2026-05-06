# =========================
# config.py
# Central configuration for the agentic topic modelling + sentiment pipeline
# Restructured: analysis is done per YEAR, with sentiment run per TOPIC per year.
# =========================

# --- File paths ---
RAW_FILE            = "data/panagbenga2013-2026_cleaned=9013.csv"
PREPROCESSED_FILE   = "data/prep_dataset_v3.csv"
CLUSTERED_FILE      = "data/clustered_dataset.csv"   # still used; now has a 'year' column
TOPIC_OUTPUT_FILE   = "outputs/topic_results.json"
SENTIMENT_OUTPUT_FILE = "outputs/sentiment_results.json"
FINAL_OUTPUT_FILE   = "outputs/results.json"

# --- Year range ---
YEAR_START = 2013
YEAR_END   = 2026          # inclusive

# --- Topic Modelling ---
TOPIC_MODEL_TYPE        = "bertopic"   # "bertopic" or "lda"
BERTOPIC_MIN_TOPIC_SIZE = 10
BERTOPIC_NR_TOPICS      = "auto"       # or int to force a count
LDA_N_TOPICS            = 8
LDA_MAX_ITER            = 50
TOP_N_WORDS             = 10

# Minimum docs in a year-slice to attempt topic modelling
MIN_DOCS_PER_YEAR = 5

# --- Sentiment Analysis ---
SENTIMENT_MODEL         = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_BATCH_SIZE    = 32
SENTIMENT_USE_PROCESSED = True   # True = 'processed' col; False = 'text'

# --- Orchestrator Quality Thresholds ---
# Topic modelling (evaluated across all years)
MIN_TOPICS          = 3           # minimum total topics across all years
MAX_NOISE_RATIO     = 0.40        # max avg noise ratio across years
MIN_TOPIC_COHERENCE = 0.80        # min avg coherence (BERTopic only)

# Sentiment (evaluated across all year-topic pairs)
MIN_SENTIMENT_COVERAGE   = 0.85
MIN_SENTIMENT_CONFIDENCE = 0.60

# --- Feedback / Retry ---
MAX_RETRIES = 3

BERTOPIC_MIN_TOPIC_SIZE_DELTA = -2   # reduce min_topic_size each retry
LDA_N_TOPICS_DELTA            = 2    # add more topics each retry

# --- Logging ---
LOG_LEVEL = "INFO"   # DEBUG | INFO | WARNING | ERROR
