# =========================
# config.py
# Central configuration for the agentic topic modelling + sentiment pipeline
# =========================

# --- File paths ---
PREPROCESSED_FILE = "data/prep_dataset_v3.csv"
CLUSTERED_FILE = "data/clustered_dataset.csv"
TOPIC_OUTPUT_FILE = "outputs/topic_results.json"
SENTIMENT_OUTPUT_FILE = "outputs/sentiment_results.json"
FINAL_OUTPUT_FILE = "outputs/results.json"

# --- Topic Modelling ---
TOPIC_MODEL_TYPE = "bertopic"          # "bertopic" or "lda"
BERTOPIC_MIN_TOPIC_SIZE = 10
BERTOPIC_NR_TOPICS = "auto"            # or int to force a count
LDA_N_TOPICS = 8
LDA_MAX_ITER = 50
TOP_N_WORDS = 10

# --- Sentiment Analysis ---
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_USE_PROCESSED = True         # True = use 'processed' col; False = use 'text'

# --- Orchestrator Quality Thresholds ---
# Topic modelling
MIN_TOPICS = 5                         # minimum acceptable topics discovered
MAX_NOISE_RATIO = 0.40                 # max fraction of docs labelled as noise topic (-1)
MIN_TOPIC_COHERENCE = 0.80             # min avg coherence score (BERTopic only)

# Sentiment
MIN_SENTIMENT_COVERAGE = 0.85         # fraction of docs that must have a sentiment label
MIN_SENTIMENT_CONFIDENCE = 0.60       # min avg confidence score across all docs

# --- Feedback / Retry ---
MAX_RETRIES = 3

# Adjustments applied per retry for BERTopic
BERTOPIC_MIN_TOPIC_SIZE_DELTA = -2    # reduce min_topic_size each retry

# Adjustments per retry for LDA
LDA_N_TOPICS_DELTA = 2                # add more topics each retry

# --- Logging ---
LOG_LEVEL = "INFO"                     # DEBUG | INFO | WARNING | ERROR
