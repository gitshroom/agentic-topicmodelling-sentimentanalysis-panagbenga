# =========================
# config.py (LLM Version)
# Central configuration for the agentic topic modelling + sentiment pipeline
# =========================
import os

# --- File paths ---
PREPROCESSED_FILE = "data/prep_dataset_v3.csv"
CLUSTERED_FILE = "data/clustered_dataset.csv"
TOPIC_OUTPUT_FILE = "outputs/topic_results.json"
SENTIMENT_OUTPUT_FILE = "outputs/sentiment_results.json"
FINAL_OUTPUT_FILE = "outputs/results.json"

# --- LLM Configuration ---
LLM_API_KEY = "ollama"           # Just a placeholder, Ollama ignores this
LLM_MODEL_NAME = "llama3.2"

# --- Topic Modelling (LLM Params) ---
LLM_N_TOPICS = 3               # Default number of topics to extract per cluster
TOP_N_WORDS = 5                # Number of keywords per topic

# --- Sentiment Analysis (LLM Params) ---
SENTIMENT_USE_PROCESSED = True # True = use 'processed' col; False = use 'text'

# --- Orchestrator Quality Thresholds ---
# Topic modelling
MIN_TOPICS = 5                 # minimum acceptable topics discovered across ALL clusters combined
MAX_NOISE_RATIO = 0.40         # (Ignored for LLMs, kept for compatibility)
MIN_TOPIC_COHERENCE = 0.80     # (Ignored for LLMs, kept for compatibility)

# Sentiment
MIN_SENTIMENT_COVERAGE = 0.85  # fraction of docs that must have a sentiment label
MIN_SENTIMENT_CONFIDENCE = 0.60# (LLMs default to 1.0 confidence on success)

# --- Feedback / Retry ---
MAX_RETRIES = 3

# Adjustments applied per retry for LLM Topic Modelling
LLM_N_TOPICS_DELTA = 1         # Ask the LLM for more topics if the total count is too low

# --- Logging ---
LOG_LEVEL = "INFO"             # DEBUG | INFO | WARNING | ERROR