# =========================
# config.py (LLM Version)
# Central configuration for the agentic topic modelling + sentiment pipeline
# =========================
import os

# --- File paths ---
RAW_FILE = "data/panagbenga2013-2026_cleaned=9013.csv" # Add your raw file path if needed
PREPROCESSED_FILE = "data/prep_dataset_v3.csv"
CLUSTERED_FILE = "data/clustered_dataset.csv"
TOPIC_OUTPUT_FILE = "outputs/topic_results.json"
SENTIMENT_OUTPUT_FILE = "outputs/sentiment_results.json"
FINAL_OUTPUT_FILE = "outputs/results.json"

# --- Year Range ---
YEAR_START = 2022
YEAR_END = 2026

# --- Embeddings & Clustering Config ---
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
UMAP_NEIGHBORS = 15
UMAP_COMPONENTS = 5
UMAP_MIN_DIST = 0.1      # Changed to 0.1 as per Iteration 3
MIN_CLUSTER_SIZE = 10    # Changed to 10 as per Iteration 3
MIN_SAMPLES = 5          # Changed to 5 as per Iteration 3

# --- LLM Configuration ---
LLM_API_KEY = "ollama"           
LLM_MODEL_NAME = "llama3.2"

# --- Topic Modelling (LLM Params) ---
LLM_N_TOPICS = 3               
TOP_N_WORDS = 5                

# --- Sentiment Analysis (LLM Params) ---
SENTIMENT_USE_PROCESSED = True 

# --- Orchestrator Quality Thresholds ---
MIN_TOPICS = 5                 
MAX_NOISE_RATIO = 0.40         
MIN_TOPIC_COHERENCE = 0.80     
MIN_SENTIMENT_COVERAGE = 0.85  
MIN_SENTIMENT_CONFIDENCE = 0.60

# --- Feedback / Retry ---
MAX_RETRIES = 3
LLM_N_TOPICS_DELTA = 1         

# --- Logging ---
LOG_LEVEL = "INFO"