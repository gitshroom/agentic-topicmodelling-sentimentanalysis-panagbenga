# =========================
# run_pipeline.py
# Master runner for Pipeline 2 (with clustering).
#
# Steps:
#   1. (Optional) data collection — skipped by default; uses local CSV
#   2. Preprocessing               — year extraction + cleaning
#   3. Embeddings + Clustering     — UMAP + HDBSCAN + per-year visuals
#   4. Topic Modelling             — BERTopic per cluster (uses persisted labels)
#   5. Sentiment Analysis          — XLM-RoBERTa per topic per cluster
#   6. Results Formatter           — merges everything into results.json
#   7. Explainer Agent             — plain-language descriptions (Ollama)
#   8. Dashboard                   — Flask app at config.DASHBOARD_PORT
#
# Usage:
#   python run_pipeline.py                    # full run
#   python run_pipeline.py --skip-preprocess  # skip step 2
#   python run_pipeline.py --skip-embeddings  # skip step 3
#   python run_pipeline.py --no-dashboard     # don't launch the dashboard
#   python run_pipeline.py --no-explainer     # skip step 7
# =========================

import argparse
import os
import shutil
import sys
import time
from datetime import datetime

import config

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
DIM   = "\033[2m"
RESET = "\033[0m"


def log(msg, color=RESET):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"{DIM}[{t}]{RESET} {color}{msg}{RESET}")


def banner(title):
    bar = "─" * 54
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{CYAN}  {BOLD}{title}{RESET}")
    print(f"{CYAN}{bar}{RESET}")


# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

def ensure_raw_dataset():
    """Copy the raw dataset from the fallback location if not present."""
    if os.path.exists(config.RAW_FILE):
        log(f"Raw dataset found → {config.RAW_FILE}", DIM)
        return
    if os.path.exists(config.FALLBACK_RAW_FILE):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        log(
            f"Copying raw dataset from {config.FALLBACK_RAW_FILE} → {config.RAW_FILE}",
            CYAN,
        )
        shutil.copy2(config.FALLBACK_RAW_FILE, config.RAW_FILE)
        return
    raise FileNotFoundError(
        f"No raw dataset at {config.RAW_FILE} (and fallback "
        f"{config.FALLBACK_RAW_FILE} also missing)."
    )


# -----------------------------------------------------------------------
# Step runners (in-process so failures propagate cleanly)
# -----------------------------------------------------------------------

def step_preprocessing():
    banner("Step 2 — Preprocessing + Year Extraction")
    import preprocessing
    preprocessing.main()


def step_embeddings():
    banner("Step 3 — Embeddings + UMAP + HDBSCAN Clustering")
    import embeddings
    embeddings.main()


def step_orchestrator():
    banner("Step 4–7 — Orchestrator (Topic + Sentiment + Format + Explain)")
    import orchestrator
    orchestrator.main()


def step_dashboard():
    banner(f"Step 8 — Dashboard (http://localhost:{config.DASHBOARD_PORT})")
    import dashboard_agent
    # Flask app blocks until Ctrl+C
    dashboard_agent.app.run(
        debug=False, port=config.DASHBOARD_PORT, host="127.0.0.1"
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Panagbenga Pipeline 2 (with clustering)."
    )
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-orchestrator", action="store_true")
    parser.add_argument("--no-dashboard",    action="store_true")
    parser.add_argument("--no-explainer",    action="store_true",
                        help="Disable the explainer agent step.")
    args = parser.parse_args()

    print(f"""
{BOLD}╔════════════════════════════════════════════════════════╗
║   Panagbenga Pipeline 2 — With Clustering             ║
║   year → cluster → topic → sentiment → explain        ║
╚════════════════════════════════════════════════════════╝{RESET}
  Embedding model : {CYAN}{config.EMBEDDING_MODEL}{RESET}
  Sentiment model : {CYAN}{config.SENTIMENT_MODEL}{RESET}
  Year range      : {CYAN}{config.YEAR_START}–{config.YEAR_END}{RESET}
  Output dir      : {CYAN}{config.OUT_DIR}{RESET}
""")

    if args.no_explainer:
        os.environ["WITH_CLUSTERING_NO_EXPLAINER"] = "1"

    ensure_raw_dataset()

    start = time.time()

    try:
        if not args.skip_preprocess:
            step_preprocessing()
        else:
            log("Skipping preprocessing (--skip-preprocess).", CYAN)

        if not args.skip_embeddings:
            step_embeddings()
        else:
            log("Skipping embeddings (--skip-embeddings).", CYAN)

        if not args.skip_orchestrator:
            step_orchestrator()
        else:
            log("Skipping orchestrator (--skip-orchestrator).", CYAN)

    except Exception as e:
        log(f"Pipeline failed: {e}", RED)
        raise

    elapsed = time.time() - start
    print(f"""
{GREEN}{BOLD}══════════════════════════════════════════════════
  Pipeline complete in {elapsed:.1f}s
  Results       → {config.FINAL_OUTPUT_FILE}
  Explained     → {config.EXPLAINED_OUTPUT_FILE}
  Visuals dir   → {config.VIZ_DIR}
  Cluster summary → {config.CLUSTER_SUMMARY_FILE}
══════════════════════════════════════════════════{RESET}
""")

    if not args.no_dashboard:
        try:
            step_dashboard()
        except KeyboardInterrupt:
            log("Dashboard stopped.", DIM)


if __name__ == "__main__":
    main()
