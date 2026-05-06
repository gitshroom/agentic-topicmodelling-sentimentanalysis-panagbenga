# =========================
# run_pipeline.py
# Master runner — executes the full pipeline in order:
#   1. Data Collection
#   2. Preprocessing
#   3. Embeddings & Clustering
#   4. Topic Modelling + Sentiment Analysis (parallel)
#   5. Orchestrator (validate + feedback loop)
#   6. Dashboard
# =========================

import os
import sys
import subprocess
import argparse
from datetime import datetime

# -----------------------------------------------------------------------
# CONFIG — edit these before running
# -----------------------------------------------------------------------

APIFY_TOKEN = os.getenv("APIFY_TOKEN")  # get from https://my.apify.com/account
QUERY        = "panagbenga"
PLATFORMS    = "all"          # all | facebook tiktok twitter instagram
MAX_ITEMS    = 700            # items per platform

SKIP_COLLECTION = False       # True = skip data collection, use existing CSV
OPEN_DASHBOARD  = True        # True = open browser when dashboard starts

# -----------------------------------------------------------------------

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
    bar = "─" * 52
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{CYAN}  {BOLD}{title}{RESET}")
    print(f"{CYAN}{bar}{RESET}")

def run_step(label: str, cmd: list[str]) -> bool:
    banner(label)
    log(f"Running: {' '.join(cmd)}", DIM)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        log(f"✗ FAILED: {label}", RED)
        return False
    log(f"✓ Done: {label}", GREEN)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full Panagbenga analysis pipeline.")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection, use existing CSV.")
    parser.add_argument("--no-dashboard",    action="store_true", help="Don't launch the dashboard at the end.")
    parser.add_argument("--query",           type=str, default=QUERY,     help="Search query for data collection.")
    parser.add_argument("--platforms",       type=str, default=PLATFORMS,  help="Platforms: all | facebook tiktok twitter instagram")
    parser.add_argument("--max-items",       type=int, default=MAX_ITEMS,  help="Max items per platform.")
    args = parser.parse_args()

    skip_collection = args.skip_collection or SKIP_COLLECTION
    open_dashboard  = not args.no_dashboard and OPEN_DASHBOARD
    query           = args.query
    platforms       = args.platforms.split() if args.platforms != "all" else ["all"]
    max_items       = args.max_items

    python = sys.executable  # use same Python that launched this script

    print(f"""
{BOLD}╔══════════════════════════════════════════════════════╗
║       Panagbenga Analysis — Full Pipeline Runner     ║
╚══════════════════════════════════════════════════════╝{RESET}
  Query      : {CYAN}{query}{RESET}
  Platforms  : {CYAN}{platforms}{RESET}
  Max items  : {CYAN}{max_items}{RESET}
  Skip collect: {CYAN}{skip_collection}{RESET}
  Dashboard  : {CYAN}{open_dashboard}{RESET}
""")

    steps_run  = 0
    steps_fail = 0

    # ── Step 1: Data Collection ──────────────────────────────────────────
    if not skip_collection:
        ok = run_step("Step 1 — Data Collection (Apify)", [
            python, "data_collection_agent.py",
            "--output", "data/panagbenga-dataset.csv",
            "--query",      query,
            "--platforms",  *platforms,
            "--max_items",  str(max_items),
            "--api_token",  APIFY_TOKEN,
        ])
        steps_run += 1
        if not ok:
            log("Data collection failed. Fix the error above and rerun, or use --skip-collection to skip.", RED)
            sys.exit(1)
    else:
        log("Skipping data collection — using existing panagbenga-dataset.csv", CYAN)

    # ── Step 2: Preprocessing ────────────────────────────────────────────
    ok = run_step("Step 2 — Preprocessing", [python, "preprocessing.py"])
    steps_run += 1
    if not ok:
        steps_fail += 1
        log("Preprocessing failed. Cannot continue.", RED)
        sys.exit(1)

    # ── Step 3: Embeddings & Clustering ──────────────────────────────────
    ok = run_step("Step 3 — Embeddings & Clustering", [python, "embeddings.py"])
    steps_run += 1
    if not ok:
        steps_fail += 1
        log("Embeddings failed. Cannot continue.", RED)
        sys.exit(1)

    # ── Step 4+5: Orchestrator (runs topic + sentiment in parallel internally)
    ok = run_step("Step 4 & 5 — Topic Modelling + Sentiment (via Orchestrator)", [python, "orchestrator.py"])
    steps_run += 1
    if not ok:
        steps_fail += 1
        log("Orchestrator failed.", RED)
        sys.exit(1)

    # ── Step 6: Dashboard ────────────────────────────────────────────────
    banner("Step 6 — Dashboard")
    if open_dashboard:
        import threading, webbrowser, time
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:5000")
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"""
{GREEN}{BOLD}══════════════════════════════════════════
  Pipeline complete!
  Results  → results.json
  Dataset  → panagbenga-dataset.csv
  Dashboard→ http://localhost:5000
══════════════════════════════════════════{RESET}
""")

    # Block here running the dashboard (Ctrl+C to stop)
    try:
        subprocess.run([python, "dashboard_agent.py"])
    except KeyboardInterrupt:
        log("Dashboard stopped.", DIM)


if __name__ == "__main__":
    main()
