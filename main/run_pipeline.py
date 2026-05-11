# =========================
# run_pipeline.py
# Master runner — executes the full pipeline in order:
#   1. Data Collection      (optional, skip with --skip-collection)
#   2. Preprocessing        (extracts year column, cleans text)
#   3. Embeddings & Clustering  (per-year UMAP + HDBSCAN)
#   4+5. Orchestrator       (topic modelling + sentiment per year/topic, with retry)
#   6. Dashboard            (interactive year timeline at http://localhost:5000)
# =========================

import os
import sys
import subprocess
import argparse
from datetime import datetime

# -----------------------------------------------------------------------
# CONFIG — edit before running
# -----------------------------------------------------------------------
APIFY_TOKEN      = os.getenv("APIFY_TOKEN")
QUERY            = "panagbenga"
PLATFORMS        = "all"
MAX_ITEMS        = 700

SKIP_COLLECTION  = False
OPEN_DASHBOARD   = True
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
    bar = "─" * 54
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
    parser = argparse.ArgumentParser(description="Panagbenga analysis pipeline (2013–2026).")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip data collection; use existing CSV.")
    parser.add_argument("--no-dashboard",    action="store_true",
                        help="Don't launch the dashboard at the end.")
    parser.add_argument("--query",     type=str, default=QUERY)
    parser.add_argument("--platforms", type=str, default=PLATFORMS)
    parser.add_argument("--max-items", type=int, default=MAX_ITEMS)
    args = parser.parse_args()

    skip_collection = args.skip_collection or SKIP_COLLECTION
    open_dashboard  = not args.no_dashboard and OPEN_DASHBOARD
    query           = args.query
    platforms       = args.platforms.split() if args.platforms != "all" else ["all"]
    max_items       = args.max_items
    python          = sys.executable

    print(f"""
{BOLD}╔════════════════════════════════════════════════════════╗
║   Panagbenga Analysis — Full Pipeline (2013–2026)     ║
╚════════════════════════════════════════════════════════╝{RESET}
  Query         : {CYAN}{query}{RESET}
  Platforms     : {CYAN}{platforms}{RESET}
  Max items     : {CYAN}{max_items}{RESET}
  Skip collect  : {CYAN}{skip_collection}{RESET}
  Dashboard     : {CYAN}{open_dashboard}{RESET}
""")

    # ── Step 1: Data Collection ──────────────────────────────────────────
    if not skip_collection:
        ok = run_step("Step 1 — Data Collection (Apify)", [
            python, "data_collection_agent.py",
            "--output",    "data/panagbenga-dataset.csv",
            "--query",     query,
            "--platforms", *platforms,
            "--max_items", str(max_items),
            "--api_token", APIFY_TOKEN or "",
        ])
        if not ok:
            log("Data collection failed. Use --skip-collection to bypass.", RED)
            sys.exit(1)
    else:
        log("Skipping data collection — using existing CSV.", CYAN)

    # ── Step 2: Preprocessing (extracts year column) ─────────────────────
    ok = run_step("Step 2 — Preprocessing + Year Extraction", [python, "preprocessing.py"])
    if not ok:
        log("Preprocessing failed. Cannot continue.", RED)
        sys.exit(1)

    # ── Step 3: Embeddings & Per-Year Clustering ──────────────────────────
    ok = run_step("Step 3 — Embeddings & Per-Year Clustering", [python, "embeddings.py"])
    if not ok:
        log("Embeddings failed. Cannot continue.", RED)
        sys.exit(1)

    # ── Step 4+5: Orchestrator ────────────────────────────────────────────
    ok = run_step(
        "Step 4+5 — Topic Modelling + Sentiment per Year/Topic (Orchestrator)",
        [python, "orchestrator.py"],
    )
    if not ok:
        log("Orchestrator failed.", RED)
        sys.exit(1)

    # ── Step 6: Dashboard ─────────────────────────────────────────────────
    banner("Step 6 — Dashboard")
    if open_dashboard:
        import threading, webbrowser, time
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:5000")
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"""
{GREEN}{BOLD}══════════════════════════════════════════════════
  Pipeline complete!
  Results   → outputs/results.json
  Dashboard → http://localhost:5000
══════════════════════════════════════════════════{RESET}
""")

    try:
        subprocess.run([python, "dashboard_agent.py"])
    except KeyboardInterrupt:
        log("Dashboard stopped.", DIM)


if __name__ == "__main__":
    main()
