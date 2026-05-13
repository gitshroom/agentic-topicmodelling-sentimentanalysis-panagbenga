# =========================
# utils.py
# Shared helpers for the with_clustering agentic pipeline.
# =========================

import json
import logging
import os
from datetime import datetime

from config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger(name)


def save_json(data: dict, path: str) -> None:
    """Persist a dict as a pretty-printed JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str) -> dict:
    """Load a JSON file. Returns empty dict if file doesn't exist."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_banner(logger: logging.Logger, title: str) -> None:
    bar = "=" * 50
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def ensure_dir(path: str) -> str:
    """Create dir if missing and return the path."""
    os.makedirs(path, exist_ok=True)
    return path
