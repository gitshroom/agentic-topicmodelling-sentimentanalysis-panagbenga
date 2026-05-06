# =========================
# preprocessing.py
# Cleans and tokenises raw social media text.
# Extracts a 'year' column from the timestamp for downstream year-based analysis.
# Can be run standalone:  python preprocessing.py
# Or imported:            from preprocessing import main; main()
# =========================

# pip install pandas nltk emoji beautifulsoup4 stopwordsiso

import re
import string

import emoji
import nltk
import pandas as pd
import stopwordsiso
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import config


# =========================
# NLP RESOURCES (loaded once at import time)
# =========================
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

_stop_en = set(stopwords.words("english"))
_stop_tl = stopwordsiso.stopwords("tl")
ALL_STOP_WORDS = _stop_en.union(_stop_tl)

DOMAIN_STOPWORDS = {
    "panagbenga",
    "panagbengafestival",
    "panagbengafestival2024",
    "panagbenga2024",
}


# =========================
# HELPER FUNCTIONS
# =========================

def is_noise_token(token: str) -> bool:
    if len(token) > 20:
        return True
    if re.search(r"[^\x00-\x7F]", token):
        return True
    if len(re.findall(r"[aeiou]", token)) <= 1 and len(token) > 5:
        return True
    return False


def extract_emojis(text: str) -> list[str]:
    return [c for c in text if c in emoji.EMOJI_DATA]


def clean_text(text: str) -> tuple[str, list, list]:
    text = str(text).lower()
    emojis = extract_emojis(text)
    hashtags = re.findall(r"#\w+", text)
    hashtags = [tag.replace("#", "HASHTAG_") for tag in hashtags]

    text = re.sub(r"https\S+|@\w+", "", text)
    text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)
    text = re.sub(r"\d+", "", text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, emojis, hashtags


def is_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    return SequenceMatcher(None, a, b).ratio() > threshold


def dynamic_token_cleanup(tokens: list, hashtags: list) -> list:
    hashtag_words = [tag.replace("HASHTAG_", "") for tag in hashtags]
    combined = tokens + hashtag_words
    cleaned: list[str] = []
    for token in combined:
        if is_noise_token(token):
            continue
        if any(is_similar(token, existing) for existing in cleaned):
            continue
        cleaned.append(token)
    return cleaned


def clean_hashtags(hashtags: list) -> list:
    return [tag for tag in hashtags if tag.replace("HASHTAG_", "") not in DOMAIN_STOPWORDS]


def extract_year(ts) -> int | None:
    """
    Best-effort year extraction from a timestamp value.
    Handles ISO strings ('2025-03-14T...'), plain year strings ('2025'),
    Unix epoch ints/floats, and NaN/None.
    Returns None when the year cannot be determined.
    """
    if pd.isna(ts):
        return None
    ts_str = str(ts).strip()

    # ISO / date strings: '2025-03-14', '2025-03-14T10:00:00+08:00', etc.
    m = re.match(r"(\d{4})", ts_str)
    if m:
        yr = int(m.group(1))
        if config.YEAR_START <= yr <= config.YEAR_END:
            return yr

    # Unix epoch (numeric string or actual number)
    try:
        epoch = float(ts_str)
        yr = pd.Timestamp(epoch, unit="s").year
        if config.YEAR_START <= yr <= config.YEAR_END:
            return yr
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
        pass

    return None


# =========================
# MAIN
# =========================

def main(input_file: str = None, output_file: str = None) -> pd.DataFrame:
    src  = input_file  or config.RAW_FILE
    dest = output_file or config.PREPROCESSED_FILE

    print(f"[preprocessing] Loading dataset from {src}…")
    df = pd.read_csv(src)
    df = df[["id", "text", "source", "timestamp", "engagement"]].copy()
    df["text"] = df["text"].astype(str)
    print(f"[preprocessing] Rows loaded: {df.shape[0]}")

    # ── Extract year ──────────────────────────────────────────────────────
    df["year"] = df["timestamp"].apply(extract_year)
    missing_year = df["year"].isna().sum()
    if missing_year:
        print(f"[preprocessing] WARNING: {missing_year} rows have no parseable year — they will be dropped.")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    year_dist = df["year"].value_counts().sort_index()
    print("[preprocessing] Posts per year:")
    for yr, cnt in year_dist.items():
        print(f"  {yr}: {cnt} posts")

    # ── Clean & tokenise ─────────────────────────────────────────────────
    cleaned          = df["text"].apply(clean_text)
    df["clean_text"] = cleaned.apply(lambda x: x[0])
    df["emojis"]     = cleaned.apply(lambda x: x[1])
    df["hashtags"]   = cleaned.apply(lambda x: x[2])
    df["hashtags"]   = df["hashtags"].apply(clean_hashtags)
    df["tokenized"]  = df["clean_text"].apply(word_tokenize)

    # ── Stopword filtering ───────────────────────────────────────────────
    df["filtered"] = df["tokenized"].apply(
        lambda tokens: [
            w for w in tokens
            if w not in ALL_STOP_WORDS and w not in DOMAIN_STOPWORDS
        ]
    )

    # ── Dynamic normalisation ─────────────────────────────────────────────
    df["normalized_tokens"] = df.apply(
        lambda row: dynamic_token_cleanup(row["filtered"], row["hashtags"]),
        axis=1,
    )

    # ── Final processed text ──────────────────────────────────────────────
    df["processed"] = df.apply(
        lambda row: " ".join(row["normalized_tokens"] + row["emojis"]),
        axis=1,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    cols = ["id", "text", "processed", "clean_text", "emojis",
            "hashtags", "source", "timestamp", "year", "engagement"]
    df[cols].to_csv(dest, index=False)
    print(f"[preprocessing] Done. Saved {len(df)} rows → {dest}")
    return df[cols]


if __name__ == "__main__":
    main()
