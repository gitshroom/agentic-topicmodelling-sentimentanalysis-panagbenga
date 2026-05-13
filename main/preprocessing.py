# =========================
# preprocessing.py
# Optimized multilingual preprocessing
# =========================

import re
import string
import unicodedata

import emoji
import nltk
import pandas as pd
import stopwordsiso

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import config

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# =========================================================
# STOPWORDS
# =========================================================

STOPWORDS_EN = set(stopwords.words("english"))
STOPWORDS_TL = stopwordsiso.stopwords("tl")

# Ilocano conversational fillers
STOPWORDS_ILO = {
    "wen",
    "haan",
    "nga",
    "gayam",
    "adi",
    "met",
    "man",
}
SOCIAL_MEDIA_STOPWORDS = {

    # Instagram / TikTok junk
    "instagood",
    "photooftheday",
    "picoftheday",
    "igdaily",
    "vscocam",
    "lateupload",
    "followme",
    "selfie",
    "repost",
    "trending",
    "viral",

    # TikTok algorithm bait
    "fyp",
    "fypp",
    "fyppp",
    "fypppp",
    "fyppppp",
    "foryou",
    "foryoupage",
    "foryouu",
    "xyzbca",
    "viralvideo",

    # Generic spam
    "lol",
    "lmao",
    "haha",
    "hahaha",
    "hehe",
    "omg",

    # Generic low-information fillers
    "post",
    "share",
    "comment",
    "like",
    "video",
    "tiktok",
}

DOMAIN_STOPWORDS = {
    "panagbenga",
    "baguio",
    "baguiocity",
    "festival",
    "flowerfestival",
    "philippines",
}

ALL_STOPWORDS = (
    STOPWORDS_EN
    | STOPWORDS_TL
    | STOPWORDS_ILO
    | SOCIAL_MEDIA_STOPWORDS
    | DOMAIN_STOPWORDS
)

# =========================================================
# HELPERS
# =========================================================

def normalize_repeated_chars(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

def remove_urls_mentions(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text

def clean_hashtags(text: str) -> str:
    text = re.sub(r"#(\w+?)(20\d{2})", r"\1", text)
    text = re.sub(r"#", "", text)
    return text

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text)

def extract_year(ts):
    if pd.isna(ts):
        return None

    ts = str(ts)

    match = re.match(r"(\d{4})", ts)

    if match:
        year = int(match.group(1))
        if config.YEAR_START <= year <= config.YEAR_END:
            return year

    return None

# =========================================================
# MAIN CLEANING
# =========================================================

def preprocess_text(text: str) -> str:

    text = str(text).lower()

    text = normalize_unicode(text)

    text = remove_urls_mentions(text)

    text = clean_hashtags(text)

    text = normalize_repeated_chars(text)

    text = BeautifulSoup(text, "html.parser").get_text()

    text = emoji.replace_emoji(text, replace=" ")

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = []

    for token in tokens:

        if len(token) <= 2:
            continue

        if token in ALL_STOPWORDS:
            continue

        if token.isnumeric():
            continue

        if re.fullmatch(r"[a-z]*\d+[a-z]*", token):
            continue

        cleaned_tokens.append(token)

    # Remove duplicates while preserving order
    cleaned_tokens = list(dict.fromkeys(cleaned_tokens))

    return " ".join(cleaned_tokens)

# =========================================================
# MAIN
# =========================================================

def main():

    print("[preprocessing] Loading dataset...")

    df = pd.read_csv(config.RAW_FILE)

    df = df[["id", "text", "source", "timestamp", "engagement"]]

    df["year"] = df["timestamp"].apply(extract_year)

    df = df.dropna(subset=["year"])

    print("[preprocessing] Cleaning text...")

    df["processed"] = df["text"].apply(preprocess_text)

    # Remove empty docs AFTER preprocessing
    df = df[df["processed"].str.strip().astype(bool)]

    # Remove tiny docs
    df = df[df["processed"].str.split().apply(len) >= 3]

    df.to_csv(config.PREPROCESSED_FILE, index=False)

    print(f"[preprocessing] Saved {len(df)} rows")

    return df

if __name__ == "__main__":
    main()