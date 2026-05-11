# =========================
# embeddings.py
# Generates multilingual sentence embeddings per doc and saves as pickle.
# UMAP + HDBSCAN clustering is handled inside BERTopic in topic_modelling.py.
# =========================

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config

INPUT_FILE  = config.PREPROCESSED_FILE
OUTPUT_FILE = config.CLUSTERED_FILE   # named CLUSTERED_FILE in config; now just embeds


def main():
    print("[embeddings] Loading preprocessed dataset...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["processed"])

    if "year" not in df.columns:
        raise ValueError("'year' column missing — run preprocessing.py first.")

    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print(f"[embeddings] Encoding {len(df)} docs with {config.EMBEDDING_MODEL}…")
    embeddings = model.encode(
        df["processed"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    df["embedding"] = embeddings.tolist()

    # Save as pickle (preserves list dtype of embedding column)
    df.to_pickle(OUTPUT_FILE)
    print(f"[embeddings] Saved {len(df)} rows with embeddings → {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    main()