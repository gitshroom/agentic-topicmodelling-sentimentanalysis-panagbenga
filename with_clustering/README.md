# Panagbenga Pipeline 2 — With Clustering

End-to-end agentic pipeline for the Panagbenga Festival corpus:

```
data collection -> preprocessing -> embeddings + UMAP + HDBSCAN clustering
                -> BERTopic per cluster -> XLM-RoBERTa sentiment per topic
                -> explanation agent -> Flask dashboard
```

The "with_clustering" variant differs from Pipeline 1 in that posts are
**hard-clustered per year** before topic modelling. Topics, coherence,
and sentiment are then computed *inside each cluster*, giving more
coherent thematic groupings and sentiment readings.

---

## Directory layout

```
with_clustering/
  config.py              # all knobs, paths, models
  utils.py               # logging, JSON helpers
  preprocessing.py       # year extraction + multilingual text cleaning (default)
  preprocessing_second.py  # stricter v2 cleaning + blocked-account filter
  embeddings.py          # sentence embeddings + UMAP + HDBSCAN + visuals
  topic_modelling.py     # BERTopic per (year, cluster) + c_v coherence
  sentiment_analysis.py  # xlm-roberta per topic + confidence aggregations
  results_formatter.py   # merges everything into outputs/results.json
  explainer_agent.py     # plain-language descriptions via Ollama
  dashboard_agent.py     # Flask dashboard (Top 10 + Clusters + Timeline)
  orchestrator.py        # runs steps 4–7
  run_pipeline.py        # one-command end-to-end runner
  data/                  # input + intermediate data
    panagbenga2013-2026_cleaned=9013.csv
    prep_dataset_v4.csv      # default preprocessing output
    prep_dataset_second.csv  # output when using --preprocessing second
    clustered_dataset.pkl
  outputs/
    cluster_summary.json
    topic_results.json
    sentiment_results.json
    results.json
    explained_results.json
    visualizations/
      cluster_viz_<year>.png   # 2D UMAP scatter per year
      coherence_overview.png   # c_v per cluster bar chart
      sentiment_overview.png   # sentiment ratio + avg confidence
```

---

## How to run

From the project root, on Windows PowerShell:

```powershell
cd with_clustering
python run_pipeline.py
```

**Stricter preprocessing (thesis v2 corpus):** use the second preprocessor
and its dedicated CSV so the default `prep_dataset_v4.csv` is left intact:

```powershell
python run_pipeline.py --preprocessing second
```

Useful flags:

| Flag                    | Effect                                         |
|-------------------------|------------------------------------------------|
| `--preprocessing default` | `preprocessing.py` → `prep_dataset_v4.csv` (default) |
| `--preprocessing second` | `preprocessing_second.py` → `prep_dataset_second.csv` |
| `--skip-preprocess`     | Reuse the preprocessed CSV for the active variant (see above) |
| `--skip-embeddings`     | Reuse `data/clustered_dataset.pkl`             |
| `--skip-orchestrator`   | Only build embeddings/clusters                 |
| `--no-explainer`        | Do not call Ollama                             |
| `--no-dashboard`        | Skip the Flask UI                              |

The dashboard runs at `http://localhost:5050` by default. Change the port
in `config.DASHBOARD_PORT`.

If `data/panagbenga2013-2026_cleaned=9013.csv` is missing, the runner
auto-copies it from `../main/data/`.

The Ollama explainer is optional. If `ollama serve` is not running, the
agent simply logs a warning and the dashboard falls back to the raw
BERTopic labels.

---

## Methodology and what we document

### 1. Data collection
The raw CSV `panagbenga2013-2026_cleaned=9013.csv` is the deduplicated
multi-platform dataset (Facebook / Instagram / TikTok / X). We do not
re-collect by default; the runner just copies it into `data/`.

### 2. Preprocessing (`preprocessing.py`)
- Year extraction from `timestamp` (kept only `YEAR_START..YEAR_END`).
- Lowercasing, Unicode normalization, emoji stripping, URL/mention/hashtag
  cleanup.
- Tokenisation + stopword removal (EN + TL + Ilocano + social-media
  fillers + domain words like `panagbenga`, `baguio`).
- Drops empty docs and docs with fewer than 3 tokens.
- Writes `data/prep_dataset_v4.csv` with the `processed` column.
- Logs the per-year document count.

### 2b. Preprocessing v2 (`preprocessing_second.py`)

Optional stricter pass, selected with `--preprocessing second`:

- Same year window and column schema as the default preprocessor.
- **Blocked-account filter:** drops rows whose text matches known
  off-topic spam accounts (e.g. crystal-bracelet ads) before cleaning.
- **Richer stopword lists** (Tagalog function words, social-media junk,
  domain hashtags) and **ASCII-only token pass** after emoji removal so
  stray non-Latin characters from compound hashtags do not survive.
- **Unicode-normalization order** fixed so stylised Unicode letters
  still map to lowercase tokens that match stopwords.
- Writes `data/prep_dataset_second.csv`. Embeddings always read
  `config.PREPROCESSED_INPUT_FILE`, which `run_pipeline.py` sets from this flag.

### 3. Embeddings + clustering (`embeddings.py`)
- Encodes the `processed` column with
  `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  (L2-normalised).
- **Per year**: UMAP-reduces to `UMAP_COMPONENTS=5` and clusters with
  HDBSCAN (`PRE_CLUSTER_MIN_CLUSTER_SIZE=20`,
  `PRE_CLUSTER_MIN_SAMPLES=5`). The labels are persisted as
  `pre_cluster_label` so downstream steps see exactly the same clusters.
- A 2-D UMAP projection is computed and saved as
  `outputs/visualizations/cluster_viz_<year>.png`.
- `outputs/cluster_summary.json` records per-year `n_clusters`,
  `noise_ratio`, and `cluster_sizes`, plus `preprocessing_variant` and
  `preprocessed_input_csv` when embeddings ran after `set_preprocessing_variant()`.

### 4. Topic modelling (`topic_modelling.py`)
- Loads the pickled dataframe and uses `pre_cluster_label` to iterate
  clusters in deterministic order.
- For each cluster ≥ `MIN_DOCS_PER_CLUSTER=20`, fits a `BERTopic` model
  (multilingual MPNet embeddings, `KeyBERTInspired` representation,
  HDBSCAN inside BERTopic).
- For each topic we record: `topic_id`, `label`, `count`, `top_words`,
  `top_word_scores`.
- **Coherence:** we compute the `c_v` score per cluster with
  `gensim.models.CoherenceModel` using the topics' top words and the
  cluster's documents. We also aggregate `avg_coherence` per year.
- Per-year/per-cluster coherence is plotted in
  `outputs/visualizations/coherence_overview.png` (dashed line at 0.5
  is a common acceptability threshold).

> **Reading c_v:** values usually fall in `[0, 1]`. Higher is better.
> `> 0.55` is generally considered a strong, well-formed topic;
> `0.40–0.55` is acceptable for noisy social-media corpora; below `0.40`
> indicates topic incoherence.

### 5. Sentiment analysis (`sentiment_analysis.py`)
- Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (3-class
  POSITIVE / NEUTRAL / NEGATIVE) on CPU by default.
- For each topic inside its cluster, docs are selected by top-word
  overlap (fallback: all cluster docs).
- Per topic we record:
  - `n_docs`, `coverage`, `dominant_sentiment`
  - `avg_confidence` (mean softmax score over the predicted class)
  - `std_confidence` (spread)
  - `high_confidence_ratio` — share of predictions with score
    `≥ SENTIMENT_HIGH_CONFIDENCE (=0.75)`
  - `label_distribution`, `label_ratios`
- Cluster-level and year-level summaries aggregate the above
  weighted by `n_docs`.
- `outputs/visualizations/sentiment_overview.png` plots the per-year
  POSITIVE/NEUTRAL/NEGATIVE share with a line for the average
  confidence.

> **Reading confidence:** `avg_confidence` is the mean softmax of the
> predicted class. `> 0.80` is high model certainty; `0.60–0.80` is
> typical for code-switched social text; `< 0.60` suggests the topic
> contains ambiguous or mixed-sentiment posts.

### 6. Explanation agent (`explainer_agent.py`)
- Flattens topics across clusters per year, ranks by post count, keeps
  top 10, and asks a local Ollama model (`qwen2.5:3b` by default) to
  describe each topic in ≤2 sentences.
- Outputs `outputs/explained_results.json` augmented with
  `explanation`, `top_10_rank`, and `top_10_topics`.
- Non-fatal: if Ollama is unavailable, the dashboard reads the raw
  `results.json` instead.

### 7. Dashboard (`dashboard_agent.py`)
Three tabs:

1. **Top 10 Topics** — flattened, sorted by count. Shows cluster tag,
   sentiment badge, average confidence, and high-confidence ratio.
2. **Clusters** — per-year cluster cards with coherence and sentiment
   confidence bars + the year's UMAP scatter PNG.
3. **Timeline** — year cards with POSITIVE/NEUTRAL/NEGATIVE bar +
   coherence + confidence numbers.

---

## Output schema highlights (`outputs/results.json`)

```jsonc
{
  "summary": {
    "year_range": "2022–2026",
    "total_docs": 8421,
    "total_clusters": 27,
    "total_topics": 142,
    "avg_coherence": 0.49,
    "avg_confidence": 0.71,
    "preprocessing_variant": "second",
    "preprocessed_input_csv": "prep_dataset_second.csv"
  },
  "models": {
    "topic_model_type": "bertopic",
    "sentiment_model": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "sentiment_confidence_threshold": 0.75
  },
  "years": [
    {
      "year": 2024,
      "n_clusters": 6,
      "avg_coherence": 0.52,
      "noise_ratio": 0.18,
      "sentiment_summary": { "dominant_sentiment": "POSITIVE",
                             "avg_confidence": 0.74,
                             "high_confidence_ratio": 0.55 },
      "visualization": "outputs/visualizations/cluster_viz_2024.png",
      "clusters": [
        {
          "cluster_id": 0,
          "coherence": 0.58,
          "sentiment_summary": { "...": "..." },
          "topics": [
            {
              "topic_id": 0,
              "label": "Float / Parade / Grand",
              "count": 312,
              "top_words": ["float", "parade", "grand", "..."],
              "sentiment": { "dominant": "POSITIVE",
                             "avg_confidence": 0.79,
                             "high_confidence_ratio": 0.63,
                             "label_distribution": { "POSITIVE": 240, "NEUTRAL": 50, "NEGATIVE": 22 } }
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Reproducibility notes

- All UMAP / HDBSCAN steps fix `random_state=42`.
- Models are pinned by name in `config.py`.
- All intermediate artefacts (preprocessed CSV, embeddings pickle,
  JSONs, PNGs) live under `with_clustering/` so the pipeline does not
  pollute Pipeline 1.
- `outputs/visualizations/` is regenerated on every run.

---

## Troubleshooting

### `LookupError: Resource punkt_tab not found`
Run `python -c "import nltk; nltk.download('punkt_tab')"` once. The
`preprocessing.py` script also tries to download it automatically.

### `ImportError: cannot import name 'builder' from 'google.protobuf.internal'`
The `sentencepiece` tokenizer used by `xlm-roberta` needs protobuf
`>= 3.20`. TensorFlow 2.10 (a transitive dependency on some setups)
needs protobuf `< 4`. The working version is **`protobuf 3.20.x`**:

```powershell
pip install "protobuf>=3.20.0,<3.21"
```

If you also see `ValueError: 'tiktoken' is required`, install it too:

```powershell
pip install tiktoken
```

### `BERTopic failed for cluster X: max_df corresponds to < documents than min_df`
This is raised by sklearn's `CountVectorizer` when BERTopic re-runs
the vectorizer on a per-topic doc subset that is smaller than
`min_df`. The pipeline now passes `min_df=1, max_df=1.0` so this
should not happen; if you bump those values up, very small topics
may fail.

### Ollama is offline
The explainer agent probes `http://localhost:11434` once at startup
and falls back to a deterministic template explanation per topic if
it is unreachable. Outputs are still written to
`outputs/explained_results.json` so the dashboard renders normally.

### Dashboard port already in use
Change `DASHBOARD_PORT` in `config.py` (default `5050`).
