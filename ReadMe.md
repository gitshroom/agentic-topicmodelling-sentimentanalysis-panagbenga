# Panagbenga AI Analysis Pipeline

An end-to-end agentic AI pipeline for analyzing social media data related to the Panagbenga Festival.  
The project performs data collection, preprocessing, embeddings, clustering, topic modeling, sentiment analysis, and dashboard visualization.

## Repository

GitHub: [https://github.com/gitshroom/agentic-topicmodelling-sentimentanalysis-panagbenga](https://github.com/gitshroom/agentic-topicmodelling-sentimentanalysis-panagbenga)

## Features

- Automated data collection through Apify
- Text preprocessing (normalization, stopword filtering, cleaning)
- Embeddings with transformer models
- Clustering with UMAP + HDBSCAN
- Topic modeling with BERTopic
- Sentiment analysis with XLM-RoBERTa
- Local LLM explanations through Ollama
- Flask dashboard for interactive exploration

## Installation

Install the required Python packages:

```bash
pip install pandas nltk emoji beautifulsoup4 stopwordsiso sentence-transformers umap-learn hdbscan scikit-learn matplotlib numpy "bertopic[representation]" transformers torch apify-client flask sentencepiece blobfile requests
```

## Environment Variables

Set environment variables before running data collection and LLM explanation features.

### Windows (PowerShell)

```powershell
setx APIFY_TOKEN "your_apify_token"
setx OLLAMA_HOST "http://localhost:11434"
```

Restart the terminal after `setx` so values are available in new sessions.

### macOS/Linux

```bash
export APIFY_TOKEN="your_apify_token"
export OLLAMA_HOST="http://localhost:11434"
```

## Local LLM Setup (Optional but Recommended)

Install Ollama from [https://ollama.com](https://ollama.com), then run:

```bash
ollama pull qwen2.5:3b
ollama serve
```

## How to Run

### Pipeline 1 (`main/`)

```bash
cd main
python run_pipeline.py
```

Run with existing dataset only (skip collection):

```bash
python run_pipeline.py --skip-collection
```

### Pipeline 2 (`with_clustering/`)

```bash
cd with_clustering
python run_pipeline.py
```

Run with stricter preprocessing:

```bash
python run_pipeline.py --preprocessing second
```

See `with_clustering/README.md` for detailed methodology and output schema.

## Output

- Dataset: `data/panagbenga-dataset.csv` (Pipeline 1) or preprocessed variants in `with_clustering/data/`
- JSON results: `outputs/results.json`, `outputs/topic_results.json`, `outputs/sentiment_results.json`
- Dashboard: `http://localhost:5000` (Pipeline 1) and `http://localhost:5050` (Pipeline 2)

## Pipeline Overview

This repository implements **two parallel pipelines** sharing the same
input data:

| | Pipeline 1 — `main/`                  | Pipeline 2 — `with_clustering/`             |
|-|----------------------------------------|----------------------------------------------|
|Flow | year → topic → sentiment           | year → **cluster** → topic → sentiment       |
|Topic backbone | BERTopic per year           | BERTopic per (year, cluster)                  |
|Cluster step | implicit (inside BERTopic)    | explicit UMAP + HDBSCAN pre-pass             |
|Sentiment | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | same |
|Confidence reporting | avg per topic        | avg + std + high-confidence ratio at every level |
|Coherence | per year                       | per cluster + per year aggregate             |
|Visualizations | cluster scatter per year  | cluster scatter + coherence + sentiment      |
|Dashboard port | 5000                       | 5050                                          |

Pipeline steps (both):

1. **Data Collection** – Fetches social media data via Apify  
2. **Preprocessing** – Cleans and normalizes text  
3. **Embeddings & Clustering** – Groups similar content  
4. **Topic Modeling** – Extracts key discussion themes  
5. **Sentiment Analysis** – Identifies public sentiment  
6. **LLM Explainer Agent** – Generates human-readable topic descriptions using Ollama + Qwen  
7. **Orchestrator Agent** – Validates and refines outputs  
8. **Dashboard** – Visualizes results interactively  

See [with_clustering/README.md](with_clustering/README.md) for the full
methodology, output schema, and how to run Pipeline 2.

---

## Notes

- Ensure your Apify token is valid before running data collection
- Make sure Ollama is running before executing explanation steps
- Large datasets may require substantial time for embedding and clustering
- Use skip flags to speed up repeated experiments
- The first Qwen2.5:3B run can take longer while model weights load

---

## Future Improvements

- Real-time streaming data support
- Expanded multilingual sentiment coverage
- Fine-tuned local LLM for Baguio/Panagbenga context
- Cloud deployment for the dashboard
- Automated hyperparameter optimization for topic modeling
