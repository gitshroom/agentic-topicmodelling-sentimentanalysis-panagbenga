# 🌸 Panagbenga AI Analysis Pipeline

An end-to-end **agentic AI pipeline** for analyzing social media data related to the Panagbenga Festival.  
It performs **data collection, preprocessing, embeddings, clustering, topic modeling, sentiment analysis, and visualization**.

---

## 🚀 Features

- 📥 Automated data collection via Apify  
- 🧹 Text preprocessing (emoji, stopwords, normalization)  
- 🧠 Embeddings using transformer models  
- 🔍 Clustering with UMAP + HDBSCAN  
- 🏷️ Topic modeling via BERTopic  
- 😊 Sentiment analysis  
- 📊 Interactive dashboard (Flask-based)  
- 🔁 Orchestrated multi-step pipeline  

---

## 📦 Installation

Install all required dependencies:

```bash
pip install pandas nltk emoji beautifulsoup4 stopwordsiso sentence-transformers umap-learn hdbscan scikit-learn matplotlib numpy "bertopic[representation]" transformers torch apify-client flask sentencepiece blobfile
```

---

## 🔐 Environment Setup (IMPORTANT)

⚠️ Do **NOT** hardcode your API key inside the code.

Instead, set your Apify token as an environment variable:

### On Windows (PowerShell)
```bash
setx APIFY_TOKEN "your_actual_token_here"
```

### On macOS/Linux
```bash
export APIFY_TOKEN="your_actual_token_here"
```

---

## ▶️ Usage

### Run the full pipeline
```bash
python run_pipeline.py
```

---

### Skip data collection (use existing dataset)
```bash
python run_pipeline.py --skip-collection
```

---

## ⚙️ Configuration

You can modify parameters inside `run_pipeline.py`:

- `QUERY` → search keyword (default: `"panagbenga"`)
- `PLATFORMS` → `all | facebook tiktok twitter instagram`
- `MAX_ITEMS` → number of posts per platform

---

## 📊 Output

- 📄 Dataset → `data/panagbenga-dataset.csv`  
- 📊 Results → `results.json`  
- 🌐 Dashboard → http://localhost:5000  

---

## 🧠 Pipeline Overview

1. **Data Collection** – Fetches social media data via Apify  
2. **Preprocessing** – Cleans and normalizes text  
3. **Embeddings & Clustering** – Groups similar content  
4. **Topic + Sentiment Analysis** – Extracts insights  
5. **Orchestrator** – Validates and refines outputs  
6. **Dashboard** – Visualizes results  

---

## ⚠️ Notes

- Ensure your Apify token is valid before running data collection  
- Large datasets may take time during embedding and clustering  
- Use `--skip-collection` to speed up repeated experiments  

---

## 📌 Future Improvements

- Real-time streaming data  
- Multi-language sentiment support  
- Model fine-tuning for local context  
- Cloud deployment for dashboard  

---
