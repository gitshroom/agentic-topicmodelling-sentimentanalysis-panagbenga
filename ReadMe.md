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
- 🤖 Local LLM topic explanation using Ollama with Qwen2.5:3B  
- 📊 Interactive dashboard (Flask-based)  
- 🔁 Orchestrated multi-step pipeline  

---

## 📦 Installation

Install all required Python dependencies:

```bash
pip install pandas nltk emoji beautifulsoup4 stopwordsiso sentence-transformers umap-learn hdbscan scikit-learn matplotlib numpy "bertopic[representation]" transformers torch apify-client flask sentencepiece blobfile requests
```

---

## 🧠 Install Ollama + Qwen2.5:3B

This project uses a local LLM for generating topic explanations and summaries.

### 1. Install Ollama

Visit:

https://ollama.com

Download and install Ollama for your operating system.

---

### 2. Pull the Qwen2.5:3B model

After installing Ollama, run:

```bash
ollama pull qwen2.5:3b
```

---

### 3. Start Ollama

```bash
ollama serve
```

By default, Ollama runs on:

```text
http://localhost:11434
```

---

### 4. Test the model

```bash
ollama run qwen2.5:3b
```

If successful, the model should respond interactively in the terminal.

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
- `PLATFORMS` → `all | facebook | tiktok | twitter | instagram`
- `MAX_ITEMS` → number of posts per platform
- `OLLAMA_MODEL` → local LLM model (default: `qwen2.5:3b`)

---

## 📊 Output

- 📄 Dataset → `data/panagbenga-dataset.csv`  
- 📊 Results → `results.json`  
- 🌐 Dashboard → `http://localhost:5000`

---

## 🧠 Pipeline Overview

1. **Data Collection** – Fetches social media data via Apify  
2. **Preprocessing** – Cleans and normalizes text  
3. **Embeddings & Clustering** – Groups similar content  
4. **Topic Modeling** – Extracts key discussion themes  
5. **Sentiment Analysis** – Identifies public sentiment  
6. **LLM Explainer Agent** – Generates human-readable topic descriptions using Ollama + Qwen  
7. **Orchestrator Agent** – Validates and refines outputs  
8. **Dashboard** – Visualizes results interactively  

---

## ⚠️ Notes

- Ensure your Apify token is valid before running data collection  
- Make sure Ollama is running before executing the pipeline  
- Large datasets may take time during embedding and clustering  
- Use `--skip-collection` to speed up repeated experiments  
- The first run of Qwen2.5:3B may take longer while the model loads into memory  

---

## 📌 Future Improvements

- Real-time streaming data  
- Multi-language sentiment support  
- Fine-tuned local LLM for Baguio/Panagbenga context  
- Cloud deployment for dashboard  
- Automated hyperparameter optimization for topic modeling  
