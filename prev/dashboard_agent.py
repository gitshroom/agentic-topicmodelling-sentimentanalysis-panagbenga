"""
dashboard_agent.py
Reads results.json and serves an interactive analysis dashboard at http://localhost:5000
Run: python dashboard_agent.py
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template_string, jsonify

app = Flask(__name__)
RESULTS_FILE = "outputs/results.json"

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Panagbenga Analysis Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #0b0f0e;
    --surface: #131918;
    --card: #1a2120;
    --border: #2a3533;
    --accent: #4dffa0;
    --accent2: #ff6b6b;
    --accent3: #ffd166;
    --text: #e8f0ef;
    --muted: #6b8a86;
    --pos: #4dffa0;
    --neg: #ff6b6b;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body {
    font-family: 'DM Mono', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* scanline overlay */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
    pointer-events: none; z-index: 1000;
  }

  header {
    border-bottom: 1px solid var(--border);
    padding: 2rem 2.5rem;
    display: flex; align-items: flex-end; justify-content: space-between;
    position: sticky; top: 0; background: rgba(11,15,14,0.92);
    backdrop-filter: blur(12px); z-index: 100;
  }
  .logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 800; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--accent);
  }
  .logo span { color: var(--text); }
  .status-pill {
    font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.3rem 0.8rem; border-radius: 100px;
    border: 1px solid;
  }
  .status-pill.fail { border-color: var(--neg); color: var(--neg); background: rgba(255,107,107,0.08); }
  .status-pill.pass { border-color: var(--pos); color: var(--pos); background: rgba(77,255,160,0.08); }

  main { padding: 2.5rem; max-width: 1400px; margin: 0 auto; }

  /* summary strip */
  .summary-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2.5rem;
  }
  .stat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 4px; padding: 1.4rem 1.6rem;
    position: relative; overflow: hidden;
    animation: fadeUp 0.5s ease both;
  }
  .stat-card::after {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; background: var(--accent);
  }
  .stat-card:nth-child(2)::after { background: var(--accent3); }
  .stat-card:nth-child(3)::after { background: var(--accent2); }
  .stat-card:nth-child(4)::after { background: #7eb8ff; }
  .stat-label { font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem; }
  .stat-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1; }

  /* validation */
  .validation-block {
    background: rgba(255,107,107,0.06); border: 1px solid rgba(255,107,107,0.25);
    border-radius: 4px; padding: 1.2rem 1.6rem; margin-bottom: 2.5rem;
    animation: fadeUp 0.5s ease 0.1s both;
  }
  .validation-block.pass { background: rgba(77,255,160,0.06); border-color: rgba(77,255,160,0.25); }
  .val-title { font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.6rem; }
  .val-issue { font-size: 0.8rem; color: var(--neg); margin-top: 0.3rem; }
  .val-issue::before { content: '→ '; }

  /* section title */
  .section-title {
    font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted);
    margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem;
  }
  .section-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

  /* cluster grid */
  .cluster-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 1.2rem; }
  .cluster-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 4px; overflow: hidden;
    animation: fadeUp 0.5s ease both;
    cursor: pointer; transition: border-color 0.2s, transform 0.15s;
  }
  .cluster-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .cluster-card.noise-cluster { border-color: var(--border); opacity: 0.6; }

  .card-header {
    padding: 1rem 1.4rem; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }
  .cluster-id {
    font-family: 'Syne', sans-serif; font-size: 0.8rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
  }
  .cluster-id .num { color: var(--accent); font-size: 1.1rem; }
  .doc-count { font-size: 0.7rem; color: var(--muted); }

  .card-body { padding: 1.2rem 1.4rem; }

  /* sentiment bar */
  .sentiment-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.2rem; }
  .sent-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); width: 60px; flex-shrink: 0; }
  .bar-track { flex: 1; height: 6px; background: var(--border); border-radius: 0; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 0; transition: width 0.8s cubic-bezier(.4,0,.2,1); }
  .bar-fill.pos { background: var(--pos); }
  .bar-fill.neg { background: var(--neg); }
  .sent-pct { font-size: 0.7rem; color: var(--muted); width: 36px; text-align: right; flex-shrink: 0; }
  .sent-dominant {
    font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.2rem 0.6rem; border-radius: 100px; border: 1px solid;
  }
  .sent-dominant.pos { border-color: var(--pos); color: var(--pos); }
  .sent-dominant.neg { border-color: var(--neg); color: var(--neg); }

  .confidence-row { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--muted); margin-bottom: 1.2rem; }

  /* topics */
  .topics-section { border-top: 1px solid var(--border); padding-top: 1rem; margin-top: 0.4rem; }
  .topic-entry { margin-bottom: 0.8rem; }
  .topic-label {
    font-size: 0.65rem; letter-spacing: 0.06em; color: var(--accent3);
    margin-bottom: 0.4rem; text-transform: uppercase;
  }
  .topic-words { display: flex; flex-wrap: wrap; gap: 0.3rem; }
  .word-chip {
    font-size: 0.62rem; padding: 0.15rem 0.5rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 2px; color: var(--text);
    transition: border-color 0.15s, color 0.15s;
  }
  .word-chip:hover { border-color: var(--accent); color: var(--accent); }
  .word-chip.hot { border-color: rgba(77,255,160,0.3); }

  .error-block { font-size: 0.7rem; color: var(--neg); padding: 0.6rem; background: rgba(255,107,107,0.06); border-radius: 2px; }

  .noise-tag { font-size: 0.62rem; color: var(--muted); letter-spacing: 0.08em; margin-left: 0.6rem; }

  /* coherence mini bar */
  .meta-row { display: flex; gap: 1rem; margin-bottom: 0.8rem; }
  .mini-stat { font-size: 0.65rem; color: var(--muted); }
  .mini-stat span { color: var(--text); margin-left: 0.3rem; }
  .mini-stat .warn { color: var(--neg); }
  .mini-stat .ok { color: var(--pos); }

  /* model info footer */
  .model-footer {
    margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
    display: flex; gap: 2rem; flex-wrap: wrap;
    animation: fadeUp 0.5s ease 0.3s both;
  }
  .model-field { font-size: 0.7rem; }
  .model-field .mkey { color: var(--muted); margin-right: 0.5rem; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @media (max-width: 900px) {
    .summary-grid { grid-template-columns: repeat(2, 1fr); }
    .cluster-grid { grid-template-columns: 1fr; }
    header { flex-direction: column; align-items: flex-start; gap: 0.8rem; }
  }
</style>
</head>
<body>
<header>
  <div class="logo">Panagbenga<span> Analysis</span></div>
  <div id="status-pill" class="status-pill"></div>
</header>
<main>
  <div class="summary-grid" id="summary-grid"></div>
  <div id="validation-block"></div>
  <div class="section-title">Cluster Analysis</div>
  <div class="cluster-grid" id="cluster-grid"></div>
  <div class="model-footer" id="model-footer"></div>
</main>

<script>
fetch('/data')
  .then(r => r.json())
  .then(data => render(data));

function render(d) {
  const passed = d.quality_passed;
  const s = d.summary;
  const pill = document.getElementById('status-pill');
  pill.textContent = passed ? 'Quality passed' : 'Quality check failed';
  pill.className = 'status-pill ' + (passed ? 'pass' : 'fail');

  // Summary stats
  const sg = document.getElementById('summary-grid');
  const stats = [
    { label: 'Clusters', value: s.total_clusters },
    { label: 'Docs clustered', value: s.total_docs_clustered },
    { label: 'Topics found', value: s.total_topics_discovered },
    { label: 'Generated', value: d.generated_at.slice(0,10) },
  ];
  sg.innerHTML = stats.map((st, i) => `
    <div class="stat-card" style="animation-delay:${i*0.06}s">
      <div class="stat-label">${st.label}</div>
      <div class="stat-value">${st.value}</div>
    </div>`).join('');

  // Validation
  const vb = document.getElementById('validation-block');
  const issues = [...(d.validation.topic_issues||[]), ...(d.validation.sentiment_issues||[])];
  if (issues.length) {
    vb.className = 'validation-block' + (passed ? ' pass' : '');
    vb.innerHTML = `<div class="val-title">Validation issues</div>` +
      issues.map(i => `<div class="val-issue">${i}</div>`).join('');
  }

  // Clusters
  const cg = document.getElementById('cluster-grid');
  
  // FIX: Sort safely as strings instead of subtracting them
  const clusters = [...d.clusters].sort((a,b) => String(a.cluster_id).localeCompare(String(b.cluster_id)));
  cg.innerHTML = clusters.map((c, idx) => renderCluster(c, idx)).join('');

  // Model info
  const m = d.models;
  document.getElementById('model-footer').innerHTML = `
    <div class="model-field"><span class="mkey">Model type</span>${m.topic_model_type}</div>
    <div class="model-field"><span class="mkey">Sentiment model</span>${m.sentiment_model}</div>
    <div class="model-field"><span class="mkey">min_topic_size</span>${m.topic_parameters?.min_topic_size ?? '—'}</div>
    <div class="model-field"><span class="mkey">nr_topics</span>${m.topic_parameters?.nr_topics ?? '—'}</div>
    <div class="model-field"><span class="mkey">Generated</span>${d.generated_at}</div>
  `;
}

function renderCluster(c, idx) {
  // FIX: Identify string-based noise clusters
  const isNoise = String(c.cluster_id).endsWith('_-1') || c.cluster_id === -1 || c.cluster_id === '-1';
  
  const sent = c.sentiment || {};
  const topics = c.topics || {};
  const domPos = sent.dominant === 'POSITIVE';
  const posRatio = (sent.label_ratios?.POSITIVE || 0) * 100;
  const negRatio = (sent.label_ratios?.NEGATIVE || 0) * 100;
  const delay = idx * 0.05;

  const noiseRatioVal = topics.noise_ratio != null ? (topics.noise_ratio * 100).toFixed(0) + '%' : '—';
  const coherenceVal = topics.coherence_proxy != null ? topics.coherence_proxy.toFixed(3) : '—';
  const coherenceWarn = topics.coherence_proxy != null && topics.coherence_proxy < 0.3;

  const topicHTML = topics.error
    ? `<div class="error-block">${topics.error}</div>`
    : (topics.top_topics || []).length === 0
      ? `<div style="font-size:0.7rem;color:var(--muted)">No topics extracted</div>`
      : topics.top_topics.map((t, ti) => {
          
          // FIX: Handle both BERTopic (has scores) and Agentic LLM (no scores)
          const hasScores = t.top_word_scores && t.top_word_scores.length > 0;
          const maxScore = hasScores ? Math.max(...t.top_word_scores) : 1;
          
          const words = (t.top_words || []).map((w, wi) => {
            const heat = hasScores ? (t.top_word_scores[wi] / maxScore) : 0.8;
            return `<span class="word-chip ${heat > 0.6 ? 'hot' : ''}">${w}</span>`;
          }).join('');
          
          // FIX: Use LLM-generated label if it exists, otherwise fallback to topic ID
          const displayLabel = t.label ? t.label : `topic ${t.topic_id}`;
          
          return `
            <div class="topic-entry">
              <div class="topic-label">${displayLabel} · ${t.count || 0} docs</div>
              <div class="topic-words">${words}</div>
            </div>`;
        }).join('');

  return `
    <div class="cluster-card ${isNoise ? 'noise-cluster' : ''}" style="animation-delay:${delay}s">
      <div class="card-header">
        <div class="cluster-id">
          Cluster <span class="num">${isNoise ? 'noise' : c.cluster_id}</span>
          ${isNoise ? '<span class="noise-tag">unassigned</span>' : ''}
        </div>
        <div style="display:flex;align-items:center;gap:0.6rem">
          <span class="sent-dominant ${domPos ? 'pos' : 'neg'}">${sent.dominant || 'N/A'}</span>
          <span class="doc-count">${c.n_docs || 0} docs</span>
        </div>
      </div>
      <div class="card-body">
        <div class="sentiment-row">
          <span class="sent-label">positive</span>
          <div class="bar-track"><div class="bar-fill pos" style="width:${posRatio}%"></div></div>
          <span class="sent-pct">${posRatio.toFixed(0)}%</span>
        </div>
        <div class="sentiment-row">
          <span class="sent-label">negative</span>
          <div class="bar-track"><div class="bar-fill neg" style="width:${negRatio}%"></div></div>
          <span class="sent-pct">${negRatio.toFixed(0)}%</span>
        </div>
        <div class="confidence-row">
          <span>avg confidence <strong>${sent.avg_confidence?.toFixed(3) ?? '—'}</strong></span>
          <span>coverage <strong>${((sent.coverage || 0)*100).toFixed(0)}%</strong></span>
        </div>
        ${!isNoise ? `
        <div class="meta-row">
          <div class="mini-stat">noise ratio <span>${noiseRatioVal}</span></div>
          <div class="mini-stat">coherence <span class="${coherenceWarn ? 'warn' : 'ok'}">${coherenceVal}</span></div>
          <div class="mini-stat">n_topics <span>${topics.n_topics || 0}</span></div>
        </div>` : ''}
        <div class="topics-section">${topicHTML}</div>
      </div>
    </div>`;
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/data")
def data():
    path = Path(RESULTS_FILE)
    if not path.exists():
        return jsonify({"error": f"{RESULTS_FILE} not found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    if not Path(RESULTS_FILE).exists():
        print(f"[WARNING] {RESULTS_FILE} not found in current directory.")
        print(f"          Place results.json in the same folder and rerun.")
    else:
        print(f"[INFO] Loaded {RESULTS_FILE}")
    print("[INFO] Dashboard running at http://localhost:5000")
    app.run(debug=False, port=5000)