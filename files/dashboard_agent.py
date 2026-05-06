"""
dashboard_agent.py
Reads results.json and serves an interactive dashboard at http://localhost:5000
Displays: year timeline → per-year topic breakdown → per-topic sentiment
Run: python dashboard_agent.py
"""

import json
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
<title>Panagbenga Analysis · 2013–2026</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:#0b0f0e; --surface:#131918; --card:#1a2120; --border:#2a3533;
    --accent:#4dffa0; --accent2:#ff6b6b; --accent3:#ffd166; --accent4:#7eb8ff;
    --text:#e8f0ef; --muted:#6b8a86;
    --pos:#4dffa0; --neg:#ff6b6b; --neu:#ffd166;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{font-family:'DM Mono',monospace;background:var(--bg);color:var(--text);min-height:100vh}
  body::before{
    content:'';position:fixed;inset:0;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.03) 2px,rgba(0,0,0,.03) 4px);
    pointer-events:none;z-index:1000
  }

  /* ── Header ── */
  header{
    border-bottom:1px solid var(--border);padding:1.6rem 2.5rem;
    display:flex;align-items:flex-end;justify-content:space-between;
    position:sticky;top:0;background:rgba(11,15,14,.94);
    backdrop-filter:blur(14px);z-index:100
  }
  .logo{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;
        letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
  .logo span{color:var(--text)}
  .pill{font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;
        padding:.28rem .8rem;border-radius:100px;border:1px solid}
  .pill.pass{border-color:var(--pos);color:var(--pos);background:rgba(77,255,160,.08)}
  .pill.fail{border-color:var(--neg);color:var(--neg);background:rgba(255,107,107,.08)}

  main{padding:2.5rem;max-width:1500px;margin:0 auto}

  /* ── Summary strip ── */
  .summary-grid{
    display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-bottom:2.5rem
  }
  .stat-card{
    background:var(--card);border:1px solid var(--border);border-radius:4px;
    padding:1.2rem 1.4rem;position:relative;overflow:hidden;animation:fadeUp .4s ease both
  }
  .stat-card::after{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent)}
  .stat-card:nth-child(2)::after{background:var(--accent3)}
  .stat-card:nth-child(3)::after{background:var(--accent2)}
  .stat-card:nth-child(4)::after{background:var(--accent4)}
  .stat-card:nth-child(5)::after{background:#c084fc}
  .stat-label{font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem}
  .stat-value{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;line-height:1}

  /* ── Validation ── */
  .val-block{
    border-radius:4px;padding:1rem 1.4rem;margin-bottom:2rem;
    background:rgba(255,107,107,.06);border:1px solid rgba(255,107,107,.25);
    animation:fadeUp .4s ease .1s both
  }
  .val-block.pass{background:rgba(77,255,160,.06);border-color:rgba(77,255,160,.25)}
  .val-title{font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem}
  .val-issue{font-size:.78rem;color:var(--neg);margin-top:.25rem}
  .val-issue::before{content:'→ '}

  /* ── Section title ── */
  .sec-title{
    font-family:'Syne',sans-serif;font-size:.72rem;font-weight:700;
    letter-spacing:.2em;text-transform:uppercase;color:var(--muted);
    margin-bottom:1.2rem;display:flex;align-items:center;gap:.8rem
  }
  .sec-title::after{content:'';flex:1;height:1px;background:var(--border)}

  /* ── Year timeline ── */
  .timeline{
    display:flex;gap:.5rem;overflow-x:auto;padding-bottom:.8rem;
    margin-bottom:2rem;scrollbar-width:thin;scrollbar-color:var(--border) transparent
  }
  .yr-btn{
    flex-shrink:0;font-family:'DM Mono',monospace;font-size:.72rem;
    padding:.45rem 1rem;border-radius:3px;cursor:pointer;
    border:1px solid var(--border);background:var(--surface);color:var(--muted);
    transition:all .15s
  }
  .yr-btn:hover{border-color:var(--accent);color:var(--accent)}
  .yr-btn.active{background:var(--accent);color:var(--bg);border-color:var(--accent);font-weight:700}
  .yr-btn .yr-cnt{font-size:.6rem;opacity:.7;display:block;margin-top:.1rem}

  /* ── Year panel ── */
  .year-panel{display:none}
  .year-panel.visible{display:block;animation:fadeUp .35s ease both}

  .year-header{
    display:flex;align-items:center;gap:1.5rem;margin-bottom:1.5rem;flex-wrap:wrap
  }
  .year-big{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;color:var(--accent);line-height:1}
  .year-meta{display:flex;gap:1.2rem;flex-wrap:wrap}
  .meta-tag{font-size:.68rem;color:var(--muted);padding:.25rem .7rem;
            border:1px solid var(--border);border-radius:100px}
  .meta-tag span{color:var(--text);margin-left:.3rem}

  /* ── Sentiment bar (year level) ── */
  .year-sent-bar{
    display:flex;height:8px;border-radius:2px;overflow:hidden;
    margin-bottom:1.5rem;width:100%;max-width:500px
  }
  .ysb-pos{background:var(--pos);transition:width .7s ease}
  .ysb-neg{background:var(--neg);transition:width .7s ease}

  /* ── Cluster cards ── */
  .cluster-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:1.2rem}
  .cluster-card{
    background:var(--card);border:1px solid var(--border);border-radius:4px;
    overflow:hidden;animation:fadeUp .35s ease both
  }
  .cc-header{
    padding:.9rem 1.3rem;border-bottom:1px solid var(--border);
    display:flex;justify-content:space-between;align-items:center
  }
  .cc-title{font-family:'Syne',sans-serif;font-size:.78rem;font-weight:700;
            letter-spacing:.1em;text-transform:uppercase}
  .cc-title .cnum{color:var(--accent);font-size:1rem}
  .cc-meta{font-size:.65rem;color:var(--muted)}

  /* ── Topic rows ── */
  .cc-body{padding:.8rem 1.3rem}
  .topic-row{
    border:1px solid var(--border);border-radius:3px;margin-bottom:.8rem;
    padding:.8rem 1rem;transition:border-color .15s
  }
  .topic-row:hover{border-color:var(--accent3)}
  .topic-row:last-child{margin-bottom:0}

  .tr-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:.6rem}
  .tr-label{font-size:.65rem;color:var(--accent3);letter-spacing:.06em;text-transform:uppercase}
  .tr-dom{
    font-size:.6rem;padding:.18rem .55rem;border-radius:100px;border:1px solid;
    text-transform:uppercase;letter-spacing:.08em
  }
  .tr-dom.pos{border-color:var(--pos);color:var(--pos)}
  .tr-dom.neg{border-color:var(--neg);color:var(--neg)}
  .tr-dom.unknown{border-color:var(--muted);color:var(--muted)}

  /* mini sentiment bars */
  .mini-bars{margin-bottom:.6rem}
  .mb-row{display:flex;align-items:center;gap:.6rem;margin-bottom:.3rem}
  .mb-lbl{font-size:.6rem;color:var(--muted);width:52px;flex-shrink:0;text-transform:uppercase}
  .mb-track{flex:1;height:4px;background:var(--border);border-radius:0;overflow:hidden}
  .mb-fill{height:100%;border-radius:0;transition:width .7s ease}
  .mb-fill.p{background:var(--pos)}
  .mb-fill.n{background:var(--neg)}
  .mb-pct{font-size:.6rem;color:var(--muted);width:32px;text-align:right;flex-shrink:0}

  .tr-conf{font-size:.62rem;color:var(--muted);margin-bottom:.6rem}
  .tr-conf span{color:var(--text);margin-left:.3rem}

  /* word chips */
  .word-chips{display:flex;flex-wrap:wrap;gap:.25rem}
  .chip{
    font-size:.58rem;padding:.12rem .45rem;
    background:var(--surface);border:1px solid var(--border);
    border-radius:2px;color:var(--text);transition:border-color .12s,color .12s
  }
  .chip:hover{border-color:var(--accent);color:var(--accent)}
  .chip.hot{border-color:rgba(77,255,160,.35)}
  .no-topics{font-size:.72rem;color:var(--muted);padding:.5rem 0}

  /* ── Model footer ── */
  .model-footer{
    margin-top:3rem;padding-top:1.5rem;border-top:1px solid var(--border);
    display:flex;gap:2rem;flex-wrap:wrap;animation:fadeUp .4s ease .2s both
  }
  .mf{font-size:.68rem}
  .mk{color:var(--muted);margin-right:.4rem}

  @keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}

  @media(max-width:900px){
    .summary-grid{grid-template-columns:repeat(2,1fr)}
    .cluster-grid{grid-template-columns:1fr}
    header{flex-direction:column;align-items:flex-start;gap:.6rem}
  }
</style>
</head>
<body>
<header>
  <div class="logo">Panagbenga<span> Analysis · 2013–2026</span></div>
  <div id="status-pill" class="pill"></div>
</header>
<main>
  <div class="summary-grid" id="summary-grid"></div>
  <div id="val-block"></div>
  <div class="sec-title">Year Timeline</div>
  <div class="timeline" id="timeline"></div>
  <div id="year-panels"></div>
  <div class="model-footer" id="model-footer"></div>
</main>

<script>
let globalData = null;

fetch('/data')
  .then(r => r.json())
  .then(d => { globalData = d; render(d); });

function render(d) {
  /* pill */
  const pill = document.getElementById('status-pill');
  pill.textContent = d.quality_passed ? 'Quality passed' : 'Quality check failed';
  pill.className   = 'pill ' + (d.quality_passed ? 'pass' : 'fail');

  /* summary */
  const s = d.summary;
  const stats = [
    { label:'Year range',    value: s.year_range        },
    { label:'Total years',   value: s.total_years       },
    { label:'Total docs',    value: s.total_docs        },
    { label:'Total clusters',value: s.total_clusters    },
    { label:'Total topics',  value: s.total_topics      },
  ];
  document.getElementById('summary-grid').innerHTML =
    stats.map((st,i) => `
      <div class="stat-card" style="animation-delay:${i*.06}s">
        <div class="stat-label">${st.label}</div>
        <div class="stat-value">${st.value}</div>
      </div>`).join('');

  /* validation */
  const vb = document.getElementById('val-block');
  const issues = [...(d.validation.topic_issues||[]), ...(d.validation.sentiment_issues||[])];
  if (issues.length) {
    vb.className = 'val-block' + (d.quality_passed ? ' pass' : '');
    vb.innerHTML = `<div class="val-title">Validation issues</div>` +
      issues.map(i => `<div class="val-issue">${i}</div>`).join('');
  }

  /* timeline buttons + year panels */
  const tl  = document.getElementById('timeline');
  const pnl = document.getElementById('year-panels');
  const years = [...d.years].sort((a,b) => a.year - b.year);

  tl.innerHTML  = '';
  pnl.innerHTML = '';

  years.forEach((yr, idx) => {
    // button
    const btn = document.createElement('button');
    btn.className = 'yr-btn' + (idx === 0 ? ' active' : '');
    btn.innerHTML = `${yr.year}<span class="yr-cnt">${yr.total_docs} docs</span>`;
    btn.onclick = () => selectYear(yr.year);
    tl.appendChild(btn);

    // panel
    const panel = document.createElement('div');
    panel.className = 'year-panel' + (idx === 0 ? ' visible' : '');
    panel.id = `yr-${yr.year}`;
    panel.innerHTML = buildYearPanel(yr);
    pnl.appendChild(panel);
  });

  /* model footer */
  const m = d.models;
  document.getElementById('model-footer').innerHTML = `
    <div class="mf"><span class="mk">Model type</span>${m.topic_model_type}</div>
    <div class="mf"><span class="mk">Sentiment model</span>${m.sentiment_model}</div>
    <div class="mf"><span class="mk">min_topic_size</span>${m.topic_parameters?.min_topic_size ?? '—'}</div>
    <div class="mf"><span class="mk">nr_topics</span>${m.topic_parameters?.nr_topics ?? '—'}</div>
    <div class="mf"><span class="mk">Generated</span>${d.generated_at}</div>
  `;
}

function selectYear(year) {
  document.querySelectorAll('.yr-btn').forEach((b,i) => {
    b.classList.toggle('active', parseInt(b.textContent) === year);
  });
  document.querySelectorAll('.year-panel').forEach(p => {
    p.classList.toggle('visible', p.id === `yr-${year}`);
  });
}

function buildYearPanel(yr) {
  /* year-level sent bar */
  const dist = yr.dominant_sentiment_dist || {};
  const pos  = dist['POSITIVE'] || 0;
  const neg  = dist['NEGATIVE'] || 0;
  const tot  = pos + neg || 1;
  const posPct = (pos/tot*100).toFixed(0);
  const negPct = (neg/tot*100).toFixed(0);

  const headerHTML = `
    <div class="year-header">
      <div class="year-big">${yr.year}</div>
      <div class="year-meta">
        <div class="meta-tag">docs <span>${yr.total_docs}</span></div>
        <div class="meta-tag">clusters <span>${yr.total_clusters}</span></div>
        <div class="meta-tag">topics <span>${yr.total_topics}</span></div>
        <div class="meta-tag">pos <span style="color:var(--pos)">${posPct}%</span></div>
        <div class="meta-tag">neg <span style="color:var(--neg)">${negPct}%</span></div>
      </div>
    </div>
    <div class="year-sent-bar">
      <div class="ysb-pos" style="width:${posPct}%"></div>
      <div class="ysb-neg" style="width:${negPct}%"></div>
    </div>`;

  const clusters = [...(yr.clusters || [])].sort((a,b) => a.cluster_id - b.cluster_id);
  const clustersHTML = clusters.length === 0
    ? `<div style="font-size:.8rem;color:var(--muted)">No clusters found for this year.</div>`
    : `<div class="sec-title">Clusters</div>
       <div class="cluster-grid">${clusters.map(buildClusterCard).join('')}</div>`;

  return headerHTML + clustersHTML;
}

function buildClusterCard(c) {
  const topics = c.topics || [];
  const topicsHTML = topics.length === 0
    ? `<div class="no-topics">No topics extracted.</div>`
    : topics.map(buildTopicRow).join('');

  return `
    <div class="cluster-card">
      <div class="cc-header">
        <div class="cc-title">Cluster <span class="cnum">${c.cluster_id}</span></div>
        <div class="cc-meta">${c.n_docs ?? '?'} docs · ${topics.length} topics</div>
      </div>
      <div class="cc-body">${topicsHTML}</div>
    </div>`;
}

function buildTopicRow(t) {
  const s       = t.sentiment || {};
  const dom     = (s.dominant || 'UNKNOWN').toUpperCase();
  const domCls  = dom === 'POSITIVE' ? 'pos' : dom === 'NEGATIVE' ? 'neg' : 'unknown';
  const ratios  = s.label_ratios || {};
  const posPct  = ((ratios['POSITIVE'] || 0) * 100).toFixed(0);
  const negPct  = ((ratios['NEGATIVE'] || 0) * 100).toFixed(0);

  const words   = t.top_words || [];
  const scores  = t.top_word_scores || [];
  const maxSc   = Math.max(...scores, 1);
  const chipsHTML = words.map((w,i) => {
    const heat = (scores[i] || 0) / maxSc;
    return `<span class="chip ${heat > .6 ? 'hot' : ''}">${w}</span>`;
  }).join('');

  return `
    <div class="topic-row">
      <div class="tr-head">
        <div class="tr-label">topic ${t.topic_id} · ${t.count ?? '?'} docs</div>
        <span class="tr-dom ${domCls}">${dom}</span>
      </div>
      <div class="mini-bars">
        <div class="mb-row">
          <span class="mb-lbl">positive</span>
          <div class="mb-track"><div class="mb-fill p" style="width:${posPct}%"></div></div>
          <span class="mb-pct">${posPct}%</span>
        </div>
        <div class="mb-row">
          <span class="mb-lbl">negative</span>
          <div class="mb-track"><div class="mb-fill n" style="width:${negPct}%"></div></div>
          <span class="mb-pct">${negPct}%</span>
        </div>
      </div>
      <div class="tr-conf">
        avg confidence<span>${s.avg_confidence?.toFixed(3) ?? '—'}</span>
        &nbsp;&nbsp;coverage<span>${((s.coverage||0)*100).toFixed(0)}%</span>
      </div>
      <div class="word-chips">${chipsHTML}</div>
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
    path = Path(RESULTS_FILE)
    if not path.exists():
        print(f"[WARNING] {RESULTS_FILE} not found. Run the pipeline first.")
    else:
        print(f"[INFO] Loaded {RESULTS_FILE}")
    print("[INFO] Dashboard → http://localhost:5000")
    app.run(debug=False, port=5000)
