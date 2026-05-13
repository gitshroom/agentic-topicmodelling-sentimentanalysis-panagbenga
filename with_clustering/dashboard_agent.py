"""
dashboard_agent.py
Pipeline 2 (with_clustering) — Flask dashboard.

Shows three tabs:
  1. Top 10 Topics    — flattened across clusters, sorted by count
  2. Clusters         — per-year cluster cards with coherence score
                        and avg sentiment confidence
  3. Timeline         — year cards + sentiment ratio bars

Run:  python dashboard_agent.py
URL:  http://localhost:<DASHBOARD_PORT>  (configured in config.py)
"""

import json
import os
from pathlib import Path

from flask import Flask, render_template_string, jsonify, send_from_directory

import config

app = Flask(__name__)

# Prefer the explained file when available, fall back to the raw results.
RESULTS_FILES = [config.EXPLAINED_OUTPUT_FILE, config.FINAL_OUTPUT_FILE]


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Panagbenga · Pipeline 2 (with clustering)</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#0b0f0e;--surface:#131918;--card:#1a2120;--border:#2a3533;--accent:#4dffa0;--accent2:#ff6b6b;--accent3:#ffd166;--accent4:#7eb8ff;--text:#e8f0ef;--muted:#6b8a86;--pos:#4dffa0;--neg:#ff6b6b;--neu:#ffd166}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:'DM Mono',monospace;background:var(--bg);color:var(--text);min-height:100vh}
header{border-bottom:1px solid var(--border);padding:1.2rem 2rem;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;background:rgba(11,15,14,.94);backdrop-filter:blur(12px);z-index:100}
.logo{font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.logo span{color:var(--text)}
.subtitle{font-size:.55rem;color:var(--muted);letter-spacing:.15em;text-transform:uppercase}
.tab-bar{display:flex;border-bottom:1px solid var(--border);padding:0 2rem;background:var(--surface)}
.tab-btn{font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;padding:.8rem 1.4rem;background:none;border:none;border-bottom:2px solid transparent;color:var(--muted);cursor:pointer;transition:all .2s}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-btn:hover:not(.active){color:var(--text)}
main{padding:2rem;max-width:1400px;margin:0 auto}
.summary-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:.8rem;margin-bottom:2rem}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:1rem 1.2rem;position:relative;overflow:hidden}
.stat-card::after{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent)}
.stat-card:nth-child(2)::after{background:var(--accent3)}
.stat-card:nth-child(3)::after{background:var(--accent2)}
.stat-card:nth-child(4)::after{background:var(--accent4)}
.stat-card:nth-child(5)::after{background:var(--accent)}
.stat-label{font-size:.55rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:.3rem}
.stat-value{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;line-height:1}
.year-nav{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:1.4rem}
.year-btn{font-size:.65rem;padding:.3rem .75rem;background:var(--card);border:1px solid var(--border);border-radius:2px;color:var(--muted);cursor:pointer;font-family:'DM Mono',monospace}
.year-btn.active,.year-btn:hover{border-color:var(--accent);color:var(--accent)}
.section-title{font-family:'Syne',sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem;display:flex;align-items:center;gap:.6rem}
.section-title::after{content:'';flex:1;height:1px;background:var(--border)}
.top10-list{display:flex;flex-direction:column;gap:.55rem;margin-bottom:2rem}
.topic-row{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:1rem 1.4rem;display:grid;grid-template-columns:2.5rem 1fr 4rem;gap:1rem;align-items:start;cursor:pointer}
.topic-row:hover{border-color:var(--accent)}
.topic-row.open{border-color:var(--accent3);background:rgba(255,209,102,.02)}
.rank-num{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:var(--muted);line-height:1;padding-top:.05rem}
.rank-num.top3{color:var(--accent)}
.topic-label-text{font-size:.78rem;font-weight:500;color:var(--text);margin-bottom:.35rem}
.cluster-tag{display:inline-block;font-size:.52rem;background:rgba(126,184,255,.12);border:1px solid rgba(126,184,255,.4);color:var(--accent4);padding:.05rem .35rem;border-radius:2px;margin-right:.35rem;letter-spacing:.05em;text-transform:uppercase}
.word-chips{display:flex;flex-wrap:wrap;gap:.22rem;margin-bottom:.4rem}
.chip{font-size:.57rem;padding:.1rem .4rem;background:var(--surface);border:1px solid var(--border);border-radius:2px;color:var(--muted)}
.chip.hot{border-color:rgba(77,255,160,.35);color:var(--accent)}
.topic-meta{display:flex;align-items:center;gap:.5rem;font-size:.6rem;color:var(--muted);flex-wrap:wrap}
.sbadge{font-size:.54rem;padding:.12rem .5rem;border-radius:100px;border:1px solid}
.sbadge.POSITIVE{border-color:var(--pos);color:var(--pos)}
.sbadge.NEGATIVE{border-color:var(--neg);color:var(--neg)}
.sbadge.NEUTRAL{border-color:var(--neu);color:var(--neu)}
.expl-box{margin-top:.65rem;padding:.75rem 1rem;background:rgba(77,255,160,.04);border:1px solid rgba(77,255,160,.15);border-radius:3px;font-size:.71rem;line-height:1.65;color:var(--text);display:none}
.expl-box.show{display:block}
.expl-hint{font-size:.54rem;color:var(--muted);margin-top:.25rem}
.doc-stat{text-align:right}
.doc-stat .big{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700}
.doc-stat .small{font-size:.54rem;color:var(--muted)}
.cluster-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:.8rem;margin-bottom:2rem}
.cluster-card{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:1.1rem 1.2rem}
.cluster-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.5rem}
.cluster-id{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:var(--accent4)}
.cluster-id.noise{color:var(--muted)}
.cluster-docs{font-size:.55rem;color:var(--muted)}
.metric-row{display:flex;justify-content:space-between;align-items:center;font-size:.6rem;color:var(--muted);margin-bottom:.25rem}
.metric-row b{color:var(--text);font-weight:500}
.bar-track{height:5px;background:var(--surface);border-radius:2px;overflow:hidden;margin-top:.15rem}
.bar-fill{height:100%;background:var(--accent)}
.bar-fill.coh{background:var(--accent3)}
.tl-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:.8rem}
.tl-card{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:1.2rem;cursor:pointer}
.tl-card:hover{border-color:var(--accent)}
.tl-year{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:var(--accent);margin-bottom:.3rem}
.tl-stats{font-size:.6rem;color:var(--muted);margin-bottom:.6rem}
.tl-top{font-size:.66rem;color:var(--text);margin-bottom:.5rem}
.tl-top span{color:var(--accent3)}
.mini-bar{display:flex;height:4px;overflow:hidden;gap:1px}
.viz-img{display:block;width:100%;border:1px solid var(--border);border-radius:3px;margin-top:1rem;background:#000}
.empty{text-align:center;padding:3rem;color:var(--muted);font-size:.78rem}
.tab-panel{display:none}
.tab-panel.active{display:block}
@media(max-width:768px){.summary-grid{grid-template-columns:repeat(2,1fr)}.tl-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div>
    <div class="logo">Panagbenga<span> · with clustering</span></div>
    <div class="subtitle">Pipeline 2 · year → cluster → topic → sentiment</div>
  </div>
  <span id="yr-range" style="font-size:.6rem;color:var(--muted)"></span>
</header>
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('top10',this)">★ Top 10 Topics</button>
  <button class="tab-btn" onclick="switchTab('clusters',this)">Clusters</button>
  <button class="tab-btn" onclick="switchTab('timeline',this)">Timeline</button>
</div>
<main>
  <div class="summary-grid" id="summary"></div>

  <div class="tab-panel active" id="tab-top10">
    <div class="year-nav" id="ynav-top10"></div>
    <div class="section-title" id="top10-title">Top 10 Topics</div>
    <div class="top10-list" id="top10-list"></div>
  </div>

  <div class="tab-panel" id="tab-clusters">
    <div class="year-nav" id="ynav-clusters"></div>
    <div class="section-title" id="cluster-title">Cluster overview</div>
    <div class="cluster-grid" id="cluster-grid"></div>
    <div class="section-title">UMAP visualization</div>
    <img id="viz-img" class="viz-img" alt="cluster visualization" style="display:none"/>
    <div id="viz-fallback" class="empty" style="display:none">No visualization available for this year.</div>
  </div>

  <div class="tab-panel" id="tab-timeline">
    <div class="section-title">Year overview — click a year to explore its topics</div>
    <div class="tl-grid" id="tl-grid"></div>
  </div>
</main>

<script>
let D=null;

fetch('/data').then(r=>r.json()).then(d=>{D=d;boot(d)});

function boot(d){
  document.getElementById('yr-range').textContent=d.summary?.year_range||'';

  const s=d.summary||{};
  const cards=[
    {l:'Years',v:s.total_years||0},
    {l:'Posts',v:(s.total_docs||0).toLocaleString()},
    {l:'Clusters',v:s.total_clusters||0},
    {l:'Topics',v:s.total_topics||0},
    {l:'Avg confidence',v:(s.avg_confidence||0).toFixed(2)},
  ];
  document.getElementById('summary').innerHTML=cards
    .map(x=>`<div class="stat-card"><div class="stat-label">${x.l}</div><div class="stat-value">${x.v}</div></div>`).join('');

  const years=(d.years||[]).map(y=>y.year).sort((a,b)=>a-b);
  mkNav('ynav-top10', years, y=>selTop10(y));
  mkNav('ynav-clusters', years, y=>selClusters(y));
  mkTimeline(d.years||[]);

  if(years.length){
    selTop10(years[years.length-1]);
    selClusters(years[years.length-1]);
  }
}

function switchTab(name,btn){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}

function mkNav(id,years,cb){
  document.getElementById(id).innerHTML=years.map(y=>
    `<button class="year-btn" data-y="${y}" onclick="window._cbs['${id}'](${y})">${y}</button>`
  ).join('');
  window._cbs=window._cbs||{};
  window._cbs[id]=cb;
}
function setActive(navId,year){
  document.querySelectorAll('#'+navId+' .year-btn').forEach(b=>
    b.classList.toggle('active',parseInt(b.dataset.y)===year));
}

function flattenTopics(yearEntry){
  const out=[];
  for(const c of (yearEntry?.clusters||[])){
    for(const t of (c.topics||[])){
      out.push({...t, cluster_id:c.cluster_id});
    }
  }
  return out;
}

function getTop10(yearEntry){
  const all=flattenTopics(yearEntry);
  all.sort((a,b)=>(b.count||0)-(a.count||0));
  return all.slice(0,10).map((t,i)=>({...t,rank:i+1}));
}

function selTop10(year){
  setActive('ynav-top10',year);
  document.getElementById('top10-title').textContent=`Top 10 Topics — ${year}`;
  const ye=(D.years||[]).find(y=>y.year===year);
  const topics=getTop10(ye);
  const list=document.getElementById('top10-list');

  if(!topics.length){
    list.innerHTML='<div class="empty">No topics found for '+year+'.</div>';
    return;
  }

  list.innerHTML=topics.map((t,idx)=>{
    const rank=t.rank||idx+1;
    const dom=t.sentiment?.dominant||'NEUTRAL';
    const words=(t.top_words||[]).slice(0,8);
    const chips=words.map((w,wi)=>`<span class="chip ${wi<2?'hot':''}">${w}</span>`).join('');
    const hasExpl=!!t.explanation;
    const cnt=t.count||0;
    const label=t.label||`Topic ${t.topic_id}`;
    const conf=t.sentiment?.avg_confidence;
    const highConf=t.sentiment?.high_confidence_ratio;
    const cid=t.cluster_id;
    const clusterTag=cid===-1?'<span class="cluster-tag" style="color:var(--muted);border-color:var(--border)">Noise</span>'
                              :`<span class="cluster-tag">Cluster ${cid}</span>`;

    return `<div class="topic-row" id="tr-${year}-${idx}" onclick="toggle('${year}',${idx})">
      <div class="rank-num ${rank<=3?'top3':''}">#${rank}</div>
      <div>
        <div class="topic-label-text">${clusterTag}${label}</div>
        <div class="word-chips">${chips}</div>
        <div class="topic-meta">
          <span class="sbadge ${dom}">${dom}</span>
          <span>${cnt.toLocaleString()} posts</span>
          ${conf!==undefined&&conf!==null?`<span>conf ${(+conf).toFixed(2)}</span>`:''}
          ${highConf!==undefined&&highConf!==null?`<span>high-conf ${(highConf*100).toFixed(0)}%</span>`:''}
        </div>
        ${hasExpl?`
        <div class="expl-box" id="eb-${year}-${idx}">${t.explanation}</div>
        <div class="expl-hint" id="eh-${year}-${idx}">click to read plain-language explanation</div>`:''}
      </div>
      <div class="doc-stat">
        <div class="big">${cnt}</div>
        <div class="small">posts</div>
      </div>
    </div>`;
  }).join('');
}

function toggle(year,idx){
  const eb=document.getElementById(`eb-${year}-${idx}`);
  const eh=document.getElementById(`eh-${year}-${idx}`);
  const tr=document.getElementById(`tr-${year}-${idx}`);
  if(!eb) return;
  const open=eb.classList.toggle('show');
  tr.classList.toggle('open',open);
  if(eh) eh.style.opacity=open?'0':'1';
}

function selClusters(year){
  setActive('ynav-clusters',year);
  document.getElementById('cluster-title').textContent=`Clusters — ${year}`;
  const ye=(D.years||[]).find(y=>y.year===year);
  const grid=document.getElementById('cluster-grid');
  if(!ye||!(ye.clusters||[]).length){
    grid.innerHTML='<div class="empty">No clusters found for '+year+'.</div>';
  }else{
    grid.innerHTML=(ye.clusters||[]).map(c=>{
      const isNoise=c.cluster_id===-1;
      const coh=(c.coherence||0);
      const conf=c.sentiment_summary?.avg_confidence||0;
      const dom=c.sentiment_summary?.dominant_sentiment||'UNKNOWN';
      return `<div class="cluster-card">
        <div class="cluster-head">
          <span class="cluster-id ${isNoise?'noise':''}">${isNoise?'Noise':'Cluster '+c.cluster_id}</span>
          <span class="cluster-docs">${c.n_docs||0} posts · ${c.n_topics||0} topics</span>
        </div>
        <div class="metric-row"><span>Coherence (c_v)</span><b>${coh.toFixed(3)}</b></div>
        <div class="bar-track"><div class="bar-fill coh" style="width:${Math.min(100,coh*100).toFixed(0)}%"></div></div>
        <div class="metric-row" style="margin-top:.5rem"><span>Avg sentiment confidence</span><b>${conf.toFixed(3)}</b></div>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100,conf*100).toFixed(0)}%"></div></div>
        <div class="metric-row" style="margin-top:.5rem"><span>Dominant sentiment</span>
          <span class="sbadge ${dom}">${dom}</span></div>
      </div>`;
    }).join('');
  }

  const viz=document.getElementById('viz-img');
  const fallback=document.getElementById('viz-fallback');
  const vizPath=ye&&ye.visualization?ye.visualization:null;
  if(vizPath){
    viz.src='/viz/'+vizPath.split(/[\\\/]/).pop()+'?y='+year;
    viz.style.display='block';
    fallback.style.display='none';
  }else{
    viz.style.display='none';
    fallback.style.display='block';
  }
}

function mkTimeline(years){
  const grid=document.getElementById('tl-grid');
  grid.innerHTML=[...years].sort((a,b)=>a.year-b.year).map(y=>{
    const top=getTop10(y)[0];
    const topLabel=top?(top.label||(top.top_words||[]).slice(0,3).join(', ')):'—';
    const dist=y.dominant_sentiment_dist||{};
    const tot=Object.values(dist).reduce((a,b)=>a+b,0)||1;
    const pW=k=>((dist[k]||0)/tot*100).toFixed(0);
    const coh=y.avg_coherence||0;
    const conf=y.sentiment_summary?.avg_confidence||0;
    return `<div class="tl-card" onclick="goYear(${y.year})">
      <div class="tl-year">${y.year}</div>
      <div class="tl-stats">${(y.total_docs||0).toLocaleString()} posts · ${y.n_clusters||0} clusters · ${y.total_topics||0} topics</div>
      ${top?`<div class="tl-top">Top: <span>${topLabel}</span></div>`:''}
      <div class="mini-bar">
        <div style="width:${pW('POSITIVE')}%;background:var(--pos)"></div>
        <div style="width:${pW('NEGATIVE')}%;background:var(--neg)"></div>
        <div style="width:${pW('NEUTRAL')}%;background:var(--neu)"></div>
      </div>
      <div style="display:flex;gap:.7rem;margin-top:.5rem;font-size:.55rem;color:var(--muted)">
        <span>coh ${coh.toFixed(2)}</span><span>conf ${conf.toFixed(2)}</span>
      </div>
    </div>`;
  }).join('');
}

function goYear(year){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-top10').classList.add('active');
  document.querySelector('.tab-btn').classList.add('active');
  selTop10(year);
  selClusters(year);
  window.scrollTo({top:0,behavior:'smooth'});
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/data")
def data():
    for path in RESULTS_FILES:
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
    return jsonify({"error": "No results file found. Run the pipeline first."}), 404


@app.route("/viz/<path:filename>")
def viz(filename):
    """Serve cluster visualization PNGs."""
    return send_from_directory(config.VIZ_DIR, filename)


if __name__ == "__main__":
    found = next((p for p in RESULTS_FILES if Path(p).exists()), None)
    msg = ("Loading: " + found) if found else "WARNING: No results file - run the pipeline first"
    print(f"[INFO] {msg}")
    port = config.DASHBOARD_PORT
    print(f"[INFO] Dashboard -> http://localhost:{port}")
    app.run(debug=False, port=port, host="127.0.0.1")
