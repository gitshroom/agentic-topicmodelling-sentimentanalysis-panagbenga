"""
dashboard_agent.py  (updated)
Top-10 Topics per Year + Plain-Language Explainer + Cluster + Timeline tabs
Run: python dashboard_agent.py  →  http://localhost:5000
"""

import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify

app = Flask(__name__)
RESULTS_FILES = ["outputs/explained_results.json", "outputs/results.json"]

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Panagbenga Festival Analysis · 2013–2026</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#0b0f0e;--surface:#131918;--card:#1a2120;--border:#2a3533;--accent:#4dffa0;--accent2:#ff6b6b;--accent3:#ffd166;--accent4:#7eb8ff;--text:#e8f0ef;--muted:#6b8a86;--pos:#4dffa0;--neg:#ff6b6b;--neu:#ffd166}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:'DM Mono',monospace;background:var(--bg);color:var(--text);min-height:100vh}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.03) 2px,rgba(0,0,0,.03) 4px);pointer-events:none;z-index:1000}
header{border-bottom:1px solid var(--border);padding:1.2rem 2rem;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;background:rgba(11,15,14,.94);backdrop-filter:blur(12px);z-index:100}
.logo{font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.logo span{color:var(--text)}
.status-pill{font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;padding:.25rem .7rem;border-radius:100px;border:1px solid}
.status-pill.pass{border-color:var(--pos);color:var(--pos);background:rgba(77,255,160,.08)}
.status-pill.fail{border-color:var(--neg);color:var(--neg);background:rgba(255,107,107,.08)}
.tab-bar{display:flex;border-bottom:1px solid var(--border);padding:0 2rem;background:var(--surface)}
.tab-btn{font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;padding:.8rem 1.4rem;background:none;border:none;border-bottom:2px solid transparent;color:var(--muted);cursor:pointer;transition:all .2s}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-btn:hover:not(.active){color:var(--text)}
main{padding:2rem;max-width:1400px;margin:0 auto}
.summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:2rem}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:1.2rem 1.4rem;position:relative;overflow:hidden;animation:fadeUp .4s ease both}
.stat-card::after{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent)}
.stat-card:nth-child(2)::after{background:var(--accent3)}
.stat-card:nth-child(3)::after{background:var(--accent2)}
.stat-card:nth-child(4)::after{background:var(--accent4)}
.stat-label{font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem}
.stat-value{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;line-height:1}
.year-nav{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:1.4rem}
.year-btn{font-size:.65rem;padding:.3rem .75rem;background:var(--card);border:1px solid var(--border);border-radius:2px;color:var(--muted);cursor:pointer;transition:all .15s;font-family:'DM Mono',monospace}
.year-btn.active,.year-btn:hover{border-color:var(--accent);color:var(--accent)}
.section-title{font-family:'Syne',sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem;display:flex;align-items:center;gap:.6rem}
.section-title::after{content:'';flex:1;height:1px;background:var(--border)}
/* Top-10 */
.top10-list{display:flex;flex-direction:column;gap:.55rem;margin-bottom:2rem}
.topic-row{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:1rem 1.4rem;display:grid;grid-template-columns:2.5rem 1fr 3.5rem;gap:1rem;align-items:start;cursor:pointer;transition:border-color .15s,background .15s;animation:fadeUp .35s ease both}
.topic-row:hover{border-color:var(--accent)}
.topic-row.open{border-color:var(--accent3);background:rgba(255,209,102,.02)}
.rank-num{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:var(--muted);line-height:1;padding-top:.05rem}
.rank-num.top3{color:var(--accent)}
.topic-label-text{font-size:.75rem;font-weight:500;color:var(--text);margin-bottom:.35rem}
.word-chips{display:flex;flex-wrap:wrap;gap:.22rem;margin-bottom:.4rem}
.chip{font-size:.57rem;padding:.1rem .4rem;background:var(--surface);border:1px solid var(--border);border-radius:2px;color:var(--muted)}
.chip.hot{border-color:rgba(77,255,160,.35);color:var(--accent)}
.topic-meta{display:flex;align-items:center;gap:.5rem;font-size:.58rem;color:var(--muted)}
.sbadge{font-size:.54rem;padding:.12rem .5rem;border-radius:100px;border:1px solid}
.sbadge.POSITIVE{border-color:var(--pos);color:var(--pos)}
.sbadge.NEGATIVE{border-color:var(--neg);color:var(--neg)}
.sbadge.NEUTRAL{border-color:var(--neu);color:var(--neu)}
.expl-box{margin-top:.65rem;padding:.75rem 1rem;background:rgba(77,255,160,.04);border:1px solid rgba(77,255,160,.15);border-radius:3px;font-size:.71rem;line-height:1.65;color:var(--text);display:none}
.expl-box.show{display:block;animation:fadeUp .25s ease}
.expl-hint{font-size:.54rem;color:var(--muted);margin-top:.25rem;display:flex;align-items:center;gap:.3rem}
.doc-stat{text-align:right}
.doc-stat .big{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700}
.doc-stat .small{font-size:.54rem;color:var(--muted)}
/* Clusters */
.cluster-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:1rem}
.cluster-card{background:var(--card);border:1px solid var(--border);border-radius:4px;overflow:hidden;animation:fadeUp .35s ease both}
.c-header{padding:.8rem 1.2rem;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.c-body{padding:1rem 1.2rem}
.bar-row{display:flex;align-items:center;gap:.55rem;margin-bottom:.55rem}
.bar-lbl{font-size:.57rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);width:50px;flex-shrink:0}
.bar-track{flex:1;height:5px;background:var(--border);overflow:hidden}
.bar-fill{height:100%}
.bar-fill.p{background:var(--pos)}
.bar-fill.n{background:var(--neg)}
.bar-fill.u{background:var(--neu)}
.bar-pct{font-size:.58rem;color:var(--muted);width:26px;text-align:right;flex-shrink:0}
/* Timeline */
.tl-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:.8rem}
.tl-card{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:1.2rem;cursor:pointer;transition:border-color .15s;animation:fadeUp .35s ease both}
.tl-card:hover{border-color:var(--accent)}
.tl-year{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:var(--accent);margin-bottom:.3rem}
.tl-stats{font-size:.6rem;color:var(--muted);margin-bottom:.6rem}
.tl-top{font-size:.66rem;color:var(--text);margin-bottom:.5rem}
.tl-top span{color:var(--accent3)}
.mini-bar{display:flex;height:4px;overflow:hidden;gap:1px}
.tab-panel{display:none}
.tab-panel.active{display:block}
.empty{text-align:center;padding:3rem;color:var(--muted);font-size:.78rem}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@media(max-width:768px){.summary-grid{grid-template-columns:repeat(2,1fr)}.cluster-grid,.tl-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div class="logo">Panagbenga<span> Analysis</span></div>
  <div style="display:flex;align-items:center;gap:1rem">
    <span id="yr-range" style="font-size:.6rem;color:var(--muted)"></span>
    <div class="status-pill" id="status-pill"></div>
  </div>
</header>
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('top10',this)">&#9733; Top 10 Topics</button>
  <button class="tab-btn" onclick="switchTab('clusters',this)">Clusters</button>
  <button class="tab-btn" onclick="switchTab('timeline',this)">Timeline</button>
</div>
<main>
  <div class="summary-grid" id="summary"></div>

  <!-- TOP-10 -->
  <div class="tab-panel active" id="tab-top10">
    <div class="year-nav" id="ynav-top10"></div>
    <div class="section-title" id="top10-title">Top 10 Topics</div>
    <div class="top10-list" id="top10-list"></div>
  </div>

  <!-- CLUSTERS -->
  <div class="tab-panel" id="tab-clusters">
    <div class="year-nav" id="ynav-clusters"></div>
    <div class="cluster-grid" id="cluster-grid"></div>
  </div>

  <!-- TIMELINE -->
  <div class="tab-panel" id="tab-timeline">
    <div class="section-title">Year Overview — click a year to explore its topics</div>
    <div class="tl-grid" id="tl-grid"></div>
  </div>
</main>

<script>
let D=null, CY_TOP10=null, CY_CLUST=null;

fetch('/data').then(r=>r.json()).then(d=>{D=d;boot(d)});

function boot(d){
  // Status
  const p=document.getElementById('status-pill');
  p.textContent=d.quality_passed?'Quality passed':'Quality check failed';
  p.className='status-pill '+(d.quality_passed?'pass':'fail');
  document.getElementById('yr-range').textContent=d.summary?.year_range||'';

  // Summary
  const s=d.summary||{};
  document.getElementById('summary').innerHTML=
    [{l:'Years',v:s.total_years||0},{l:'Posts',v:(s.total_docs||0).toLocaleString()},
     {l:'Topics',v:s.total_topics||0},{l:'Clusters',v:s.total_clusters||0}]
    .map((x,i)=>`<div class="stat-card" style="animation-delay:${i*.06}s"><div class="stat-label">${x.l}</div><div class="stat-value">${x.v}</div></div>`).join('');

  const years=(d.years||[]).map(y=>y.year).sort((a,b)=>a-b);
  mkNav('ynav-top10',   years, y=>selTop10(y));
  mkNav('ynav-clusters',years, y=>selClusters(y));
  mkTimeline(d.years||[]);

  if(years.length){
    selTop10(years[years.length-1]);
    selClusters(years[years.length-1]);
  }
}

/* ── Tab ── */
function switchTab(name,btn){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}

/* ── Year nav ── */
function mkNav(id,years,cb){
  document.getElementById(id).innerHTML=years.map(y=>
    `<button class="year-btn" data-y="${y}" onclick="(${cb.toString()})(${y})">${y}</button>`
  ).join('');
}
function setActive(navId,year){
  document.querySelectorAll('#'+navId+' .year-btn').forEach(b=>
    b.classList.toggle('active',parseInt(b.dataset.y)===year));
}

/* ── Get top-10 topics for a year ── */
function getTop10(yearEntry){
  if(yearEntry?.top_10_topics?.length) return yearEntry.top_10_topics;
  const all=[];
  for(const c of yearEntry?.clusters||[]){
    if(c.cluster_id===-1) continue;
    for(const t of c.topics||[]) all.push({...t,cluster_id:c.cluster_id});
  }
  all.sort((a,b)=>(b.count||0)-(a.count||0));
  return all.slice(0,10).map((t,i)=>({...t,rank:i+1}));
}

/* ── TOP-10 renderer ── */
function selTop10(year){
  CY_TOP10=year;
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
    const expl=t.explanation||'';
    const cnt=t.count||0;
    const label=t.label||`Topic ${t.topic_id}`;
    const conf=t.sentiment?.avg_confidence;

    return `<div class="topic-row" id="tr-${year}-${idx}" onclick="toggle('${year}',${idx})" style="animation-delay:${idx*.04}s">
      <div class="rank-num ${rank<=3?'top3':''}">#${rank}</div>
      <div>
        <div class="topic-label-text">${label}</div>
        <div class="word-chips">${chips}</div>
        <div class="topic-meta">
          <span class="sbadge ${dom}">${dom}</span>
          <span>${cnt.toLocaleString()} posts</span>
          ${conf?`<span>conf ${conf.toFixed(2)}</span>`:''}
        </div>
        ${hasExpl?`
        <div class="expl-box" id="eb-${year}-${idx}">💬 ${expl}</div>
        <div class="expl-hint" id="eh-${year}-${idx}">▾ click to read plain-language explanation</div>`:''}
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

/* ── CLUSTERS renderer ── */
function selClusters(year){
  CY_CLUST=year;
  setActive('ynav-clusters',year);
  const ye=(D.years||[]).find(y=>y.year===year);
  const grid=document.getElementById('cluster-grid');
  const clusters=(ye?.clusters||[]).filter(c=>c.cluster_id!==-1);
  if(!clusters.length){
    grid.innerHTML='<div class="empty">No cluster data for '+year+'.</div>';
    return;
  }
  grid.innerHTML=clusters.map((c,idx)=>{
    const topics=c.topics||[];
    const agg={POSITIVE:0,NEGATIVE:0,NEUTRAL:0};
    topics.forEach(t=>{
      const lr=t.sentiment?.label_ratios||{};
      Object.entries(lr).forEach(([k,v])=>{ agg[k]=(agg[k]||0)+v; });
    });
    const tot=Object.values(agg).reduce((a,b)=>a+b,0)||1;
    const pct=k=>((agg[k]||0)/tot*100).toFixed(0);

    const topicHTML=topics.length
      ? topics.map(t=>{
          const dom=t.sentiment?.dominant||'NEUTRAL';
          const ws=(t.top_words||[]).slice(0,6).map(w=>`<span class="chip">${w}</span>`).join('');
          return `<div style="margin-bottom:.65rem">
            <div style="font-size:.58rem;color:var(--accent3);margin-bottom:.3rem">
              topic ${t.topic_id} · ${t.count||0} docs <span class="sbadge ${dom}" style="margin-left:.3rem">${dom}</span>
            </div>
            <div class="word-chips">${ws}</div>
            ${t.explanation?`<div style="font-size:.63rem;color:var(--muted);margin-top:.35rem;line-height:1.55">${t.explanation}</div>`:''}
          </div>`;
        }).join('')
      : (c.error?`<div style="font-size:.62rem;color:var(--neg)">${c.error}</div>`:'<div style="font-size:.62rem;color:var(--muted)">No topics</div>');

    return `<div class="cluster-card" style="animation-delay:${idx*.05}s">
      <div class="c-header">
        <span style="font-family:'Syne',sans-serif;font-size:.8rem;font-weight:700">
          Cluster <span style="color:var(--accent)">${c.cluster_id}</span>
        </span>
        <span style="font-size:.6rem;color:var(--muted)">${c.n_docs||0} docs · ${c.n_topics||0} topics</span>
      </div>
      <div class="c-body">
        <div class="bar-row"><span class="bar-lbl">positive</span><div class="bar-track"><div class="bar-fill p" style="width:${pct('POSITIVE')}%"></div></div><span class="bar-pct">${pct('POSITIVE')}%</span></div>
        <div class="bar-row"><span class="bar-lbl">negative</span><div class="bar-track"><div class="bar-fill n" style="width:${pct('NEGATIVE')}%"></div></div><span class="bar-pct">${pct('NEGATIVE')}%</span></div>
        <div class="bar-row"><span class="bar-lbl">neutral</span><div class="bar-track"><div class="bar-fill u" style="width:${pct('NEUTRAL')}%"></div></div><span class="bar-pct">${pct('NEUTRAL')}%</span></div>
        <div style="border-top:1px solid var(--border);padding-top:.7rem;margin-top:.3rem">${topicHTML}</div>
      </div>
    </div>`;
  }).join('');
}

/* ── TIMELINE ── */
function mkTimeline(years){
  const grid=document.getElementById('tl-grid');
  grid.innerHTML=[...years].sort((a,b)=>a.year-b.year).map((y,idx)=>{
    const top=getTop10(y)[0];
    const topLabel=top?(top.label||(top.top_words||[]).slice(0,3).join(', ')):'—';
    const dist=y.dominant_sentiment_dist||{};
    const tot=Object.values(dist).reduce((a,b)=>a+b,0)||1;
    const pW=k=>((dist[k]||0)/tot*100).toFixed(0);
    return `<div class="tl-card" style="animation-delay:${idx*.04}s" onclick="goYear(${y.year})">
      <div class="tl-year">${y.year}</div>
      <div class="tl-stats">${(y.total_docs||0).toLocaleString()} posts · ${y.total_topics||0} topics</div>
      ${top?`<div class="tl-top">Top: <span>${topLabel}</span></div>`:''}
      <div class="mini-bar">
        <div class="mini-seg" style="width:${pW('POSITIVE')}%;background:var(--pos)"></div>
        <div class="mini-seg" style="width:${pW('NEGATIVE')}%;background:var(--neg)"></div>
        <div class="mini-seg" style="width:${pW('NEUTRAL')}%;background:var(--neu)"></div>
      </div>
      <div style="display:flex;gap:.7rem;margin-top:.3rem">
        ${Object.entries(dist).map(([k,v])=>`<span style="font-size:.54rem;color:var(--muted)">${k} ${((v/tot)*100).toFixed(0)}%</span>`).join('')}
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


if __name__ == "__main__":
    found = next((p for p in RESULTS_FILES if Path(p).exists()), None)
    print(f"[INFO] {'Loading: ' + found if found else 'WARNING: No results file found — run the pipeline first'}")
    print("[INFO] Dashboard → http://localhost:5000")
    app.run(debug=False, port=5000)