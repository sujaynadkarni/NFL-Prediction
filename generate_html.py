"""
Render data/processed/predictions.json → index.html (static, open in any browser)
Views: overview (favorites, week-by-week picks, championship odds, model scorecard)
       and a team page (#team=JAX) reached through the search box.
Run: .venv/bin/python generate_html.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed" / "predictions.json"
OUT  = ROOT / "index.html"

TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D", "CAR": "#0085CA",
    "CHI": "#C83803", "CIN": "#FB4F14", "CLE": "#FF3C00", "DAL": "#003594", "DEN": "#FB4F14",
    "DET": "#0076B6", "GB": "#203731", "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778",
    "KC": "#E31837", "LA": "#003594", "LAC": "#0080C6", "LV": "#A5ACAF", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265", "NYJ": "#125740",
    "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#69BE28", "SF": "#AA0000", "TB": "#D50A0A",
    "TEN": "#4B92DB", "WAS": "#5A1414",
}

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL 2026 Predictions</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#12161c;--muted:#5d6675;--line:#e2e5ea;--soft:#eef1f5;--accent:#1f5fbf;
--good:#1a7f4b;--bad:#b3261e;--bar:#cfdcf2;--afc:#c8102e;--nfc:#013369}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e1115;--card:#161a20;--ink:#e8ebf0;--muted:#98a1b0;
--line:#272d36;--soft:#1d232b;--accent:#6ea0ff;--good:#4cc38a;--bad:#ff7a70;--bar:#243553;--afc:#ff6b7f;--nfc:#7fa8ff}}
:root[data-theme="dark"]{--bg:#0e1115;--card:#161a20;--ink:#e8ebf0;--muted:#98a1b0;
--line:#272d36;--soft:#1d232b;--accent:#6ea0ff;--good:#4cc38a;--bad:#ff7a70;--bar:#243553;--afc:#ff6b7f;--nfc:#7fa8ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:0 16px 64px}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
.bar{max-width:1100px;margin:0 auto;padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.brand{font-weight:750;font-size:18px;cursor:pointer;white-space:nowrap}.brand span{color:var(--accent)}
.search{margin-left:auto;position:relative;flex:1;max-width:320px;min-width:200px}
.search input{width:100%;font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--ink)}
.sugg{position:absolute;left:0;right:0;top:calc(100% + 4px);background:var(--card);border:1px solid var(--line);border-radius:10px;
box-shadow:0 8px 24px rgba(0,0,0,.12);overflow:hidden;display:none}
.sugg div{padding:8px 12px;cursor:pointer;display:flex;align-items:center;gap:8px}.sugg div:hover,.sugg div.on{background:var(--soft)}
h2{font-size:19px;margin:40px 0 12px}
.sub{color:var(--muted);font-size:14px}
.note{color:var(--muted);font-size:13px;margin-top:10px}
.chip{display:inline-block;width:10px;height:10px;border-radius:3px;flex:none}
.conf{font-size:10.5px;font-weight:700;padding:1px 6px;border-radius:4px;color:#fff;margin-left:6px;vertical-align:1px}
.AFC{background:var(--afc)}.NFC{background:var(--nfc)}
.hero{padding:24px 0 4px}.hero h1{font-size:28px;margin:0 0 4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;border-top:4px solid var(--tc,var(--line))}
.tile.click{cursor:pointer}.tile.click:hover{border-color:var(--accent)}
.tile .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.tile .v{font-size:21px;font-weight:700;margin-top:4px}.tile .s{color:var(--muted);font-size:13px}
.weeks{display:flex;gap:6px;overflow-x:auto;padding-bottom:6px;margin-bottom:10px;scrollbar-width:thin}
.weeks button,.seg button{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px;border-radius:999px;
cursor:pointer;font-size:13px;white-space:nowrap;flex:none}
.weeks button.on,.seg button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.weeks button.done{color:var(--muted)}.weeks button.done.on{color:#fff}
.games{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:12px}
.g{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.g .meta{color:var(--muted);font-size:12px;margin-bottom:8px}
.team{display:flex;align-items:center;gap:8px;font-weight:600;cursor:pointer}.team:hover .nm{text-decoration:underline}
.team .p{margin-left:auto;font-variant-numeric:tabular-nums}
.team.fav .nm,.team.fav .p{color:var(--accent)}
.meter{display:flex;height:8px;border-radius:4px;overflow:hidden;margin:7px 0}
.why{font-size:13px;margin-top:10px}.why div{margin:1px 0}
.up{color:var(--good)}.down{color:var(--bad)}
.foot{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:12px;margin-top:10px;border-top:1px solid var(--line);padding-top:8px}
.qb{color:var(--muted);font-size:12px;margin-top:6px}
.res{font-weight:600}
.tbl{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.tbl th,.tbl td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl th{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.tbl th.s{cursor:pointer;user-select:none}
.tbl td.l,.tbl th.l{text-align:left}.tbl tr:last-child td{border-bottom:0}
.tbl tr.link{cursor:pointer}.tbl tr.link:hover td{background:var(--soft)}
.scroll{overflow-x:auto;border-radius:12px}
.pct{position:relative}.pct span{position:relative;z-index:1}
.pct i{position:absolute;right:6px;top:7px;bottom:7px;background:var(--bar);border-radius:3px;z-index:0}
.seg{display:flex;gap:6px;margin-bottom:10px}
.big{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.stat .v{font-size:30px;font-weight:750;font-variant-numeric:tabular-nums}.stat .k{font-size:13px;color:var(--muted)}
/* team page */
.thead{display:flex;align-items:center;gap:14px;padding:26px 0 6px}
.badge{width:52px;height:52px;border-radius:12px;display:grid;place-items:center;color:#fff;font-weight:800;font-size:15px;flex:none}
.thead h1{margin:0;font-size:26px}
.summary{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;font-size:15.5px;line-height:1.6;margin-top:14px}
.outs{display:grid;gap:8px}
.orow{display:grid;grid-template-columns:190px 1fr 56px;gap:10px;align-items:center;font-size:14px}
.otrack{height:12px;background:var(--soft);border-radius:6px;overflow:hidden}.otrack b{display:block;height:100%;border-radius:6px}
.orow .n{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.seeds{display:grid;grid-template-columns:repeat(8,1fr);gap:6px;align-items:end;height:150px;margin-top:6px}
.seeds .c{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;font-size:12px;color:var(--muted)}
.seeds .c b{width:100%;border-radius:6px 6px 2px 2px;min-height:2px}
.seeds .c .pv{color:var(--ink);font-weight:600;margin-bottom:3px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.back{color:var(--accent);cursor:pointer;font-size:14px;margin-top:18px;display:inline-block}
@media (max-width:760px){.two{grid-template-columns:1fr}.orow{grid-template-columns:120px 1fr 48px}.hero h1{font-size:23px}
.search{max-width:none;margin-left:0;flex-basis:100%}}
</style>
</head>
<body>
<header><div class="bar">
  <div class="brand" onclick="go('')">NFL <span>2026</span> Predictions</div>
  <div class="search"><input id="q" placeholder="Search a team…" autocomplete="off"><div class="sugg" id="sugg"></div></div>
</div></header>
<div class="wrap" id="app"></div>

<script>
const D = __DATA__, COLORS = __COLORS__;
const T = D.teams, codes = Object.keys(T);
const pct = v => v >= 0.995 ? '>99%' : v > 0 && v < 0.005 ? '<1%' : v*100 >= 10 || v === 0 ? Math.round(v*100)+'%' : (v*100).toFixed(1)+'%';
const nick = c => T[c].name.split(' ').slice(-1)[0];
const chip = c => `<span class="chip" style="background:${COLORS[c]}"></span>`;
const rankOf = (c, k) => 1 + codes.filter(o => T[o][k] > T[c][k]).length;
let week = D.upcoming_week || 1, confFilter = 'All', sortKey = 'champion';

// ── routing ──────────────────────────────────────────────
function go(hash){ location.hash = hash; }
window.onhashchange = render;
function render(){
  const m = location.hash.match(/team=([A-Z]+)/);
  document.getElementById('app').innerHTML = m && T[m[1]] ? teamPage(m[1]) : home();
  bind(); window.scrollTo(0, 0);
}

// ── search ───────────────────────────────────────────────
const q = document.getElementById('q'), sugg = document.getElementById('sugg');
let hits = [], sel = 0;
q.oninput = () => {
  const s = q.value.trim().toLowerCase();
  hits = s ? codes.filter(c => T[c].name.toLowerCase().includes(s) || c.toLowerCase() === s).slice(0, 8) : [];
  sel = 0; drawSugg();
};
function drawSugg(){
  sugg.style.display = hits.length ? 'block' : 'none';
  sugg.innerHTML = hits.map((c,i) => `<div class="${i===sel?'on':''}" data-c="${c}">${chip(c)}${T[c].name}<span class="sub" style="margin-left:auto">${T[c].record}</span></div>`).join('');
  sugg.querySelectorAll('div').forEach(d => d.onmousedown = () => pick(d.dataset.c));
}
function pick(c){ q.value = ''; hits = []; drawSugg(); q.blur(); go('team=' + c); }
q.onkeydown = e => {
  if (e.key === 'ArrowDown') { sel = Math.min(sel+1, hits.length-1); drawSugg(); e.preventDefault(); }
  if (e.key === 'ArrowUp') { sel = Math.max(sel-1, 0); drawSugg(); e.preventDefault(); }
  if (e.key === 'Enter' && hits.length) pick(hits[sel]);
  if (e.key === 'Escape') { hits = []; drawSugg(); }
};
q.onblur = () => setTimeout(() => { hits = []; drawSugg(); }, 150);

// ── shared pieces ────────────────────────────────────────
function gameCard(g){
  const favH = g.favorite === g.home, ph = g.p_home_pre ?? g.p_home, pa = 1 - ph;
  const row = (c, p, fav, at) => `<div class="team ${fav?'fav':''}" data-team="${c}">${chip(c)}<span class="nm">${at}${T[c].name}</span><span class="p">${pct(p)}</span></div>`;
  const vegas = g.vegas_home_spread == null ? '' : 'Vegas: ' + (g.vegas_home_spread > 0 ? `${g.home} −${g.vegas_home_spread}` : g.vegas_home_spread < 0 ? `${g.away} −${-g.vegas_home_spread}` : 'pick’em');
  const why = g.why.map(w => `<div class="up">▲ ${w.label}</div>`).join('') +
              g.against.map(w => `<div class="down">▼ ${w.label} (${nick(favH ? g.away : g.home)})</div>`).join('');
  const qbOdds = [['away', g.away], ['home', g.home]].filter(([s]) => g[s+'_qb_odds'])
    .map(([s, c]) => `<div class="qb">${nick(c)} QB: ${g[s+'_qb_odds'].map(([n,p]) => `${n} ${Math.round(p*100)}%`).join(' · ')}</div>`).join('');
  const res = g.played
    ? `<span class="res ${g.correct?'up':'down'}">Final ${g.away} ${g.away_score}–${g.home_score} ${g.home} · ${g.correct?'✓ called':'✗ missed'}</span>`
    : `Projected ${g.away} ${g.proj_away}–${g.proj_home} ${g.home}`;
  return `<div class="g"><div class="meta">${new Date(g.gameday+'T12:00').toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric'})} · ${g.gametime} ET · ${g.stadium}</div>
    ${row(g.away, pa, !favH, '')}
    <div class="meter"><b style="width:${pa*100}%;background:${COLORS[g.away]}"></b><b style="width:${ph*100}%;background:${COLORS[g.home]}"></b></div>
    ${row(g.home, ph, favH, '@ ')}
    <div class="qb">QBs: ${g.away_qb || '?'} vs ${g.home_qb || '?'}</div>${qbOdds}
    <div class="why"><b>Why ${nick(g.favorite)}:</b>${why || '<div>Near coin flip</div>'}</div>
    <div class="foot"><span>${res}</span><span>${vegas}</span></div></div>`;
}

// ── overview ─────────────────────────────────────────────
function home(){
  const byK = k => codes.slice().sort((a,b) => T[b][k] - T[a][k]);
  const topConf = c => codes.filter(o => T[o].conf === c).sort((a,b) => T[b].super_bowl - T[a].super_bowl)[0];
  const sb = byK('champion')[0], af = topConf('AFC'), nf = topConf('NFC'), sc = D.scorecard.all;
  const tile = (k, c, s) => `<div class="tile click" data-team="${c}" style="--tc:${COLORS[c]}"><div class="k">${k}</div><div class="v">${T[c].name}</div><div class="s">${s}</div></div>`;
  const weeks = [...new Set(D.games.map(g => g.week))].sort((a,b) => a-b);
  const done = w => D.games.filter(g => g.week === w).every(g => g.played);
  return `<div class="hero"><h1>NFL 2026 Predictions</h1>
    <div class="sub">Updated ${D.generated} · ${D.n_sim.toLocaleString()} simulated seasons · ${sc.games} games played and locked in · next up: Week ${D.upcoming_week}</div></div>
  <div class="tiles">
    ${tile('Super Bowl favorite', sb, pct(T[sb].champion) + ' to win it all')}
    ${tile('AFC favorite', af, pct(T[af].super_bowl) + ' to win the AFC')}
    ${tile('NFC favorite', nf, pct(T[nf].super_bowl) + ' to win the NFC')}
    <div class="tile"><div class="k">2026 picks so far</div><div class="v">${sc.correct} / ${sc.games}</div><div class="s">winners called · Vegas ${sc.vegas_correct} / ${sc.vegas_games}</div></div>
  </div>

  <h2>Week-by-week picks</h2>
  <div class="weeks">${weeks.map(w => `<button data-week="${w}" class="${w===week?'on':''} ${done(w)?'done':''}">Week ${w}</button>`).join('')}</div>
  <div class="games">${D.games.filter(g => g.week === week).map(gameCard).join('')}</div>

  <h2>Championship odds</h2>
  <div class="seg">${['All','AFC','NFC'].map(c => `<button data-conf="${c}" class="${c===confFilter?'on':''}">${c}</button>`).join('')}</div>
  <div class="scroll">${oddsTable()}</div>
  <div class="note">Click a column to sort, or a team to open its page. "Win conference" = win the AFC/NFC Championship and reach the Super Bowl.</div>

  ${modelSection()}`;
}

function oddsTable(){
  const cols = [['name','Team'],['record','Record'],['avg_wins','Proj. wins'],['playoffs','Playoffs'],['division','Win division'],
                ['seed1','#1 seed'],['conf_game','Conf. game'],['super_bowl','Win conference'],['champion','Win Super Bowl']];
  const rows = codes.filter(c => confFilter === 'All' || T[c].conf === confFilter)
    .sort((a,b) => typeof T[a][sortKey] === 'string' ? T[a][sortKey].localeCompare(T[b][sortKey]) : T[b][sortKey] - T[a][sortKey]);
  return `<table class="tbl"><tr>${cols.map(([k,l],i) => `<th class="s ${i?'':'l'}" data-sort="${k}">${l}${k===sortKey?' ▾':''}</th>`).join('')}</tr>` +
    rows.map(c => `<tr class="link" data-team="${c}">` + cols.map(([k]) => {
      const v = T[c][k];
      if (k === 'name') return `<td class="l">${chip(c)} ${v}<span class="conf ${T[c].conf}">${T[c].conf}</span></td>`;
      if (k === 'record' || k === 'avg_wins') return `<td>${v}</td>`;
      return `<td class="pct"><i style="width:${Math.min(v,1)*80}%"></i><span>${pct(v)}</span></td>`;
    }).join('') + '</tr>').join('') + '</table>';
}

function modelSection(){
  const sc = D.scorecard, B = D.backtest, f3 = v => v == null ? '—' : v.toFixed(3);
  const live = sc.live.games, rec = sc.reconstructed.games;
  const wk = sc.by_week.map(w => `<tr><td class="l">Week ${w.week}</td><td>${w.games}</td><td>${w.correct}</td><td>${w.vegas_correct}</td>
      <td>${f3(w.brier)}</td><td>${f3(w.vegas_brier)}</td><td class="l">${w.live===w.games?'live':w.live?`${w.live} live`:'reconstructed'}</td></tr>`).join('');
  return `<h2>How good is our model?</h2>
  <div class="big">
    <div class="stat"><div class="k">2026 season so far</div><div class="v">${sc.all.games ? Math.round(100*sc.all.correct/sc.all.games) : 0}%</div>
      <div class="k">${sc.all.correct} of ${sc.all.games} winners · Vegas ${sc.all.vegas_correct} of ${sc.all.vegas_games}</div></div>
    <div class="stat"><div class="k">2024–25 test (${B.model.games} games)</div><div class="v">${(B.model.accuracy*100).toFixed(1)}%</div>
      <div class="k">winners called · Vegas ${(B.vegas.accuracy*100).toFixed(1)}%</div></div>
    <div class="stat"><div class="k">Average miss on the final margin</div><div class="v">${B.model.mae.toFixed(1)} pts</div>
      <div class="k">Vegas spread ${B.vegas.mae.toFixed(1)} pts · Elo alone ${B.elo_only.mae.toFixed(1)} pts</div></div>
  </div>
  <h2 style="font-size:16px;margin-top:24px">2026 week by week</h2>
  <div class="scroll"><table class="tbl"><tr><th class="l">Week</th><th>Games</th><th>We called</th><th>Vegas called</th><th>Our Brier</th><th>Vegas Brier</th><th class="l">Picks</th></tr>${wk}
    <tr><td class="l"><b>Season</b></td><td>${sc.all.games}</td><td>${sc.all.correct}</td><td>${sc.all.vegas_correct}</td><td>${f3(sc.all.brier)}</td><td>${f3(sc.all.vegas_brier)}</td>
    <td class="l">${live} live · ${rec} reconstructed</td></tr></table></div>
  <div class="note"><b>Live</b> picks were saved before kickoff and frozen at kickoff. <b>Reconstructed</b> picks (games played before tracking began) are the model's
  prediction using only pre-game data. Brier score measures how good the probabilities are (lower is better; always saying 50% scores 0.250).
  Vegas is the benchmark: the betting line is very hard to beat.</div>`;
}

// ── team page ────────────────────────────────────────────
function teamPage(c){
  const t = T[c], seeds = t.seeds, miss = Math.max(0, 1 - t.playoffs);
  const wc = seeds[4] + seeds[5] + seeds[6], divTeams = codes.filter(o => T[o].div_name === t.div_name);
  const divFav = divTeams.slice().sort((a,b) => T[b].division - T[a].division)[0];
  const bestSeed = seeds.map((p,i) => [i === 0 ? '#1 seed (first-round bye)' : `#${i+1} seed`, p]).sort((a,b) => b[1]-a[1])[0];
  const finishLine = t.playoffs >= 0.5
    ? `If they make it, their most likely spot is the <b>${bestSeed[0]}</b> (${pct(bestSeed[1])} overall).`
    : `Most likely outcome: <b>missing the playoffs</b> (${pct(miss)}).`;
  const tier = p => p >= .8 ? 'a strong favorite to make the playoffs' : p >= .6 ? 'likely to make the playoffs' : p >= .4 ? 'a coin flip to make the playoffs'
                  : p >= .2 ? 'a long shot for the playoffs' : 'unlikely to make the playoffs';
  const divLine = divFav === c
    ? `They're the favorite in the ${t.div_name} (${pct(t.division)} to win it).`
    : `They win the ${t.div_name} ${pct(t.division)} of the time; ${T[divFav].name} are the division favorite (${pct(T[divFav].division)}).`;
  const summary = `The <b>${t.name}</b> (${t.record}) are projected to finish with about <b>${t.avg_wins} wins</b> and are ${tier(t.playoffs)}:
    they make it in <b>${pct(t.playoffs)}</b> of simulations — ${pct(t.division)} as division champions and ${pct(wc)} as a wild card. ${divLine}
    ${finishLine} They reach the ${t.conf} Championship Game ${pct(t.conf_game)} of the time,
    win the ${t.conf} ${pct(t.super_bowl)}, and win the Super Bowl <b>${pct(t.champion)}</b> — #${rankOf(c,'champion')} of 32.`;
  const bar = (label, p) => `<div class="orow"><span>${label}</span><div class="otrack"><b style="width:${p*100}%;background:${COLORS[c]}"></b></div><span class="n">${pct(p)}</span></div>`;
  const maxSeed = Math.max(...seeds, miss, .01);
  const seedCol = (label, p, col) => `<div class="c"><span class="pv">${pct(p)}</span><b style="height:${p/maxSeed*100}%;background:${col}"></b><span>${label}</span></div>`;
  const sched = D.games.filter(g => g.home === c || g.away === c).sort((a,b) => a.week - b.week).map(g => {
    const home = g.home === c, opp = home ? g.away : g.home, p = home ? (g.p_home_pre ?? g.p_home) : 1 - (g.p_home_pre ?? g.p_home);
    let result = '';
    if (g.played) {
      const us = home ? g.home_score : g.away_score, them = home ? g.away_score : g.home_score;
      result = `<span class="${us>them?'up':'down'}">${us>them?'W':us<them?'L':'T'} ${us}–${them}</span>`;
    } else result = `<span class="sub">proj. ${home ? g.proj_home : g.proj_away}–${home ? g.proj_away : g.proj_home}</span>`;
    return `<tr class="link" data-team="${opp}"><td class="l">Wk ${g.week}</td><td class="l">${home?'vs':'@'} ${chip(opp)} ${T[opp].name}</td>
      <td class="pct"><i style="width:${p*70}%"></i><span>${pct(p)}</span></td><td>${result}</td></tr>`;
  }).join('');
  const divRows = divTeams.sort((a,b) => T[b].division - T[a].division).map(o => `<tr class="link" data-team="${o}" ${o===c?'style="font-weight:700"':''}>
    <td class="l">${chip(o)} ${T[o].name}</td><td>${T[o].record}</td><td>${T[o].avg_wins}</td><td>${pct(T[o].division)}</td><td>${pct(T[o].playoffs)}</td></tr>`).join('');
  return `<span class="back" onclick="go('')">← All predictions</span>
  <div class="thead"><div class="badge" style="background:${COLORS[c]}">${c}</div>
    <div><h1>${t.name}</h1><div class="sub">${t.div_name} · ${t.record} · QB ${t.qb || '—'} · ${pct(t.champion)} to win the Super Bowl</div></div></div>
  <div class="summary">${summary}</div>
  <div class="two">
    <div><h2>Season outcomes</h2><div class="outs">
      ${bar('Make the playoffs', t.playoffs)}${bar('Win the division', t.division)}${bar('Wild card', wc)}${bar('#1 seed (bye)', t.seed1)}
      ${bar('Reach divisional round', t.div_round)}${bar('Reach conf. championship', t.conf_game)}${bar('Win the ' + t.conf, t.super_bowl)}${bar('Win the Super Bowl', t.champion)}
    </div></div>
    <div><h2>Where they finish</h2>
      <div class="seeds">${seeds.map((p,i) => seedCol('#'+(i+1), p, COLORS[c])).join('')}${seedCol('Out', miss, 'var(--muted)')}</div>
      <div class="note">Chance of each playoff seed in ${t.conf}. Seeds 1–4 are division winners, 5–7 wild cards; only #1 gets a bye.</div>
    </div>
  </div>
  <div class="two">
    <div><h2>Schedule</h2><div class="scroll"><table class="tbl"><tr><th class="l">Week</th><th class="l">Opponent</th><th>Win chance</th><th>Result</th></tr>${sched}</table></div></div>
    <div><h2>${t.div_name} race</h2><div class="scroll"><table class="tbl"><tr><th class="l">Team</th><th>Record</th><th>Proj. wins</th><th>Win div.</th><th>Playoffs</th></tr>${divRows}</table></div></div>
  </div>`;
}

// ── events ───────────────────────────────────────────────
function bind(){
  document.querySelectorAll('[data-team]').forEach(el => el.onclick = e => { e.stopPropagation(); go('team=' + el.dataset.team); });
  document.querySelectorAll('[data-week]').forEach(b => b.onclick = () => { week = +b.dataset.week; rerenderKeep(); });
  document.querySelectorAll('[data-conf]').forEach(b => b.onclick = () => { confFilter = b.dataset.conf; rerenderKeep(); });
  document.querySelectorAll('[data-sort]').forEach(th => th.onclick = () => { sortKey = th.dataset.sort; rerenderKeep(); });
}
function rerenderKeep(){ const y = window.scrollY; document.getElementById('app').innerHTML = home(); bind(); window.scrollTo(0, y); }
render();
</script>
</body></html>
"""


def main():
    data = json.loads(DATA.read_text())
    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__COLORS__", json.dumps(TEAM_COLORS))
    OUT.write_text(html)
    print(f"Wrote {OUT.name} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
