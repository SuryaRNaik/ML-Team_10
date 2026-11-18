const $ = id => document.getElementById(id);
const nums = t => t.trim().split(/\r?\n/).filter(l => l.trim()).map(l => l.trim().split(/[\s,]+/).map(Number));
const msg = m => $("msg").textContent = m || "";
function parseCSV(t) {
  const rows = []; let r = [], f = "", q = false;
  for (let i = 0; i < t.length; i++) { const c = t[i];
    if (q) { if (c === '"') { if (t[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true; else if (c === ",") { r.push(f); f = ""; }
    else if (c === "\n") { r.push(f); rows.push(r); r = []; f = ""; } else if (c !== "\r") f += c; }
  if (f !== "" || r.length) { r.push(f); rows.push(r); }
  const h = rows[0].map(s => s.trim());
  return rows.slice(1).filter(x => x.length > 1).map(x => { const o = {}; h.forEach((k, i) => { const v = x[i]; o[k] = v !== "" && v !== undefined && !isNaN(v) ? Number(v) : v; }); return o; });
}
// ---------- tabs
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll("section").forEach(s => s.hidden = s.id !== "tab-" + b.dataset.tab);
  if (b.dataset.tab === "map") setTimeout(() => map.invalidateSize(), 50); });
// ---------- map
const map = L.map("map", {preferCanvas: true}).setView([12.97, 77.59], 14);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom: 19, attribution: "© OpenStreetMap"}).addTo(map);
const LY = {net: L.layerGroup().addTo(map), ref: L.layerGroup().addTo(map), route: L.layerGroup().addTo(map), gps: L.layerGroup().addTo(map), sel: L.layerGroup().addTo(map)};
const S = {files: {}, track: null, nodes: null, arcs: null, pred: null, ref: null}; let byPt = {};
function seg(id, opt, tip) { if (!S.arcs || !S.nodes) return null; const a = S.arcs[id]; if (!a) return null;
  const A = S.nodes[a[0]], B = S.nodes[a[1]]; if (!A || !B) return null;
  return L.polyline([[A[1], A[0]], [B[1], B[0]]], opt).bindTooltip(tip || "arc " + id); }
function draw() {
  Object.values(LY).forEach(l => l.clearLayers());
  const {track, nodes, arcs, pred, ref} = S, pts = []; byPt = {}; let st = "";
  if (pred) pred.forEach(r => byPt[r.pt_idx] = r);
  if (nodes && arcs) arcs.forEach((_, i) => { const p = seg(i, {color: "#aaa", weight: 2}); if (p) p.addTo(LY.net); });
  if (ref) ref.forEach(id => { const p = seg(id, {color: "#2ca02c", weight: 9, opacity: .4, dashArray: "6 6"}, "reference arc " + id); if (p) p.addTo(LY.ref); });
  if (track) { track.forEach((p, i) => { pts.push([p[1], p[0]]);
      L.circleMarker([p[1], p[0]], {radius: 3, color: "#1f77b4", weight: 1}).bindTooltip(`#${i} t=${p[2]}` + (byPt[i] ? ` → arc ${byPt[i].arc_id}` : "")).addTo(LY.gps); });
    st += `GPS points: ${track.length}<br>`; $("scrub").max = Math.max(0, track.length - 1); $("scrub").value = 0; }
  if (pred && nodes && arcs) { const seq = [];
    pred.slice().sort((a, b) => a.pt_idx - b.pt_idx).forEach(r => { if (!seq.length || seq[seq.length - 1] !== r.arc_id) seq.push(r.arc_id); });
    seq.forEach(id => { const p = seg(id, {color: "#d62728", weight: 5, opacity: .8}); if (p) p.addTo(LY.route); });
    let ok = 0; for (let i = 1; i < seq.length; i++) { const a = arcs[seq[i - 1]], b = arcs[seq[i]]; if (a && b && a[1] === b[0]) ok++; }
    st += `Predicted arcs (distinct, in order): ${seq.length}<br>Continuous transitions: ${seq.length > 1 ? (100 * ok / (seq.length - 1)).toFixed(1) + "%" : "n/a"}`; }
  $("stats").innerHTML = st; $("pinfo").textContent = ""; toggles();
  if (pts.length) map.fitBounds(pts, {padding: [30, 30]});
}
function toggles() { [["net", "t_net"], ["gps", "t_gps"], ["route", "t_route"], ["ref", "t_ref"]].forEach(([k, id]) => $(id).checked ? map.addLayer(LY[k]) : map.removeLayer(LY[k])); }
["t_net", "t_gps", "t_route", "t_ref"].forEach(id => $(id).onchange = toggles);
$("scrub").oninput = e => { LY.sel.clearLayers(); const i = +e.target.value, p = S.track && S.track[i]; if (!p) return;
  L.circleMarker([p[1], p[0]], {radius: 9, color: "#ff7f0e", fillOpacity: .3}).addTo(LY.sel);
  const r = byPt[i]; if (r) { const l = seg(r.arc_id, {color: "#ff7f0e", weight: 7}, `point ${i} → arc ${r.arc_id}`); if (l) l.addTo(LY.sel); }
  $("pinfo").textContent = `point ${i}` + (r ? ` → arc ${r.arc_id}` + (r.prob != null && !isNaN(r.prob) ? ` (p=${(+r.prob).toFixed(3)})` : "") : " (no prediction)"); map.panTo([p[1], p[0]]); };
[["f_track", "track"], ["f_nodes", "nodes"], ["f_arcs", "arcs"]].forEach(([id, k]) => $(id).onchange = async e => {
  const f = e.target.files[0]; if (!f) return; S.files[k] = f; S[k] = nums(await f.text());
  if (k === "arcs") S.arcs = S.arcs.map(r => r.map(Math.round)); S.pred = null; draw(); });
$("f_pred").onchange = async e => { const f = e.target.files[0]; if (f) { S.pred = parseCSV(await f.text()); draw(); } };
$("f_ref").onchange = async e => { const f = e.target.files[0]; if (f) { S.ref = nums(await f.text()).map(r => Math.round(r[0])); draw(); } };
$("run").onclick = async () => { try { msg(); const f = S.files; if (!f.track || !f.nodes || !f.arcs) throw new Error("Load .track, .nodes and .arcs first.");
  S.pred = await predictTrip(f); draw(); } catch (e) { msg(e.message); } };
$("demo").onclick = () => { msg(); const N = 6, d = .002, lon0 = 77.59, lat0 = 12.97, nodes = [], arcs = [], id = {};
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) nodes.push([lon0 + c * d, lat0 + r * d]);
  const add = (a, b) => { id[a + "_" + b] = arcs.length; arcs.push([a, b]); };
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) { const n = r * N + c; if (c < N - 1) { add(n, n + 1); add(n + 1, n); } if (r < N - 1) { add(n, n + N); add(n + N, n); } }
  const path = []; for (let c = 0; c < N; c++) path.push(2 * N + c); for (let r = 3; r < N; r++) path.push(r * N + 5);
  const ref = path.slice(1).map((b, i) => id[path[i] + "_" + b]), track = []; let t = 0;
  for (let i = 0; i < path.length - 1; i++) { const A = nodes[path[i]], B = nodes[path[i + 1]];
    for (let s = 0; s < 8; s++, t += 3) { const f = s / 8, j = () => (Math.random() - .5) * 1.4e-4; track.push([A[0] + (B[0] - A[0]) * f + j(), A[1] + (B[1] - A[1]) * f + j(), t]); } }
  const m = (lo, la) => [(lo - lon0) * Math.cos(lat0 * Math.PI / 180) * 111320, (la - lat0) * 110540];
  const sd = (p, a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy || 1, u = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2)); return Math.hypot(p[0] - a[0] - u * dx, p[1] - a[1] - u * dy); };
  const pred = track.map((p, i) => { const q = m(p[0], p[1]), n = m(...track[Math.min(i + 1, track.length - 1)]), pv = m(...track[Math.max(i - 1, 0)]);
    const h = [n[0] - pv[0], n[1] - pv[1]], hl = Math.hypot(h[0], h[1]) || 1; let best = 0, bs = 1e9;
    arcs.forEach(([a, b], k) => { const A = m(...nodes[a]), B = m(...nodes[b]), ul = Math.hypot(B[0] - A[0], B[1] - A[1]);
      const cos = ((B[0] - A[0]) * h[0] + (B[1] - A[1]) * h[1]) / (ul * hl), sc = sd(q, A, B) - 15 * cos; if (sc < bs) { bs = sc; best = k; } });
    return {pt_idx: i, arc_id: best, prob: 1}; });
  Object.assign(S, {track, nodes, arcs, pred, ref}); draw(); msg("Demo data (synthetic). Real predictions replace this once connected."); };
// ---------- models tab
let ROWS = BASELINE.slice(), DEC = [];
const COLS = ["src", "model", "cv_top1_mean", "test_top1", "test_macro_f1", "test_precision", "test_recall", "test_log_loss", "route_continuity", "train_time_s", "inference_ms_per_obs"];
const fmt = v => typeof v === "number" ? (Math.abs(v) < 100 ? v.toFixed(4) : v.toFixed(1)) : (v == null ? "" : v);
function table(el, rows, cols) { el.innerHTML = "<table><tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr>" + rows.map(r => "<tr>" + cols.map(c => `<td>${fmt(r[c])}</td>`).join("") + "</tr>").join("") + "</table>"; }
function bars(el, rows, label, metric) {
  const rs = rows.filter(r => typeof r[metric] === "number"), frac = /top1|f1|precision|recall|continuity|acc/.test(metric), low = /loss|time|ms/.test(metric);
  const den = frac ? 1 : Math.max(...rs.map(r => r[metric]), 1e-9);
  el.innerHTML = rs.sort((a, b) => low ? a[metric] - b[metric] : b[metric] - a[metric]).map(r =>
    `<div class="bar"><span>${label(r)}</span><div><i style="width:${Math.min(100, 100 * r[metric] / den)}%"></i>${/top1|acc/.test(metric) ? '<u style="left:90%"></u>' : ""}</div><b>${fmt(r[metric])}</b></div>`).join("") || "<p class='muted'>No numeric data for this metric.</p>"; }
function renderModels() { bars($("mbars"), ROWS.slice(), r => r.src + " · " + r.model, $("metric").value); table($("mtable"), ROWS, COLS); }
$("metric").onchange = renderModels; $("reset").onclick = () => { ROWS = BASELINE.slice(); renderModels(); };
$("f_models").onchange = async e => { const f = e.target.files[0]; if (!f) return; const nr = parseCSV(await f.text()); nr.forEach(r => r.src = r.src || f.name);
  ROWS = ROWS.filter(r => r.src !== f.name).concat(nr); renderModels(); };
// ---------- decoding tab
function renderDec() { if (!DEC.length) { $("dtable").innerHTML = "<p class='muted'>Nothing loaded yet. Load the independent-vs-decoded comparison file once the pipeline produces it.</p>"; $("dbars").innerHTML = ""; return; }
  const cols = Object.keys(DEC[0]), nc = cols.filter(c => typeof DEC[0][c] === "number"); table($("dtable"), DEC, cols);
  const cur = $("dmetric").value; $("dmetric").innerHTML = nc.map(c => `<option${c === cur ? " selected" : ""}>${c}</option>`).join(""); dbars(); }
function dbars() { bars($("dbars"), DEC.slice(), r => r.method || r.model || Object.values(r)[0], $("dmetric").value); }
$("dmetric").onchange = dbars;
$("f_dec").onchange = async e => { const f = e.target.files[0]; if (!f) return; const t = await f.text(); DEC = f.name.endsWith(".json") ? JSON.parse(t) : parseCSV(t); renderDec(); };
// ---------- connection tab
$("api_save").onclick = async () => { CONFIG.API_URL = $("api_url").value.trim().replace(/\/$/, "");
  try { const h = await apiHealth(); $("api_out").textContent = "Connected: " + JSON.stringify(h); $("apistate").textContent = "backend: connected"; }
  catch (e) { $("api_out").textContent = "Failed: " + e.message; $("apistate").textContent = "backend: unreachable"; } };
$("api_metrics").onclick = async () => { try { const rows = await apiMetrics(); rows.forEach(r => r.src = "backend");
  ROWS = ROWS.filter(r => r.src !== "backend").concat(rows); renderModels(); $("api_out").textContent = `Loaded ${rows.length} rows into the Models tab.`; }
  catch (e) { $("api_out").textContent = "Failed: " + e.message; } };
renderModels(); renderDec();
/* v2 */
(function () {
  const conf = $("t_conf"), btn = $("run"), r0 = btn.onclick;
  function paint() { let i = 0;
    LY.gps.eachLayer(l => { const r = byPt[i++], p = r && r.prob != null ? Number(r.prob) : NaN;
      l.setStyle({color: conf.checked && !isNaN(p) ? `hsl(${Math.round(120 * Math.max(0, Math.min(1, p)))},80%,40%)` : "#1f77b4"}); }); }
  function extra() {
    if (S.pred && S.ref && S.arcs) { const P = new Set(S.pred.map(r => r.arc_id)), R = new Set(S.ref); let hit = 0; P.forEach(a => { if (R.has(a)) hit++; });
      if (P.size && R.size) $("stats").innerHTML += `<br>Arc overlap with reference (rough): precision ${(100 * hit / P.size).toFixed(1)}%, recall ${(100 * hit / R.size).toFixed(1)}%`; }
    paint(); }
  const d0 = draw; draw = function () { d0(); extra(); };
  conf.onchange = paint;
  $("dl").onclick = () => { if (!S.pred || !S.pred.length) return msg("No predictions to download yet.");
    const rows = S.pred.slice().sort((a, b) => a.pt_idx - b.pt_idx).map(r => `${r.pt_idx},${r.arc_id},${r.prob ?? ""}`);
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob(["pt_idx,arc_id,prob\n" + rows.join("\n")], {type: "text/csv"}));
    a.download = "predictions.csv"; a.click(); };
  btn.onclick = async () => { const t = btn.textContent; btn.disabled = true; btn.textContent = "Running…";
    try { await r0(); } finally { btn.disabled = false; btn.textContent = t; }
    if (/Failed to fetch|NetworkError/i.test($("msg").textContent)) msg("Cannot reach the backend. Check the URL in the Connection tab, that the server is running, and that CORS is enabled."); };
})();
