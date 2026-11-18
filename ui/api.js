// The ONLY file that talks to the backend. Backend contract (enable CORS on it):
//  GET  /health   -> {"status":"ok","model":"..."}
//  POST /predict  multipart: track, nodes, arcs (raw files, NO .route) -> {"trip_id":"..","predictions":[{"pt_idx":0,"arc_id":12,"prob":0.9},...]}
//  GET  /metrics  -> [ {model, test_top1, test_macro_f1, ...}, ... ]   (optional, same columns as model_comparison.csv)
const need = () => { if (!CONFIG.API_URL) throw new Error("No backend set. Use the Connection tab."); };
async function apiHealth() { need(); const r = await fetch(CONFIG.API_URL + "/health"); if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }
async function apiMetrics() { need(); const r = await fetch(CONFIG.API_URL + "/metrics"); if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }
async function predictTrip(f) {
  need(); const fd = new FormData(); fd.append("track", f.track); fd.append("nodes", f.nodes); fd.append("arcs", f.arcs);
  const r = await fetch(CONFIG.API_URL + "/predict", { method: "POST", body: fd });
  if (!r.ok) throw new Error("Backend error " + r.status); return (await r.json()).predictions;
}
