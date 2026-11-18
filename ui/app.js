const $ = id => document.getElementById(id);
const state = { result: null, trips: [] };
const layers = {
  gps: L.layerGroup(),
  route: L.layerGroup(),
  selected: L.layerGroup(),
};
const map = L.map("map", {preferCanvas: true}).setView([12.97, 77.59], 12);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "© OpenStreetMap",
}).addTo(map);
Object.values(layers).forEach(layer => layer.addTo(map));

function message(text) { $("msg").textContent = text || ""; }
function setApiState(text) { $("apistate").textContent = text; }
function toggleLayers() {
  [["gps", "t_gps"], ["route", "t_route"]].forEach(([name, control]) => {
    if ($(control).checked) map.addLayer(layers[name]); else map.removeLayer(layers[name]);
  });
}
function clearLayers() {
  Object.values(layers).forEach(layer => layer.clearLayers());
}

function renderResult() {
  const result = state.result;
  clearLayers();
  if (!result) return;
  const gps = result.track.map(p => [p[1], p[0]]);
  if (gps.length) {
    L.polyline(gps, {color: "#2563eb", weight: 2, opacity: .7}).addTo(layers.gps);
    const stride = Math.max(1, Math.ceil(gps.length / 900));
    gps.forEach((point, index) => {
      if (index % stride === 0) {
        L.circleMarker(point, {radius: 2, color: "#2563eb", weight: 1})
          .bindTooltip(`GPS point ${index}`).addTo(layers.gps);
      }
    });
    $("scrub").max = Math.max(0, gps.length - 1);
    $("scrub").disabled = false;
    map.fitBounds(gps, {padding: [30, 30]});
  }
  const route = result.route_geometry.flatMap((segment, index) =>
    index === 0 ? segment : [segment[1]]);
  if (route.length) L.polyline(route, {
    color: "#dc2626", weight: 6, opacity: .85,
  }).addTo(layers.route);
  const m = result.metrics.hmm_viterbi;
  $("stats").innerHTML = [
    `<b>${result.trip_id}</b> · ${result.n_points} GPS points`,
    `Model: ${result.model}`,
    `Top-1 accuracy: ${(100 * m.top1_accuracy).toFixed(2)}%`,
    `Route continuity: ${(100 * m.route_continuity).toFixed(2)}%`,
    `Route F1: ${(100 * m.route_f1).toFixed(2)}%`,
    `Route precision / recall: ${(100 * m.route_precision).toFixed(2)}% / ${(100 * m.route_recall).toFixed(2)}%`,
  ].join("<br>");
  renderDecode();
  toggleLayers();
}

function selectPoint(index) {
  layers.selected.clearLayers();
  if (!state.result) return;
  const row = state.result.predictions[index];
  if (!row) return;
  const point = [row.lat, row.lon];
  L.circleMarker(point, {radius: 9, color: "#f97316", fillOpacity: .35})
    .bindTooltip(`GPS ${row.pt_idx} → decoded arc ${row.arc_decoded}`).addTo(layers.selected);
  $("pinfo").textContent = `Point ${row.pt_idx} · classifier arc ${row.arc_top1} · decoded arc ${row.arc_decoded} · probability ${(100 * row.probability).toFixed(1)}%`;
  map.panTo(point);
}

function renderDecode() {
  const result = state.result;
  if (!result) {
    $("dbars").innerHTML = "<p class='muted'>Select a trip to run the comparison.</p>";
    $("dtable").innerHTML = "";
    return;
  }
  const rows = [
    {method: "Classifier only", ...result.metrics.classifier_only},
    {method: "Classifier + HMM/Viterbi", ...result.metrics.hmm_viterbi},
  ];
  $("dtable").innerHTML = `<table><tr><th>Method</th><th>Top-1</th><th>Continuity</th><th>Route F1</th><th>Precision</th><th>Recall</th></tr>${
    rows.map(row => `<tr><td>${row.method}</td><td>${(100 * row.top1_accuracy).toFixed(2)}%</td><td>${(100 * row.route_continuity).toFixed(2)}%</td><td>${(100 * row.route_f1).toFixed(2)}%</td><td>${(100 * row.route_precision).toFixed(2)}%</td><td>${(100 * row.route_recall).toFixed(2)}%</td></tr>`).join("")
  }</table>`;
  $("dbars").innerHTML = rows.map(row =>
    `<div class="bar"><span>${row.method} · top-1</span><div><i style="width:${100 * row.top1_accuracy}%"></i></div><b>${(100 * row.top1_accuracy).toFixed(2)}%</b></div>`
  ).join("");
}

async function loadTrips() {
  const payload = await apiTrips();
  state.trips = payload.trips;
  const select = $("trip_select");
  select.innerHTML = state.trips.map(t => `<option value="${t.trip_id}">${t.trip_id} (${t.n_points} points)</option>`).join("");
  select.disabled = state.trips.length === 0;
  $("run").disabled = state.trips.length === 0;
  $("api_out").textContent = `Loaded ${state.trips.length} evaluation trips.`;
}

async function connect() {
  const health = await apiHealth();
  setApiState("backend: connected");
  $("api_out").textContent = `Connected to ${health.model}.`;
  await loadTrips();
}

document.querySelectorAll("nav button").forEach(button => {
  button.onclick = () => {
    document.querySelectorAll("nav button").forEach(item => item.classList.toggle("on", item === button));
    document.querySelectorAll("section").forEach(section => {
      section.hidden = section.id !== `tab-${button.dataset.tab}`;
    });
    if (button.dataset.tab === "map") setTimeout(() => map.invalidateSize(), 50);
  };
});

["t_gps", "t_route"].forEach(id => $(id).onchange = toggleLayers);
$("scrub").oninput = event => selectPoint(Number(event.target.value));
$("load_trips").onclick = async () => {
  try { message(""); await loadTrips(); setApiState("backend: connected"); }
  catch (error) { message(error.message); setApiState("backend: unreachable"); }
};
$("run").onclick = async () => {
  const button = $("run");
  try {
    button.disabled = true; button.textContent = "Running HMM…"; message("");
    state.result = await apiPredict($("trip_select").value);
    renderResult();
  } catch (error) { message(error.message); }
  finally { button.disabled = false; button.textContent = "Run HMM prediction"; }
};
$("api_save").onclick = async () => {
  CONFIG.API_URL = $("api_url").value.trim().replace(/\/$/, "");
  try { await connect(); } catch (error) {
    $("api_out").textContent = `Connection failed: ${error.message}`;
    setApiState("backend: unreachable");
  }
};
$("api_refresh").onclick = async () => {
  try { await loadTrips(); } catch (error) { $("api_out").textContent = error.message; }
};
$("api_metrics").onclick = async () => {
  try {
    const payload = await apiMetrics();
    const rows = payload.rows.map(row => ({
      model: row.method, test_top1: row.top1, route_continuity: row.continuity,
      test_macro_f1: row.route_f1, test_precision: row.route_precision,
      test_recall: row.route_recall, inference_ms_per_obs: row.ms_per_obs,
    }));
    renderModelTable(rows);
  } catch (error) { $("mtable").textContent = error.message; }
};

let modelRows = BASELINE.map(row => ({...row}));
const modelColumns = ["model", "cv_top1_mean", "test_top1", "test_macro_f1", "test_precision", "test_recall", "route_continuity"];
function renderModelTable(rows = modelRows) {
  modelRows = rows;
  const metric = $("metric").value;
  const numeric = rows.filter(row => typeof row[metric] === "number");
  const maximum = Math.max(...numeric.map(row => row[metric]), 1e-9);
  $("mbars").innerHTML = numeric.map(row =>
    `<div class="bar"><span>${row.model}</span><div><i style="width:${100 * row[metric] / maximum}%"></i></div><b>${Number(row[metric]).toFixed(4)}</b></div>`
  ).join("");
  $("mtable").innerHTML = `<table><tr>${modelColumns.map(c => `<th>${c}</th>`).join("")}</tr>${
    rows.map(row => `<tr>${modelColumns.map(c => `<td>${typeof row[c] === "number" ? row[c].toFixed(4) : (row[c] || "")}</td>`).join("")}</tr>`).join("")
  }</table>`;
}
$("metric").onchange = () => renderModelTable();
renderModelTable();
renderDecode();
