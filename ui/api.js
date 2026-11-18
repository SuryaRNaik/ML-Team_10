// Laptop 2 talks to Laptop 1 only through this file.
const needApi = () => {
  const url = CONFIG.API_URL.trim().replace(/\/$/, "");
  if (!url) throw new Error("Set the Laptop 1 API URL in the Connection tab.");
  return url;
};

async function request(path, options = {}) {
  const response = await fetch(needApi() + path, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

async function apiHealth() { return request("/health"); }
async function apiTrips() { return request("/trips"); }
async function apiMetrics() { return request("/metrics"); }
async function apiPredict(tripId) {
  return request("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({trip_id: tripId}),
  });
}
