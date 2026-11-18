# PathSnap UI
Static frontend (no build step). Run: `cd ui && python -m http.server 8080`, open http://localhost:8080.
Backend: `python scripts/serve_api.py` (port 8000), then set the URL in the Connection tab.
Contract: GET /health, POST /predict (JSON: trip_id, track, nodes, arcs; no .route), optional GET /metrics. CORS enabled.
