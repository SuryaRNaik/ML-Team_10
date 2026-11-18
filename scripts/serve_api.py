"""Laptop 1 API for the PathSnap two-laptop evaluation architecture.

Laptop 1 owns the trained model, road network, evaluator trips and reference
routes. Laptop 2 sends only a trip_id. Prediction is produced with labels
disabled; the reference route is loaded separately only after inference for
server-side scoring. Reference arc IDs are never returned to the UI.

Run from the repository root:
    python scripts/serve_api.py

Environment variables:
    PATHSNAP_MODEL=RandomForest
    PATHSNAP_API_HOST=0.0.0.0
    PATHSNAP_API_PORT=8000
    PATHSNAP_RAW_ROOT=data/raw
    PATHSNAP_EVALUATOR_DIR=data/raw/evaluator_test
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathsnap.data_io import load_trip  # noqa: E402
from pathsnap.hmm import (  # noqa: E402
    HMMParams,
    RoadGraph,
    arc_lengths,
    classifier_only,
    decode_hmm,
    evaluate,
    reconstruct_route,
)
from pathsnap.prepare import prepare_trip  # noqa: E402

MODEL_NAME = os.environ.get("PATHSNAP_MODEL", "RandomForest")
MODEL_PATH = ROOT / "models" / f"{MODEL_NAME}.joblib"
FEATURE_LOG_PATH = ROOT / "reports" / "experiment_log.json"
HMM_CONFIG_PATH = ROOT / "configs" / "hmm.json"
METRICS_PATH = ROOT / "reports" / "hmm_comparison.csv"
RAW_ROOT = Path(os.environ.get("PATHSNAP_RAW_ROOT", str(ROOT / "data" / "raw"))).expanduser()
EVALUATOR_DIR = Path(
    os.environ.get("PATHSNAP_EVALUATOR_DIR", str(RAW_ROOT / "evaluator_test"))
).expanduser()

app = FastAPI(
    title="PathSnap API",
    description=(
        "GPS map matching and route reconstruction. Laptop 1 keeps evaluator "
        "reference routes private; Laptop 2 receives predictions and scores only."
    ),
    version="1.1.0",
)

# Laptop 2 serves the static UI from a different local origin. No cookies or
# authorization credentials are used by this hackathon demo API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_MODEL: Any | None = None
_FEATURES: list[str] | None = None
_HMM_PARAMS: HMMParams | None = None


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=128)


def _load_runtime():
    """Load the selected model and inference configuration once per process."""
    global _MODEL, _FEATURES, _HMM_PARAMS
    if _MODEL is None:
        missing = [p for p in (MODEL_PATH, FEATURE_LOG_PATH, HMM_CONFIG_PATH) if not p.is_file()]
        if missing:
            raise HTTPException(
                status_code=503,
                detail="Missing backend artifact(s): " + ", ".join(str(p) for p in missing),
            )
        try:
            model = joblib.load(MODEL_PATH)
            log = json.loads(FEATURE_LOG_PATH.read_text(encoding="utf-8"))
            features = log.get("features", [])
            if not features:
                raise ValueError("The experiment log does not contain a feature list.")
            if not hasattr(model, "predict_proba"):
                raise TypeError(f"{MODEL_NAME} does not expose predict_proba().")
            params = HMMParams.load(HMM_CONFIG_PATH)
            _MODEL, _FEATURES, _HMM_PARAMS = model, list(features), params
        except Exception as exc:
            _MODEL, _FEATURES, _HMM_PARAMS = None, None, None
            raise HTTPException(status_code=503, detail=f"Could not load model/configuration: {exc}") from exc
    return _MODEL, _FEATURES, _HMM_PARAMS


def _safe_trip_dir(trip_id: str) -> Path:
    """Resolve a trip folder without allowing traversal or symlink escape."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trip_id):
        raise HTTPException(status_code=400, detail="trip_id must be a simple trip-folder name.")
    root = EVALUATOR_DIR.resolve()
    candidate = (root / trip_id).resolve()
    if candidate.parent != root:
        raise HTTPException(status_code=400, detail="Invalid trip_id path.")
    required = [candidate / f"{trip_id}.{suffix}" for suffix in ("track", "nodes", "arcs", "route")]
    if (
        not candidate.is_dir()
        or not all(path.is_file() for path in required)
        or required[-1].stat().st_size == 0
    ):
        raise HTTPException(status_code=404, detail=f"Evaluator trip not found or incomplete: {trip_id}")
    return candidate


def _trip_ids() -> list[str]:
    """List complete evaluator trips without reading or returning their labels."""
    if not EVALUATOR_DIR.is_dir():
        raise HTTPException(
            status_code=503,
            detail=f"Evaluator data directory not found: {EVALUATOR_DIR}. Set PATHSNAP_EVALUATOR_DIR on Laptop 1.",
        )
    root = EVALUATOR_DIR.resolve()
    result: list[str] = []
    for folder in sorted(EVALUATOR_DIR.iterdir()):
        if not folder.is_dir() or folder.resolve().parent != root:
            continue
        trip_id = folder.name
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trip_id):
            continue
        required = [folder / f"{trip_id}.{suffix}" for suffix in ("track", "nodes", "arcs", "route")]
        if all(path.is_file() for path in required) and required[-1].stat().st_size > 0:
            result.append(trip_id)
    return result


def _load_trip_for_prediction(trip_dir: Path, trip_id: str) -> dict[str, Any]:
    """Load only track/network data. This deliberately does not read .route."""
    try:
        trip = load_trip(trip_dir, with_route=False)
        track = np.asarray(trip.get("track"), dtype=float)
        nodes = np.asarray(trip.get("nodes"), dtype=float)
        arcs = np.asarray(trip.get("arcs"), dtype=np.int64)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read evaluator trip {trip_id}: {exc}") from exc

    if track.ndim != 2 or track.shape[1] != 3 or len(track) == 0:
        raise HTTPException(status_code=422, detail=f"Trip {trip_id} has an invalid or empty .track file.")
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) == 0:
        raise HTTPException(status_code=422, detail=f"Trip {trip_id} has an invalid or empty .nodes file.")
    if arcs.ndim != 2 or arcs.shape[1] != 2 or len(arcs) == 0:
        raise HTTPException(status_code=422, detail=f"Trip {trip_id} has an invalid or empty .arcs file.")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise HTTPException(status_code=422, detail=f"Trip {trip_id} contains non-finite coordinates or timestamps.")
    if np.any(np.abs(track[:, 0]) > 180) or np.any(np.abs(track[:, 1]) > 90):
        raise HTTPException(status_code=422, detail=f"Trip {trip_id} contains invalid GPS coordinates.")
    if np.any(np.abs(nodes[:, 0]) > 180) or np.any(np.abs(nodes[:, 1]) > 90):
        raise HTTPException(status_code=422, detail=f"Trip {trip_id} contains invalid road-node coordinates.")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise HTTPException(status_code=422, detail=f"Trip {trip_id} contains arc endpoints outside its node array.")
    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": None,
    }


def _score_predictions(
    trip_dir: Path,
    predicted: pd.DataFrame,
    points_for_prediction: pd.DataFrame,
    graph: RoadGraph,
    params: HMMParams,
    classifier_pred: pd.DataFrame,
) -> dict[str, Any] | None:
    """Read .route after inference and calculate scores locally on Laptop 1."""
    try:
        labelled_trip = load_trip(trip_dir, with_route=True)
        route = labelled_trip.get("route")
        if route is None or len(route) == 0:
            return None
        labelled_points, _, _ = prepare_trip(labelled_trip, with_labels=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not evaluate the server-side reference route: {exc}") from exc

    # Both passes use the same raw trip and cleaning rules; only label columns differ.
    raw_pred = points_for_prediction["raw_idx"].to_numpy(dtype=int)
    raw_label = labelled_points["raw_idx"].to_numpy(dtype=int)
    if not np.array_equal(raw_pred, raw_label):
        raise HTTPException(status_code=500, detail="Prediction/evaluation point alignment mismatch.")

    classifier_scores = evaluate(classifier_pred, labelled_points, graph, params)
    hmm_scores = evaluate(predicted, labelled_points, graph, params)

    def summarize(scores: dict[str, Any]) -> dict[str, Any]:
        n_obs = int(scores["n_obs"])
        n_consec = int(scores["n_consec"])
        return {
            "top1_accuracy": float(scores["n_hit"] / n_obs) if n_obs else None,
            "route_continuity": float(scores["n_connected"] / n_consec) if n_consec else None,
            "route_f1": float(scores["route_f1"]),
            "route_precision": float(scores["route_precision"]),
            "route_recall": float(scores["route_recall"]),
            "scored_observations": n_obs,
        }

    return {"classifier_only": summarize(classifier_scores), "hmm_viterbi": summarize(hmm_scores)}


def _route_geometry(nodes: np.ndarray, arcs: np.ndarray, route: list[int]) -> list[list[list[float]]]:
    geometry: list[list[list[float]]] = []
    for arc_id in route:
        if arc_id < 0 or arc_id >= len(arcs):
            continue
        start, end = arcs[arc_id]
        a, b = nodes[int(start)], nodes[int(end)]
        # Leaflet expects [latitude, longitude].
        geometry.append([[float(a[1]), float(a[0])], [float(b[1]), float(b[0])]])
    return geometry


def _build_prediction(trip: dict[str, Any], trip_dir: Path) -> dict[str, Any]:
    model, feature_names, params = _load_runtime()
    try:
        # Prediction path is intentionally label-free. Evaluator .route is only
        # opened later by _score_predictions, after all predictions are formed.
        points, candidates, quality = prepare_trip(trip, with_labels=False)
        if points.empty:
            raise HTTPException(status_code=422, detail="No usable GPS observations were found.")
        if candidates.empty:
            raise HTTPException(status_code=422, detail="No candidate road segments were generated for this trip.")

        missing_features = [name for name in feature_names if name not in candidates.columns]
        if missing_features:
            raise HTTPException(status_code=500, detail="Backend is missing feature(s): " + ", ".join(missing_features))

        candidates = candidates.copy()
        candidates["p"] = model.predict_proba(candidates[feature_names].to_numpy(dtype=np.float32))[:, 1]
        lengths = arc_lengths(
            trip["nodes"], trip["arcs"], float(points.lon.median()), float(points.lat.median())
        )
        graph = RoadGraph(trip["arcs"], lengths)
        classifier_pred = classifier_only(candidates, len(points))
        decoded = decode_hmm(points, candidates, graph, params)
        predicted_route = reconstruct_route(decoded.pred_arc.to_numpy(), graph, params)

        # Top-ranked candidate probability is shown as confidence for each GPS point.
        best_candidate = (
            candidates.sort_values(["pt_idx", "p"], ascending=[True, False])
            .drop_duplicates("pt_idx")
            .set_index("pt_idx")["p"]
        )
        top_arcs = classifier_pred.set_index("pt_idx")["pred_arc"]
        decoded_arcs = decoded.set_index("pt_idx")["pred_arc"]
        prediction_rows = []
        track_rows = []
        for row in points.itertuples(index=False):
            idx = int(row.pt_idx)
            top_arc = int(top_arcs.loc[idx])
            decoded_arc = int(decoded_arcs.loc[idx])
            prediction_rows.append({
                "pt_idx": int(row.raw_idx),
                "lon": float(row.lon),
                "lat": float(row.lat),
                "t": float(row.t),
                "arc_top1": top_arc if top_arc >= 0 else None,
                "arc_decoded": decoded_arc if decoded_arc >= 0 else None,
                "probability": float(best_candidate.get(idx, 0.0)),
            })
            track_rows.append([float(row.lon), float(row.lat), float(row.t)])

        # Score only on Laptop 1. No reference route/arc labels are included in
        # the response sent to Laptop 2.
        metrics = _score_predictions(
            trip_dir, decoded, points, graph, params, classifier_pred
        )
        return {
            "trip_id": trip["trip_id"],
            "model": f"{MODEL_NAME} + HMM/Viterbi",
            "n_points": len(track_rows),
            "track": track_rows,
            "predictions": prediction_rows,
            "route_geometry": _route_geometry(trip["nodes"], trip["arcs"], predicted_route),
            "metrics": metrics,
            "data_quality": quality,
        }
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/health")
def health():
    _load_runtime()
    if not EVALUATOR_DIR.is_dir():
        raise HTTPException(status_code=503, detail=f"Evaluator directory not found: {EVALUATOR_DIR}")
    return {
        "ok": True,
        "status": "ok",
        "model": f"{MODEL_NAME} + HMM/Viterbi",
        "evaluator_data_available": True,
    }


@app.get("/trips")
def trips():
    ids = _trip_ids()
    result = []
    for trip_id in ids:
        folder = _safe_trip_dir(trip_id)
        # Only metadata is read here. Reference .route is not opened or returned.
        try:
            trip = load_trip(folder, with_route=False)
            n_points = int(len(trip["track"]))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not inspect trip {trip_id}: {exc}") from exc
        result.append({"trip_id": trip_id, "n_points": n_points})
    return {"trips": result, "count": len(result)}


@app.get("/metrics")
def metrics():
    """Return development HMM-comparison metrics (never evaluator-test labels)."""
    if not METRICS_PATH.is_file():
        raise HTTPException(status_code=404, detail="HMM comparison metrics are not available yet.")
    frame = pd.read_csv(METRICS_PATH)
    frame = frame.astype(object).where(pd.notna(frame), None)
    return {"model": MODEL_NAME, "rows": frame.to_dict(orient="records")}


@app.post("/predict")
def predict(request: PredictionRequest):
    folder = _safe_trip_dir(request.trip_id)
    trip = _load_trip_for_prediction(folder, request.trip_id)
    return _build_prediction(trip, folder)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("PATHSNAP_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("PATHSNAP_API_PORT", "8000")),
        reload=False,
    )
