"""Laptop 1 API for the PathSnap two-system deployment.

The API owns the raw trips, model, reference routes and HMM decoder. The UI
only sends a trip ID and receives GPS coordinates, predictions, route geometry
and aggregate evaluation metrics; reference arcs never leave this process.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
RAW_ROOT = Path(os.environ.get("PATHSNAP_RAW_ROOT", str(ROOT / "data" / "raw")))
STUDENT_ROOT = Path(os.environ.get("PATHSNAP_STUDENT_ROOT", str(RAW_ROOT / "student_test")))
MODEL = joblib.load(ROOT / "models" / f"{MODEL_NAME}.joblib")
FEATURES = json.loads((ROOT / "reports" / "experiment_log.json").read_text(encoding="utf-8"))["features"]
PARAMS = HMMParams.load(ROOT / "configs" / "hmm.json")
app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    trip_id: str


def _trip_dir(trip_id: str) -> Path:
    if not trip_id or Path(trip_id).name != trip_id:
        raise HTTPException(status_code=400, detail="trip_id must be a directory name")
    path = STUDENT_ROOT / trip_id
    required = [path / f"{trip_id}.{suffix}" for suffix in ("track", "nodes", "arcs", "route")]
    if not path.is_dir() or not all(p.is_file() for p in required):
        raise HTTPException(status_code=404, detail=f"student-test trip not found: {trip_id}")
    return path


def _trip_ids() -> list[str]:
    return sorted(
        p.name
        for p in STUDENT_ROOT.iterdir()
        if p.is_dir()
        and (p / f"{p.name}.track").is_file()
        and (p / f"{p.name}.nodes").is_file()
        and (p / f"{p.name}.arcs").is_file()
        and (p / f"{p.name}.route").is_file()
    )


@app.get("/health")
def health():
    return {"ok": True, "status": "ok", "model": f"{MODEL_NAME} + HMM/Viterbi"}


@app.get("/trips")
def trips():
    result = []
    for trip_id in _trip_ids():
        trip = load_trip(STUDENT_ROOT / trip_id, with_route=False)
        result.append({"trip_id": trip_id, "n_points": int(len(trip["track"]))})
    return {"trips": result}


@app.get("/metrics")
def metrics():
    path = ROOT / "reports" / "hmm_comparison.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="final HMM metrics are not available")
    return {"model": MODEL_NAME, "rows": pd.read_csv(path).to_dict(orient="records")}


def _geometry(nodes: np.ndarray, arcs: np.ndarray, route: list[int]) -> list[list[list[float]]]:
    return [
        [
            [float(nodes[arcs[arc_id, 0], 1]), float(nodes[arcs[arc_id, 0], 0])],
            [float(nodes[arcs[arc_id, 1], 1]), float(nodes[arcs[arc_id, 1], 0])],
        ]
        for arc_id in route
    ]


@app.post("/predict")
def predict(request: PredictionRequest):
    trip_id = request.trip_id
    trip_dir = _trip_dir(trip_id)
    trip = load_trip(trip_dir, with_route=True)
    points, candidates, _ = prepare_trip(trip, with_labels=True)
    if candidates.empty:
        raise HTTPException(status_code=422, detail="trip has no generated road candidates")

    candidates = candidates.copy()
    candidates["p"] = MODEL.predict_proba(candidates[FEATURES].to_numpy("float32"))[:, 1]
    lengths = arc_lengths(trip["nodes"], trip["arcs"], points.lon.median(), points.lat.median())
    graph = RoadGraph(trip["arcs"], lengths)
    classifier = classifier_only(candidates, len(points))
    decoded = decode_hmm(points, candidates, graph, PARAMS)
    classifier_evaluation = evaluate(classifier, points, graph, PARAMS)
    evaluation = evaluate(decoded, points, graph, PARAMS)
    route = reconstruct_route(decoded.pred_arc.to_numpy(), graph, PARAMS)

    confidence = candidates.sort_values(["pt_idx", "p"], ascending=[True, False])
    confidence = confidence.drop_duplicates("pt_idx").set_index("pt_idx")["p"]
    prediction_rows = []
    for i, row in points.reset_index(drop=True).iterrows():
        prediction_rows.append(
            {
                "pt_idx": int(row.pt_idx),
                "lon": float(row.lon),
                "lat": float(row.lat),
                "t": float(row.t),
                "arc_top1": int(classifier.pred_arc.iloc[i]),
                "arc_decoded": int(decoded.pred_arc.iloc[i]),
                "probability": float(confidence.get(int(row.pt_idx), 0.0)),
            }
        )

    return {
        "trip_id": trip_id,
        "model": f"{MODEL_NAME} + HMM/Viterbi",
        "n_points": len(prediction_rows),
        "track": [[float(row.lon), float(row.lat), float(row.t)] for row in points.itertuples()],
        "route_geometry": _geometry(trip["nodes"], trip["arcs"], route),
        "predictions": prediction_rows,
        "metrics": {
            "usable_observations": evaluation["n_obs"],
            "classifier_only": {
                "top1_accuracy": classifier_evaluation["n_hit"] / max(classifier_evaluation["n_obs"], 1),
                "route_continuity": classifier_evaluation["n_connected"] / max(classifier_evaluation["n_consec"], 1),
                "route_precision": classifier_evaluation["route_precision"],
                "route_recall": classifier_evaluation["route_recall"],
                "route_f1": classifier_evaluation["route_f1"],
            },
            "hmm_viterbi": {
                "top1_accuracy": evaluation["n_hit"] / max(evaluation["n_obs"], 1),
                "route_continuity": evaluation["n_connected"] / max(evaluation["n_consec"], 1),
                "route_precision": evaluation["route_precision"],
                "route_recall": evaluation["route_recall"],
                "route_f1": evaluation["route_f1"],
            },
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("PATHSNAP_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("PATHSNAP_API_PORT", "8000")),
        reload=False,
    )
