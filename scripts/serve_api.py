"""Local API for the PathSnap UI. Same JSON contract as the teammate's 09_api.py.
Pipeline: prepare_trip(with_labels=False) -> model probabilities -> HMM/Viterbi. No .route is read."""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from flask import Flask, jsonify, request, Response

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from pathsnap.prepare import prepare_trip                                                  # noqa: E402
from pathsnap.hmm import (HMMParams, RoadGraph, arc_lengths, decode_hmm,                   # noqa: E402
                          classifier_only, reconstruct_route)

NAME = os.environ.get("PATHSNAP_MODEL", "HistGradientBoosting")
MODEL = joblib.load(ROOT / "models" / f"{NAME}.joblib")
FEATURES = json.loads((ROOT / "reports" / "experiment_log.json").read_text(encoding="utf-8"))["features"]
PRM = HMMParams.load(ROOT / "configs" / "hmm.json")
app = Flask(__name__)


@app.after_request
def cors(r):
    r.headers.update({"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*",
                      "Access-Control-Allow-Methods": "GET,POST,OPTIONS"})
    return r


@app.route("/health")
def health():
    return jsonify(ok=True, status="ok", model=NAME + " + HMM")


@app.route("/metrics")
def metrics():
    return Response(pd.read_csv(ROOT / "reports" / "model_comparison.csv").to_json(orient="records"), mimetype="application/json")


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        j = request.get_json(force=True)
        trip = {"trip_id": str(j.get("trip_id", "trip")), "track": np.array(j["track"], float),
                "nodes": np.array(j["nodes"], float), "arcs": np.array(j["arcs"], np.int64), "route": None}
        pts, cands, _ = prepare_trip(trip, with_labels=False)
        if cands.empty:
            return jsonify(route=[], route_geometry=[], points=[], n_points=0)
        cands = cands.copy()
        cands["p"] = MODEL.predict_proba(cands[FEATURES].to_numpy("float32"))[:, 1]
        n = len(pts)
        graph = RoadGraph(trip["arcs"], arc_lengths(trip["nodes"], trip["arcs"], pts.lon.median(), pts.lat.median()))
        top = classifier_only(cands, n)
        dec = decode_hmm(pts, cands, graph, PRM)
        route = reconstruct_route(dec.pred_arc.to_numpy(), graph, PRM)
        nd, a = trip["nodes"], trip["arcs"]
        geom = [[[float(nd[a[r, 0], 1]), float(nd[a[r, 0], 0])], [float(nd[a[r, 1], 1]), float(nd[a[r, 1], 0])]] for r in route]
        raw = pts["raw_idx"].to_numpy()
        def arc(v): return None if pd.isna(v) else int(v)
        points = [{"pt_idx": int(raw[i]), "lon": float(pts.lon.iloc[i]), "lat": float(pts.lat.iloc[i]), "t": float(pts.t.iloc[i]),
                   "arc_top1": arc(top.pred_arc.iloc[i]), "arc_decoded": arc(dec.pred_arc.iloc[i])} for i in range(n)]
        return jsonify(route=[int(r) for r in route], route_geometry=geom, points=points, n_points=n)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
