"""Contract tests for the two-laptop PathSnap API (no real dataset/model needed)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("pathsnap_serve_api", ROOT / "scripts" / "serve_api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class FakeModel:
    def predict_proba(self, features):
        p = np.full(len(features), 0.9)
        return np.column_stack([1.0 - p, p])


@pytest.fixture()
def client_setup(tmp_path, monkeypatch):
    trip_id = "00000052"
    trip_dir = tmp_path / trip_id
    trip_dir.mkdir()
    for suffix in ("track", "nodes", "arcs", "route"):
        (trip_dir / f"{trip_id}.{suffix}").write_text("fixture\n", encoding="utf-8")

    monkeypatch.setattr(api, "EVALUATOR_DIR", tmp_path)
    params = api.HMMParams()
    monkeypatch.setattr(api, "_load_runtime", lambda: (FakeModel(), ["feature"], params))

    trip = {
        "trip_id": trip_id,
        "track": np.array([[77.0002, 12.0, 0], [77.0008, 12.0, 1], [77.0018, 12.0, 2]], dtype=float),
        "nodes": np.array([[77.0, 12.0], [77.001, 12.0], [77.002, 12.0]], dtype=float),
        "arcs": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "route": None,
    }

    def fake_load_trip(folder, with_route=True):
        data = dict(trip)
        if with_route:
            data["route"] = np.array([0, 1], dtype=np.int64)
        return data

    def fake_prepare_trip(data, with_labels=False):
        points = pd.DataFrame({
            "trip_id": [trip_id] * 3,
            "pt_idx": [0, 1, 2],
            "raw_idx": [0, 1, 2],
            "lon": [77.0002, 77.0008, 77.0018],
            "lat": [12.0, 12.0, 12.0],
            "t": [0.0, 1.0, 2.0],
            "x": [0.0, 60.0, 160.0],
            "y": [0.0, 0.0, 0.0],
        })
        if with_labels:
            points["label_ok"] = [1, 1, 1]
            points["true_arc"] = [0, 0, 1]
        candidates = pd.DataFrame({
            "trip_id": [trip_id] * 4,
            "pt_idx": [0, 1, 1, 2],
            "arc_id": [0, 0, 1, 1],
            "rank": [0, 0, 1, 0],
            "along_m": [20.0, 80.0, 20.0, 80.0],
            "feature": [1.0, 1.0, 0.5, 1.0],
        })
        return points, candidates, {"n_points": 3, "n_candidates": 4}

    class DummyGraph:
        def __init__(self, arcs, lengths):
            self.arcs = np.asarray(arcs)
            self.len = np.asarray(lengths)

    monkeypatch.setattr(api, "load_trip", fake_load_trip)
    monkeypatch.setattr(api, "prepare_trip", fake_prepare_trip)
    monkeypatch.setattr(api, "arc_lengths", lambda *args: np.array([100.0, 100.0]))
    monkeypatch.setattr(api, "RoadGraph", DummyGraph)
    monkeypatch.setattr(api, "classifier_only", lambda cands, n: pd.DataFrame({"pt_idx": range(n), "pred_arc": [0, 0, 1]}))
    monkeypatch.setattr(api, "decode_hmm", lambda pts, cands, graph, params: pd.DataFrame({"pt_idx": range(len(pts)), "pred_arc": [0, 0, 1]}))
    monkeypatch.setattr(api, "reconstruct_route", lambda arcs, graph, params: [0, 1])
    monkeypatch.setattr(api, "evaluate", lambda pred, pts, graph, params: {
        "n_obs": 3, "n_hit": 3, "n_consec": 2, "n_connected": 2,
        "route_f1": 0.95, "route_precision": 0.9, "route_recall": 1.0,
    })

    with TestClient(api.app) as client:
        yield client, trip_id


def test_health_reports_ready_model_and_evaluator_data(client_setup):
    client, _ = client_setup
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["evaluator_data_available"] is True


def test_trip_list_returns_only_trip_metadata(client_setup):
    client, trip_id = client_setup
    response = client.get("/trips")
    assert response.status_code == 200
    assert response.json() == {"trips": [{"trip_id": trip_id, "n_points": 3}], "count": 1}
    assert "route" not in response.text.lower()


def test_predict_returns_map_payload_and_server_side_scores_not_reference_labels(client_setup):
    client, trip_id = client_setup
    response = client.post("/predict", json={"trip_id": trip_id})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["trip_id"] == trip_id
    assert payload["n_points"] == 3
    assert len(payload["track"]) == 3
    assert len(payload["predictions"]) == 3
    assert len(payload["route_geometry"]) == 2
    assert payload["metrics"]["hmm_viterbi"]["scored_observations"] == 3
    assert "route" not in payload
    assert "true_arc" not in response.text
    assert "reference_route" not in response.text


def test_trip_id_path_traversal_is_rejected(client_setup):
    client, _ = client_setup
    response = client.post("/predict", json={"trip_id": "../secret"})
    assert response.status_code == 400
