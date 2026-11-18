"""Where do the classifier / HMM make mistakes?  (student-test or train, never evaluator data)

For every observation with a reliable label whose top-1 arc is wrong, the error is classified as
  reverse    : predicted arc is the opposite-direction twin of the true arc
  adjacent   : predicted arc is directly connected (head-to-tail) to the true arc, either direction
  other      : anything else (parallel street, wrong junction branch, ...)
and the errors are broken down by distance of the point to the end of the true arc ("junction" = <10 m),
GPS gap flag and number of candidates. This tells you which fix to prioritise.

usage: python scripts/05_error_analysis.py [--model HistGradientBoosting | --model-path models/x.joblib] [--partition student_test]
Outputs: reports/error_analysis.csv, reports/error_analysis_per_trip.csv
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data_io import load_trip                              # noqa: E402
from pathsnap.hmm import HMMParams, RoadGraph, arc_lengths, decode_hmm, classifier_only   # noqa: E402
from pathsnap.features2 import add_group_features                   # noqa: E402

PROC, REP, MOD, RAW = ROOT / "data/processed", ROOT / "reports", ROOT / "models", ROOT / "data/raw"


def load_model(path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj["model"], obj["features"], bool(obj.get("group_features", False))
    return obj, json.loads((REP / "experiment_log.json").read_text(encoding="utf-8"))["features"], False


def classify_errors(pred, true, arcs):
    """pred/true: arrays of arc ids. returns array of 'ok' | 'reverse' | 'adjacent' | 'other'."""
    out = np.full(len(pred), "other", dtype=object)
    out[pred == true] = "ok"
    miss = (pred != true) & (pred >= 0)
    a, b = pred[miss], true[miss]
    rev = (arcs[a, 0] == arcs[b, 1]) & (arcs[a, 1] == arcs[b, 0])
    adj = (arcs[a, 1] == arcs[b, 0]) | (arcs[b, 1] == arcs[a, 0]) | (arcs[a, 0] == arcs[b, 0]) | (arcs[a, 1] == arcs[b, 1])
    kind = np.where(rev, "reverse", np.where(adj, "adjacent", "other")).astype(object)
    out[miss] = kind
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HistGradientBoosting")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--partition", default="student_test", choices=["student_test", "train"])
    a = ap.parse_args()
    model, feats, grp = load_model(a.model_path or MOD / f"{a.model}.joblib")
    pts = pd.read_parquet(PROC / f"points_{a.partition}.parquet"); c = pd.read_parquet(PROC / f"candidates_{a.partition}.parquet")
    if grp:
        c = add_group_features(c)
    c["p"] = model.predict_proba(c[feats].to_numpy(np.float32))[:, 1]
    prm = HMMParams.load(ROOT / "configs/hmm.json")
    rows, per_trip = [], []
    for t in sorted(pts.trip_id.unique()):
        p_t = pts[pts.trip_id == t].sort_values("pt_idx"); c_t = c[c.trip_id == t]
        trip = load_trip(RAW / a.partition / t)
        g = RoadGraph(trip["arcs"], arc_lengths(trip["nodes"], trip["arcs"], p_t.lon.median(), p_t.lat.median()))
        n = int(p_t.pt_idx.max()) + 1
        clf, hmm = classifier_only(c_t, n).pred_arc.to_numpy(), decode_hmm(p_t, c_t, g, prm).pred_arc.to_numpy()
        ok = p_t[p_t.label_ok == 1]
        idx, true = ok.pt_idx.to_numpy(), ok.true_arc.to_numpy()
        # distance of the point to the end of the TRUE arc (from the candidate row of the true arc)
        te = c_t[c_t.label == 1].set_index("pt_idx").dist_to_end_m
        d_end = te.reindex(idx).to_numpy()
        ncand = ok.n_cands.to_numpy(); gap = ok.gap_flag.to_numpy()
        for name, pred in (("classifier", clf), ("hmm", hmm)):
            kind = classify_errors(pred[idx], true, g.arcs)
            d = pd.DataFrame({"trip_id": t, "method": name, "kind": kind, "junction": d_end < 10, "gap": gap == 1,
                              "few_cands": ncand <= 2, "many_cands": ncand >= 10})
            rows.append(d)
    d = pd.concat(rows)
    err = d[d.kind != "ok"]
    summ = []
    for m, gdf in d.groupby("method"):
        e = gdf[gdf.kind != "ok"]
        s = dict(method=m, n_obs=len(gdf), accuracy=1 - len(e) / len(gdf), n_errors=len(e),
                 err_reverse=(e.kind == "reverse").mean(), err_adjacent=(e.kind == "adjacent").mean(), err_other=(e.kind == "other").mean(),
                 err_near_junction=e.junction.mean(), err_at_gap=e.gap.mean(), err_many_cands=e.many_cands.mean(),
                 acc_near_junction=1 - (gdf[gdf.junction].kind != "ok").mean(), acc_away_from_junction=1 - (gdf[~gdf.junction].kind != "ok").mean())
        summ.append(s)
    REP.mkdir(exist_ok=True)
    pd.DataFrame(summ).to_csv(REP / "error_analysis.csv", index=False)
    pt = d.groupby(["trip_id", "method"]).apply(lambda x: pd.Series({"accuracy": (x.kind == "ok").mean(), "n_obs": len(x)}), include_groups=False).reset_index()
    pt.to_csv(REP / "error_analysis_per_trip.csv", index=False)
    print(pd.DataFrame(summ).round(4).T.to_string(header=False))


if __name__ == "__main__":
    main()
