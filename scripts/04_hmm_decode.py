"""HMM / Viterbi route reconstruction: classifier-only vs classifier + HMM.

Examples
--------
# evaluate (uses configs/hmm.json if present, else defaults) on the 10 student-test trips
python scripts/04_hmm_decode.py --model HistGradientBoosting

# tune decoder parameters on TRAIN trips only (out-of-fold probabilities), then evaluate on student-test
python scripts/04_hmm_decode.py --model HistGradientBoosting --tune --tune-trials 60

# use a tuned model saved elsewhere (e.g. after Bayesian optimisation)
python scripts/04_hmm_decode.py --model-path models/HistGradientBoosting_tuned.joblib

Needs: data/processed/*.parquet (02_preprocess.py), data/raw/<partition>/<trip>/ (arcs/nodes/route), models/<name>.joblib
Outputs: reports/hmm_comparison.csv, reports/hmm_per_trip.csv, reports/hmm_predictions_<partition>.csv,
         reports/hmm_tuning.csv + configs/hmm.json (with --tune), reports/figures/07_hmm_vs_classifier.png,
         reports/maps/hmm_<partition>_<trip>.html
Evaluator-test data is never read. Decoder parameters are chosen on TRAIN data only.
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data_io import load_trip                                                    # noqa: E402
from pathsnap.hmm import (HMMParams, RoadGraph, arc_lengths, build_lattice, viterbi,      # noqa: E402
                          decode_hmm, classifier_only, reconstruct_route, evaluate)
from pathsnap.viz import trip_map                                                         # noqa: E402

PROC, REP, MOD, RAW, CFG = ROOT / "data/processed", ROOT / "reports", ROOT / "models", ROOT / "data/raw", ROOT / "configs/hmm.json"
SEED = 42


def features():
    f = REP / "experiment_log.json"
    return json.loads(f.read_text(encoding="utf-8"))["features"]


def load_part(part):
    return (pd.read_parquet(PROC / f"points_{part}.parquet"), pd.read_parquet(PROC / f"candidates_{part}.parquet"))


def make_graph(part, trip_id, pts):
    t = load_trip(RAW / part / trip_id)
    alen = arc_lengths(t["nodes"], t["arcs"], pts.lon.median(), pts.lat.median())
    return t, RoadGraph(t["arcs"], alen)


def pool(rows):
    d = pd.DataFrame(rows)
    return dict(top1=d.n_hit.sum() / d.n_obs.sum(), continuity=d.n_connected.sum() / max(d.n_consec.sum(), 1),
                route_f1=d.route_f1.mean(), route_precision=d.route_precision.mean(), route_recall=d.route_recall.mean(),
                n_obs=int(d.n_obs.sum()))


# ------------------------------------------------------------------ tuning (TRAIN only, out-of-fold)
def tune(args):
    """Random search over decoder parameters. Probabilities are OUT-OF-FOLD (a trip is never scored by a model that saw it)
    so the decoder is not tuned on over-confident in-sample probabilities."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold
    spec = importlib.util.spec_from_file_location("train_compare", ROOT / "scripts/03_train_compare.py")
    tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
    FEATS = tc.FEATURES
    pts, cands = load_part("train")
    ok = cands[cands.label_ok == 1]
    tr, _ = tc.sample_observations(ok, pts, args.obs_per_trip, SEED)
    trips = np.array(sorted(pts.trip_id.unique()))
    rng = np.random.default_rng(SEED)
    tune_trips = set(rng.choice(trips, size=min(args.tune_trips, len(trips)), replace=False))
    est = tc.model_zoo()[args.model][2]
    folds = list(GroupKFold(n_splits=5).split(tr, tr.label, tr.trip_id))
    trip_fold = {}
    for k, (_, va) in enumerate(folds):
        for t in tr.trip_id.iloc[va].unique():
            trip_fold[t] = k
    # trips that have no sampled rows can't be assigned -> skip them
    tune_trips = [t for t in sorted(tune_trips) if t in trip_fold]
    print(f"tuning on {len(tune_trips)} train trips, window={args.tune_window} obs, {args.tune_trials} trials")

    probs = {}
    for k, (i_tr, _) in enumerate(folds):
        mine = [t for t in tune_trips if trip_fold[t] == k]
        if not mine:
            continue
        m = clone(est).fit(tr[FEATS].iloc[i_tr].to_numpy(np.float32), tr.label.iloc[i_tr].to_numpy())
        for t in mine:
            probs[t] = m.predict_proba(cands[cands.trip_id == t][FEATS].to_numpy(np.float32))[:, 1]

    base = HMMParams()
    data = []
    for t in tune_trips:
        p_t = pts[pts.trip_id == t].sort_values("pt_idx"); c_t = cands[cands.trip_id == t].copy()
        c_t["p"] = probs[t]
        lo = int(p_t.pt_idx.iloc[0]); hi = lo + args.tune_window
        p_t = p_t[(p_t.pt_idx >= lo) & (p_t.pt_idx < hi)]
        c_t = c_t[(c_t.pt_idx >= lo) & (c_t.pt_idx < hi)]
        _, graph = make_graph("train", t, p_t)
        # window: re-index to start at 0 so evaluate()/decode work on a self-contained sequence
        p_t = p_t.assign(pt_idx=p_t.pt_idx - lo); c_t = c_t.assign(pt_idx=c_t.pt_idx - lo)
        data.append((p_t, c_t, graph, build_lattice(p_t, c_t, graph, base)))

    def score(prm):
        rows = []
        for p_t, c_t, graph, lat in data:
            rows.append(evaluate(decode_hmm(p_t, c_t, graph, prm, lattice=lat), p_t, graph, prm))
        return pool(rows)

    clf = pool([evaluate(classifier_only(c, p.pt_idx.max() + 1), p, g, base) for p, c, g, _ in data])
    print("classifier-only on tuning window:", {k: round(v, 4) for k, v in clf.items() if k != "n_obs"})
    trials = []
    for i in range(args.tune_trials):
        prm = base if i == 0 else HMMParams(
            alpha=float(rng.uniform(0.3, 2.0)), beta_m=float(np.exp(rng.uniform(np.log(5), np.log(200)))),
            log_disc=float(-rng.uniform(4, 30)), uturn_pen=float(rng.uniform(0, 10)),
            gap_break_s=float(rng.choice([10, 20, 30, 60])), max_hops=base.max_hops, max_route_m=base.max_route_m)
        s = score(prm)
        trials.append({**{k: getattr(prm, k) for k in ("alpha", "beta_m", "log_disc", "uturn_pen", "gap_break_s")}, **s})
        print(f"  trial {i:3d}: top1={s['top1']:.4f} cont={s['continuity']:.4f} routeF1={s['route_f1']:.4f}")
    df = pd.DataFrame(trials).sort_values(["top1", "continuity"], ascending=False)
    REP.mkdir(exist_ok=True); df.to_csv(REP / "hmm_tuning.csv", index=False)
    b = df.iloc[0]
    best = HMMParams(alpha=float(b.alpha), beta_m=float(b.beta_m), log_disc=float(b.log_disc), uturn_pen=float(b.uturn_pen),
                     gap_break_s=float(b.gap_break_s))
    CFG.parent.mkdir(exist_ok=True); best.save(CFG)
    print(f"best params saved to {CFG}:", best)
    return best


# ------------------------------------------------------------------ evaluation
def evaluate_partition(args, prm):
    part = args.partition
    pts, cands = load_part(part)
    model = joblib.load(args.model_path or MOD / f"{args.model}.joblib")
    name = Path(args.model_path).stem if args.model_path else args.model
    FEATS = features()
    t0 = time.time()
    cands = cands.assign(p=model.predict_proba(cands[FEATS].to_numpy(np.float32))[:, 1])
    t_model = time.time() - t0
    rows_c, rows_h, per_trip, preds, t_hmm = [], [], [], [], 0.0
    for t in sorted(pts.trip_id.unique()):
        p_t = pts[pts.trip_id == t].sort_values("pt_idx"); c_t = cands[cands.trip_id == t]
        trip, graph = make_graph(part, t, p_t)
        n = int(p_t.pt_idx.max()) + 1
        pc = classifier_only(c_t, n)
        t1 = time.time(); ph = decode_hmm(p_t, c_t, graph, prm); t_hmm += time.time() - t1
        ec, eh = evaluate(pc, p_t, graph, prm), evaluate(ph, p_t, graph, prm)
        rows_c.append(ec); rows_h.append(eh)
        per_trip.append(dict(trip_id=t, n_obs=ec["n_obs"], clf_top1=ec["n_hit"] / ec["n_obs"], hmm_top1=eh["n_hit"] / eh["n_obs"],
                             clf_continuity=ec["n_connected"] / max(ec["n_consec"], 1), hmm_continuity=eh["n_connected"] / max(eh["n_consec"], 1),
                             clf_route_f1=ec["route_f1"], hmm_route_f1=eh["route_f1"], n_seg_breaks=int(ph.seg_break.sum()),
                             n_filled=int(ph.filled.sum())))
        preds.append(pd.DataFrame({"trip_id": t, "pt_idx": pc.pt_idx, "clf_arc": pc.pred_arc, "hmm_arc": ph.pred_arc}))
        print(f"  {t}: top1 {per_trip[-1]['clf_top1']:.4f} -> {per_trip[-1]['hmm_top1']:.4f} | continuity "
              f"{per_trip[-1]['clf_continuity']:.4f} -> {per_trip[-1]['hmm_continuity']:.4f}")
    C, Hh = pool(rows_c), pool(rows_h)
    comp = pd.DataFrame([dict(method=f"Classifier only ({name})", **C), dict(method=f"Classifier + HMM/Viterbi ({name})", **Hh)])
    comp["decode_time_s"] = [t_model, t_model + t_hmm]
    comp["ms_per_obs"] = 1000 * comp.decode_time_s / C["n_obs"]
    REP.mkdir(exist_ok=True)
    comp.to_csv(REP / "hmm_comparison.csv", index=False)
    pt = pd.DataFrame(per_trip); pt.to_csv(REP / "hmm_per_trip.csv", index=False)
    pd.concat(preds).to_csv(REP / f"hmm_predictions_{part}.csv", index=False)
    print("\n", comp.round(4).to_string(index=False))

    # figure (real results only)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); y = np.arange(len(pt)); w = .38
    ax[0].barh(y - w / 2, pt.clf_top1, w, label="classifier only", color="#9aa0a6"); ax[0].barh(y + w / 2, pt.hmm_top1, w, label="+ HMM/Viterbi", color="#2166ac")
    ax[0].set_yticks(y); ax[0].set_yticklabels(pt.trip_id); ax[0].set_title(f"Top-1 accuracy per {part} trip"); ax[0].legend(loc="lower right")
    ax[1].barh(y - w / 2, pt.clf_continuity, w, color="#9aa0a6"); ax[1].barh(y + w / 2, pt.hmm_continuity, w, color="#2166ac")
    ax[1].set_yticks(y); ax[1].set_yticklabels(pt.trip_id); ax[1].set_title("Route continuity per trip")
    for a in ax: a.invert_yaxis(); a.grid(alpha=.3)
    (REP / "figures").mkdir(exist_ok=True); plt.tight_layout(); plt.savefig(REP / "figures/07_hmm_vs_classifier.png", dpi=140); plt.close()

    # maps: best-improvement trip, worst (HMM did not help) trip, or those given by --map-trips
    pt["gain"] = pt.hmm_top1 - pt.clf_top1
    chosen = args.map_trips or [pt.sort_values("gain").trip_id.iloc[-1], pt.sort_values("gain").trip_id.iloc[0]]
    allp = pd.concat(preds)
    (REP / "maps").mkdir(parents=True, exist_ok=True)
    for t in dict.fromkeys(chosen):
        p_t = pts[pts.trip_id == t]; trip, graph = make_graph(part, t, p_t)
        route = reconstruct_route(allp[allp.trip_id == t].sort_values("pt_idx").hmm_arc.to_numpy(), graph, prm)
        near = np.unique(cands[cands.trip_id == t].arc_id.to_numpy())
        out = REP / "maps" / f"hmm_{part}_{t}.html"
        trip_map(trip, p_t.rename(columns={}), network_arc_ids=near, pred_arcs=route, out_html=str(out))
        print("map ->", out)
    return comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HistGradientBoosting")
    ap.add_argument("--model-path", default=None, help="explicit .joblib (e.g. Bayesian-tuned model)")
    ap.add_argument("--partition", default="student_test", choices=["student_test", "train"])
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-trials", type=int, default=60)
    ap.add_argument("--tune-trips", type=int, default=12)
    ap.add_argument("--tune-window", type=int, default=800, help="contiguous observations per trip used while tuning")
    ap.add_argument("--obs-per-trip", type=int, default=150)
    ap.add_argument("--map-trips", nargs="*", default=None)
    a = ap.parse_args()
    prm = tune(a) if a.tune else HMMParams.load(CFG)
    print("HMM params:", prm)
    evaluate_partition(a, prm)