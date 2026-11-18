"""Train and compare >= 5 candidate-scoring models.

Task: for every (GPS point, candidate arc) row predict P(this arc is the true road).
Protocol (follows the hackathon guidelines):
  * fit + group cross-validation on the 80 TRAIN trips only (GroupKFold by trip_id, no trip split across folds)
  * development comparison on the 10 STUDENT-TEST trips (never used for fitting)
  * only label_ok == 1 points are used (reference route actually covers the point)
  * evaluator-test data is never touched
Run:  python scripts/03_train_compare.py [--obs-per-trip 300] [--folds 5] [--only RandomForest HistGB]
Outputs: reports/model_comparison.csv, reports/cv_fold_scores.csv, reports/experiment_log.json,
         reports/figures/06_*.png, models/<name>.joblib
"""
import argparse, json, sys, time, platform
from pathlib import Path
import numpy as np, pandas as pd
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC, REP, MOD, RAW = ROOT / "data/processed", ROOT / "reports", ROOT / "models", ROOT / "data/raw"
SEED = 42
FEATURES = ["dist", "dist_rel", "dist_prev", "dist_next", "dist_ctx_mean", "t_frac", "along_m",
            "dist_to_end_m", "arc_len", "cos_ang", "abs_cos_ang", "rank", "n_cands", "speed",
            "heading_valid", "gap_flag", "out_deg_to", "in_deg_from", "has_reverse"]


def model_zoo():
    """name -> (algorithm family, key configuration, estimator). Edit hyper-parameters HERE."""
    return {
        "LogisticRegression": ("Linear", "C=1, class_weight=balanced, scaled",
            make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=300))),
        "GaussianNB": ("Probabilistic", "var_smoothing=1e-9, scaled",
            make_pipeline(StandardScaler(), GaussianNB())),
        "DecisionTree": ("Tree", "max_depth=12, min_samples_leaf=20",
            DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED)),
        "RandomForest": ("Bagging ensemble", "100 trees, max_depth=16, min_samples_leaf=5",
            RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "ExtraTrees": ("Randomised ensemble", "100 trees, max_depth=18, min_samples_leaf=5",
            ExtraTreesClassifier(n_estimators=100, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "HistGradientBoosting": ("Boosting", "lr=0.1, max_iter=200, max_leaf_nodes=31",
            HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, random_state=SEED)),
        "MLP": ("Neural network", "(64,32) ReLU, adam, early_stopping, scaled",
            make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                          max_iter=60, random_state=SEED))),
    }


# ---------------------------------------------------------------- metrics
_ARCS = {}
def arcs_of(part, trip):
    k = (part, trip)
    if k not in _ARCS:
        _ARCS[k] = pd.read_csv(RAW / part / trip / f"{trip}.arcs", sep=r"\s+", header=None).to_numpy()
    return _ARCS[k]


def observation_metrics(df, proba, n_points, part):
    """Top-1 accuracy over observations (a point whose true arc is not a candidate counts as wrong)
    and route continuity of the top-1 arcs (same arc or head-to-tail connected on consecutive points)."""
    d = df[["trip_id", "pt_idx", "arc_id", "label"]].copy()
    d["p"] = proba
    top = d.sort_values(["trip_id", "pt_idx", "p"], ascending=[True, True, False]).drop_duplicates(["trip_id", "pt_idx"])
    top1 = top.label.sum() / n_points
    ok = tot = 0
    for trip, g in top.groupby("trip_id"):
        a = arcs_of(part, trip); idx, arc = g.pt_idx.to_numpy(), g.arc_id.to_numpy()
        consec = np.diff(idx) == 1
        a0, a1 = arc[:-1][consec], arc[1:][consec]
        ok += int(((a0 == a1) | (a[a0, 1] == a[a1, 0])).sum()); tot += int(consec.sum())
    return float(top1), float(ok / max(tot, 1))


def candidate_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(macro_f1=f1_score(y, pred, average="macro"),
                precision_pos=precision_score(y, pred, zero_division=0),
                recall_pos=recall_score(y, pred, zero_division=0),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))


def sample_observations(c, pts, per_trip, seed):
    """Sample whole observations (all candidates of a point stay together)."""
    ok = pts[pts.label_ok == 1][["trip_id", "pt_idx"]]
    ok = ok.sample(frac=1.0, random_state=seed)                       # shuffle, then keep first N per trip
    s = ok[ok.groupby("trip_id").cumcount() < per_trip]
    return c.merge(s, on=["trip_id", "pt_idx"], how="inner"), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-per-trip", type=int, default=150, help="max training observations sampled per trip")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    REP.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True); (REP / "figures").mkdir(exist_ok=True)

    ctr = pd.read_parquet(PROC / "candidates_train.parquet"); ptr = pd.read_parquet(PROC / "points_train.parquet")
    cte = pd.read_parquet(PROC / "candidates_student_test.parquet"); pte = pd.read_parquet(PROC / "points_student_test.parquet")
    ctr = ctr[ctr.label_ok == 1]; cte = cte[cte.label_ok == 1]
    tr, sampled = sample_observations(ctr, ptr, a.obs_per_trip, SEED); n_tr = len(sampled)
    n_te = int((pte.label_ok == 1).sum())
    print(f"train sample: {n_tr:,} observations / {len(tr):,} candidate rows / {tr.trip_id.nunique()} trips | "
          f"student-test: {n_te:,} observations / {len(cte):,} rows")
    X, y, g = tr[FEATURES].to_numpy(np.float32), tr.label.to_numpy(), tr.trip_id.to_numpy()
    Xte, yte = cte[FEATURES].to_numpy(np.float32), cte.label.to_numpy()

    zoo = {k: v for k, v in model_zoo().items() if not a.only or k in a.only}
    gkf = GroupKFold(n_splits=a.folds)
    rows, folds_log = [], []
    for name, (family, cfg, est) in zoo.items():
        print(f"\n=== {name} [{family}] {cfg}")
        fold_f1, fold_top1 = [], []
        for k, (i_tr, i_va) in enumerate(gkf.split(X, y, g)):
            m = clone(est)
            m.fit(X[i_tr], y[i_tr]); p = m.predict_proba(X[i_va])[:, 1]
            va = tr.iloc[i_va]
            # denominator = ALL sampled observations of the validation trips (points with no candidate count as wrong)
            n_va_pts = int(sampled.trip_id.isin(set(va.trip_id)).sum())
            f1 = candidate_metrics(y[i_va], p)["macro_f1"]
            t1, _ = observation_metrics(va, p, n_va_pts, "train")
            fold_f1.append(f1); fold_top1.append(t1)
            folds_log.append(dict(model=name, fold=k, macro_f1=f1, top1=t1))
            print(f"  fold {k}: macro-F1={f1:.4f}  top-1={t1:.4f}")
        m = clone(est); t0 = time.time(); m.fit(X, y); train_s = time.time() - t0
        t0 = time.time(); p = m.predict_proba(Xte)[:, 1]; inf_s = time.time() - t0
        cm = candidate_metrics(yte, p); top1, cont = observation_metrics(cte, p, n_te, "student_test")
        # train-set score (for over/under-fitting discussion)
        ptrn = m.predict_proba(X)[:, 1]; train_f1 = candidate_metrics(y, ptrn)["macro_f1"]
        joblib.dump(m, MOD / f"{name}.joblib")
        row = dict(model=name, family=family, config=cfg, cv_macro_f1_mean=np.mean(fold_f1), cv_macro_f1_std=np.std(fold_f1),
                   cv_top1_mean=np.mean(fold_top1), train_macro_f1=train_f1, test_top1=top1, test_macro_f1=cm["macro_f1"],
                   test_precision=cm["precision_pos"], test_recall=cm["recall_pos"], test_log_loss=cm["log_loss"],
                   route_continuity=cont, train_time_s=train_s, inference_time_s=inf_s,
                   inference_ms_per_obs=1000 * inf_s / n_te)
        rows.append(row); print("  ->", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("config",)})
        pd.DataFrame(rows).to_csv(REP / "model_comparison.csv", index=False)   # save progressively

    res = pd.DataFrame(rows); pd.DataFrame(folds_log).to_csv(REP / "cv_fold_scores.csv", index=False)
    (REP / "experiment_log.json").write_text(json.dumps(dict(
        seed=SEED, features=FEATURES, folds=a.folds, obs_per_trip=a.obs_per_trip, n_train_obs=int(n_tr),
        n_train_rows=int(len(tr)), n_test_obs=n_te, python=platform.python_version(),
        models={k: dict(family=v[0], config=v[1]) for k, v in zoo.items()}), indent=2), encoding="utf-8")

    # ---- figures (real results only)
    r = res.set_index("model")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    ax[0].barh(r.index, r.cv_macro_f1_mean, xerr=r.cv_macro_f1_std, color="#2166ac"); ax[0].set_title("CV macro-F1 (grouped by trip)")
    w = .38; yy = np.arange(len(r))
    ax[1].barh(yy - w / 2, r.test_top1, w, label="Top-1 accuracy", color="#1a9850")
    ax[1].barh(yy + w / 2, r.test_macro_f1, w, label="Macro-F1", color="#d6604d"); ax[1].set_yticks(yy); ax[1].set_yticklabels(r.index)
    ax[1].set_title("Student-test: Top-1 accuracy and macro-F1"); ax[1].legend(loc="lower right")
    ax[2].barh(r.index, r.test_log_loss, color="#9970ab"); ax[2].set_title("Student-test log loss (lower = better)")
    for x in ax: x.invert_yaxis(); x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(REP / "figures/06_model_comparison.png", dpi=140); plt.close()

    show = res[["model", "cv_macro_f1_mean", "test_top1", "test_macro_f1", "test_log_loss", "route_continuity", "inference_ms_per_obs"]]
    print("\n", show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
