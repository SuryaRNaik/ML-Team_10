"""Bayesian hyper-parameter optimisation (Optuna TPE) for the selected model + relative-feature upgrade.

Model            : HistGradientBoostingClassifier (best log-loss / top-1 in 03_train_compare, fast, good probabilities for the HMM)
Features         : the 19 base features + 10 per-point *relative* features (src/pathsnap/features2.py)
Objective        : mean observation-level Top-1 accuracy over group-K-fold CV (GroupKFold by trip_id) on the 80 TRAIN trips.
                   Top-1 is the quantity the route is built from, and it is what we are scored on; log-loss and macro-F1 are reported too.
Search space     : learning_rate, max_iter, max_leaf_nodes, max_depth, min_samples_leaf, l2_regularization, max_bins
Trial 0          : the untuned configuration of 03_train_compare.py, so the search is guaranteed to be compared with a baseline.
Evaluation       : untuned vs tuned on the SAME features / partitions: CV top-1, CV macro-F1, student-test top-1 / macro-F1 / log-loss,
                   train + inference time. Evaluator-test data is never read.

usage: python scripts/06_bayes_opt.py [--trials 40] [--obs-per-trip 400] [--folds 5]
Outputs: reports/bayes_trials.csv, reports/bayes_best_params.json, reports/bayes_comparison.csv,
         reports/figures/08_bayes_search.png, models/HistGradientBoosting_tuned.joblib  (dict: model, features, params)
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features2 import add_group_features, GROUP_FEATURES   # noqa: E402

PROC, REP, MOD = ROOT / "data/processed", ROOT / "reports", ROOT / "models"
SEED = 42
spec = importlib.util.spec_from_file_location("train_compare", ROOT / "scripts/03_train_compare.py")
tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
FEATS = tc.FEATURES + GROUP_FEATURES
BASELINE = dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=1e-3, max_bins=255)


def top1(df, p, n_pts):
    d = pd.DataFrame({"k": df.trip_id.to_numpy(), "pt": df.pt_idx.to_numpy(), "p": p, "y": df.label.to_numpy()})
    best = d.groupby(["k", "pt"], sort=False).p.idxmax()
    return float(d.y.loc[best].sum() / n_pts)


def make(params):
    return HistGradientBoostingClassifier(random_state=SEED, **params)


def cv_scores(tr, sampled, params, folds):
    X, y, g = tr[FEATS].to_numpy(np.float32), tr.label.to_numpy(), tr.trip_id.to_numpy()
    t1, f1 = [], []
    for i_tr, i_va in GroupKFold(n_splits=folds).split(X, y, g):
        m = make(params).fit(X[i_tr], y[i_tr]); p = m.predict_proba(X[i_va])[:, 1]
        va = tr.iloc[i_va]
        t1.append(top1(va, p, int(sampled.trip_id.isin(set(va.trip_id)).sum())))
        f1.append(f1_score(y[i_va], (p >= .5).astype(int), average="macro"))
    return float(np.mean(t1)), float(np.std(t1)), float(np.mean(f1))


def test_scores(model, X, y, te, n_te):
    t0 = time.time(); p = model.predict_proba(X)[:, 1]; inf = time.time() - t0
    pred = (p >= .5).astype(int)
    return dict(test_top1=top1(te, p, n_te), test_macro_f1=f1_score(y, pred, average="macro"), test_precision=precision_score(y, pred, zero_division=0),
                test_recall=recall_score(y, pred, zero_division=0), test_log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
                inference_time_s=inf, inference_ms_per_obs=1000 * inf / n_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--obs-per-trip", type=int, default=400, help="observations sampled per train trip (03 used 150)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--timeout-min", type=float, default=None)
    a = ap.parse_args()
    REP.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True); (REP / "figures").mkdir(exist_ok=True)

    ptr, ctr = pd.read_parquet(PROC / "points_train.parquet"), pd.read_parquet(PROC / "candidates_train.parquet")
    pte, cte = pd.read_parquet(PROC / "points_student_test.parquet"), pd.read_parquet(PROC / "candidates_student_test.parquet")
    tr, sampled = tc.sample_observations(ctr[ctr.label_ok == 1], ptr, a.obs_per_trip, SEED)
    tr = add_group_features(tr).reset_index(drop=True)
    cte = add_group_features(cte[cte.label_ok == 1]).reset_index(drop=True)
    n_te = int((pte.label_ok == 1).sum())
    Xte, yte = cte[FEATS].to_numpy(np.float32), cte.label.to_numpy()
    print(f"train: {len(sampled):,} obs / {len(tr):,} rows / {tr.trip_id.nunique()} trips | student-test: {n_te:,} obs | features: {len(FEATS)}")

    def objective(trial):
        p = dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                 max_iter=trial.suggest_int("max_iter", 100, 600, step=50),
                 max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
                 max_depth=trial.suggest_int("max_depth", 4, 20),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200, log=True),
                 l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
                 max_bins=trial.suggest_int("max_bins", 64, 255))
        t0 = time.time(); m, s, f = cv_scores(tr, sampled, p, a.folds)
        trial.set_user_attr("cv_top1_std", s); trial.set_user_attr("cv_macro_f1", f); trial.set_user_attr("cv_time_s", time.time() - t0)
        return m

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                study_name="hgb_top1")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.enqueue_trial({**BASELINE, "max_depth": 20})        # seed TPE with (almost) the untuned config; depth 20 is ~unconstrained, l2 1e-3 ~ none
    def cb(st, tr_):
        print(f"  trial {tr_.number:3d}: cv top-1={tr_.value:.4f} (best {st.best_value:.4f})  {tr_.user_attrs['cv_time_s']:.0f}s", flush=True)
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, timeout=None if a.timeout_min is None else a.timeout_min * 60, callbacks=[cb])
    search_s = time.time() - t0

    df = study.trials_dataframe(); df.to_csv(REP / "bayes_trials.csv", index=False)
    best = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in study.best_params.items()}
    (REP / "bayes_best_params.json").write_text(json.dumps(dict(best_params=best, best_cv_top1=study.best_value, n_trials=len(study.trials),
                                                                search_seconds=search_s, seed=SEED, objective="mean grouped-CV observation top-1",
                                                                obs_per_trip=a.obs_per_trip, folds=a.folds, features=FEATS), indent=2), encoding="utf-8")
    print("best:", best, f"cv top-1 {study.best_value:.4f}  ({search_s/60:.1f} min)")

    # ---- untuned vs tuned on identical data / metrics
    X, y = tr[FEATS].to_numpy(np.float32), tr.label.to_numpy()
    rows = []
    for name, params in (("HGB untuned (03 config, + relative feats)", {**BASELINE, "l2_regularization": 0.0}), ("HGB Bayesian-tuned", best)):
        cvm, cvs, cvf = cv_scores(tr, sampled, params, a.folds)
        t0 = time.time(); m = make(params).fit(X, y); train_s = time.time() - t0
        rows.append(dict(model=name, config=json.dumps(params), cv_top1_mean=cvm, cv_top1_std=cvs, cv_macro_f1_mean=cvf, train_time_s=train_s,
                         **test_scores(m, Xte, yte, cte, n_te)))
        if "tuned" in name and "untuned" not in name:
            joblib.dump(dict(model=m, features=FEATS, group_features=True, params=params, seed=SEED), MOD / "HistGradientBoosting_tuned.joblib")
    res = pd.DataFrame(rows); res.to_csv(REP / "bayes_comparison.csv", index=False)
    print("\n", res.drop(columns=["config"]).round(4).T.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    v = df["value"].to_numpy(); ax[0].plot(v, "o", alpha=.6, label="trial"); ax[0].plot(np.maximum.accumulate(v), "-", label="best so far")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV top-1"); ax[0].set_title("Bayesian (TPE) search history"); ax[0].legend(); ax[0].grid(alpha=.3)
    r = res.set_index("model"); xx = np.arange(2); w = .38
    ax[1].bar(xx - w / 2, r.cv_top1_mean, w, label="CV top-1"); ax[1].bar(xx + w / 2, r.test_top1, w, label="student-test top-1")
    ax[1].set_xticks(xx); ax[1].set_xticklabels(["untuned", "tuned"]); ax[1].set_ylim(max(0, r[["cv_top1_mean", "test_top1"]].min().min() - .03), 1)
    ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title("Untuned vs tuned")
    plt.tight_layout(); plt.savefig(REP / "figures/08_bayes_search.png", dpi=140); plt.close()


if __name__ == "__main__":
    main()
