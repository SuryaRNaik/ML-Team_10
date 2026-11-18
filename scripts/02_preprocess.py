"""Run preprocessing for every trip of a partition and write parquet tables.

usage: python scripts/02_preprocess.py [--partitions train student_test] [--raw data/raw] [--out data/processed]
Outputs (per partition): points_<p>.parquet, candidates_<p>.parquet, trip_summary_<p>.csv
"""
import argparse, json, sys, time
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data_io import load_trip, list_trips      # noqa: E402
from pathsnap.prepare import prepare_trip, DEFAULT_CFG  # noqa: E402


def work(d):
    return prepare_trip(load_trip(d), DEFAULT_CFG, with_labels=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--raw", default=str(ROOT / "data/raw"))
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "preprocess_config_used.json").write_text(json.dumps(DEFAULT_CFG, indent=2), encoding="utf-8")
    for part in a.partitions:
        t0 = time.time()
        res = Parallel(n_jobs=2)(delayed(work)(d) for d in list_trips(Path(a.raw) / part))
        pts = pd.concat([r[0] for r in res], ignore_index=True)
        cands = pd.concat([r[1] for r in res], ignore_index=True)
        info = pd.DataFrame([r[2] for r in res])
        pts.to_parquet(out / f"points_{part}.parquet", index=False)
        cands.to_parquet(out / f"candidates_{part}.parquet", index=False)
        info.to_csv(out / f"trip_summary_{part}.csv", index=False)
        print(f"[{part}] trips={len(info)} points={len(pts):,} candidates={len(cands):,} "
              f"pos_rate={cands.label.mean():.4f} miss_rate={1-pts.true_in_cands.mean():.5f} "
              f"label_ok={pts.label_ok.mean():.4f} ({time.time()-t0:.0f}s)")
