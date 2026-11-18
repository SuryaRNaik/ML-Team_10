"""Dataset inventory + EDA figures + Folium maps. Run AFTER 02_preprocess.py.
Outputs -> reports/ (dataset_inventory.csv, figures/*.png, maps/*.html, data_quality_log.md)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data_io import load_trip, list_trips   # noqa: E402
from pathsnap.viz import trip_map                    # noqa: E402

RAW, PROC, REP = ROOT / "data/raw", ROOT / "data/processed", ROOT / "reports"
PARTS = ["train", "student_test"]
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"train": "#2166ac", "student_test": "#d6604d"}

# ------------------------------------------------------------------ inventory
rows = []
for part in PARTS:
    for d in list_trips(RAW / part):
        tr = load_trip(d); t, nodes, arcs, route = tr["track"], tr["nodes"], tr["arcs"], tr["route"]
        dt = np.diff(t[:, 2]); m = len(nodes)
        g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(m, m))
        ncomp, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab); deg = np.bincount(arcs.ravel(), minlength=m)
        keys = set((arcs[:, 0].astype(np.int64) * m + arcs[:, 1]).tolist())
        rev = np.mean([(int(b) * m + int(a)) in keys for a, b in arcs[:2000]])
        rconn = np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0]) if len(route) > 1 else np.nan
        rows.append(dict(partition=part, trip_id=d.name, n_points=len(t), n_nodes=m, n_arcs=len(arcs),
            n_route_arcs=len(route), n_route_unique=len(set(route.tolist())),
            n_nan=int(np.isnan(t).sum()), lon_min=t[:, 0].min(), lon_max=t[:, 0].max(), lat_min=t[:, 1].min(),
            lat_max=t[:, 1].max(), duration_s=t[-1, 2] - t[0, 2], pct_dt_1s=100 * (dt == 1).mean(),
            median_dt=np.median(dt), max_dt_s=dt.max(), n_dt_gt5=int((dt > 5).sum()), n_dt_le0=int((dt <= 0).sum()),
            net_components=ncomp, net_largest_comp_pct=100 * sizes.max() / m, isolated_nodes=int((deg == 0).sum()),
            pct_arcs_with_reverse=100 * rev, route_connected_pct=100 * rconn,
            route_arc_ids_valid=bool(route.max() < len(arcs)), arcs_node_ids_valid=bool(arcs.max() < m)))
inv = pd.DataFrame(rows)
pts = {p: pd.read_parquet(PROC / f"points_{p}.parquet") for p in PARTS}
cands = {p: pd.read_parquet(PROC / f"candidates_{p}.parquet") for p in PARTS}
trip_noise = pd.concat([pts[p].groupby("trip_id").agg(median_gps_to_road_m=("d_nearest", "median"),
        track_len_km=("speed", lambda s: s.sum() / 1000), label_ok_pct=("label_ok", lambda s: 100 * s.mean()))
        for p in PARTS])
inv = inv.merge(trip_noise, left_on="trip_id", right_index=True, how="left")
inv.to_csv(REP / "dataset_inventory.csv", index=False)

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for p in PARTS:
    s = inv[inv.partition == p]
    ax[0].hist(s.n_points, bins=25, alpha=.7, color=COL[p], label=f"{p} (n={len(s)})")
    ax[1].hist(np.log10(s.n_arcs), bins=20, alpha=.7, color=COL[p], label=p)
    ax[2].scatter((s.lon_min + s.lon_max) / 2, (s.lat_min + s.lat_max) / 2, c=COL[p], s=22, alpha=.8, label=p)
ax[0].set(title="GPS observations per trip", xlabel="points", ylabel="trips"); ax[0].legend()
ax[1].set(title="Road-network size per trip", xlabel="log10(#arcs)", ylabel="trips")
ax[2].set(title="Trip locations (centroids)", xlabel="longitude", ylabel="latitude")
ax[2].legend(); plt.tight_layout(); plt.savefig(REP / "figures/01_dataset_overview.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
alldt = []
for p in PARTS:
    for d in list_trips(RAW / p):
        alldt.append(np.diff(np.loadtxt(d / f"{d.name}.track", usecols=2)))
alldt = np.concatenate(alldt)
vals, cnt = np.unique(np.clip(alldt, -1, 30), return_counts=True)
ax[0].bar(vals, cnt, color="#2166ac"); ax[0].set_yscale("log")
ax[0].set(title="Sampling interval dt (clipped at 30 s)", xlabel="seconds between consecutive fixes", ylabel="count (log)")
ax[1].bar(["dt = 1 s", "1 < dt ≤ 5", "dt > 5 s (gap)", "dt ≤ 0"],
          [(alldt == 1).mean() * 100, ((alldt > 1) & (alldt <= 5)).mean() * 100, (alldt > 5).mean() * 100,
           (alldt <= 0).mean() * 100], color=["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"])
ax[1].set(title="Share of steps (%)", ylabel="%"); plt.tight_layout()
plt.savefig(REP / "figures/02_sampling_intervals.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for p in PARTS:
    q = pts[p][pts[p].label_ok == 1]
    ax[0].hist(q.d_nearest.clip(upper=60), bins=60, alpha=.7, color=COL[p], density=True, label=p)
ax[0].set(title="GPS noise proxy: distance to nearest road (usable points)", xlabel="metres (clipped 60)", ylabel="density")
ax[0].legend()
tr_c = cands["train"]; pos = tr_c[(tr_c.label == 1) & (tr_c.label_ok == 1)]
rk = pos["rank"].value_counts(normalize=True).sort_index().cumsum() * 100
ax[1].plot(rk.index + 1, rk.values, marker="o", color="#2166ac")
ax[1].set(title="True arc found within top-k nearest candidates (train)", xlabel="k (nearest-arc rank)",
          ylabel="cumulative % of observations"); plt.tight_layout()
plt.savefig(REP / "figures/03_noise_and_candidate_coverage.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for p in PARTS:
    ax[0].hist(pts[p].label_dist.clip(upper=300), bins=60, alpha=.7, color=COL[p], density=True, label=p)
ax[0].axvline(50, color="k", ls="--"); ax[0].text(52, ax[0].get_ylim()[1] * .8, "label_ok ≤ 50 m")
ax[0].set(title="Distance from GPS point to its aligned reference arc", xlabel="metres (clipped 300)", ylabel="density"); ax[0].legend()
s = inv[inv.partition == "train"].sort_values("label_ok_pct").head(12)
ax[1].barh(s.trip_id, s.label_ok_pct, color="#d6604d"); ax[1].invert_yaxis()
ax[1].set(title="Train trips with least reliable reference labels", xlabel="% points with label_ok"); plt.tight_layout()
plt.savefig(REP / "figures/04_label_quality.png"); plt.close()

fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
c = cands["train"]; c = c[c.label_ok == 1].sample(300000, random_state=0)
for a, f, lim in zip(ax, ["dist", "dist_rel", "cos_ang", "dist_ctx_mean"], [100, 100, None, 120]):
    for lab, colr, nm in [(0, "#9aa0a6", "wrong arc"), (1, "#1a9850", "true arc")]:
        a.hist(c.loc[c.label == lab, f], bins=50, range=(-1, 1) if f == "cos_ang" else (0, lim), alpha=.7,
               color=colr, density=True, label=nm)
    a.set(title=f, ylabel="density"); a.legend()
plt.suptitle("Feature separation: true vs wrong candidates (train, usable points)", y=1.02)
plt.tight_layout(); plt.savefig(REP / "figures/05_feature_separation.png", bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------ folium maps
for part, tid in [("student_test", inv[inv.partition == "student_test"].sort_values("n_arcs").trip_id.iloc[0]),
                  ("train", "00000048")]:
    tr = load_trip(RAW / part / tid); P = pts[part][pts[part].trip_id == tid]
    near = np.unique(cands[part].loc[cands[part].trip_id == tid, "arc_id"])
    trip_map(tr, P, network_arc_ids=near, out_html=str(REP / f"maps/map_{part}_{tid}.html"))

# ------------------------------------------------------------------ data-quality log
def pct(x): return f"{x:.2f}%"
tr_i, te_i = inv[inv.partition == "train"], inv[inv.partition == "student_test"]
sm = {p: pd.read_csv(PROC / f"trip_summary_{p}.csv") for p in PARTS}
bad = tr_i[tr_i.label_ok_pct < 90].sort_values("label_ok_pct")
md = f"""# Data-quality log (auto-generated by scripts/01_eda.py)

## Inventory
| | train | student_test |
|---|---|---|
| trips | {len(tr_i)} | {len(te_i)} |
| GPS observations | {tr_i.n_points.sum():,} | {te_i.n_points.sum():,} |
| points / trip (min / median / max) | {tr_i.n_points.min()} / {int(tr_i.n_points.median())} / {tr_i.n_points.max()} | {te_i.n_points.min()} / {int(te_i.n_points.median())} / {te_i.n_points.max()} |
| road nodes / trip (median) | {int(tr_i.n_nodes.median()):,} | {int(te_i.n_nodes.median()):,} |
| road arcs / trip (median) | {int(tr_i.n_arcs.median()):,} | {int(te_i.n_arcs.median()):,} |
| reference-route arcs / trip (median) | {int(tr_i.n_route_arcs.median())} | {int(te_i.n_route_arcs.median())} |
| steps with dt = 1 s | {pct(100*(alldt==1).mean())} (all trips) | |
| NaN / invalid rows | {int(tr_i.n_nan.sum()+te_i.n_nan.sum())} | |
| non-increasing timestamps (dt ≤ 0) | {int(inv.n_dt_le0.sum())} | |
| steps with gap > 5 s | {int(inv.n_dt_gt5.sum())} ({inv.n_dt_gt5.gt(0).sum()} trips) | |
| max single gap | {inv.max_dt_s.max():,.0f} s (trip {inv.loc[inv.max_dt_s.idxmax(),'trip_id']}) | |
| trips with isolated nodes | {int((inv.isolated_nodes>0).sum())} | |
| trips whose road graph has >1 weakly-connected component | {int((inv.net_components>1).sum())} / {len(inv)} | |
| arcs with a reverse (two-way) twin (first 2000 arcs per trip) | {inv.pct_arcs_with_reverse.mean():.1f}% (mean) | |
| reference routes where consecutive arcs are connected | {inv.route_connected_pct.mean():.1f}% (mean) | |
| all arc/route indices valid | {bool(inv.route_arc_ids_valid.all() and inv.arcs_node_ids_valid.all())} | |

## Decisions and assumptions
1. **Each trip has its own road network** (node/arc ids are local to the trip) -> everything is computed per trip; ids are never mixed across trips.
2. **Projection:** each trip is projected to its local UTM zone (EPSG 326xx/327xx) so distances are in metres (degrees are not distances).
3. **Cleaning:** only non-finite/out-of-range rows would be dropped ({int(tr_i.n_nan.sum()+te_i.n_nan.sum())} found). Gaps and odd timestamps are *flagged* (`gap_flag`), not dropped, so a prediction can be returned for every observation.
4. **Candidates:** all arcs within {100} m of a point, nearest 20 kept. Coverage of the reference arc on usable points: train {pct(100*(1-sm['train'].candidate_miss_rate_label_ok.mean()))}, student_test {pct(100*(1-sm['student_test'].candidate_miss_rate_label_ok.mean()))} (candidate miss rate on usable points {pct(100*sm['train'].candidate_miss_rate_label_ok.mean())} / {pct(100*sm['student_test'].candidate_miss_rate_label_ok.mean())}).
5. **Labels:** `.route` has no timestamps, so GPS points are aligned to the route with a monotonic dynamic-programming alignment (stay / advance 1-{3} arcs). The label of a candidate is 1 iff it equals the aligned route arc. The route is a *reference*, not perfect truth.
6. **Unreliable labels (`label_ok = 0`)**: aligned arc further than 50 m from the point. train: {pct(100-100*pts['train'].label_ok.mean())} of points, student_test: {pct(100-100*pts['student_test'].label_ok.mean())}. These are excluded from training and from metrics. Worst train trips: {', '.join(f"{r.trip_id} ({r.label_ok_pct:.0f}% usable)" for r in bad.itertuples())}. In 00000054 and 00000033 the track keeps going (slow/stationary, up to 19k points) after the reference route has ended, so the route does not cover the whole track.
7. **No leakage:** features use only the track and the road network; the route is used only to create labels. Scaling/normalisation is NOT applied here (fit it on train inside the modelling pipeline).
"""
(REP / "data_quality_log.md").write_text(md, encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8"); print(md[:1800])
