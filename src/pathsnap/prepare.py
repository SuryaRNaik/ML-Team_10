"""Per-trip preprocessing: cleaning -> projection -> candidate generation ->
(optional) label alignment -> feature engineering.

`prepare_trip` is deliberately a pure function so the *same* code is used for
training data, student-test data and (later) the API at inference time.
Labels are only produced when a reference route is passed in; none of the
features ever look at the route, so there is no label leakage.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .geo import make_projector, point_segment

DEFAULT_CFG = json.loads((Path(__file__).parents[2] / "configs" / "preprocess.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. cleaning
# --------------------------------------------------------------------------- #
def clean_track(track):
    """Drop only physically impossible rows; *flag* (never drop) everything else,
    so predictions can still be returned for every observation."""
    df = pd.DataFrame(track, columns=["lon", "lat", "t"])
    df["raw_idx"] = np.arange(len(df))
    log = {"n_raw": len(df)}
    bad = ~np.isfinite(df[["lon", "lat", "t"]]).all(axis=1)
    bad |= (df.lon.abs() > 180) | (df.lat.abs() > 90)
    log["n_dropped_invalid"] = int(bad.sum())
    df = df[~bad].reset_index(drop=True)
    dt = df.t.diff()
    log["n_nonmonotonic_time"] = int((dt < 0).sum())
    log["n_duplicate_time"] = int((dt == 0).sum())
    return df, log


# --------------------------------------------------------------------------- #
# 2. label alignment (reference route has no timestamps -> align monotonically)
# --------------------------------------------------------------------------- #
def align_points_to_route(px, py, route, ax, ay, bx, by, cfg):
    """Monotonic dynamic-programming alignment of GPS points to the reference route.

    Point i is assigned to route position j(i); j can only stay, or advance by
    1..align_max_skip along the route (skipping >1 arc is penalised).
    Returns (route_pos[N], dist_to_assigned_arc[N]).
    """
    n, m = len(px), len(route)
    cap, k, lam = cfg["align_dist_cap_m"], cfg["align_max_skip"], cfg["align_skip_penalty_m"]
    ra, rb = route, route
    d = np.empty((n, m), dtype=np.float32)
    step = max(1, int(2e7 // max(m, 1)))
    for s in range(0, n, step):
        e = min(n, s + step)
        dist, _ = point_segment(px[s:e, None], py[s:e, None], ax[ra][None, :], ay[ra][None, :],
                                bx[rb][None, :], by[rb][None, :])
        d[s:e] = dist
    dc = np.minimum(d, cap)
    pen = np.concatenate([[0.0], [0.0], lam * np.arange(1, k)]).astype(np.float32)[: k + 1]
    cost = dc[0] + lam * np.arange(m, dtype=np.float32)          # soft "start near route start"
    back = np.zeros((n, m), dtype=np.int8)
    inf = np.float32(1e9)
    for i in range(1, n):
        best = cost + pen[0]
        arg = np.zeros(m, dtype=np.int8)
        for s in range(1, k + 1):
            if s >= m:
                break
            cand = np.concatenate([np.full(s, inf, dtype=np.float32), cost[:-s]]) + pen[s]
            mask = cand < best
            best = np.where(mask, cand, best)
            arg = np.where(mask, s, arg).astype(np.int8)
        cost = best + dc[i]
        back[i] = arg
    end_pen = lam * (m - 1 - np.arange(m, dtype=np.float32))     # soft "end near route end"
    j = int(np.argmin(cost + end_pen))
    pos = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j -= int(back[i, j])
    return pos, d[np.arange(n), pos].astype(float)


# --------------------------------------------------------------------------- #
# 3. main entry point
# --------------------------------------------------------------------------- #
def prepare_trip(trip, cfg=None, with_labels=True):
    """Returns (points_df, cands_df, info).

    points_df : one row per GPS observation (cleaned, projected, kinematics, labels)
    cands_df  : one row per (observation, candidate arc) with features (+ label)
    """
    cfg = cfg or DEFAULT_CFG
    tid = trip["trip_id"]
    pts, log = clean_track(trip["track"])
    n = len(pts)
    nodes, arcs = trip["nodes"], trip["arcs"]

    # --- projection (local UTM, metres) -------------------------------------
    proj = make_projector(float(pts.lon.median()), float(pts.lat.median()))
    pts["x"], pts["y"] = proj(pts.lon.to_numpy(), pts.lat.to_numpy())
    nx_, ny_ = proj(nodes[:, 0], nodes[:, 1])
    ax, ay = nx_[arcs[:, 0]], ny_[arcs[:, 0]]
    bx, by = nx_[arcs[:, 1]], ny_[arcs[:, 1]]
    arc_len = np.hypot(bx - ax, by - ay)
    arc_ux = np.where(arc_len > 0, (bx - ax) / np.where(arc_len > 0, arc_len, 1), 0.0)
    arc_uy = np.where(arc_len > 0, (by - ay) / np.where(arc_len > 0, arc_len, 1), 0.0)
    px, py, pt = pts.x.to_numpy(), pts.y.to_numpy(), pts.t.to_numpy()

    # --- point kinematics (uses only the track) -----------------------------
    dt_prev = np.r_[np.nan, np.diff(pt)]
    dt_next = np.r_[np.diff(pt), np.nan]
    w = cfg["heading_half_window"]
    lo, hi = np.clip(np.arange(n) - w, 0, n - 1), np.clip(np.arange(n) + w, 0, n - 1)
    mvx, mvy = px[hi] - px[lo], py[hi] - py[lo]
    mv = np.hypot(mvx, mvy)
    span_t = pt[hi] - pt[lo]
    head_ok = mv >= cfg["min_move_for_heading_m"]
    hx = np.where(head_ok, mvx / np.where(mv > 0, mv, 1), 0.0)
    hy = np.where(head_ok, mvy / np.where(mv > 0, mv, 1), 0.0)
    speed = np.where(span_t > 0, mv / np.where(span_t > 0, span_t, 1), 0.0)
    pts["dt_prev"], pts["dt_next"] = dt_prev, dt_next
    pts["gap_flag"] = ((np.nan_to_num(dt_prev) > cfg["gap_threshold_s"]) |
                       (np.nan_to_num(dt_next) > cfg["gap_threshold_s"])).astype(np.int8)
    pts["heading_valid"] = head_ok.astype(np.int8)
    pts["hx"], pts["hy"], pts["speed"] = hx, hy, speed
    pts.insert(0, "trip_id", tid)
    pts.insert(1, "pt_idx", np.arange(n))

    # --- candidate generation (all arcs within R metres, keep K nearest) ----
    segs = shapely.linestrings(np.stack([np.stack([ax, ay], 1), np.stack([bx, by], 1)], 1))
    tree = STRtree(segs)
    q = tree.query(shapely.points(px, py), predicate="dwithin", distance=cfg["candidate_radius_m"])
    ci, ca = q[0], q[1]
    dist, tfrac = point_segment(px[ci], py[ci], ax[ca], ay[ca], bx[ca], by[ca])
    order = np.lexsort((dist, ci))
    ci, ca, dist, tfrac = ci[order], ca[order], dist[order], tfrac[order]
    first = np.r_[0, np.flatnonzero(np.diff(ci)) + 1]
    counts_all = np.diff(np.r_[first, len(ci)])
    rank = np.arange(len(ci)) - np.repeat(first, counts_all)
    keep = rank < cfg["max_candidates_per_point"]
    ci, ca, dist, tfrac, rank = ci[keep], ca[keep], dist[keep], tfrac[keep], rank[keep]
    n_cands = np.bincount(ci, minlength=n)
    d_min = np.full(n, np.nan)
    d_min[ci[rank == 0]] = dist[rank == 0]
    pts["n_cands"] = n_cands
    pts["d_nearest"] = d_min          # per-point GPS-to-network distance ~ noise level

    # --- features per (point, candidate) ------------------------------------
    off = cfg["context_offset"]
    ip, inx = np.clip(ci - off, 0, n - 1), np.clip(ci + off, 0, n - 1)
    d_prev, _ = point_segment(px[ip], py[ip], ax[ca], ay[ca], bx[ca], by[ca])
    d_next, _ = point_segment(px[inx], py[inx], ax[ca], ay[ca], bx[ca], by[ca])
    m = len(nodes)
    out_deg = np.bincount(arcs[:, 0], minlength=m)
    in_deg = np.bincount(arcs[:, 1], minlength=m)
    keys = arcs[:, 0].astype(np.int64) * m + arcs[:, 1]
    has_rev = np.isin(arcs[:, 1].astype(np.int64) * m + arcs[:, 0], keys)
    cos_ang = hx[ci] * arc_ux[ca] + hy[ci] * arc_uy[ca]       # +1 same direction, -1 opposite

    cands = pd.DataFrame({
        "trip_id": tid, "pt_idx": ci, "arc_id": ca,
        "dist": dist, "t_frac": tfrac, "arc_len": arc_len[ca],
        "along_m": tfrac * arc_len[ca],
        "dist_to_end_m": np.minimum(tfrac, 1 - tfrac) * arc_len[ca],
        "cos_ang": cos_ang, "abs_cos_ang": np.abs(cos_ang),
        "rank": rank, "n_cands": n_cands[ci],
        "dist_rel": dist - d_min[ci],
        "dist_prev": d_prev, "dist_next": d_next,
        "dist_ctx_mean": (dist + d_prev + d_next) / 3.0,
        "speed": speed[ci], "heading_valid": head_ok[ci].astype(np.int8),
        "gap_flag": pts["gap_flag"].to_numpy()[ci],
        "out_deg_to": out_deg[arcs[ca, 1]], "in_deg_from": in_deg[arcs[ca, 0]],
        "has_reverse": has_rev[ca].astype(np.int8),
    })

    info = {"trip_id": tid, **log, "n_points": n, "n_nodes": m, "n_arcs": len(arcs),
            "n_candidates": len(cands), "pct_points_no_candidate": float((n_cands == 0).mean() * 100)}

    # --- labels (only when a reference route is supplied) -------------------
    if with_labels and trip.get("route") is not None and len(trip["route"]) > 0:
        route = trip["route"]
        pos, adist = align_points_to_route(px, py, route, ax, ay, bx, by, cfg)
        true_arc = route[pos]
        pts["true_route_pos"], pts["true_arc"], pts["label_dist"] = pos, true_arc, adist
        pts["label_ok"] = (adist <= cfg["label_max_dist_m"]).astype(np.int8)
        cands["label"] = (cands["arc_id"].to_numpy() == true_arc[ci]).astype(np.int8)
        # label_ok = the aligned reference arc is within label_max_dist_m of the GPS point.
        # Points where the reference route is missing/truncated (e.g. long stationary tails
        # after the route ends) are NOT reliable labels -> exclude them from training/metrics.
        cands["label_ok"] = pts["label_ok"].to_numpy()[ci]
        hit = np.zeros(n, bool)
        hit[ci[cands["label"].to_numpy() == 1]] = True
        pts["true_in_cands"] = hit.astype(np.int8)
        ok = pts["label_ok"].to_numpy() == 1
        info.update({"candidate_miss_rate_label_ok": float(1 - hit[ok].mean()) if ok.any() else np.nan,
                     "candidate_miss_rate": float(1 - hit.mean()),
                     "label_ok_rate": float(pts["label_ok"].mean()),
                     "median_label_dist_m": float(np.median(adist))})
    return pts, cands, info
