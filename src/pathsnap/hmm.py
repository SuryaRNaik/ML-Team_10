"""HMM / Viterbi route reconstruction on top of the candidate classifier.

Model
-----
* Hidden state at GPS observation t  : one of the candidate road arcs generated for that
  observation (the same candidate rows the classifier scores, so the candidate <-> arc
  correspondence is preserved).
* Emission   log P(obs_t | arc)      : alpha * log(p_clf), where p_clf = P(arc is the true road)
  from the selected classifier (clipped to [p_floor, 1]).  Per-observation normalisation is a
  constant for Viterbi, so it is skipped.
* Transition log P(arc_t | arc_t-1)  : Newson & Krumm style.
      connected (same arc or reachable through the directed network within `max_hops` arcs):
          -| route_distance - great_circle_distance | / beta_m
          route_distance = (remaining length of previous arc) + (lengths of arcs in between) + (offset on new arc)
          same arc       : route_distance = max(along_new - along_prev, 0)
          reverse arc (U-turn) gets an extra penalty `uturn_pen`
      NOT connected: constant `log_disc` (very unlikely but never impossible, so decoding can recover).
* Missing candidates : observations with 0 candidates are skipped by the lattice and get the
  nearest decoded arc afterwards (flag `filled = 1`).
* GPS gaps          : if the time gap between two consecutive lattice steps > `gap_break_s`, the
  transition is uninformative (all zeros) = the chain restarts there (flag `seg_break = 1`).
* Decoder           : standard Viterbi, O(T * K^2) with K <= max_candidates_per_point (20).

Everything is a pure function of (points, candidates, probabilities, arcs, nodes) so the FastAPI
backend can call exactly the same code at inference time.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import make_projector


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
@dataclass
class HMMParams:
    alpha: float = 1.0          # emission exponent (weight of the classifier vs the transitions)
    beta_m: float = 30.0        # transition scale in metres (smaller = stricter route/GPS agreement)
    log_disc: float = -12.0     # log-prob of a transition between NOT connected arcs
    uturn_pen: float = 3.0      # extra penalty for jumping onto the reverse arc
    gap_break_s: float = 30.0   # time gap that restarts the chain
    p_floor: float = 1e-6       # clip for log(p)
    max_hops: int = 6           # structural: how many arcs ahead we search for connectivity
    max_route_m: float = 600.0  # structural: max length of arcs skipped in between

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# road graph
# --------------------------------------------------------------------------- #
def arc_lengths(nodes, arcs, lon0, lat0):
    """Arc lengths in metres, projected with the same per-trip UTM projector as prepare_trip."""
    proj = make_projector(float(lon0), float(lat0))
    nx_, ny_ = proj(nodes[:, 0], nodes[:, 1])
    return np.hypot(nx_[arcs[:, 1]] - nx_[arcs[:, 0]], ny_[arcs[:, 1]] - ny_[arcs[:, 0]])


class RoadGraph:
    """Directed arc graph: arc a -> arc b when to_node(a) == from_node(b)."""

    def __init__(self, arcs, arc_len):
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.len = np.asarray(arc_len, dtype=float)
        n_nodes = int(self.arcs.max()) + 1 if len(self.arcs) else 0
        order = np.argsort(self.arcs[:, 0], kind="stable")
        self._out = order
        self._ptr = np.searchsorted(self.arcs[order, 0], np.arange(n_nodes + 1))
        self._cache = {}

    def successors(self, a):
        node = self.arcs[a, 1]
        return self._out[self._ptr[node]:self._ptr[node + 1]]

    def reach(self, a, max_hops, max_m):
        """{b: (hops, mid_len_m, parent_arc)} for arcs reachable from a in 1..max_hops steps (BFS)."""
        key = (int(a), max_hops, max_m)
        r = self._cache.get(key)
        if r is not None:
            return r
        r = {}
        q = deque()
        for s in self.successors(a):
            s = int(s)
            if s != a and s not in r:
                r[s] = (1, 0.0, int(a))
                q.append(s)
        while q:
            cur = q.popleft()
            h, mid, _ = r[cur]
            if h >= max_hops:
                continue
            mid2 = mid + self.len[cur]
            if mid2 > max_m:
                continue
            for s in self.successors(cur):
                s = int(s)
                if s != a and s not in r:
                    r[s] = (h + 1, mid2, cur)
                    q.append(s)
        self._cache[key] = r
        return r

    def path(self, a, b, max_hops, max_m):
        """Arcs strictly between a and b on the BFS path (None if not reachable)."""
        r = self.reach(a, max_hops, max_m)
        if b not in r:
            return None
        mid, cur = [], r[b][2]
        while cur != a:
            mid.append(cur)
            cur = r[cur][2]
        return mid[::-1]


# --------------------------------------------------------------------------- #
# lattice (parameter-free part: all geometry / connectivity) and Viterbi
# --------------------------------------------------------------------------- #
@dataclass
class Lattice:
    pt_idx: np.ndarray            # observation index of every lattice step
    arcs: list                    # arcs[i]  -> candidate arc ids at step i
    p: list                       # p[i]     -> classifier probabilities at step i
    H: list                       # H[i]     -> (K_prev, K_cur) hops matrix (-1 = not connected, 0 = same arc); H[0] is None
    RD: list                      # RD[i]    -> route-distance matrix (m)
    U: list                       # U[i]     -> U-turn indicator matrix
    gc: np.ndarray                # great-circle (euclidean in UTM) distance between consecutive steps (m)
    dt: np.ndarray                # time difference between consecutive steps (s)
    n_points: int


def build_lattice(pts, cands, graph, prm: HMMParams, p_col="p"):
    """pts: points DataFrame (pt_idx, x, y, t) ; cands: candidates DataFrame with
    pt_idx, arc_id, along_m, rank and the classifier probability column `p_col`."""
    n_points = int(pts.pt_idx.max()) + 1 if len(pts) else 0
    c = cands.sort_values(["pt_idx", "rank"], kind="stable")
    if len(c) == 0:
        return Lattice(np.empty(0, int), [], [], [], [], [], np.empty(0), np.empty(0), n_points)
    pt_arr = c.pt_idx.to_numpy()
    arc_arr = c.arc_id.to_numpy().astype(np.int64)
    along_arr = c.along_m.to_numpy().astype(float)
    p_arr = c[p_col].to_numpy().astype(float)
    starts = np.r_[0, np.flatnonzero(np.diff(pt_arr)) + 1]
    ends = np.r_[starts[1:], len(c)]
    P = pts.set_index("pt_idx")
    X, Y, T = P.x.to_numpy(), P.y.to_numpy(), P.t.to_numpy()
    pos = {int(k): i for i, k in enumerate(P.index.to_numpy())}

    idx, arcs_l, p_l, along_l = [], [], [], []
    for s, e in zip(starts, ends):
        idx.append(int(pt_arr[s])); arcs_l.append(arc_arr[s:e]); p_l.append(p_arr[s:e]); along_l.append(along_arr[s:e])

    n = len(idx)
    H, RD, U = [None], [None], [None]
    gc, dt = np.zeros(n), np.zeros(n)
    ga = graph.arcs
    for i in range(1, n):
        ap, ac, lp, lc = arcs_l[i - 1], arcs_l[i], along_l[i - 1], along_l[i]
        h = np.full((len(ap), len(ac)), -1, dtype=np.int16)
        rd = np.zeros((len(ap), len(ac)))
        for a_i, a in enumerate(ap):
            reach = graph.reach(int(a), prm.max_hops, prm.max_route_m)
            rem = graph.len[a] - lp[a_i]
            for b_i, b in enumerate(ac):
                if a == b:
                    h[a_i, b_i] = 0
                    rd[a_i, b_i] = max(lc[b_i] - lp[a_i], 0.0)
                else:
                    r = reach.get(int(b))
                    if r is not None:
                        h[a_i, b_i] = r[0]
                        rd[a_i, b_i] = rem + r[1] + lc[b_i]
        u = ((ga[ap, 0][:, None] == ga[ac, 1][None, :]) & (ga[ap, 1][:, None] == ga[ac, 0][None, :])
             & (ap[:, None] != ac[None, :]))
        H.append(h); RD.append(rd); U.append(u)
        i0, i1 = pos[idx[i - 1]], pos[idx[i]]
        gc[i] = np.hypot(X[i1] - X[i0], Y[i1] - Y[i0])
        dt[i] = T[i1] - T[i0]
    return Lattice(np.asarray(idx), arcs_l, p_l, H, RD, U, gc, dt, n_points)


def viterbi(lat: Lattice, prm: HMMParams):
    """Returns (chosen arc per lattice step, seg_break flag per step)."""
    n = len(lat.pt_idx)
    if n == 0:
        return np.empty(0, int), np.empty(0, bool)
    emis = [prm.alpha * np.log(np.clip(p, prm.p_floor, 1.0)) for p in lat.p]
    score = emis[0].copy()
    back = [None]
    brk = np.zeros(n, bool)
    for i in range(1, n):
        if lat.dt[i] > prm.gap_break_s or lat.dt[i] < 0:
            T = np.zeros((len(lat.arcs[i - 1]), len(lat.arcs[i])))      # restart: transition carries no information
            brk[i] = True
        else:
            h = lat.H[i]
            T = np.where(h >= 0, -np.abs(lat.RD[i] - lat.gc[i]) / prm.beta_m, prm.log_disc)
            T = T - prm.uturn_pen * lat.U[i]
        tot = score[:, None] + T
        bp = np.argmax(tot, axis=0)
        score = tot[bp, np.arange(tot.shape[1])] + emis[i]
        back.append(bp)
    k = int(np.argmax(score))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = lat.arcs[i][k]
        if i > 0:
            k = int(back[i][k])
    return path, brk


# --------------------------------------------------------------------------- #
# public helpers
# --------------------------------------------------------------------------- #
def _fill(arr):
    s = pd.Series(arr, dtype="float64").ffill().bfill()
    return s.fillna(-1).to_numpy().astype(np.int64)


def decode_hmm(pts, cands, graph, prm: HMMParams, p_col="p", lattice: Lattice | None = None):
    """Full-trip HMM decoding. Returns DataFrame[pt_idx, pred_arc, filled, seg_break] with one row per observation."""
    lat = lattice or build_lattice(pts, cands, graph, prm, p_col)
    path, brk = viterbi(lat, prm)
    n = lat.n_points
    pred = np.full(n, np.nan); seg = np.zeros(n, np.int8)
    pred[lat.pt_idx] = path
    seg[lat.pt_idx] = brk
    has = ~np.isnan(pred)
    return pd.DataFrame({"pt_idx": np.arange(n), "pred_arc": _fill(pred),
                         "filled": (~has).astype(np.int8), "seg_break": seg})


def classifier_only(cands, n_points, p_col="p"):
    """Baseline: highest-probability candidate for every observation independently."""
    top = cands.sort_values(["pt_idx", p_col], ascending=[True, False]).drop_duplicates("pt_idx")
    pred = np.full(n_points, np.nan)
    pred[top.pt_idx.to_numpy()] = top.arc_id.to_numpy()
    has = ~np.isnan(pred)
    return pd.DataFrame({"pt_idx": np.arange(n_points), "pred_arc": _fill(pred), "filled": (~has).astype(np.int8)})


def reconstruct_route(pred_arcs, graph, prm: HMMParams):
    """Per-observation arcs -> ordered route: drop repeats and bridge consecutive arcs with the shortest
    BFS path when they are connected within max_hops. Returns a list of arc ids."""
    route = []
    for a in pred_arcs:
        a = int(a)
        if a < 0:
            continue
        if not route:
            route.append(a); continue
        if a == route[-1]:
            continue
        mid = graph.path(route[-1], a, prm.max_hops, prm.max_route_m)
        if mid:
            route.extend(mid)
        route.append(a)
    return route


def evaluate(pred, pts, graph, prm: HMMParams):
    """Scores a per-observation prediction against the reference (only label_ok observations are scored).
    Returns raw counts so trips can be pooled, plus route-level F1."""
    m = pts[pts.label_ok == 1].sort_values("pt_idx")
    pa = pred.set_index("pt_idx").pred_arc.reindex(m.pt_idx).to_numpy()
    ta = m.true_arc.to_numpy()
    n_ok, hit = len(m), int((pa == ta).sum())
    idx = m.pt_idx.to_numpy()
    consec = (np.diff(idx) == 1) & (pa[:-1] >= 0) & (pa[1:] >= 0)
    a0, a1 = pa[:-1][consec], pa[1:][consec]
    ok = (a0 == a1) | (graph.arcs[a0, 1] == graph.arcs[a1, 0])
    # route-level (sequence) quality: predicted route arcs vs reference route arcs (set overlap)
    pr = set(reconstruct_route(pa, graph, prm))
    ref = set(int(x) for x in np.unique(ta))
    tp = len(pr & ref)
    prec, rec = tp / max(len(pr), 1), tp / max(len(ref), 1)
    return dict(n_obs=n_ok, n_hit=hit, n_consec=int(consec.sum()), n_connected=int(ok.sum()),
                route_precision=prec, route_recall=rec,
                route_f1=(2 * prec * rec / (prec + rec)) if prec + rec else 0.0)