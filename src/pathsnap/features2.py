"""Second-stage, per-observation *relative* features.

The base features (prepare.py) describe one (GPS point, candidate arc) pair in isolation. Top-1 accuracy, however,
is decided by comparing the candidates of the SAME point with each other, so these features express every
candidate relative to its competitors at that point. Pure function of the candidates table (no labels, no raw
files) so training, the Bayesian search and the FastAPI backend all call exactly the same code.
"""
import numpy as np
import pandas as pd

GROUP_FEATURES = ["dist_z", "dist_share", "ctx_rel", "dprev_rel", "dnext_rel", "cos_gap", "ctx_rank", "cos_rank",
                  "fwd_best", "d_gap_to_best"]


def add_group_features(c: pd.DataFrame) -> pd.DataFrame:
    """Adds GROUP_FEATURES (in place on a copy). Requires the base feature columns from prepare_trip."""
    c = c.copy()
    key = c.groupby(["trip_id", "pt_idx"], sort=False).ngroup().to_numpy()
    g = c.groupby(key, sort=False)
    dmin = g["dist"].transform("min")
    c["dist_z"] = c["dist"] / (dmin + 1.0)                                  # how many times farther than the nearest arc
    w = np.exp(-c["dist"] / 20.0)
    c["dist_share"] = w / w.groupby(key).transform("sum")                  # soft-max style share of the point's "mass"
    c["ctx_rel"] = c["dist_ctx_mean"] - g["dist_ctx_mean"].transform("min")  # distance over a +-5 s window, relative
    c["dprev_rel"] = c["dist_prev"] - g["dist_prev"].transform("min")
    c["dnext_rel"] = c["dist_next"] - g["dist_next"].transform("min")
    c["cos_gap"] = g["cos_ang"].transform("max") - c["cos_ang"]            # heading agreement relative to the best candidate
    c["ctx_rank"] = g["dist_ctx_mean"].rank(method="min")
    c["cos_rank"] = g["cos_ang"].rank(ascending=False, method="min")
    # arcs whose heading agrees with the motion AND are close in the context window are the plausible ones
    fwd = (c["cos_ang"] > 0.5).astype(float) * (1.0 / (1.0 + c["dist_ctx_mean"] / 10.0))
    c["fwd_best"] = fwd - fwd.groupby(key).transform("max")
    c["d_gap_to_best"] = c["dist"] - c.groupby(key)["dist"].transform("min")
    return c
