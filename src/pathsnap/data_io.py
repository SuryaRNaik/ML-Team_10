"""Raw-file readers for one PathSnap trip folder.

Raw formats (all whitespace separated, no header):
  .track : lon, lat, timestamp_s
  .nodes : lon, lat                (row index = node_id)
  .arcs  : from_node, to_node      (row index = arc_id)
  .route : arc_id                  (chronological reference route, optional)
"""
from pathlib import Path
import numpy as np
import pandas as pd


def _read(path, dtype, ncols):
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype)
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c")
    return df.to_numpy()


def load_trip(trip_dir, with_route=True):
    """Load one trip. Returns dict with track(N,3), nodes(M,2), arcs(K,2), route(R,)|None."""
    d = Path(trip_dir)
    tid = d.name
    out = {
        "trip_id": tid,
        "track": _read(d / f"{tid}.track", float, 3),
        "nodes": _read(d / f"{tid}.nodes", float, 2),
        "arcs": _read(d / f"{tid}.arcs", np.int64, 2),
        "route": None,
    }
    if with_route and (d / f"{tid}.route").exists():
        out["route"] = _read(d / f"{tid}.route", np.int64, 1).reshape(-1)
    return out


def list_trips(partition_dir):
    return sorted(p for p in Path(partition_dir).iterdir() if p.is_dir())
