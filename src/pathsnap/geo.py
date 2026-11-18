"""Geometry helpers. Every trip is projected to its own local UTM zone (metres),
because trips come from different parts of the world and degrees are not distances."""
import numpy as np
from pyproj import Transformer


def utm_epsg(lon0, lat0):
    zone = int((lon0 + 180) // 6) + 1
    return (32600 if lat0 >= 0 else 32700) + zone


def make_projector(lon0, lat0):
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg(lon0, lat0)}", always_xy=True)
    def proj(lon, lat):
        x, y = t.transform(lon, lat)
        return np.asarray(x), np.asarray(y)
    return proj


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments A->B and the projection fraction t in [0,1]."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t
