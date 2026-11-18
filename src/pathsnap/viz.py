"""Folium map builder: GPS points, road network, reference route and (optionally) predicted route."""
import folium
import numpy as np


def _arc_lines(nodes, arcs, arc_ids):
    return [[(nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]), (nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0])]
            for a in arc_ids]


def trip_map(trip, points, network_arc_ids=None, pred_arcs=None, max_points=3000, out_html=None):
    """trip: dict from load_trip. points: cleaned points DataFrame (lon, lat, t).
    network_arc_ids: arcs to draw as the road network (default: all, so pass a subset for big trips).
    pred_arcs: optional list of predicted arc ids (route order)."""
    nodes, arcs, route = trip["nodes"], trip["arcs"], trip.get("route")
    lat0, lon0 = points.lat.median(), points.lon.median()
    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap", control_scale=True)

    if network_arc_ids is None:
        network_arc_ids = np.arange(len(arcs))
    fg = folium.FeatureGroup(name="Road network (arcs near track)", show=True)
    folium.PolyLine(_arc_lines(nodes, arcs, network_arc_ids), color="#9aa0a6", weight=2, opacity=0.7).add_to(fg)
    fg.add_to(m)

    if route is not None and len(route):
        fg = folium.FeatureGroup(name="Reference route (green)", show=True)
        folium.PolyLine(_arc_lines(nodes, arcs, route), color="#1a9850", weight=6, opacity=0.8,
                        tooltip="reference route").add_to(fg)
        fg.add_to(m)

    if pred_arcs is not None and len(pred_arcs):
        fg = folium.FeatureGroup(name="Predicted route (red)", show=True)
        folium.PolyLine(_arc_lines(nodes, arcs, pred_arcs), color="#d73027", weight=4, opacity=0.9,
                        dash_array="8", tooltip="predicted route").add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup(name="Raw GPS points (blue)", show=True)
    step = max(1, len(points) // max_points)
    for r in points.iloc[::step].itertuples():
        folium.CircleMarker((r.lat, r.lon), radius=2.5, color="#2166ac", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"pt {r.pt_idx if hasattr(r, 'pt_idx') else ''} t={r.t:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.PolyLine(points[["lat", "lon"]].to_numpy()[::step].tolist(), color="#2166ac", weight=1,
                    opacity=0.4).add_to(m)

    legend = """<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;
    border:1px solid #888;border-radius:6px;font-size:12px;line-height:1.5">
    <b>PathSnap</b><br><span style='color:#2166ac'>&#9679;</span> raw GPS<br>
    <span style='color:#1a9850'>&#9632;</span> reference route<br>
    <span style='color:#d73027'>&#9632;</span> predicted route<br>
    <span style='color:#9aa0a6'>&#9632;</span> road network</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    b = [[points.lat.min(), points.lon.min()], [points.lat.max(), points.lon.max()]]
    m.fit_bounds(b)
    if out_html:
        m.save(out_html)
    return m
