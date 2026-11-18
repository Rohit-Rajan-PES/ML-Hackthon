import streamlit as st, requests, folium
from streamlit_folium import st_folium

API = st.sidebar.text_input("API URL", "http://10.1.25.48:8000")
st.title("PathSnap: Map Matching Demo")

try:
    ok = requests.get(f"{API}/health", timeout=3).json()
    st.sidebar.success(f"API online: {ok}")
except Exception:
    st.sidebar.error("API not reachable")

try:
    trips = requests.get(f"{API}/trips", timeout=5).json()["trips"]
except Exception:
    trips = []
trip_id = st.sidebar.selectbox("Trip", trips) if trips else st.sidebar.text_input("Trip ID", "trip_01")
run = st.sidebar.button("Run prediction")


def flat(x):
    """Turn [[lat,lon],...] or [[[lat,lon],[lat,lon]],...] into a flat list of [lat,lon]."""
    out = []
    for item in x or []:
        if item and isinstance(item[0], (list, tuple)):
            out.extend(flat(item))
        else:
            out.append(item)
    return out


if run:
    with st.spinner("Calling model..."):
        r = requests.post(f"{API}/evaluate", json={"trip_id": trip_id}, timeout=300)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
    else:
        st.session_state["res"] = r.json()

res = st.session_state.get("res")
if res:
    met = res.get("metrics", {})
    gps = flat(res.get("gps_points"))
    pred = flat(res.get("predicted_route_geometry"))
    ref = flat(res.get("reference_route_geometry"))

    c1, c2, c3, c4 = st.columns(4)
    c1.write(f"**Points:** {len(gps)}")
    c2.write(f"**Classifier Top-1:** {met.get('classifier_top1', 'n/a')}")
    c3.write(f"**HMM Top-1:** {met.get('hmm_top1', 'n/a')}")
    c4.write(f"**Route continuity:** {met.get('route_continuity', 'n/a')}")

    if not (gps or pred):
        st.warning("API returned no coordinates to draw.")
    else:
        pts = pred or gps
        m = folium.Map(location=pts[len(pts) // 2], zoom_start=15, tiles=None)
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", name="Base map").add_to(m)

        if gps:
            g = folium.FeatureGroup("GPS observations (red)").add_to(m)
            for p in gps:
                folium.CircleMarker(p, radius=3, color="red", fill=True).add_to(g)
        if ref:
            gr = folium.FeatureGroup("Reference route (green)").add_to(m)
            folium.PolyLine(ref, color="green", weight=5, tooltip="Reference route").add_to(gr)
        if pred:
            gp = folium.FeatureGroup("Predicted route (blue)").add_to(m)
            folium.PolyLine(pred, color="blue", weight=4, tooltip="Predicted route (HMM)").add_to(gp)
            folium.Marker(pred[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(pred[-1], tooltip="End", icon=folium.Icon(color="black")).add_to(m)

        allp = gps + pred + ref
        m.fit_bounds([[min(p[0] for p in allp), min(p[1] for p in allp)],
                      [max(p[0] for p in allp), max(p[1] for p in allp)]])
        folium.LayerControl(collapsed=False).add_to(m)

        legend = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
         background: white; color: black; padding: 8px 12px; border: 1px solid #888;
         border-radius: 6px; font-size: 13px;">
         <b>Legend</b><br>
         <span style="color:red">&#9679;</span> GPS observations<br>
         <span style="color:green">&#9473;</span> Reference route<br>
         <span style="color:blue">&#9473;</span> Predicted route (HMM)<br>
         <span style="color:green">&#9679;</span> Start &nbsp; <span style="color:black">&#9679;</span> End
        </div>"""
        m.get_root().html.add_child(folium.Element(legend))
        st_folium(m, use_container_width=True, height=550, returned_objects=[])

    # classifier vs HMM comparison (Section 5)
    ca, ha = res.get("classifier_arcs", []), res.get("hmm_arcs", [])
    if ca and ha:
        changed = sum(a != b for a, b in zip(ca, ha))
        st.caption(f"Classifier vs HMM: {changed} of {len(ca)} point decisions changed by HMM decoding.")

    with st.expander("Raw API response (debug)"):
        st.json(res)