"""Prediction pipeline. NEVER reads .route files and never imports scoring.py."""
import json, os, pickle, time
from pathlib import Path

import numpy as np

import pathsnap_features as pf
from pathsnap_hmm import RoadGraph, decode_trip

BASE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", BASE / "data" / "evaluator_test"))
MODEL_PATH = BASE / "artifacts" / "pathsnap_logreg_tuned.pkl"
HMM_PATH = BASE / "artifacts" / "hmm_config.json"


def load_pipeline():
    """Runs once at startup."""
    try:
        import joblib
        artifact = joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
    if not isinstance(artifact, dict):           # bare sklearn model
        artifact = {"model": artifact, "features": pf.FEATURES}
    artifact.setdefault("features", pf.FEATURES)
    with open(HMM_PATH) as f:
        hmm_params = json.load(f)["params"]
    return {"artifact": artifact, "hmm_params": hmm_params}


def list_trip_ids():
    return pf.get_trip_ids(DATA_DIR)


def _geometry(arc_ids, nodes, arcs):
    """Consecutive-deduplicated arcs -> list of segments [[lat,lon],[lat,lon]]."""
    segs, prev = [], None
    for a in arc_ids:
        a = int(a)
        if a == prev:
            continue
        prev = a
        fn, tn = arcs["from_node_id"].values[a], arcs["to_node_id"].values[a]
        segs.append([[float(nodes["lat"].values[fn]), float(nodes["lon"].values[fn])],
                     [float(nodes["lat"].values[tn]), float(nodes["lon"].values[tn])]])
    return segs


def predict_trip(pipe, trip_id):
    t = {}
    t0 = time.time()
    d = pf.load_trip(DATA_DIR, trip_id, with_route=False)     # track, nodes, arcs only
    track, nodes, arcs = d["track"], d["nodes"], d["arcs"]
    t["load_s"] = time.time() - t0

    t0 = time.time()
    feats = pf.build_trip_features(track, nodes, arcs, route=None, trip_id=trip_id)
    t["features_s"] = time.time() - t0

    t0 = time.time()
    art = pipe["artifact"]
    feats["prob"] = art["model"].predict_proba(feats[art["features"]])[:, 1]
    t["classifier_s"] = time.time() - t0

    t0 = time.time()
    clean, _ = pf.clean_track(track)
    res = decode_trip(feats, RoadGraph(arcs), clean["timestamp"].values.astype(float),
                      pipe["hmm_params"])
    t["hmm_s"] = time.time() - t0

    res = res.sort_values("point_idx")
    gps = clean.loc[res["point_idx"].values, ["lat", "lon"]].values.tolist()   # [lat, lon]
    clf_arcs = res["clf_arc"].astype(int).tolist()
    hmm_arcs = res["hmm_arc"].astype(int).tolist()
    return {
        "gps_points": gps,
        "classifier_arcs": clf_arcs,
        "hmm_arcs": hmm_arcs,
        "classifier_route_geometry": _geometry(clf_arcs, nodes, arcs),
        "predicted_route_geometry": _geometry(hmm_arcs, nodes, arcs),
        "timings": {k: round(v, 3) for k, v in t.items()},
    }
