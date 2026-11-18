"""Ground-truth scoring. Only main.py imports this, AFTER prediction is finished.
pipeline.py must never import this module."""
import os
from pathlib import Path

import pandas as pd

import pathsnap_features as pf
from pathsnap_hmm import RoadGraph, route_metrics

DATA_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR",
                               Path(__file__).resolve().parent / "data" / "evaluator_test"))


def _load_route(trip_id):
    trip_dir = DATA_DIR / trip_id
    m = [f for f in os.listdir(trip_dir) if f.endswith(".route")]
    if not m:
        return None
    return pd.read_csv(trip_dir / m[0], sep=r"\s+", header=None, names=["arc_id"])["arc_id"].astype(int).tolist()


def score_trip(trip_id, pred):
    route = _load_route(trip_id)
    if route is None:
        return {"metrics": {"classifier_top1": None, "hmm_top1": None, "route_continuity": None},
                "reference_route_geometry": []}
    d = pf.load_trip(DATA_DIR, trip_id, with_route=False)
    nodes, arcs = d["nodes"], d["arcs"]
    graph = RoadGraph(arcs)

    clf = route_metrics(pred["classifier_arcs"], route, graph, with_seq=True)
    hmm = route_metrics(pred["hmm_arcs"], route, graph, with_seq=True)

    segs, prev = [], None
    for a in route:
        if a == prev:
            continue
        prev = a
        fn, tn = arcs["from_node_id"].values[a], arcs["to_node_id"].values[a]
        segs.append([[float(nodes["lat"].values[fn]), float(nodes["lon"].values[fn])],
                     [float(nodes["lat"].values[tn]), float(nodes["lon"].values[tn])]])

    return {
        "metrics": {
            "classifier_top1": clf["point_acc"],
            "hmm_top1": hmm["point_acc"],
            "route_continuity": hmm["continuity"],
            "classifier_continuity": clf["continuity"],
            "hmm_route_f1": hmm["route_f1"],
            "classifier_route_f1": clf["route_f1"],
            "hmm_seq_sim": hmm["seq_sim"],
            "classifier_seq_sim": clf["seq_sim"],
            "hmm_jumps": hmm["n_jumps"],
            "classifier_jumps": clf["n_jumps"],
        },
        "reference_route_geometry": segs,
    }
