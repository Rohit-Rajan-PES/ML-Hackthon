"""PathSnap shared data + feature code.

Used by the training notebook AND by the API (api/pipeline.py) so that training
and inference build features in exactly the same way.
Ground truth (.route) is only used when `route` is passed in (training / scoring).
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

R_EARTH = 6371000.0

# ---- candidate generation / feature config (saved with the model) ----
K_FINAL = 10        # candidates kept per GPS point
K_PRE = 40          # arcs pre-selected by midpoint distance, then re-ranked by exact point-to-segment distance
HEAD_WIN = 2        # heading/speed measured from point i-2 to i+2
NBR_OFFSETS = (-2, -1, 1, 2)   # neighbouring points used for the "neighbour support" feature

FEATURES = [
    "dist_m", "dist_rank", "dist_gap_m", "nbr_dist_m",
    "heading_diff_deg", "cos_heading", "cos_heading_w",
    "speed_mps", "along_frac", "arc_len_m",
]


# ---------------------------------------------------------------- loaders
def get_trip_ids(partition_dir):
    return sorted(os.listdir(partition_dir))


def find_file(trip_dir, ext):
    matches = [f for f in os.listdir(trip_dir) if f.endswith(ext)]
    return Path(trip_dir) / matches[0] if matches else None


def load_track(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])


def load_nodes(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat"])
    df.index.name = "node_id"
    return df


def load_arcs(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])
    df.index.name = "arc_id"
    return df


def load_route(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["arc_id"])


def load_trip(partition_dir, trip_id, with_route=True):
    trip_dir = Path(partition_dir) / trip_id
    spec = [("track", ".track", load_track), ("nodes", ".nodes", load_nodes),
            ("arcs", ".arcs", load_arcs)]
    if with_route:
        spec.append(("route", ".route", load_route))
    data = {}
    for key, ext, loader in spec:
        p = find_file(trip_dir, ext)
        data[key] = loader(p) if p is not None else None
    return data


# ---------------------------------------------------------------- cleaning
def clean_track(track):
    """Drop NaN / impossible coordinates, sort by time, drop duplicate timestamps.
    Keeps `orig_idx` (row number in the raw file) so predictions can be mapped back."""
    t = track.copy()
    t["orig_idx"] = np.arange(len(t))
    n0 = len(t)
    t = t.dropna()
    t = t[t["lat"].between(-90, 90) & t["lon"].between(-180, 180)]
    t = t.sort_values("timestamp", kind="stable")
    t = t.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    return t, n0 - len(t)


# ---------------------------------------------------------------- geometry
def _project(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres (accurate at city scale)."""
    x = R_EARTH * np.radians(np.asarray(lon) - lon0) * np.cos(np.radians(lat0))
    y = R_EARTH * np.radians(np.asarray(lat) - lat0)
    return x, y


def _seg_dist(P, A, B):
    """Distance from points P to segments AB (broadcasting). Returns (dist, t in [0,1])."""
    AB = B - A
    denom = np.maximum((AB ** 2).sum(-1), 1e-9)
    t = np.clip(((P - A) * AB).sum(-1) / denom, 0.0, 1.0)
    proj = A + t[..., None] * AB
    return np.sqrt(((P - proj) ** 2).sum(-1)), t


def _bearing(dx, dy):
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def _hdiff(b1, b2):
    d = np.abs(b1 - b2) % 360.0
    return np.minimum(d, 360.0 - d)


# ---------------------------------------------------------------- features
def build_trip_features(track, nodes, arcs, route=None, trip_id="trip"):
    """Candidate table for one trip: K_FINAL candidate arcs per GPS point.
    If `route` is given a `label` column is added (1 = candidate arc is in the reference route)."""
    track, _ = clean_track(track)
    n = len(track)
    if n == 0:
        return None

    lon0, lat0 = track["lon"].mean(), track["lat"].mean()
    px, py = _project(track["lon"].values, track["lat"].values, lon0, lat0)
    P = np.column_stack([px, py])
    ts = track["timestamp"].values.astype(float)

    nx, ny = _project(nodes["lon"].values, nodes["lat"].values, lon0, lat0)
    N = np.column_stack([nx, ny])
    A = N[arcs["from_node_id"].values]
    B = N[arcs["to_node_id"].values]
    arc_len = np.sqrt(((B - A) ** 2).sum(1))
    arc_brg = _bearing(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])

    # 1) pre-select by midpoint distance, 2) re-rank by exact point-to-segment distance
    n_arcs = len(arcs)
    kpre = min(K_PRE, n_arcs)
    kfin = min(K_FINAL, kpre)
    tree = cKDTree((A + B) / 2.0)
    _, pre = tree.query(P, k=kpre)
    pre = pre.reshape(n, kpre)
    d_pre, t_pre = _seg_dist(P[:, None, :], A[pre], B[pre])
    order = np.argsort(d_pre, axis=1)[:, :kfin]
    cand = np.take_along_axis(pre, order, axis=1)          # (n, kfin) arc ids
    dist = np.take_along_axis(d_pre, order, axis=1)
    along = np.take_along_axis(t_pre, order, axis=1)

    # neighbour support: distance of neighbouring GPS points to the same candidate arc
    idx = np.arange(n)
    nb = []
    for o in NBR_OFFSETS:
        j = np.clip(idx + o, 0, n - 1)
        dj, _ = _seg_dist(P[j][:, None, :], A[cand], B[cand])
        nb.append(dj)
    nbr = np.mean(nb, axis=0)

    # smoothed heading + speed of the vehicle
    lo, hi = np.clip(idx - HEAD_WIN, 0, n - 1), np.clip(idx + HEAD_WIN, 0, n - 1)
    dxy = P[hi] - P[lo]
    dt = np.maximum(ts[hi] - ts[lo], 1e-6)
    speed = np.sqrt((dxy ** 2).sum(1)) / dt
    gps_brg = _bearing(dxy[:, 0], dxy[:, 1])
    hd = _hdiff(gps_brg[:, None], arc_brg[cand])
    cosh = np.cos(np.radians(hd))
    conf = np.clip(speed / 5.0, 0, 1)[:, None]

    out = {
        "trip_id": np.repeat(trip_id, n * kfin),
        "point_idx": np.repeat(idx, kfin),
        "orig_idx": np.repeat(track["orig_idx"].values, kfin),
        "candidate_arc_id": cand.ravel(),
        "dist_m": dist.ravel(),
        "dist_rank": np.tile(np.arange(kfin), n),
        "dist_gap_m": (dist - dist[:, [0]]).ravel(),
        "nbr_dist_m": nbr.ravel(),
        "heading_diff_deg": hd.ravel(),
        "cos_heading": cosh.ravel(),
        "cos_heading_w": (cosh * conf).ravel(),
        "speed_mps": np.repeat(speed, kfin),
        "along_frac": along.ravel(),
        "arc_len_m": arc_len[cand].ravel(),
    }
    df = pd.DataFrame(out)
    if route is not None:
        gt = set(route["arc_id"].astype(int).tolist())
        df["label"] = np.isin(df["candidate_arc_id"].values, list(gt)).astype(np.int8)
    return df


def build_feature_dataset(partition_dir, trip_ids, with_route=True):
    tables = []
    for tid in trip_ids:
        d = load_trip(partition_dir, tid, with_route=with_route)
        if any(d[k] is None for k in d):
            print("skipping (missing file):", tid)
            continue
        t = build_trip_features(d["track"], d["nodes"], d["arcs"],
                                d.get("route"), trip_id=tid)
        if t is not None:
            tables.append(t)
    return pd.concat(tables, ignore_index=True)


# ---------------------------------------------------------------- metrics
def observation_metrics(trip_id, point_idx, prob, label):
    """Observation-level metrics. For each GPS point take the highest-scored candidate.
    top1_all  : correct / all points (points whose reference arc is missing count as wrong)
    top1_cond : correct / points where the reference arc is in the candidate set
    miss_rate : share of points with NO reference arc among the candidates"""
    key = pd.factorize(pd.Series(trip_id).astype(str) + "_" + pd.Series(point_idx).astype(str))[0]
    prob, label = np.asarray(prob), np.asarray(label)
    order = np.lexsort((-prob, key))
    ks = key[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    top = order[first]
    n_pts = len(top)
    has_pos = np.bincount(key, weights=label) > 0
    correct = label[top].sum()
    return {
        "top1_all": correct / n_pts,
        "top1_cond": correct / max(has_pos.sum(), 1),
        "miss_rate": 1 - has_pos.mean(),
        "n_points": n_pts,
    }
