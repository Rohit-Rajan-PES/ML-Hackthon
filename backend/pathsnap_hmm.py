"""PathSnap HMM / Viterbi route reconstruction (shared by the notebook and the API).

Hidden state  : at GPS point i, the state is WHICH of the K candidate road arcs the vehicle is on.
Emission      : w_emit * log(P(candidate is correct)) from the classifier (floored for stability).
Transition    : based on road-network connectivity between candidate arcs of consecutive points.
Decoding      : Viterbi over the time-varying candidate lattice (log-space).
"""
import difflib
import numpy as np
import pandas as pd

DEFAULT_PARAMS = dict(
    w_emit=1.0,        # scale of the classifier log-probability
    lam_hop=1.0,       # cost per road hop when moving forward along connected arcs
    lam_back=2.5,      # cost for a small backwards step (GPS noise jitter)
    pen_disc=6.0,      # cost for a disconnected jump (finite, so decoding never breaks)
    max_hops=4,        # forward hops searched between consecutive GPS points
    gap_soft_s=3.0,    # every this many seconds of gap widens the hop search by 1
    gap_reset_s=30.0,  # gaps this long (or longer) -> no transition constraint (restart)
    p_floor=1e-4,      # floor for classifier probabilities before taking log
)


class RoadGraph:
    """Directed arc graph: arc a -> arc b if arcs[a].to_node == arcs[b].from_node."""

    def __init__(self, arcs):
        self.from_n = arcs["from_node_id"].values.astype(np.int64)
        self.to_n = arcs["to_node_id"].values.astype(np.int64)
        n_nodes = int(max(self.from_n.max(), self.to_n.max())) + 1
        order = np.argsort(self.from_n, kind="stable")
        self.out_arcs = order
        self.out_ptr = np.searchsorted(self.from_n[order], np.arange(n_nodes + 1))
        self._cache = {}

    def successors(self, a):
        n = self.to_n[a]
        return self.out_arcs[self.out_ptr[n]:self.out_ptr[n + 1]]

    def reach(self, a, max_hops):
        """{arc: hops} reachable from arc a within max_hops (a itself = 0 hops)."""
        key = (a, max_hops)
        if key in self._cache:
            return self._cache[key]
        dist = {a: 0}
        frontier = [a]
        for h in range(1, max_hops + 1):
            nxt = []
            for x in frontier:
                for s in self.successors(x):
                    s = int(s)
                    if s not in dist:
                        dist[s] = h
                        nxt.append(s)
            frontier = nxt
            if not frontier:
                break
        self._cache[key] = dist
        return dist

    def transition(self, prev, nxt, dt, p):
        """Log-transition matrix (len(prev) x len(nxt)). Never -inf: invalid transitions get a finite penalty."""
        if dt >= p["gap_reset_s"]:                       # long GPS gap: restart, no constraint
            return np.zeros((len(prev), len(nxt)))
        extra = int(max(dt - 1.0, 0.0) // p["gap_soft_s"])   # shorter gaps: search further, penalise less
        H = min(int(p["max_hops"]) + extra, 10)
        pen = p["pen_disc"] / (1 + extra)

        T = np.full((len(prev), len(nxt)), -pen)
        nidx = {int(b): j for j, b in enumerate(nxt)}
        pidx = {int(a): i for i, a in enumerate(prev)}

        for i, a in enumerate(prev):                      # forward moves (same arc = 0 hops)
            for b, h in self.reach(int(a), H).items():
                j = nidx.get(b)
                if j is not None:
                    T[i, j] = -min(p["lam_hop"] * h, pen)
        back = -min(p["lam_back"], pen)                   # small backwards step: a is a successor of b
        for j, b in enumerate(nxt):
            for a, h in self.reach(int(b), 1).items():
                if h == 1:
                    i = pidx.get(a)
                    if i is not None and T[i, j] < back:
                        T[i, j] = back
        return T


def decode_trip(feats, graph, timestamps, params=None):
    """feats: candidate table of ONE trip with columns point_idx, dist_rank, orig_idx,
    candidate_arc_id, prob.  timestamps: cleaned-track timestamps indexed by point_idx.
    Returns one row per GPS point with the classifier-only arc and the HMM/Viterbi arc."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    f = feats.sort_values(["point_idx", "dist_rank"], kind="stable")
    if len(f) == 0:
        return pd.DataFrame(columns=["point_idx", "orig_idx", "clf_arc", "hmm_arc"])

    uniq, start = np.unique(f["point_idx"].values, return_index=True)
    cands = np.split(f["candidate_arc_id"].values.astype(np.int64), start[1:])
    probs = np.split(f["prob"].values.astype(float), start[1:])
    orig = f["orig_idx"].values[start]
    ts = np.asarray(timestamps, dtype=float)[uniq]
    n = len(uniq)

    emit = [p["w_emit"] * np.log(np.clip(pr, p["p_floor"], 1.0)) for pr in probs]
    clf = np.array([c[int(np.argmax(pr))] for c, pr in zip(cands, probs)])

    delta, back = emit[0], []
    for i in range(1, n):
        T = graph.transition(cands[i - 1], cands[i], ts[i] - ts[i - 1], p)
        sc = delta[:, None] + T
        bp = sc.argmax(axis=0)
        delta = sc[bp, np.arange(len(cands[i]))] + emit[i]
        back.append(bp)

    k = int(np.argmax(delta))
    path = [k]
    for bp in reversed(back):
        k = int(bp[k])
        path.append(k)
    path.reverse()
    hmm = np.array([cands[i][path[i]] for i in range(n)])
    return pd.DataFrame({"point_idx": uniq, "orig_idx": orig, "clf_arc": clf, "hmm_arc": hmm})


# ---------------------------------------------------------------- metrics
def dedup(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def route_metrics(pred_arcs, ref_route, graph, with_seq=False):
    """Pointwise accuracy, route-set P/R/F1, route continuity and (optionally) sequence similarity."""
    pred = np.asarray(pred_arcs)
    ref_list = dedup([int(x) for x in ref_route])
    ref_set = set(ref_list)
    d = dedup([int(x) for x in pred])
    ps = set(d)
    inter = len(ps & ref_set)
    prec = inter / max(len(ps), 1)
    rec = inter / max(len(ref_set), 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    pairs = len(d) - 1
    jumps = sum(1 for a, b in zip(d[:-1], d[1:]) if graph.to_n[a] != graph.from_n[b])
    out = {
        "point_acc": float(np.isin(pred, list(ref_set)).mean()),
        "route_prec": prec, "route_rec": rec, "route_f1": f1,
        "continuity": 1.0 - jumps / pairs if pairs > 0 else 1.0,
        "n_jumps": jumps,
        "n_segments": len(d),
    }
    if with_seq:
        out["seq_sim"] = difflib.SequenceMatcher(None, d, ref_list, autojunk=False).ratio()
    return out


# ---------------------------------------------------------------- one-call API helper
def reconstruct_trip(pf, artifact, hmm_params, track, nodes, arcs, trip_id="trip"):
    """Full inference for ONE trip (no route / ground truth needed):
    features -> classifier probabilities -> Viterbi. Returns the per-point DataFrame."""
    feats = pf.build_trip_features(track, nodes, arcs, route=None, trip_id=trip_id)
    feats["prob"] = artifact["model"].predict_proba(feats[artifact["features"]])[:, 1]
    clean, _ = pf.clean_track(track)
    return decode_trip(feats, RoadGraph(arcs), clean["timestamp"].values.astype(float), hmm_params)
