#!/usr/bin/env python3
"""E7: the E6 audit system ported to the high-DOF libraries.

Libraries: panda55 (7-DOF, 55 regions, 741 overlaps) and bimanual219
(14-DOF, 219 regions, 2757 overlaps).  Nothing to draw in 14-D; the
audits are the deliverable, plus first-solution latency and anytime
quality against the stored cluster references (GCS* t_best / cost,
global lower bounds from the certified runs) on the SAME queries.

Ground-truth layers (the 2-D obstacle polygons get replaced by the real
thing):
  * geometric: samples / edge probes / path probes inside their declared
    convex objects -- dimension-free, identical to 2-D;
  * physical: the scene's SceneGraphCollisionChecker re-checks every
    incumbent path and a sample subset.  IrisZo regions are only
    probabilistically sound, so this doubles as a region audit
    (the lesson of the UAV library saga);
  * junctions: every incumbent transition needs interface dim >= d-1
    (all interfaces here are full-dimensional overlaps -- verified at
    classification time -- and the audit re-checks each path anyway);
  * soundness: final cost >= the stored global lower bound.

High-D adaptations (both disclosed in output):
  * informed keys are closed-form bbox lower bounds on ell(P) (exact
    per-interface SOCPs would dominate the anytime clock); any lower
    bound keeps the filter admissible;
  * intra-region cliques are k-NN-sparsified above a size threshold
    (a subset of valid edges: soundness untouched, path quality may
    dip slightly).

    python3 e7_highd_audit.py --preset panda55
    python3 e7_highd_audit.py --preset biman219
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import e6_sampling as e6
import stageB_bimanual_ipgcs as sb
from e6_step2_anytime import AnytimeInterfacePlanner

PRESETS = {
    "panda55": {
        "lib": "out/panda_regions.pkl",
        "topo_cache": "out/e1_topo_panda55.pkl",
        "iface_cache": "out/e7_ifaces_panda55.pkl",
        "bench": "out/bench_panda1raw55.jsonl",
        "ref_arm": "gcsstar", "lb_field": ("ipgcs", "lb"),
        "scene": "panda",
    },
    "biman219": {
        "lib": "out/bimanual_regions_scale.pkl",
        "topo_cache": "out/bimanual219_topo.pkl",
        "iface_cache": "out/e7_ifaces_biman219.pkl",
        "bench": "out/bench_219focus_K1.jsonl",
        "ref_arm": "igcsstar", "lb_field": ("certify", "lb"),
        "scene": "bimanual",
    },
    # pocket-excised (repaired) libraries; queries and references stay
    # those of the original benchmark -- the old geometric LB remains a
    # valid lower bound because repair only shrinks the feasible set
    "panda55R": {
        "lib": "out/panda_regionsR.pkl",
        "topo_cache": "out/panda55R_topo.pkl",
        "iface_cache": "out/e7_ifaces_panda55R.pkl",
        "bench": "out/bench_panda1raw55.jsonl",
        "ref_arm": "gcsstar", "lb_field": ("ipgcs", "lb"),
        "scene": "panda",
    },
    "biman219R": {
        "lib": "out/bimanual_regions_scaleR.pkl",
        "topo_cache": "out/bimanual219R_topo.pkl",
        "iface_cache": "out/e7_ifaces_biman219R.pkl",
        "bench": "out/bench_219focus_K1.jsonl",
        "ref_arm": "igcsstar", "lb_field": ("certify", "lb"),
        "scene": "bimanual",
    },
}


def load_topo(cfg):
    data = pickle.load(open(cfg["lib"], "rb"))
    reg_dict = data["regions"] if isinstance(data, dict) and \
        "regions" in data else data
    regions = [(np.asarray(A), np.asarray(b)) for A, b in reg_dict.values()]
    topo = pickle.load(open(cfg["topo_cache"], "rb"))
    if isinstance(topo, dict) and "topo" in topo:
        topo = topo["topo"]
    assert topo["n"] == len(regions), "topology/library mismatch"
    return topo


def load_ifaces(cfg, topo):
    import os
    if os.path.exists(cfg["iface_cache"]):
        return pickle.load(open(cfg["iface_cache"], "rb"))
    t0 = time.perf_counter()
    ifs = e6.classify_interfaces(topo)
    print(f"classified {len(ifs)} interfaces in "
          f"{time.perf_counter()-t0:.1f}s", flush=True)
    pickle.dump(ifs, open(cfg["iface_cache"], "wb"))
    return ifs


def bbox_ell_keys(topo, ifaces, s, g):
    """Closed-form admissible keys: ell(P_uv) >= dist(s, B_uv) +
    dist(g, B_uv) with B_uv = bbox(X_u) ^ bbox(X_v) >= P_uv."""
    los, his = np.asarray(topo["bbox_lo"]), np.asarray(topo["bbox_hi"])
    eids = list(ifaces)
    u = np.array([ifaces[e]["u"] for e in eids])
    v = np.array([ifaces[e]["v"] for e in eids])
    lo = np.maximum(los[u], los[v])
    hi = np.minimum(his[u], his[v])

    def dist(p):
        d = np.maximum(lo - p, 0) + np.maximum(p - hi, 0)
        return np.linalg.norm(d, axis=1)

    vals = dist(s) + dist(g)
    return {e: float(x) for e, x in zip(eids, vals)}


def polish_waypoints(rgq, pseq, s, g):
    """Restriction SOCP returning the waypoint PATH (not just the cost),
    so the polished output can be physically validated like any other."""
    import cvxpy as cp
    if not pseq:
        return None, None
    d = len(s)
    xs = [cp.Variable(d) for _ in pseq]
    cons = []
    for x, eid in zip(xs, pseq):
        p = rgq["portals"][eid]
        cons.append(p["A"] @ x <= p["b"])
    cost = cp.norm(xs[0] - s)
    for a, b_ in zip(xs, xs[1:]):
        cost += cp.norm(b_ - a)
    cost += cp.norm(g - xs[-1])
    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver="CLARABEL")
    if xs[0].value is None:
        return None, None
    return float(prob.value), np.vstack([s] + [x.value for x in xs] + [g])


def build_path_audit(scene):
    """Physical ground truth, upgraded to a CONTINUOUS certificate: the
    clearance/Lipschitz sweep of e8_validate (no probe-resolution
    loophole).  Returns (audit_fn, checker, validator)."""
    from e8_validate import ContinuousValidator, L_BOUNDS
    try:
        if scene == "panda":
            from panda_scene import build_scene
            checker, _, _, _ = build_scene()
        elif scene == "bimanual":
            from stageB_bimanual_regions import build
            checker, _ = build()
        else:
            return None, None, None
    except Exception as exc:
        print(f"scene build failed ({str(exc)[:80]}) -> "
              f"collision layer OFF", flush=True)
        return None, None, None
    validator = ContinuousValidator(checker, L_BOUNDS[scene])
    return validator.path, checker, validator


class HighDPlanner(AnytimeInterfacePlanner):
    """kNN-sparsified intra-region edges above a clique threshold, plus
    lazy validation: a physically-failed incumbent is rejected and its
    offending edges blacklisted, so the next solve routes around the
    unsound region pocket instead of re-finding the same path."""

    KNN = 10
    CLIQUE_MAX = 24

    def __init__(self, *a, validator=None, **kw):
        super().__init__(*a, **kw)
        self.strict_audit = False
        self.blacklist = set()        # frozensets of node ids / 'S',''G'
        self.validator = validator    # for local repair of failed paths
        self._region_hulls = {}
        self._repair_rng = np.random.default_rng(int(self.seed) + 977)
        self.n_repaired = 0

    def _attempt_repair(self, path, cost, aud):
        if self.validator is None:
            return path, cost, False
        from e8_validate import path_cost, repair_path
        p2, ok, n_vias = repair_path(path, self.validator, self.regions,
                                     self._region_hulls, self._repair_rng)
        if not ok:
            return path, cost, False
        # geometric sanity on the repaired path before accepting it
        okj = e6.audit_junctions(p2, self.regions,
                                 dim_cache=self._dim_cache)
        if okj["bad_junctions"]:
            return path, cost, False
        self.n_repaired += 1
        return p2, path_cost(p2), True

    def _edge_key(self, a, b, N):
        aa = "S" if a == N else ("G" if a == N + 1 else int(a))
        bb = "S" if b == N else ("G" if b == N + 1 else int(b))
        return frozenset((aa, bb))

    def _on_reject(self, path, aud):
        idx = self._last_idx
        if idx is None:
            return
        N = len(self.pos)
        bad = aud.get("bad_segments")
        segs = bad if bad else range(len(idx) - 1)
        for k in segs:
            self.blacklist.add(self._edge_key(idx[k], idx[k + 1], N))

    def _solve(self):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        from scipy.spatial import cKDTree
        pos = np.asarray(self.pos)
        N = len(pos)
        alive = self._alive_nodes()
        s_idx, g_idx = N, N + 1
        ii, jj = [], []
        for r, nodes in self.region_nodes.items():
            nd = [k for k in nodes if alive is None or k in alive]
            nd = sorted(set(nd))
            if len(nd) < 2:
                continue
            a = np.asarray(nd)
            if len(a) <= self.CLIQUE_MAX:
                iu, ju = np.triu_indices(len(a), k=1)
                ii.append(a[iu])
                jj.append(a[ju])
            else:
                P = pos[a]
                k = min(self.KNN + 1, len(a))
                _, nb = cKDTree(P).query(P, k=k)
                src = np.repeat(a, k - 1)
                dst = a[nb[:, 1:]].ravel()
                # kNN alone can split a region's subgraph into islands
                # (all edges of the full clique are valid, connectivity
                # is not guaranteed by nearest neighbors in high-D);
                # a chain over the node list restores it for n-1 edges
                src = np.concatenate([src, a[:-1]])
                dst = np.concatenate([dst, a[1:]])
                lo = np.minimum(src, dst)
                hi = np.maximum(src, dst)
                ii.append(lo)
                jj.append(hi)
        s_mem = e6.region_membership(self.s, self.regions)
        g_mem = e6.region_membership(self.g, self.regions)
        qi, qj = [], []
        for r in s_mem:
            for k in self.region_nodes.get(r, []):
                if alive is None or k in alive:
                    qi.append(s_idx)
                    qj.append(k)
        for r in g_mem:
            for k in self.region_nodes.get(r, []):
                if alive is None or k in alive:
                    qi.append(g_idx)
                    qj.append(k)
        if s_mem & g_mem:
            qi.append(s_idx)
            qj.append(g_idx)
        if not qi and not ii:
            return np.inf, None
        allpos = np.vstack([pos, self.s, self.g])
        if ii:
            ii = np.concatenate(ii).astype(np.int64)
            jj = np.concatenate(jj).astype(np.int64)
            keys = np.unique(ii * (N + 2) + jj)
            ii, jj = keys // (N + 2), keys % (N + 2)
        else:
            ii = jj = np.zeros(0, np.int64)
        qi = np.asarray(qi, np.int64)
        qj = np.asarray(qj, np.int64)
        I = np.concatenate([ii, qi])
        J = np.concatenate([jj, qj])
        if self.blacklist:
            keep = np.array([self._edge_key(a, b, N) not in self.blacklist
                             for a, b in zip(I, J)])
            I, J = I[keep], J[keep]
            if len(I) == 0:
                return np.inf, None
        W = np.linalg.norm(allpos[I] - allpos[J], axis=1)
        m = coo_matrix((np.concatenate([W, W]),
                        (np.concatenate([I, J]), np.concatenate([J, I]))),
                       shape=(N + 2, N + 2)).tocsr()
        dist, pred = dijkstra(m, indices=s_idx, return_predecessors=True)
        if not np.isfinite(dist[g_idx]):
            self._last_idx = None
            return np.inf, None
        path = [g_idx]
        while path[-1] != s_idx:
            path.append(pred[path[-1]])
        path = path[::-1]
        self._last_idx = list(path)
        return float(dist[g_idx]), allpos[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    ap.add_argument("--modes", default="contour,area")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--no-scene", action="store_true",
                    help="skip the collision-checker layer")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    out_file = args.out or f"out/e7_{args.preset}.json"

    topo = load_topo(cfg)
    ifaces = load_ifaces(cfg, topo)
    dims = {f["dim"] for f in ifaces.values()}
    d = len(topo["bbox_lo"][0])
    print(f"{args.preset}: {topo['n']} regions, {len(ifaces)} interfaces, "
          f"dims={sorted(dims)} (d={d})", flush=True)
    full_dim = all(f["dim"] == d for f in ifaces.values())
    print(f"all interfaces full-dimensional: {full_dim} "
          f"(raw contour {'legal' if full_dim else 'NOT legal -> inset'})",
          flush=True)

    path_audit, checker, validator = (None, None, None) if args.no_scene \
        else build_path_audit(cfg["scene"])
    if path_audit is None:
        print("collision-checker layer: OFF", flush=True)
    else:
        print("collision-checker layer: ON (continuous certificate, "
              f"L={validator.L})", flush=True)

    bench = [json.loads(l) for l in open(cfg["bench"])]
    if args.max_queries:
        bench = bench[:args.max_queries]
    dim_cache = {}
    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}
    regions = topo["regions"]
    rows = []
    for rec in bench:
        s, g = np.asarray(rec["s"]), np.asarray(rec["g"])
        ref = rec.get(cfg["ref_arm"], {})
        ref_t = ref.get("t_best") or ref.get("t")
        ref_c = ref.get("cost")
        lb_arm, lb_key = cfg["lb_field"]
        ref_lb = (rec.get(lb_arm) or {}).get(lb_key)
        best = rec.get("best_known")
        ell = bbox_ell_keys(topo, ifaces, s, g)
        for mode in args.modes.split(","):
            pl = HighDPlanner(topo, ifaces, s, g, mode=mode, M=args.M,
                              informed=True, workers=1, seed=0,
                              ell=ell, dim_cache=dim_cache,
                              path_audit=path_audit,
                              validator=validator).run(args.rounds)
            ev = pl.events
            t_first = ev[0][0] if ev else None
            c_first = ev[0][1] if ev else None
            # corridor SOCP polish: sampling discovers the corridor, the
            # convex program places the waypoints exactly.  The polished
            # path MOVES waypoints, so it must re-pass the physical
            # layer; if dirty it gets the same repair treatment, and on
            # failure the pre-polish physical path stands.
            c_pol, t_pol, pol_valid = np.inf, 0.0, None
            if pl.best_path is not None:
                t0p = time.perf_counter()
                mems = [e6.region_membership(p, regions)
                        for p in pl.best_path]
                pseq = e6.corridor_from_path(mems, pair_eid)
                if pseq is not None:
                    rgq = sb.with_query(topo, s, g)
                    c_geo, ppath = polish_waypoints(rgq, pseq, s, g)
                    if ppath is not None:
                        if validator is not None:
                            from e8_validate import path_cost, repair_path
                            okp, _ = validator.path(ppath)
                            if not okp:
                                ppath, okp, _ = repair_path(
                                    ppath, validator, regions,
                                    pl._region_hulls,
                                    np.random.default_rng(7))
                            pol_valid = bool(okp)
                            if okp:
                                c_pol = path_cost(ppath)
                        else:
                            c_pol = c_geo
                t_pol = time.perf_counter() - t0p
            c_out = min(pl.c_best, c_pol)
            sound = True
            if ref_lb and np.isfinite(c_out):
                sound = c_out >= ref_lb - 1e-6 * max(1.0, ref_lb)
            row = {"i": rec["i"], "mode": mode,
                   "t_first": t_first, "c_first": c_first,
                   "c_final": float(pl.c_best),
                   "t_total": sum(r["t_round"] for r in pl.rows),
                   "n_events": len(ev), "n_nodes": len(pl.pos),
                   "alive_frac": pl.rows[-1]["n_ifaces_alive"] / len(ifaces),
                   "ref_t_best": ref_t, "ref_cost": ref_c,
                   "best_known": best, "ref_lb": ref_lb,
                   "excess_vs_best": (pl.c_best / best - 1)
                   if best and np.isfinite(pl.c_best) else None,
                   "c_polished": float(c_pol), "t_polish": t_pol,
                   "polish_valid": pol_valid,
                   "n_repaired": pl.n_repaired,
                   "excess_polished": (c_out / best - 1)
                   if best and np.isfinite(c_out) else None,
                   "sound_ge_lb": bool(sound),
                   "n_rejections": len(pl.rejections),
                   "n_blacklisted": len(pl.blacklist),
                   "events": [(float(t), float(c)) for t, c in ev]}
            rows.append(row)
            ex = row["excess_vs_best"]
            exp = row["excess_polished"]
            print(f"[q{rec['i']:2d}] {mode:8s} "
                  f"first {t_first if t_first is None else round(t_first,2)}s"
                  f"/{'-' if c_first is None else round(c_first,3)} "
                  f"final {pl.c_best:8.3f} "
                  f"({'-' if ex is None else f'{100*ex:+.1f}%'}) "
                  f"polished {c_pol:8.3f} "
                  f"({'-' if exp is None else f'{100*exp:+.1f}%'}, "
                  f"+{t_pol:.2f}s) "
                  f"| ref {cfg['ref_arm']}: "
                  f"{'-' if ref_t is None else round(ref_t,2)}s"
                  f"/{'-' if ref_c is None else round(ref_c,3)} "
                  f"| alive {100*row['alive_frac']:.0f}% "
                  f"{'OK' if sound else 'LB-FAIL'}", flush=True)

    # roadmap-sample subset physical audit (region soundness spot check)
    subset_bad = None
    if checker is not None and rows:
        rng = np.random.default_rng(0)
        pl = HighDPlanner(topo, ifaces,
                          np.asarray(bench[0]["s"]), np.asarray(bench[0]["g"]),
                          mode="contour", M=2, informed=False, seed=1,
                          ell=bbox_ell_keys(topo, ifaces,
                                            np.asarray(bench[0]["s"]),
                                            np.asarray(bench[0]["g"])),
                          dim_cache=dim_cache).run(2)
        pts = np.asarray(pl.pos)
        idx = rng.choice(len(pts), min(2000, len(pts)), replace=False)
        subset_bad = int(sum(
            0 if checker.CheckConfigCollisionFree(pts[k]) else 1
            for k in idx))
        print(f"region-soundness spot check: {subset_bad}/{len(idx)} "
              f"interface samples in collision", flush=True)

    ok = all(r["sound_ge_lb"] for r in rows)
    print(f"\nALL LB CHECKS PASS: {ok}")
    json.dump({"preset": args.preset, "rows": rows,
               "interface_dims": sorted(dims), "full_dim": full_dim,
               "collision_layer": path_audit is not None,
               "sample_subset_collisions": subset_bad,
               "all_lb_pass": ok},
              open(out_file, "w"), indent=1)
    print(f"written {out_file}")


if __name__ == "__main__":
    main()
