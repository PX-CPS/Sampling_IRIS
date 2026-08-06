#!/usr/bin/env python3
"""Stage B: the five-step loop on the real bimanual 14-DOF C-space.

Loads the IRIS regions from stageB_bimanual_regions.py, builds the
region-intersection graph in 14D (Chebyshev-center waypoints via LP),
and compares full-graph GCS vs iP-GCS (cold start -> restriction SOCP ->
per-region informed SOCP prune -> guard -> subgraph GCS) over seed-pair
queries. With tens of regions this is a correctness/bring-up experiment
(the C-space scale story needs larger region libraries); it also measures
the multi-query amortization case native to GCS roadmaps.

    python3 stageB_bimanual_ipgcs.py --regions out/bimanual_regions.pkl \
        --pairs 6 --out out/stageB_bimanual.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np

for p in (os.environ.get("GCS_SR", ""),
          os.path.expanduser("~/gcs-science-robotics")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ipgcs_core as core


def chebyshev_center(A, b):
    """LP: max r s.t. A x + ||a_i|| r <= b. Returns (center, radius)."""
    from scipy.optimize import linprog
    norms = np.linalg.norm(A, axis=1)
    d = A.shape[1]
    c = np.zeros(d + 1)
    c[-1] = -1.0
    Aub = np.hstack([A, norms[:, None]])
    res = linprog(c, A_ub=Aub, b_ub=b, bounds=[(None, None)] * d + [(0, None)],
                  method="highs")
    if not res.success:
        return None, -1.0
    return res.x[:-1], res.x[-1]


def region_bboxes(regions):
    """Per-dimension bounds of each H-polytope (2d LPs per region). Computed
    once at topology time; powers the closed-form prune tiers per query."""
    from scipy.optimize import linprog
    los, his = [], []
    for A, b in regions:
        d = A.shape[1]
        lo = np.empty(d)
        hi = np.empty(d)
        for k in range(d):
            c = np.zeros(d)
            c[k] = 1.0
            r = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * d,
                        method="highs")
            lo[k] = r.x[k] if r.success else -np.inf
            r = linprog(-c, A_ub=A, b_ub=b, bounds=[(None, None)] * d,
                        method="highs")
            hi[k] = r.x[k] if r.success else np.inf
        los.append(lo)
        his.append(hi)
    return np.asarray(los), np.asarray(his)


def build_topology(regions):
    """Query-independent intersection graph (portals, adjacency). Computed
    ONCE and reused across queries — the expensive part (n^2 Chebyshev LPs)."""
    n = len(regions)
    portals, region_portals = {}, {i: [] for i in range(n)}
    edges = []
    eid = 0
    for i, j in itertools.combinations(range(n), 2):
        A = np.vstack([regions[i][0], regions[j][0]])
        b = np.concatenate([regions[i][1], regions[j][1]])
        ctr, r = chebyshev_center(A, b)
        if ctr is None or r < 1e-9:
            continue
        portals[eid] = {"u": i, "v": j, "A": A, "b": b, "center": ctr,
                        "radius": r}
        region_portals[i].append(eid)
        region_portals[j].append(eid)
        edges.append((i, j))
        eid += 1
    los, his = region_bboxes(regions)
    return {"regions": regions, "edges": edges, "portals": portals,
            "region_portals": region_portals, "n": n,
            "bbox_lo": los, "bbox_hi": his}


def with_query(topo, start, goal):
    """Attach a start/goal to a precomputed topology (cheap: 2n containment)."""
    tol = 1e-7
    regions = topo["regions"]
    sregs = [i for i, (A, b) in enumerate(regions)
             if np.all(A @ start <= b + tol)]
    gregs = [i for i, (A, b) in enumerate(regions)
             if np.all(A @ goal <= b + tol)]
    return {**topo, "start": np.asarray(start), "goal": np.asarray(goal),
            "start_regions": sregs, "goal_regions": gregs}


def build_region_graph(regions, start, goal):
    return with_query(build_topology(regions), start, goal)


def build_waypoint_graph(rg, alive=None):
    alive = alive if alive is not None else set(range(rg["n"]))
    pos = [rg["start"], rg["goal"]]
    node_portal, portal_nodes = {}, {}
    for eid, p in rg["portals"].items():
        if p["u"] not in alive or p["v"] not in alive:
            continue
        node_portal[len(pos)] = eid
        portal_nodes[eid] = [len(pos)]
        pos.append(p["center"])
    pos = np.asarray(pos)
    adj = [[] for _ in range(len(pos))]
    n_edges = 0

    def connect(i, j):
        nonlocal n_edges
        d = float(np.linalg.norm(pos[i] - pos[j]))
        adj[i].append((j, d))
        adj[j].append((i, d))
        n_edges += 1

    for r in alive:
        eids = [e for e in rg["region_portals"][r] if e in portal_nodes]
        for a in range(len(eids)):
            for c in range(a + 1, len(eids)):
                connect(portal_nodes[eids[a]][0], portal_nodes[eids[c]][0])
        if r in rg["start_regions"]:
            for e in eids:
                connect(0, portal_nodes[e][0])
        if r in rg["goal_regions"]:
            for e in eids:
                connect(1, portal_nodes[e][0])
    if set(rg["start_regions"]) & set(rg["goal_regions"]) & alive:
        connect(0, 1)
    return core.WaypointGraph(pos, adj, node_portal, portal_nodes, n_edges)


def restriction_socp(rg, portal_seq):
    import cvxpy as cp
    K = len(portal_seq)
    if K == 0:
        return float(np.linalg.norm(rg["goal"] - rg["start"])), 0.0
    d = len(rg["start"])
    xs = [cp.Variable(d) for _ in range(K)]
    cons = []
    for x, eid in zip(xs, portal_seq):
        p = rg["portals"][eid]
        cons.append(p["A"] @ x <= p["b"])
    cost = cp.norm(xs[0] - rg["start"])
    for a, b_ in zip(xs, xs[1:]):
        cost += cp.norm(b_ - a)
    cost += cp.norm(rg["goal"] - xs[-1])
    prob = cp.Problem(cp.Minimize(cost), cons)
    t0 = time.perf_counter()
    prob.solve(solver="CLARABEL")
    return float(prob.value), time.perf_counter() - t0


def informed_prune(rg, c_best):
    """3-tier nD prune (fix for the 219-region timeout): closed-form bbox and
    point-to-bbox tiers kill most regions; the per-region SOCP only touches
    borderline cases. Bboxes come precomputed from build_topology."""
    import cvxpy as cp
    s, g = rg["start"], rg["goal"]
    keep = set(rg["start_regions"]) | set(rg["goal_regions"])
    tol = max(1e-9, 1e-6 * c_best)
    los, his = rg["bbox_lo"], rg["bbox_hi"]
    # AABB of the prolate hyperspheroid
    sg = g - s
    c_min = float(np.linalg.norm(sg))
    a = 0.5 * c_best
    bmin = np.sqrt(max(a * a - 0.25 * c_min * c_min, 0.0))
    u = sg / (c_min + 1e-12)
    ext = np.sqrt((a * u) ** 2 + (bmin ** 2) * (1.0 - u ** 2))
    ctr = 0.5 * (s + g)

    survivors = set()
    t0 = time.perf_counter()
    n_socp = 0
    for rid, (A, b) in enumerate(rg["regions"]):
        if rid in keep:
            survivors.add(rid)
            continue
        lo, hi = los[rid], his[rid]
        # tier 1: box vs spheroid AABB
        if np.any(lo > ctr + ext) or np.any(hi < ctr - ext):
            continue
        # tier 2: closed-form point-to-box lower bound
        ds = np.linalg.norm(np.maximum(lo - s, 0) + np.maximum(s - hi, 0))
        dg = np.linalg.norm(np.maximum(lo - g, 0) + np.maximum(g - hi, 0))
        if ds + dg > c_best + tol:
            continue
        # tier 3: exact SOCP (borderline only)
        n_socp += 1
        d = A.shape[1]
        x = cp.Variable(d)
        prob = cp.Problem(
            cp.Minimize(cp.norm(x - s) + cp.norm(x - g)), [A @ x <= b])
        try:
            prob.solve(solver="CLARABEL")
            lb = float(prob.value)
        except Exception:
            lb = 0.0                       # conservative keep is admissible
        if lb <= c_best + tol:
            survivors.add(rid)
    return survivors, time.perf_counter() - t0


def solve_gcs(rg, alive=None):
    from pydrake.geometry.optimization import HPolyhedron
    from gcs.linear import LinearGCS
    alive = sorted(alive if alive is not None else range(rg["n"]))
    ridx = {r: k for k, r in enumerate(alive)}
    regions = [HPolyhedron(*rg["regions"][r]) for r in alive]
    edges = []
    for (i, j) in rg["edges"]:
        if i in ridx and j in ridx:
            edges.append((ridx[i], ridx[j]))
            edges.append((ridx[j], ridx[i]))
    t0 = time.perf_counter()
    g = LinearGCS(regions, edges)
    g.addSourceTarget(rg["start"], rg["goal"])
    wp, rd = g.SolvePath(rounding=True, verbose=False, preprocessing=False)
    return {"lb": rd.get("relaxation_cost"), "ub": rd.get("rounded_cost"),
            "t": time.perf_counter() - t0, "n_regions": len(regions),
            "waypoints": None if wp is None else np.asarray(wp).T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="out/bimanual_regions.pkl")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--guard", type=float, default=0.2)
    ap.add_argument("--full-limit", type=int, default=10 ** 9,
                    help="run the (expensive) full-graph baseline only on the "
                         "first N pairs; the rest report iP-GCS + certificate")
    ap.add_argument("--out", default="out/stageB_bimanual.jsonl")
    args = ap.parse_args()

    from reproduction.bimanual.helpers import getConfigurationSeeds
    seeds = getConfigurationSeeds()
    names = list(seeds.keys())
    reg_pkl = pickle.load(open(args.regions, "rb"))
    regions = [(np.asarray(A), np.asarray(b)) for A, b in reg_pkl.values()]
    rnames = list(reg_pkl.keys())
    print(f"{len(regions)} regions, dim={regions[0][0].shape[1]}", flush=True)

    # Build the query-independent intersection graph ONCE (the n^2-LP cost),
    # then reuse it for the reachability picker and every query.
    t0 = time.perf_counter()
    topo = build_topology(regions)
    print(f"topology: {len(topo['portals'])} portals in "
          f"{time.perf_counter()-t0:.1f}s", flush=True)
    reg_A = [np.asarray(A) for A, _ in reg_pkl.values()]
    reg_b = [np.asarray(b) for _, b in reg_pkl.values()]

    def region_of(q):
        for k in range(len(regions)):
            if np.all(reg_A[k] @ q <= reg_b[k] + 1e-7):
                return k
        return -1

    adj = {k: set() for k in range(len(regions))}
    for (i, j) in topo["edges"]:
        adj[i].add(j)
        adj[j].add(i)

    def component(k):
        seen, stack = {k}, [k]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    seed_region = {nm: region_of(np.asarray(seeds[nm])) for nm in names}
    rng = np.random.default_rng(7)
    pairs, tries = [], 0
    while len(pairs) < args.pairs and tries < 4000:
        tries += 1
        a, b = rng.choice(len(names), 2, replace=False)
        ra, rb = seed_region[names[a]], seed_region[names[b]]
        if ra < 0 or rb < 0 or ra == rb:
            continue
        if rb in component(ra) and (names[a], names[b]) not in pairs:
            pairs.append((names[a], names[b]))
    if not pairs:                          # fallback: any distinct pair
        pairs = [(names[0], names[1])]
    print(f"selected {len(pairs)} reachable seed-pairs", flush=True)

    with open(args.out, "w") as f:
        for pair_idx, (sname, gname) in enumerate(pairs):
            s, g = np.asarray(seeds[sname]), np.asarray(seeds[gname])
            rg = with_query(topo, s, g)
            if not rg["start_regions"] or not rg["goal_regions"]:
                print(f"{sname}->{gname}: seed not in any region, skip",
                      flush=True)
                continue
            run_full = pair_idx < args.full_limit
            full = solve_gcs(rg) if run_full else None
            t0 = time.perf_counter()
            wg = build_waypoint_graph(rg)
            path, _, exp, _ = core.astar(wg)
            if path is None:
                print(f"{sname}->{gname}: disconnected waypoint graph",
                      flush=True)
                continue
            pseq = [wg.node_portal[i] for i in path if i in wg.node_portal]
            c0, t_socp = restriction_socp(rg, pseq)
            t_first = time.perf_counter() - t0
            alive, t_prune = informed_prune(rg, c0)
            ratio = 1 - len(alive) / rg["n"]
            if ratio < args.guard:
                if full is None:            # guard needs the full solve
                    full = solve_gcs(rg)
                    run_full = True
                sub = dict(full)
                ip_total = t_first + t_prune + full["t"]
                guarded = True
            else:
                sub = solve_gcs(rg, alive=alive)
                ip_total = t_first + t_prune + sub["t"]
                guarded = False
            ub = min(c0, sub["ub"]) if sub["ub"] else c0
            row = {"start": sname, "goal": gname, "n_regions": rg["n"],
                   "n_portals": len(rg["portals"]),
                   "full_t": full["t"] if run_full else None,
                   "full_lb": full["lb"] if run_full else None,
                   "full_ub": full["ub"] if run_full else None,
                   "t_first": t_first, "c_first": c0,
                   "expansions": exp, "t_prune": t_prune,
                   "alive": len(alive), "prune_ratio": ratio,
                   "guard": guarded, "ip_total": ip_total, "ub": ub,
                   "lb": sub["lb"],
                   "gap": (ub - sub["lb"]) / max(ub, 1e-12),
                   "speedup": (full["t"] / ip_total) if run_full else None}
            f.write(json.dumps(row) + "\n")
            f.flush()
            full_txt = (f"full={full['t']:.2f}s ({row['speedup']:.1f}x)"
                        if run_full else "full=skipped (iP-only)")
            print(f"{sname}->{gname}: {full_txt} iP={ip_total:.2f}s "
                  f"prune={100*ratio:.0f}% gap={100*row['gap']:.2f}% "
                  f"first_ub={t_first*1e3:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
