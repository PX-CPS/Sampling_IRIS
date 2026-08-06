#!/usr/bin/env python3
"""Faithful reimplementation of GCS* (Chia et al., WAFR 2024,
arXiv:2407.08848) — the SAMPLING-based domination variant — on our
region-intersection GCS, plus the informed/GFP-augmented iGCS*.

Feasible set matches gcs-science-robotics LinearGCS exactly (portal-chain
trajectories), so costs are comparable across all methods in the bench.

GCS* (paper Alg. 2): best-first over region paths with
  f~(v) = ConvexRestriction(path, free terminal x) + |g - x|,
sampling-based ReachesCheaper domination on fixed per-vertex samples.

iGCS* adds three switchable ingredients (for ablation):
  prune  — admissible informed region pruning w.r.t. the restriction UB c0
  warm   — c0 as initial incumbent (anytime; enables pruning from t=0)
  deg    — GFP layer: (a) plateau-bucketed ordering: within an f~ tolerance
           band, expand higher-degree (hub) regions first; (b) domination
           sample budgets proportional to region degree. With plateau
           bucketing the returned solution is tau-optimal (tau = plateau
           fraction of the query scale); exact-f ordering when disabled.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time

import numpy as np


def _chain_socp(portals_Ab, s, terminal, fixed_x=None, goal=None):
    """min |y1-s| + sum |y_{i+1}-y_i| + (|x-ym|, x in terminal or fixed)
    (+ |goal-x| if goal is not None). Returns (cost, x_end)."""
    import cvxpy as cp
    d = len(s)
    ys = [cp.Variable(d) for _ in portals_Ab]
    cons = []
    for y, (A, b) in zip(ys, portals_Ab):
        cons.append(A @ y <= b)
    if fixed_x is not None:
        x = fixed_x
    else:
        x = cp.Variable(d)
        At, bt = terminal
        cons.append(At @ x <= bt)
    pts = [s] + ys + [x]
    cost = 0
    for a, b_ in zip(pts, pts[1:]):
        cost = cost + cp.norm(b_ - a)
    if goal is not None:
        cost = cost + cp.norm(goal - x)
    prob = cp.Problem(cp.Minimize(cost), cons)
    try:
        prob.solve(solver="CLARABEL")
    except cp.SolverError:
        return np.inf, None
    if prob.value is None or not np.isfinite(prob.value):
        return np.inf, None
    xv = fixed_x if fixed_x is not None else np.asarray(x.value)
    return float(prob.value), xv


def _sample_region(A, b, lo, hi, rng, k, tries=4000):
    out = []
    for _ in range(tries):
        if len(out) >= k:
            break
        q = lo + rng.random(len(lo)) * (hi - lo)
        if np.all(A @ q <= b + 1e-9):
            out.append(q)
    return out


def gcs_star(rg, timeout=60.0, K=5, max_revisits=1, seed=0, verbose=False,
             alive=None, incumbent=None, order_prior=None,
             plateau=None, deg_samples=False):
    """rg: query graph from stageB_bimanual_ipgcs.with_query.
    Returns dict(cost, path, t, t_best, expansions, status, lb, gap)."""
    rng = np.random.default_rng(seed)
    s, g = rg["start"], rg["goal"]
    n = rg["n"]
    portals, rp = rg["portals"], rg["region_portals"]
    alive = set(range(n)) if alive is None else set(alive)
    deg = {v: len(rp[v]) for v in range(n)}
    mean_deg = max(np.mean([deg[v] for v in alive]) if alive else 1.0, 1.0)

    def portal_between(a, b_):
        for eid in rp[a]:
            p = portals[eid]
            if {p["u"], p["v"]} == {a, b_}:
                return (p["A"], p["b"])
        return None

    regionAb = rg["regions"]
    lo, hi = rg["bbox_lo"], rg["bbox_hi"]
    samples = {}
    bestG = {}

    goal_regions = set(rg["goal_regions"]) & alive
    start_regions = [r for r in rg["start_regions"] if r in alive]
    if not start_regions or not goal_regions:
        return {"cost": np.inf, "path": None, "t": 0.0, "t_best": None,
                "expansions": 0, "status": "no_start_or_goal_region",
                "lb": 0.0, "gap": None}

    scale = max(float(np.linalg.norm(np.asarray(g) - np.asarray(s))), 1e-6)
    tau = (plateau or 0.0) * scale

    t0 = time.perf_counter()
    cnt = itertools.count()
    Q = []          # entries: (key_tuple, cnt, f, path)
    best = {"cost": np.inf, "path": None, "t_best": None}
    if incumbent is not None:
        best = {"cost": float(incumbent), "path": "warm", "t_best": 0.0}
    expansions = 0
    lb = 0.0

    def K_of(v):
        if not deg_samples:
            return K
        return int(np.clip(round(K * deg[v] / mean_deg), 2, 2 * K))

    def chain_of(path):
        return [portal_between(a, b_) for a, b_ in zip(path, path[1:])]

    def push(path):
        nonlocal best
        chain = chain_of(path)
        v = path[-1]
        if v in goal_regions:
            c_exact, _ = _chain_socp(chain, s, None, fixed_x=g)
            if c_exact < best["cost"]:
                best = {"cost": c_exact, "path": list(path),
                        "t_best": time.perf_counter() - t0}
        f, _ = _chain_socp(chain, s, regionAb[v], goal=g)
        if f >= best["cost"] - 1e-9:
            return
        if v not in samples:
            A, b_ = regionAb[v]
            pts_ = _sample_region(A, b_, lo[v], hi[v], rng, K_of(v))
            if not pts_:
                import stageB_bimanual_ipgcs as sb
                c, _r = sb.chebyshev_center(A, b_)
                pts_ = [np.asarray(c)] if c is not None else []
            samples[v] = pts_
            bestG[v] = np.full(len(pts_), np.inf)
        if len(samples[v]) == 0:
            return
        Gv = np.array([_chain_socp(chain, s, None, fixed_x=x)[0]
                       for x in samples[v]])
        if np.all(Gv >= bestG[v] - 1e-9):
            return
        bestG[v] = np.minimum(bestG[v], Gv)
        if tau > 0 and order_prior is not None:
            key = (math.floor(f / tau), -order_prior(v), f)
        elif order_prior is not None:
            key = (f, -order_prior(v), 0.0)
        else:
            key = (f, 0.0, 0.0)
        heapq.heappush(Q, (key, next(cnt), f, list(path)))

    for r0 in start_regions:
        push([r0])
    status = "optimal"
    while Q:
        if time.perf_counter() - t0 > timeout:
            status = "timeout"
            break
        key, _, f, path = heapq.heappop(Q)
        lb = max(lb, key[0] * tau if tau > 0 else f)
        if f >= best["cost"] - 1e-9:
            if tau == 0:
                status = "optimal"
                lb = best["cost"]
                break
            # bucketed order: terminate only when the bucket floor clears
            # the incumbent (tau-optimality)
            if key[0] * tau >= best["cost"] - 1e-9:
                status = "tau_optimal"
                lb = best["cost"] - tau
                break
            continue
        expansions += 1
        v = path[-1]
        for eid in rp[v]:
            p = portals[eid]
            w = p["v"] if p["u"] == v else p["u"]
            if w not in alive or path.count(w) > max_revisits:
                continue
            push(path + [w])
    else:
        status = "optimal" if tau == 0 else "tau_optimal"
        if np.isfinite(best["cost"]):
            lb = best["cost"] if tau == 0 else best["cost"] - tau
    if status == "timeout" and Q:
        lb = max(lb, min(item[2] for item in Q))
    gap = (best["cost"] - lb) / best["cost"] \
        if np.isfinite(best["cost"]) and best["cost"] > 0 else None
    return {"cost": best["cost"], "path": best["path"],
            "t": time.perf_counter() - t0, "t_best": best["t_best"],
            "expansions": expansions, "status": status,
            "lb": float(lb), "gap": gap}


def igcs_star(rg, timeout=60.0, K=5, max_revisits=1, seed=0,
              use_prune=True, use_warm=True, use_deg=False,
              plateau=1e-3):
    # default = PW (ablation-best); deg/GFP layer measured as a null
    # (adds per-expansion sampling cost, changes no expansion outcome)
    """informed GCS* with ablation switches (prune / warm / deg=GFP)."""
    import ipgcs_core as core
    import stageB_bimanual_ipgcs as sb
    t0 = time.perf_counter()
    alive, c0 = None, None
    if use_prune or use_warm:
        wg = sb.build_waypoint_graph(rg)
        path, _, _, _ = core.astar(wg)
        if path is None:
            return {"cost": np.inf, "path": None, "t": 0.0, "t_best": None,
                    "expansions": 0, "status": "disconnected",
                    "lb": 0.0, "gap": None}
        pseq = [wg.node_portal[i] for i in path if i in wg.node_portal]
        c0, _ = sb.restriction_socp(rg, pseq)
        if use_prune:
            alive, _ = sb.informed_prune(rg, c0)
    deg = {v: len(rg["region_portals"][v]) for v in range(rg["n"])}
    r = gcs_star(
        rg, timeout=timeout, K=K, max_revisits=max_revisits, seed=seed,
        alive=alive,
        incumbent=(c0 * (1 + 1e-9) if (use_warm and c0 is not None)
                   else None),
        order_prior=(lambda v: deg.get(v, 0)) if use_deg else None,
        plateau=plateau if use_deg else None,
        deg_samples=use_deg)
    r["t"] = time.perf_counter() - t0
    r["c0"] = float(c0) if c0 is not None else None
    r["n_alive"] = len(alive) if alive is not None else rg["n"]
    if r["path"] == "warm":
        r["cost"] = float(c0)
        r["path"] = None
    return r
