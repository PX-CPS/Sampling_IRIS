#!/usr/bin/env python3
"""E1 prototype: sampled interface roadmap over an IRIS region library.

Nodes = portal Chebyshev centers + hit-and-run samples inside portal
polytopes (+ optional region-interior samples). Edges connect nodes that
share a region, so every edge is collision-free by convexity (zero motion
validation). Dijkstra gives an instantly-valid upper bound; the corridor's
restriction SOCP polishes it. Compared per query against the stored
cluster results (GCS*, iP-GCS) on the same library and queries.

Arms:
  centers   K=0 (portal Chebyshev centers only)  ~= current iP cold start
  iface     centers + K samples per portal
  iface+int iface + M interior samples per region
  volume    same TOTAL node budget as iface+int, all uniform in regions
            (control for the interface-biased-sampling hypothesis)

    python3 e1_interface_roadmap.py --lib out/panda_regions.pkl \
        --bench out/bench_panda1raw55.jsonl --out out/e1_roadmap_panda55.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ipgcs_core as core
import stageB_bimanual_ipgcs as sb


def hit_and_run(A, b, x0, n_samples, rng, burn=20, thin=10):
    """Uniform-ish samples in {x: Ax<=b} starting from interior point x0."""
    x = np.asarray(x0, float).copy()
    out = []
    total = burn + n_samples * thin
    d = len(x)
    for it in range(total):
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        Au = A @ u
        s = b - A @ x
        with np.errstate(divide="ignore"):
            t = s / Au
        t_hi = np.min(t[Au > 1e-12]) if np.any(Au > 1e-12) else 1e6
        t_lo = np.max(t[Au < -1e-12]) if np.any(Au < -1e-12) else -1e6
        if t_hi <= t_lo:
            continue
        x = x + rng.uniform(t_lo, t_hi) * u
        if it >= burn and (it - burn) % thin == thin - 1:
            out.append(x.copy())
    return out


def build_roadmap(topo, portal_K, interior_M, volume_only=0, seed=0):
    """Return dict with node positions, per-node portal id / region sets,
    and deduped intra-region edge arrays."""
    rng = np.random.default_rng(seed)
    regions = topo["regions"]
    n = topo["n"]
    pos, node_portal, node_regions = [], [], []

    def membership(x):
        return frozenset(
            r for r, (A, b) in enumerate(regions)
            if np.all(A @ x <= b + 1e-9))

    t0 = time.perf_counter()
    if not volume_only:
        for eid, p in topo["portals"].items():
            pts = [p["center"]]
            if portal_K:
                pts += hit_and_run(p["A"], p["b"], p["center"], portal_K, rng)
            for x in pts:
                node_portal.append(eid)
                node_regions.append(membership(x))
                pos.append(x)
        if interior_M:
            for r, (A, b) in enumerate(regions):
                c, _ = sb.chebyshev_center(A, b)
                if c is None:
                    continue
                for x in hit_and_run(A, b, c, interior_M, rng):
                    node_portal.append(None)
                    node_regions.append(membership(x))
                    pos.append(x)
    else:
        per_region = int(np.ceil(volume_only / n))
        for r, (A, b) in enumerate(regions):
            c, _ = sb.chebyshev_center(A, b)
            if c is None:
                continue
            for x in hit_and_run(A, b, c, per_region, rng):
                node_portal.append(None)
                node_regions.append(membership(x))
                pos.append(x)
    pos = np.asarray(pos)
    t_sample = time.perf_counter() - t0

    t0 = time.perf_counter()
    region_nodes = {r: [] for r in range(n)}
    for i, regs in enumerate(node_regions):
        for r in regs:
            region_nodes[r].append(i)
    ii, jj = [], []
    for r, nodes in region_nodes.items():
        if len(nodes) < 2:
            continue
        a = np.asarray(nodes)
        iu, ju = np.triu_indices(len(a), k=1)
        ii.append(a[iu])
        jj.append(a[ju])
    if ii:
        ii = np.concatenate(ii)
        jj = np.concatenate(jj)
        keys = np.unique(ii.astype(np.int64) * len(pos) + jj)
        ii = (keys // len(pos)).astype(np.int32)
        jj = (keys % len(pos)).astype(np.int32)
        ww = np.linalg.norm(pos[ii] - pos[jj], axis=1)
    else:
        ii = jj = np.zeros(0, np.int32)
        ww = np.zeros(0)
    t_edges = time.perf_counter() - t0
    return {"pos": pos, "node_portal": node_portal,
            "node_regions": node_regions, "region_nodes": region_nodes,
            "ii": ii, "jj": jj, "ww": ww,
            "t_sample": t_sample, "t_edges": t_edges}


def corridor_from_path(path_mems, pair_eid):
    """Portal sequence via a greedy region walk over the path's node
    memberships. Guarantees consecutive portals share the transit region,
    so the restriction SOCP chain is a valid path (tag-based extraction is
    unsound when an edge connects two portal nodes through a third region,
    and yields an empty — hence straight-line, invalid — corridor for
    interior-only paths)."""
    edge_regs = [path_mems[k] & path_mems[k + 1]
                 for k in range(len(path_mems) - 1)]
    if any(not er for er in edge_regs):
        return None
    pseq, cur = [], None
    for k, er in enumerate(edge_regs):
        if cur is not None and cur in er:
            continue
        best_r, best_len = None, -1
        for r in er:
            L = 0
            while k + L < len(edge_regs) and r in edge_regs[k + L]:
                L += 1
            if L > best_len:
                best_r, best_len = r, L
        if cur is not None:
            eid = pair_eid.get(frozenset((cur, best_r)))
            if eid is None:
                return None
            pseq.append(eid)
        cur = best_r
    return pseq


def roadmap_query(rm, rg, pair_eid):
    """Dijkstra s->g on the roadmap. Returns (ub_path, corridor pseq, t)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    t0 = time.perf_counter()
    pos, N = rm["pos"], len(rm["pos"])
    s_idx, g_idx = N, N + 1
    qi, qj = [], []
    for r in rg["start_regions"]:
        for k in rm["region_nodes"][r]:
            qi.append(s_idx)
            qj.append(k)
    for r in rg["goal_regions"]:
        for k in rm["region_nodes"][r]:
            qi.append(g_idx)
            qj.append(k)
    if set(rg["start_regions"]) & set(rg["goal_regions"]):
        qi.append(s_idx)
        qj.append(g_idx)
    if not qi:
        return None, None, time.perf_counter() - t0
    qi = np.asarray(qi)
    qj = np.asarray(qj)
    qpos = np.vstack([pos, rg["start"], rg["goal"]])
    qw = np.linalg.norm(qpos[qi] - qpos[qj], axis=1)
    ii = np.concatenate([rm["ii"], qi])
    jj = np.concatenate([rm["jj"], qj])
    ww = np.concatenate([rm["ww"], qw])
    m = coo_matrix((np.concatenate([ww, ww]),
                    (np.concatenate([ii, jj]), np.concatenate([jj, ii]))),
                   shape=(N + 2, N + 2)).tocsr()
    dist, pred = dijkstra(m, indices=s_idx, return_predecessors=True)
    if not np.isfinite(dist[g_idx]):
        return None, None, time.perf_counter() - t0
    path = [g_idx]
    while path[-1] != s_idx:
        path.append(pred[path[-1]])
    path = path[::-1]
    path_mems = []
    for k in path:
        if k == s_idx:
            path_mems.append(frozenset(rg["start_regions"]))
        elif k == g_idx:
            path_mems.append(frozenset(rg["goal_regions"]))
        else:
            path_mems.append(rm["node_regions"][k])
    pseq = corridor_from_path(path_mems, pair_eid)
    return float(dist[g_idx]), pseq, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="out/panda_regions.pkl")
    ap.add_argument("--bench", default="out/bench_panda1raw55.jsonl")
    ap.add_argument("--topo-cache", default="out/e1_topo_panda55.pkl")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/e1_roadmap_panda55.jsonl")
    args = ap.parse_args()

    data = pickle.load(open(args.lib, "rb"))
    reg_dict = data["regions"] if isinstance(data, dict) and \
        "regions" in data else data
    regions = [(np.asarray(A), np.asarray(b)) for A, b in reg_dict.values()]

    if isinstance(data, dict) and "topo" in data:
        # wall-aware topologies (UAV/maze) ship with the library; rebuilding
        # from raw intersections would resurrect fake through-wall portals
        topo = data["topo"]
        print(f"topology from library: {len(topo['portals'])} portals")
    elif os.path.exists(args.topo_cache):
        topo = pickle.load(open(args.topo_cache, "rb"))
        print(f"topology from cache: {len(topo['portals'])} portals")
    else:
        t0 = time.perf_counter()
        topo = sb.build_topology(regions)
        print(f"topology: {len(topo['portals'])} portals "
              f"in {time.perf_counter()-t0:.1f}s", flush=True)
        pickle.dump(topo, open(args.topo_cache, "wb"))

    bench = [json.loads(l) for l in open(args.bench)]
    print(f"{len(bench)} bench queries loaded", flush=True)

    n_port = len(topo["portals"])
    budget_iface_int = n_port * (1 + args.K) + topo["n"] * args.M
    arms = {
        "centers": dict(portal_K=0, interior_M=0),
        "iface": dict(portal_K=args.K, interior_M=0),
        "iface_int": dict(portal_K=args.K, interior_M=args.M),
        "volume": dict(portal_K=0, interior_M=0,
                       volume_only=budget_iface_int),
    }
    roadmaps = {}
    for name, kw in arms.items():
        rm = build_roadmap(topo, seed=args.seed, **kw)
        roadmaps[name] = rm
        print(f"[{name}] {len(rm['pos'])} nodes, {len(rm['ww'])} edges, "
              f"sample {rm['t_sample']:.1f}s edges {rm['t_edges']:.1f}s",
              flush=True)

    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}

    fout = open(args.out, "w")
    for rec in bench:
        s = np.asarray(rec["s"])
        g = np.asarray(rec["g"])
        rg = sb.with_query(topo, s, g)
        best = rec.get("best_known")
        row = {"i": rec["i"], "best_known": best,
               "gcsstar_t": rec.get("gcsstar", {}).get("t"),
               "gcsstar_cost": rec.get("gcsstar", {}).get("cost"),
               "ipgcs_t": rec.get("ipgcs", {}).get("t"),
               "ipgcs_cost": rec.get("ipgcs", {}).get("cost"),
               "ipgcs_prune": rec.get("ipgcs", {}).get("prune")}

        # current cold start reproduced locally (center A* + SOCP)
        t0 = time.perf_counter()
        wg = sb.build_waypoint_graph(rg)
        path, _, _, _ = core.astar(wg)
        if path is not None:
            pseq0 = [wg.node_portal[i] for i in path if i in wg.node_portal]
            c0_old, _ = sb.restriction_socp(rg, pseq0)
        else:
            c0_old = np.inf
        t_old = time.perf_counter() - t0
        alive_old, t_pr_old = sb.informed_prune(rg, c0_old)
        row["old"] = {"c0": c0_old, "t": t_old,
                      "alive": len(alive_old), "t_prune": t_pr_old}

        # oracle prune: ceiling on what ANY tighter UB could buy
        alive_orc, t_orc = sb.informed_prune(rg, best * (1 + 1e-9))
        row["oracle"] = {"alive": len(alive_orc), "t_prune": t_orc}

        for name, rm in roadmaps.items():
            ub_path, pseq, t_q = roadmap_query(rm, rg, pair_eid)
            if ub_path is None:
                row[name] = {"status": "disconnected", "t": t_q}
                continue
            if pseq is not None:
                c0_new, t_socp = sb.restriction_socp(rg, pseq)
                c0 = min(ub_path, c0_new)
            else:
                c0, t_socp = ub_path, 0.0
            alive, t_pr = sb.informed_prune(rg, c0)
            row[name] = {
                "ub_path": ub_path, "c0": c0, "t_query": t_q,
                "t_socp": t_socp, "t_prune": t_pr,
                "corridor_len": len(pseq) if pseq is not None else -1,
                "alive": len(alive),
                "excess_vs_best": (c0 / best - 1) if best else None}
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        msg = " ".join(
            f"{nm}:{row[nm].get('c0', float('nan')):.3f}"
            f"/p{row[nm].get('ub_path', float('nan')):.3f}"
            f"(a{row[nm].get('alive', -1)})"
            for nm in arms if nm in row and "c0" in row[nm])
        print(f"[q{rec['i']}] best={best:.3f} oracle_a{len(alive_orc)} "
              f"old:{c0_old:.3f}(a{len(alive_old)}) {msg}", flush=True)
    fout.close()

    # summary table
    rows = [json.loads(l) for l in open(args.out)]
    print("\n=== per-arm medians over", len(rows), "queries ===")
    print(f"{'arm':10s} {'excess%':>8s} {'alive':>6s} {'t_query':>8s} "
          f"{'t_socp':>7s} {'t_prune':>8s}")
    print(f"{'oracle':10s} {'0.00':>8s} "
          f"{np.median([r['oracle']['alive'] for r in rows]):6.0f} "
          f"{'-':>8s} {'-':>7s} "
          f"{np.median([r['oracle']['t_prune'] for r in rows]):8.2f}"
          "   <- prune ceiling with the true optimum as UB")
    ex_old = [r["old"]["c0"] / r["best_known"] - 1 for r in rows
              if r.get("best_known")]
    al_old = [r["old"]["alive"] for r in rows]
    print(f"{'old':10s} {100*np.median(ex_old):8.2f} "
          f"{np.median(al_old):6.0f} {'-':>8s} {'-':>7s} "
          f"{np.median([r['old']['t_prune'] for r in rows]):8.2f}")
    for nm in ("centers", "iface", "iface_int", "volume"):
        ok = [r for r in rows if nm in r and "c0" in r[nm]]
        if not ok:
            print(f"{nm:10s}  (all disconnected)")
            continue
        ex = [r[nm]["excess_vs_best"] for r in ok
              if r[nm]["excess_vs_best"] is not None]
        print(f"{nm:10s} {100*np.median(ex):8.2f} "
              f"{np.median([r[nm]['alive'] for r in ok]):6.0f} "
              f"{np.median([r[nm]['t_query'] for r in ok]):8.3f} "
              f"{np.median([r[nm]['t_socp'] for r in ok]):7.2f} "
              f"{np.median([r[nm]['t_prune'] for r in ok]):8.2f}")
    gt = [r["gcsstar_t"] for r in rows if r.get("gcsstar_t")]
    print(f"\nGCS* median t on these queries: {np.median(gt):.1f}s "
          f"(cost = best_known on all optimal rows)")


if __name__ == "__main__":
    main()
