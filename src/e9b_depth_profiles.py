#!/usr/bin/env python3
"""E9-B: full min-clearance depth profiles for the 18 colliding
task-pair reference answers.

The task sweep recorded only the FIRST-crossing depth (0-1 mm, shallow
entry); the one pair profiled during video work reached -41.2 mm.  This
sweep re-derives each reference answer (same solve as the sweep),
checks the cost reproduces, and walks the whole path with a dense
clearance scan: worst penetration, where it happens, in-collision
fraction, contiguous dirty intervals, colliding geometry at the worst
point, and an endpoint sanity check (named grasp configs should be
clean, unlike q8's sampled start).  Reference waypoints are persisted
this time (the sweep never saved them).

    python3 e9b_depth_profiles.py   # out/e9b_depth_profiles.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for p in (os.environ.get("GCS_SR", ""),
          os.path.expanduser("~/gcs-science-robotics")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import baseline_gcsstar as bgs
import stageB_bimanual_ipgcs as sb
from e7_highd_audit import build_path_audit
from e9_verify_gcsstar import chain_waypoints, colliding_pairs
from reproduction.bimanual.helpers import getConfigurationSeeds

N_SCAN = 4000
INFLUENCE = 0.10


def clearance(checker, q):
    rc = checker.CalcRobotClearance(q, INFLUENCE)
    d = np.asarray(rc.distances())
    return float(d.min()) if d.size else INFLUENCE


def profile(checker, path):
    p = np.asarray(path)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    ts = np.linspace(0, L[-1], N_SCAN)
    phi = np.empty(N_SCAN)
    qs = np.empty((N_SCAN, p.shape[1]))
    for j, t in enumerate(ts):
        k = min(max(int(np.searchsorted(L, t, "right") - 1), 0),
                len(seg) - 1)
        lam = 0.0 if seg[k] < 1e-12 else (t - L[k]) / seg[k]
        qs[j] = (1 - lam) * p[k] + lam * p[k + 1]
        phi[j] = clearance(checker, qs[j])
    bad = phi < 0
    j0 = int(np.argmin(phi))
    # contiguous dirty intervals as arc-length fractions
    ivals, j = [], 0
    while j < N_SCAN:
        if bad[j]:
            j1 = j
            while j1 + 1 < N_SCAN and bad[j1 + 1]:
                j1 += 1
            ivals.append((j / (N_SCAN - 1), j1 / (N_SCAN - 1)))
            j = j1 + 1
        else:
            j += 1
    pairs, worst = colliding_pairs(checker, qs[j0], INFLUENCE)
    return {"worst_mm": 1000 * float(phi[j0]),
            "worst_at_fraction": j0 / (N_SCAN - 1),
            "in_collision_fraction": float(bad.mean()),
            "n_dirty_intervals": len(ivals),
            "dirty_intervals": [(round(a, 4), round(b, 4))
                                for a, b in ivals],
            "worst_pairs": pairs,
            "phi_start_mm": 1000 * float(phi[0]),
            "phi_goal_mm": 1000 * float(phi[-1])}


def main():
    seeds = {k: np.asarray(v) for k, v in getConfigurationSeeds().items()}
    topo = pickle.load(open("out/bimanual219_topo.pkl", "rb"))
    if isinstance(topo, dict) and "topo" in topo:
        topo = topo["topo"]
    verdicts = json.load(open("out/e9_task_sweep_verdicts.json"))
    bad = [r for r in verdicts if r.get("verdict") == "collision"]
    _, checker, _ = build_path_audit("bimanual")
    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}
    print(f"{len(bad)} colliding task pairs to profile", flush=True)

    out, paths = [], {}
    for r in bad:
        s, g = seeds[r["a"]], seeds[r["b"]]
        t0 = time.perf_counter()
        # the sweep's reference recipe (as in e9_task_violation_video)
        rg = sb.with_query(topo, s, g)
        res = bgs.gcs_star(rg, timeout=60.0, K=1, max_revisits=1, seed=0)
        if res is None or res.get("path") is None:
            print(f"{r['a']} -> {r['b']}: NO PATH on re-solve", flush=True)
            out.append({**r, "repro": "no_path"})
            continue
        chain = [(topo["portals"][pair_eid[frozenset((u, v))]]["A"],
                  topo["portals"][pair_eid[frozenset((u, v))]]["b"])
                 for u, v in zip(res["path"], res["path"][1:])]
        c, wp = chain_waypoints(chain, s, g)
        if wp is None:
            print(f"{r['a']} -> {r['b']}: chain infeasible", flush=True)
            out.append({**r, "repro": "chain_infeasible"})
            continue
        dc = abs(c - r["cost"])
        prof = profile(checker, wp)
        t = time.perf_counter() - t0
        out.append({**r, "repro_cost": c, "d_cost": dc,
                    "repro": "exact" if dc < 1e-6 else
                    ("close" if dc < 1e-3 else "DRIFT"), **prof, "t": t})
        paths[f"{r['a']}|{r['b']}"] = wp
        print(f"{r['a']} -> {r['b']}: cost {c:.4f} (d={dc:.1e}) "
              f"worst {prof['worst_mm']:+.1f}mm at "
              f"{100*prof['worst_at_fraction']:.0f}% "
              f"({prof['in_collision_fraction']*100:.1f}% dirty, "
              f"{prof['n_dirty_intervals']} intervals) "
              f"ends {prof['phi_start_mm']:+.0f}/"
              f"{prof['phi_goal_mm']:+.0f}mm [{t:.0f}s]", flush=True)

    ws = [o["worst_mm"] for o in out if "worst_mm" in o]
    print(f"\nprofiled {len(ws)}/{len(bad)}; worst overall "
          f"{min(ws):+.1f}mm; median {np.median(ws):+.1f}mm", flush=True)
    json.dump(out, open("out/e9b_depth_profiles.json", "w"), indent=1)
    np.savez("out/e9b_ref_task_paths.npz", **paths)
    print("written out/e9b_depth_profiles.json + ref paths npz",
          flush=True)


if __name__ == "__main__":
    main()
