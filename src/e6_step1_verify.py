#!/usr/bin/env python3
"""E6 step 1: correctness verification of interface sampling roadmaps.

For each sampler (center / contour / area) and each per-interface budget
M, build the roadmap, run every audit, solve the query, and compare the
answer against ground truth obtained two independent ways:

  * the exact GCS convex relaxation + rounding over the SAME regions
    (global lower bound LB and upper bound UB), and
  * the raw obstacle polygons (2D testbed), which never passed through
    the planner.

Soundness criterion (must hold on every row): roadmap cost >= LB - tol,
zero constraint violations beyond solver tolerance, zero probe points in
obstacles, zero edges without a witness region.

    python3 e6_step1_verify.py --lib out/e6_iris2d.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6_sampling as e6
import stageB_bimanual_ipgcs as sb
from e6_build_iris2d import penetration_depth

PEN_TOL = 1e-9          # boundary contact vs real penetration


def no_obstacles(p, obstacles):
    return 0.0


def ground_truth(topo, s, g):
    rg = sb.with_query(topo, s, g)
    out = sb.solve_gcs(rg)
    return {"lb": out["lb"], "ub": out["ub"], "t": out["t"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="out/e6_iris2d.pkl")
    ap.add_argument("--modes", default="center,contour,area")
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--membership", default="declared",
                    choices=["declared", "geometric"])
    ap.add_argument("--walls", action="store_true",
                    help="reconstruct zero-thickness walls (box libraries) "
                         "and audit every edge/path against them")
    ap.add_argument("--out", default="out/e6_step1_verify.json")
    args = ap.parse_args()

    lib = pickle.load(open(args.lib, "rb"))
    topo = lib["topo"]
    regions = topo["regions"]
    obstacles = lib.get("obstacles", [])
    penetration = penetration_depth if obstacles else no_obstacles
    if "task_configs" in lib:
        tc = lib["task_configs"]
        s, g = np.asarray(tc["start"]), np.asarray(tc["goal"])
    else:                                    # maze: corner to corner
        s = np.asarray(topo["bbox_lo"][0]) + 0.5
        s[:] = [0.5, 0.5]
        g = np.asarray(topo["bbox_hi"][-1]) - 0.5

    ifaces = e6.classify_interfaces(topo)
    kinds = {}
    for f in ifaces.values():
        kinds[(f["kind"], f["dim"])] = kinds.get((f["kind"], f["dim"]), 0) + 1
    print(f"library {args.lib}: {len(regions)} regions, "
          f"{len(ifaces)} interfaces")
    for k, c in sorted(kinds.items()):
        print(f"  {k[0]:8s} dim={k[1]}  x{c}")

    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}
    gt = ground_truth(topo, s, g)
    print(f"ground truth (full GCS on same regions): LB={gt['lb']:.6f} "
          f"UB={gt['ub']:.6f} in {gt['t']:.2f}s", flush=True)
    walls = e6.box_walls(topo) if args.walls else []
    if args.walls:
        print(f"reconstructed {len(walls)} zero-thickness walls "
              f"(adjacent boxes with no declared portal)", flush=True)
    lb_tol = 1e-6 * max(1.0, abs(gt["lb"]))

    rows = []
    print(f"\n{'mode':8s} {'M':>3s} {'nodes':>6s} {'edges':>8s} "
          f"{'cost':>9s} {'polished':>9s} {'vs UB':>8s} {'>=LB':>5s} "
          f"{'maxViol':>9s} {'penSmp':>9s} {'noWit':>6s} {'penEdge':>9s} "
          f"{'penPath':>9s} {'pathOut':>7s} {'wall':>7s} {'badJnc':>8s}")
    for mode in args.modes.split(","):
        for M in [int(x) for x in args.budgets.split(",")]:
            rm = e6.build_roadmap(topo, ifaces, mode, M, seed=args.seed,
                                  membership=args.membership)
            a_s = e6.audit_samples(rm, ifaces, regions, obstacles, penetration)
            a_e = e6.audit_edges(rm, regions, obstacles, penetration,
                                 seed=args.seed)
            cost, path = e6.shortest_path(rm, regions, s, g)
            a_p = e6.audit_path(path, regions, obstacles, penetration)
            # corridor polish: the exact optimum WITHIN the chosen corridor
            c_pol = np.inf
            if path is not None:
                mems = [e6.region_membership(p, regions) for p in path]
                pseq = e6.corridor_from_path(mems, pair_eid)
                if pseq is not None:
                    rgq = sb.with_query(topo, s, g)
                    c_pol, _ = sb.restriction_socp(rgq, pseq)
            a_w = e6.audit_wall_crossings(rm, walls, path, seed=args.seed)
            a_j = e6.audit_junctions(path, regions)
            a_w = {**a_w, **a_j}
            sound = bool(np.isfinite(cost) and cost >= gt["lb"] - lb_tol
                         and (not np.isfinite(c_pol)
                              or c_pol >= gt["lb"] - lb_tol))
            row = {"mode": mode, "M": M, "nodes": int(len(rm["pos"])),
                   "edges": a_e["n_edges"], "cost": float(cost),
                   "cost_polished": float(c_pol),
                   "excess_vs_ub": float(cost / gt["ub"] - 1),
                   "excess_polished": float(c_pol / gt["ub"] - 1),
                   "sound_ge_lb": sound,
                   "t_sample": rm["t_sample"], "t_member": rm["t_member"],
                   **{f"aud_{k}": v for k, v in
                      {**a_s, **a_e, **a_p, **a_w}.items()}}
            rows.append(row)
            viol = max(a_s["max_interface_violation"],
                       a_s["max_parent_violation"],
                       a_e["max_probe_region_violation"])
            print(f"{mode:8s} {M:3d} {row['nodes']:6d} {row['edges']:8d} "
                  f"{cost:9.4f} {c_pol:9.4f} {100*row['excess_vs_ub']:7.2f}% "
                  f"{'OK' if sound else 'FAIL':>5s} "
                  f"{viol:9.1e} {a_s['max_sample_penetration']:9.1e} "
                  f"{a_e['edges_without_witness_region']:6d} "
                  f"{a_e['max_probe_penetration']:9.1e} "
                  f"{a_p.get('max_path_penetration', -1):9.1e} "
                  f"{a_p.get('probe_outside_union', -1):8d} "
                  f"{a_w['edge_wall_crossings']:5d}/"
                  f"{a_w['path_wall_crossings']:d} "
                  f"{a_w['bad_junctions']:4d}/{a_w['junctions']:d}",
                  flush=True)

    ok = all(r["sound_ge_lb"]
             and r["aud_max_sample_penetration"] <= PEN_TOL
             and r["aud_edges_without_witness_region"] == 0
             and r["aud_max_probe_penetration"] <= PEN_TOL
             and r["aud_max_path_penetration"] <= PEN_TOL
             and r["aud_max_interface_violation"] <= 1e-9
             and r["aud_max_parent_violation"] <= 1e-9
             and r["aud_max_probe_region_violation"] <= 1e-9
             and r["aud_edge_wall_crossings"] == 0
             and r["aud_path_wall_crossings"] == 0
             and r["aud_bad_junctions"] == 0
             and r["aud_probe_outside_union"] == 0 for r in rows)
    print(f"\nALL AUDITS PASS: {ok}")
    json.dump({"lib": args.lib, "ground_truth": gt, "rows": rows,
               "all_audits_pass": ok,
               "interface_kinds": {f"{k[0]}_dim{k[1]}": c
                                   for k, c in kinds.items()}},
              open(args.out, "w"), indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
