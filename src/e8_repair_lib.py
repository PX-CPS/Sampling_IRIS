#!/usr/bin/env python3
"""E8 part 3: pocket excision -- offline library repair.

Per region: sample configurations inside the polytope; every colliding
sample is bisected toward the region's interior point to the free/
colliding boundary, and a halfspace cut through that boundary point
removes the pocket cap (the region stays convex, one H-row per cut).
Repeat until M consecutive samples are clean, which certifies
"colliding volume fraction <= eps with confidence conf" by the standard
Bernoulli argument (M = ln(1/(1-conf))/eps) -- the same test IrisZo
uses, continued to a stricter target.  Then rebuild the topology on the
repaired regions and re-audit interface contamination.

C-space analog of the UAV workspace repair (uav_export --repair).

    python3 e8_repair_lib.py --preset biman219
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

SCENES = {
    "panda55": ("out/panda_regions.pkl", "panda",
                "out/panda_regionsR.pkl", "out/panda55R_topo.pkl"),
    "biman219": ("out/bimanual_regions_scale.pkl", "bimanual",
                 "out/bimanual_regions_scaleR.pkl",
                 "out/bimanual219R_topo.pkl"),
}


def build_checker(scene):
    if scene == "panda":
        from panda_scene import build_scene
        return build_scene()[0]
    from stageB_bimanual_regions import build
    return build()[0]


def bisect_to_boundary(checker, q_bad, q_good, iters=16):
    """Boundary point between a colliding and a free configuration."""
    a, b = np.asarray(q_bad, float), np.asarray(q_good, float)
    for _ in range(iters):
        m = 0.5 * (a + b)
        if checker.CheckConfigCollisionFree(m):
            b = m
        else:
            a = m
    return b                                # free side of the boundary


def repair_region(checker, A, b, eps, conf, max_cuts, rng, batch=64):
    """Excise collision pockets; returns (A', b', report)."""
    A = np.asarray(A, float).copy()
    b = np.asarray(b, float).copy()
    M_clean = int(np.ceil(np.log(1.0 / (1.0 - conf)) / eps))
    streak, cuts, n_samples, n_bad = 0, 0, 0, 0
    iface = None

    def rebuild():
        nonlocal iface
        h = e6.reduce_to_affine_hull(A, b)
        iface = {"A": A, "b": b, "hull": h, "dim": h["dim"] if h else 0,
                 "u": 0, "v": 0, "kind": "region"}
        return h is not None and h["dim"] > 0

    if not rebuild():
        return None, None, {"status": "degenerate", "cuts": 0}
    ctr = iface["hull"]["yc"] if iface["hull"]["dim"] == A.shape[1] else None
    ctr = ctr if ctr is not None else np.zeros(A.shape[1])
    if not checker.CheckConfigCollisionFree(ctr):
        # interior anchor itself dirty: look for a clean anchor first
        anchor = None
        for x in e6.sample_area(iface, 64, rng):
            n_samples += 1
            if checker.CheckConfigCollisionFree(np.asarray(x)):
                anchor = np.asarray(x)
                break
        if anchor is None:
            return None, None, {"status": "irreparable", "cuts": 0,
                                "samples": n_samples}
        ctr = anchor

    while streak < M_clean and cuts < max_cuts:
        for x in e6.sample_area(iface, batch, rng):
            x = np.asarray(x)
            n_samples += 1
            if checker.CheckConfigCollisionFree(x):
                streak += 1
                if streak >= M_clean:
                    break
                continue
            # excise: cut through the boundary point toward the anchor
            n_bad += 1
            streak = 0
            qb = bisect_to_boundary(checker, x, ctr)
            a_dir = x - ctr
            nrm = np.linalg.norm(a_dir)
            if nrm < 1e-12:
                continue
            a_dir /= nrm
            A = np.vstack([A, a_dir])
            b = np.concatenate([b, [float(a_dir @ qb)]])
            cuts += 1
            if not rebuild():
                return None, None, {"status": "vanished", "cuts": cuts,
                                    "samples": n_samples}
            if cuts >= max_cuts:
                break
    status = "clean" if streak >= M_clean else "cut_budget"
    return A, b, {"status": status, "cuts": cuts, "samples": n_samples,
                  "bad_found": n_bad, "M_clean": M_clean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(SCENES))
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--conf", type=float, default=0.99)
    ap.add_argument("--max-cuts", type=int, default=200)
    ap.add_argument("--audit-per-iface", type=int, default=3)
    args = ap.parse_args()
    lib_file, scene, out_regions, out_topo = SCENES[args.preset]

    data = pickle.load(open(lib_file, "rb"))
    reg_dict = data["regions"] if isinstance(data, dict) and \
        "regions" in data else data
    checker = build_checker(scene)
    print(f"{args.preset}: {len(reg_dict)} regions, eps={args.eps} "
          f"conf={args.conf}", flush=True)

    rng = np.random.default_rng(0)
    repaired, reports, dropped = {}, {}, []
    t0 = time.perf_counter()
    for key, (A, b) in reg_dict.items():
        A2, b2, rep = repair_region(checker, A, b, args.eps, args.conf,
                                    args.max_cuts, rng)
        reports[str(key)] = rep
        if A2 is None:
            dropped.append(key)
            continue
        repaired[key] = (A2, b2)
        if rep["cuts"]:
            print(f"  region {key}: {rep['status']}, {rep['cuts']} cuts, "
                  f"{rep['bad_found']} bad of {rep['samples']}", flush=True)
    t_rep = time.perf_counter() - t0
    n_cuts = sum(r.get("cuts", 0) for r in reports.values())
    print(f"repair: {len(repaired)}/{len(reg_dict)} regions kept, "
          f"{len(dropped)} dropped, {n_cuts} cuts total, {t_rep:.0f}s",
          flush=True)

    pickle.dump(repaired, open(out_regions, "wb"))
    regions = [(np.asarray(A), np.asarray(b))
               for A, b in repaired.values()]
    t0 = time.perf_counter()
    topo = sb.build_topology(regions)
    print(f"topology: {len(topo['portals'])} portals in "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    pickle.dump(topo, open(out_topo, "wb"))

    # interface contamination re-audit (same measure as the E7 audit)
    ifs = e6.classify_interfaces(topo)
    rng2 = np.random.default_rng(1)
    n_tot, n_bad = 0, 0
    for f in ifs.values():
        for x in e6.sample_area(f, args.audit_per_iface, rng2):
            n_tot += 1
            if not checker.CheckConfigCollisionFree(np.asarray(x)):
                n_bad += 1
    print(f"REPAIRED interface contamination: {n_bad}/{n_tot} "
          f"({100*n_bad/max(n_tot,1):.2f}%)", flush=True)

    json.dump({"preset": args.preset, "eps": args.eps, "conf": args.conf,
               "kept": len(repaired), "dropped": [str(k) for k in dropped],
               "total_cuts": n_cuts, "t_repair_s": t_rep,
               "portals": len(topo["portals"]),
               "iface_contamination": [n_bad, n_tot],
               "reports": reports},
              open(f"out/e8_repair_{args.preset}.json", "w"), indent=1)
    print(f"written out/e8_repair_{args.preset}.json", flush=True)


if __name__ == "__main__":
    main()
