#!/usr/bin/env python3
"""E8 part 5: erosion sweep -- contamination vs margin vs connectivity.

Planning-time mitigation: shrink every region by a margin delta
(b_i -> b_i - delta * ||a_i||).  Collision pockets hug region
boundaries, so small delta should kill most contamination; the price is
thinner interfaces -- portals vanish and queries disconnect as delta
grows.  This sweep measures all three curves so the trade-off (and a
usable delta*) is a figure, not a guess.

    python3 e8_erosion.py --preset biman219
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
from e8_repair_lib import SCENES, build_checker

BENCH = {"panda55": "out/bench_panda1raw55.jsonl",
         "biman219": "out/bench_219focus_K1.jsonl"}
TOPO = {"panda55": "out/e1_topo_panda55.pkl",
        "biman219": "out/bimanual219_topo.pkl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(SCENES))
    ap.add_argument("--deltas", default="0,0.002,0.005,0.01,0.02,0.05")
    ap.add_argument("--per-iface", type=int, default=3)
    args = ap.parse_args()
    lib_file, scene, _, _ = SCENES[args.preset]

    data = pickle.load(open(lib_file, "rb"))
    reg_dict = data["regions"] if isinstance(data, dict) and \
        "regions" in data else data
    regions = [(np.asarray(A), np.asarray(b)) for A, b in reg_dict.values()]
    topo = pickle.load(open(TOPO[args.preset], "rb"))
    if isinstance(topo, dict) and "topo" in topo:
        topo = topo["topo"]
    checker = build_checker(scene)
    bench = [json.loads(l) for l in open(BENCH[args.preset])]
    norms = [np.linalg.norm(A, axis=1) for A, _ in regions]
    print(f"{args.preset}: {len(regions)} regions, "
          f"{len(topo['portals'])} portals, {len(bench)} queries",
          flush=True)

    out = []
    for delta in [float(x) for x in args.deltas.split(",")]:
        t0 = time.perf_counter()
        er = [(A, b - delta * n) for (A, b), n in zip(regions, norms)]
        rng = np.random.default_rng(0)
        surv, n_tot, n_bad = [], 0, 0
        for eid, p in topo["portals"].items():
            u, v = p["u"], p["v"]
            A = np.vstack([er[u][0], er[v][0]])
            b = np.concatenate([er[u][1], er[v][1]])
            ctr, rad = e6.chebyshev(A, b)
            if ctr is None or rad < 1e-9:
                continue
            surv.append((eid, u, v))
            h = dict(x0=np.zeros(A.shape[1]), N=np.eye(A.shape[1]),
                     Ar=A, br=b, dim=A.shape[1], yc=ctr, rad=rad)
            for x in e6.sample_area({"hull": h}, args.per_iface, rng):
                n_tot += 1
                if not checker.CheckConfigCollisionFree(np.asarray(x)):
                    n_bad += 1
        # region-graph connectivity per bench query on surviving portals
        adj = {}
        for _, u, v in surv:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        n_conn = 0
        for rec in bench:
            s, g = np.asarray(rec["s"]), np.asarray(rec["g"])
            sm = [i for i, (A, b) in enumerate(er)
                  if np.all(A @ s <= b + 1e-9)]
            gm = [i for i, (A, b) in enumerate(er)
                  if np.all(A @ g <= b + 1e-9)]
            if not sm or not gm:
                continue
            seen, st = set(sm), list(sm)
            while st:
                x = st.pop()
                for y in adj.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        st.append(y)
            if seen & set(gm):
                n_conn += 1
        row = {"delta": delta, "portals_alive": len(surv),
               "portals_total": len(topo["portals"]),
               "contamination": [n_bad, n_tot],
               "queries_connected": n_conn, "queries_total": len(bench),
               "t_s": time.perf_counter() - t0}
        out.append(row)
        print(f"delta={delta:6.3f}: portals {len(surv)}/"
              f"{len(topo['portals'])}  contamination {n_bad}/{n_tot} "
              f"({100*n_bad/max(n_tot,1):.2f}%)  connected "
              f"{n_conn}/{len(bench)}  [{row['t_s']:.0f}s]", flush=True)

    json.dump({"preset": args.preset, "rows": out},
              open(f"out/e8_erosion_{args.preset}.json", "w"), indent=1)
    print(f"written out/e8_erosion_{args.preset}.json", flush=True)


if __name__ == "__main__":
    main()
