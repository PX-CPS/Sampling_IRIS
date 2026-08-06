#!/usr/bin/env python3
"""E5-B: certify-after-search on the dense 7-DOF library (fills the
panda55 certify cells of the dense-library table).

For one logged query: incumbent = GCS* cost from the paired benchmark
file; admissible prune at c_hat*(1+1e-6); subgraph relaxation LB;
record (t_prune, n_alive, t_relax, lb, gap).  One query per invocation
so an expensive relaxation cannot starve the rest -- the slurm array
task wall clock is the budget cap, and a start-stamp without a result
file marks a right-censored query.

    python3 e5_certify_panda55.py --idx 0
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
import stageB_bimanual_ipgcs as sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="out/panda_regions.pkl")
    ap.add_argument("--bench", default="out/bench_panda1raw55.jsonl")
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--outdir", default="out/e5_certify_panda55_parts")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    bench = [json.loads(l) for l in open(args.bench)]
    rec = bench[args.idx]
    c_hat = rec.get("gcsstar", {}).get("cost")
    if not c_hat or not np.isfinite(c_hat):
        print(f"q{args.idx}: no finite GCS* incumbent, skip", flush=True)
        return

    stamp = os.path.join(args.outdir, f"q{args.idx}.started")
    open(stamp, "w").write(json.dumps({"c_hat": c_hat, "t0": time.time()}))

    data = pickle.load(open(args.lib, "rb"))
    reg_dict = data["regions"] if isinstance(data, dict) and \
        "regions" in data else data
    regions = [(np.asarray(A), np.asarray(b)) for A, b in reg_dict.values()]
    t0 = time.perf_counter()
    topo = sb.build_topology(regions)
    print(f"topology {len(topo['portals'])} portals "
          f"{time.perf_counter()-t0:.1f}s", flush=True)

    rg = sb.with_query(topo, np.asarray(rec["s"]), np.asarray(rec["g"]))
    alive, t_prune = sb.informed_prune(rg, c_hat * (1 + 1e-6))
    print(f"q{args.idx}: c_hat={c_hat:.4f} alive={len(alive)}/{rg['n']} "
          f"prune {t_prune:.2f}s", flush=True)
    t0 = time.perf_counter()
    sub = sb.solve_gcs(rg, alive=alive)
    t_relax = time.perf_counter() - t0
    lb = sub["lb"]
    gap = (c_hat - lb) / c_hat if lb else None
    row = {"i": args.idx, "c_hat": c_hat, "straight": rec["straight"],
           "n_alive": len(alive), "t_prune": t_prune,
           "t_relax": t_relax, "t_cert": t_prune + t_relax,
           "lb": lb, "gap": gap,
           "gcsstar_t": rec["gcsstar"].get("t")}
    with open(os.path.join(args.outdir, f"q{args.idx}.json"), "w") as f:
        json.dump(row, f)
    print(f"q{args.idx}: cert {row['t_cert']:.1f}s gap "
          f"{100*(gap or -1):.2f}%", flush=True)


if __name__ == "__main__":
    main()
