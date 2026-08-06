#!/usr/bin/env python3
"""E10-A: can OUR pipeline solve the 18 task queries whose GCS*
reference answers are physically colliding?

For each violated pair: HighDPlanner (interface sampling + informed +
lazy physical validation + in-region repair) -> corridor SOCP polish ->
re-certification.  Records certified cost vs the (colliding) reference
cost and saves the paths for the comparison video.

    python3 e10_ours_on_tasks.py
"""
from __future__ import annotations

import json
import pickle
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.environ.get("GCS_SR",
                            os.path.expanduser("~/gcs-science-robotics")))

import e6_sampling as e6
import stageB_bimanual_ipgcs as sb
from e7_highd_audit import (HighDPlanner, bbox_ell_keys, build_path_audit,
                            polish_waypoints)
from e8_validate import path_cost
from reproduction.bimanual.helpers import getConfigurationSeeds


def main():
    seeds = {k: np.asarray(v) for k, v in getConfigurationSeeds().items()}
    topo = pickle.load(open("out/bimanual219_topo.pkl", "rb"))
    if isinstance(topo, dict) and "topo" in topo:
        topo = topo["topo"]
    ifaces = pickle.load(open("out/e7_ifaces_biman219.pkl", "rb"))
    regions = topo["regions"]
    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}
    verdicts = json.load(open("out/e9_task_sweep_verdicts.json"))
    bad = [r for r in verdicts if r.get("verdict") == "collision"]
    audit_fn, checker, validator = build_path_audit("bimanual")
    print(f"{len(bad)} colliding task pairs to attempt", flush=True)

    out, paths = [], {}
    for r in bad:
        s, g = seeds[r["a"]], seeds[r["b"]]
        ell = bbox_ell_keys(topo, ifaces, s, g)
        t0 = time.perf_counter()
        pl = HighDPlanner(topo, ifaces, s, g, mode="area", M=1,
                          informed=True, seed=0, ell=ell,
                          path_audit=audit_fn,
                          validator=validator).run(8)
        path = pl.best_path
        # polish + re-certify (use polished only if it stays clean)
        if path is not None:
            mems = [e6.region_membership(p, regions) for p in path]
            pseq = e6.corridor_from_path(mems, pair_eid)
            if pseq is not None:
                rgq = sb.with_query(topo, s, g)
                _, pp = polish_waypoints(rgq, pseq, s, g)
                if pp is not None:
                    okp, _ = validator.path(pp)
                    if okp and path_cost(pp) < path_cost(path):
                        path = pp
        t = time.perf_counter() - t0
        if path is None:
            print(f"{r['a']} -> {r['b']}: NO certified path "
                  f"({len(pl.rejections)} rejections) [{t:.0f}s]",
                  flush=True)
            out.append({**r, "ours": None, "t": t,
                        "rejections": len(pl.rejections)})
            continue
        ok, info = validator.path(path)
        c = path_cost(path)
        excess = c / r["cost"] - 1
        print(f"{r['a']} -> {r['b']}: CERTIFIED {c:.3f} "
              f"(ref colliding {r['cost']:.3f}, +{100*excess:.1f}%) "
              f"[{t:.0f}s, rep {pl.n_repaired}]", flush=True)
        out.append({**r, "ours": float(c), "excess_vs_ref": excess,
                    "t": t, "certified": bool(ok),
                    "n_repaired": pl.n_repaired})
        paths[f"{r['a']}|{r['b']}"] = np.asarray(path)

    n_ok = sum(1 for o in out if o.get("ours"))
    print(f"\nOURS: certified clean answers on {n_ok}/{len(bad)} "
          f"colliding-reference task pairs")
    json.dump(out, open("out/e10_ours_on_tasks.json", "w"), indent=1)
    np.savez("out/e10_ours_task_paths.npz",
             **{k: v for k, v in paths.items()})
    print("written out/e10_ours_on_tasks.json + paths npz", flush=True)


if __name__ == "__main__":
    main()
