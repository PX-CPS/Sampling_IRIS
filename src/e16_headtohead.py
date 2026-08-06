#!/usr/bin/env python3
"""E16: head-to-head on the 29 connected task pairs -- the closing
experiment.

Same library (bimanual219), same machine, same queries.  Two arms:

  * REFERENCE: GCS* with the sweep's exact configuration
    (timeout=60, K=1, max_revisits=1, seed=0) -> portal-chain SOCP
    argmin -> physical validation of the answer it would deliver.
  * OURS: interface sampling (area, M=1) + informed pruning + lazy
    physical validation + in-region repair -> corridor SOCP polish
    (re-validated, repaired, else pre-polish path stands) -> the
    delivered answer carries a continuous certificate.

Per pair we record: wall time (reference solve vs our first certified
incumbent and total), delivered-answer verdict (dense continuous
validation for both arms), costs, and certificate effort.  Summary:
delivered-colliding counts, honest no-paths, speed percentiles,
optimality ratio on the pairs whose reference answer is clean, and
the price of truth on the pairs where it is not.

    python3 e16_headtohead.py            # out/e16_headtohead.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for p in (os.environ.get("GCS_SR", ""),
          os.path.expanduser("~/gcs-science-robotics")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import pickle

import baseline_gcsstar as bgs
import e6_sampling as e6
import stageB_bimanual_ipgcs as sb
from e7_highd_audit import (HighDPlanner, bbox_ell_keys,
                            build_path_audit, polish_waypoints)
from e8_validate import path_cost, repair_path
from e9_verify_gcsstar import chain_waypoints, colliding_pairs
from reproduction.bimanual.helpers import getConfigurationSeeds

ROUNDS = 8


def main():
    seeds = {k: np.asarray(v) for k, v in getConfigurationSeeds().items()}
    topo = pickle.load(open("out/bimanual219_topo.pkl", "rb"))
    if isinstance(topo, dict) and "topo" in topo:
        topo = topo["topo"]
    ifaces = pickle.load(open("out/e7_ifaces_biman219.pkl", "rb"))
    regions = topo["regions"]
    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}
    sweep = json.load(open("out/e9_task_sweep.json"))
    audit_fn, checker, validator = build_path_audit("bimanual")
    print(f"{len(sweep)} connected task pairs", flush=True)

    rows = []
    for r in sweep:
        s, g = seeds[r["a"]], seeds[r["b"]]
        row = {"a": r["a"], "b": r["b"], "ref_cost_stored": r["cost"]}

        # ---------------- reference arm ----------------
        t0 = time.perf_counter()
        rg = sb.with_query(topo, s, g)
        res = bgs.gcs_star(rg, timeout=60.0, K=1, max_revisits=1,
                           seed=0)
        row["t_ref_solve"] = time.perf_counter() - t0
        if res is None or res.get("path") is None:
            row["ref"] = "no_path"
        else:
            chain = [(topo["portals"][pair_eid[frozenset((u, v))]]["A"],
                      topo["portals"][pair_eid[frozenset((u, v))]]["b"])
                     for u, v in zip(res["path"], res["path"][1:])]
            c_ref, wp = chain_waypoints(chain, s, g)
            row["t_ref_total"] = time.perf_counter() - t0
            if wp is None:
                row["ref"] = "chain_infeasible"
            else:
                row["ref_cost"] = float(c_ref)
                ok, info = validator.path(wp)
                row["ref_valid_calls"] = info.get("clearance_calls")
                if ok:
                    row["ref"] = "clean"
                else:
                    _, worst = colliding_pairs(
                        checker, wp[len(wp) // 2], 0.05)
                    row["ref"] = "colliding"
                    row["ref_bad_segments"] = len(
                        info.get("bad_segments", []))

        # ---------------- our arm ----------------
        ell = bbox_ell_keys(topo, ifaces, s, g)
        t0 = time.perf_counter()
        pl = HighDPlanner(topo, ifaces, s, g, mode="area", M=1,
                          informed=True, seed=0, ell=ell,
                          path_audit=audit_fn,
                          validator=validator).run(ROUNDS)
        t_sample = time.perf_counter() - t0
        ev = pl.events
        row["t_ours_first"] = float(ev[0][0]) if ev else None
        row["c_ours_first"] = float(ev[0][1]) if ev else None
        path = pl.best_path
        # canonical polish block: re-validate, repair, else pre-polish
        t0p = time.perf_counter()
        if path is not None:
            mems = [e6.region_membership(p, regions) for p in path]
            pseq = e6.corridor_from_path(mems, pair_eid)
            if pseq is not None:
                rgq = sb.with_query(topo, s, g)
                _, ppath = polish_waypoints(rgq, pseq, s, g)
                if ppath is not None:
                    okp, _ = validator.path(ppath)
                    if not okp:
                        ppath, okp, _ = repair_path(
                            ppath, validator, regions,
                            pl._region_hulls,
                            np.random.default_rng(7))
                    if okp and path_cost(ppath) < path_cost(path):
                        path = ppath
        row["t_ours_polish"] = time.perf_counter() - t0p
        row["t_ours_total"] = t_sample + row["t_ours_polish"]
        if path is None:
            row["ours"] = "no_certified_path"
            row["ours_rejections"] = len(pl.rejections)
        else:
            ok, info = validator.path(path)
            row["ours"] = "certified" if ok else "VALIDATION_FAIL"
            row["ours_cost"] = float(path_cost(path))
            row["ours_cert_calls"] = info.get("clearance_calls")
        rows.append(row)
        print(f'{r["a"]:>22s}->{r["b"]:<22s} ref {row.get("ref","?"):10s}'
              f' {row.get("ref_cost", float("nan")):7.3f}'
              f' [{row["t_ref_solve"]:5.1f}s]  ours '
              f'{row.get("ours","?"):16s}'
              f' {row.get("ours_cost", float("nan")):7.3f}'
              f' [first {row.get("t_ours_first") or -1:5.2f}s'
              f' total {row["t_ours_total"]:5.1f}s]', flush=True)

    # ---------------- summary ----------------
    n = len(rows)
    ref_col = [r for r in rows if r.get("ref") == "colliding"]
    ref_clean = [r for r in rows if r.get("ref") == "clean"]
    ours_cert = [r for r in rows if r.get("ours") == "certified"]
    ours_nop = [r for r in rows if r.get("ours") == "no_certified_path"]
    ours_bad = [r for r in rows if r.get("ours") == "VALIDATION_FAIL"]
    print(f"\nREFERENCE delivered: {len(ref_clean)} clean, "
          f"{len(ref_col)} colliding ({100*len(ref_col)/n:.0f}%), "
          f"{n-len(ref_clean)-len(ref_col)} other", flush=True)
    print(f"OURS delivered: {len(ours_cert)} certified, "
          f"{len(ours_bad)} colliding, {len(ours_nop)} honest no-path",
          flush=True)
    tf = [r["t_ours_first"] for r in ours_cert
          if r.get("t_ours_first")]
    tr = [r["t_ref_solve"] for r in rows]
    print(f"speed: ref solve median {np.median(tr):.2f}s | ours "
          f"t_first median {np.median(tf):.2f}s, total median "
          f"{np.median([r['t_ours_total'] for r in rows]):.2f}s",
          flush=True)
    both_clean = [r for r in ref_clean if r.get("ours_cost")]
    if both_clean:
        ratio = [r["ours_cost"] / r["ref_cost"] for r in both_clean]
        print(f"optimality on {len(both_clean)} clean-reference pairs: "
              f"ours/ref median {np.median(ratio):.4f} "
              f"(max {max(ratio):.4f})", flush=True)
    truth = [r for r in ref_col if r.get("ours_cost")]
    if truth:
        exc = [r["ours_cost"] / r["ref_cost"] - 1 for r in truth]
        print(f"price of truth on {len(truth)} colliding-reference "
              f"pairs solved by ours: median +{100*np.median(exc):.1f}%"
              f" (range +{100*min(exc):.1f}% to +{100*max(exc):.1f}%)",
              flush=True)
    json.dump(rows, open("out/e16_headtohead.json", "w"), indent=1)
    print("written out/e16_headtohead.json", flush=True)


if __name__ == "__main__":
    main()
