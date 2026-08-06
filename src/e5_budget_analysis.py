#!/usr/bin/env python3
"""E5-A: budgeted-certification policy analysis from logged per-query data.

Policy: after search returns incumbent c_hat at t_search, attempt
certification (admissible prune + subgraph relaxation LB); abort at
budget B.  On success report the certified gap; on abort report the
trivial-but-sound straight-line lower bound, gap_triv = 1 - ||g-s||/c_hat.
Both outcomes are deterministic certificates; the policy converts the
unbounded certification tail into the hard bound t_search + B.

Evaluated exactly from logged (t_cert, gap, straight, c_hat) per query --
a timeout rule needs no re-execution.

    python3 e5_budget_analysis.py
"""
from __future__ import annotations

import glob
import json

import numpy as np

BUDGETS = [1, 5, 15, 60, 300, 1800]
OUT = "out/e5_budget_analysis.json"


def load(fams):
    rows = {}
    for name, paths in fams.items():
        rr = []
        for p in paths:
            for line in open(p):
                r = json.loads(line)
                cert = r.get("certify")
                if not cert or cert.get("t") is None or not cert.get("lb"):
                    continue
                inc = r.get("igcsstar") or r.get("gcsstar") or {}
                c_hat = inc.get("cost")
                if not c_hat or not np.isfinite(c_hat):
                    continue
                # certificate LB = max(subgraph relaxation LB, straight
                # line): both sound, the max is free and strictly tighter
                lb = max(cert["lb"], r["straight"])
                rr.append({
                    "t_search": inc.get("t") or 0.0,
                    "t_cert": cert["t"],
                    "gap_cert": max(0.0, (c_hat - lb) / c_hat),
                    "gap_triv": max(0.0, 1.0 - r["straight"] / c_hat),
                    "n_alive": cert.get("n_alive"),
                })
        rows[name] = rr
    return rows


def load_panda55():
    """E5-B parts: certify-after-search on the 55-region 7-DOF library.
    Queries where the search itself returned no finite incumbent are
    excluded (nothing to certify); a .started stamp without a result
    file would be right-censored at the 3h array wall (none occurred)."""
    bench = {r["i"]: r for r in
             (json.loads(l) for l in open("out/bench_panda1raw55.jsonl"))}
    rr = []
    for i, b in bench.items():
        c_hat = b.get("gcsstar", {}).get("cost")
        if not c_hat or not np.isfinite(c_hat):
            continue
        part = f"out/e5_certify_panda55_parts/q{i}.json"
        started = f"out/e5_certify_panda55_parts/q{i}.started"
        import os
        if os.path.exists(part):
            r = json.load(open(part))
            lb = max(r["lb"] or 0.0, b["straight"])
            rr.append({"t_search": b["gcsstar"].get("t") or 0.0,
                       "t_cert": r["t_cert"],
                       "gap_cert": max(0.0, (c_hat - lb) / c_hat),
                       "gap_triv": max(0.0, 1.0 - b["straight"] / c_hat),
                       "n_alive": r["n_alive"]})
        elif os.path.exists(started):
            rr.append({"t_search": b["gcsstar"].get("t") or 0.0,
                       "t_cert": float("inf"),
                       "gap_cert": None,
                       "gap_triv": max(0.0, 1.0 - b["straight"] / c_hat),
                       "n_alive": None})
    return rr


def main():
    fams = {
        "maze1600": ["out/bench_maze2d_1600.jsonl"],
        "uav (12 libs)": sorted(glob.glob("out/cluster_uav/bench_uavR*.jsonl")),
        "bimanual219": ["out/bench_219focus_K1.jsonl"],
    }
    data = load(fams)
    data["panda55 (E5-B)"] = load_panda55()
    report = {}
    for name, rr in data.items():
        n = len(rr)
        if n == 0:
            print(f"{name}: no certify rows")
            continue
        tc = np.array([r["t_cert"] for r in rr])
        print(f"\n=== {name} (n={n}) ===")
        print(f"certify t: median {np.median(tc):.2f}s  p90 "
              f"{np.percentile(tc,90):.1f}s  max {tc.max():.1f}s")
        dv = [r["gap_triv"] - r["gap_cert"] for r in rr
              if r["gap_cert"] is not None]
        if dv:
            print(f"certification value-added (triv - cert gap): median "
                  f"{100*np.median(dv):.2f} pts, max {100*max(dv):.2f} pts")
        print(f"{'B(s)':>6} {'cover%':>7} {'medGap%':>8} {'p90Gap%':>8} "
              f"{'p90 total(s)':>12}")
        rep = []
        for B in BUDGETS:
            gaps, totals = [], []
            cov = 0
            for r in rr:
                if r["t_cert"] <= B:
                    cov += 1
                    gaps.append(r["gap_cert"])
                    totals.append(r["t_search"] + r["t_cert"])
                else:
                    gaps.append(r["gap_triv"])
                    totals.append(r["t_search"] + B)
            row = {"B": B, "coverage": cov / n,
                   "med_gap": float(np.median(gaps)),
                   "p90_gap": float(np.percentile(gaps, 90)),
                   "p90_total": float(np.percentile(totals, 90))}
            rep.append(row)
            print(f"{B:6d} {100*row['coverage']:7.0f} "
                  f"{100*row['med_gap']:8.2f} {100*row['p90_gap']:8.2f} "
                  f"{row['p90_total']:12.1f}")
        # guard predictor: does n_alive at prune time predict t_cert?
        al = [r["n_alive"] for r in rr if r["n_alive"] is not None]
        if len(al) == n and n >= 8:
            al = np.array(al, float)
            rho = np.corrcoef(np.log1p(al), np.log1p(tc))[0, 1]
            print(f"guard predictor: corr(log n_alive, log t_cert) = "
                  f"{rho:.2f}")
            report[name + "_guard_corr"] = float(rho)
        report[name] = rep
    json.dump(report, open(OUT, "w"), indent=1)
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
