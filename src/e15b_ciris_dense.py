#!/usr/bin/env python3
"""E15-B: what it takes to make the C-IRIS library usable.

Keeps the 7 equal-budget regions from e15, adds farthest-point-sampled
extra seeds (C-IRIS side only) until the library connects, then
re-measures coverage, the overlap graph, query success, and -- once
something solves -- the metric price: s-space optima re-measured as
true q-space arc length against the IrisZo library's answers on the
SAME query pairs.

    python3 e15b_ciris_dense.py      # out/e15b_ciris_dense.json
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import e6_sampling as e6
import stageB_bimanual_ipgcs as sb
from e15_ciris_2dof import (SEED_Q, build, dense_ok, grow_ciris,
                            q_arclength)

N_EXTRA = 12


def in_any(x, regs, tol=1e-9):
    return any(np.all(A @ x <= b + tol) for A, b in regs)


def overlap_edges(regs):
    out = []
    for i, j in itertools.combinations(range(len(regs)), 2):
        A2 = np.vstack([regs[i][0], regs[j][0]])
        b2 = np.concatenate([regs[i][1], regs[j][1]])
        ch = e6.chebyshev(A2, b2)
        if ch is not None and ch[1] > 1e-9:
            out.append((i, j))
    return out


def main():
    d = pickle.load(open("out/e15_regions.pkl", "rb"))
    diagram, checker, arm = build()
    plant = checker.plant()
    lo, hi = (plant.GetPositionLowerLimits(),
              plant.GetPositionUpperLimits())
    q_star = np.zeros(2)

    # farthest-point extra seeds from a free-grid, away from existing
    qs = np.linspace(lo[0] + 0.05, hi[0] - 0.05, 31)
    grid = np.array([(a, b) for a in qs for b in qs
                     if checker.CheckConfigCollisionFree(
                         np.array([a, b]))])
    chosen = list(SEED_Q)
    extras = []
    for _ in range(N_EXTRA):
        D = np.min([np.linalg.norm(grid - c, axis=1) for c in chosen],
                   axis=0)
        k = int(np.argmax(D))
        extras.append(grid[k].copy())
        chosen.append(grid[k].copy())
    print("extra seeds:", np.round(extras, 2).tolist(), flush=True)

    t0 = time.perf_counter()
    rc_extra, tc, rfk = grow_ciris(diagram, extras, q_star)
    t_extra = time.perf_counter() - t0
    regs = list(d["ciris"]) + rc_extra
    print(f"{len(regs)} C-IRIS regions total "
          f"(+{len(rc_extra)} in {t_extra:.0f}s)", flush=True)

    def to_s(q):
        return rfk.ComputeSValue(np.asarray(q, float), q_star)

    def to_q(s):
        return rfk.ComputeQValue(np.asarray(s, float), q_star)

    edges = overlap_edges(regs)
    print(f"overlap edges: {len(edges)}", flush=True)
    X = lo + np.random.default_rng(1).random((8000, 2)) * (hi - lo)
    free = np.array([checker.CheckConfigCollisionFree(x) for x in X])
    cov = np.mean([in_any(to_s(x), regs) for x in X[free]])
    print(f"free-space coverage: {100*cov:.1f}%", flush=True)

    # same 30 pairs as e15 (same rng recipe)
    qrng = np.random.default_rng(9)
    pairs, tries = [], 0
    while len(pairs) < 30 and tries < 5000:
        tries += 1
        a = lo + qrng.random(2) * (hi - lo)
        b = lo + qrng.random(2) * (hi - lo)
        if (checker.CheckConfigCollisionFree(a)
                and checker.CheckConfigCollisionFree(b)):
            pairs.append((a, b))

    topo_c = sb.build_topology(regs)
    zt = sb.build_topology(list(d["iriszo"]))
    ok = bad = nosol = 0
    rows = []
    for a, b in pairs:
        try:
            rg = sb.with_query(topo_c, to_s(a), to_s(b))
            rr = sb.solve_gcs(rg)
        except Exception:
            rr = None
        if rr is None or rr.get("waypoints") is None:
            nosol += 1
            continue
        wp = np.asarray(rr["waypoints"])
        qdense = []
        for k in range(len(wp) - 1):
            for t in np.linspace(0, 1, 60, endpoint=False):
                qdense.append(to_q((1 - t) * wp[k] + t * wp[k + 1]))
        qdense.append(to_q(wp[-1]))
        clean = dense_ok(checker, np.asarray(qdense), step=2e-3)
        ok += clean
        bad += (not clean)
        row = {"q_len_ciris": q_arclength(to_q, wp), "clean": clean}
        # IrisZo answer on the same pair, for the metric-price ratio
        try:
            rz = sb.solve_gcs(sb.with_query(zt, a, b))
            if rz is not None and rz.get("waypoints") is not None:
                wz = np.asarray(rz["waypoints"])
                row["q_len_iriszo"] = float(np.linalg.norm(
                    np.diff(wz, axis=0), axis=1).sum())
        except Exception:
            pass
        rows.append(row)
    both = [r for r in rows
            if r.get("q_len_iriszo") and r["clean"]]
    ratio = (float(np.median([r["q_len_ciris"] / r["q_len_iriszo"]
                              for r in both])) if both else None)
    print(f"queries: {ok} clean / {bad} colliding / {nosol} no-path; "
          f"metric price (median q-length ratio C-IRIS/IrisZo on "
          f"{len(both)} shared pairs): {ratio}", flush=True)

    json.dump({"n_regions": len(regs), "extra_time_s": t_extra,
               "edges": len(edges), "coverage": float(cov),
               "queries": {"clean": ok, "colliding": bad,
                           "no_path": nosol},
               "metric_price_ratio": ratio, "rows": rows},
              open("out/e15b_ciris_dense.json", "w"), indent=1)
    pickle.dump({"ciris_dense": regs},
                open("out/e15b_regions.pkl", "wb"))
    print("written out/e15b_ciris_dense.json", flush=True)


if __name__ == "__main__":
    main()
