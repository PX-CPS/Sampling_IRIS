#!/usr/bin/env python3
"""E9: direct verification -- does the GCS* implementation return
physically invalid paths on the 219-region 14-DOF benchmark?

Protocol (closes the "same cost => same path" inference gap):
  1. Re-run gcs_star exactly as the benchmark did (K=1, seed=0,
     timeout=60, max_revisits=1) on EVERY query; the reported cost must
     reproduce the stored benchmark cost (identity check).
  2. Re-solve the SAME chain SOCP that produced that reported cost,
     keeping all waypoints: that argmin IS the solution the planner
     commits to (its cost is the number in the benchmark).
  3. Validate that exact path with the continuous clearance certificate
     AND independently with dense boolean probes (two mechanisms).
  4. For every failure, bisect to a colliding configuration and report
     the colliding geometry PAIRS, their type (environment vs self;
     self would indicate a collision-filter mismatch, i.e. an audit
     artifact) and the penetration depth in meters.
The collision checker is built by the SAME function that generated the
regions (stageB_bimanual_regions.build), so filters are identical by
construction.

    python3 e9_verify_gcsstar.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import baseline_gcsstar as bgs
import stageB_bimanual_ipgcs as sb
from e8_validate import ContinuousValidator, L_BOUNDS


def chain_waypoints(chain_Ab, s, g):
    """Full-waypoint argmin of the exact program whose value gcs_star
    reports as its final cost."""
    import cvxpy as cp
    d = len(s)
    ys = [cp.Variable(d) for _ in chain_Ab]
    cons = [A @ y <= b for y, (A, b) in zip(ys, chain_Ab)]
    pts = [s] + ys + [g]
    cost = 0
    for a, b_ in zip(pts, pts[1:]):
        cost = cost + cp.norm(b_ - a)
    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver="CLARABEL")
    if ys and ys[0].value is None:
        return None, None
    return float(prob.value), np.vstack([s] + [y.value for y in ys] + [g])


def colliding_pairs(checker, q, influence=0.05):
    """(pairs, worst_penetration_m): colliding geometry at q."""
    plant = checker.plant()
    rc = checker.CalcRobotClearance(q, influence)
    d = np.asarray(rc.distances())
    out = []
    worst = 0.0
    if d.size == 0:
        return out, worst
    ri = rc.robot_indices()
    oi = rc.other_indices()
    ct = rc.collision_types()
    from pydrake.multibody.tree import BodyIndex
    for k in np.argsort(d):
        if d[k] >= 0:
            break
        b1 = plant.get_body(BodyIndex(int(ri[k])))
        b2 = plant.get_body(BodyIndex(int(oi[k])))
        n1 = f"{plant.GetModelInstanceName(b1.model_instance())}::{b1.name()}"
        n2 = f"{plant.GetModelInstanceName(b2.model_instance())}::{b2.name()}"
        out.append((n1, n2, str(ct[k]).split(".")[-1], float(d[k])))
        worst = min(worst, float(d[k]))
        if len(out) >= 4:
            break
    return out, worst


def dense_probe(checker, path, n=2000):
    p = np.asarray(path)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    bad = 0
    witness = None
    for t in np.linspace(0, L[-1], n):
        k = min(max(int(np.searchsorted(L, t, "right") - 1), 0),
                len(seg) - 1)
        lam = 0.0 if seg[k] < 1e-12 else (t - L[k]) / seg[k]
        q = (1 - lam) * p[k] + lam * p[k + 1]
        if not checker.CheckConfigCollisionFree(q):
            bad += 1
            if witness is None:
                witness = q
    return bad, witness


def main():
    data = pickle.load(open("out/bimanual_regions_scale.pkl", "rb"))
    reg_dict = data["regions"] if isinstance(data, dict) and \
        "regions" in data else data
    regions = [(np.asarray(A), np.asarray(b)) for A, b in reg_dict.values()]
    topo = pickle.load(open("out/bimanual219_topo.pkl", "rb"))
    if isinstance(topo, dict) and "topo" in topo:
        topo = topo["topo"]
    bench = [json.loads(l) for l in open("out/bench_219focus_K1.jsonl")]

    from stageB_bimanual_regions import build
    checker, _ = build()          # SAME builder as region generation
    validator = ContinuousValidator(checker, L_BOUNDS["bimanual"])
    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}

    n_invalid, n_clean, rows = 0, 0, []
    for rec in bench:
        i = rec["i"]
        s, g = np.asarray(rec["s"]), np.asarray(rec["g"])
        stored = rec.get("gcsstar", {}).get("cost")
        rg = sb.with_query(topo, s, g)
        t0 = time.perf_counter()
        r = bgs.gcs_star(rg, timeout=60.0, K=1, max_revisits=1, seed=0)
        t_solve = time.perf_counter() - t0
        if r["path"] is None or r["path"] == "warm" \
                or not np.isfinite(r["cost"]):
            print(f"[q{i:2d}] gcs_star returned no path "
                  f"(stored {stored}); skip", flush=True)
            rows.append({"i": i, "status": "no_path", "stored": stored})
            continue
        chain = []
        okc = True
        for a, b_ in zip(r["path"], r["path"][1:]):
            eid = pair_eid.get(frozenset((a, b_)))
            if eid is None:
                okc = False
                break
            p = topo["portals"][eid]
            chain.append((p["A"], p["b"]))
        if not okc:
            print(f"[q{i:2d}] chain reconstruction failed", flush=True)
            continue
        c2, wpath = chain_waypoints(chain, s, g)
        d_run = abs(r["cost"] - (stored or r["cost"]))
        d_chain = abs(c2 - r["cost"]) if c2 is not None else np.inf
        ok_cont, info = validator.path(wpath)
        n_bool, witness = dense_probe(checker, wpath)
        agree = (not ok_cont) == (n_bool > 0) or ok_cont == (n_bool == 0)
        pairs, worst = ([], 0.0)
        if witness is not None:
            pairs, worst = colliding_pairs(checker, witness)
        verdict = "CLEAN" if (ok_cont and n_bool == 0) else "INVALID"
        if verdict == "INVALID":
            n_invalid += 1
        else:
            n_clean += 1
        rows.append({"i": i, "stored": stored, "rerun": r["cost"],
                     "chain": c2, "d_stored": d_run, "d_chain": d_chain,
                     "verdict": verdict, "bool_bad": int(n_bool),
                     "cont_bad_segments": info["bad_segments"],
                     "worst_penetration_m": worst,
                     "pairs": pairs, "mech_agree": bool(agree)})
        pstr = "; ".join(f"{a} vs {b} [{t}] {d*1000:.1f}mm"
                         for a, b, t, d in pairs[:2])
        print(f"[q{i:2d}] stored {stored:.4f} rerun {r['cost']:.4f} "
              f"(d={d_run:.1e}) chain {c2:.4f} (d={d_chain:.1e}) "
              f"[{t_solve:.1f}s] -> {verdict}"
              + (f"  boolProbes {n_bool}/2000, pen {1000*worst:.1f}mm, "
                 f"{pstr}" if verdict == "INVALID" else ""), flush=True)

    print(f"\nVERDICT: {n_invalid}/{n_invalid+n_clean} reference answers "
          f"physically INVALID; both mechanisms agreed on "
          f"{sum(1 for r in rows if r.get('mech_agree'))} rows")
    env_only = all(all(t == "kEnvironmentCollision" for _, _, t, _
                       in r.get("pairs", []))
                   for r in rows if r.get("verdict") == "INVALID")
    print(f"all flagged collisions are robot-ENVIRONMENT (no self-"
          f"collision filter artifacts): {env_only}")
    json.dump(rows, open("out/e9_verify_gcsstar.json", "w"), indent=1)
    print("written out/e9_verify_gcsstar.json")


if __name__ == "__main__":
    main()
