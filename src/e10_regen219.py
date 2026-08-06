#!/usr/bin/env python3
"""E10: regenerate the 219-region bimanual library with strict IrisZo
settings and a generation-time audit gate -- the fix the E8 regime
diagnosis prescribed for deep contamination.

Original library (stageB_scale_regions.py) used IrisZoOptions()
DEFAULTS: epsilon=0.01 (1% colliding volume tolerated), delta=0.05
(95% confidence), max_iterations=3.  Measured consequence: 3.2%
interface contamination, 62% of task-query reference answers colliding.

Strict recipe, same seed protocol for comparability (22 grasp + 200
random, same rng seed):
  * epsilon 0.001, delta 0.01, num_particles 4000, max_iterations 10,
    more separating-plane iterations, more mixing;
  * AUDIT GATE per region: M = ln(1/delta)/epsilon consecutive clean
    samples (hit-and-run x collision checker) or the region is regrown
    from a jittered seed (2 retries) and finally dropped -- max-iter
    hard landings can no longer enter the library silently;
  * then topology + interface-contamination measurement in-job.

Engineering (protocol-neutral): every seed's outcome is checkpointed to
--ckpt-dir the moment it is decided, so wall-clock kills lose nothing;
--stride/--stride-id shard the seed list for a Slurm job array; audit
and jitter rngs are keyed per seed name, so each seed's outcome is
independent of the sharding.  Rejected regions (last attempt) are kept
too -- the gate's refusal pattern is data.  --merge gathers checkpoints
into the library + topology + contamination report.

    python3 e10_regen219.py --stride 8 --stride-id 3   # one array task
    python3 e10_regen219.py --merge                    # after the array
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for p in (os.environ.get("GCS_SR", ""),
          os.path.expanduser("~/gcs-science-robotics")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import e6_sampling as e6
import stageB_bimanual_ipgcs as sb


def strict_options(n_threads=None):
    from pydrake.planning import IrisZoOptions
    o = IrisZoOptions()
    so = o.sampled_iris_options
    so.epsilon = 0.001
    so.delta = 0.01
    so.num_particles = 4000
    so.max_iterations = 10
    so.max_iterations_separating_planes = 40
    so.mixing_steps = 80
    if n_threads:
        # otherwise IrisZo spawns one thread per hardware core and
        # oversubscribes the Slurm cpu allocation
        try:
            from pydrake.common import Parallelism
            so.parallelism = Parallelism(n_threads)
        except Exception as e:
            print(f"parallelism not settable: {e}", flush=True)
    return o


def seed_rng(name, salt):
    return np.random.default_rng(zlib.crc32(f"{salt}:{name}".encode()))


def audit_region(checker, A, b, eps, delta, rng):
    """(passed, n_bad, n_tested): Bernoulli gate at (eps, delta)."""
    M = int(np.ceil(np.log(1.0 / delta) / eps))
    h = e6.reduce_to_affine_hull(np.asarray(A), np.asarray(b))
    if h is None or h["dim"] == 0:
        return False, -1, 0
    iface = {"A": np.asarray(A), "b": np.asarray(b), "hull": h}
    streak, tested, bad = 0, 0, 0
    while streak < M:
        for x in e6.sample_area(iface, 256, rng):
            tested += 1
            if checker.CheckConfigCollisionFree(np.asarray(x)):
                streak += 1
                if streak >= M:
                    break
            else:
                bad += 1
                return False, bad, tested
        if tested > 4 * M:
            break
    return streak >= M, bad, tested


def build_seeds(checker, dlo, dhi, n_random, seed):
    from reproduction.bimanual.helpers import getConfigurationSeeds
    grasp = {k: np.clip(np.asarray(v), dlo + 1e-3, dhi - 1e-3)
             for k, v in getConfigurationSeeds().items()}
    rng = np.random.default_rng(seed)
    seeds = dict(grasp)
    tries = 0
    while len(seeds) < len(grasp) + n_random and tries < 50 * n_random:
        tries += 1
        q = dlo + rng.random(len(dlo)) * (dhi - dlo)
        if checker.CheckConfigCollisionFree(q):
            seeds[f"rand_{len(seeds)}"] = q
    return seeds, len(grasp)


def ckpt_path(ckpt_dir, name):
    return os.path.join(ckpt_dir, name.replace("/", "__") + ".pkl")


def merge(args, checker, seeds):
    import json
    regions, report, missing = {}, {}, []
    n_rej = 0
    for name in seeds:
        p = ckpt_path(args.ckpt_dir, name)
        if not os.path.exists(p):
            missing.append(name)
            continue
        c = pickle.load(open(p, "rb"))
        report[name] = c["report"]
        if c.get("A") is not None and c["report"]["outcome"].startswith(
                "clean"):
            regions[name] = (c["A"], c["b"])
        else:
            n_rej += 1
    if missing:
        print(f"WARNING: {len(missing)} seeds missing checkpoints "
              f"(first: {missing[:4]})", flush=True)
    print(f"merge: {len(regions)} accepted, {n_rej} dropped, "
          f"{len(missing)} missing", flush=True)
    pickle.dump(regions, open(args.out, "wb"))

    reg_list = [(np.asarray(A), np.asarray(b))
                for A, b in regions.values()]
    t0 = time.perf_counter()
    topo = sb.build_topology(reg_list)
    print(f"topology: {len(topo['portals'])} portals in "
          f"{(time.perf_counter()-t0)/60:.0f} min", flush=True)
    pickle.dump(topo, open(args.topo_out, "wb"))

    ifs = e6.classify_interfaces(topo)
    rng2 = np.random.default_rng(2)
    n_tot, n_bad = 0, 0
    for f in ifs.values():
        for x in e6.sample_area(f, 3, rng2):
            n_tot += 1
            if not checker.CheckConfigCollisionFree(np.asarray(x)):
                n_bad += 1
    print(f"STRICT library interface contamination: {n_bad}/{n_tot} "
          f"({100*n_bad/max(n_tot,1):.2f}%)", flush=True)
    json.dump({"seeds": len(seeds), "accepted": len(regions),
               "dropped": n_rej, "missing": missing,
               "portals": len(topo["portals"]),
               "iface_contamination": [n_bad, n_tot],
               "report": report},
              open("out/e10_regen219_report.json", "w"), indent=1)
    print("written out/e10_regen219_report.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--stride-id", type=int, default=0)
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)))
    ap.add_argument("--ckpt-dir", default="out/e10_ckpt")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default="out/bimanual_regions_scaleS.pkl")
    ap.add_argument("--topo-out", default="out/bimanual219S_topo.pkl")
    args = ap.parse_args()

    from pydrake.geometry.optimization import Hyperellipsoid
    from pydrake.planning import IrisZo
    from stageB_bimanual_regions import build

    checker, domain = build()
    plant = checker.plant()
    dlo = plant.GetPositionLowerLimits()
    dhi = plant.GetPositionUpperLimits()
    seeds, n_grasp = build_seeds(checker, dlo, dhi,
                                 args.n_random, args.seed)
    if args.merge:
        merge(args, checker, seeds)
        return

    os.makedirs(args.ckpt_dir, exist_ok=True)
    mine = [(i, kv) for i, kv in enumerate(seeds.items())
            if i % args.stride == args.stride_id]
    todo = [(i, kv) for i, kv in mine
            if not os.path.exists(ckpt_path(args.ckpt_dir, kv[0]))]
    print(f"{len(seeds)} seeds ({n_grasp} grasp), strict IrisZo: "
          f"eps=0.001 delta=0.01 particles=4000 iters=10 | shard "
          f"{args.stride_id}/{args.stride}: {len(mine)} seeds, "
          f"{len(mine)-len(todo)} checkpointed, {len(todo)} to run, "
          f"{args.threads or 'all'} threads", flush=True)

    opts = strict_options(args.threads)
    t_all = time.perf_counter()
    for n_done, (i, (name, q0)) in enumerate(todo):
        q = np.asarray(q0, float)
        # per-seed rngs: outcome independent of sharding/restarts
        audit_rng = seed_rng(name, "audit")
        jit_rng = seed_rng(name, "jitter")
        outcome, best, rejects = "dropped", None, []
        rep = {"attempt": -1}
        for attempt in range(args.retries + 1):
            try:
                t0 = time.perf_counter()
                hp = IrisZo(checker, Hyperellipsoid.MakeHypersphere(
                    1e-2, q), domain, opts)
                t_grow = time.perf_counter() - t0
                A, b = np.asarray(hp.A()), np.asarray(hp.b())
                ok, bad, tested = audit_region(
                    checker, A, b, 0.001, 0.01, audit_rng)
                if ok:
                    best = (A, b)
                    outcome = f"clean({attempt})"
                    rep = {"attempt": attempt, "faces": len(b),
                           "t_grow": t_grow, "tested": tested}
                    break
                rejects.append({"A": A, "b": b, "bad": bad,
                                "tested": tested})
                print(f"[{i+1}/{len(seeds)}] {name} attempt {attempt}: "
                      f"audit FAILED ({bad} bad of {tested})", flush=True)
            except Exception as e:
                print(f"[{i+1}/{len(seeds)}] {name} attempt {attempt} "
                      f"grow error: {str(e)[:70]}", flush=True)
            for _ in range(200):
                qj = np.clip(q0 + 0.05 * jit_rng.standard_normal(
                    len(q0)), dlo + 1e-3, dhi - 1e-3)
                if checker.CheckConfigCollisionFree(qj):
                    q = qj
                    break
        rep["outcome"] = outcome
        A, b = best if best is not None else (None, None)
        tmp = ckpt_path(args.ckpt_dir, name) + f".tmp{args.stride_id}"
        pickle.dump({"name": name, "A": A, "b": b, "report": rep,
                     "rejects": rejects}, open(tmp, "wb"))
        os.replace(tmp, ckpt_path(args.ckpt_dir, name))
        print(f"[{i+1}/{len(seeds)}] {name}: {outcome}  "
              f"({n_done+1}/{len(todo)} in shard, "
              f"{(time.perf_counter()-t_all)/60:.0f} min)", flush=True)
    print(f"shard {args.stride_id} complete: {len(todo)} seeds in "
          f"{(time.perf_counter()-t_all)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
