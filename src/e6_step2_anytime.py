#!/usr/bin/env python3
"""E6 step 2: informed, parallel, batched interface sampling -- the
anytime protocol ("shortest path per unit time").

Round loop:
  1. sample a batch of M points on every ALIVE interface (equal budget,
     parallelized over interfaces with a worker pool),
  2. rebuild the intra-region graph over ALIVE nodes, Dijkstra s->g,
  3. if the incumbent improves, AUDIT the new path (pinch junctions,
     zero-thickness walls, obstacle penetration) before accepting it,
  4. tighten the informed filter with the new incumbent.

Informed filter (admissible, Lemma-1 style at interface/node level):
  * interface P stays alive iff ell(P) = min_{x in P} |x-s|+|x-g|
    <= c_best  (ell is query-constant: ONE SOCP per interface, cached;
    the filter afterwards is a comparison),
  * node x stays alive iff |x-s|+|x-g| <= c_best (cached at creation).
  Both prune only paths provably longer than the incumbent, and every
  node on the incumbent satisfies the bound (triangle inequality), so
  the incumbent is never lost and the cost curve is monotone.

Samplers follow the traversability rule from step 1: contour on
full-dimensional overlaps, inset contour on degenerate contact faces
(mode "contour"), or hit-and-run everywhere (mode "area").

    python3 e6_step2_anytime.py --lib out/e6_iris2d.pkl \
        --out out/e6_step2_iris.json
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
import e6_sampling as e6
import stageB_bimanual_ipgcs as sb
from e6_build_iris2d import penetration_depth

TOL = 1e-9


def interface_ell(iface, s, g):
    """Exact min_{x in P} |x-s| + |x-g| (SOCP, solved once per query)."""
    import cvxpy as cp
    d = len(s)
    x = cp.Variable(d)
    prob = cp.Problem(cp.Minimize(cp.norm(x - s) + cp.norm(x - g)),
                      [iface["A"] @ x <= iface["b"]])
    prob.solve(solver="CLARABEL")
    return float(prob.value)


def _sample_batch(task):
    """Worker task: (iface, mode, M, seed) -> list of points."""
    iface, mode, M, seed = task
    rng = np.random.default_rng(seed)
    return e6.SAMPLERS[mode](iface, M, rng)


class AnytimeInterfacePlanner:
    def __init__(self, topo, ifaces, s, g, mode="contour", M=2,
                 informed=True, workers=1, seed=0,
                 obstacles=None, walls=None, ell=None, dim_cache=None,
                 path_audit=None):
        self.topo = topo
        self.regions = topo["regions"]
        self.d = len(s)
        self.ifaces = ifaces
        self.s, self.g = np.asarray(s, float), np.asarray(g, float)
        self.mode = mode
        self.M = M
        self.informed = informed
        self.workers = workers
        self.seed = seed
        self.obstacles = obstacles or []
        self.walls = walls or []
        self.c_best = np.inf
        self.best_path = None
        self.pos, self.node_iface, self.mems, self.metric = [], [], [], []
        self.seen = {}
        self.region_nodes = {r: [] for r in range(len(self.regions))}
        self.events = []            # (t_wall, c_best) staircase
        self.event_paths = []       # incumbent path at each improvement
        self.rows = []              # per-round diagnostics
        self.n_sampled = 0
        self._dim_cache = {} if dim_cache is None else dim_cache
        self.path_audit = path_audit
        self.strict_audit = True      # 2-D testbeds: any audit failure is
        self.rejections = []          # a bug; high-D subclasses override
        self._last_idx = None         # node-index path of the last solve
        self._t0 = None
        self._pool = None
        if workers > 1:
            import multiprocessing as mp
            self._pool = mp.get_context("fork").Pool(workers)
        if ell is not None:
            # precomputed admissible keys (e.g. closed-form bbox lower
            # bounds in high-D, where exact per-interface SOCPs would
            # dominate the anytime clock); any LOWER bound on ell keeps
            # the filter admissible
            self.ell = ell
            self.t_ell = 0.0
        else:
            # query-constant informed keys, one SOCP per interface
            t0 = time.perf_counter()
            self.ell = {e: interface_ell(f, self.s, self.g)
                        for e, f in ifaces.items()}
            self.t_ell = time.perf_counter() - t0

    def _sampler_for(self, iface):
        if self.mode == "area":
            return "area"
        # traversability rule (step 1): raw contour only on full-dim
        return "contour" if iface["dim"] == self.d else "contour_in"

    def _add_samples(self, eid, pts):
        f = self.ifaces[eid]
        parents = (f["u"], f["v"])
        for x in pts:
            x = np.asarray(x, float)
            key = tuple(np.round(x * 1e9).astype(np.int64))
            if key in self.seen:               # merged node: union parents
                i = self.seen[key]
                new = self.mems[i] | frozenset(parents)
                if new != self.mems[i]:
                    for r in new - self.mems[i]:
                        self.region_nodes[r].append(i)
                    self.mems[i] = new
                continue
            i = len(self.pos)
            self.seen[key] = i
            self.pos.append(x)
            self.node_iface.append(eid)
            self.mems.append(frozenset(parents))
            self.metric.append(float(np.linalg.norm(x - self.s)
                                     + np.linalg.norm(x - self.g)))
            for r in parents:
                self.region_nodes[r].append(i)
            self.n_sampled += 1

    def _on_reject(self, path, aud):
        """Hook for subclasses (edge blacklisting etc.)."""

    def _attempt_repair(self, path, cost, aud):
        """Hook: try to fix a physically-failed incumbent.  Base: no."""
        return path, cost, False

    def _alive_nodes(self):
        if not (self.informed and np.isfinite(self.c_best)):
            return None
        return {i for i, m in enumerate(self.metric)
                if m <= self.c_best + TOL}

    def _solve(self):
        rm = {"pos": np.asarray(self.pos), "region_nodes": self.region_nodes,
              "mems": self.mems}
        return e6.shortest_path(rm, self.regions, self.s, self.g,
                                self._alive_nodes())

    def _audit(self, path, cost):
        a_j = e6.audit_junctions(path, self.regions,
                                 dim_cache=self._dim_cache)
        n_wall = 0
        if self.walls:
            n_wall = sum(
                1 for k in range(len(path) - 1)
                if e6.segment_crosses_wall(path[k], path[k + 1],
                                           self.walls) > 0)
        pen = 0.0
        if self.obstacles:
            for k in range(len(path) - 1):
                for t in np.linspace(0, 1, 20):
                    p = (1 - t) * path[k] + t * path[k + 1]
                    pen = max(pen, penetration_depth(p, self.obstacles))
        ok = (a_j["bad_junctions"] == 0 and n_wall == 0 and pen <= 1e-8)
        info = {"bad_junctions": a_j["bad_junctions"],
                "wall_crossings": n_wall, "penetration": pen}
        if ok and self.path_audit is not None:
            ok2, extra = self.path_audit(path)
            ok = ok and ok2
            info.update(extra)
        return ok, info

    def run(self, rounds=10):
        self._t0 = time.perf_counter()
        for rnd in range(rounds):
            t_r = time.perf_counter()
            if self.informed and np.isfinite(self.c_best):
                alive_if = [e for e in self.ifaces
                            if self.ell[e] <= self.c_best + TOL]
            else:
                alive_if = list(self.ifaces)
            tasks = [(self.ifaces[e], self._sampler_for(self.ifaces[e]),
                      self.M, self.seed * 1000003 + 7919 * rnd + e)
                     for e in alive_if]
            t_s0 = time.perf_counter()
            if self._pool is not None:
                batches = self._pool.map(_sample_batch, tasks)
            else:
                batches = [_sample_batch(t) for t in tasks]
            t_sample = time.perf_counter() - t_s0
            for e, pts in zip(alive_if, batches):
                self._add_samples(e, pts)
            t_g0 = time.perf_counter()
            cost, path = self._solve()
            t_solve = time.perf_counter() - t_g0
            accepted = False
            if np.isfinite(cost) and cost < self.c_best - 1e-12:
                ok, aud = self._audit(path, cost)
                if not ok and self.strict_audit:
                    raise RuntimeError(f"audit failed on incumbent: {aud}")
                if not ok:
                    # lazy-validation semantics: try to repair first,
                    # else reject + blacklist and keep planning
                    p2, c2, ok2 = self._attempt_repair(path, cost, aud)
                    if ok2 and c2 < self.c_best - 1e-12:
                        self.c_best = c2
                        self.best_path = p2
                        accepted = True
                        self.events.append(
                            (time.perf_counter() - self._t0, c2))
                        self.event_paths.append(np.asarray(p2).copy())
                    else:
                        self.rejections.append(
                            {"round": rnd, "cost": float(cost),
                             **{k: v for k, v in aud.items()
                                if np.isscalar(v)}})
                        self._on_reject(path, aud)
                else:
                    self.c_best = cost
                    self.best_path = path
                    accepted = True
                    self.events.append((time.perf_counter() - self._t0,
                                        cost))
                    self.event_paths.append(np.asarray(path).copy())
            self.rows.append({
                "round": rnd, "t_round": time.perf_counter() - t_r,
                "t_sample": t_sample, "t_solve": t_solve,
                "n_ifaces_alive": len(alive_if),
                "n_nodes": len(self.pos), "n_sampled": self.n_sampled,
                "c_best": float(self.c_best), "improved": accepted})
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
        return self

    def summary(self):
        return {"mode": self.mode, "M": self.M, "informed": self.informed,
                "workers": self.workers, "seed": self.seed,
                "t_ell": self.t_ell, "c_best": float(self.c_best),
                "events": [(float(t), float(c)) for t, c in self.events],
                "event_paths": [p.tolist() for p in self.event_paths],
                "ell": {int(e): float(v) for e, v in self.ell.items()},
                "rows": self.rows,
                "alive_ifaces_final": [int(e) for e in self.ifaces
                                       if self.ell[e] <= self.c_best + TOL],
                "best_path": None if self.best_path is None
                else np.asarray(self.best_path).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="out/e6_iris2d.pkl")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--timing-M", type=int, default=64,
                    help="heavier per-round budget for the worker-scaling "
                         "measurement")
    ap.add_argument("--out", default="out/e6_step2_iris.json")
    args = ap.parse_args()

    lib = pickle.load(open(args.lib, "rb"))
    topo = lib["topo"]
    obstacles = [np.asarray(o) for o in lib.get("obstacles", [])]
    if "task_configs" in lib:
        s = np.asarray(lib["task_configs"]["start"])
        g = np.asarray(lib["task_configs"]["goal"])
    else:
        s = np.array([0.5, 0.5])
        g = np.asarray(topo["bbox_hi"][-1]) - 0.5
    ifaces = e6.classify_interfaces(topo)
    walls = e6.box_walls(topo) if not obstacles else []
    print(f"{args.lib}: {topo['n']} regions, {len(ifaces)} interfaces, "
          f"{len(walls)} walls", flush=True)

    gt = sb.solve_gcs(sb.with_query(topo, s, g))
    print(f"ground truth: LB={gt['lb']:.6f} UB={gt['ub']:.6f}", flush=True)

    seeds = [int(x) for x in args.seeds.split(",")]
    runs = []
    for mode in ("contour", "area"):
        for informed in (True, False):
            for sd in seeds:
                pl = AnytimeInterfacePlanner(
                    topo, ifaces, s, g, mode=mode, M=args.M,
                    informed=informed, workers=1, seed=sd,
                    obstacles=obstacles, walls=walls).run(args.rounds)
                assert pl.c_best >= gt["lb"] - 1e-6 * max(1, abs(gt["lb"]))
                runs.append(pl.summary())
                print(f"  {mode:8s} informed={int(informed)} seed={sd}: "
                      f"c={pl.c_best:.4f} "
                      f"(vs UB {100*(pl.c_best/gt['ub']-1):+.2f}%), "
                      f"alive ifaces "
                      f"{runs[-1]['rows'][-1]['n_ifaces_alive']}"
                      f"/{len(ifaces)}", flush=True)

    timing = []
    for workers in (1, 2, 4):
        pl = AnytimeInterfacePlanner(
            topo, ifaces, s, g, mode="contour", M=args.timing_M,
            informed=True, workers=workers, seed=0,
            obstacles=obstacles, walls=walls).run(4)
        ts = float(np.sum([r["t_sample"] for r in pl.rows]))
        timing.append({"workers": workers, "t_sample_total": ts,
                       "t_total": float(np.sum([r["t_round"]
                                                for r in pl.rows])),
                       "c_best": float(pl.c_best)})
        print(f"  workers={workers}: sample {ts:.2f}s "
              f"total {timing[-1]['t_total']:.2f}s c={pl.c_best:.4f}",
              flush=True)

    json.dump({"lib": args.lib, "gt": {"lb": gt["lb"], "ub": gt["ub"]},
               "s": s.tolist(), "g": g.tolist(),
               "n_ifaces": len(ifaces), "runs": runs, "timing": timing},
              open(args.out, "w"), indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
