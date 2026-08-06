#!/usr/bin/env python3
"""E15: certified regions x certified layer -- C-IRIS meets the
interface-sampling pipeline, on a 2-DoF planar arm.

Same scene, same seeds, two libraries:
  * IrisZo (probabilistic (eps, delta) contract, q-space polytopes);
  * C-IRIS  (deterministic SOS certificates, polytopes in the rational
    s = tan((q - q*)/2) space).
On top of both, the SAME layer: build_topology -> interface
classification -> area sampling -> audit, plus the geometric GCS
answer and ground-truth dense validation of every path.

Measured axes: generation wall-clock, interface/interior contamination
(the audit should read exactly 0 on C-IRIS -- its selling point),
query coverage/connectivity, and the metric price (C-IRIS optimizes
s-space lengths; we re-measure its answers as true q-space arc
length).

2-DoF because C-IRIS SOS programs are practical there in minutes and
MOSEK is licensed on this machine only (smoke-scale, run locally).

    python3 e15_ciris_2dof.py        # out/e15_ciris.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, ".")

ARM_URDF = """<?xml version="1.0"?>
<robot name="arm2">
  <link name="base"/>
  <link name="link1">
    <collision><origin xyz="0.5 0 0"/>
      <geometry><box size="0.86 0.08 0.05"/></geometry></collision>
  </link>
  <link name="link2">
    <collision><origin xyz="0.4 0 0"/>
      <geometry><box size="0.66 0.07 0.05"/></geometry></collision>
  </link>
  <joint name="q1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.0" upper="3.0" effort="10" velocity="10"/>
  </joint>
  <joint name="q2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="1.0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.0" upper="3.0" effort="10" velocity="10"/>
  </joint>
</robot>"""

OBSTACLES = [  # (cx, cy, sx, sy) axis-aligned boxes in the plane
    (1.30, 0.95, 0.42, 0.42),
    (1.25, -0.85, 0.38, 0.38),
    (-1.05, 0.85, 0.42, 0.42),
    (-0.35, -1.25, 0.38, 0.38),
]

SEED_Q = [np.array([0.0, 0.0]), np.array([1.9, 0.4]),
          np.array([-2.17, -0.48]), np.array([2.42, -1.21]),
          np.array([-2.7, 1.2]), np.array([0.9, -2.2]),
          np.array([-0.9, 2.2])]

N_AUDIT = 2000


def build():
    from pydrake.geometry import Box, ProximityProperties, Role
    from pydrake.math import RigidTransform
    from pydrake.multibody.plant import CoulombFriction
    from pydrake.planning import (RobotDiagramBuilder,
                                  SceneGraphCollisionChecker)
    rdb = RobotDiagramBuilder(time_step=0.0)
    plant = rdb.plant()
    arm, = rdb.parser().AddModelsFromString(ARM_URDF, "urdf")
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("base"))
    for i, (cx, cy, sx, sy) in enumerate(OBSTACLES):
        plant.RegisterCollisionGeometry(
            plant.world_body(), RigidTransform([cx, cy, 0.0]),
            Box(sx, sy, 0.05), f"obs{i}", CoulombFriction(0.9, 0.8))
    plant.Finalize()
    diagram = rdb.Build()
    checker = SceneGraphCollisionChecker(
        model=diagram, robot_model_instances=[arm],
        edge_step_size=0.02)
    return diagram, checker, arm


def grow_iriszo(checker, seeds):
    from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
    from pydrake.planning import IrisZo, IrisZoOptions
    plant = checker.plant()
    dom = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(),
                              plant.GetPositionUpperLimits())
    regions, times = [], []
    for q in seeds:
        t0 = time.perf_counter()
        hp = IrisZo(checker, Hyperellipsoid.MakeHypersphere(1e-2, q),
                    dom, IrisZoOptions())
        times.append(time.perf_counter() - t0)
        regions.append((np.asarray(hp.A()), np.asarray(hp.b())))
    return regions, times


def grow_ciris(diagram, seeds, q_star):
    from pydrake.geometry.optimization import (CspaceFreePolytope,
                                               SeparatingPlaneOrder)
    from pydrake.multibody.rational import RationalForwardKinematics
    plant = diagram.plant()
    sg = diagram.scene_graph()
    rfk = RationalForwardKinematics(plant)
    cfp = CspaceFreePolytope(plant, sg, SeparatingPlaneOrder.kAffine,
                             q_star)
    # the certified set is {s | C s <= d, s_lo <= s <= s_up}; the
    # joint-limit rows are part of its definition and MUST ship with
    # the exported polytope (after alternation reshapes C, the bare
    # (C, d) can even be unbounded)
    s_lo = rfk.ComputeSValue(plant.GetPositionLowerLimits(), q_star)
    s_up = rfk.ComputeSValue(plant.GetPositionUpperLimits(), q_star)
    nq = len(s_lo)
    B_A = np.vstack([np.eye(nq), -np.eye(nq)])
    B_b = np.concatenate([s_up, -s_lo])
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    C = np.c_[np.cos(ang), np.sin(ang)]
    opts = CspaceFreePolytope.BinarySearchOptions()
    opts.scale_max = 120.0        # default 1.0 pins the region to the
    opts.max_iter = 18            # seed box -- let it actually grow
    # BinarySearch scales the seed 12-gon isotropically, so the first
    # obstacle met in ANY direction stops growth in ALL of them; the
    # paper's max-volume tool is bilinear alternation, which reshapes
    # C and d anisotropically.  Chain the two, as the paper does.
    bopts = CspaceFreePolytope.BilinearAlternationOptions()
    bopts.max_iter = 12
    regions, times = [], []
    for q in seeds:
        sc = rfk.ComputeSValue(np.asarray(q, float), q_star)
        d0 = C @ sc + 0.03
        t0 = time.perf_counter()
        res = cfp.BinarySearch(set(), C, d0, sc, opts)
        if res is None:
            times.append(time.perf_counter() - t0)
            print(f"  C-IRIS seed {q}: no certified polytope",
                  flush=True)
            continue
        Cb, db = np.asarray(res.C()), np.asarray(res.d())
        try:
            # back off the certification boundary slightly, or the
            # first Lagrangian step of the alternation is marginal
            db_in = C @ sc + 0.98 * (db - C @ sc)
            hist = cfp.SearchWithBilinearAlternation(set(), Cb, db_in,
                                                     bopts)
            print(f"  seed {q}: alternation history {len(hist)}",
                  flush=True)
            if hist:
                Cb, db = (np.asarray(hist[-1].C()),
                          np.asarray(hist[-1].d()))
        except Exception as e:
            print(f"  bilinear alternation failed ({str(e)[:60]}); "
                  f"keeping binary-search polytope", flush=True)
        times.append(time.perf_counter() - t0)
        regions.append((np.vstack([Cb, B_A]),
                        np.concatenate([db, B_b])))
    return regions, times, rfk


def audit(regions, checker, to_q, rng, n=N_AUDIT, interfaces=True):
    """Sample interfaces (or interiors) and re-check physically."""
    import e6_sampling as e6
    import stageB_bimanual_ipgcs as sb
    topo = sb.build_topology(regions)
    ifs = e6.classify_interfaces(topo)
    tgt = (list(ifs.values()) if interfaces else
           [{"A": A, "b": b,
             "hull": e6.reduce_to_affine_hull(np.asarray(A),
                                              np.asarray(b))}
            for A, b in regions])
    tgt = [f for f in tgt if f.get("hull") or interfaces]
    bad = tot = 0
    per = max(1, n // max(1, len(tgt)))
    for f in tgt:
        for x in e6.sample_area(f, per, rng):
            tot += 1
            if not checker.CheckConfigCollisionFree(to_q(np.asarray(x))):
                bad += 1
    return bad, tot, topo, ifs


def dense_ok(checker, qpath, step=1e-3):
    p = np.asarray(qpath)
    for k in range(len(p) - 1):
        L = np.linalg.norm(p[k + 1] - p[k])
        for t in np.linspace(0, 1, max(2, int(np.ceil(L / step)))):
            if not checker.CheckConfigCollisionFree(
                    (1 - t) * p[k] + t * p[k + 1]):
                return False
    return True


def q_arclength(to_q, spath, nsub=200):
    """True q-space length of a straight-in-s path."""
    p = np.asarray(spath)
    tot = 0.0
    for k in range(len(p) - 1):
        qs = [to_q((1 - t) * p[k] + t * p[k + 1])
              for t in np.linspace(0, 1, nsub)]
        tot += float(np.linalg.norm(np.diff(np.asarray(qs), axis=0),
                                    axis=1).sum())
    return tot


def main():
    import stageB_bimanual_ipgcs as sb

    diagram, checker, arm = build()
    for q in SEED_Q:
        assert checker.CheckConfigCollisionFree(q), f"seed {q} dirty"
    q_star = np.zeros(2)

    print("=== IrisZo (probabilistic) ===", flush=True)
    rz, tz = grow_iriszo(checker, SEED_Q)
    print(f"{len(rz)} regions, {sum(tz):.1f}s total "
          f"({np.median(tz):.2f}s median)", flush=True)

    print("=== C-IRIS (SOS-certified) ===", flush=True)
    rc, tc, rfk = grow_ciris(diagram, SEED_Q, q_star)
    print(f"{len(rc)} regions, {sum(tc):.1f}s total "
          f"({np.median(tc):.2f}s median)", flush=True)

    def sq(s):
        return rfk.ComputeQValue(np.asarray(s, float), q_star)

    rng = np.random.default_rng(3)
    bz, nz, topo_z, _ = audit(rz, checker, lambda x: x, rng)
    bc, nc, topo_c, _ = audit(rc, checker, sq, rng)
    bzi, nzi, _, _ = audit(rz, checker, lambda x: x, rng,
                           interfaces=False)
    bci, nci, _, _ = audit(rc, checker, sq, rng, interfaces=False)
    print(f"interface contamination: IrisZo {bz}/{nz} "
          f"({100*bz/max(nz,1):.2f}%)  C-IRIS {bc}/{nc} "
          f"({100*bc/max(nc,1):.2f}%)", flush=True)
    print(f"interior  contamination: IrisZo {bzi}/{nzi} "
          f"({100*bzi/max(nzi,1):.2f}%)  C-IRIS {bci}/{nci} "
          f"({100*bci/max(nci,1):.2f}%)", flush=True)

    # ---- coverage + the canonical query through the pipeline ----
    plant = checker.plant()
    lo, hi = (plant.GetPositionLowerLimits(),
              plant.GetPositionUpperLimits())
    qrng = np.random.default_rng(9)
    pairs, tries = [], 0
    while len(pairs) < 30 and tries < 5000:
        tries += 1
        a = lo + qrng.random(2) * (hi - lo)
        b = lo + qrng.random(2) * (hi - lo)
        if (checker.CheckConfigCollisionFree(a)
                and checker.CheckConfigCollisionFree(b)):
            pairs.append((a, b))

    def solved(topo, to_s, to_q, a, b):
        rg = sb.with_query(topo, to_s(a), to_s(b))
        try:
            r = sb.solve_gcs(rg)
        except Exception:
            return None
        if r is None or r.get("waypoints") is None:
            return None
        wp = np.asarray(r["waypoints"])
        qlen = q_arclength(to_q, wp)
        clean = dense_ok(checker,
                         [to_q(w) for w in
                          np.vstack([wp[0]] + [w for w in wp])])
        return {"s_cost": float(np.linalg.norm(np.diff(wp, axis=0),
                                               axis=1).sum()),
                "q_len": qlen, "clean_endpoints_only": clean}

    stats = {}
    for name, topo, to_s, to_q in (
            ("iriszo", topo_z, lambda q: q, lambda s: s),
            ("ciris", topo_c,
             lambda q: rfk.ComputeSValue(np.asarray(q, float), q_star),
             sq)):
        ok = bad_path = nosol = 0
        qlens = []
        for a, b in pairs:
            r = solved(topo, to_s, to_q, a, b)
            if r is None:
                nosol += 1
                continue
            # ground-truth: densified q-space walk of the answer
            wp = None
            rg = sb.with_query(topo, to_s(a), to_s(b))
            rr = sb.solve_gcs(rg)
            wp = np.asarray(rr["waypoints"])
            qdense = []
            for k in range(len(wp) - 1):
                for t in np.linspace(0, 1, 60, endpoint=False):
                    qdense.append(to_q((1 - t) * wp[k] + t * wp[k + 1]))
            qdense.append(to_q(wp[-1]))
            if dense_ok(checker, np.asarray(qdense), step=2e-3):
                ok += 1
                qlens.append(r["q_len"])
            else:
                bad_path += 1
        stats[name] = {"solved_clean": ok, "solved_colliding": bad_path,
                       "unsolved": nosol,
                       "median_q_len": float(np.median(qlens))
                       if qlens else None}
        print(f"{name}: {ok} clean / {bad_path} colliding / "
              f"{nosol} no-path over {len(pairs)} random queries; "
              f"median q-length {stats[name]['median_q_len']}",
              flush=True)

    json.dump({"gen_time": {"iriszo": tz, "ciris": tc},
               "n_regions": {"iriszo": len(rz), "ciris": len(rc)},
               "iface_contam": {"iriszo": [bz, nz], "ciris": [bc, nc]},
               "interior_contam": {"iriszo": [bzi, nzi],
                                   "ciris": [bci, nci]},
               "queries": stats},
              open("out/e15_ciris.json", "w"), indent=1)
    pickle.dump({"iriszo": rz, "ciris": rc, "q_star": q_star},
                open("out/e15_regions.pkl", "wb"))
    print("written out/e15_ciris.json + regions pkl", flush=True)


if __name__ == "__main__":
    main()
