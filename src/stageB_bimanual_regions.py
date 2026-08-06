#!/usr/bin/env python3
"""Stage B: generate bimanual (14-DOF) IRIS regions, headless, Drake >= 1.5x.

The paper repo used IrisInConfigurationSpace (removed in modern Drake); this
uses its successor IrisZo (sampled, certified) with SceneGraphCollisionChecker.
Same models, same seeds, same collision-filter declarations (applied at MODEL
level via a shim so checker-created contexts inherit them).

Output: out/bimanual_regions.pkl {seed_name: (A, b)}.
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

for p in (os.environ.get("GCS_SR", ""),
          os.path.expanduser("~/gcs-science-robotics")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
from pydrake.multibody.parsing import LoadModelDirectives, ProcessModelDirectives
from pydrake.planning import (IrisZo, IrisZoOptions, RobotDiagramBuilder,
                              SceneGraphCollisionChecker)

from reproduction.bimanual.helpers import (filterCollsionGeometry,
                                           getConfigurationSeeds)
from reproduction.util import FindModelFile, GcsDir

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "out", "bimanual_regions.pkl")


class _ModelLevelSG:
    """Shim: route the helper's context-scoped filter manager to the
    model-level one so filters survive into checker-created contexts;
    forward everything else the helper touches to the real SceneGraph."""

    def __init__(self, sg):
        self._sg = sg

    def collision_filter_manager(self, _context=None):
        return self._sg.collision_filter_manager()

    def model_inspector(self):
        return self._sg.model_inspector()

    def __getattr__(self, name):
        return getattr(self._sg, name)


def build():
    rdb = RobotDiagramBuilder(time_step=0.0)
    parser = rdb.parser()
    parser.package_map().Add("gcs", GcsDir())
    directives = LoadModelDirectives(FindModelFile("models/bimanual_iiwa.yaml"))
    ProcessModelDirectives(directives, rdb.plant(), parser)
    rdb.plant().Finalize()
    plant = rdb.plant()
    robots = [plant.GetModelInstanceByName(n)
              for n in ("iiwa_1", "wsg_1", "iiwa_2", "wsg_2")]
    lo = plant.GetPositionLowerLimits()
    hi = plant.GetPositionUpperLimits()
    diagram = rdb.Build()
    ctx = diagram.CreateDefaultContext()
    sg = diagram.scene_graph()
    sg_ctx = sg.GetMyContextFromRoot(ctx)
    # Apply the paper's filter declarations at MODEL level (shim), while the
    # helper's diagnostics Eval on a real context.
    filterCollsionGeometry(_ModelLevelSG(sg), sg_ctx)
    checker = SceneGraphCollisionChecker(
        model=diagram, robot_model_instances=robots, edge_step_size=0.125)
    domain = HPolyhedron.MakeBox(lo, hi)
    return checker, domain


def main():
    checker, domain = build()
    seeds = getConfigurationSeeds()
    opts = IrisZoOptions()
    part_dir = OUT.replace(".pkl", "_parts")
    os.makedirs(part_dir, exist_ok=True)
    regions = {}
    for i, (name, q) in enumerate(seeds.items()):
        part = os.path.join(part_dir, f"{name.replace('/', '_')}.pkl")
        if os.path.exists(part):
            regions[name] = pickle.load(open(part, "rb"))
            print(f"[{i+1}/{len(seeds)}] {name}: cached", flush=True)
            continue
        t0 = time.time()
        try:
            q = np.asarray(q, dtype=float)
            # clamp into the domain interior: seeds on/beyond joint limits
            # make IrisZo abort (DRAKE_DEMAND, uncatchable)
            hb = domain.b()
            d = domain.ambient_dimension()
            hi_lim, lo_lim = hb[:d], -hb[d:]
            eps = 1e-4 * (hi_lim - lo_lim)
            q = np.clip(q, lo_lim + eps, hi_lim - eps)
            ell = Hyperellipsoid.MakeHypersphere(1e-2, q)
            hp = IrisZo(checker, ell, domain, opts)
            regions[name] = (hp.A(), hp.b())
            pickle.dump(regions[name], open(part, "wb"))
            print(f"[{i+1}/{len(seeds)}] {name}: {len(hp.b())} faces, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(seeds)}] {name} FAILED: {e}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(regions, f)
    print(f"saved {len(regions)}/{len(seeds)} regions -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
