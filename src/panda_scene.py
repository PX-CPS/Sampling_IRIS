#!/usr/bin/env python3
"""Franka Panda demo scene for iP-GCS: arm on a table facing a two-compartment
shelf. Provides the plant/checker/domain used by region generation, planning,
and the meshcat simulation.

Scene frame: panda base at world origin on the table surface (z=0); shelf in
front of the robot along +x with two openings facing it.
"""
from __future__ import annotations

import numpy as np
from pydrake.geometry import Box, Rgba
from pydrake.geometry.optimization import HPolyhedron
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker

PANDA_URDF = "package://drake_models/franka_description/urdf/panda_arm.urdf"

# obstacle boxes: (name, size_xyz, center_xyz, rgba)
SHELF_X0, SHELF_X1 = 0.55, 0.83          # front/back of the shelf
SHELF_W = 0.75                            # total width  (y)
PANEL = 0.03
OBSTACLES = [
    ("table", (1.8, 1.8, 0.10), (0.35, 0.0, -0.05), (0.55, 0.42, 0.30, 1.0)),
    ("shelf_bottom", (SHELF_X1 - SHELF_X0, SHELF_W, PANEL),
     ((SHELF_X0 + SHELF_X1) / 2, 0.0, 0.015), (0.80, 0.65, 0.45, 1.0)),
    ("shelf_mid", (SHELF_X1 - SHELF_X0, SHELF_W, PANEL),
     ((SHELF_X0 + SHELF_X1) / 2, 0.0, 0.395), (0.80, 0.65, 0.45, 1.0)),
    ("shelf_top", (SHELF_X1 - SHELF_X0, SHELF_W, PANEL),
     ((SHELF_X0 + SHELF_X1) / 2, 0.0, 0.775), (0.80, 0.65, 0.45, 1.0)),
    ("shelf_left", (SHELF_X1 - SHELF_X0, PANEL, 0.79),
     ((SHELF_X0 + SHELF_X1) / 2, SHELF_W / 2, 0.395), (0.80, 0.65, 0.45, 1.0)),
    ("shelf_right", (SHELF_X1 - SHELF_X0, PANEL, 0.79),
     ((SHELF_X0 + SHELF_X1) / 2, -SHELF_W / 2, 0.395), (0.80, 0.65, 0.45, 1.0)),
    ("shelf_back", (PANEL, SHELF_W, 0.79),
     (SHELF_X1, 0.0, 0.395), (0.80, 0.65, 0.45, 1.0)),
]

# end-effector target boxes (inside the compartments) for task-config search
EE_TARGETS = {
    "top_bin": (np.array([0.58, -0.10, 0.50]), np.array([0.74, 0.10, 0.72])),
    "bottom_bin": (np.array([0.58, -0.10, 0.10]), np.array([0.74, 0.10, 0.34])),
}
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def add_obstacles(plant):
    from pydrake.multibody.plant import CoulombFriction
    for name, size, center, rgba in OBSTACLES:
        shape = Box(*size)
        X = RigidTransform(np.asarray(center))
        plant.RegisterCollisionGeometry(plant.world_body(), X, shape,
                                        name + "_col",
                                        CoulombFriction(0.9, 0.8))
        plant.RegisterVisualGeometry(plant.world_body(), X, shape,
                                     name + "_vis", np.asarray(rgba))


def build_scene():
    """RobotDiagram + collision checker + joint-limit domain for the Panda."""
    rdb = RobotDiagramBuilder(time_step=0.0)
    parser = Parser(rdb.plant())
    (panda,) = parser.AddModels(url=PANDA_URDF)
    plant = rdb.plant()
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("panda_link0", panda))
    add_obstacles(plant)
    plant.Finalize()
    lo = plant.GetPositionLowerLimits()
    hi = plant.GetPositionUpperLimits()
    diagram = rdb.Build()
    checker = SceneGraphCollisionChecker(
        model=diagram, robot_model_instances=[panda], edge_step_size=0.05)
    domain = HPolyhedron.MakeBox(lo, hi)
    return checker, domain, np.asarray(lo), np.asarray(hi)


def make_fk(checker):
    """Forward kinematics for the end-effector (panda_link8 origin)."""
    plant = checker.plant()
    ctx = checker.plant_context()

    def fk(q):
        plant.SetPositions(ctx, np.asarray(q))
        X = plant.EvalBodyPoseInWorld(
            ctx, plant.GetBodyByName("panda_link8"))
        return X.translation()
    return fk


def find_task_configs(checker, lo, hi, rng, n_tries=20000):
    """Rejection-sample collision-free configs whose EE lands in each target
    box; return {name: q} including home."""
    fk = make_fk(checker)
    assert checker.CheckConfigCollisionFree(Q_HOME), "home config collides!"
    out = {"home": Q_HOME.copy()}
    for name, (blo, bhi) in EE_TARGETS.items():
        best = None
        for _ in range(n_tries):
            q = lo + rng.random(len(lo)) * (hi - lo)
            p = fk(q)
            if np.all(p >= blo) and np.all(p <= bhi) \
                    and checker.CheckConfigCollisionFree(q):
                best = q
                break
        if best is None:
            raise RuntimeError(f"no config found for {name}")
        out[name] = best
    return out
