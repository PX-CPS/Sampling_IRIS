#!/usr/bin/env python3
"""E9 task video: a REAL coordinated task motion -- arm 1 goes
top-shelf -> right bin while arm 2 goes left bin -> top-shelf -- planned
by our pipeline and carrying a continuous clearance certificate along
its entire length.  Companion piece to the q8 violation video: same
scene, same renderer, opposite verdict.

    python3 e9_task_video.py           # out/e9_task_certified.mp4
"""
from __future__ import annotations

import pickle
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.environ.get("GCS_SR",
                            os.path.expanduser("~/gcs-science-robotics")))

FPS = 60
W, H = 1280, 720
OUT = "out/e9_task_certified.mp4"
PAIR = ("top_shelf/bin_L", "bin_R/top_shelf")


def plan_task_path():
    """Our pipeline on the named pair: sampled interfaces -> corridor ->
    SOCP polish -> continuous physical certificate."""
    import e6_sampling as e6
    import stageB_bimanual_ipgcs as sb
    from e7_highd_audit import (HighDPlanner, bbox_ell_keys,
                                build_path_audit, polish_waypoints)
    from e8_validate import path_cost
    from reproduction.bimanual.helpers import getConfigurationSeeds

    seeds = {k: np.asarray(v) for k, v in getConfigurationSeeds().items()}
    s, g = seeds[PAIR[0]], seeds[PAIR[1]]
    topo = pickle.load(open("out/bimanual219_topo.pkl", "rb"))
    if isinstance(topo, dict) and "topo" in topo:
        topo = topo["topo"]
    ifaces = pickle.load(open("out/e7_ifaces_biman219.pkl", "rb"))
    audit_fn, checker, validator = build_path_audit("bimanual")
    ell = bbox_ell_keys(topo, ifaces, s, g)
    pl = HighDPlanner(topo, ifaces, s, g, mode="area", M=1, informed=True,
                      seed=0, ell=ell, path_audit=audit_fn,
                      validator=validator).run(8)
    assert pl.best_path is not None, "planner found no certified path"
    # polish, then re-certify the polished waypoints
    regions = topo["regions"]
    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}
    mems = [e6.region_membership(p, regions) for p in pl.best_path]
    pseq = e6.corridor_from_path(mems, pair_eid)
    path = pl.best_path
    if pseq is not None:
        rgq = sb.with_query(topo, s, g)
        c_geo, ppath = polish_waypoints(rgq, pseq, s, g)
        if ppath is not None:
            okp, _ = validator.path(ppath)
            if okp and path_cost(ppath) < path_cost(path):
                path = ppath
    ok, info = validator.path(path)
    assert ok, f"final path failed certification: {info}"
    print(f"task path: cost {path_cost(path):.3f}, "
          f"{len(path)} waypoints, CERTIFIED clean "
          f"({info['clearance_calls']} clearance checks)", flush=True)
    return np.asarray(path), checker


def main():
    import imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from e9_q8_video3 import build_model, fk_align_check

    wpath, checker = plan_task_path()
    cplant = checker.plant()

    seg = np.linalg.norm(np.diff(wpath, axis=0), axis=1)
    Lc = np.concatenate([[0], np.cumsum(seg)])
    n_pts = 420
    ts = np.linspace(0, Lc[-1], n_pts)
    qs = []
    for t in ts:
        k = min(max(int(np.searchsorted(Lc, t, "right") - 1), 0),
                len(seg) - 1)
        lam = 0.0 if seg[k] < 1e-12 else (t - Lc[k]) / seg[k]
        qs.append((1 - lam) * wpath[k] + lam * wpath[k + 1])
    qs = np.asarray(qs)

    phi = []
    for q in qs:
        rc = checker.CalcRobotClearance(q, 0.5)
        d = np.asarray(rc.distances())
        phi.append(float(d.min()) if d.size else 0.5)
    phi = np.asarray(phi)
    print(f"clearance along the motion: min {1000*phi.min():.1f} mm",
          flush=True)
    assert phi.min() > 0

    m, _ = build_model(checker)
    worst = fk_align_check(checker, m, qs[[0, n_pts // 2, -1]])
    print(f"FK alignment: worst {1000*worst:.2f} mm")
    assert worst < 0.005

    mdata = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    overview = np.array([0.35, 0.25, 0.45])

    def render(pi):
        mdata.qpos[:14] = qs[pi]
        mujoco.mj_forward(m, mdata)
        prog = pi / n_pts
        cam.lookat = overview
        cam.distance = 2.9 - 0.35 * np.sin(np.pi * prog)
        cam.azimuth = 135 + 55 * prog
        cam.elevation = -20
        renderer.update_scene(mdata, camera=cam)
        return renderer.render()

    try:
        font_b = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        font_s = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font_b = font_s = ImageFont.load_default()

    def compose(pi, cap=None):
        img = Image.fromarray(render(pi))
        dr = ImageDraw.Draw(img)
        dr.text((28, 20), "task query, certified answer -- 14-DOF "
                "bimanual", fill=(20, 20, 25), font=font_b)
        dr.text((28, 62), "arm 1: top shelf to right bin   |   arm 2: "
                "left bin to top shelf", fill=(50, 50, 60), font=font_s)
        c_mm = 1000 * phi[pi]
        col = (20, 130, 40) if c_mm > 10 else (200, 130, 0)
        dr.text((28, H - 64), f"clearance  +{c_mm:6.1f} mm", fill=col,
                font=font_b)
        dr.text((28, H - 100), cap or "continuous clearance certificate "
                "holds along the entire motion", fill=(20, 130, 40),
                font=font_s)
        return img

    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8)
    for _ in range(int(1.0 * FPS)):
        writer.append_data(np.asarray(compose(0, "start: arm 1 at the "
                                              "top shelf, arm 2 at the "
                                              "left bin")))
    for pi in np.repeat(np.arange(n_pts), 2):
        writer.append_data(np.asarray(compose(pi)))
    for _ in range(int(1.5 * FPS)):
        writer.append_data(np.asarray(compose(n_pts - 1, "goal: arms "
                                              "exchanged their "
                                              "stations -- zero "
                                              "collision, certified")))
    writer.close()
    n_tot = int(1.0 * FPS) + 2 * n_pts + int(1.5 * FPS)
    print(f"written {OUT} ({n_tot} frames, {n_tot/FPS:.1f}s)")


if __name__ == "__main__":
    main()
