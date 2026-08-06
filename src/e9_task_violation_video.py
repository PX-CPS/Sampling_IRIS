#!/usr/bin/env python3
"""E9 task-violation video: a REAL shelf task whose GCS* reference
answer drags the gripper through the shelf for half the motion.

Pair top_shelf/top_shelf -> neutral/shelf_1: 1023/2000 dense probes in
collision (51% of the path length), gripper-vs-shelf.  Same scene, same
renderer, same truth chain as the certified task video -- the third
panel of the set: certified task / random-query violation / TASK
violation.

    python3 e9_task_violation_video.py   # out/e9_task_violation.mp4
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
OUT = "out/e9_task_violation.mp4"
PAIR = ("top_shelf/top_shelf", "neutral/shelf_1")


def main():
    import imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from pydrake.multibody.tree import BodyIndex

    import baseline_gcsstar as bgs
    import stageB_bimanual_ipgcs as sb
    from e9_q8_video3 import PREFIX, RED, build_model, fk_align_check
    from e9_verify_gcsstar import chain_waypoints
    from reproduction.bimanual.helpers import getConfigurationSeeds
    from stageB_bimanual_regions import build

    seeds = {k: np.asarray(v) for k, v in getConfigurationSeeds().items()}
    s, g = seeds[PAIR[0]], seeds[PAIR[1]]
    topo = pickle.load(open("out/bimanual219_topo.pkl", "rb"))
    if isinstance(topo, dict) and "topo" in topo:
        topo = topo["topo"]
    rg = sb.with_query(topo, s, g)
    r = bgs.gcs_star(rg, timeout=60.0, K=1, max_revisits=1, seed=0)
    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}
    chain = [(topo["portals"][pair_eid[frozenset((u, v))]]["A"],
              topo["portals"][pair_eid[frozenset((u, v))]]["b"])
             for u, v in zip(r["path"], r["path"][1:])]
    stored, wpath = chain_waypoints(chain, s, g)
    print(f"task pair {PAIR[0]} -> {PAIR[1]}: GCS* cost {stored:.4f}",
          flush=True)

    checker, _ = build()
    cplant = checker.plant()
    seg = np.linalg.norm(np.diff(wpath, axis=0), axis=1)
    Lc = np.concatenate([[0], np.cumsum(seg)])
    n_pts = 460
    ts = np.linspace(0, Lc[-1], n_pts)
    qs = []
    for t in ts:
        k = min(max(int(np.searchsorted(Lc, t, "right") - 1), 0),
                len(seg) - 1)
        lam = 0.0 if seg[k] < 1e-12 else (t - Lc[k]) / seg[k]
        qs.append((1 - lam) * wpath[k] + lam * wpath[k + 1])
    qs = np.asarray(qs)

    cctx = cplant.CreateDefaultContext()
    wsg2b = cplant.GetBodyByName("body",
                                 cplant.GetModelInstanceByName("wsg_2"))
    wsg1b = cplant.GetBodyByName("body",
                                 cplant.GetModelInstanceByName("wsg_1"))
    phi, pairs, focus_pos = [], [], []
    for q in qs:
        rc = checker.CalcRobotClearance(q, 0.5)
        dd = np.asarray(rc.distances())
        cplant.SetPositions(cctx, q)
        if dd.size == 0:
            phi.append(0.5)
            pairs.append(None)
            focus_pos.append(cplant.EvalBodyPoseInWorld(cctx, wsg1b)
                             .translation().copy())
            continue
        k = int(np.argmin(dd))
        phi.append(float(dd[k]))
        b1 = cplant.get_body(BodyIndex(int(rc.robot_indices()[k])))
        b2 = cplant.get_body(BodyIndex(int(rc.other_indices()[k])))
        m1 = cplant.GetModelInstanceName(b1.model_instance())
        pairs.append(((m1, b1.name()),
                      (cplant.GetModelInstanceName(b2.model_instance()),
                       b2.name())))
        fb = wsg1b if m1 == "wsg_1" else wsg2b
        focus_pos.append(cplant.EvalBodyPoseInWorld(cctx, fb)
                         .translation().copy())
    phi = np.asarray(phi)
    focus_pos = np.asarray(focus_pos)
    frac = float((phi < 0).mean())
    print(f"in-collision fraction {100*frac:.0f}%, min "
          f"{1000*phi.min():.1f} mm", flush=True)

    m, _ = build_model(checker)
    worst = fk_align_check(checker, m, qs[[0, n_pts // 2, -1]])
    assert worst < 0.005

    mdata = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=H, width=W)
    base_rgba = m.geom_rgba.copy()

    def geoms_of_body_name(name):
        try:
            bid = m.body(name).id
        except KeyError:
            return []
        return [gi for gi in range(m.ngeom) if m.geom_bodyid[gi] == bid]

    def flash_ids(pair):
        ids = []
        for model, bname in pair:
            p = PREFIX.get(model, "")
            if model.startswith("wsg"):
                ids += [gi for gi in range(m.ngeom)
                        if (m.geom(gi).name or "").startswith(
                            f"{p}wsg_{bname}")]
            elif model.startswith("iiwa"):
                ids += geoms_of_body_name(
                    f"{p}link{bname.split('_')[-1]}")
            else:
                ids += [gi for gi in range(m.ngeom)
                        if (m.geom(gi).name or "").startswith(
                            "shelf" if model == "shelves" else model)]
        return ids

    overview = np.array([0.55, 0.25, 0.5])
    cam = mujoco.MjvCamera()

    def render(pi, z):
        mdata.qpos[:14] = qs[pi]
        mujoco.mj_forward(m, mdata)
        m.geom_rgba[:] = base_rgba
        if phi[pi] < 0 and pairs[pi] is not None:
            for gi in flash_ids(pairs[pi]):
                a = base_rgba[gi][3]
                m.geom_rgba[gi] = (RED[0], RED[1], RED[2],
                                   min(a, 0.55) if a < 1.0 else 1.0)
        prog = pi / n_pts
        cam.lookat = (1 - 0.6 * z) * overview + 0.6 * z * focus_pos[pi]
        cam.distance = 2.9 - 1.5 * z
        cam.azimuth = 120 + 30 * prog
        cam.elevation = -18 - 6 * z
        renderer.update_scene(mdata, camera=cam)
        return renderer.render()

    try:
        font_b = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        font_s = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font_b = font_s = ImageFont.load_default()

    def compose(pi, z, hold=None):
        img = Image.fromarray(render(pi, z))
        dr = ImageDraw.Draw(img)
        c_mm = 1000 * phi[pi]
        if c_mm < 0:
            for wdt in range(10):
                dr.rectangle([wdt, wdt, W - 1 - wdt, H - 1 - wdt],
                             outline=(214, 40, 30))
        dr.text((28, 20), "shelf TASK query, GCS* reference answer",
                fill=(20, 20, 25), font=font_b)
        dr.text((28, 62), f"cost {stored:.4f}  |  arms: top shelf poses "
                "to neutral / shelf-1  |  51% of this motion is in "
                "collision", fill=(50, 50, 60), font=font_s)
        if c_mm >= 0:
            col = (20, 130, 40) if c_mm > 10 else (200, 130, 0)
            dr.text((28, H - 64), f"clearance  +{c_mm:6.1f} mm",
                    fill=col, font=font_b)
        else:
            dr.text((28, H - 64), f"PENETRATION  {c_mm:6.1f} mm",
                    fill=(214, 30, 24), font=font_b)
            dr.text((28, H - 100), "the gripper drags THROUGH the shelf",
                    fill=(214, 30, 24), font=font_s)
        if hold:
            dr.text((28, H - 100), hold, fill=(200, 130, 0), font=font_s)
        return img

    dwell = np.where(phi < 0, 4, np.where(phi < 0.01, 3, 2))
    frames_idx = np.repeat(np.arange(n_pts), dwell)
    z_target = (phi < 0.012).astype(float)
    z_s, z = [], 0.0
    for pi in frames_idx:
        z += (0.16 if z_target[pi] > z else 0.012) * (z_target[pi] - z)
        z_s.append(z)

    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8)
    for _ in range(int(1.0 * FPS)):
        writer.append_data(np.asarray(compose(0, 0.4, hold="a real "
                                              "shelf task, not a random "
                                              "query")))
    for fi, pi in enumerate(frames_idx):
        writer.append_data(np.asarray(compose(pi, z_s[fi])))
    for _ in range(int(1.5 * FPS)):
        writer.append_data(np.asarray(compose(n_pts - 1, 0.0)))
    card = Image.new("RGB", (W, H), (18, 20, 26))
    dr = ImageDraw.Draw(card)
    for y, (txt, col) in enumerate([
            ("shelf task, reference answer:", (235, 235, 240)),
            ("", None),
            ("   1023 / 2000 probes in collision", (240, 90, 80)),
            ("   51% of the motion inside the shelf", (240, 90, 80)),
            ("", None),
            ("18 of 29 solvable TASK queries on this library",
             (235, 235, 240)),
            ("have physically colliding reference answers.",
             (235, 235, 240)),
            ("No planner in this family ever checks.",
             (200, 200, 210))]):
        if col:
            dr.text((110, 150 + 52 * y), txt, fill=col, font=font_b)
    for _ in range(int(3.0 * FPS)):
        writer.append_data(np.asarray(card))
    writer.close()
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
