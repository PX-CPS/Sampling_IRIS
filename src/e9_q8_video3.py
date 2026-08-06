#!/usr/bin/env python3
"""E9 presentation video, v3: menagerie KUKA iiwa14 meshes (the pretty
ones), WSG grippers injected from the planning scene's own collision
boxes, translucent shelf so the buried gripper is visible.

Truth chain unchanged: the path is the bit-exact GCS* reference answer;
clearance numbers come from the planning scene's collision checker; an
FK alignment assert (Drake vs MuJoCo, same configs) guards against
frame mismatches before a single frame is rendered.

    python3 e9_q8_video3.py --test-frame 0.02
    python3 e9_q8_video3.py            # out/e9_q8_violation_hd.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from e9_q8_video import q8_exact_path

FPS = 60
W, H = 1280, 720
OUT = "out/e9_q8_violation_hd.mp4"
XML = (os.environ.get("MENAGERIE",
       os.path.expanduser("~/mujoco_menagerie")) + "/kuka_iiwa_14/"
       "bimanual_race.xml")
PREFIX = {"iiwa_1": "r1_", "wsg_1": "r1_", "iiwa_2": "r2_", "wsg_2": "r2_"}
RED = np.array([0.90, 0.12, 0.10, 1.0])


def wsg_boxes(checker):
    """Per arm: the gripper's collision boxes with poses in the link_7
    frame (welded, hence constant)."""
    from pydrake.geometry import Box
    plant = checker.plant()
    diagram = checker.model()
    insp = diagram.GetSubsystemByName("scene_graph").model_inspector()
    ctx = plant.CreateDefaultContext()
    out = {"wsg_1": [], "wsg_2": []}
    l7 = {"wsg_1": plant.GetBodyByName(
              "iiwa_link_7", plant.GetModelInstanceByName("iiwa_1")),
          "wsg_2": plant.GetBodyByName(
              "iiwa_link_7", plant.GetModelInstanceByName("iiwa_2"))}
    for gid in insp.GetAllGeometryIds():
        if insp.GetProximityProperties(gid) is None:
            continue
        body = plant.GetBodyFromFrameId(insp.GetFrameId(gid))
        model = plant.GetModelInstanceName(body.model_instance())
        if model not in out:
            continue
        shape = insp.GetShape(gid)
        if not isinstance(shape, Box):
            continue
        X_WG = plant.EvalBodyPoseInWorld(ctx, body) @ \
            insp.GetPoseInFrame(gid)
        X_WL7 = plant.EvalBodyPoseInWorld(ctx, l7[model])
        X_L7G = X_WL7.inverse() @ X_WG
        q = X_L7G.rotation().ToQuaternion()
        out[model].append({
            "bname": body.name(),
            "half": (shape.width() / 2, shape.depth() / 2,
                     shape.height() / 2),
            "pos": X_L7G.translation(),
            "quat": (q.w(), q.x(), q.y(), q.z())})
    return out


def build_model(checker):
    import mujoco
    spec = mujoco.MjSpec.from_file(XML)
    # translucent shelf so a buried gripper stays visible
    for g in spec.geoms:
        if g.name.startswith("shelf"):
            g.rgba[3] = 0.45
    boxes = wsg_boxes(checker)
    for model, items in boxes.items():
        p = PREFIX[model]
        body = spec.body(f"{p}link7")
        for k, it in enumerate(items):
            gname = f"{p}wsg_{it['bname']}_{k}"
            g = body.add_geom()
            g.name = gname
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size[:] = it["half"]
            g.pos[:] = it["pos"]
            g.quat[:] = it["quat"]
            g.rgba[:] = (0.42, 0.45, 0.50, 1.0)
            g.contype = 0
            g.conaffinity = 0
    m = spec.compile()
    return m, boxes


def fk_align_check(checker, m, qs):
    """Drake vs MuJoCo world positions of both grippers at test configs."""
    import mujoco
    plant = checker.plant()
    ctx = plant.CreateDefaultContext()
    d = mujoco.MjData(m)
    worst = 0.0
    for q in qs:
        plant.SetPositions(ctx, q)
        d.qpos[:14] = q
        mujoco.mj_forward(m, d)
        for model in ("wsg_1", "wsg_2"):
            b = plant.GetBodyByName("body",
                                    plant.GetModelInstanceByName(model))
            p_drake = plant.EvalBodyPoseInWorld(ctx, b).translation()
            gname = f"{PREFIX[model]}wsg_body_0"
            p_mj = d.geom(gname).xpos
            worst = max(worst, float(np.linalg.norm(p_drake - p_mj)))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frame", type=float, default=None)
    args = ap.parse_args()
    import imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from pydrake.multibody.tree import BodyIndex

    wpath, stored = q8_exact_path()
    from stageB_bimanual_regions import build
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
    phi, pairs, wsg2_pos = [], [], []
    for q in qs:
        rc = checker.CalcRobotClearance(q, 0.5)
        dd = np.asarray(rc.distances())
        cplant.SetPositions(cctx, q)
        wsg2_pos.append(cplant.EvalBodyPoseInWorld(cctx, wsg2b)
                        .translation().copy())
        if dd.size == 0:
            phi.append(0.5)
            pairs.append(None)
            continue
        k = int(np.argmin(dd))
        phi.append(float(dd[k]))
        b1 = cplant.get_body(BodyIndex(int(rc.robot_indices()[k])))
        b2 = cplant.get_body(BodyIndex(int(rc.other_indices()[k])))
        pairs.append((
            (cplant.GetModelInstanceName(b1.model_instance()), b1.name()),
            (cplant.GetModelInstanceName(b2.model_instance()), b2.name())))
    phi = np.asarray(phi)
    wsg2_pos = np.asarray(wsg2_pos)

    run_caption = {}
    i0 = 0
    while i0 < n_pts:
        if phi[i0] < 0:
            j = i0
            while j < n_pts and phi[j] < 0:
                j += 1
            k = i0 + int(np.argmin(phi[i0:j]))
            models = {m for m, _ in pairs[k]}
            if i0 == 0:
                cap = ("benchmark START configuration is inside the "
                       "shelf (sampled from a contaminated region)")
            elif models & {"shelves", "table", "binR", "binL"}:
                cap = "the optimal path dives into the furniture"
            else:
                cap = ("the optimal path drives the two arms THROUGH "
                       "each other")
            for t in range(i0, j):
                run_caption[t] = cap
            i0 = j
        else:
            i0 += 1

    m, _ = build_model(checker)
    worst = fk_align_check(checker, m, qs[[0, n_pts // 2, -1]])
    print(f"FK alignment Drake vs MuJoCo: worst {1000*worst:.2f} mm")
    assert worst < 0.005, "frame mismatch between Drake and MuJoCo"

    mdata = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=H, width=W)
    base_rgba = m.geom_rgba.copy()

    # colliding entity -> geom ids for the red flash
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
            else:                       # furniture: shelf*/bin*/table
                ids += [gi for gi in range(m.ngeom)
                        if (m.geom(gi).name or "").startswith(
                            "shelf" if model == "shelves" else model)]
        return ids

    overview = np.array([0.35, 0.25, 0.45])
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
        cam.lookat = (1 - 0.72 * z) * overview + 0.72 * z * wsg2_pos[pi]
        cam.distance = 3.2 - 1.9 * z
        cam.azimuth = 150 + 24 * prog
        cam.elevation = -22 - 5 * z
        renderer.update_scene(mdata, camera=cam)
        return renderer.render()

    try:
        font_b = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        font_s = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font_b = font_s = ImageFont.load_default()

    def compose(pi, z, hold_caption=None):
        img = Image.fromarray(render(pi, z))
        dr = ImageDraw.Draw(img)
        c_mm = 1000 * phi[pi]
        if c_mm < 0:
            for wdt in range(10):
                dr.rectangle([wdt, wdt, W - 1 - wdt, H - 1 - wdt],
                             outline=(214, 40, 30))
        dr.text((28, 20), "GCS* reference answer, 14-DOF bimanual "
                "benchmark q8", fill=(20, 20, 25), font=font_b)
        dr.text((28, 62), f"reported cost {stored:.4f}  |  clearance "
                "measured by the scene's own collision checker",
                fill=(50, 50, 60), font=font_s)
        if c_mm >= 0:
            col = (20, 130, 40) if c_mm > 10 else (200, 130, 0)
            dr.text((28, H - 64), f"clearance  +{c_mm:6.1f} mm",
                    fill=col, font=font_b)
        else:
            dr.text((28, H - 64), f"PENETRATION  {c_mm:6.1f} mm",
                    fill=(214, 30, 24), font=font_b)
            dr.text((28, H - 100), run_caption.get(pi, "in collision"),
                    fill=(214, 30, 24), font=font_s)
        if hold_caption:
            dr.text((28, H - 100), hold_caption, fill=(20, 130, 40),
                    font=font_s)
        return img

    if args.test_frame is not None:
        pi = int(args.test_frame * (n_pts - 1))
        z = 1.0 if phi[pi] < 0.012 else 0.0
        compose(pi, z).save("out/e9_q8_hd_test.png")
        print(f"test frame ({1000*phi[pi]:.1f} mm) -> "
              "out/e9_q8_hd_test.png")
        return

    dwell = np.where(phi < 0, 7, np.where(phi < 0.01, 3, 2))
    frames_idx = np.repeat(np.arange(n_pts), dwell)
    z_target = (phi < 0.012).astype(float)
    z_s, z = [], 0.0
    for pi in frames_idx:
        z += (0.16 if z_target[pi] > z else 0.012) * (z_target[pi] - z)
        z_s.append(z)

    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8)
    for _ in range(int(1.2 * FPS)):
        writer.append_data(np.asarray(compose(0, 1.0)))
    for fi, pi in enumerate(frames_idx):
        writer.append_data(np.asarray(compose(pi, z_s[fi])))
    for _ in range(int(1.8 * FPS)):
        writer.append_data(np.asarray(compose(
            n_pts - 1, 0.0,
            hold_caption="goal reached -- clean stretches are real, but "
            "three collision intervals sit between them")))
    card = Image.new("RGB", (W, H), (18, 20, 26))
    dr = ImageDraw.Draw(card)
    for y, (txt, col) in enumerate([
            ("the reference answer of query 8:", (235, 235, 240)),
            ("", None),
            ("   start config 63.5 mm inside the shelf", (240, 90, 80)),
            ("   arms through each other, 41.7 mm deep", (240, 90, 80)),
            ("   arms through each other, 56.2 mm deep", (240, 90, 80)),
            ("", None),
            ("14.4% of the path length is in collision.",
             (235, 235, 240)),
            ("Every planner without a physical layer returns it",
             (200, 200, 210)),
            ("and scores it as optimal.", (200, 200, 210))]):
        if col:
            dr.text((110, 150 + 52 * y), txt, fill=col, font=font_b)
    for _ in range(int(3.0 * FPS)):
        writer.append_data(np.asarray(card))
    writer.close()
    n_tot = int(1.2 * FPS) + len(frames_idx) + int(1.8 * FPS) \
        + int(3.0 * FPS)
    print(f"written {OUT} ({n_tot} frames, {n_tot/FPS:.1f}s)")


if __name__ == "__main__":
    main()
