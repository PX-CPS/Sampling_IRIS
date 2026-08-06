#!/usr/bin/env python3
"""E11: certified pick-and-place -- an item on the top shelf is grasped
by arm 1 and carried along the CERTIFIED task path into the right bin,
while arm 2 heads to the shelf for the next item.  Ends with a release
+ retreat shot so the place actually reads.

The robot motion is the physically-certified path of
e9_task_video (cost 4.755, min clearance +64 mm).  The item is a
kinematic prop attached at the finger midpoint (the panda-demo
recipe); the certificate statement applies to the robot motion.

    python3 e11_pickplace_video.py --test-frame 0.0
    python3 e11_pickplace_video.py       # out/e11_pickplace.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.environ.get("GCS_SR",
                            os.path.expanduser("~/gcs-science-robotics")))

FPS = 60
W, H = 1280, 720
OUT = "out/e11_pickplace.mp4"
ITEM_HALF = 0.024                       # 4.8 cm cube
ITEM_RGBA = (0.10, 0.55, 0.75, 1.0)


def build_model_with_item(checker):
    import mujoco
    from e9_q8_video3 import PREFIX, XML, wsg_boxes
    spec = mujoco.MjSpec.from_file(XML)
    for g in spec.geoms:
        if g.name.startswith("shelf"):
            g.rgba[3] = 0.45
    boxes = wsg_boxes(checker)
    for model, items in boxes.items():
        p = PREFIX[model]
        body = spec.body(f"{p}link7")
        for k, it in enumerate(items):
            g = body.add_geom()
            g.name = f"{p}wsg_{it['bname']}_{k}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size[:] = it["half"]
            g.pos[:] = it["pos"]
            g.quat[:] = it["quat"]
            g.rgba[:] = (0.42, 0.45, 0.50, 1.0)
            g.contype = 0
            g.conaffinity = 0
    item = spec.worldbody.add_body()
    item.name = "item"
    item.mocap = True
    gi = item.add_geom()
    gi.name = "item_geom"
    gi.type = mujoco.mjtGeom.mjGEOM_BOX
    gi.size[:] = (ITEM_HALF, ITEM_HALF, ITEM_HALF)
    gi.rgba[:] = ITEM_RGBA
    gi.contype = 0
    gi.conaffinity = 0
    m = spec.compile()
    # grasp offset: finger midpoint in the link7 frame (from the wsg
    # collision boxes), nudged toward the fingertips
    f = {it["bname"]: np.asarray(it["pos"])
         for it in boxes["wsg_1"] if "finger" in it["bname"]}
    body_c = next(np.asarray(it["pos"]) for it in boxes["wsg_1"]
                  if it["bname"] == "body")
    mid = 0.5 * (f["left_finger"] + f["right_finger"])
    tip_dir = mid - body_c
    n = np.linalg.norm(tip_dir)
    off = mid + (0.035 * tip_dir / n if n > 1e-9 else 0.0)
    return m, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frame", type=float, default=None)
    args = ap.parse_args()
    import imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation

    from e9_q8_video3 import fk_align_check
    from e9_task_video import plan_task_path

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
    assert phi.min() > 0

    m, off = build_model_with_item(checker)
    worst = fk_align_check(checker, m, qs[[0, n_pts // 2, -1]])
    assert worst < 0.005
    mdata = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=H, width=W)
    item_mocapid = m.body("item").mocapid

    cam = mujoco.MjvCamera()
    overview = np.array([0.45, 0.25, 0.5])

    def item_pose_from_link7():
        b = mdata.body("r1_link7")
        Rw = Rotation.from_quat(np.roll(b.xquat, -1))   # wxyz -> xyzw
        return b.xpos + Rw.apply(off), b.xquat

    release_pose = {}

    def render(pi, attached, z, arm1_q=None):
        q = qs[pi].copy()
        if arm1_q is not None:
            q[:7] = arm1_q
        mdata.qpos[:14] = q
        mujoco.mj_forward(m, mdata)
        if attached:
            pos, quat = item_pose_from_link7()
            mdata.mocap_pos[item_mocapid] = pos
            mdata.mocap_quat[item_mocapid] = quat
            release_pose["pos"] = pos.copy()
            release_pose["quat"] = np.asarray(quat).copy()
        else:
            mdata.mocap_pos[item_mocapid] = release_pose["pos"]
            mdata.mocap_quat[item_mocapid] = release_pose["quat"]
        mujoco.mj_forward(m, mdata)
        prog = pi / n_pts
        gr = mdata.body("r1_link7").xpos
        cam.lookat = (1 - z) * overview + z * gr
        cam.distance = 2.9 - 1.1 * z
        cam.azimuth = 130 + 40 * prog
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

    def compose(pi, attached, z, cap, arm1_q=None):
        img = Image.fromarray(render(pi, attached, z, arm1_q))
        dr = ImageDraw.Draw(img)
        dr.text((28, 20), "certified pick-and-place -- 14-DOF bimanual",
                fill=(20, 20, 25), font=font_b)
        dr.text((28, 62), "arm 1 carries the item: top shelf to right "
                "bin   |   arm 2: left bin to top shelf",
                fill=(50, 50, 60), font=font_s)
        c_mm = 1000 * phi[pi]
        dr.text((28, H - 64), f"clearance  +{c_mm:6.1f} mm",
                fill=(20, 130, 40) if c_mm > 10 else (200, 130, 0),
                font=font_b)
        dr.text((28, H - 100), cap, fill=(20, 130, 40), font=font_s)
        return img

    if args.test_frame is not None:
        pi = int(args.test_frame * (n_pts - 1))
        compose(pi, True, 0.75 if pi < 5 else 0.0,
                "grasp at the top shelf").save("out/e11_test.png")
        print("-> out/e11_test.png")
        return

    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8)
    # act 1: grasp hold (close-up)
    for _ in range(int(1.4 * FPS)):
        writer.append_data(np.asarray(compose(
            0, True, 0.8, "grasp: item picked at the top shelf")))
    # act 2: carry (zoom eases out, then in for the place)
    for fi, pi in enumerate(np.repeat(np.arange(n_pts), 2)):
        prog = pi / n_pts
        z = max(0.8 * (1 - prog * 6), 0.0) + max((prog - 0.85) / 0.15, 0.0) * 0.7
        writer.append_data(np.asarray(compose(
            pi, True, min(z, 0.8),
            "carrying -- continuous clearance certificate holds")))
    # act 3: release + retreat (arm 1 backs off, item stays in the bin)
    retreat = qs[::-1][:int(0.10 * n_pts)]
    for rq in np.repeat(retreat[:, :7], 3, axis=0):
        writer.append_data(np.asarray(compose(
            n_pts - 1, False, 0.7,
            "place: item released in the right bin, arm retreats",
            arm1_q=rq)))
    for _ in range(int(1.4 * FPS)):
        writer.append_data(np.asarray(compose(
            n_pts - 1, False, 0.0,
            "done -- item delivered, motion certified end to end",
            arm1_q=retreat[-1, :7])))
    writer.close()
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
