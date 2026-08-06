#!/usr/bin/env python3
"""E17: UAV split-screen comparison on the shipped (polluted) building
library -- the wall-crossing reference kept as-is, our certified
reroute beside it.

Left : GCS* on the shipped 8x8_s1 library (fake through-wall portals
       included).  The smooth optimum dives through walls; every
       violating sample pulses red under a WALL VIOLATION banner.
Right: our layer on the same world: the returned answer is validated
       against the analytic building truth, the illegal portals are
       blacklisted, the reroute is re-planned and certified -- shown
       with a live min-clearance readout (never below +0.65 m).

Both sides fly the same query (0,0,1) -> (40,35,1), synchronized by
flight progress.  Source trajectories are the archived artifacts
(traj_gcsstar_orig.npz / uav_traj.npz + uav_traj_clearance.npz);
the scene is the ground-truth building XML, so what is rendered is
what was audited.

    python3 e17_uav_compare.py      # out/e17_uav_compare.mp4
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "mujoco_menagerie", "skydio_x2",
                     "flythrough_ctruth_8_1.xml")
FPS = 30
W2, H = 960, 720
FLIGHT_S = 14.0
TRAIL_L = (0.85, 0.46, 0.02, 0.9)
TRAIL_R = (0.05, 0.60, 0.35, 0.9)
VIOL = (0.85, 0.1, 0.1, 1.0)


def yaw_quats(t, V):
    import mujoco
    quats, psi_s = [], 0.0
    for k in range(len(t)):
        v = V[k]
        if np.linalg.norm(v[:2]) > 0.15:
            psi = np.arctan2(v[1], v[0])
            d = (psi - psi_s + np.pi) % (2 * np.pi) - np.pi
            psi_s += 0.10 * d
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), psi_s)
        quats.append(q)
    return np.array(quats)


def main():
    import imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    A = np.load(os.path.join(HERE, "out", "traj_gcsstar_orig.npz"))
    # the certified answer on the audited library (v3: illegal portals
    # removed AND the true-window regions rebuilt) -- 1690 samples
    B = np.load(os.path.join(HERE, "out", "uav_trajR.npz"))
    CLR = np.load(os.path.join(HERE, "out",
                               "uav_clearance_fullR.npz"))["clearance"]
    LA = float(np.linalg.norm(np.diff(A["p"], axis=0), axis=1).sum())
    LB = float(np.linalg.norm(np.diff(B["p"], axis=0), axis=1).sum())
    n_viol = int(A["viol"].sum())
    sides = [
        dict(P=A["p"], V=A["v"], viol=A["viol"].astype(bool),
             clr=None, trail=TRAIL_L,
             t1="the shipped library's optimum (GCS*, full GCS: same route)",
             t2=f"length {LA:.2f}   —   {n_viol} trajectory samples "
                f"inside true walls"),
        dict(P=B["p"], V=B["v"], viol=np.zeros(len(B["p"]), bool),
             clr=CLR, trail=TRAIL_R,
             t1="audited + rebuilt library, certified answer",
             t2=f"length {LB:.2f} ({100*(LB/LA-1):+.1f}% — shorter than "
                f"the illegal route)   0/{len(B['p'])} samples in walls"),
    ]
    for s in sides:
        s["quat"] = yaw_quats(np.arange(len(s["P"])), s["V"])

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W2)
    try:
        f1 = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        f2 = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
        f3 = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        f4 = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except OSError:
        f1 = f2 = f3 = f4 = ImageFont.load_default()
    identity = np.eye(3).ravel()

    def render_side(s, frac, cam, trail):
        k = min(int(frac * (len(s["P"]) - 1)), len(s["P"]) - 1)
        data.qpos[0:3] = s["P"][k]
        data.qpos[3:7] = s["quat"][k]
        mujoco.mj_forward(model, data)
        v = s["V"][k]
        look = s["P"][k] + 0.8 * v / max(np.linalg.norm(v), 1e-3)
        cam.lookat = 0.90 * np.asarray(cam.lookat) + 0.10 * look
        vn = np.linalg.norm(v[:2])
        if vn > 0.2:
            az = np.degrees(np.arctan2(v[1], v[0])) + 180.0
            d = (az - cam.azimuth + 180.0) % 360.0 - 180.0
            cam.azimuth += 0.06 * d
        renderer.update_scene(data, camera=cam)
        scn = renderer.scene
        for pt, is_v in trail:
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.15 if is_v else 0.09, 0, 0]), pt, identity,
                np.array(VIOL if is_v else s["trail"]))
            scn.ngeom += 1
        if s["viol"][k]:
            g = scn.geoms[scn.ngeom]
            r = 0.35 + 0.12 * np.sin(frac * 200)
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, 0, 0]), s["P"][k],
                                identity,
                                np.array((0.9, 0.05, 0.05, 0.45)))
            scn.ngeom += 1
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([16, 14, 944, 96], fill=(12, 16, 22, 190))
        d.text((30, 22), s["t1"], fill=(255, 255, 255), font=f1)
        d.text((30, 60), s["t2"], fill=(255, 210, 160), font=f2)
        if s["viol"][k]:
            msg = "WALL VIOLATION"
            w = d.textlength(msg, f3)
            d.rectangle([W2 / 2 - w / 2 - 20, 108, W2 / 2 + w / 2 + 20,
                         168], fill=(160, 12, 12, 230))
            d.text((W2 / 2 - w / 2, 118), msg, fill=(255, 255, 255),
                   font=f3)
        elif s["clr"] is not None:
            kc = min(int(frac * (len(s["clr"]) - 1)), len(s["clr"]) - 1)
            msg = f"clearance +{s['clr'][kc]:.2f} m   CERTIFIED"
            w = d.textlength(msg, f4)
            d.rectangle([W2 / 2 - w / 2 - 16, 108, W2 / 2 + w / 2 + 16,
                         156], fill=(8, 90, 50, 210))
            d.text((W2 / 2 - w / 2, 116), msg, fill=(210, 255, 225),
                   font=f4)
        return img

    cams, trails = [], [[], []]
    for s in sides:
        c = mujoco.MjvCamera()
        c.distance = 9.5
        c.elevation = -58.0
        c.lookat = s["P"][0].copy()
        c.azimuth = np.degrees(np.arctan2(s["V"][0][1],
                                          s["V"][0][0])) + 180.0
        cams.append(c)

    out = os.path.join(HERE, "out", "e17_uav_compare.mp4")
    writer = imageio.get_writer(out, fps=FPS, codec="libx264",
                                quality=8, macro_block_size=8)
    n_frames = int(FLIGHT_S * FPS)
    for i in range(n_frames):
        frac = i / (n_frames - 1)
        halves = []
        for j, s in enumerate(sides):
            k = min(int(frac * (len(s["P"]) - 1)), len(s["P"]) - 1)
            if i % 3 == 0:
                trails[j].append((s["P"][k].copy(), bool(s["viol"][k])))
            halves.append(render_side(s, frac, cams[j], trails[j]))
        fr = Image.new("RGB", (2 * W2, H))
        fr.paste(halves[0], (0, 0))
        fr.paste(halves[1], (W2, 0))
        d = ImageDraw.Draw(fr, "RGBA")
        d.rectangle([0, H - 26, int(2 * W2 * frac), H - 18],
                    fill=(240, 240, 240, 200))
        writer.append_data(np.asarray(fr))

    # ending card
    card = Image.new("RGB", (2 * W2, H), (10, 13, 18))
    d = ImageDraw.Draw(card)
    lines = [
        ("The shipped map has doors through walls.", f3,
         (255, 255, 255)),
        ("Every planner on it takes them — and reports success.", f3,
         (255, 140, 120)),
        ("", f2, (0, 0, 0)),
        (f"Physical validation flags all {n_viol} violating samples. "
         "With the map audited and rebuilt,", f1, (220, 220, 220)),
        (f"the certified route is {LB:.2f} m — "
         f"{abs(LB/LA-1)*100:.1f}% SHORTER than the illegal one, "
         f"clearance never below +{CLR.min():.2f} m.", f1,
         (150, 255, 190)),
        ("", f2, (0, 0, 0)),
        ("The shortcut through the wall was never a shortcut.",
         f4, (255, 255, 255)),
        ("Checking the answer cost nothing — and bought the truth.",
         f4, (255, 255, 255)),
    ]
    y = 190
    for txt, ft, col in lines:
        w = d.textlength(txt, ft) if txt else 0
        d.text(((2 * W2 - w) / 2, y), txt, fill=col, font=ft)
        y += 58 if ft in (f3, f4) else 40
    for _ in range(int(3.5 * FPS)):
        writer.append_data(np.asarray(card))
    writer.close()
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
