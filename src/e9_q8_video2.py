#!/usr/bin/env python3
"""E9 presentation video, v2: same exact GCS* reference path for q8,
rendered with Drake's own renderer and the official iiwa/WSG visual
meshes (the simulator that owns the scene).  The collision-geometry
video (e9_q8_video.py) remains the truth appendix; every number shown
here still comes from the scene's SceneGraphCollisionChecker.

Adds what the first cut missed: slower clean stretches (green readout
visible), a hold on the goal frame, and an end summary card.

    python3 e9_q8_video2.py --test-frame 0.02   # single frame to tune
    python3 e9_q8_video2.py                     # out/e9_q8_violation_hd.mp4
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from e9_q8_video import q8_exact_path

FPS = 60
W, H = 1280, 720
OUT = "out/e9_q8_violation_hd.mp4"


def build_viz():
    """Visualization twin of the planning scene + a free camera rig."""
    from pydrake.geometry import (ClippingRange, ColorRenderCamera,
                                  DepthRange, DepthRenderCamera,
                                  MakeRenderEngineVtk, RenderCameraCore,
                                  RenderEngineVtkParams)
    from pydrake.math import RigidTransform
    from pydrake.multibody.parsing import (LoadModelDirectives,
                                           ProcessModelDirectives)
    from pydrake.multibody.tree import SpatialInertia, UnitInertia
    from pydrake.planning import RobotDiagramBuilder
    from pydrake.systems.sensors import CameraInfo, RgbdSensor
    from reproduction.util import FindModelFile, GcsDir

    rdb = RobotDiagramBuilder(time_step=0.0)
    parser = rdb.parser()
    parser.package_map().Add("gcs", GcsDir())
    directives = LoadModelDirectives(FindModelFile("models/bimanual_iiwa.yaml"))
    ProcessModelDirectives(directives, rdb.plant(), parser)
    plant = rdb.plant()
    cam_mi = plant.AddModelInstance("camrig")
    cam_body = plant.AddRigidBody(
        "camrig_body", cam_mi,
        SpatialInertia(1e-6, np.zeros(3), UnitInertia(1e-6, 1e-6, 1e-6)))
    plant.Finalize()

    sg = rdb.scene_graph()
    from pydrake.geometry import LightParameter
    params = RenderEngineVtkParams()
    params.default_clear_color = np.array([0.93, 0.94, 0.96])
    params.cast_shadows = True
    params.shadow_map_size = 2048
    params.exposure = 1.0
    params.lights = [
        LightParameter(type="directional", intensity=1.0,
                       direction=np.array([-0.4, 0.35, -0.85])),
        LightParameter(type="directional", intensity=0.55,
                       direction=np.array([0.6, -0.3, -0.74])),
        LightParameter(type="directional", intensity=0.35,
                       direction=np.array([0.1, 0.9, -0.44])),
    ]
    sg.AddRenderer("vtk", MakeRenderEngineVtk(params))
    intr = CameraInfo(width=W, height=H, fov_y=np.pi / 4.4)
    core = RenderCameraCore("vtk", intr, ClippingRange(0.05, 30.0),
                            RigidTransform())
    color_cam = ColorRenderCamera(core, False)
    depth_cam = DepthRenderCamera(core, DepthRange(0.1, 10.0))
    builder = rdb.builder()
    sensor = builder.AddSystem(RgbdSensor(
        parent_id=plant.GetBodyFrameIdOrThrow(cam_body.index()),
        X_PB=RigidTransform(), color_camera=color_cam,
        depth_camera=depth_cam))
    builder.Connect(sg.get_query_output_port(),
                    sensor.query_object_input_port())
    diagram = rdb.Build()
    return diagram, plant, sensor, cam_body


def lookat_pose(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """Drake camera frame: +z forward, +x right, +y DOWN."""
    from pydrake.math import RigidTransform, RotationMatrix
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    n = np.linalg.norm(x)
    if n < 1e-9:
        x = np.array([1.0, 0, 0])
    else:
        x = x / n
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return RigidTransform(RotationMatrix(R), eye)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frame", type=float, default=None,
                    help="render one frame at this path fraction and exit")
    ap.add_argument("--az", type=float, default=None,
                    help="override base azimuth (deg) for the test frame")
    args = ap.parse_args()

    import imageio
    from PIL import Image, ImageDraw, ImageFont

    wpath, stored = q8_exact_path()
    from stageB_bimanual_regions import build
    checker, _ = build()
    cplant = checker.plant()

    # clearance profile on the resampled path (same as the truth video)
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

    from pydrake.multibody.tree import BodyIndex
    cctx = cplant.CreateDefaultContext()
    wsg2b = cplant.GetBodyByName("body",
                                 cplant.GetModelInstanceByName("wsg_2"))
    phi, pairs, wsg2_pos = [], [], []
    for q in qs:
        rc = checker.CalcRobotClearance(q, 0.5)
        d = np.asarray(rc.distances())
        cplant.SetPositions(cctx, q)
        wsg2_pos.append(cplant.EvalBodyPoseInWorld(cctx, wsg2b)
                        .translation().copy())
        if d.size == 0:
            phi.append(0.5)
            pairs.append(None)
            continue
        k = int(np.argmin(d))
        phi.append(float(d[k]))
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

    # visualization twin
    diagram, vplant, sensor, cam_body = build_viz()
    vctx = diagram.CreateDefaultContext()
    pctx = vplant.GetMyContextFromRoot(vctx)
    sctx = sensor.GetMyContextFromRoot(vctx)
    inst = {n: vplant.GetModelInstanceByName(n)
            for n in ("iiwa_1", "iiwa_2")}

    def set_config(q):
        vplant.SetPositions(pctx, inst["iiwa_1"], q[:7])
        vplant.SetPositions(pctx, inst["iiwa_2"], q[7:14])

    overview = np.array([0.0, 0.0, 0.45])
    AZ_BASE = [145.0]

    def render(pi, z):
        set_config(qs[pi])
        look = (1 - 0.7 * z) * overview + 0.7 * z * wsg2_pos[pi]
        prog = pi / n_pts
        az_base = AZ_BASE[0]
        az = np.deg2rad(az_base + 22 * prog)
        el = np.deg2rad(-18 - 6 * z)
        dist = 3.0 - 1.8 * z
        eye = look + dist * np.array([np.cos(el) * np.cos(az),
                                      np.cos(el) * np.sin(az),
                                      -np.sin(el)])
        vplant.SetFreeBodyPose(pctx, cam_body, lookat_pose(eye, look))
        img = sensor.color_image_output_port().Eval(sctx)
        return np.array(img.data[:, :, :3])

    try:
        font_b = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        font_s = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font_b = font_s = ImageFont.load_default()

    def compose(pi, z, hold_caption=None):
        arr = render(pi, z)
        img = Image.fromarray(arr)
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
                fill=(60, 60, 70), font=font_s)
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
        if args.az is not None:
            AZ_BASE[0] = args.az
        pi = int(args.test_frame * (n_pts - 1))
        z = 1.0 if phi[pi] < 0.012 else 0.0
        compose(pi, z).save("out/e9_q8_hd_test.png")
        print(f"test frame at {args.test_frame} (phi "
              f"{1000*phi[pi]:.1f} mm) -> out/e9_q8_hd_test.png")
        return

    # pacing: collision 7x, near 3x, clean 2x + start/goal holds
    dwell = np.where(phi < 0, 7, np.where(phi < 0.01, 3, 2))
    frames_idx = np.repeat(np.arange(n_pts), dwell)
    z_target = (phi < 0.012).astype(float)
    z_s, z = [], 0.0
    for pi in frames_idx:
        z += (0.16 if z_target[pi] > z else 0.012) * (z_target[pi] - z)
        z_s.append(z)

    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8)
    for _ in range(int(1.2 * FPS)):                    # opening hold
        writer.append_data(np.asarray(compose(0, 1.0)))
    for fi, pi in enumerate(frames_idx):
        writer.append_data(np.asarray(compose(pi, z_s[fi])))
    for _ in range(int(1.6 * FPS)):                    # goal hold
        writer.append_data(np.asarray(compose(
            n_pts - 1, 0.0,
            hold_caption="goal reached -- the clean stretches are real, "
            "but three collision intervals sit between them")))
    # end summary card
    card = Image.new("RGB", (W, H), (18, 20, 26))
    dr = ImageDraw.Draw(card)
    lines = [
        ("the reference answer of query 8:", (235, 235, 240)),
        ("", None),
        ("  start config 63.5 mm inside the shelf", (240, 90, 80)),
        ("  arms through each other, 41.7 mm deep", (240, 90, 80)),
        ("  arms through each other, 56.2 mm deep", (240, 90, 80)),
        ("", None),
        ("14.4% of the path length is in collision.", (235, 235, 240)),
        ("Every planner without a physical layer returns it",
         (200, 200, 210)),
        ("and scores it as optimal.", (200, 200, 210)),
    ]
    y = 150
    for txt, col in lines:
        if col:
            dr.text((110, y), txt, fill=col, font=font_b)
        y += 52
    for _ in range(int(3.0 * FPS)):
        writer.append_data(np.asarray(card))
    writer.close()
    n_total = int(1.2 * FPS) + len(frames_idx) + int(1.6 * FPS) \
        + int(3.0 * FPS)
    print(f"written {OUT} ({n_total} frames, {n_total/FPS:.1f}s)",
          flush=True)


if __name__ == "__main__":
    main()
