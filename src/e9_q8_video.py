#!/usr/bin/env python3
"""E9 video: the GCS* reference answer for query 8 driving the wsg_2
gripper 63.5 mm through the shelf, rendered in MuJoCo.

Truth discipline (the UAV lesson): every geom in the video is the
COLLISION geometry exported pose-by-pose from the planning scene itself
(48 primitives; quaternions straight from Drake, no hand-written
angles).  The clearance readout is computed by the same
SceneGraphCollisionChecker that audited the path; the pair that
penetrates is flashed red.

    python3 e9_q8_video.py            # writes out/e9_q8_violation.mp4
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
import baseline_gcsstar as bgs
import stageB_bimanual_ipgcs as sb
from e9_verify_gcsstar import chain_waypoints

FPS = 60
W, H = 1280, 720
OUT = "out/e9_q8_violation.mp4"
QUERY = 8

COLORS = {
    "iiwa_1": (0.45, 0.57, 0.75, 1.0),
    "wsg_1": (0.30, 0.42, 0.62, 1.0),
    "iiwa_2": (0.80, 0.62, 0.31, 1.0),
    "wsg_2": (0.95, 0.55, 0.10, 1.0),
    "shelves": (0.52, 0.42, 0.34, 1.0),
    "binR": (0.45, 0.48, 0.45, 1.0),
    "binL": (0.45, 0.48, 0.45, 1.0),
    "table": (0.30, 0.30, 0.32, 1.0),
}
RED = (0.90, 0.12, 0.10, 1.0)


def q8_exact_path():
    data = pickle.load(open("out/bimanual_regions_scale.pkl", "rb"))
    reg_dict = data["regions"] if isinstance(data, dict) and \
        "regions" in data else data
    topo = pickle.load(open("out/bimanual219_topo.pkl", "rb"))
    if isinstance(topo, dict) and "topo" in topo:
        topo = topo["topo"]
    rec = [json.loads(l) for l in open("out/bench_219focus_K1.jsonl")][QUERY]
    s, g = np.asarray(rec["s"]), np.asarray(rec["g"])
    rg = sb.with_query(topo, s, g)
    r = bgs.gcs_star(rg, timeout=60.0, K=1, max_revisits=1, seed=0)
    stored = rec["gcsstar"]["cost"]
    assert abs(r["cost"] - stored) < 1e-6, (r["cost"], stored)
    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}
    chain = [topo["portals"][pair_eid[frozenset((a, b_))]]
             for a, b_ in zip(r["path"], r["path"][1:])]
    c2, wpath = chain_waypoints([(p["A"], p["b"]) for p in chain], s, g)
    assert abs(c2 - stored) < 1e-6
    print(f"q{QUERY}: reproduced GCS* cost {c2:.4f} "
          f"(stored {stored:.4f}), {len(wpath)} waypoints", flush=True)
    return wpath, stored


def scene_geoms(checker):
    """(geoms, plant): every proximity geometry with body + local pose."""
    from pydrake.geometry import Box, Cylinder, Role, Sphere
    plant = checker.plant()
    diagram = checker.model()
    sg = diagram.GetSubsystemByName("scene_graph")
    insp = sg.model_inspector()
    out = []
    for gid in insp.GetAllGeometryIds():
        if insp.GetProximityProperties(gid) is None:
            continue
        shape = insp.GetShape(gid)
        body = plant.GetBodyFromFrameId(insp.GetFrameId(gid))
        X_BG = insp.GetPoseInFrame(gid)
        model = plant.GetModelInstanceName(body.model_instance())
        if isinstance(shape, Sphere):
            spec = ("sphere", (shape.radius(),))
        elif isinstance(shape, Box):
            spec = ("box", (shape.width() / 2, shape.depth() / 2,
                            shape.height() / 2))
        elif isinstance(shape, Cylinder):
            spec = ("cylinder", (shape.radius(), shape.length() / 2))
        else:
            continue
        out.append({"spec": spec, "body": body, "X_BG": X_BG,
                    "model": model, "bname": body.name()})
    return out, plant


def build_mjcf(geoms):
    lines = ["<mujoco model='e9q8'>",
             "<visual><headlight ambient='.45 .45 .45' diffuse='.7 .7 .7'/>"
             f"<global offwidth='{W}' offheight='{H}'/>"
             "<quality shadowsize='4096'/></visual>",
             "<asset><texture type='skybox' builtin='gradient' "
             "rgb1='.97 .97 .99' rgb2='.82 .85 .90' width='64' "
             "height='64'/></asset>",
             "<worldbody>",
             "<light pos='1.5 -1.5 3' dir='-.4 .4 -1' diffuse='.8 .8 .8'/>",
             "<light pos='-2 1.5 2.5' dir='.5 -.4 -1' diffuse='.45 .45 .45'/>",
             "<geom name='floor' type='plane' size='6 6 .1' "
             "pos='0 0 -0.02' rgba='.93 .93 .94 1'/>"]
    for i, gm in enumerate(geoms):
        kind, dims = gm["spec"]
        rgba = COLORS.get(gm["model"], (0.6, 0.6, 0.6, 1.0))
        sz = " ".join(f"{x:.6f}" for x in dims)
        lines.append(
            f"<body name='b{i}' mocap='true'><geom name='g{i}' "
            f"type='{kind}' size='{sz}' rgba='{rgba[0]} {rgba[1]} "
            f"{rgba[2]} {rgba[3]}'/></body>")
    lines += ["</worldbody></mujoco>"]
    return "\n".join(lines)


def main():
    import imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    wpath, stored = q8_exact_path()
    from stageB_bimanual_regions import build
    checker, _ = build()
    geoms, plant = scene_geoms(checker)
    print(f"{len(geoms)} proximity geoms", flush=True)

    # constant-speed resample, then clearance profile
    seg = np.linalg.norm(np.diff(wpath, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    n_pts = 460
    ts = np.linspace(0, L[-1], n_pts)
    qs = []
    for t in ts:
        k = min(max(int(np.searchsorted(L, t, "right") - 1), 0),
                len(seg) - 1)
        lam = 0.0 if seg[k] < 1e-12 else (t - L[k]) / seg[k]
        qs.append((1 - lam) * wpath[k] + lam * wpath[k + 1])
    qs = np.asarray(qs)

    from pydrake.multibody.tree import BodyIndex
    ctx0 = plant.CreateDefaultContext()
    wsg2b = plant.GetBodyByName("body",
                                plant.GetModelInstanceByName("wsg_2"))
    phi, pair_bodies, wsg2_pos = [], [], []
    for q in qs:
        rc = checker.CalcRobotClearance(q, 0.5)
        d = np.asarray(rc.distances())
        plant.SetPositions(ctx0, q)
        wsg2_pos.append(plant.EvalBodyPoseInWorld(ctx0, wsg2b)
                        .translation().copy())
        if d.size == 0:
            phi.append(0.5)
            pair_bodies.append(None)
            continue
        k = int(np.argmin(d))
        phi.append(float(d[k]))
        b1 = plant.get_body(BodyIndex(int(rc.robot_indices()[k])))
        b2 = plant.get_body(BodyIndex(int(rc.other_indices()[k])))
        pair_bodies.append((
            (plant.GetModelInstanceName(b1.model_instance()), b1.name()),
            (plant.GetModelInstanceName(b2.model_instance()), b2.name())))
    phi = np.asarray(phi)
    wsg2_pos = np.asarray(wsg2_pos)
    # collision runs -> per-run captions + pair report
    runs, i0 = [], 0
    while i0 < n_pts:
        if phi[i0] < 0:
            j = i0
            while j < n_pts and phi[j] < 0:
                j += 1
            k = i0 + int(np.argmin(phi[i0:j]))
            runs.append((i0, j, k))
            print(f"collision run {len(runs)}: [{i0/n_pts:.2f},"
                  f"{j/n_pts:.2f}] min {1000*phi[k]:.1f} mm, "
                  f"pair {pair_bodies[k]}", flush=True)
            i0 = j
        else:
            i0 += 1
    run_caption = {}
    for ri, (a, b_, k) in enumerate(runs):
        models = {m for m, _ in pair_bodies[k]}
        if a == 0:
            cap = ("benchmark START configuration is inside the shelf "
                   "(sampled from a contaminated region)")
        elif "shelves" in models or "table" in models \
                or "binR" in models or "binL" in models:
            cap = "the optimal path dives into the furniture"
        else:
            cap = ("the optimal path drives the two arms THROUGH each "
                   "other (region pocket)")
        for t in range(a, b_):
            run_caption[t] = cap

    # dwell: slow-mo where it matters
    dwell = np.where(phi < 0, 7, np.where(phi < 0.01, 3, 1))
    frames_idx = np.repeat(np.arange(n_pts), dwell)

    model = mujoco.MjModel.from_xml_string(build_mjcf(geoms))
    mdata = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    base_rgba = model.geom_rgba.copy()
    gidx_of = {i: model.geom(f"g{i}").id for i in range(len(geoms))}

    # interest-driven camera: zoom onto the gripper during events,
    # fast attack / slow release, overview otherwise
    ctx = plant.CreateDefaultContext()
    overview = np.array([0.0, 0.0, 0.45])
    cam = mujoco.MjvCamera()
    z_target = (phi < 0.012).astype(float)
    z_s, z = [], 0.0
    for pi in frames_idx:
        zt = z_target[pi]
        z += (0.16 if zt > z else 0.012) * (zt - z)
        z_s.append(z)
    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8)
    try:
        font_b = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        font_s = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font_b = font_s = ImageFont.load_default()

    for fi, pi in enumerate(frames_idx):
        q = qs[pi]
        plant.SetPositions(ctx, q)
        for i, gm in enumerate(geoms):
            X = plant.EvalBodyPoseInWorld(ctx, gm["body"]) @ gm["X_BG"]
            mdata.mocap_pos[i] = X.translation()
            quat = X.rotation().ToQuaternion()
            mdata.mocap_quat[i] = [quat.w(), quat.x(), quat.y(), quat.z()]
        mujoco.mj_forward(model, mdata)
        # red flash on the penetrating pair (instance-qualified match)
        model.geom_rgba[:] = base_rgba
        if phi[pi] < 0 and pair_bodies[pi] is not None:
            keys = set(pair_bodies[pi])
            for i, gm in enumerate(geoms):
                if (gm["model"], gm["bname"]) in keys:
                    model.geom_rgba[gidx_of[i]] = RED
        prog = pi / n_pts
        z = z_s[fi]
        cam.lookat = (1 - 0.7 * z) * overview + 0.7 * z * wsg2_pos[pi]
        cam.distance = 3.1 - 1.85 * z
        cam.azimuth = 145 + 22 * prog
        cam.elevation = -18 - 6 * z
        renderer.update_scene(mdata, camera=cam)
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img)
        dr.text((28, 20, ), "GCS* reference answer, 14-DOF bimanual "
                f"benchmark q{QUERY}", fill=(20, 20, 25), font=font_b)
        dr.text((28, 62), f"reported cost {stored:.4f}  |  arm 1 blue, "
                "arm 2 orange  |  every geom = the planning scene's own "
                "collision geometry", fill=(60, 60, 70), font=font_s)
        c_mm = 1000 * phi[pi]
        if c_mm >= 0:
            col = (20, 130, 40) if c_mm > 10 else (200, 130, 0)
            dr.text((28, H - 64), f"clearance  +{c_mm:6.1f} mm",
                    fill=col, font=font_b)
        else:
            dr.text((28, H - 64), f"PENETRATION  {c_mm:6.1f} mm",
                    fill=(210, 25, 20), font=font_b)
            dr.text((28, H - 100),
                    run_caption.get(pi, "in collision"),
                    fill=(210, 25, 20), font=font_s)
        writer.append_data(np.asarray(img))
    writer.close()
    print(f"written {OUT} ({len(frames_idx)} frames, "
          f"{len(frames_idx)/FPS:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
