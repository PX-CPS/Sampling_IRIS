#!/usr/bin/env python3
"""E12: pick-and-place COMPARISON -- both arms unload the top shelf
into the bins, side by side.

Left: the GCS* reference answer (212/2000 probes in collision -- the
item-laden gripper drags through the shelf).  Right: our certified
answer (+27.8% cost, clean).  Same task, same library, same camera;
items are kinematic props attached at the finger midpoints on both
sides; certificates apply to the robot motion.

Pair: top_shelf/top_shelf -> bin_R/bin_L.

    python3 e12_compare_pickplace.py --test-frame 0.0
    python3 e12_compare_pickplace.py     # out/e12_compare_pickplace.mp4
"""
from __future__ import annotations

import argparse
import pickle
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.environ.get("GCS_SR",
                            os.path.expanduser("~/gcs-science-robotics")))

FPS = 60
PW, PH = 960, 720
W, H = 2 * PW, PH
OUT = "out/e12_compare_pickplace.mp4"
PAIR = ("top_shelf/top_shelf", "bin_R/bin_L")
ITEM_HALF = 0.024
ITEM_RGBA = {"r1_": (0.10, 0.55, 0.75, 1.0),
             "r2_": (0.92, 0.72, 0.15, 1.0)}


def build_model_two_items(checker):
    import mujoco
    from e9_q8_video3 import PREFIX, XML, wsg_boxes
    spec = mujoco.MjSpec.from_file(XML)
    for g in spec.geoms:
        if g.name.startswith("shelf"):
            g.rgba[3] = 0.45
    boxes = wsg_boxes(checker)
    offs = {}
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
        f = {it["bname"]: np.asarray(it["pos"])
             for it in items if "finger" in it["bname"]}
        body_c = next(np.asarray(it["pos"]) for it in items
                      if it["bname"] == "body")
        mid = 0.5 * (f["left_finger"] + f["right_finger"])
        tip = mid - body_c
        offs[p] = mid + 0.035 * tip / max(np.linalg.norm(tip), 1e-9)
        item = spec.worldbody.add_body()
        item.name = f"{p}item"
        item.mocap = True
        gi = item.add_geom()
        gi.name = f"{p}item_geom"
        gi.type = mujoco.mjtGeom.mjGEOM_BOX
        gi.size[:] = (ITEM_HALF, ITEM_HALF, ITEM_HALF)
        gi.rgba[:] = ITEM_RGBA[p]
        gi.contype = 0
        gi.conaffinity = 0
    return spec.compile(), offs


def resample(wpath, n):
    seg = np.linalg.norm(np.diff(wpath, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    out = []
    for t in np.linspace(0, L[-1], n):
        k = min(max(int(np.searchsorted(L, t, "right") - 1), 0),
                len(seg) - 1)
        lam = 0.0 if seg[k] < 1e-12 else (t - L[k]) / seg[k]
        out.append((1 - lam) * wpath[k] + lam * wpath[k + 1])
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frame", type=float, default=None)
    args = ap.parse_args()
    import imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from pydrake.multibody.tree import BodyIndex
    from scipy.spatial.transform import Rotation

    import baseline_gcsstar as bgs
    import stageB_bimanual_ipgcs as sb
    from e9_q8_video3 import PREFIX, RED, fk_align_check
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
    c_ref, wp_ref = chain_waypoints(chain, s, g)
    wp_ours = np.load("out/e10_ours_task_paths.npz")[
        f"{PAIR[0]}|{PAIR[1]}"]
    c_ours = float(np.linalg.norm(np.diff(wp_ours, axis=0),
                                  axis=1).sum())
    print(f"ref {c_ref:.3f} (colliding) vs ours {c_ours:.3f} "
          f"(+{100*(c_ours/c_ref-1):.1f}%)", flush=True)

    checker, _ = build()
    cplant = checker.plant()
    n_pts = 420
    QL, QR = resample(wp_ref, n_pts), resample(wp_ours, n_pts)

    def profile(qs):
        phi, pairs = [], []
        for q in qs:
            rc = checker.CalcRobotClearance(q, 0.5)
            d = np.asarray(rc.distances())
            if d.size == 0:
                phi.append(0.5)
                pairs.append(None)
                continue
            k = int(np.argmin(d))
            phi.append(float(d[k]))
            b1 = cplant.get_body(BodyIndex(int(rc.robot_indices()[k])))
            b2 = cplant.get_body(BodyIndex(int(rc.other_indices()[k])))
            pairs.append((
                (cplant.GetModelInstanceName(b1.model_instance()),
                 b1.name()),
                (cplant.GetModelInstanceName(b2.model_instance()),
                 b2.name())))
        return np.asarray(phi), pairs

    phiL, pairsL = profile(QL)
    phiR, pairsR = profile(QR)
    print(f"left in-collision {100*(phiL<0).mean():.0f}% | right min "
          f"{1000*phiR.min():.1f} mm", flush=True)
    assert phiR.min() > 0

    m, offs = build_model_two_items(checker)
    assert fk_align_check(checker, m, QL[[0, -1]]) < 0.005
    mdata = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=PH, width=PW)
    base_rgba = m.geom_rgba.copy()
    item_id = {p: m.body(f"{p}item").mocapid for p in ("r1_", "r2_")}

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

    cam = mujoco.MjvCamera()

    def render_panel(q, phi_v, pair_v, prog):
        mdata.qpos[:14] = q
        mujoco.mj_forward(m, mdata)
        for p in ("r1_", "r2_"):
            b = mdata.body(f"{p}link7")
            Rw = Rotation.from_quat(np.roll(b.xquat, -1))
            mdata.mocap_pos[item_id[p]] = b.xpos + Rw.apply(offs[p])
            mdata.mocap_quat[item_id[p]] = b.xquat
        mujoco.mj_forward(m, mdata)
        m.geom_rgba[:] = base_rgba
        if phi_v < 0 and pair_v is not None:
            for gi in flash_ids(pair_v):
                a = base_rgba[gi][3]
                m.geom_rgba[gi] = (RED[0], RED[1], RED[2],
                                   min(a, 0.55) if a < 1.0 else 1.0)
        cam.lookat = np.array([0.5, 0.25, 0.5])
        cam.distance = 2.75
        cam.azimuth = 120 + 30 * prog
        cam.elevation = -19
        renderer.update_scene(mdata, camera=cam)
        return renderer.render()

    try:
        font_b = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        font_m = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_s = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font_b = font_m = font_s = ImageFont.load_default()

    def compose(pi):
        prog = pi / n_pts
        Lp = render_panel(QL[pi], phiL[pi], pairsL[pi], prog)
        Rp = render_panel(QR[pi], phiR[pi], pairsR[pi], prog)
        img = Image.new("RGB", (W, H))
        img.paste(Image.fromarray(Lp), (0, 0))
        img.paste(Image.fromarray(Rp), (PW, 0))
        dr = ImageDraw.Draw(img)
        dr.rectangle([PW - 2, 0, PW + 2, H], fill=(30, 30, 36))
        dr.text((24, 16), "GCS* reference answer",
                fill=(214, 30, 24), font=font_b)
        dr.text((24, 54), f"cost {c_ref:.3f} -- carries the items "
                "THROUGH the shelf", fill=(60, 60, 70), font=font_s)
        dr.text((PW + 24, 16), "ours: certified answer",
                fill=(20, 130, 40), font=font_b)
        dr.text((PW + 24, 54), f"cost {c_ours:.3f} "
                f"(+{100*(c_ours/c_ref-1):.1f}%) -- items delivered "
                "clean", fill=(60, 60, 70), font=font_s)
        dr.text((int(W/2) - 330, H - 40),
                "same task: both arms unload the top shelf into the "
                "bins  |  items ride the grippers on both sides",
                fill=(80, 80, 90), font=font_s)
        cL = 1000 * phiL[pi]
        if cL < 0:
            for wdt in range(8):
                dr.rectangle([wdt, wdt, PW - 1 - wdt, H - 1 - wdt],
                             outline=(214, 40, 30))
            dr.text((24, H - 104), f"PENETRATION {cL:6.1f} mm",
                    fill=(214, 30, 24), font=font_m)
        else:
            dr.text((24, H - 104), f"clearance  +{cL:6.1f} mm",
                    fill=(20, 130, 40) if cL > 10 else (200, 130, 0),
                    font=font_m)
        cR = 1000 * phiR[pi]
        dr.text((PW + 24, H - 104), f"clearance  +{cR:6.1f} mm",
                fill=(20, 130, 40) if cR > 10 else (200, 130, 0),
                font=font_m)
        return img

    if args.test_frame is not None:
        pi = int(args.test_frame * (n_pts - 1))
        compose(pi).save("out/e12_test.png")
        print("-> out/e12_test.png")
        return

    dwell = np.where(phiL < 0, 4, 2)
    frames_idx = np.repeat(np.arange(n_pts), dwell)
    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8)
    for _ in range(int(1.2 * FPS)):
        writer.append_data(np.asarray(compose(0)))
    for pi in frames_idx:
        writer.append_data(np.asarray(compose(pi)))
    for _ in range(int(1.8 * FPS)):
        writer.append_data(np.asarray(compose(n_pts - 1)))
    card = Image.new("RGB", (W, H), (18, 20, 26))
    dr = ImageDraw.Draw(card)
    for y, (txt, col) in enumerate([
            ("both arms unload the top shelf -- same library:",
             (235, 235, 240)),
            ("", None),
            ("   reference: 212/2000 probes in collision,",
             (240, 90, 80)),
            ("   the item-laden gripper drags through the shelf",
             (240, 90, 80)),
            ("   ours: certified clean delivery, +27.8% cost",
             (110, 220, 130)),
            ("", None),
            ("The certificate is the difference between",
             (200, 200, 210)),
            ("a delivered item and a broken shelf.", (200, 200, 210))]):
        if col:
            dr.text((150, 160 + 56 * y), txt, fill=col, font=font_b)
    for _ in range(int(3.2 * FPS)):
        writer.append_data(np.asarray(card))
    writer.close()
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
