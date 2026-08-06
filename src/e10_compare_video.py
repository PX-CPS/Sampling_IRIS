#!/usr/bin/env python3
"""E10 comparison video: SAME task, SAME library, side by side --
left: the GCS* reference answer (drags through the shelf, 329/2000
probes in collision); right: our physically-certified answer (+15.0%
cost).  Same camera, same renderer, same clearance instrument.

Pair: neutral/shelf_1 -> shelf_1/neutral (the two arms swap roles).

    python3 e10_compare_video.py     # out/e10_compare.mp4
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
PW, PH = 960, 720                       # per-panel size
W, H = 2 * PW, PH
OUT = "out/e10_compare.mp4"
PAIR = ("neutral/shelf_1", "shelf_1/neutral")


def resample(wpath, n):
    seg = np.linalg.norm(np.diff(wpath, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    qs = []
    for t in np.linspace(0, L[-1], n):
        k = min(max(int(np.searchsorted(L, t, "right") - 1), 0),
                len(seg) - 1)
        lam = 0.0 if seg[k] < 1e-12 else (t - L[k]) / seg[k]
        qs.append((1 - lam) * wpath[k] + lam * wpath[k + 1])
    return np.asarray(qs)


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
    c_ref, wp_ref = chain_waypoints(chain, s, g)
    paths = np.load("out/e10_ours_task_paths.npz")
    wp_ours = paths[f"{PAIR[0]}|{PAIR[1]}"]
    c_ours = float(np.linalg.norm(np.diff(wp_ours, axis=0),
                                  axis=1).sum())
    print(f"ref {c_ref:.3f} (colliding) vs ours {c_ours:.3f} "
          f"(+{100*(c_ours/c_ref-1):.1f}%)", flush=True)

    checker, _ = build()
    cplant = checker.plant()
    n_pts = 420
    QL = resample(wp_ref, n_pts)
    QR = resample(wp_ours, n_pts)

    def profile(qs):
        cctx = cplant.CreateDefaultContext()
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
    print(f"left in-collision {100*(phiL<0).mean():.0f}%  |  right min "
          f"clearance {1000*phiR.min():.1f} mm", flush=True)
    assert phiR.min() > 0, "our path must be clean"

    m, _ = build_model(checker)
    assert fk_align_check(checker, m, QL[[0, -1]]) < 0.005
    mdata = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=PH, width=PW)
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

    cam = mujoco.MjvCamera()

    def render_panel(q, phi_v, pair_v, prog):
        mdata.qpos[:14] = q
        mujoco.mj_forward(m, mdata)
        m.geom_rgba[:] = base_rgba
        if phi_v < 0 and pair_v is not None:
            for gi in flash_ids(pair_v):
                a = base_rgba[gi][3]
                m.geom_rgba[gi] = (RED[0], RED[1], RED[2],
                                   min(a, 0.55) if a < 1.0 else 1.0)
        cam.lookat = np.array([0.5, 0.25, 0.5])
        cam.distance = 2.75
        cam.azimuth = 125 + 25 * prog
        cam.elevation = -20
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
        L = render_panel(QL[pi], phiL[pi], pairsL[pi], prog)
        R = render_panel(QR[pi], phiR[pi], pairsR[pi], prog)
        img = Image.new("RGB", (W, H))
        img.paste(Image.fromarray(L), (0, 0))
        img.paste(Image.fromarray(R), (PW, 0))
        dr = ImageDraw.Draw(img)
        dr.rectangle([PW - 2, 0, PW + 2, H], fill=(30, 30, 36))
        dr.text((24, 16), "GCS* reference answer",
                fill=(214, 30, 24), font=font_b)
        dr.text((24, 54), f"cost {c_ref:.3f} -- shortest, but through "
                "the shelf", fill=(60, 60, 70), font=font_s)
        dr.text((PW + 24, 16), "ours: certified answer",
                fill=(20, 130, 40), font=font_b)
        dr.text((PW + 24, 54), f"cost {c_ours:.3f} "
                f"(+{100*(c_ours/c_ref-1):.1f}%) -- continuous clearance "
                "certificate", fill=(60, 60, 70), font=font_s)
        dr.text((int(W/2) - 290, H - 40),
                "same task, same library, same camera:  arms swap "
                "neutral and shelf-1 stations", fill=(80, 80, 90),
                font=font_s)
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

    dwell = np.where(phiL < 0, 4, 2)
    frames_idx = np.repeat(np.arange(n_pts), dwell)
    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8)
    for _ in range(int(1.0 * FPS)):
        writer.append_data(np.asarray(compose(0)))
    for pi in frames_idx:
        writer.append_data(np.asarray(compose(pi)))
    for _ in range(int(1.6 * FPS)):
        writer.append_data(np.asarray(compose(n_pts - 1)))
    card = Image.new("RGB", (W, H), (18, 20, 26))
    dr = ImageDraw.Draw(card)
    for y, (txt, col) in enumerate([
            ("same task, same region library:", (235, 235, 240)),
            ("", None),
            ("   reference answer: 329/2000 probes in collision",
             (240, 90, 80)),
            ("   ours: certified collision-free, +15.0% cost",
             (110, 220, 130)),
            ("", None),
            ("The physical layer costs a fraction of a second",
             (200, 200, 210)),
            ("and 15% path length.  Not having it costs the truth.",
             (200, 200, 210))]):
        if col:
            dr.text((150, 170 + 56 * y), txt, fill=col, font=font_b)
    for _ in range(int(3.2 * FPS)):
        writer.append_data(np.asarray(card))
    writer.close()
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
