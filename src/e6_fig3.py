#!/usr/bin/env python3
"""E6 figure 3: the informed spheroid ITERATING in 3-D.

Audited UAV building (42 axis-aligned rooms, 52 door/window contact
faces).  Mid-range query chosen so pruning engages (oracle alive 27%).
Top row: three snapshots of the anytime run -- incumbent path, the
prolate spheroid {x : |x-s|+|x-g| <= c_best}, and the alive (amber) vs
pruned (gray) interfaces, at the first / middle / final improvement.
Bottom row: overview with the final state, the cost staircase, and the
alive-interface count stepping down with each improvement.

    python3 e6_fig3.py
"""
from __future__ import annotations

import json
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import (Line3DCollection, Poly3DCollection)

sys.path.insert(0, ".")
import e6_sampling as e6
from e6_fig1 import poly_vertices

C_AREA = "#059669"
C_GCS = "#0ea5e9"
C_IFACE = "#f59e0b"
C_PRUNED = "#9ca3af"
C_PATH = "#dc2626"
C_ROOM = "#64748b"


def iface_corners(f):
    """3-D corner points of a (possibly degenerate) interface polytope."""
    h = f["hull"]
    if h["dim"] == 0:
        return None
    V = poly_vertices(h["Ar"], h["br"]) if h["dim"] == 2 else None
    if h["dim"] == 1:
        lo = hi = None
        # endpoints along the 1-D hull
        for sgn in (1.0, -1.0):
            Au = h["Ar"] @ np.array([sgn])
            slack = h["br"] - h["Ar"] @ h["yc"]
            pos = Au > 1e-12
            if not np.any(pos):
                return None
            t = float(np.min(slack[pos] / Au[pos]))
            p = h["yc"] + t * np.array([sgn])
            lo = p if sgn < 0 else lo
            hi = p if sgn > 0 else hi
        V = np.array([lo, hi])
    if V is None or len(V) < 2:
        return None
    return np.array([h["x0"] + h["N"] @ v for v in V])


def draw_building(ax, topo, ifaces, alive, path=None, spheroid=None,
                  s=None, g=None):
    los, his = np.asarray(topo["bbox_lo"]), np.asarray(topo["bbox_hi"])
    segs = []
    for lo, hi in zip(los, his):
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
             (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        segs += [[c[a], c[b]] for a, b in E]
    ax.add_collection3d(Line3DCollection(segs, colors=C_ROOM, alpha=0.16,
                                         linewidths=0.5))
    polys_a, polys_p = [], []
    for eid, f in ifaces.items():
        V = iface_corners(f)
        if V is None or len(V) < 3:
            continue
        (polys_a if eid in alive else polys_p).append(V)
    if polys_p:
        ax.add_collection3d(Poly3DCollection(
            polys_p, facecolors=C_PRUNED, alpha=0.18, edgecolors="none"))
    if polys_a:
        ax.add_collection3d(Poly3DCollection(
            polys_a, facecolors=C_IFACE, alpha=0.60,
            edgecolors="#b45309", linewidths=0.6))
    if spheroid is not None:
        c, sq, gq = spheroid
        f2 = np.linalg.norm(gq - sq)
        if c > f2 * (1 + 1e-9):
            a, b = c / 2, np.sqrt(c * c - f2 * f2) / 2
            u = (gq - sq) / f2
            t = np.array([1.0, 0, 0])
            if abs(u @ t) > 0.9:
                t = np.array([0, 1.0, 0])
            v = np.cross(u, t)
            v /= np.linalg.norm(v)
            w = np.cross(u, v)
            th, ph = np.meshgrid(np.linspace(0, np.pi, 26),
                                 np.linspace(0, 2 * np.pi, 40))
            ctr = 0.5 * (sq + gq)
            P = (ctr[None, None] + a * np.cos(th)[..., None] * u
                 + b * (np.sin(th) * np.cos(ph))[..., None] * v
                 + b * (np.sin(th) * np.sin(ph))[..., None] * w)
            ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=C_PATH,
                            alpha=0.09, linewidth=0, antialiased=False,
                            shade=False)
            ax.plot_wireframe(P[..., 0], P[..., 1], P[..., 2], color=C_PATH,
                              alpha=0.16, linewidth=0.4, rstride=5,
                              cstride=10)
    if path is not None:
        p = np.asarray(path)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-", color=C_PATH, lw=2.6,
                zorder=10)
    if s is not None:
        ax.scatter(*s, marker="*", s=170, color="#16a34a", zorder=11)
        ax.scatter(*g, marker="*", s=170, color=C_PATH, zorder=11)
    lo, hi = los.min(0), his.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=24, azim=-58)


def main():
    data = json.load(open("out/e6_step3_3d.json"))
    lib = pickle.load(open(data["lib"], "rb"))
    topo = lib["topo"]
    ifaces = e6.classify_interfaces(topo)
    q = data["query"]
    s, g = np.asarray(q["s"]), np.asarray(q["g"])
    run = data["run"]
    ev = run["events"]
    paths = [np.asarray(p) for p in run["event_paths"]]
    alive = [set(a) for a in run["event_alive"]]
    n_if = data["n_ifaces"]

    ks = [0, len(ev) // 2, len(ev) - 1]
    fig = plt.figure(figsize=(15.5, 9.2))

    for col, k in enumerate(ks):
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        draw_building(ax, topo, ifaces, alive[k], path=paths[k],
                      spheroid=(ev[k][1], s, g), s=s, g=g)
        f2 = np.linalg.norm(g - s)
        b_ax = np.sqrt(ev[k][1] ** 2 - f2 ** 2)
        ax.set_title(f"({'abc'[col]}) improvement {k + 1}/{len(ev)}:  "
                     f"$c$={ev[k][1]:.2f}\nspheroid minor axis "
                     f"{b_ax:.1f} m,  alive {len(alive[k])}/{n_if}",
                     fontsize=9)

    axd = fig.add_subplot(2, 3, 4, projection="3d")
    draw_building(axd, topo, ifaces, alive[-1], path=paths[-1],
                  spheroid=(ev[-1][1], s, g), s=s, g=g)
    axd.view_init(elev=64, azim=-90)
    axd.set_title("(d) final state, top view: the surviving amber\n"
                  "interfaces hug the corridor; the rest is pruned",
                  fontsize=9)

    axe = fig.add_subplot(2, 3, 5)
    ts = [t for t, _ in ev]
    cs = [c for _, c in ev]
    t_end = sum(r["t_round"] for r in run["rows"])
    axe.step(ts + [t_end], cs + [cs[-1]], where="post", color=C_AREA,
             lw=2.0)
    axe.plot(ts, cs, "o", color=C_AREA, ms=4)
    for col, k in enumerate(ks):
        axe.annotate(f"({'abc'[col]})", (ts[k], cs[k]),
                     textcoords="offset points", xytext=(6, 8),
                     fontsize=9, color=C_PATH)
        axe.plot(ts[k], cs[k], "o", ms=9, mfc="none", mec=C_PATH, mew=1.8)
    axe.axhline(q["ub"], color=C_GCS, ls="--", lw=1.5)
    axe.text(t_end, q["ub"], f"  GCS optimum {q['ub']:.2f}", color=C_GCS,
             fontsize=8, ha="right", va="bottom")
    axe.set_xlabel("wall-clock time (s)")
    axe.set_ylabel("incumbent cost (m)")
    axe.grid(alpha=0.22)
    axe.set_title(f"(e) anytime staircase: {len(ev)} improvements,\n"
                  f"final {cs[-1]:.2f} = +"
                  f"{100 * (cs[-1] / q['ub'] - 1):.1f}% of optimum",
                  fontsize=9)

    axf = fig.add_subplot(2, 3, 6)
    na = [len(a) for a in alive]
    axf.step(range(1, len(na) + 1), na, where="post", color=C_IFACE, lw=2.0)
    axf.plot(range(1, len(na) + 1), na, "o", color=C_IFACE, ms=4)
    axf.axhline(n_if, color=C_PRUNED, ls=":", lw=1.4)
    axf.text(1, n_if, f" all {n_if} interfaces (before first incumbent)",
             fontsize=8, color="#4b5563", va="bottom")
    for col, k in enumerate(ks):
        axf.plot(k + 1, na[k], "o", ms=9, mfc="none", mec=C_PATH, mew=1.8)
    axf.set_xlabel("improvement event")
    axf.set_ylabel("interfaces alive")
    axf.set_ylim(0, n_if * 1.12)
    axf.grid(alpha=0.22)
    axf.set_title(f"(f) informed filter tightens with every incumbent:\n"
                  f"{n_if} $\\to$ {na[0]} at the first incumbent "
                  f"$\\to$ {na[-1]} at the last", fontsize=9)

    fig.suptitle(
        "Informed spheroid iteration in 3-D (audited UAV building, 42 "
        "rooms / 52 door-window interfaces; mid-range query, detour "
        f"ratio {q['detour']:.2f}).  Every incumbent passed the pinch / "
        "wall / containment audits; final cost within "
        f"{100 * (cs[-1] / q['ub'] - 1):.1f}% of the full-graph GCS "
        "optimum computed on the same regions.", fontsize=10, y=0.985)
    out = "out/fig_e6_3d_informed.png"
    fig.savefig(out, dpi=155, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
