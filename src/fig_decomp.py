#!/usr/bin/env python3
"""Two teaching figures for the method's geometry, each a 3-panel triptych:
  (a) available free space (obstacles carved out)
  (b) IRIS "inflation" convex decomposition (overlapping convex regions)
  (c) portals only — the two kinds of region adjacency the method keeps:
        volume overlap  (X_i ∩ X_j has positive measure)
        face touching   (X_i ∩ X_j is a shared facet, zero measure)

    python3 fig_decomp.py   -> out/fig_decomp_2d.png, out/fig_decomp_3d.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COL = ["#2a78d6", "#eb6834", "#008300", "#8a5cc7", "#eda100"]
OBST = "#2b2b2b"
GRID = "#cdc9c1"
OVL = "#c0362c"          # volume-overlap portal
FACE = "#d4179a"         # face-touching portal
INK = "#14181d"
MUT = "#5c5a55"
SURF = "#ffffff"

LEG = [Patch(fc=OVL, ec="white", label="volume-overlap portal   "
             "(X_i ∩ X_j has positive area/volume)"),
       Line2D([0], [0], color=FACE, lw=6, solid_capstyle="round",
              label="face-touching portal   (shared facet, zero measure)")]


def cap(ax, s):
    ax.text(0.5, -0.045, s, transform=ax.transAxes, ha="center", va="top",
            color=MUT, fontsize=10)


# ----------------------------------------------------------------------- 2D
def fig_2d():
    R = {"A": (0.4, 0.4, 5.0, 7.6), "B": (4.0, 3.2, 8.0, 4.8),
         "C": (7.0, 0.4, 11.6, 4.0), "D": (7.0, 4.0, 11.6, 7.6)}
    order = ["A", "B", "C", "D"]
    walls = [(0.0, 0.0, 12.0, 0.4), (0.0, 7.6, 12.0, 8.0),
             (0.0, 0.0, 0.4, 8.0), (11.6, 0.0, 12.0, 8.0),
             (5.0, 0.4, 7.0, 3.2), (5.0, 4.8, 7.0, 7.6)]

    def draw_walls(ax):
        for x0, y0, x1, y1 in walls:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=OBST,
                                   ec="none", zorder=2))

    def setup(ax, title):
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12.5, color=INK, pad=7)
        for s in ax.spines.values():
            s.set_edgecolor(GRID)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), facecolor=SURF)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.83, bottom=0.14,
                        wspace=0.07)

    # (a)
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0), 12, 8, fc="#e9edf1", ec="none"))
    for k in order:
        x0, y0, x1, y1 = R[k]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="white",
                               ec="none", zorder=1))
    draw_walls(ax)
    setup(ax, "(a)  available free space")
    ax.text(2.7, 4.0, "free", color=MUT, ha="center", fontsize=12)
    ax.text(6.0, 6.2, "wall", color="white", ha="center", fontsize=10.5)
    cap(ax, "white = free space to plan in      black = obstacles")

    # (b)
    ax = axes[1]
    ax.add_patch(Rectangle((0, 0), 12, 8, fc="#f3f1ec", ec="none"))
    draw_walls(ax)
    for idx, k in enumerate(order):
        x0, y0, x1, y1 = R[k]
        c = COL[idx]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=c, ec=c,
                               alpha=0.28, lw=2.4, zorder=3))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.add_patch(Ellipse((cx, cy), 0.72 * (x1 - x0), 0.72 * (y1 - y0),
                             fill=False, ec=c, ls=(0, (4, 3)), lw=1.4,
                             alpha=0.9, zorder=4))
        ax.plot([cx], [cy], marker="*", ms=14, color=c, mec="white",
                mew=0.9, zorder=5)
        ax.text(cx + 0.28, cy, k, color=INK, fontsize=12, ha="left",
                va="center", fontweight="bold", zorder=6)
    setup(ax, "(b)  IRIS convex decomposition (inflated seeds)")
    cap(ax, "seeds inflate into overlapping convex regions")

    # (c)
    ax = axes[2]
    ax.add_patch(Rectangle((0, 0), 12, 8, fc="#fbfaf8", ec="none"))
    for k in order:
        x0, y0, x1, y1 = R[k]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none",
                               ec=GRID, ls=(0, (5, 4)), lw=1.3, zorder=2))
    n_ovl = n_face = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = R[order[i]], R[order[j]]
            xl, yl = max(a[0], b[0]), max(a[1], b[1])
            xh, yh = min(a[2], b[2]), min(a[3], b[3])
            if xl > xh + 1e-9 or yl > yh + 1e-9:
                continue
            if xh - xl > 1e-6 and yh - yl > 1e-6:
                ax.add_patch(Rectangle((xl, yl), xh - xl, yh - yl, fc=OVL,
                                       ec="white", lw=1.3, alpha=0.95,
                                       zorder=5))
                n_ovl += 1
            else:
                ax.plot([xl, xh], [yl, yh], color=FACE, lw=6,
                        solid_capstyle="round", zorder=6)
                n_face += 1
    setup(ax, "(c)  portals only  (the decomposition graph)")
    cap(ax, f"keep ONLY the portals:  {n_ovl} volume-overlap  +  "
        f"{n_face} face-touching")

    fig.legend(handles=LEG, loc="lower center", ncol=2, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.6, columnspacing=2.4)
    fig.suptitle("Query-independent geometry:   free space   →   overlapping "
                 "convex regions   →   portal graph  (2D)",
                 fontsize=14, color=INK, y=0.965)
    fig.savefig("out/fig_decomp_2d.png", dpi=150, facecolor=SURF,
                bbox_inches="tight")
    print("wrote out/fig_decomp_2d.png")


# ----------------------------------------------------------------------- 3D
def box_faces(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return [[c[0], c[1], c[2], c[3]], [c[4], c[5], c[6], c[7]],
            [c[0], c[1], c[5], c[4]], [c[2], c[3], c[7], c[6]],
            [c[1], c[2], c[6], c[5]], [c[0], c[3], c[7], c[4]]]


def fig_3d():
    R = {"A": ((0.0, 0.0, 0.0), (6.0, 6.0, 2.5)),
         "B": ((0.0, 0.0, 2.5), (6.0, 6.0, 5.0)),
         "C": ((4.0, 2.0, 0.5), (9.0, 4.0, 2.0)),
         "D": ((8.0, 0.0, 0.0), (13.0, 6.0, 2.5))}
    order = ["A", "B", "C", "D"]

    def setup(ax, title):
        ax.set_title(title, fontsize=12.5, color=INK, pad=-2)
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 8)
        ax.set_zlim(0, 6)
        ax.set_box_aspect((13, 8, 6.5))
        ax.view_init(elev=22, azim=-58)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_alpha(0.0)
            a.line.set_color(GRID)

    def add_box(ax, lo, hi, color, alpha, ec, lw=1.0, ls="-"):
        pc = Poly3DCollection(box_faces(lo, hi), facecolor=color,
                              edgecolor=ec, alpha=alpha, linewidths=lw)
        pc.set_linestyle(ls)
        ax.add_collection3d(pc)

    fig = plt.figure(figsize=(16, 5.4), facecolor=SURF)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.1,
                        wspace=0.02)

    ax = fig.add_subplot(131, projection="3d")
    for k in order:
        add_box(ax, R[k][0], R[k][1], "#c9cdd2", 0.15, "#9aa0a6", 0.8)
    setup(ax, "(a)  available free workspace")
    ax.text2D(0.5, 0.02, "the connected free space (obstacles = the "
              "complement)", transform=ax.transAxes, ha="center",
              color=MUT, fontsize=10)

    ax = fig.add_subplot(132, projection="3d")
    for idx, k in enumerate(order):
        lo, hi = R[k]
        add_box(ax, lo, hi, COL[idx], 0.24, COL[idx], 1.5)
        c = 0.5 * (np.array(lo) + np.array(hi))
        ax.scatter([c[0]], [c[1]], [c[2]], marker="*", s=170,
                   color=COL[idx], edgecolors="white", linewidths=0.9,
                   depthshade=False)
        ax.text(c[0], c[1], c[2] + 0.35, k, color=INK, fontsize=13,
                fontweight="bold", ha="center")
    setup(ax, "(b)  IRIS convex decomposition (inflated boxes)")
    ax.text2D(0.5, 0.02, "convex boxes grown from seeds, overlapping to "
              "cover the workspace", transform=ax.transAxes, ha="center",
              color=MUT, fontsize=10)

    ax = fig.add_subplot(133, projection="3d")
    for k in order:
        add_box(ax, R[k][0], R[k][1], GRID, 0.035, GRID, 0.7, ls=(0, (4, 4)))
    n_ovl = n_face = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = R[order[i]], R[order[j]]
            lo = np.maximum(a[0], b[0])
            hi = np.minimum(a[1], b[1])
            d = hi - lo
            if np.any(d < -1e-9):
                continue
            pos = d > 1e-6
            if np.all(pos):
                add_box(ax, lo, hi, OVL, 0.9, "white", 1.0)
                n_ovl += 1
            elif pos.sum() == 2:
                ax0 = int(np.argmin(d))
                a2 = [k for k in range(3) if k != ax0]
                v = lo[ax0]
                pts = []
                for sx in (lo[a2[0]], hi[a2[0]]):
                    for sy in (lo[a2[1]], hi[a2[1]]):
                        q = [0, 0, 0]
                        q[ax0], q[a2[0]], q[a2[1]] = v, sx, sy
                        pts.append(tuple(q))
                pts = [pts[0], pts[1], pts[3], pts[2]]
                ax.add_collection3d(Poly3DCollection(
                    [pts], facecolor=FACE, edgecolor="white", alpha=0.82,
                    linewidths=1.3))
                n_face += 1
    setup(ax, "(c)  portals only  (the decomposition graph)")
    ax.text2D(0.5, 0.02, f"keep ONLY: {n_ovl} volume-overlap (red boxes) "
              f" +  {n_face} face-touching (magenta facet)",
              transform=ax.transAxes, ha="center", color=MUT, fontsize=10)

    fig.legend(handles=LEG, loc="lower center", ncol=2, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.015),
               handlelength=1.6, columnspacing=2.4)
    fig.suptitle("Same idea in 3D:   free workspace   →   overlapping "
                 "convex boxes   →   portal graph",
                 fontsize=14, color=INK, y=0.98)
    fig.savefig("out/fig_decomp_3d.png", dpi=150, facecolor=SURF,
                bbox_inches="tight")
    print("wrote out/fig_decomp_3d.png")


if __name__ == "__main__":
    fig_2d()
    fig_3d()


# ------------------------------------------------------ complex 2D building
def fig_2d_complex():
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union

    R = {"A": (1.0, 1.0, 7.0, 6.0), "B": (1.0, 9.0, 7.0, 14.0),
         "C": (3.0, 4.0, 6.0, 11.0), "D": (4.5, 2.0, 13.0, 5.0),
         "E": (10.0, 5.0, 15.0, 10.0), "F": (8.0, 10.0, 14.0, 13.0),
         "G": (16.0, 9.0, 23.0, 14.0), "H": (16.0, 1.0, 23.0, 6.0),
         "I": (18.0, 4.0, 21.0, 11.0), "J": (14.0, 6.0, 18.0, 9.0)}
    order = list(R)
    PAL = ["#2a78d6", "#eb6834", "#008300", "#8a5cc7", "#eda100",
           "#0f9e9e", "#d0417e", "#6b8e23", "#7b6cf0", "#c25b2e"]
    col = {k: PAL[i % len(PAL)] for i, k in enumerate(order)}
    DOM = (0, 24, 0, 15)
    dom_poly = box(0, 0, 24, 15)
    walls = dom_poly.difference(unary_union([box(*R[k]) for k in order]))

    def draw_walls(ax):
        polys = walls.geoms if walls.geom_type == "MultiPolygon" else [walls]
        for p in polys:
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, fc=OBST, ec="none", zorder=2)

    def setup(ax, title):
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 15)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12.5, color=INK, pad=7)
        for s in ax.spines.values():
            s.set_edgecolor(GRID)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4), facecolor=SURF)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.83, bottom=0.14,
                        wspace=0.06)

    # (a)
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0), 24, 15, fc=OBST, ec="none"))
    for k in order:
        x0, y0, x1, y1 = R[k]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="white",
                               ec="none", zorder=1))
    setup(ax, "(a)  available free space")
    cap(ax, "white = free space to plan in      black = obstacles / walls")

    # (b)
    ax = axes[1]
    ax.add_patch(Rectangle((0, 0), 24, 15, fc="#f3f1ec", ec="none"))
    draw_walls(ax)
    for k in order:
        x0, y0, x1, y1 = R[k]
        c = col[k]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=c, ec=c,
                               alpha=0.24, lw=2.0, zorder=3))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.add_patch(Ellipse((cx, cy), 0.66 * (x1 - x0), 0.66 * (y1 - y0),
                             fill=False, ec=c, ls=(0, (4, 3)), lw=1.2,
                             alpha=0.85, zorder=4))
        ax.plot([cx], [cy], marker="*", ms=12, color=c, mec="white",
                mew=0.8, zorder=5)
        ax.text(cx + 0.32 * (x1 - x0), cy + 0.30 * (y1 - y0), k, color=INK,
                fontsize=10.5, ha="center", va="center", fontweight="bold",
                zorder=6)
    setup(ax, "(b)  IRIS convex decomposition (10 inflated regions)")
    cap(ax, "seeds inflate into overlapping convex regions covering the "
        "free space")

    # (c) portals + portal graph
    ax = axes[2]
    ax.add_patch(Rectangle((0, 0), 24, 15, fc="#fbfaf8", ec="none"))
    for k in order:
        x0, y0, x1, y1 = R[k]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none",
                               ec=GRID, ls=(0, (5, 4)), lw=1.1, zorder=2))
    ctr = {k: ((R[k][0] + R[k][2]) / 2, (R[k][1] + R[k][3]) / 2)
           for k in order}
    n_ovl = n_face = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = R[order[i]], R[order[j]]
            inter = box(*a).intersection(box(*b))
            if inter.is_empty:
                continue
            ci, cj = ctr[order[i]], ctr[order[j]]
            if inter.geom_type in ("Polygon",) and inter.area > 1e-6:
                xs, ys = inter.exterior.xy
                ax.fill(xs, ys, fc=OVL, ec="white", lw=1.0, alpha=0.9,
                        zorder=5)
                ax.plot([ci[0], cj[0]], [ci[1], cj[1]], color=OVL, lw=1.4,
                        alpha=0.55, zorder=4)
                n_ovl += 1
            elif inter.geom_type == "LineString" and inter.length > 1e-6:
                xs, ys = inter.xy
                ax.plot(xs, ys, color=FACE, lw=5.5, solid_capstyle="round",
                        zorder=6)
                ax.plot([ci[0], cj[0]], [ci[1], cj[1]], color=FACE, lw=1.4,
                        alpha=0.5, ls=(0, (2, 2)), zorder=4)
                n_face += 1
    for k in order:                      # graph nodes
        ax.plot([ctr[k][0]], [ctr[k][1]], marker="o", ms=11, color="#20242b",
                mec="white", mew=1.3, zorder=7)
        ax.text(ctr[k][0], ctr[k][1], k, color="white", fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=8)
    setup(ax, "(c)  portals + portal graph")
    cap(ax, f"the query-independent graph:  {len(order)} nodes,  "
        f"{n_ovl} volume-overlap + {n_face} face-touching portals")

    fig.legend(handles=LEG, loc="lower center", ncol=2, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.6, columnspacing=2.4)
    fig.suptitle("A richer floor-plan:   free space   →   overlapping convex "
                 "regions   →   portal graph  (2D)", fontsize=14,
                 color=INK, y=0.965)
    fig.savefig("out/fig_decomp_2d_complex.png", dpi=150, facecolor=SURF,
                bbox_inches="tight")
    print("wrote out/fig_decomp_2d_complex.png")


if __name__ == "__main__":
    fig_2d_complex()


# ------------------------------------------------ complex 3D two-storey bldg
def fig_3d_complex():
    R = {"A": ((1, 1, 0.0), (8, 6, 2.4)), "B": ((1, 6, 0.0), (8, 10, 2.4)),
         "C": ((6, 2, 0.0), (16, 9, 2.4)), "D": ((14, 1, 0.0), (23, 10, 2.4)),
         "E": ((1, 1, 2.8), (9, 5.5, 5.2)),
         "H": ((1, 5.5, 2.8), (9, 10, 5.2)),
         "F": ((7, 2, 2.8), (17, 9, 5.2)), "G": ((15, 1, 2.8), (23, 10, 5.2)),
         "S": ((7, 3, 0.0), (10, 6, 5.2))}
    order = list(R)
    PAL = ["#2a78d6", "#eb6834", "#008300", "#8a5cc7", "#eda100",
           "#0f9e9e", "#d0417e", "#6b8e23", "#c25b2e"]
    col = {k: PAL[i] for i, k in enumerate(order)}

    def setup(ax, title):
        ax.set_title(title, fontsize=12.5, color=INK, pad=-2)
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 11)
        ax.set_zlim(0, 5.4)
        ax.set_box_aspect((24, 11, 8))
        ax.view_init(elev=20, azim=-58)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_alpha(0.0)
            a.line.set_color(GRID)

    def add_box(ax, lo, hi, color, alpha, ec, lw=1.0, ls="-"):
        pc = Poly3DCollection(box_faces(lo, hi), facecolor=color,
                              edgecolor=ec, alpha=alpha, linewidths=lw)
        pc.set_linestyle(ls)
        ax.add_collection3d(pc)

    def facet(ax, lo, hi, ax0):
        a2 = [k for k in range(3) if k != ax0]
        v = lo[ax0]
        pts = []
        for sx in (lo[a2[0]], hi[a2[0]]):
            for sy in (lo[a2[1]], hi[a2[1]]):
                q = [0, 0, 0]
                q[ax0], q[a2[0]], q[a2[1]] = v, sx, sy
                pts.append(tuple(q))
        pts = [pts[0], pts[1], pts[3], pts[2]]
        ax.add_collection3d(Poly3DCollection([pts], facecolor=FACE,
                            edgecolor="white", alpha=0.85, linewidths=1.2))

    fig = plt.figure(figsize=(16.5, 5.4), facecolor=SURF)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.1, wspace=0.02)

    ax = fig.add_subplot(131, projection="3d")
    for k in order:
        add_box(ax, R[k][0], R[k][1], "#c9cdd2", 0.13, "#9aa0a6", 0.7)
    setup(ax, "(a)  available free workspace (2-storey)")
    ax.text2D(0.5, 0.03, "connected free space of a two-floor building",
              transform=ax.transAxes, ha="center", color=MUT, fontsize=10)

    ax = fig.add_subplot(132, projection="3d")
    for k in order:
        lo, hi = np.array(R[k][0]), np.array(R[k][1])
        add_box(ax, lo, hi, col[k], 0.20, col[k], 1.4)
        c = 0.5 * (lo + hi)
        ax.text(c[0], c[1], c[2], k, color=INK, fontsize=11,
                fontweight="bold", ha="center", va="center")
    setup(ax, "(b)  IRIS decomposition (9 regions + stairwell S)")
    ax.text2D(0.5, 0.03, "rooms & corridors per floor, joined vertically by "
              "stairwell S", transform=ax.transAxes, ha="center",
              color=MUT, fontsize=10)

    ax = fig.add_subplot(133, projection="3d")
    for k in order:
        add_box(ax, R[k][0], R[k][1], GRID, 0.03, GRID, 0.6, ls=(0, (3, 4)))
    ctr = {k: 0.5 * (np.array(R[k][0]) + np.array(R[k][1])) for k in order}
    n_ovl = n_face = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = R[order[i]], R[order[j]]
            lo = np.maximum(a[0], b[0])
            hi = np.minimum(a[1], b[1])
            d = hi - lo
            if np.any(d < -1e-9):
                continue
            ci, cj = ctr[order[i]], ctr[order[j]]
            pos = d > 1e-6
            if np.all(pos):
                add_box(ax, lo, hi, OVL, 0.9, "white", 0.8)
                ax.plot(*zip(ci, cj), color=OVL, lw=1.2, alpha=0.5)
                n_ovl += 1
            elif pos.sum() == 2:
                facet(ax, lo, hi, int(np.argmin(d)))
                ax.plot(*zip(ci, cj), color=FACE, lw=1.2, alpha=0.5,
                        ls=(0, (2, 2)))
                n_face += 1
    for k in order:
        ax.scatter(*ctr[k], s=120, color="#20242b", edgecolors="white",
                   linewidths=1.2, depthshade=False, zorder=6)
        ax.text(ctr[k][0], ctr[k][1], ctr[k][2], k, color="white",
                fontsize=7.5, ha="center", va="center", fontweight="bold")
    setup(ax, "(c)  portals + portal graph")
    ax.text2D(0.5, 0.03, f"{len(order)} nodes,  {n_ovl} volume-overlap + "
              f"{n_face} face-touching portals", transform=ax.transAxes,
              ha="center", color=MUT, fontsize=10)

    fig.legend(handles=LEG, loc="lower center", ncol=2, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.015),
               handlelength=1.6, columnspacing=2.4)
    fig.suptitle("A richer building in 3D:   free workspace   →   overlapping "
                 "convex boxes   →   portal graph", fontsize=14, color=INK,
                 y=0.98)
    fig.savefig("out/fig_decomp_3d_complex.png", dpi=150, facecolor=SURF,
                bbox_inches="tight")
    print("wrote out/fig_decomp_3d_complex.png")


if __name__ == "__main__":
    fig_3d_complex()
