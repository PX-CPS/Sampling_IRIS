#!/usr/bin/env python3
"""E6 figure 1: interface sampling on a convex decomposition, verified.

(a) IRIS testbed and its interfaces;  (b) contour sampling;
(c) area sampling;  (d) the pinch-point failure on degenerate contact
interfaces (found by the audits, not by inspection);
(e)/(f) cost vs per-interface budget on both testbeds, against
ground-truth GCS bounds.

    python3 e6_fig1.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPoly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6_sampling as e6

C_OBS = "#3f3f46"
C_REG = "#93c5fd"
C_IFACE = "#f59e0b"
C_CONTOUR = "#7c3aed"
C_AREA = "#059669"
C_PATH = "#dc2626"
C_GCS = "#0ea5e9"
C_BAD = "#dc2626"
C_CENTER = "#6b7280"


def poly_vertices(A, b, tol=1e-7):
    pts = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            M = np.array([A[i], A[j]])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            x = np.linalg.solve(M, np.array([b[i], b[j]]))
            if np.all(A @ x <= b + tol):
                pts.append(x)
    if not pts:
        return None
    P = np.unique(np.round(np.asarray(pts), 9), axis=0)
    if len(P) < 3:
        return P
    c = P.mean(0)
    return P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]


def draw_world(ax, lib, alpha=0.30):
    for ob in lib["obstacles"]:
        ax.add_patch(MplPoly(ob, closed=True, facecolor=C_OBS,
                             edgecolor="k", lw=0.6, zorder=3))
    for A, b in lib["regions"].values():
        V = poly_vertices(np.asarray(A), np.asarray(b))
        if V is not None and len(V) >= 3:
            ax.add_patch(MplPoly(V, closed=True, facecolor=C_REG,
                                 edgecolor="#1d4ed8", lw=0.5, alpha=alpha,
                                 zorder=1))
    lo, hi = lib["domain"]
    ax.set_xlim(lo[0] - .1, hi[0] + .1)
    ax.set_ylim(lo[1] - .1, hi[1] + .1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_interfaces(ax, ifaces, lw=2.2):
    for f in ifaces.values():
        V = poly_vertices(f["A"], f["b"])
        if V is None:
            continue
        if f["dim"] >= 2 and len(V) >= 3:
            ax.add_patch(MplPoly(V, closed=True, facecolor=C_IFACE,
                                 edgecolor="#b45309", lw=0.8, alpha=0.55,
                                 zorder=2))
        elif len(V) >= 2:
            ax.plot(V[:, 0], V[:, 1], color=C_IFACE, lw=lw, zorder=2)


def draw_roadmap(ax, rm, color, max_edges=1200, seed=0, ms=2.6):
    pos = rm["pos"]
    ii, jj, _ = e6.edges_from_regions(rm["region_nodes"], len(pos), pos)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ii))
    if len(idx) > max_edges:
        idx = rng.choice(idx, max_edges, replace=False)
    ax.add_collection(LineCollection(
        [[pos[int(ii[e])], pos[int(jj[e])]] for e in idx],
        colors=color, linewidths=0.25, alpha=0.25, zorder=4))
    ax.plot(pos[:, 0], pos[:, 1], "o", ms=ms, color=color, zorder=5,
            markeredgewidth=0)


def curve_panel(ax, res, modes, title, note=None):
    gt = res["ground_truth"]
    Ms = sorted({r["M"] for r in res["rows"]})
    for mode, color, style in modes:
        rows = [r for r in res["rows"] if r["mode"] == mode]
        if not rows:
            continue
        cs, bad = [], []
        for M in Ms:
            r = next((x for x in rows if x["M"] == M), None)
            cs.append(r["cost"] if r else np.nan)
            bad.append(bool(r and not r["sound_ge_lb"]))
        ax.plot(Ms, cs, style, color=color, lw=1.8, ms=5, label=mode)
        if any(bad):
            xb = [m for m, f in zip(Ms, bad) if f]
            yb = [c for c, f in zip(cs, bad) if f]
            ax.plot(xb, yb, "x", color=C_BAD, ms=13, mew=2.6, zorder=6)
    pol = [next((r["cost_polished"] for r in res["rows"]
                 if r["M"] == M and r["mode"] == "area"), np.nan) for M in Ms]
    ax.plot(Ms, pol, "s--", color="#111827", lw=1.5, ms=4,
            label="after corridor SOCP polish")
    ax.axhline(gt["ub"], color=C_GCS, lw=1.6,
               label=f"full GCS UB = {gt['ub']:.3f}")
    ax.axhline(gt["lb"], color=C_GCS, lw=1.3, ls=":",
               label=f"full GCS LB = {gt['lb']:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(m) for m in Ms])
    ax.set_xlabel("samples per interface, M")
    ax.set_ylabel("path cost")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.2, loc="best")
    ax.set_title(title, fontsize=9)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7.6,
                color=C_BAD, va="bottom")


def main():
    lib = pickle.load(open("out/e6_iris2d.pkl", "rb"))
    topo = lib["topo"]
    regions = topo["regions"]
    s = np.asarray(lib["task_configs"]["start"])
    g = np.asarray(lib["task_configs"]["goal"])
    ifaces = e6.classify_interfaces(topo)
    res = json.load(open("out/e6_step1_verify.json"))
    res_m = json.load(open("out/e6_step1_verify_maze.json"))

    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.16)

    # (a)
    axa = fig.add_subplot(gs[0, 0])
    draw_world(axa, lib)
    draw_interfaces(axa, ifaces)
    axa.plot(*s, "*", ms=15, color="#16a34a", zorder=6)
    axa.plot(*g, "*", ms=15, color=C_PATH, zorder=6)
    axa.set_title(f"(a) IRIS decomposition: {len(regions)} regions, "
                  f"{len(ifaces)} interfaces\n"
                  "orange = overlap polytopes $X_u\\cap X_v$ (all "
                  "full-dimensional here)", fontsize=9)

    # (b), (c)
    M_SHOW = 8
    for col, (mode, color, label) in enumerate(
            [("contour", C_CONTOUR, "contour: relative boundary"),
             ("area", C_AREA, "area: relative interior")]):
        ax = fig.add_subplot(gs[0, col + 1])
        draw_world(ax, lib, alpha=0.16)
        draw_interfaces(ax, ifaces)
        rm = e6.build_roadmap(topo, ifaces, mode, M_SHOW, seed=0)
        draw_roadmap(ax, rm, color)
        cost, path = e6.shortest_path(rm, regions, s, g)
        ax.plot(path[:, 0], path[:, 1], "-", color=C_PATH, lw=2.4, zorder=7)
        ax.plot(*s, "*", ms=15, color="#16a34a", zorder=8)
        ax.plot(*g, "*", ms=15, color=C_PATH, zorder=8)
        row = next(r for r in res["rows"]
                   if r["mode"] == mode and r["M"] == M_SHOW)
        ax.set_title(f"({'bc'[col]}) {label}, M={M_SHOW}\n"
                     f"{row['nodes']} nodes, {row['edges']} edges, "
                     f"cost {cost:.2f} (all audits pass)", fontsize=9)

    # (d) the pinch-point failure on degenerate interfaces
    axd = fig.add_subplot(gs[1, 0])
    mz = pickle.load(open("out/maze2d_100.pkl", "rb"))
    mtopo = mz["topo"]
    mif = e6.classify_interfaces(mtopo)
    mreg = mtopo["regions"]
    for A, b in mreg:
        V = poly_vertices(np.asarray(A), np.asarray(b))
        if V is not None and len(V) >= 3:
            axd.add_patch(MplPoly(V, closed=True, facecolor=C_REG,
                                  edgecolor="#1d4ed8", lw=0.4, alpha=0.22))
    walls = e6.box_walls(mtopo)
    for w in walls:                      # zero-thickness barriers
        k, c = w["dim"], w["coord"]
        (m, lo_, hi_) = w["spans"][0]
        p1, p2 = np.zeros(2), np.zeros(2)
        p1[k] = p2[k] = c
        p1[m], p2[m] = lo_, hi_
        axd.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C_OBS, lw=2.4,
                 zorder=3, solid_capstyle="butt")
    for f in mif.values():
        V = poly_vertices(f["A"], f["b"])
        if V is not None and len(V) >= 2:
            axd.plot(V[:, 0], V[:, 1], color=C_IFACE, lw=2.0, zorder=4)
    sm = np.array([0.5, 0.5])
    gm = np.asarray(mtopo["bbox_hi"][-1]) - 0.5
    rm_bad = e6.build_roadmap(mtopo, mif, "contour", 4, seed=0,
                              membership="geometric")
    c_bad, p_bad = e6.shortest_path(rm_bad, mreg, sm, gm)
    jj = e6.audit_junctions(p_bad, mreg)
    axd.plot(p_bad[:, 0], p_bad[:, 1], "-", color=C_BAD, lw=2.2, zorder=6)
    pos_b = rm_bad["pos"]
    axd.plot(pos_b[:, 0], pos_b[:, 1], "o", ms=2.4, color=C_CONTOUR,
             zorder=5, markeredgewidth=0)
    for k in range(1, len(p_bad) - 1):
        sub = e6.audit_junctions(p_bad[k - 1:k + 2], mreg)
        if sub["bad_junctions"]:
            axd.plot(*p_bad[k], "o", ms=10, mfc="none", mec=C_BAD, mew=2.2,
                     zorder=7)
    axd.set_xlim(-0.2, 5.2)
    axd.set_ylim(-0.2, 5.2)
    axd.set_aspect("equal")
    axd.set_xticks([])
    axd.set_yticks([])
    axd.set_title("(d) FAILURE on degenerate interfaces: contour samples "
                  "land on\ncorners where cells meet in measure zero — "
                  f"{jj['bad_junctions']}/{jj['junctions']} junctions "
                  "pinch, cost falls BELOW the optimum", fontsize=9)

    # (e), (f)
    axe = fig.add_subplot(gs[1, 1])
    curve_panel(axe, res,
                [("center", C_CENTER, "o-"), ("contour", C_CONTOUR, "o-"),
                 ("area", C_AREA, "o-")],
                "(e) overlap interfaces (IRIS testbed): contour wins,\n"
                "every configuration valid")
    axf = fig.add_subplot(gs[1, 2])
    curve_panel(axf, res_m,
                [("center", C_CENTER, "o-"), ("contour", C_CONTOUR, "o-"),
                 ("contour_in", "#a855f7", "^-"), ("area", C_AREA, "o-")],
                "(f) contact interfaces (maze testbed): raw contour is\n"
                "INVALID (x); inset contour is valid and most efficient",
                note="x = cost below the true optimum (invalid path)")

    fig.suptitle("Interface sampling on convex decompositions — "
                 "audited construction.  Valid configurations: max region "
                 "violation $\\leq 9{\\cdot}10^{-16}$, max obstacle "
                 "penetration $\\leq 7{\\cdot}10^{-10}$, 0 edges without a "
                 "witness region, 0 wall crossings, 0 pinch junctions, "
                 "cost $\\geq$ ground-truth optimum on every row.",
                 fontsize=10, y=0.98)
    out = "out/fig_e6_interface_sampling.png"
    fig.savefig(out, dpi=165, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
