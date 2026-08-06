#!/usr/bin/env python3
"""E6 figure 2: the anytime protocol -- informed, parallel, batched
interface sampling ("shortest path per unit time").

(a)/(b) incumbent-cost staircases vs wall clock on both testbeds;
(c) informed filter at work: fraction of interfaces alive per round;
(d)/(e) the informed set drawn on the map (maze prunes, IRIS cannot);
(f) worker scaling of the sampling stage.

    python3 e6_fig2.py
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
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon as MplPoly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6_sampling as e6
from e6_fig1 import poly_vertices

C_CONTOUR = "#7c3aed"
C_AREA = "#059669"
C_GCS = "#0ea5e9"
C_IFACE = "#f59e0b"
C_PRUNED = "#d4d4d8"
C_OBS = "#3f3f46"
C_REG = "#93c5fd"
C_PATH = "#dc2626"
C_MUTED = "#6b7280"


def staircase(ax, ev, t_end, color, alpha=1.0, lw=1.8, label=None):
    if not ev:
        return
    ts = [t for t, _ in ev] + [t_end]
    cs = [c for _, c in ev]
    ax.step(ts, cs + [cs[-1]], where="post", color=color, alpha=alpha,
            lw=lw, label=label)


def anytime_panel(ax, data, title):
    gt = data["gt"]
    t_end = 0.0
    for r in data["runs"]:
        if r["rows"]:
            t_end = max(t_end, sum(x["t_round"] for x in r["rows"]))
    for mode, color in (("contour", C_CONTOUR), ("area", C_AREA)):
        runs = [r for r in data["runs"]
                if r["mode"] == mode and r["informed"]]
        finals = [r["c_best"] for r in runs]
        rep = runs[int(np.argsort(finals)[len(finals) // 2])]
        for r in runs:
            staircase(ax, r["events"], t_end, color, alpha=0.30, lw=1.0)
        staircase(ax, rep["events"], t_end, color, lw=2.2,
                  label=f"{mode} (4 seeds, median shown bold)")
    ax.axhline(gt["ub"], color=C_GCS, lw=1.6, ls="--")
    ax.text(t_end, gt["ub"], f"  full-graph GCS optimum {gt['ub']:.3f}",
            color=C_GCS, fontsize=8, va="center", ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("incumbent path cost")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=9)


def alive_panel(ax, data_i, data_m):
    for data, color, name in ((data_m, C_IFACE, "maze (109 interfaces)"),
                              (data_i, C_MUTED, "IRIS (10 interfaces)")):
        runs = [r for r in data["runs"]
                if r["informed"] and r["mode"] == "area"]
        for k, r in enumerate(runs):
            fr = [row["n_ifaces_alive"] / data["n_ifaces"]
                  for row in r["rows"]]
            ax.plot(range(len(fr)), fr, "o-", color=color, ms=4,
                    lw=1.6 if k == 0 else 1.0,
                    alpha=1.0 if k == 0 else 0.35,
                    label=name if k == 0 else None)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("round")
    ax.set_ylabel("fraction of interfaces alive")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(c) informed filter: fraction of interfaces\n"
                 "alive per round", fontsize=9)


def informed_map(ax, libfile, data, run_sel, title, walls=False):
    lib = pickle.load(open(libfile, "rb"))
    topo = lib["topo"]
    ifaces = e6.classify_interfaces(topo)
    run = next(r for r in data["runs"] if
               (r["mode"], r["informed"], r["seed"]) == run_sel)
    alive = set(run["alive_ifaces_final"])
    s, g = np.asarray(data["s"]), np.asarray(data["g"])
    c = run["c_best"]

    for A, b in topo["regions"]:
        V = poly_vertices(np.asarray(A), np.asarray(b))
        if V is not None and len(V) >= 3:
            ax.add_patch(MplPoly(V, closed=True, facecolor=C_REG,
                                 edgecolor="#1d4ed8", lw=0.4, alpha=0.20))
    if "obstacles" in lib:
        for ob in lib["obstacles"]:
            ax.add_patch(MplPoly(np.asarray(ob), closed=True,
                                 facecolor=C_OBS, edgecolor="k", lw=0.5,
                                 zorder=3))
    if walls:
        for w in e6.box_walls(topo):
            k, cd = w["dim"], w["coord"]
            (m, lo_, hi_) = w["spans"][0]
            p1, p2 = np.zeros(2), np.zeros(2)
            p1[k] = p2[k] = cd
            p1[m], p2[m] = lo_, hi_
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C_OBS, lw=2.2,
                    zorder=3, solid_capstyle="butt")
    for eid, f in ifaces.items():
        V = poly_vertices(f["A"], f["b"])
        if V is None:
            continue
        col = C_IFACE if eid in alive else C_PRUNED
        z = 4 if eid in alive else 2
        if f["dim"] >= 2 and len(V) >= 3:
            ax.add_patch(MplPoly(V, closed=True, facecolor=col,
                                 edgecolor="none", alpha=0.75, zorder=z))
        elif len(V) >= 2:
            ax.plot(V[:, 0], V[:, 1], color=col, lw=2.4, zorder=z)
    # informed ellipse for the final incumbent
    f2 = np.linalg.norm(g - s)
    if np.isfinite(c) and c > f2:
        ctr = 0.5 * (s + g)
        ang = np.degrees(np.arctan2(g[1] - s[1], g[0] - s[0]))
        ax.add_patch(Ellipse(ctr, width=c, height=np.sqrt(c * c - f2 * f2),
                             angle=ang, fill=False, color=C_PATH, lw=1.4,
                             ls="--", zorder=6))
    p = np.asarray(run["best_path"])
    ax.plot(p[:, 0], p[:, 1], "-", color=C_PATH, lw=2.2, zorder=7)
    ax.plot(*s, "*", ms=13, color="#16a34a", zorder=8)
    ax.plot(*g, "*", ms=13, color=C_PATH, zorder=8)
    ax.set_xlim(-0.25, 5.25)
    ax.set_ylim(-0.25, 5.25)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def worker_panel(ax, data_i, data_m):
    shades = ["#9ca3af", "#4b5563", "#111827"]          # 1 / 2 / 4 workers
    groups = [("IRIS", data_i["timing"]), ("maze", data_m["timing"])]
    xw = 0.24
    for gi, (name, tim) in enumerate(groups):
        for k, t in enumerate(tim):
            x = gi + (k - 1) * xw
            ax.bar(x, t["t_sample_total"], width=xw * 0.92,
                   color=shades[k], edgecolor="white", linewidth=1.2)
            ax.text(x, t["t_sample_total"], f" {t['t_sample_total']:.2f}",
                    ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([n for n, _ in groups])
    ax.set_ylabel("sampling time, 4 heavy rounds (s)")
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in shades]
    ax.legend(hs, ["1 worker", "2 workers", "4 workers"], fontsize=8)
    sp = data_i["timing"][0]["t_sample_total"] / \
        data_i["timing"][-1]["t_sample_total"]
    ax.set_title("(f) parallel sampling stage, M=64/interface\n"
                 f"(IRIS {sp:.1f}$\\times$ w/ 4 workers; totals are "
                 "solve-dominated)", fontsize=9)
    ax.grid(alpha=0.22, axis="y")


def main():
    data_i = json.load(open("out/e6_step2_iris.json"))
    data_m = json.load(open("out/e6_step2_maze.json"))

    fig = plt.figure(figsize=(14.5, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.26)

    anytime_panel(fig.add_subplot(gs[0, 0]), data_i,
                  "(a) IRIS testbed: incumbent cost vs time\n"
                  "(informed on; batches of M=2 per alive interface)")
    anytime_panel(fig.add_subplot(gs[0, 1]), data_m,
                  "(b) maze: incumbent cost vs time\n"
                  "(area $\\to$ +0.5%; inset-contour plateaus)")
    alive_panel(fig.add_subplot(gs[0, 2]), data_i, data_m)
    informed_map(fig.add_subplot(gs[1, 0]), "out/maze2d_100.pkl", data_m,
                 ("area", True, 0),
                 "(d) maze: informed set at the final incumbent\n"
                 "amber = alive interfaces (80$\\to$70), gray = pruned, "
                 "dashed = ellipse", walls=True)
    informed_map(fig.add_subplot(gs[1, 1]), "out/e6_iris2d.pkl", data_i,
                 ("contour", True, 3),
                 "(e) IRIS: ellipse $\\supset$ map, zero pruning\n"
                 "(detour ratio 1.7: effectiveness law in miniature)")
    worker_panel(fig.add_subplot(gs[1, 2]), data_i, data_m)

    fig.suptitle("Anytime interface sampling: informed + parallel + "
                 "batched.  Soundness: every accepted incumbent passes the "
                 "pinch/wall/penetration audits, informed pruning is "
                 "admissible (identical costs with the filter on/off on "
                 "every seed), and every final cost $\\geq$ the full-graph "
                 "GCS lower bound.", fontsize=10, y=0.99)
    out = "out/fig_e6_anytime.png"
    fig.savefig(out, dpi=165, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
