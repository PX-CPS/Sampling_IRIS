#!/usr/bin/env python3
"""E13: the 2-D map of the whole story on one picture.

A genuine IRIS decomposition of a polygonal world, with one small
collision pocket injected into a region -- present in the ground truth,
invisible to the region description (exactly what a sampling-based
C-space generator leaves behind).  The geometric optimum (what every
GCS-family planner returns) cuts straight through the pocket; our
layer detects it, repairs in-region, and returns a certified path a
few percent longer.  Same regions, same query, opposite verdicts.

    python3 e13_fig2d_contamination.py   # out/fig_2d_contamination.png
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
import e6_sampling as e6
import stageB_bimanual_ipgcs as sb
from e6_build_iris2d import penetration_depth
from e8_validate import path_cost, repair_path

POCKET_HALF = (0.16, 0.07)             # small box, generator never saw it


class Validator2D:
    """Ground-truth validator with the e8_validate interface."""

    def __init__(self, obstacles, step=0.004):
        self.obstacles = obstacles
        self.step = step
        self.n_calls = 0

    def _free(self, q):
        # 1 micrometer: physically zero, above conic-solver graze noise
        # (full-graph rounding paths legitimately touch region corners)
        self.n_calls += 1
        return penetration_depth(q, self.obstacles) <= 1e-6

    def segment(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        L = float(np.linalg.norm(b - a))
        n = max(2, int(np.ceil(L / self.step)))
        ts = np.linspace(0, 1, n)
        free = [self._free(a + t * (b - a)) for t in ts]
        if all(free):
            return True, None
        i0 = free.index(False)
        j = i0
        while j < n and not free[j]:
            j += 1
        # FULL dirty run, so repair anchors land on clean ground
        lo = ts[max(i0 - 1, 0)]
        hi = ts[min(j, n - 1)]
        return False, (float(lo), float(hi))

    def path(self, path):
        self.n_calls = 0
        p = np.asarray(path)
        bad = []
        for k in range(len(p) - 1):
            ok, itv = self.segment(p[k], p[k + 1])
            if not ok:
                bad.append((k, itv))
        return len(bad) == 0, {"collision_probes_bad": len(bad),
                               "bad_segments": [k for k, _ in bad],
                               "bad_intervals": bad,
                               "clearance_calls": self.n_calls}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly
    from e6_fig1 import poly_vertices

    lib = pickle.load(open("out/e6_iris2d.pkl", "rb"))
    topo = lib["topo"]
    regions = topo["regions"]
    s = np.asarray(lib["task_configs"]["start"])
    g = np.asarray(lib["task_configs"]["goal"])
    obstacles = [np.asarray(o) for o in lib["obstacles"]]

    # the GCS answer on the (believed-clean) regions
    rg = sb.with_query(topo, s, g)
    full = sb.solve_gcs(rg)
    P_geo = np.asarray(full["waypoints"])
    c_geo = float(np.linalg.norm(np.diff(P_geo, axis=0), axis=1).sum())
    print(f"geometric optimum (the GCS answer): cost {c_geo:.3f}",
          flush=True)

    # inject the pocket on a mid-path segment whose witness region is
    # WIDE (an in-region detour must exist -- the demo is a pocket, not
    # a blocked corridor)
    seg = np.linalg.norm(np.diff(P_geo, axis=0), axis=1)
    best_k, best_r = None, -1.0
    for kk in range(len(seg)):
        if seg[kk] < 0.35:
            continue
        m = 0.5 * (P_geo[kk] + P_geo[kk + 1])
        for A, b in regions:
            if np.all(A @ m <= b + 1e-9):
                _, rad = e6.chebyshev(np.asarray(A), np.asarray(b))
                best_k, best_r = (kk, rad) if rad > best_r else \
                    (best_k, best_r)
                break
    k = best_k
    print(f"pocket host segment {k}, witness-region radius "
          f"{best_r:.2f}", flush=True)
    mid = 0.5 * (P_geo[k] + P_geo[k + 1])
    hx, hy = POCKET_HALF
    pocket = np.array([[mid[0] - hx, mid[1] - hy],
                       [mid[0] + hx, mid[1] - hy],
                       [mid[0] + hx, mid[1] + hy],
                       [mid[0] - hx, mid[1] + hy]])
    truth = obstacles + [pocket]
    val = Validator2D(truth)
    ok_geo, info_geo = val.path(P_geo)
    pen = max(penetration_depth(0.5 * (P_geo[k] + P_geo[k + 1]), truth),
              penetration_depth(mid, truth))
    assert not ok_geo, "pocket must intersect the geometric optimum"
    print(f"GCS answer collides with the pocket "
          f"(segments {info_geo['bad_segments']})", flush=True)

    # our layer: validated roadmap -> corridor -> polish -> repair
    ifaces = e6.classify_interfaces(topo)
    rm = e6.build_roadmap(topo, ifaces, "area", 8, seed=0)
    # lazy validation, done eagerly for the demo: drop dirty samples
    keep = {i for i, x in enumerate(rm["pos"])
            if penetration_depth(x, truth) <= 1e-9}
    cost_rm, path_rm = e6.shortest_path(rm, regions, s, g,
                                        alive_nodes=keep)
    # node filtering is not edge filtering: candidates are validated and
    # repaired exactly like the high-D pipeline, cheapest certified wins
    hulls = {}
    rng = np.random.default_rng(0)
    cands = []
    p1, ok1, nv1 = repair_path(path_rm, val, regions, hulls, rng,
                               tries=200)
    if ok1:
        cands.append((path_cost(p1), np.asarray(p1),
                      f"roadmap+repair({nv1})"))
    pair_eid = {frozenset((p["u"], p["v"])): eid
                for eid, p in topo["portals"].items()}
    mems = [e6.region_membership(p, regions) for p in path_rm]
    pseq = e6.corridor_from_path(mems, pair_eid)
    if pseq is not None:
        from e7_highd_audit import polish_waypoints
        rgq = sb.with_query(topo, s, g)
        c_pol, pp = polish_waypoints(rgq, pseq, s, g)
        if pp is not None:
            p2, ok2, nv2 = repair_path(pp, val, regions, hulls,
                                       rng, tries=200)
            if ok2:
                cands.append((path_cost(p2), np.asarray(p2),
                              f"polish+repair({nv2})"))
    assert cands, "no certified candidate"
    cands.sort(key=lambda t: t[0])
    _, P_ours, note = cands[0]
    ok_ours, _ = val.path(P_ours)
    assert ok_ours
    c_ours = path_cost(P_ours)
    print(f"ours ({note}): cost {c_ours:.3f} "
          f"(+{100*(c_ours/c_geo-1):.1f}%), certified clean", flush=True)

    # ---------------- draw ----------------
    C_OBS, C_REG, C_IF = "#3f3f46", "#93c5fd", "#f59e0b"
    C_BAD, C_OK, C_PKT = "#dc2626", "#059669", "#dc2626"
    fig, ax = plt.subplots(figsize=(9.6, 8.6))
    for ob in obstacles:
        ax.add_patch(MplPoly(ob, closed=True, facecolor=C_OBS,
                             edgecolor="k", lw=0.6, zorder=3))
    for A, b in regions:
        V = poly_vertices(np.asarray(A), np.asarray(b))
        if V is not None and len(V) >= 3:
            ax.add_patch(MplPoly(V, closed=True, facecolor=C_REG,
                                 edgecolor="#1d4ed8", lw=0.5,
                                 alpha=0.25, zorder=1))
    for f in ifaces.values():
        V = poly_vertices(f["A"], f["b"])
        if V is not None and len(V) >= 3:
            ax.add_patch(MplPoly(V, closed=True, facecolor=C_IF,
                                 edgecolor="#b45309", lw=0.6,
                                 alpha=0.45, zorder=2))
    ax.add_patch(MplPoly(pocket, closed=True, facecolor=C_PKT,
                         edgecolor="#7f1d1d", lw=1.4, alpha=0.85,
                         hatch="////", zorder=4))
    ax.annotate("collision pocket the region\ngenerator never saw\n"
                "(region still claims this is free)",
                xy=mid, xytext=(mid[0] + 1.05, mid[1] - 0.9),
                fontsize=10, color="#7f1d1d",
                arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.3),
                zorder=8)

    ax.plot(P_geo[:, 0], P_geo[:, 1], "--", color=C_BAD, lw=2.6,
            zorder=6, label=f"every geometric GCS answer: cost "
            f"{c_geo:.2f} — COLLIDES")
    bks = info_geo["bad_segments"]
    bk = min(bks, key=lambda kk: np.linalg.norm(
        0.5 * (P_geo[kk] + P_geo[kk + 1]) - mid))
    ax.plot(*mid, "x", ms=17, mew=4, color=C_BAD, zorder=9)

    ax.plot(P_ours[:, 0], P_ours[:, 1], "-", color=C_OK, lw=2.8,
            zorder=7, label=f"ours (validate → repair → certify): cost "
            f"{c_ours:.2f} (+{100*(c_ours/c_geo-1):.1f}%) — certified "
            "clean")
    ax.plot(*s, "*", ms=19, color="#16a34a", zorder=10)
    ax.plot(*g, "*", ms=19, color=C_BAD, zorder=10)
    ax.annotate("start", s, textcoords="offset points", xytext=(8, 8),
                fontsize=10)
    ax.annotate("goal", g, textcoords="offset points", xytext=(-4, 10),
                fontsize=10)

    lo, hi = lib["domain"]
    ax.set_xlim(lo[0] - .12, hi[0] + .12)
    ax.set_ylim(lo[1] - .12, hi[1] + .12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.set_title(
        "Contaminated region library, one query, two answers\n"
        "regions (blue) believed collision-free; the hatched pocket is "
        "real but absent from every region description.\n"
        "Geometric planners keep the path inside the regions — and "
        "inside the pocket.  The physical layer detects, repairs "
        "in-region, certifies.", fontsize=10.5)
    fig.tight_layout()
    out = "out/fig_2d_contamination.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
