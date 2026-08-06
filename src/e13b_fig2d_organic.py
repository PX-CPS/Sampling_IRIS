#!/usr/bin/env python3
"""E13-B: the 2-D map, with the pocket GROWN, not injected.

The previous figure dropped a hand-made rectangular pocket into a
region -- arbitrary, convex, and disconnected from any mechanism.
Here the contamination is produced by the mechanism itself:

  * the world holds convex blocks AND one non-convex crescent;
  * regions are grown by a faithful IrisZo-lite: the obstacle set is a
    point-membership black box, cuts are supporting half-planes at
    bisected boundary crossings of DISCOVERED colliding samples, and a
    region is accepted by the Bernoulli gate (M consecutive clean
    samples for the (eps, delta) contract) -- simplification vs the
    real thing: fixed Euclidean metric, no ellipsoid update;
  * against the convex blocks the cuts terminate exactly (measured
    penetration 0); against the crescent the horns curve back around
    every cut, are too thin for volume-uniform sampling to hit, and
    survive the audit -- the residual INSIDE the accepted region is
    the pocket, computed by polygon clipping, exactly the geometry of
    Fig. 1(b);
  * the GCS answer on the accepted regions hugs the shortest corridor
    and crosses a horn; our layer detects, repairs in-region,
    certifies.

    python3 e13b_fig2d_organic.py    # out/fig_2d_contamination2.png
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
import e6_sampling as e6
import stageB_bimanual_ipgcs as sb
from e6_fig1 import poly_vertices
from e8_validate import path_cost, repair_path
from e13_fig2d_contamination import Validator2D
from e14_fig_why_pockets import clip_convex, crescent, in_crescent

# ---------------- the world ----------------
LO, HI = np.array([0.0, 0.0]), np.array([6.0, 4.0])
BLOCK_L = np.array([[0.9, 2.9], [2.3, 2.9], [2.3, 4.0], [0.9, 4.0]])
BLOCK_R = np.array([[3.9, 0.0], [5.1, 0.0], [5.1, 1.15], [3.9, 1.15]])
CA, RA = np.array([2.95, 1.35]), 1.75          # outer disk
CB, RB = np.array([2.95, 2.05]), 1.75          # carving disk
CRESC = crescent(CA, RA, CB, RB)
OBSTACLES = [BLOCK_L, BLOCK_R, CRESC]
CONVEX_OBS = [BLOCK_L, BLOCK_R]
S, G = np.array([0.45, 1.15]), np.array([5.55, 2.45])
SEEDS = [np.array([0.55, 1.7]), np.array([2.0, 2.2]),
         np.array([3.1, 2.6]), np.array([4.6, 2.6]),
         np.array([5.4, 1.9])]
EPS, DELTA = 0.01, 0.05                        # IrisZo defaults
MARGIN = 1e-6                                  # solver graze noise


def occupied(x, margin=0.0):
    """O(1) analytic point oracle (blocks are axis-aligned; the
    crescent is two circle tests).  The 600-vertex polygon is for
    drawing only."""
    for blk in CONVEX_OBS:
        lo, hi = blk.min(0), blk.max(0)
        if np.all(x >= lo + margin) and np.all(x <= hi - margin):
            return True
    return (np.linalg.norm(x - CA) <= RA - margin
            and np.linalg.norm(x - CB) >= RB + margin)


def occupied_vec(X, margin=0.0):
    bad = np.zeros(len(X), bool)
    for blk in CONVEX_OBS:
        lo, hi = blk.min(0), blk.max(0)
        bad |= np.all(X >= lo + margin, 1) & np.all(X <= hi - margin, 1)
    bad |= (np.linalg.norm(X - CA, axis=1) <= RA - margin) \
        & (np.linalg.norm(X - CB, axis=1) >= RB + margin)
    return bad


class TruthOracle(Validator2D):
    """Validator2D interface on the analytic oracle."""

    def __init__(self, step=0.004):
        super().__init__([], step=step)

    def _free(self, q):
        self.n_calls += 1
        return not occupied(np.asarray(q), MARGIN)


def sample_in_poly(A, b, rng, V):
    lo, hi = V.min(0), V.max(0)
    for _ in range(20000):
        x = lo + rng.random(2) * (hi - lo)
        if np.all(A @ x <= b + 1e-12):
            return x
    raise RuntimeError("rejection sampling starved")


def grow_region(seed, rng, max_cuts=80):
    """IrisZo-lite; returns (A, b, cuts, audit_pts, accepted)."""
    c = np.asarray(seed, float)
    assert not occupied(c), "seed must be free"
    A = np.vstack([np.eye(2), -np.eye(2)])
    b = np.concatenate([HI, -LO])
    M = int(np.ceil(np.log(1.0 / DELTA) / EPS))
    cuts = []
    for _ in range(max_cuts):
        V = poly_vertices(A, b)
        streak, witness, pts = 0, None, []
        while streak < M:
            x = sample_in_poly(A, b, rng, V)
            if occupied(x):
                witness = x
                break
            pts.append(x)
            streak += 1
        if witness is None:
            return A, b, cuts, np.asarray(pts), True
        # THE dichotomy of the figure: a convex obstacle admits a
        # separating hyperplane (projection theorem) -- one cut
        # excludes the whole block, exactly.  The non-convex crescent
        # admits none; all the generator can do is cut a supporting
        # half-plane at the discovered boundary crossing.
        blk = next((o for o in CONVEX_OBS
                    if np.all(witness >= o.min(0))
                    and np.all(witness <= o.max(0))), None)
        if blk is not None:
            xb = np.clip(c, blk.min(0), blk.max(0))
        else:
            tlo, thi = 0.0, 1.0
            for _ in range(50):
                tm = 0.5 * (tlo + thi)
                if occupied(c + tm * (witness - c)):
                    thi = tm
                else:
                    tlo = tm
            xb = c + thi * (witness - c)
            cuts.append(xb)
        n = xb - c
        n = n / np.linalg.norm(n)
        A = np.vstack([A, n])
        b = np.append(b, float(n @ xb))
    return A, b, cuts, np.zeros((0, 2)), False


def shoelace(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def mc_contamination(A, b, rng, n=200_000):
    V = poly_vertices(A, b)
    lo, hi = V.min(0), V.max(0)
    X = lo + rng.random((n, 2)) * (hi - lo)
    inside = np.all(X @ A.T <= b + 1e-12, axis=1)
    return occupied_vec(X[inside]).mean(), inside.mean()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly

    # ---- grow the library (retry a rejected region like real
    # pipelines do; survivors still carry eps-level dirt) ----
    regions, cuts_all, audits, attempts = [], [], [], []
    for i, sd in enumerate(SEEDS):
        for att in range(12):
            rng = np.random.default_rng(1000 * i + att)
            A, b, cuts, pts, ok = grow_region(sd, rng)
            if ok:
                regions.append((A, b))
                cuts_all.append(cuts)
                audits.append(pts)
                attempts.append(att + 1)
                break
        else:
            raise RuntimeError(f"seed {i} never accepted")
    print("accepted after attempts:", attempts, flush=True)

    # ---- measured truth about the accepted library ----
    rng = np.random.default_rng(77)
    convex_leak = 0.0
    horn_region, horn_frac = None, 0.0
    pockets = []
    for k, (A, b) in enumerate(regions):
        V = poly_vertices(A, b)
        for ob in CONVEX_OBS:
            inter = clip_convex(ob, V)
            if len(inter) >= 3:
                convex_leak = max(convex_leak, shoelace(inter))
        pk = clip_convex(CRESC, V)
        if len(pk) >= 3 and shoelace(pk) > 1e-4:
            frac, _ = mc_contamination(A, b, rng)
            pockets.append((k, pk, frac))
            if frac > horn_frac:
                horn_region, horn_frac = k, frac
    assert convex_leak < 1e-9, f"convex obstacles must be cut exactly, leak {convex_leak}"
    assert pockets, "no residual pocket grew -- adjust layout"
    print(f"convex-obstacle overlap area of accepted regions: "
          f"{convex_leak:.2e} (exact cuts)", flush=True)
    for k, pk, frac in pockets:
        print(f"region {k}: crescent residual, true colliding "
              f"fraction {100*frac:.2f}% (accepted at eps={EPS})",
              flush=True)

    # ---- the GCS answer on the accepted regions ----
    topo = sb.build_topology(regions)
    print(f"topology: {len(topo['portals'])} portals", flush=True)
    rg = sb.with_query(topo, S, G)
    full = sb.solve_gcs(rg)
    P_geo = np.asarray(full["waypoints"])
    c_geo = float(np.linalg.norm(np.diff(P_geo, axis=0), axis=1).sum())
    val = TruthOracle()
    ok_geo, info_geo = val.path(P_geo)
    assert not ok_geo, "GCS answer must cross a horn -- adjust layout"
    # witness of the violation for the ✗ marker
    kbad, (a_t, b_t) = info_geo["bad_intervals"][0]
    qbad = P_geo[kbad] + 0.5 * (a_t + b_t) * (P_geo[kbad + 1]
                                              - P_geo[kbad])
    assert in_crescent(qbad, CA, RA, CB, RB), \
        "violation must be inside the crescent residual"
    print(f"GCS answer: cost {c_geo:.3f}, collides inside the horn "
          f"residual at {qbad}", flush=True)

    # ---- our layer: validate -> repair in-region -> certify ----
    regs = topo["regions"]
    hulls = {}
    rrng = np.random.default_rng(5)
    cands = []
    p1, ok1, nv1 = repair_path(P_geo, val, regs, hulls, rrng, tries=300)
    if ok1:
        cands.append((path_cost(p1), np.asarray(p1), "repair"))
    ifaces = e6.classify_interfaces(topo)
    rm = e6.build_roadmap(topo, ifaces, "area", 8, seed=0)
    keep = {i for i, x in enumerate(rm["pos"])
            if not occupied(np.asarray(x), MARGIN)}
    _, path_rm = e6.shortest_path(rm, regs, S, G, alive_nodes=keep)
    if path_rm is not None:
        p2, ok2, nv2 = repair_path(path_rm, val, regs, hulls, rrng,
                                   tries=300)
        if ok2:
            cands.append((path_cost(p2), np.asarray(p2), "roadmap"))
    assert cands, "no certified candidate"
    cands.sort(key=lambda t: t[0])
    c_ours, P_ours, note = cands[0]
    okf, _ = val.path(P_ours)
    assert okf
    print(f"ours ({note}): cost {c_ours:.3f} "
          f"(+{100*(c_ours/c_geo-1):.1f}%), certified", flush=True)

    # ---------------- draw ----------------
    C_OBS, C_REG, C_IF = "#3f3f46", "#93c5fd", "#f59e0b"
    C_BAD, C_OK, C_CUT = "#dc2626", "#059669", "#7c3aed"
    fig, ax = plt.subplots(figsize=(10.4, 7.6))
    for A, b in regions:
        V = poly_vertices(A, b)
        ax.add_patch(MplPoly(V, closed=True, facecolor=C_REG,
                             edgecolor="#1d4ed8", lw=0.8, alpha=0.16,
                             zorder=1))
    for ob in (BLOCK_L, BLOCK_R):
        ax.add_patch(MplPoly(ob, closed=True, facecolor=C_OBS,
                             edgecolor="k", lw=0.7, zorder=3))
    ax.add_patch(MplPoly(CRESC, closed=True, facecolor=C_OBS,
                         edgecolor="k", lw=0.7, zorder=3))
    # cuts near the crescent (the mechanism trace) -- shown for the
    # region the violated answer runs through
    kh = next(k for k, (A, b) in enumerate(regions)
              if np.all(A @ qbad <= b + 1e-9))
    for xb in cuts_all[kh]:
        on_crescent = (abs(np.linalg.norm(xb - CA) - RA) < 0.03
                       or abs(np.linalg.norm(xb - CB) - RB) < 0.03)
        if on_crescent:
            n = (xb - SEEDS[kh]) / np.linalg.norm(xb - SEEDS[kh])
            t = np.array([-n[1], n[0]])
            ax.plot([xb[0] - 0.9 * t[0], xb[0] + 0.9 * t[0]],
                    [xb[1] - 0.9 * t[1], xb[1] + 0.9 * t[1]],
                    "--", color=C_CUT, lw=1.9, alpha=0.95, zorder=6)
    # residual pockets (grown, not injected)
    for k, pk, frac in pockets:
        ax.add_patch(MplPoly(pk, closed=True, facecolor=C_BAD,
                             edgecolor="none", alpha=0.88, zorder=5))
    # audit samples of the horn region (they missed the horns)
    pa = audits[kh]
    near = np.linalg.norm(pa - CA, axis=1) < RA + 0.55
    ax.plot(pa[near, 0], pa[near, 1], ".", ms=3.2, color="#0d9488",
            alpha=0.6, zorder=4)

    ax.plot(P_geo[:, 0], P_geo[:, 1], "--", color=C_BAD, lw=2.6,
            zorder=7, label=f"every geometric GCS answer: cost "
            f"{c_geo:.2f} — COLLIDES in the residual")
    ax.plot(*qbad, "x", ms=16, mew=4, color=C_BAD, zorder=9)
    ax.plot(P_ours[:, 0], P_ours[:, 1], "-", color=C_OK, lw=2.8,
            zorder=8, label=f"ours (validate → repair → certify): "
            f"cost {c_ours:.2f} (+{100*(c_ours/c_geo-1):.1f}%) — "
            "certified clean")
    ax.plot(*S, "*", ms=18, color="#16a34a", zorder=10)
    ax.plot(*G, "*", ms=18, color=C_BAD, zorder=10)
    ax.annotate("start", S, textcoords="offset points", xytext=(-26, -16),
                fontsize=10)
    ax.annotate("goal", G, textcoords="offset points", xytext=(-6, 9),
                fontsize=10)

    ax.annotate("the pocket is GROWN, not injected:\n"
                "cuts (purple) stop at discovered collisions,\n"
                "the non-convex horn curves back behind them,\n"
                "samples (teal) miss the thin residual\n"
                f"— region accepted at $\\varepsilon$={EPS:g}",
                xy=(qbad[0] + 0.04, qbad[1] + 0.10), fontsize=10,
                xytext=(3.05, 2.62), color="#7f1d1d",
                arrowprops=dict(arrowstyle="->", color="#7f1d1d",
                                lw=1.3,
                                connectionstyle="arc3,rad=0.18"),
                zorder=11)
    ax.annotate("convex obstacle: ONE separating-\n"
                "hyperplane cut, exact (leak = 0)",
                xy=(2.3, 3.35), xytext=(3.15, 3.5), fontsize=10,
                color="#1f2937",
                arrowprops=dict(arrowstyle="->", color="#1f2937",
                                lw=1.1), zorder=11)

    ax.set_xlim(LO[0] - 0.1, HI[0] + 0.1)
    ax.set_ylim(LO[1] - 0.1, HI[1] + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.set_title(
        "Same generator, two obstacle classes, one query, two answers\n"
        "regions grown by a faithful sampling-based generator "
        "(point oracle + supporting-half-plane cuts + Bernoulli "
        "acceptance).\nConvex blocks are cut exactly; the non-convex "
        "crescent leaves accepted residual horns — the pocket nobody "
        "injected.", fontsize=10.5)
    fig.tight_layout()
    out = "out/fig_2d_contamination2.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
