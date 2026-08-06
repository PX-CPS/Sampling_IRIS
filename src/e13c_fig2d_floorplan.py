#!/usr/bin/env python3
"""E13-C: the 2-D map as a floor plan -- walls with doorways, grown
pockets at the jambs, a wall-clipping GCS* answer, and our certified
reroute.

World: several wall assemblies with door openings (a wall with a
doorway is NON-convex as a union -- the residual wedges concentrate at
the jambs and inner corners, exactly where taut paths hug) plus two
freestanding convex pillars (projection-theorem cuts, exact -- the
contrast).  Regions are grown by the same IrisZo-lite as e13b: point
oracle, supporting-half-plane cuts at bisected crossings for the
non-convex wall unions, one exact separating cut for a convex pillar,
Bernoulli (eps, delta) acceptance.  The GCS geometric optimum threads
the doorways and clips a jamb wedge the audit accepted; our layer
detects it, reroutes through the legitimate openings, and certifies.

    python3 e13c_fig2d_floorplan.py   # out/fig_2d_contamination3.png
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

# ---------------- the floor plan ----------------
LO, HI = np.array([0.0, 0.0]), np.array([8.0, 5.0])
TH = 0.18                                     # wall thickness


def hwall(y, x0, x1, gaps):
    """horizontal wall strip with door gaps [(a,b),...]."""
    xs = sorted([x0] + [g for ab in gaps for g in ab] + [x1])
    out = []
    for a, b in zip(xs[::2], xs[1::2]):
        if b - a > 1e-9:
            out.append(np.array([[a, y - TH / 2], [b, y - TH / 2],
                                 [b, y + TH / 2], [a, y + TH / 2]]))
    return out


def vwall(x, y0, y1, gaps):
    ys = sorted([y0] + [g for ab in gaps for g in ab] + [y1])
    out = []
    for a, b in zip(ys[::2], ys[1::2]):
        if b - a > 1e-9:
            out.append(np.array([[x - TH / 2, a], [x + TH / 2, a],
                                 [x + TH / 2, b], [x - TH / 2, b]]))
    return out


WALLS = (vwall(2.6, 0.0, 5.0, [(1.15, 1.75), (3.35, 3.95)])
         + hwall(2.5, 2.6, 8.0, [(4.15, 4.75), (6.25, 6.85)])
         + vwall(5.4, 2.5, 5.0, [(3.55, 4.15)])
         + hwall(1.3, 5.4, 8.0, [(6.4, 7.0)]))
PILLARS = [np.array([[1.05, 3.55], [1.55, 3.55],
                     [1.55, 4.05], [1.05, 4.05]]),
           np.array([[3.6, 0.7], [4.1, 0.7], [4.1, 1.2], [3.6, 1.2]])]
RECTS = WALLS + PILLARS
S, G = np.array([0.7, 0.7]), np.array([7.35, 4.35])
SEEDS = [np.array([0.9, 1.3]), np.array([1.6, 2.6]),
         np.array([3.4, 1.6]), np.array([3.5, 3.3]),
         np.array([4.9, 3.9]), np.array([6.6, 4.1]),
         np.array([5.0, 1.6]), np.array([7.0, 2.0]),
         np.array([2.0, 4.4]), np.array([4.45, 2.5]),
         np.array([6.55, 2.5]), np.array([6.7, 1.3]),
         np.array([5.9, 2.0])]
EPS, DELTA = 0.01, 0.05
MARGIN = 1e-6


def in_rect(x, r, margin=0.0):
    lo, hi = r.min(0), r.max(0)
    return np.all(x >= lo + margin) and np.all(x <= hi - margin)


def occupied(x, margin=0.0):
    return any(in_rect(x, r, margin) for r in RECTS)


def occupied_vec(X, margin=0.0):
    bad = np.zeros(len(X), bool)
    for r in RECTS:
        lo, hi = r.min(0), r.max(0)
        bad |= np.all(X >= lo + margin, 1) & np.all(X <= hi - margin, 1)
    return bad


class TruthOracle(Validator2D):
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
    raise RuntimeError("rejection starved")


def grow_region(seed, rng, max_cuts=120):
    """IrisZo-lite.  Convex pillars (known geometry) get the exact
    projection-theorem separating cut; the non-convex wall unions are
    a black box -- supporting half-plane at the bisected crossing."""
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
        pil = next((p for p in PILLARS if in_rect(witness, p)), None)
        if pil is not None:
            xb = np.clip(c, pil.min(0), pil.max(0))
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
    return 0.5 * abs(np.dot(x, np.roll(y, -1))
                     - np.dot(y, np.roll(x, -1)))


def clip_rect(rect, A, b):
    """Sutherland-Hodgman of a rect against convex (A,b)."""
    out = [r for r in rect]
    for ai, bi in zip(A, b):
        if not out:
            break
        res = []
        m = len(out)
        for i in range(m):
            p, q = np.asarray(out[i]), np.asarray(out[(i + 1) % m])
            ina, inb = ai @ p <= bi, ai @ q <= bi
            if ina:
                res.append(p)
            if ina != inb:
                t = (bi - ai @ p) / (ai @ (q - p))
                res.append(p + t * (q - p))
        out = res
    return np.asarray(out) if out else np.zeros((0, 2))


def shortcut(path, val, rng, iters=250):
    P = [np.asarray(q, float) for q in path]
    for _ in range(iters):
        if len(P) <= 2:
            break
        i = rng.integers(0, len(P) - 2)
        j = rng.integers(i + 2, min(i + 6, len(P)))
        ok, _ = val.segment(P[i], P[j])
        if ok:
            P = P[:i + 1] + P[j:]
    return np.asarray(P)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly

    regions, cuts_all, audits, attempts = [], [], [], []
    for i, sd in enumerate(SEEDS):
        for att in range(14):
            rng = np.random.default_rng(3000 * i + att)
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

    # residual pockets = obstacle rect ∩ region, per rect class
    pockets, pillar_leak = [], 0.0
    rng = np.random.default_rng(7)
    for k, (A, b) in enumerate(regions):
        for r in WALLS:
            pk = clip_rect(r, A, b)
            if len(pk) >= 3 and shoelace(pk) > 1e-6:
                pockets.append((k, pk))
        for p in PILLARS:
            pk = clip_rect(p, A, b)
            if len(pk) >= 3:
                pillar_leak = max(pillar_leak, shoelace(pk))
    assert pillar_leak < 1e-9, f"pillar leak {pillar_leak}"
    assert pockets, "no wall residuals grew"
    print(f"{len(pockets)} wall residual wedges; pillar leak "
          f"{pillar_leak:.1e} (exact cuts)", flush=True)

    topo = sb.build_topology(regions)
    print(f"topology: {len(topo['portals'])} portals", flush=True)
    rg = sb.with_query(topo, S, G)
    full = sb.solve_gcs(rg)
    P_geo = np.asarray(full["waypoints"])
    c_geo = float(np.linalg.norm(np.diff(P_geo, axis=0), axis=1).sum())
    val = TruthOracle()
    ok_geo, info_geo = val.path(P_geo)
    assert not ok_geo, "GCS answer must clip a wall -- adjust layout"
    kbad, (a_t, b_t) = info_geo["bad_intervals"][0]
    qbad = P_geo[kbad] + 0.5 * (a_t + b_t) * (P_geo[kbad + 1]
                                              - P_geo[kbad])
    assert any(in_rect(qbad, r) for r in WALLS), \
        "violation must be inside a true wall"
    print(f"GCS answer: cost {c_geo:.3f}, clips a wall at {qbad}",
          flush=True)

    regs = topo["regions"]
    hulls = {}
    rrng = np.random.default_rng(11)
    cands = []
    p1, ok1, _ = repair_path(P_geo, val, regs, hulls, rrng, tries=300)
    if ok1:
        cands.append((path_cost(p1), np.asarray(p1)))
    ifaces = e6.classify_interfaces(topo)
    rm = e6.build_roadmap(topo, ifaces, "area", 8, seed=0)
    keep = {i for i, x in enumerate(rm["pos"])
            if not occupied(np.asarray(x), MARGIN)}
    _, path_rm = e6.shortest_path(rm, regs, S, G, alive_nodes=keep)
    if path_rm is not None:
        p2, ok2, _ = repair_path(path_rm, val, regs, hulls, rrng,
                                 tries=300)
        if ok2:
            cands.append((path_cost(p2), np.asarray(p2)))
    assert cands, "no certified candidate"
    cands.sort(key=lambda t: t[0])
    _, P_ours = cands[0]
    P_ours = shortcut(P_ours, val, np.random.default_rng(2))
    okf, _ = val.path(P_ours)
    assert okf
    c_ours = path_cost(P_ours)
    print(f"ours: cost {c_ours:.3f} (+{100*(c_ours/c_geo-1):.1f}%), "
          f"certified", flush=True)

    # ---------------- draw ----------------
    C_OBS, C_REG = "#3f3f46", "#93c5fd"
    C_BAD, C_OK, C_CUT = "#dc2626", "#059669", "#7c3aed"
    fig, ax = plt.subplots(figsize=(11.2, 7.4))
    for A, b in regions:
        V = poly_vertices(A, b)
        ax.add_patch(MplPoly(V, closed=True, facecolor=C_REG,
                             edgecolor="#1d4ed8", lw=0.7, alpha=0.15,
                             zorder=1))
    for r in WALLS:
        ax.add_patch(MplPoly(r, closed=True, facecolor=C_OBS,
                             edgecolor="k", lw=0.6, zorder=3))
    for p in PILLARS:
        ax.add_patch(MplPoly(p, closed=True, facecolor="#27272a",
                             edgecolor="k", lw=0.6, zorder=3))
    for k, pk in pockets:
        ax.add_patch(MplPoly(pk, closed=True, facecolor=C_BAD,
                             edgecolor="none", alpha=0.9, zorder=5))

    ax.plot(P_geo[:, 0], P_geo[:, 1], "--", color=C_BAD, lw=2.6,
            zorder=7, label=f"GCS* / every geometric GCS answer: "
            f"cost {c_geo:.2f} — CLIPS THE WALL")
    ax.plot(P_ours[:, 0], P_ours[:, 1], "-", color=C_OK, lw=2.8,
            zorder=8, label=f"ours (validate → repair → certify): "
            f"cost {c_ours:.2f} (+{100*(c_ours/c_geo-1):.1f}%) — "
            "through the doorways, certified")
    ax.plot(*S, "*", ms=18, color="#16a34a", zorder=10)
    ax.plot(*G, "*", ms=18, color=C_BAD, zorder=10)
    ax.annotate("start", S, textcoords="offset points",
                xytext=(-8, -18), fontsize=10)
    ax.annotate("goal", G, textcoords="offset points", xytext=(-4, 10),
                fontsize=10)
    ax.annotate("residual wedges the audit\n"
                f"accepted (red, $\\varepsilon$={EPS:g})\n"
                "on faces and door jambs",
                xy=(2.56, 2.30), fontsize=10,
                xytext=(0.02, 0.56), textcoords="axes fraction",
                color="#7f1d1d",
                arrowprops=dict(arrowstyle="->", color="#7f1d1d",
                                lw=1.2), zorder=11)


    ax.annotate("convex pillar: ONE separating-\nhyperplane cut, "
                "exact (leak = 0)",
                xy=(1.55, 3.8), xytext=(0.02, 0.86),
                textcoords="axes fraction", fontsize=10,
                color="#1f2937",
                arrowprops=dict(arrowstyle="->", color="#1f2937",
                                lw=1.1), zorder=11)

    ax.set_xlim(LO[0] - 0.1, HI[0] + 0.1)
    ax.set_ylim(LO[1] - 0.1, HI[1] + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.set_title(
        "A floor plan, one query, two answers\n"
        "walls-with-doorways are non-convex unions: the sampling-based "
        "generator's cuts leave accepted wedges at the jambs.\n"
        "The taut geometric optimum hugs the jambs and clips the wall; "
        "the physical layer reroutes through the doorways and "
        "certifies.", fontsize=10.5)
    fig.tight_layout()
    for out in ("out/fig_2d_contamination3.pdf",
                "out/fig_2d_contamination3.png"):
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print("written", out, flush=True)


if __name__ == "__main__":
    main()
