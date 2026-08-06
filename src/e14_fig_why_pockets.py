#!/usr/bin/env python3
"""E14: the derivation as a picture -- why pockets MUST exist.

Four panels, one per step of the argument:
 (a) workspace, convex obstacle: the separating-hyperplane theorem
     makes one cut a COMPLETE exclusion proof;
 (b) C-space, non-convex obstacle (preimage of a convex set under
     forward kinematics): a cut at the discovered collision point only
     removes a supporting half-space -- the horns curve back around
     the cut and REMAIN inside the region (computed by actual polygon
     clipping, not drawn by hand);
 (c) finite sampling cannot see the residual: M clean interior samples
     accept the region under the (eps, delta) contract;
 (d) the optimizer must exploit the surplus: whenever the through-
     pocket shortcut is shorter, every modeled optimum is infeasible.

    python3 e14_fig_why_pockets.py    # out/fig_why_pockets.png
"""
from __future__ import annotations

import numpy as np

C_OBS = "#3f3f46"
C_REG = "#93c5fd"
C_REGE = "#1d4ed8"
C_RES = "#dc2626"
C_OK = "#059669"
C_CUT = "#7c3aed"


def clip_halfplane(poly, n, c):
    """Sutherland–Hodgman: keep {x : n.x <= c}."""
    out = []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        ina, inb = np.dot(n, a) <= c, np.dot(n, b) <= c
        if ina:
            out.append(a)
        if ina != inb:
            t = (c - np.dot(n, a)) / np.dot(n, b - a)
            out.append(a + t * (b - a))
    return np.asarray(out) if out else np.zeros((0, 2))


def point_in_convex(p, convex):
    sign = 0
    m = len(convex)
    for i in range(m):
        a, b = convex[i], convex[(i + 1) % m]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cr) < 1e-12:
            continue
        sg = 1 if cr > 0 else -1
        if sign == 0:
            sign = sg
        elif sg != sign:
            return False
    return True


def clip_convex(poly, convex):
    """Clip arbitrary polygon by a convex polygon (list of CCW verts)."""
    out = np.asarray(poly, float)
    m = len(convex)
    for i in range(m):
        a, b = np.asarray(convex[i]), np.asarray(convex[(i + 1) % m])
        e = b - a
        n = np.array([e[1], -e[0]])          # inward for CCW
        c = np.dot(n, a)
        if len(out) == 0:
            break
        out = clip_halfplane(out, n, c)
    return out


def crescent(A, rA, B, rB, n=600):
    """Polygon of disk(A,rA) minus disk(B,rB) (single lune)."""
    th = np.linspace(0, 2 * np.pi, n)
    pa = np.c_[A[0] + rA * np.cos(th), A[1] + rA * np.sin(th)]
    keep = np.linalg.norm(pa - B, axis=1) >= rB
    # roll so the kept arc is contiguous
    i0 = np.argmax(~keep)
    pa = np.roll(pa, -i0, axis=0)
    keep = np.roll(keep, -i0)
    outer = pa[keep]
    pb = np.c_[B[0] + rB * np.cos(th), B[1] + rB * np.sin(th)]
    keepb = np.linalg.norm(pb - A, axis=1) <= rA
    i0 = np.argmax(~keepb)
    pb = np.roll(pb, -i0, axis=0)
    keepb = np.roll(keepb, -i0)
    inner = pb[keepb][::-1]
    return np.vstack([outer, inner])


def in_crescent(p, A, rA, B, rB):
    return (np.linalg.norm(p - A) <= rA) and (np.linalg.norm(p - B) > rB)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as MplPoly

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.2))
    (axa, axb), (axc, axd) = axes

    # ------------------------------------------------ (a) convex world
    P_a = np.array([[-2.2, -1.5], [0.35, -1.5], [0.35, 1.9], [-1.2, 2.2],
                    [-2.2, 1.0]], float)
    axa.add_patch(MplPoly(P_a, closed=True, facecolor=C_REG,
                          edgecolor=C_REGE, alpha=0.55, lw=1.5))
    axa.add_patch(Circle((1.7, 0.3), 0.95, facecolor=C_OBS,
                         edgecolor="k", lw=1.0))
    axa.plot([0.6, 0.6], [-1.9, 2.5], "--", color=C_CUT, lw=2.4)
    axa.annotate("separating hyperplane:\nBOTH sets convex $\\Rightarrow$"
                 " it exists\nand excludes the WHOLE obstacle",
                 xy=(0.6, 1.6), xytext=(-2.75, 2.15), fontsize=10.5,
                 color=C_CUT,
                 arrowprops=dict(arrowstyle="->", color=C_CUT))
    axa.text(-2.75, -2.75,
             "one cut $=$ a complete exclusion proof:  "
             "$P\\subseteq\\mathcal{F}$ certified exactly "
             "(residual $\\equiv 0$)", fontsize=10.5)
    axa.text(1.15, 0.25, "convex\nobstacle", color="white", fontsize=11,
             ha="center")
    axa.text(-1.15, 0.2, "region $P$", color="#1e3a8a", fontsize=13,
             ha="center")
    axa.set_title("(a)  workspace: convex obstacle — cuts CAN finish "
                  "the proof", fontsize=12.5)

    # ------------------------------------------- (b) nonconvex C-space
    A, rA = np.array([0.0, 0.0]), 2.0
    B, rB = np.array([0.0, 0.9]), 2.0
    ban = crescent(A, rA, B, rB)
    P_b = np.array([[-2.15, -1.42], [2.15, -1.42], [2.5, 0.4],
                    [1.5, 1.75], [-1.5, 1.75], [-2.5, 0.4]], float)
    cuty = -1.02
    # region after the cut (keep y >= cuty)
    P_cut = clip_halfplane(P_b, np.array([0.0, -1.0]), -cuty)
    def area(poly):
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    Pc = P_cut if area(P_cut) > 0 else P_cut[::-1]
    residual = clip_convex(ban, Pc)
    axb.add_patch(MplPoly(ban, closed=True, facecolor=C_OBS,
                          edgecolor="k", lw=1.0, zorder=2))
    axb.add_patch(MplPoly(P_b, closed=True, facecolor=C_REG,
                          edgecolor=C_REGE, alpha=0.45, lw=1.5, zorder=1))
    axb.axhspan(-2.6, cuty, xmin=0.02, xmax=0.98, facecolor="none",
                edgecolor="#9ca3af", hatch="..", lw=0.0, zorder=0)
    axb.plot([-2.75, 2.75], [cuty, cuty], "--", color=C_CUT, lw=2.4,
             zorder=4)
    axb.plot(0, -1.28, "x", ms=15, mew=3.5, color="#f59e0b", zorder=5)
    axb.annotate("discovered collision\n$\\Rightarrow$ cut its "
                 "supporting half-space", xy=(0.05, -1.24),
                 xytext=(0.55, -2.35), fontsize=11, color="#b45309",
                 arrowprops=dict(arrowstyle="->", color="#b45309"),
                 zorder=6)
    if len(residual) >= 3:
        axb.add_patch(MplPoly(residual, closed=True, facecolor=C_RES,
                              edgecolor="#7f1d1d", lw=1.2, alpha=0.95,
                              zorder=5))
    for sx in (-1, 1):
        axb.annotate("", xy=(sx * 1.75, 0.35), xytext=(sx * 2.6, -0.85),
                     arrowprops=dict(arrowstyle="->", color=C_RES,
                                     lw=2.0), zorder=6)
    axb.text(0, -0.55, "the horns curve BACK\naround the cut:\n"
             "residual $\\subset P$ remains", color=C_RES, fontsize=12,
             ha="center", zorder=6, fontweight="bold")
    axb.text(-2.75, 2.25,
             "$O=\\{q:B(q)\\cap W\\neq\\emptyset\\}$ (FK preimage) "
             "is NON-convex:\nno hyperplane separates it; finitely many "
             "cuts cannot finish", fontsize=10.5)
    axb.set_title("(b)  C-space: non-convex obstacle — every cut is "
                  "only LOCAL", fontsize=12.5)

    # ------------------------------------------------- (c) sampling
    axc.add_patch(MplPoly(P_cut, closed=True, facecolor=C_REG,
                          edgecolor=C_REGE, alpha=0.45, lw=1.5, zorder=1))
    if len(residual) >= 3:
        axc.add_patch(MplPoly(residual, closed=True, facecolor=C_RES,
                              edgecolor="#7f1d1d", lw=1.2, alpha=0.95,
                              zorder=4))
    rng = np.random.default_rng(3)
    pts = []
    guard = 0
    while len(pts) < 46 and guard < 20000:
        guard += 1
        p = rng.uniform([-2.4, cuty], [2.4, 1.75])
        if not point_in_convex(p, P_cut):
            continue
        if in_crescent(p, A, rA, B, rB):
            continue
        pts.append(p)
    pts = np.asarray(pts)
    axc.plot(pts[:, 0], pts[:, 1], "o", ms=6, color=C_OK, zorder=3,
             markeredgewidth=0)
    axc.annotate("undetected: sliver volume $<$ detection\n"
                 "probability of $M$ samples", xy=(1.9, 0.42),
                 xytext=(-2.6, 2.05), fontsize=10.5, color=C_RES,
                 arrowprops=dict(arrowstyle="->", color=C_RES), zorder=6)
    axc.text(-2.75, -2.85,
             "$\\mathrm{Pr}[M\\ \\mathrm{clean\\ samples} \\mid "
             "\\mathrm{colliding\\ fraction}=\\varepsilon]"
             "=(1-\\varepsilon)^M$\n"
             "accept $\\Rightarrow$ contamination $\\leq\\varepsilon$ "
             "at confidence $1-\\delta$\n"
             "--- with $\\varepsilon>0$ BY CONSTRUCTION", fontsize=10.5)
    axc.set_title("(c)  finite sampling accepts the residual: "
                  "$\\forall$ cannot be sampled", fontsize=12.5)

    # ------------------------------------------------ (d) optimizer
    P_d = np.array([[-2.5, -0.85], [2.5, -0.85], [2.5, 0.85],
                    [-2.5, 0.85]], float)
    pocket = np.array([[-0.45, 0.85], [0.45, 0.85], [0.45, 0.30],
                       [-0.45, 0.30]], float)
    axd.add_patch(MplPoly(P_d, closed=True, facecolor=C_REG,
                          edgecolor=C_REGE, alpha=0.45, lw=1.5))
    axd.add_patch(MplPoly(pocket, closed=True, facecolor=C_RES,
                          edgecolor="#7f1d1d", lw=1.2, alpha=0.95,
                          hatch="///"))
    s, g = np.array([-2.25, 0.62]), np.array([2.25, 0.62])
    axd.plot(*s, "*", ms=17, color=C_OK)
    axd.plot(*g, "*", ms=17, color=C_RES)
    axd.plot([s[0], g[0]], [s[1], g[1]], "--", color=C_RES, lw=2.6)
    axd.plot(0, 0.62, "x", ms=15, mew=3.5, color=C_RES)
    det = np.array([s, [-0.62, 0.62], [-0.5, 0.22], [0.5, 0.22],
                    [0.62, 0.62], g])
    axd.plot(det[:, 0], det[:, 1], "-", color=C_OK, lw=2.8)
    axd.text(-0.02, 1.15, "pocket $=$ the residual of (b), accepted "
             "in (c)", color="#7f1d1d", fontsize=10.5, ha="center")
    axd.text(-2.75, -1.45,
             "$m(P)<m(P\\cap\\mathcal{F})\\Rightarrow$ EVERY modeled "
             "optimum is infeasible", fontsize=10.5)
    axd.text(-2.75, -1.95,
             "taut optima hug boundaries: $3.2\\%$ volume "
             "$\\to$ $62\\%$ of task optima (measured)",
             fontsize=10.5)
    axd.set_title("(d)  the optimizer is DRIVEN into the pocket, not "
                  "unlucky", fontsize=12.5)

    for ax, (xl, xh, yl, yh) in zip(
            (axa, axb, axc, axd),
            ((-2.9, 2.9, -3.05, 2.85), (-2.9, 2.9, -2.85, 3.05),
             (-2.9, 2.9, -3.45, 2.75), (-2.9, 2.9, -2.25, 1.85))):
        ax.set_xlim(xl, xh)
        ax.set_ylim(yl, yh)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Why pockets MUST exist:  (a) convexity makes cuts complete "
        "proofs;  (b) C-space non-convexity makes them local;\n"
        "(c) sampling cannot certify the universal claim, so the "
        "contract tolerates $\\varepsilon>0$;  (d) optimization "
        "provably exploits whatever remains.", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = "out/fig_why_pockets.png"
    fig.savefig(out, dpi=165, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
