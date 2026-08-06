#!/usr/bin/env python3
"""E6 step 1a: build a genuine 2D IRIS decomposition as the study testbed.

Uses the env_2d obstacle world (6 explicit polygons) so that every claim
can be checked against ground truth that never passed through the
planner: obstacles stay in vertex form for an independent point-in-
polygon audit.  IRIS regions genuinely overlap, which is the case the
interface study needs (maze libraries only produce degenerate
face contacts).

    python3 e6_build_iris2d.py --out out/e6_iris2d.pkl
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.environ.get("GCS_SR",
                            os.path.expanduser("~/gcs-science-robotics")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydrake.geometry.optimization import (HPolyhedron, Iris, IrisOptions,
                                           VPolytope)

import stageB_bimanual_ipgcs as sb

DOMAIN_LO = np.array([0.0, 0.0])
DOMAIN_HI = np.array([5.0, 5.0])


def triangulate(poly):
    """Ear clipping: a simple polygon -> convex triangles.

    IRIS takes CONVEX obstacles, and VPolytope silently convexifies its
    input.  Feeding a non-convex obstacle directly makes the planner see
    the convex hull, so a seed sitting in the polygon's notch (free in
    truth, inside the hull) forces IRIS to grow a region across real
    geometry.  Triangulating first makes planner geometry and audit
    geometry identical.
    """
    V = [np.asarray(v, float) for v in poly]
    n = len(V)
    if n < 3:
        return []
    area2 = sum((V[i][0] * V[(i + 1) % n][1] - V[(i + 1) % n][0] * V[i][1])
                for i in range(n))
    idx = list(range(n)) if area2 > 0 else list(range(n))[::-1]

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    def in_tri(p, a, b, c):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    tris, guard = [], 0
    while len(idx) > 3 and guard < 10 * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = (idx[k - 1], idx[k], idx[(k + 1) % len(idx)])
            a, b, c = V[i0], V[i1], V[i2]
            if cross(a, b, c) <= 1e-12:              # reflex or collinear
                continue
            if any(in_tri(V[j], a, b, c) for j in idx
                   if j not in (i0, i1, i2)):
                continue
            tris.append(np.array([a, b, c]))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            break
    if len(idx) == 3:
        tris.append(np.array([V[idx[0]], V[idx[1]], V[idx[2]]]))
    return tris


def point_in_poly(p, poly):
    """Ray-casting containment test on the raw obstacle vertices."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def in_free_space(p, obstacles, margin=0.0):
    if np.any(p < DOMAIN_LO) or np.any(p > DOMAIN_HI):
        return False
    for ob in obstacles:
        if point_in_poly(p, ob):
            return False
        if margin > 0 and poly_distance(p, ob) < margin:
            return False
    return True


def penetration_depth(p, obstacles):
    """Depth of p inside the obstacle set (0 if outside or on a face).

    Binary containment is ill-posed exactly on an obstacle face, which is
    where boundary samples of tangent regions land; depth is well defined
    everywhere and is what a collision claim actually needs.
    """
    worst = 0.0
    for ob in obstacles:
        if point_in_poly(p, ob):
            worst = max(worst, boundary_distance(p, ob))
    return worst


def boundary_distance(p, poly):
    """Distance from p to the polygon boundary (regardless of side)."""
    best = np.inf
    n = len(poly)
    for i in range(n):
        a = np.asarray(poly[i], float)
        b = np.asarray(poly[(i + 1) % n], float)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        best = min(best, np.linalg.norm(p - (a + t * ab)))
    return float(best)


def poly_distance(p, poly):
    """Distance from a point to a polygon boundary (0 if inside)."""
    if point_in_poly(p, poly):
        return 0.0
    best = np.inf
    n = len(poly)
    for i in range(n):
        a = np.asarray(poly[i], float)
        b = np.asarray(poly[(i + 1) % n], float)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        best = min(best, np.linalg.norm(p - (a + t * ab)))
    return best


def free_path(start, goal, obstacles, res=240, margin=0.03):
    """BFS on a fine occupancy grid: a witness path through free space.
    Used to seed IRIS along a route (grid seeds alone leave the narrow
    passages of env_2d uncovered, so the region graph comes out
    disconnected -- the same failure recorded for the 7-DOF bring-up)."""
    from collections import deque
    lo, hi = DOMAIN_LO, DOMAIN_HI
    xs = np.linspace(lo[0], hi[0], res)
    ys = np.linspace(lo[1], hi[1], res)
    free = np.ones((res, res), bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            free[i, j] = in_free_space(np.array([x, y]), obstacles, margin)

    def cell(p):
        return (int(np.argmin(abs(xs - p[0]))), int(np.argmin(abs(ys - p[1]))))

    s_c, g_c = cell(start), cell(goal)
    if not (free[s_c] and free[g_c]):
        return None
    prev = {s_c: None}
    q = deque([s_c])
    while q:
        c = q.popleft()
        if c == g_c:
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (c[0] + dx, c[1] + dy)
                if not (0 <= nb[0] < res and 0 <= nb[1] < res):
                    continue
                if nb in prev or not free[nb]:
                    continue
                prev[nb] = c
                q.append(nb)
    if g_c not in prev:
        return None
    path, c = [], g_c
    while c is not None:
        path.append(np.array([xs[c[0]], ys[c[1]]]))
        c = prev[c]
    return path[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/e6_iris2d.pkl")
    ap.add_argument("--grid", type=int, default=7,
                    help="seed grid resolution (seeds in free space only)")
    ap.add_argument("--seed-margin", type=float, default=0.05)
    ap.add_argument("--no-cover-walk", action="store_true")
    args = ap.parse_args()

    from models.env_2d import obstacles
    obstacles = [np.asarray(o, float) for o in obstacles]
    tris = [t for o in obstacles for t in triangulate(o)]
    obs_sets = [VPolytope(t.T) for t in tris]
    print(f"{len(obstacles)} obstacle polygons -> {len(tris)} convex "
          f"triangles for IRIS", flush=True)
    # audit the triangulation itself: every triangle centroid must be
    # inside its source polygon, and dense samples of the polygons must be
    # covered by some triangle
    rng0 = np.random.default_rng(0)
    bad_c = sum(1 for t in tris
                if not any(point_in_poly(t.mean(0), o) for o in obstacles))
    miss = 0
    for o in obstacles:
        for _ in range(400):
            w = rng0.dirichlet(np.ones(len(o)))
            p = w @ o
            if not point_in_poly(p, o):
                continue
            if not any(point_in_poly(p, t) for t in tris):
                miss += 1
    print(f"  triangulation audit: {bad_c} stray triangles, "
          f"{miss} uncovered interior samples", flush=True)
    domain = HPolyhedron.MakeBox(DOMAIN_LO, DOMAIN_HI)

    opts = IrisOptions()
    opts.require_sample_point_is_contained = True
    opts.iteration_limit = 20
    opts.termination_threshold = 2e-2
    opts.relative_termination_threshold = 1e-3

    # seeds on a grid, kept only if strictly inside free space
    xs = np.linspace(0.15, 4.85, args.grid)
    ys = np.linspace(0.15, 4.85, args.grid)
    seeds = [np.array([x, y]) for x in xs for y in ys
             if in_free_space(np.array([x, y]), obstacles,
                              margin=args.seed_margin)]
    print(f"{len(seeds)} free seeds of {args.grid**2} grid points",
          flush=True)

    # start/goal from the env_2d task
    start, goal = np.array([0.2, 0.2]), np.array([4.8, 4.8])
    if not args.no_cover_walk:
        wit = free_path(start, goal, obstacles, margin=args.seed_margin)
        if wit is None:
            print("WARNING: no witness path in free space", flush=True)
        else:
            print(f"witness path: {len(wit)} grid points", flush=True)
            seeds = [start, goal] + seeds + wit         # cover-walk below

    regions, kept_seeds = [], []
    for k, s in enumerate(seeds):
        # cover-walk: only grow a region if this seed is not covered yet
        if any(np.all(A @ s <= b + 1e-9) for A, b in regions):
            continue
        try:
            H = Iris(obs_sets, s, domain, opts)
        except Exception as e:                       # numerically hard seed
            print(f"  seed {k} failed: {str(e)[:50]}", flush=True)
            continue
        A, b = np.asarray(H.A()), np.asarray(H.b())
        # drop regions that duplicate an existing one (same seed basin)
        dup = False
        for (A2, b2) in regions:
            if A2.shape == A.shape and np.allclose(A2, A, atol=1e-6) \
                    and np.allclose(b2, b, atol=1e-6):
                dup = True
                break
        if dup:
            continue
        regions.append((A, b))
        kept_seeds.append(s)
    print(f"{len(regions)} distinct IRIS regions", flush=True)

    topo = sb.build_topology(regions)
    print(f"topology: {len(topo['portals'])} portals", flush=True)

    inside = lambda q: any(np.all(A @ q <= b + 1e-9) for A, b in regions)
    if not (inside(start) and inside(goal)):
        print("WARNING: start/goal outside the union", flush=True)

    # connectivity check (the reason cover-walk seeding exists)
    adj = {i: set() for i in range(len(regions))}
    for i, j in topo["edges"]:
        adj[i].add(j)
        adj[j].add(i)
    sr = [i for i, (A, b) in enumerate(regions) if np.all(A @ start <= b + 1e-9)]
    seen, stack = set(sr), list(sr)
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    gr = [i for i, (A, b) in enumerate(regions) if np.all(A @ goal <= b + 1e-9)]
    print(f"start/goal connected: {bool(seen & set(gr))} "
          f"(component of start: {len(seen)}/{len(regions)} regions)",
          flush=True)

    lib = {"regions": {i: r for i, r in enumerate(regions)}, "topo": topo,
           "obstacles": obstacles, "seeds": kept_seeds,
           "domain": (DOMAIN_LO, DOMAIN_HI),
           "queries": [("start", "goal")],
           "task_configs": {"start": start, "goal": goal},
           "meta": {"builder": "e6_build_iris2d.py", "world": "env_2d"}}
    pickle.dump(lib, open(args.out, "wb"))
    print(f"written {args.out}", flush=True)


if __name__ == "__main__":
    main()
