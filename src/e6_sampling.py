#!/usr/bin/env python3
"""E6 core: interface taxonomy, interface samplers, roadmap, audits.

An interface between two regions is the intersection polytope
P = X_u ^ X_v.  Two kinds occur in real decompositions:

  * OVERLAP  -- full-dimensional (dim P = d); the generic IRIS case,
  * CONTACT  -- degenerate (dim P < d); shared faces/segments, the
                generic case for box/maze/building libraries.

Both are handled by one construction: find the affine hull of P, reduce
to full-dimensional coordinates inside that hull, and sample there.  In
reduced coordinates the two requested samplers are

  * "contour" -- on the RELATIVE BOUNDARY of P (ray-shoot from an
                 interior point along random directions).  For a 1-D
                 interface this returns its two endpoints, which is
                 where taut shortest paths actually touch,
  * "area"    -- in the RELATIVE INTERIOR of P (hit-and-run).

Every roadmap edge joins two samples that share a region, so the segment
between them lies in that convex region by construction and needs no
motion validation.  Nothing here is trusted on that argument alone:
audit_* re-verifies membership, edges, and the final path against the
raw obstacle geometry, which never passed through the planner.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOL = 1e-7

OVERLAP = "overlap"
CONTACT = "contact"


# ---------------------------------------------------------------- geometry
def chebyshev(A, b):
    """(center, radius) of the largest inscribed ball; radius 0 means the
    polytope is degenerate (lies in a proper affine subspace)."""
    norms = np.linalg.norm(A, axis=1)
    d = A.shape[1]
    c = np.zeros(d + 1)
    c[-1] = -1.0
    res = linprog(c, A_ub=np.hstack([A, norms[:, None]]), b_ub=b,
                  bounds=[(None, None)] * d + [(0, None)], method="highs")
    if not res.success:
        return None, -1.0
    return res.x[:-1], float(res.x[-1])


def implicit_equalities(A, b, tol=1e-9):
    """Rows that hold with equality everywhere in P (m LPs; only called on
    degenerate polytopes, where the Chebyshev radius already vanished)."""
    m, d = A.shape
    eq = np.zeros(m, bool)
    for i in range(m):
        # row i is an implicit equality iff its slack vanishes everywhere,
        # i.e. max_P (b_i - a_i x) = b_i - min_P a_i x <= tol
        res = linprog(A[i], A_ub=A, b_ub=b, bounds=[(None, None)] * d,
                      method="highs")
        if res.success and b[i] - float(res.fun) <= tol:
            eq[i] = True
    return eq


def reduce_to_affine_hull(A, b):
    """Return (x0, N, Ar, br, dim): P = {x0 + N y : Ar y <= br}, with the
    reduced polytope full-dimensional in R^dim."""
    d = A.shape[1]
    ctr, rad = chebyshev(A, b)
    if ctr is None:
        return None
    if rad > 1e-9:                                   # already full-dimensional
        return dict(x0=np.zeros(d), N=np.eye(d), Ar=A, br=b, dim=d,
                    yc=ctr, rad=rad)
    eq = implicit_equalities(A, b)
    E, e = A[eq], b[eq]
    if len(E) == 0:                                  # numerically thin only
        return dict(x0=np.zeros(d), N=np.eye(d), Ar=A, br=b, dim=d,
                    yc=ctr, rad=max(rad, 0.0))
    # null space of the equality block
    _, s, Vt = np.linalg.svd(E)
    rank = int(np.sum(s > max(E.shape) * np.finfo(float).eps * s[0])) \
        if s.size else 0
    N = Vt[rank:].T                                  # d x (d-rank)
    dim = N.shape[1]
    x0 = ctr                                         # feasible, satisfies E
    if dim == 0:
        return dict(x0=x0, N=np.zeros((d, 0)), Ar=np.zeros((0, 0)),
                    br=np.zeros(0), dim=0, yc=np.zeros(0), rad=0.0)
    ineq = ~eq
    Ar = A[ineq] @ N
    br = b[ineq] - A[ineq] @ x0
    yc, rr = chebyshev(Ar, br)
    if yc is None:                                   # fall back to origin
        yc, rr = np.zeros(dim), 0.0
    return dict(x0=x0, N=N, Ar=Ar, br=br, dim=dim, yc=yc, rad=rr)


def classify_interfaces(topo):
    """Attach {kind, dim, hull} to every portal of a topology."""
    out = {}
    for eid, p in topo["portals"].items():
        A, b = np.asarray(p["A"], float), np.asarray(p["b"], float)
        h = reduce_to_affine_hull(A, b)
        if h is None:
            continue
        d = A.shape[1]
        out[eid] = {"u": p["u"], "v": p["v"], "A": A, "b": b, "hull": h,
                    "dim": h["dim"], "center": np.asarray(p["center"]),
                    "kind": OVERLAP if h["dim"] == d else CONTACT}
    return out


# ---------------------------------------------------------------- samplers
def _ray_to_boundary(Ar, br, yc, u):
    """Largest t with yc + t*u feasible (u a unit direction)."""
    Au = Ar @ u
    slack = br - Ar @ yc
    pos = Au > 1e-12
    if not np.any(pos):
        return None
    return float(np.min(slack[pos] / Au[pos]))


def sample_contour(iface, M, rng):
    """M points on the relative boundary of the interface polytope."""
    h = iface["hull"]
    k = h["dim"]
    if k == 0:
        return [h["x0"].copy() for _ in range(M)]
    pts = []
    if k == 1:                       # a segment: both endpoints, then repeat
        ends = []
        for sgn in (1.0, -1.0):
            t = _ray_to_boundary(h["Ar"], h["br"], h["yc"],
                                 np.array([sgn]))
            if t is not None:
                ends.append(h["x0"] + h["N"] @ (h["yc"] + t * np.array([sgn])))
        if not ends:
            return [h["x0"] + h["N"] @ h["yc"]] * M
        for i in range(M):
            pts.append(ends[i % len(ends)].copy())
        return pts
    for _ in range(M):
        u = rng.standard_normal(k)
        u /= np.linalg.norm(u)
        t = _ray_to_boundary(h["Ar"], h["br"], h["yc"], u)
        if t is None:
            continue
        pts.append(h["x0"] + h["N"] @ (h["yc"] + t * u))
    return pts


def sample_area(iface, M, rng, burn=15, thin=6):
    """M points in the relative interior (hit-and-run in the hull)."""
    h = iface["hull"]
    k = h["dim"]
    if k == 0:
        return [h["x0"].copy() for _ in range(M)]
    y = h["yc"].copy()
    pts = []
    it = 0
    while len(pts) < M and it < burn + M * thin * 4:
        it += 1
        u = rng.standard_normal(k)
        u /= np.linalg.norm(u)
        t_hi = _ray_to_boundary(h["Ar"], h["br"], y, u)
        t_lo = _ray_to_boundary(h["Ar"], h["br"], y, -u)
        if t_hi is None or t_lo is None:
            continue
        y = y + rng.uniform(-t_lo, t_hi) * u
        if it > burn and it % thin == 0:
            pts.append(h["x0"] + h["N"] @ y)
    while len(pts) < M:                              # degenerate fallback
        pts.append(h["x0"] + h["N"] @ h["yc"])
    return pts[:M]


def sample_center(iface, M, rng):
    """Baseline: the deterministic Chebyshev representative, M copies."""
    h = iface["hull"]
    x = h["x0"] + (h["N"] @ h["yc"] if h["dim"] else 0.0)
    return [np.asarray(x).copy() for _ in range(M)]


def sample_contour_in(iface, M, rng, lam=1e-3):
    """Contour samples pulled a fraction lam toward the interface centre.

    Required on interfaces of dimension < d.  The relative boundary of a
    (d-1)-dimensional contact face is a (d-2)-dimensional PINCH set --
    corners in 2-D, edges in 3-D -- where regions meet in measure zero
    and no passage exists; roadmaps that place nodes there can return
    paths cheaper than the true optimum (audit_junctions catches this).
    Blending toward the centre of a convex set keeps the sample strictly
    inside while preserving the boundary-hugging geometry that taut
    shortest paths want.
    """
    h = iface["hull"]
    c = h["x0"] + (h["N"] @ h["yc"] if h["dim"] else 0.0)
    return [(1.0 - lam) * np.asarray(x) + lam * c
            for x in sample_contour(iface, M, rng)]


SAMPLERS = {"contour": sample_contour, "contour_in": sample_contour_in,
            "area": sample_area, "center": sample_center}


# ---------------------------------------------------------------- roadmap
def region_membership(x, regions, tol=TOL):
    return frozenset(i for i, (A, b) in enumerate(regions)
                     if np.all(A @ x <= b + tol))


def build_roadmap(topo, ifaces, mode, M, seed=0, dedup_tol=1e-9,
                  membership="declared"):
    """Equal budget M per interface, of both kinds, per the study design.

    membership="declared" (default, and the only sound choice in general):
    a sample's regions are the two parents of its interface, i.e. exactly
    the transitions the topology declares traversable.

    membership="geometric" instead uses every region whose H-description
    contains the point.  That is UNSOUND on libraries whose barriers have
    zero thickness and live in the topology rather than in the geometry
    (maze/box libraries: two cells separated by a wall still share a
    closed face, so a sample on that face belongs to both cells and the
    roadmap teleports through the wall).  Kept only to reproduce the
    failure in the audit table.
    """
    rng = np.random.default_rng(seed)
    regions = topo["regions"]
    sampler = SAMPLERS[mode]
    t0 = time.perf_counter()
    pos, node_iface = [], []
    for eid, iface in sorted(ifaces.items()):
        for x in sampler(iface, M, rng):
            pos.append(np.asarray(x, float))
            node_iface.append(eid)
    if not pos:
        raise RuntimeError("no samples")
    pos = np.asarray(pos)
    # Deduplicate exact repeats (center mode; 1-D contour with M > 2; and
    # corner points shared by two interfaces of the same region).  The
    # merged node must inherit the parents of EVERY interface that
    # produced it -- keeping only the first one silently disconnects the
    # roadmap wherever adjacent portals share a vertex.
    keep, seen, merged = [], {}, []
    for i, x in enumerate(pos):
        key = tuple(np.round(x / max(dedup_tol, 1e-12)).astype(np.int64))
        if key in seen:
            merged[seen[key]].append(node_iface[i])
            continue
        seen[key] = len(keep)
        keep.append(i)
        merged.append([node_iface[i]])
    pos = pos[keep]
    node_iface = [node_iface[i] for i in keep]
    t_sample = time.perf_counter() - t0

    t0 = time.perf_counter()
    if membership == "declared":
        mems = [frozenset(r for e in es for r in
                          (ifaces[e]["u"], ifaces[e]["v"])) for es in merged]
    else:
        mems = [region_membership(x, regions) for x in pos]
    region_nodes = {r: [] for r in range(len(regions))}
    for i, m in enumerate(mems):
        for r in m:
            region_nodes[r].append(i)
    t_member = time.perf_counter() - t0
    return {"pos": pos, "node_iface": node_iface, "mems": mems,
            "region_nodes": region_nodes, "t_sample": t_sample,
            "t_member": t_member, "mode": mode, "M": M,
            "membership": membership}


def edges_from_regions(region_nodes, n_nodes, pos, alive_nodes=None):
    """Intra-region cliques -> deduped undirected edge arrays."""
    ii, jj = [], []
    for r, nodes in region_nodes.items():
        if alive_nodes is not None:
            nodes = [k for k in nodes if k in alive_nodes]
        if len(nodes) < 2:
            continue
        a = np.asarray(sorted(nodes))
        iu, ju = np.triu_indices(len(a), k=1)
        ii.append(a[iu])
        jj.append(a[ju])
    if not ii:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(0))
    ii = np.concatenate(ii).astype(np.int64)
    jj = np.concatenate(jj).astype(np.int64)
    keys = np.unique(ii * n_nodes + jj)
    ii = keys // n_nodes
    jj = keys % n_nodes
    ww = np.linalg.norm(pos[ii] - pos[jj], axis=1)
    return ii, jj, ww


def shortest_path(rm, regions, s, g, alive_nodes=None):
    """Dijkstra from s to g over the roadmap; returns (cost, path points)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    pos = rm["pos"]
    N = len(pos)
    s_idx, g_idx = N, N + 1
    ii, jj, ww = edges_from_regions(rm["region_nodes"], N + 2, pos,
                                    alive_nodes)
    s_mem = region_membership(s, regions)
    g_mem = region_membership(g, regions)
    qi, qj = [], []
    for r in s_mem:
        for k in rm["region_nodes"][r]:
            if alive_nodes is None or k in alive_nodes:
                qi.append(s_idx)
                qj.append(k)
    for r in g_mem:
        for k in rm["region_nodes"][r]:
            if alive_nodes is None or k in alive_nodes:
                qi.append(g_idx)
                qj.append(k)
    if s_mem & g_mem:
        qi.append(s_idx)
        qj.append(g_idx)
    if not qi:
        return np.inf, None
    allpos = np.vstack([pos, s, g])
    qi = np.asarray(qi, np.int64)
    qj = np.asarray(qj, np.int64)
    qw = np.linalg.norm(allpos[qi] - allpos[qj], axis=1)
    I = np.concatenate([ii, qi])
    J = np.concatenate([jj, qj])
    W = np.concatenate([ww, qw])
    m = coo_matrix((np.concatenate([W, W]),
                    (np.concatenate([I, J]), np.concatenate([J, I]))),
                   shape=(N + 2, N + 2)).tocsr()
    dist, pred = dijkstra(m, indices=s_idx, return_predecessors=True)
    if not np.isfinite(dist[g_idx]):
        return np.inf, None
    path = [g_idx]
    while path[-1] != s_idx:
        path.append(pred[path[-1]])
    path = path[::-1]
    return float(dist[g_idx]), allpos[path]


def corridor_from_path(path_mems, pair_eid):
    """Portal sequence via a greedy region walk over the path's node
    memberships.  Consecutive portals are forced to share a transit
    region, so the restriction SOCP chain is a valid path -- reading the
    portal tag off each node is unsound (an edge can join two portal
    nodes through a third region) and yields an empty, i.e. straight-line
    and infeasible, corridor for interior-only paths."""
    edge_regs = [path_mems[k] & path_mems[k + 1]
                 for k in range(len(path_mems) - 1)]
    if any(not er for er in edge_regs):
        return None
    pseq, cur = [], None
    for k, er in enumerate(edge_regs):
        if cur is not None and cur in er:
            continue
        best_r, best_len = None, -1
        for r in er:
            L = 0
            while k + L < len(edge_regs) and r in edge_regs[k + L]:
                L += 1
            if L > best_len:
                best_r, best_len = r, L
        if cur is not None:
            eid = pair_eid.get(frozenset((cur, best_r)))
            if eid is None:
                return None
            pseq.append(eid)
        cur = best_r
    return pseq


def path_corridor(rm, regions, s, g, path_idx=None, path=None):
    """Membership walk for a returned path (endpoints included)."""
    mems = [region_membership(p, regions) for p in path]
    return mems


# ------------------------------------------------- zero-thickness walls
def box_walls(topo, tol=1e-9):
    """Faces shared by adjacent boxes that the topology does NOT declare
    traversable: the maze/box libraries encode walls only in the portal
    list, so this reconstructs them from geometry + topology and gives an
    obstacle-style ground truth for libraries that ship no obstacles."""
    lo, hi = np.asarray(topo["bbox_lo"]), np.asarray(topo["bbox_hi"])
    declared = {frozenset((p["u"], p["v"])) for p in topo["portals"].values()}
    n, d = lo.shape
    walls = []
    for i in range(n):
        for j in range(i + 1, n):
            if frozenset((i, j)) in declared:
                continue
            for k in range(d):
                touch = None
                if abs(hi[i, k] - lo[j, k]) < tol:
                    touch = hi[i, k]
                elif abs(hi[j, k] - lo[i, k]) < tol:
                    touch = hi[j, k]
                if touch is None:
                    continue
                spans = []
                ok = True
                for m in range(d):
                    if m == k:
                        continue
                    a = max(lo[i, m], lo[j, m])
                    b = min(hi[i, m], hi[j, m])
                    if b - a <= tol:
                        ok = False
                        break
                    spans.append((m, a, b))
                if ok:
                    walls.append({"dim": k, "coord": float(touch),
                                  "spans": spans, "pair": (i, j)})
    return walls


def segment_crosses_wall(p, q, walls, tol=1e-9):
    """Max transversal crossing depth of segment pq through any wall
    (0 if it only grazes or touches endpoints)."""
    worst = 0.0
    for w in walls:
        k, c = w["dim"], w["coord"]
        a, b = p[k] - c, q[k] - c
        if (a > tol and b > tol) or (a < -tol and b < -tol):
            continue
        if abs(a - b) < 1e-15:
            continue
        t = a / (a - b)
        if not (tol < t < 1 - tol):
            continue
        x = p + t * (q - p)
        inside = all(lo + tol < x[m] < hi_ - tol for (m, lo, hi_) in w["spans"])
        if inside:
            worst = max(worst, min(abs(a), abs(b)))
    return worst


def audit_wall_crossings(rm, walls, path=None, max_edges=20000, seed=0):
    if not walls:
        return {"walls": 0, "edge_wall_crossings": 0, "path_wall_crossings": 0}
    pos = rm["pos"]
    ii, jj, _ = edges_from_regions(rm["region_nodes"], len(pos), pos)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ii))
    if len(idx) > max_edges:
        idx = rng.choice(idx, max_edges, replace=False)
    n_edge = sum(1 for e in idx
                 if segment_crosses_wall(pos[int(ii[e])], pos[int(jj[e])],
                                         walls) > 0)
    n_path = 0
    if path is not None:
        n_path = sum(1 for k in range(len(path) - 1)
                     if segment_crosses_wall(path[k], path[k + 1], walls) > 0)
    return {"walls": len(walls), "edge_wall_crossings": int(n_edge),
            "path_wall_crossings": int(n_path)}


# ---------------------------------------------------------------- audits
def audit_samples(rm, ifaces, regions, obstacles, penetration):
    """Samples: inside their own interface, inside both parents, and clear
    of the raw obstacle geometry.

    Obstacle clearance is reported as PENETRATION DEPTH, not a binary
    in/out test: interface polytopes are tangent to obstacles wherever
    IRIS stopped growing, so boundary samples land exactly on obstacle
    faces, where a ray-casting test is ill-posed.  Depth is the
    physically meaningful quantity and is compared against tolerance.
    """
    worst_iface, worst_parent, worst_pen = 0.0, 0.0, 0.0
    for i, x in enumerate(rm["pos"]):
        eid = rm["node_iface"][i]
        f = ifaces[eid]
        worst_iface = max(worst_iface, float(np.max(f["A"] @ x - f["b"])))
        for r in (f["u"], f["v"]):
            A, b = regions[r]
            worst_parent = max(worst_parent, float(np.max(A @ x - b)))
        worst_pen = max(worst_pen, penetration(x, obstacles))
    return {"n_samples": len(rm["pos"]),
            "max_interface_violation": worst_iface,
            "max_parent_violation": worst_parent,
            "max_sample_penetration": worst_pen}


def audit_edges(rm, regions, obstacles, penetration, n_probe=25,
                max_edges=20000, seed=0):
    """Every edge must have a witness region containing both endpoints;
    probe points along the segment must lie in that region and in free
    space (an independent check of the convexity argument)."""
    pos = rm["pos"]
    ii, jj, _ = edges_from_regions(rm["region_nodes"], len(pos), pos)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ii))
    if len(idx) > max_edges:
        idx = rng.choice(idx, max_edges, replace=False)
    no_witness = 0
    worst_region, worst_pen = 0.0, 0.0
    ts = np.linspace(0, 1, n_probe)
    for e in idx:
        a, b_ = int(ii[e]), int(jj[e])
        common = rm["mems"][a] & rm["mems"][b_]
        if not common:
            no_witness += 1
            continue
        r = min(common)
        A, bb = regions[r]
        for t in ts:
            p = (1 - t) * pos[a] + t * pos[b_]
            worst_region = max(worst_region, float(np.max(A @ p - bb)))
            worst_pen = max(worst_pen, penetration(p, obstacles))
    return {"n_edges": int(len(ii)), "edges_audited": int(len(idx)),
            "edges_without_witness_region": no_witness,
            "max_probe_region_violation": worst_region,
            "max_probe_penetration": worst_pen}


def audit_junctions(path, regions, eps=1e-4, tol=1e-9, dim_cache=None):
    """Every interior vertex of the path must be a TRAVERSABLE transition.

    A transition from region R_a to R_b at p is traversable only if the
    two regions meet in a set of dimension >= d-1 that contains p: a
    shared face (dim d-1) or a genuine overlap (dim d).  Regions touching
    in lower dimension -- a corner in 2-D, an edge in 3-D -- provide no
    passage, yet every pointwise check still succeeds there, because each
    incident segment lies in its own region and the pinch point itself is
    in both.  This is how a roadmap built from closed-region membership
    can return a path CHEAPER than the true optimum.
    """
    if path is None or len(path) < 3:
        return {"bad_junctions": 0, "junctions": 0}
    d = len(path[0])
    cache = {} if dim_cache is None else dim_cache

    def pair_dim(i, j):
        key = (i, j) if i <= j else (j, i)
        if key not in cache:
            if i == j:
                cache[key] = d
            else:
                A = np.vstack([regions[i][0], regions[j][0]])
                b = np.concatenate([regions[i][1], regions[j][1]])
                h = reduce_to_affine_hull(A, b)
                cache[key] = -1 if h is None else h["dim"]
        return cache[key]

    bad = 0
    for k in range(1, len(path) - 1):
        p = path[k]
        u, v = path[k - 1] - p, path[k + 1] - p
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-12 or nv < 1e-12:
            continue
        a = p + (eps / nu) * u
        b = p + (eps / nv) * v
        Ra = [i for i, (A, bb) in enumerate(regions)
              if np.all(A @ a <= bb + tol)]
        Rb = [i for i, (A, bb) in enumerate(regions)
              if np.all(A @ b <= bb + tol)]
        ok = False
        for i in Ra:
            for j in Rb:
                if pair_dim(i, j) < d - 1:
                    continue
                Aij = np.vstack([regions[i][0], regions[j][0]])
                bij = np.concatenate([regions[i][1], regions[j][1]])
                if np.all(Aij @ p <= bij + 1e-7):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            bad += 1
    return {"bad_junctions": int(bad), "junctions": int(len(path) - 2)}


def audit_path(path, regions, obstacles, penetration, n_probe=2000):
    """Dense resampling of the returned polyline against ground truth."""
    if path is None:
        return {"status": "no_path"}
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(seg.sum())
    L = np.concatenate([[0], np.cumsum(seg)])
    ts = np.linspace(0, total, n_probe)
    worst_pen, n_out = 0.0, 0
    for t in ts:
        k = int(np.searchsorted(L, t, "right") - 1)
        k = min(max(k, 0), len(seg) - 1)
        lam = 0.0 if seg[k] < 1e-12 else (t - L[k]) / seg[k]
        p = (1 - lam) * path[k] + lam * path[k + 1]
        worst_pen = max(worst_pen, penetration(p, obstacles))
        if not any(np.all(A @ p <= b + 1e-6) for A, b in regions):
            n_out += 1
    return {"cost": total, "probe_points": n_probe,
            "max_path_penetration": worst_pen, "probe_outside_union": n_out}
