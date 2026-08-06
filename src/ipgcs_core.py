"""iP-GCS Stage A core: waypoint graph, gate-aware A*, informed region pruning,
restriction polish, and a portal-distance lower bound with certificate gap.

Operates on the cell-graph JSON produced by GNN-DIP's `cdt_decompose`
(faces = convex cells, edges = portals). Cells are convex, so any segment
between points on two portals of the same cell is collision-free by
construction — no collision checking anywhere in this module.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

CDT_BIN = os.environ.get("CDT_BIN", "cdt_decompose")


# ── geometry ──────────────────────────────────────────────────────────────────

def point_seg_dist(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-20 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def seg_seg_dist(a1, a2, b1, b2):
    if segments_intersect(a1, a2, b1, b2):
        return 0.0
    return min(point_seg_dist(a1, b1, b2), point_seg_dist(a2, b1, b2),
               point_seg_dist(b1, a1, a2), point_seg_dist(b2, a1, a2))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1, p2, q1, q2):
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    eps = 1e-12

    def on_seg(a, b, c):
        return (abs(_cross(a, b, c)) < eps
                and min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
                and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)

    return (on_seg(q1, q2, p1) or on_seg(q1, q2, p2)
            or on_seg(p1, p2, q1) or on_seg(p1, p2, q2))


def point_in_convex_poly(p, verts):
    n = len(verts)
    sign = 0
    for i in range(n):
        c = _cross(verts[i], verts[(i + 1) % n], p)
        if abs(c) < 1e-14:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def point_poly_dist(p, verts):
    if point_in_convex_poly(p, verts):
        return 0.0
    n = len(verts)
    return min(point_seg_dist(p, verts[i], verts[(i + 1) % n]) for i in range(n))


def seg_intersects_poly(a, b, verts):
    if point_in_convex_poly(a, verts) or point_in_convex_poly(b, verts):
        return True
    n = len(verts)
    return any(segments_intersect(a, b, verts[i], verts[(i + 1) % n])
               for i in range(n))


def _seg_min_f(ax, ay, bx, by, sx, sy, gx, gy):
    """Exact min over x in segment [a,b] of |s-x|+|x-g| via the reflection
    method (f is convex along the segment, so clamping t is exact)."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-20:
        return math.hypot(sx - ax, sy - ay) + math.hypot(ax - gx, ay - gy)
    vx, vy = gx - ax, gy - ay
    k = (vx * dx + vy * dy) / len_sq
    rx, ry = ax + 2 * k * dx - vx, ay + 2 * k * dy - vy   # g reflected
    ex, ey = rx - sx, ry - sy
    denom = ex * dy - ey * dx
    if abs(denom) < 1e-15:
        t = ((sx - ax) * dx + (sy - ay) * dy) / len_sq
    else:
        t = ((ax - sx) * ey - (ay - sy) * ex) / denom
    best = float("inf")
    for tt in (0.0, min(max(t, 0.0), 1.0), 1.0):
        x, y = ax + tt * dx, ay + tt * dy
        f = math.hypot(sx - x, sy - y) + math.hypot(x - gx, y - gy)
        best = f if f < best else best
    return best


def cell_informed_lb(verts, s, g):
    """Exact min over the convex cell of |s-x|+|x-g| (tier-3)."""
    if seg_intersects_poly(s, g, verts):
        return float(np.linalg.norm(s - g))
    n = len(verts)
    sx, sy, gx, gy = float(s[0]), float(s[1]), float(g[0]), float(g[1])
    return min(_seg_min_f(verts[i][0], verts[i][1],
                          verts[(i + 1) % n][0], verts[(i + 1) % n][1],
                          sx, sy, gx, gy) for i in range(n))


# ── cell graph ────────────────────────────────────────────────────────────────

@dataclass
class CellGraph:
    cells: dict            # id -> {verts: np.array, centroid, area, clearance}
    portals: dict          # id -> {u, v, a, b, mid, length, clearance}
    cell_portals: dict     # cell id -> [portal ids]
    start: np.ndarray
    goal: np.ndarray
    start_cell: int
    goal_cell: int
    bounds: list
    obstacles: list = field(default_factory=list)


def decompose(map_path: str, cdt_bin: str = CDT_BIN,
              out_json: str | None = None) -> tuple[CellGraph, float]:
    out_json = out_json or os.path.join(
        tempfile.gettempdir(), f"ipgcs_graph_{abs(hash(map_path))}.json")
    t0 = time.perf_counter()
    subprocess.run([cdt_bin, "--map", map_path, "--out", out_json],
                   check=True, capture_output=True)
    t_decomp = time.perf_counter() - t0
    g = json.load(open(out_json))
    m = json.load(open(map_path))

    cells = {}
    for f in g["faces"]:
        verts = np.asarray(f["vertices"], dtype=float)
        vt = [(float(x), float(y)) for x, y in verts]
        cells[f["id"]] = {
            "verts": verts,
            "verts_t": vt,
            "bbox": (min(x for x, _ in vt), min(y for _, y in vt),
                     max(x for x, _ in vt), max(y for _, y in vt)),
            "centroid": np.asarray(f["centroid"], dtype=float),
            "area": f.get("area", 0.0),
            "clearance": f.get("clearance", 0.0),
        }
    portals, cell_portals = {}, {c: [] for c in cells}
    for e in g["edges"]:
        a, b = np.asarray(e["portal"][0], float), np.asarray(e["portal"][1], float)
        portals[e["id"]] = {
            "u": e["u"], "v": e["v"], "a": a, "b": b,
            "mid": np.asarray(e["midpoint"], float),
            "length": e["length"], "clearance": e.get("clearance", 0.0),
        }
        cell_portals[e["u"]].append(e["id"])
        cell_portals[e["v"]].append(e["id"])

    return CellGraph(
        cells=cells, portals=portals, cell_portals=cell_portals,
        start=np.asarray(m["start"], float), goal=np.asarray(m["goal"], float),
        start_cell=g["start_face"], goal_cell=g["goal_face"],
        bounds=g.get("bounds", m.get("bounds")),
        obstacles=[_obstacle_verts(o) for o in m.get("obstacles", [])],
    ), t_decomp


def _obstacle_verts(o: dict) -> np.ndarray:
    if "vertices" in o:
        return np.asarray(o["vertices"], float)
    return np.asarray([[o["xmin"], o["ymin"]], [o["xmax"], o["ymin"]],
                       [o["xmax"], o["ymax"]], [o["xmin"], o["ymax"]]], float)


# ── priors (gate scores in [0,1]; higher = prefer) ────────────────────────────

def prior_scores(cg: CellGraph, kind: str) -> dict | None:
    if kind == "uniform":
        return None
    if kind == "gfp_proxy":            # structural only: gate clearance
        mx = max(p["clearance"] for p in cg.portals.values()) or 1.0
        return {pid: p["clearance"] / mx for pid, p in cg.portals.items()}
    if kind == "dg_heur":              # GNN-DIP's shipped fallback: 1/(1+dg)
        return {pid: 1.0 / (1.0 + float(np.linalg.norm(p["mid"] - cg.goal)))
                for pid, p in cg.portals.items()}
    raise ValueError(kind)


# ── waypoint graph (M samples per portal) ─────────────────────────────────────

@dataclass
class WaypointGraph:
    pos: np.ndarray                 # (N,2) node positions; 0=start, 1=goal
    adj: list                       # node -> list[(nbr, length)]
    node_portal: dict               # node -> portal id (interior nodes only)
    portal_nodes: dict              # portal id -> [node indices]
    n_edges: int


def build_waypoint_graph(cg: CellGraph, M: int = 1,
                         alive_cells: set | None = None,
                         informed_c: float | None = None) -> WaypointGraph:
    """Nodes = M samples per portal (midpoint first) + start/goal.
    Edges = all pairs of nodes on distinct portals of the same cell.
    With `informed_c`, portal samples with |s-p|+|p-g| > c are dropped
    (admissible: such points cannot lie on a path better than c)."""
    alive = alive_cells if alive_cells is not None else set(cg.cells)
    pos = [cg.start, cg.goal]
    node_portal, portal_nodes = {}, {}
    for pid, p in cg.portals.items():
        if p["u"] not in alive or p["v"] not in alive:
            continue
        # Endpoints included for M>1: string-pulled optima hug portal corners.
        ts = [0.5] if M <= 1 else list(np.linspace(0.0, 1.0, M))
        pts = [p["a"] + t * (p["b"] - p["a"]) for t in ts]
        if informed_c is not None:
            pts = [q for q in pts
                   if np.linalg.norm(cg.start - q) + np.linalg.norm(q - cg.goal)
                   <= informed_c + 1e-9]
            if not pts:  # keep portal reachable: retain its best point
                best_t = min(np.linspace(0, 1, 33),
                             key=lambda t: np.linalg.norm(cg.start - (p["a"] + t * (p["b"] - p["a"])))
                             + np.linalg.norm((p["a"] + t * (p["b"] - p["a"])) - cg.goal))
                pts = [p["a"] + best_t * (p["b"] - p["a"])]
        idxs = []
        for q in pts:
            idxs.append(len(pos))
            node_portal[len(pos)] = pid
            pos.append(q)
        portal_nodes[pid] = idxs

    pos = np.asarray(pos)
    adj = [[] for _ in range(len(pos))]
    n_edges = 0

    def connect(i, j):
        nonlocal n_edges
        d = float(np.linalg.norm(pos[i] - pos[j]))
        adj[i].append((j, d))
        adj[j].append((i, d))
        n_edges += 1

    for c in alive:
        pids = [pid for pid in cg.cell_portals.get(c, []) if pid in portal_nodes]
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                for u in portal_nodes[pids[i]]:
                    for v in portal_nodes[pids[j]]:
                        connect(u, v)
        if c == cg.start_cell:
            for pid in pids:
                for v in portal_nodes[pid]:
                    connect(0, v)
        if c == cg.goal_cell:
            for pid in pids:
                for v in portal_nodes[pid]:
                    connect(1, v)
    if cg.start_cell == cg.goal_cell and cg.start_cell in alive:
        connect(0, 1)
    return WaypointGraph(pos, adj, node_portal, portal_nodes, n_edges)


# ── gate-aware A* ─────────────────────────────────────────────────────────────

def astar(wg: WaypointGraph, scores: dict | None = None, beta: float = 3.0):
    """A* start(0) -> goal(1). With `scores`, edge costs are modulated
    DIP-style (len * exp(-beta*score(target portal))) for expansion ordering;
    the returned cost is always the true geometric length of the found path."""
    pos, adj = wg.pos, wg.adj
    goal = pos[1]
    hmul = math.exp(-beta) if scores is not None else 1.0

    def h(i):
        return float(np.linalg.norm(pos[i] - goal)) * hmul

    t0 = time.perf_counter()
    dist = {0: 0.0}
    prev = {}
    pq = [(h(0), 0)]
    expansions = 0
    closed = set()
    while pq:
        f, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expansions += 1
        if u == 1:
            break
        for v, d in adj[u]:
            w = d
            if scores is not None:
                w = d * math.exp(-beta * scores.get(wg.node_portal.get(v, -1), 0.5))
            nd = dist[u] + w
            if nd < dist.get(v, float("inf")) - 1e-15:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd + h(v), v))
    t = time.perf_counter() - t0
    if 1 not in prev and 1 not in dist:
        return None, float("inf"), expansions, t
    path = [1]
    while path[-1] != 0:
        path.append(prev[path[-1]])
    path.reverse()
    true_cost = float(sum(np.linalg.norm(pos[path[i + 1]] - pos[path[i]])
                          for i in range(len(path) - 1)))
    return path, true_cost, expansions, t


# ── restriction polish (corridor-exact refinement) ────────────────────────────

def _optimize_point_on_portal(a, b, prev_p, next_p, cur=None):
    """Exact min over x in segment [a,b] of |prev-x|+|x-next|.

    NOTE: DIP's C++ optimizePointOnPortal always uses the reflection
    construction, which is only valid when prev/next lie on the SAME side
    of the portal line. When the path CROSSES the portal (the common case)
    the optimum is the direct intersection of [prev,next] with the line.
    We evaluate both candidate constructions (+ endpoints + current point)
    and take the argmin — exact by convexity, and monotone by construction."""
    d = b - a
    len_sq = float(d @ d)
    if len_sq < 1e-20:
        return a
    cands = [0.0, 1.0]

    # (i) same-side construction: reflect `next` across the portal line
    v = next_p - a
    proj = ((v @ d) / len_sq) * d
    reflected = a + 2.0 * proj - v
    for target in (reflected, next_p):        # (ii) crossing construction
        r = target - prev_p
        denom = r[0] * d[1] - r[1] * d[0]
        if abs(denom) < 1e-15:
            t = float((prev_p - a) @ d / len_sq)
        else:
            w = a - prev_p
            t = (w[0] * r[1] - w[1] * r[0]) / denom
        cands.append(float(np.clip(t, 0.0, 1.0)))
    if cur is not None:
        cands.append(cur)

    def f(t):
        x = a + t * d
        return (math.hypot(*(prev_p - x)) + math.hypot(*(x - next_p)))

    return a + min(cands, key=f) * d


def polish(wg: WaypointGraph, cg: CellGraph, path, sweeps: int = 60):
    """Coordinate sweeps of the reflection step over the path's portals.
    Converges to the corridor-optimal polyline (= restriction solution)."""
    pts = [wg.pos[i].copy() for i in path]
    segs = [(cg.portals[wg.node_portal[i]]["a"], cg.portals[wg.node_portal[i]]["b"])
            if i in wg.node_portal else None for i in path]
    for _ in range(sweeps):
        moved = 0.0
        for k in range(1, len(pts) - 1):
            if segs[k] is None:
                continue
            newp = _optimize_point_on_portal(segs[k][0], segs[k][1],
                                             pts[k - 1], pts[k + 1])
            moved += float(np.linalg.norm(newp - pts[k]))
            pts[k] = newp
        if moved < 1e-12:
            break
    cost = float(sum(np.linalg.norm(pts[i + 1] - pts[i])
                     for i in range(len(pts) - 1)))
    return pts, cost


# ── informed region pruning (3-tier) ──────────────────────────────────────────

def informed_prune(cg: CellGraph, c_best: float, alive: set):
    """Return (survivors, stats). Cell v dies iff
    min_{x in v} |s-x|+|x-g| > c_best  (admissible for path-length cost)."""
    s, g = cg.start, cg.goal
    ctr = 0.5 * (s + g)
    a_len = 0.5 * c_best
    c_len = 0.5 * float(np.linalg.norm(g - s))
    b_len = math.sqrt(max(a_len * a_len - c_len * c_len, 0.0))
    th = math.atan2(g[1] - s[1], g[0] - s[0])
    ex = math.sqrt((a_len * math.cos(th)) ** 2 + (b_len * math.sin(th)) ** 2)
    ey = math.sqrt((a_len * math.sin(th)) ** 2 + (b_len * math.cos(th)) ** 2)

    stats = {"tier1_kill": 0, "tier2_kill": 0, "tier3_kill": 0,
             "tier3_calls": 0, "alive": 0, "t_tier12": 0.0, "t_tier3": 0.0}
    sx, sy, gx, gy = float(s[0]), float(s[1]), float(g[0]), float(g[1])
    cx, cy = float(ctr[0]), float(ctr[1])
    # solver-noise-safe threshold + force-keep endpoint cells (W3 borderline
    # fix: at c_best == exact optimum, corridor cells sit AT the threshold)
    tol = max(1e-9, 1e-6 * c_best)
    keep_always = {cg.start_cell, cg.goal_cell}
    survivors = set()
    t0 = time.perf_counter()
    for cid in alive:
        if cid in keep_always:
            survivors.add(cid)
            continue
        cell = cg.cells[cid]
        x0, y0, x1, y1 = cell["bbox"]
        if x0 > cx + ex or x1 < cx - ex or y0 > cy + ey or y1 < cy - ey:
            stats["tier1_kill"] += 1
            continue
        vt = cell["verts_t"]
        lb2 = (_pt_poly_dist_s(sx, sy, vt) + _pt_poly_dist_s(gx, gy, vt))
        if lb2 > c_best + tol:
            stats["tier2_kill"] += 1
            continue
        t1 = time.perf_counter()
        stats["tier3_calls"] += 1
        lb3 = _cell_informed_lb_s(vt, sx, sy, gx, gy)
        stats["t_tier3"] += time.perf_counter() - t1
        if lb3 > c_best + tol:
            stats["tier3_kill"] += 1
            continue
        survivors.add(cid)
    stats["t_tier12"] = time.perf_counter() - t0 - stats["t_tier3"]
    stats["alive"] = len(survivors)
    return survivors, stats


# scalar fast paths for the pruning loop ──────────────────────────────────────

def _pt_in_poly_s(px, py, vt):
    sign = 0
    n = len(vt)
    for i in range(n):
        x1, y1 = vt[i]
        x2, y2 = vt[(i + 1) % n]
        c = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if -1e-14 < c < 1e-14:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _pt_seg_dist_s(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    t = 0.0 if ll < 1e-20 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _pt_poly_dist_s(px, py, vt):
    if _pt_in_poly_s(px, py, vt):
        return 0.0
    n = len(vt)
    return min(_pt_seg_dist_s(px, py, vt[i][0], vt[i][1],
                              vt[(i + 1) % n][0], vt[(i + 1) % n][1])
               for i in range(n))


def _segs_cross_s(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y):
    d1 = (q2x - q1x) * (p1y - q1y) - (q2y - q1y) * (p1x - q1x)
    d2 = (q2x - q1x) * (p2y - q1y) - (q2y - q1y) * (p2x - q1x)
    d3 = (p2x - p1x) * (q1y - p1y) - (p2y - p1y) * (q1x - p1x)
    d4 = (p2x - p1x) * (q2y - p1y) - (p2y - p1y) * (q2x - p1x)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    eps = 1e-12
    if abs(d1) < eps and min(q1x, q2x) - eps <= p1x <= max(q1x, q2x) + eps \
            and min(q1y, q2y) - eps <= p1y <= max(q1y, q2y) + eps:
        return True
    if abs(d2) < eps and min(q1x, q2x) - eps <= p2x <= max(q1x, q2x) + eps \
            and min(q1y, q2y) - eps <= p2y <= max(q1y, q2y) + eps:
        return True
    if abs(d3) < eps and min(p1x, p2x) - eps <= q1x <= max(p1x, p2x) + eps \
            and min(p1y, p2y) - eps <= q1y <= max(p1y, p2y) + eps:
        return True
    if abs(d4) < eps and min(p1x, p2x) - eps <= q2x <= max(p1x, p2x) + eps \
            and min(p1y, p2y) - eps <= q2y <= max(p1y, p2y) + eps:
        return True
    return False


def _cell_informed_lb_s(vt, sx, sy, gx, gy):
    if _pt_in_poly_s(sx, sy, vt) or _pt_in_poly_s(gx, gy, vt):
        return math.hypot(gx - sx, gy - sy)
    n = len(vt)
    for i in range(n):
        if _segs_cross_s(sx, sy, gx, gy, vt[i][0], vt[i][1],
                         vt[(i + 1) % n][0], vt[(i + 1) % n][1]):
            return math.hypot(gx - sx, gy - sy)
    return min(_seg_min_f(vt[i][0], vt[i][1],
                          vt[(i + 1) % n][0], vt[(i + 1) % n][1],
                          sx, sy, gx, gy) for i in range(n))


# ── portal-distance lower bound (poor man's relaxation) ───────────────────────

def portal_lb(cg: CellGraph, alive: set | None = None) -> float:
    """Dijkstra over portals with edge weights = min distance between portal
    segments sharing a cell. Any feasible path's length is >= this value
    (each cell transit is >= the segment-segment distance), so it is a valid
    global lower bound; on a pruned graph it is still valid because pruning
    is admissible (the optimum always survives)."""
    alive = alive if alive is not None else set(cg.cells)
    INF = float("inf")
    dist = {}
    pq = []
    for pid in cg.cell_portals.get(cg.start_cell, []):
        p = cg.portals[pid]
        if p["u"] in alive and p["v"] in alive:
            d = point_seg_dist(cg.start, p["a"], p["b"])
            dist[pid] = d
            heapq.heappush(pq, (d, pid))
    if cg.start_cell == cg.goal_cell:
        return float(np.linalg.norm(cg.goal - cg.start))
    best = INF
    goal_pids = set(pid for pid in cg.cell_portals.get(cg.goal_cell, [])
                    if cg.portals[pid]["u"] in alive and cg.portals[pid]["v"] in alive)
    closed = set()
    while pq:
        d, pid = heapq.heappop(pq)
        if pid in closed:
            continue
        closed.add(pid)
        p = cg.portals[pid]
        if pid in goal_pids:
            best = min(best, d + point_seg_dist(cg.goal, p["a"], p["b"]))
        for cell in (p["u"], p["v"]):
            if cell not in alive:
                continue
            for qid in cg.cell_portals.get(cell, []):
                if qid == pid or qid in closed:
                    continue
                q = cg.portals[qid]
                if q["u"] not in alive or q["v"] not in alive:
                    continue
                nd = d + seg_seg_dist(p["a"], p["b"], q["a"], q["b"])
                if nd < dist.get(qid, INF) - 1e-15:
                    dist[qid] = nd
                    heapq.heappush(pq, (nd, qid))
    sg = float(np.linalg.norm(cg.goal - cg.start))
    return max(best, sg) if best < INF else sg
