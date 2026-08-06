#!/usr/bin/env python3
"""E8 part 1+2: continuous physical validation and local path repair.

ContinuousValidator upgrades probe-based checking to a CERTIFIED sweep:
at configuration q with workspace clearance phi(q), every configuration
within the joint-space ball of radius phi(q)/L is collision-free, where
L bounds the workspace speed of any robot point per unit joint-space
step (conservative reach-sum bound; safety-certificate argument of
Bialkowski et al.).  Walking a segment with steps no larger than the
certified radius covers it continuously -- no resolution loophole.  If
the walk is forced below h_min without clearance, the segment counts as
FAILED (colliding or uncertifiable), with the offending interval
reported so the caller can repair it.

repair_path replaces a failed stretch by a via-point sampled inside the
segment's witness region (convexity keeps both new legs in the region),
validating candidates until one certifies.  This fixes the
q8/q13-style failure mode where lazy edge-blacklisting killed the only
corridor because of a small IrisZo collision pocket.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
import e6_sampling as e6

# conservative reach-sum Lipschitz bounds (m per rad of joint-norm step);
# derivation: |dp| <= sqrt(sum_i r_i^2) ||dq|| with r_i the distal reach
# of joint i, rounded up generously.  TODO(paper appendix): tight
# per-link derivation.
L_BOUNDS = {"panda": 4.0, "bimanual": 6.0}


class ContinuousValidator:
    def __init__(self, checker, L, h_min=2e-3, influence=0.30,
                 max_checks=30000):
        self.checker = checker
        self.L = L
        self.h_min = h_min
        self.influence = influence
        self.max_checks = max_checks
        self.n_calls = 0

    def _phi(self, q):
        self.n_calls += 1
        rc = self.checker.CalcRobotClearance(q, self.influence)
        d = np.asarray(rc.distances())
        if d.size == 0:
            return self.influence          # nothing within influence range
        return float(d.min())

    def segment(self, q1, q2):
        """(certified, bad_interval) for the straight segment q1->q2.
        bad_interval = (t0, t1) in [0,1] around the failure, else None."""
        seg = float(np.linalg.norm(q2 - q1))
        if seg < 1e-12:
            return self._phi(q1) > 0.0, None
        t = 0.0
        while t < 1.0:
            q = q1 + t * (q2 - q1)
            phi = self._phi(q)
            if phi <= 1e-9:
                return False, (max(0.0, t - self.h_min / seg),
                               min(1.0, t + 2 * self.h_min / seg))
            step = phi / self.L            # certified free radius (rad)
            if step < self.h_min:
                if not self.checker.CheckConfigCollisionFree(q):
                    return False, (max(0.0, t - self.h_min / seg),
                                   min(1.0, t + 2 * self.h_min / seg))
                # too thin to certify continuously: fail conservatively
                return False, (max(0.0, t - 5 * self.h_min / seg),
                               min(1.0, t + 5 * self.h_min / seg))
            if self.n_calls > self.max_checks:
                return False, (t, min(1.0, t + 0.1))
            # ball at q certifies [t, t + step/seg); overlap by 10%
            t += 0.9 * step / seg
        return (self._phi(q2) > 1e-9), None

    def path(self, path):
        """(ok, info) with per-segment failures for the repair layer."""
        self.n_calls = 0                   # per-path budget
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


def _region_sampler(regions, cache, rid, rng, n):
    """n interior samples of region rid (hull machinery reused)."""
    if rid not in cache:
        A, b = regions[rid]
        h = e6.reduce_to_affine_hull(np.asarray(A), np.asarray(b))
        cache[rid] = {"A": np.asarray(A), "b": np.asarray(b), "hull": h,
                      "dim": h["dim"], "u": rid, "v": rid,
                      "kind": "region"}
    return e6.sample_area(cache[rid], n, rng)


def repair_path(path, validator, regions, region_cache, rng,
                tries=15, max_rounds=4):
    """Replace failed stretches with in-region via points until the whole
    path certifies.  Returns (path, ok, n_vias)."""
    p = [np.asarray(x) for x in np.asarray(path)]
    n_vias = 0
    for _ in range(max_rounds):
        ok, info = validator.path(np.asarray(p))
        if ok:
            return np.asarray(p), True, n_vias
        k, itv = info["bad_intervals"][0]
        a, b = p[k], p[k + 1]
        common = e6.region_membership(a, regions) & \
            e6.region_membership(b, regions)
        if not common:
            return np.asarray(p), False, n_vias
        rid = min(common)
        t0, t1 = itv if itv else (0.3, 0.7)
        lo = a + max(0.0, t0 - 0.05) * (b - a)
        hi = a + min(1.0, t1 + 0.05) * (b - a)
        fixed = False
        for x in _region_sampler(regions, region_cache, rid, rng, tries):
            x = np.asarray(x)
            ok1, _ = validator.segment(lo, x)
            if not ok1:
                continue
            ok2, _ = validator.segment(x, hi)
            if not ok2:
                continue
            p = p[:k + 1] + [lo, x, hi] + p[k + 1:]
            n_vias += 1
            fixed = True
            break
        if not fixed:
            return np.asarray(p), False, n_vias
    ok, _ = validator.path(np.asarray(p))
    return np.asarray(p), ok, n_vias


def path_cost(path):
    p = np.asarray(path)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
