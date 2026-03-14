"""
FIG Icosagrid: Fibonacci Icosagrid vertices and 20G centroids.

Port of the FIG grid construction from the 2022 Mathematica simulation
(FIG Dynamics Simulation - v6.wl, lines 143-290). Generates the full
3D Fibonacci Icosagrid by intersecting 10 Fibonacci-spaced grid planes,
and extracts the 20G centroid subset (points on ALL 10 grids simultaneously).
"""

import numpy as np
from math import sqrt
from itertools import combinations
from .fibonacci_chain import (FibonacciChain, PHI, phi,
                               PROJECTION_VECTOR, PROXY_SPACING, EMBEDDING_RATIO)

# --------------------------------------------------------------------------
# 10 FIG grid vectors (icosahedral directions from v6.wl lines 143-160)
# These are the 10 face normals of the icosahedron, normalized.
# --------------------------------------------------------------------------

_phi = PHI  # golden ratio

# The 10 grid vectors (icosahedral 3-fold normals, first hemisphere)
_raw_vectors = np.array([
    [0, 1, _phi],
    [0, 1, -_phi],
    [1, _phi, 0],
    [1, -_phi, 0],
    [-_phi, 0, 1],
    [_phi, 0, 1],
    [0, -1, _phi],
    [0, -1, -_phi],
    [-1, _phi, 0],
    [-1, -_phi, 0],
], dtype=np.float64)

# Normalize to unit vectors
GRID_VECTORS_10 = _raw_vectors / np.linalg.norm(_raw_vectors, axis=1, keepdims=True)

# 20 grid axes = 10 vectors + 10 negatives (1-indexed in Mathematica)
GRID_AXES_20 = np.vstack([GRID_VECTORS_10, -GRID_VECTORS_10])

# --------------------------------------------------------------------------
# 12 five-fold symmetry axes (icosahedral vertices, normalized)
# --------------------------------------------------------------------------

_fivefold_raw = np.array([
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [0, 1/_phi, _phi],
    [0, 1/_phi, -_phi],
    [1/_phi, _phi, 0],
    [1/_phi, -_phi, 0],
    [_phi, 0, 1/_phi],
    [_phi, 0, -1/_phi],
    [0, -1/_phi, _phi],
    [0, -1/_phi, -_phi],
], dtype=np.float64)

FIVEFOLD_AXES = _fivefold_raw / np.linalg.norm(_fivefold_raw, axis=1, keepdims=True)

# --------------------------------------------------------------------------
# 5 tetrahedral axis groups (v6.wl lines 185-192, 1-indexed into GRID_AXES_20)
# Converted to 0-indexed into GRID_AXES_20
# --------------------------------------------------------------------------

TET_GROUPS = [
    [0, 8, 12, 17],   # tetA: axes 1, 9, 13, 18 (1-indexed)
    [1, 2, 5, 19],    # tetB: axes 2, 3, 6, 20
    [7, 15, 4, 16],   # tetC: axes 8, 16, 5, 17
    [9, 14, 10, 13],  # tetD: axes 10, 15, 11, 14
    [6, 3, 11, 18],   # tetE: axes 7, 4, 12, 19
]

# --------------------------------------------------------------------------
# Precompute all C(10,3)=120 triples of grid vectors with nonzero determinant
# --------------------------------------------------------------------------

def _precompute_triples():
    """Find all valid triples of grid directions for 3-line intersection."""
    triples = []
    for combo in combinations(range(10), 3):
        M = GRID_VECTORS_10[list(combo)]
        det = np.linalg.det(M)
        if abs(det) > 1e-10:
            triples.append((combo, np.linalg.inv(M)))
    return triples

VALID_TRIPLES = _precompute_triples()


class FIGIcosagrid:
    """Fibonacci Icosagrid: 3D quasicrystal from 10-grid intersection."""

    def __init__(self):
        self.fib = FibonacciChain()

    def vertices(self, center, radius):
        """All FIG vertices within a sphere.

        For each of the 88 valid grid-vector triples:
          1. Project center onto each direction to get Fibonacci interval
          2. Form all tuples via meshgrid (vectorized)
          3. Batch-solve 3x3 system for 3D positions
          4. Filter by sphere distance

        Returns deduplicated array of shape (N, 3).
        """
        center = np.asarray(center, dtype=np.float64)
        points = []
        r_sq = (radius + 1e-8) ** 2

        for (i, j, k), inv_M in VALID_TRIPLES:
            dirs = GRID_VECTORS_10[[i, j, k]]
            projs = dirs @ center  # shape (3,)

            intervals = []
            for d in range(3):
                vals = self.fib.interval_vertices(projs[d] - radius,
                                                   projs[d] + radius)
                if len(vals) == 0:
                    break
                intervals.append(vals)
            else:
                # Vectorized: meshgrid all triples, batch matmul
                aa, bb, cc = np.meshgrid(intervals[0], intervals[1],
                                          intervals[2], indexing='ij')
                rhs = np.column_stack([aa.ravel(), bb.ravel(), cc.ravel()])
                pos = rhs @ inv_M.T  # (N, 3)
                diff = pos - center
                dist_sq = np.einsum('ij,ij->i', diff, diff)
                points.append(pos[dist_sq <= r_sq])

        if not points:
            return np.empty((0, 3))

        points = np.vstack(points)
        return _deduplicate(points)

    def centroids_20g(self, center, radius):
        """20G centroids: FIG vertices that project onto ALL 10 Fibonacci grids.

        These are the maximally-constrained vertices in the icosagrid.
        Uses phi^-3 rescaling to access the finer grid structure.
        """
        center = np.asarray(center, dtype=np.float64)

        scaled_center = center / PHI**3
        scaled_radius = radius / PHI**3

        candidates = self.vertices(scaled_center, scaled_radius)
        if len(candidates) == 0:
            return np.empty((0, 3))

        # Filter: projection onto ALL 10 grid directions must be on Fibonacci chain
        mask = np.ones(len(candidates), dtype=bool)
        for d in range(10):
            projs = candidates @ GRID_VECTORS_10[d]
            on_chain = self._batch_vertex_q(projs)
            mask &= on_chain

        filtered = candidates[mask]
        return filtered * PHI**3

    def _batch_vertex_q(self, values):
        """Vectorized check: are these values on the Fibonacci chain?

        For each value, compute the nearest ~5 Fibonacci vertex positions
        and check if any is within tolerance.
        """
        tol = self.fib.TOL
        # Approximate index for each value
        proxy_idx = values * EMBEDDING_RATIO / PROXY_SPACING if PROXY_SPACING != 0 \
            else np.zeros_like(values)
        base_n = np.round(proxy_idx).astype(int)

        result = np.zeros(len(values), dtype=bool)
        for dn in range(-2, 3):
            ns = base_n + dn
            # Compute vertex(n) for all ns at once
            # lattice_vertex: (round(n*phi), round(n*phi^2))
            lv0 = np.round(ns * phi).astype(int)
            lv1 = np.round(ns * phi**2).astype(int)
            verts = lv0 * PROJECTION_VECTOR[0] + lv1 * PROJECTION_VECTOR[1]
            result |= (np.abs(verts - values) < tol)

        return result

    def centroids_along_line(self, point_a, point_b):
        """20G centroids along a line segment from A to B.

        For each grid vector not perpendicular to the line direction:
          - Find Fibonacci values between projections of A and B
          - Compute intersection point on the line
        Keep points that appear in enough grid directions.
        """
        point_a = np.asarray(point_a, dtype=np.float64)
        point_b = np.asarray(point_b, dtype=np.float64)
        direction = point_b - point_a
        length = np.linalg.norm(direction)
        if length < 1e-12:
            return np.empty((0, 3))

        d_hat = direction / length
        points = []

        for gv_idx in range(10):
            gv = GRID_VECTORS_10[gv_idx]
            denom = np.dot(gv, d_hat)
            if abs(denom) < 1e-10:
                continue

            proj_a = np.dot(gv, point_a)
            proj_b = np.dot(gv, point_b)
            lo, hi = min(proj_a, proj_b), max(proj_a, proj_b)

            fib_vals = self.fib.interval_vertices(lo, hi)
            for fv in fib_vals:
                # t such that gv . (A + t*d_hat*length) = fv
                t = (fv - proj_a) / (denom * length)
                if -1e-8 <= t <= 1 + 1e-8:
                    pt = point_a + t * direction
                    points.append(pt)

        if not points:
            return np.empty((0, 3))

        points = np.array(points)
        # Keep points that are on ALL grid chains
        mask = np.ones(len(points), dtype=bool)
        for d in range(10):
            gv = GRID_VECTORS_10[d]
            projs = points @ gv
            for idx in range(len(points)):
                if mask[idx] and not self.fib.vertex_q(projs[idx]):
                    mask[idx] = False

        filtered = points[mask]
        if len(filtered) == 0:
            return np.empty((0, 3))
        return _deduplicate(filtered)

    def empire_rays(self, emperor_pos, radius):
        """Empire = union of 20G centroids along all 20 grid axis rays.

        Shoots rays from emperor_pos along all 20 grid directions,
        collects 20G centroids along each ray.
        """
        emperor_pos = np.asarray(emperor_pos, dtype=np.float64)
        all_centroids = []

        for axis_idx in range(20):
            direction = GRID_AXES_20[axis_idx]
            endpoint = emperor_pos + direction * radius
            cents = self.centroids_along_line(emperor_pos, endpoint)
            if len(cents) > 0:
                all_centroids.append(cents)

        if not all_centroids:
            return np.empty((0, 3))

        combined = np.vstack(all_centroids)
        return _deduplicate(combined)


def _deduplicate(points, decimals=6):
    """Remove duplicate points by rounding to `decimals` places and hashing."""
    if len(points) == 0:
        return points

    rounded = np.round(points, decimals)
    seen = {}
    keep = []
    for i in range(len(rounded)):
        key = (rounded[i, 0], rounded[i, 1], rounded[i, 2])
        if key not in seen:
            seen[key] = i
            keep.append(i)

    return points[keep]
