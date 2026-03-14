"""
FIG Builder: E8 → H4fold → ESQC → 3D slice → FIG.

Produces the point set AND the segment (edge) set.
Each segment is an ordered pair of vertex indices with its 3D geometry.
"""

import numpy as np
from itertools import product as iprod
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist, squareform

import sys, os
_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'CCT-StandardModel')
if _repo not in sys.path:
    sys.path.insert(0, _repo)
from e8_utils import build_e8_roots


class FIGBuilder:
    """Build the Fibonacci Icosagrid with full coordinate data."""

    def __init__(self, max_norm_sq=4, slab_thickness=1.0):
        self.max_norm_sq = max_norm_sq
        self.slab_thickness = slab_thickness
        self._build()

    def _build(self):
        E8 = build_e8_roots()
        Phi = (1 + np.sqrt(5)) / 2
        phi = Phi - 1
        phi_sq = phi ** 2

        H4fold = np.array([
            [Phi, 0, 0, 0, phi_sq, 0, 0, 0],
            [0, phi, 1, 0, 0, -phi, 1, 0],
            [0, 1, 0, phi, 0, 1, 0, -phi],
            [0, 0, phi, 1, 0, 0, -phi, 1],
            [phi_sq, 0, 0, 0, Phi, 0, 0, 0],
            [0, -phi, 1, 0, 0, phi, 1, 0],
            [0, 1, 0, -phi, 0, 1, 0, phi],
            [0, 0, -phi, 1, 0, 0, phi, 1],
        ])
        Pi = H4fold[:4, :]
        Om = H4fold[4:, :]

        perp_roots = (Om @ E8.T).T

        # Generate E8 lattice
        points = []
        r = 2
        for v in iprod(range(-r, r + 1), repeat=8):
            v = np.array(v, dtype=float)
            if np.sum(v ** 2) <= self.max_norm_sq and np.sum(v) % 2 == 0:
                points.append(v)
        for v in iprod([x + 0.5 for x in range(-r, r)], repeat=8):
            v = np.array(v, dtype=float)
            if np.sum(v ** 2) <= self.max_norm_sq and round(np.sum(v)) % 2 == 0:
                points.append(v)
        lattice = np.array(points)

        par_all = (Pi @ lattice.T).T
        perp_all = (Om @ lattice.T).T

        # Cut window
        hull = ConvexHull(perp_roots)
        self._qc_A = hull.equations[:, :-1]
        self._qc_b = hull.equations[:, -1]
        in_window = np.all(self._qc_A @ perp_all.T + self._qc_b[:, None] <= 1e-10, axis=0)

        esqc_par = par_all[in_window]
        esqc_perp = perp_all[in_window]
        esqc_8d = lattice[in_window]

        # 3D slice
        eta = np.array([1, -1, 1, 1]) / 2
        eta = eta / np.linalg.norm(eta)
        basis = []
        for e in np.eye(4):
            v = e - np.dot(e, eta) * eta
            for b in basis:
                v -= np.dot(v, b) * b
            n = np.linalg.norm(v)
            if n > 1e-10:
                basis.append(v / n)
            if len(basis) == 3:
                break
        self._basis_3d = np.array(basis)

        heights = esqc_par @ eta
        in_slab = np.abs(heights) < self.slab_thickness
        fig_3d = esqc_par[in_slab] @ self._basis_3d.T
        fig_perp = esqc_perp[in_slab]
        fig_8d = esqc_8d[in_slab]

        # Deduplicate
        _, ui = np.unique(np.round(fig_3d, 8), axis=0, return_index=True)
        ui = np.sort(ui)

        self.pos_3d = fig_3d[ui]
        self.pos_perp = fig_perp[ui]
        self.pos_8d = fig_8d[ui]
        self.n_vertices = len(self.pos_3d)
        self.perp_radius = np.linalg.norm(self.pos_perp, axis=1)
        self.origin_idx = int(np.argmin(self.perp_radius))

    def in_qc_window(self, points_4d):
        """Test if 4D points are inside the QC cut window."""
        return np.all(self._qc_A @ points_4d.T + self._qc_b[:, None] <= 1e-8, axis=0)
