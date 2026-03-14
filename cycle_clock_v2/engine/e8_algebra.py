"""
E8 Algebra: roots, Hopf partition, Coxeter element, Cl(8) chirality.

Thin wrapper around Klee's CCT-StandardModel verification scripts
plus our Coxeter and coset computations.
"""

import numpy as np
from scipy.spatial import cKDTree
import sys, os

_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'CCT-StandardModel')
if _repo not in sys.path:
    sys.path.insert(0, _repo)
from e8_utils import build_e8_roots, hopf_map_quat, cluster_by_hopf


class E8Algebra:
    """The algebraic backbone: E8 roots, Hopf fibers, Coxeter circuit."""

    def __init__(self):
        self.roots = build_e8_roots()  # (240, 8)
        self._tree = cKDTree(self.roots)

        # Hopf partition: 10 × D4
        self.fibers = cluster_by_hopf(self.roots)
        self.root_fiber = np.zeros(240, dtype=int)
        for fi, sh in enumerate(self.fibers):
            for ri in sh:
                self.root_fiber[ri] = fi

        # Perpendicular pairs
        self.perp_pairs = []
        for i in range(10):
            for j in range(i + 1, 10):
                if np.allclose(self.roots[self.fibers[i]] @ self.roots[self.fibers[j]].T, 0):
                    self.perp_pairs.append((i, j))
        self.fiber_pair = np.zeros(10, dtype=int)
        for pi, (a, b) in enumerate(self.perp_pairs):
            self.fiber_pair[a] = pi
            self.fiber_pair[b] = pi

        # Simple roots and Coxeter element
        self._build_coxeter()

        # A8 coset grading
        self._build_cosets()

    def _build_coxeter(self):
        alpha = np.zeros((8, 8))
        alpha[0] = [1, -1, 0, 0, 0, 0, 0, 0]
        alpha[1] = [0, 1, -1, 0, 0, 0, 0, 0]
        alpha[2] = [0, 0, 1, -1, 0, 0, 0, 0]
        alpha[3] = [0, 0, 0, 1, -1, 0, 0, 0]
        alpha[4] = [0, 0, 0, 0, 1, -1, 0, 0]
        alpha[5] = [0, 0, 0, 0, 0, 1, -1, 0]
        alpha[6] = [0, 0, 0, 0, 0, 1, 1, 0]
        alpha[7] = [-0.5] * 8
        self.simple_roots = alpha

        self.coxeter_matrix = np.eye(8)
        for i in range(8):
            S = np.eye(8) - np.outer(alpha[i], alpha[i])
            self.coxeter_matrix = S @ self.coxeter_matrix

        # Precompute Coxeter permutation
        self.coxeter_perm = np.zeros(240, dtype=int)
        for i in range(240):
            v = self.coxeter_matrix @ self.roots[i]
            _, idx = self._tree.query(v)
            self.coxeter_perm[i] = idx

        # Build 8 orbits of 30
        visited = set()
        self.orbits = []
        for seed in range(240):
            if seed in visited:
                continue
            orbit = [seed]
            idx = seed
            for _ in range(29):
                idx = self.coxeter_perm[idx]
                orbit.append(idx)
            visited.update(orbit)
            self.orbits.append(orbit)

    def _build_cosets(self):
        cartan = np.array([
            [round(2 * (self.simple_roots[i] @ self.simple_roots[j]) /
                   (self.simple_roots[j] @ self.simple_roots[j]))
             for j in range(8)] for i in range(8)
        ])
        cartan_inv = np.linalg.inv(cartan.astype(float))

        self.root_coset = np.zeros(240, dtype=int)
        for i in range(240):
            n = np.array([self.roots[i] @ self.simple_roots[j] for j in range(8)])
            c = cartan_inv @ n
            self.root_coset[i] = round(c[5]) % 3

    def quantum_numbers(self, root_idx):
        f = int(self.root_fiber[root_idx])
        p = int(self.fiber_pair[f])
        c = int(self.root_coset[root_idx])
        pair_a, _ = self.perp_pairs[p]
        sign = +1 if f == pair_a else -1
        return {'fiber': f, 'pair': p, 'coset': c, 'chirality': sign}
