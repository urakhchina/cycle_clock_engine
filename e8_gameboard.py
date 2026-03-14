"""
E8 Gameboard: The lattice point set for the Cycle Clock game.

The gameboard is the E8 root system (240 vectors in R^8), equipped with:
  - Quaternionic Hopf partition (10 × D4 clusters)
  - Perpendicular pairs (5 bipartite halves)
  - Conformal selection (cuboctahedral projections to R^3)
  - A8 coset grading (L0, L1, L2 = three generations)
  - Coxeter element (order 30, the clock mechanism)
  - Empire computation (forced neighbors via inner-product structure)

Mathematical foundation:
  Irwin, "Kinematic Spinors on Division-Algebra Root Systems" (2026)
  Fang, Irwin, Hammock, Paduroiu, "Non-Local Game of Life in 2D Quasicrystals" (2018)
"""

import numpy as np
from scipy.spatial import cKDTree
import sys, os

# Import from Klee's verified repo
_repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CCT-StandardModel')
if _repo_path not in sys.path:
    sys.path.insert(0, _repo_path)
from e8_utils import build_e8_roots, hopf_map_quat, cluster_by_hopf


class E8Gameboard:
    """The E8 root lattice as a game board for cycle clocks.

    Attributes:
        roots: (240, 8) array of E8 root vectors
        shells: list of 10 lists, each containing 24 root indices (Hopf partition)
        perp_pairs: list of 5 tuples (i, j) of perpendicular cluster pairs
        adjacency: dict mapping root index to list of neighbor indices
        coxeter: (8, 8) matrix, the Coxeter element C = s1...s8
        root_fiber: array of length 240, fiber index for each root
        root_coset: array of length 240, A8 coset (0, 1, or 2) for each root
    """

    def __init__(self):
        # === Build E8 root system ===
        self.roots = build_e8_roots()  # (240, 8)
        self.n_roots = len(self.roots)
        self._tree = cKDTree(self.roots)

        # === Hopf partition: E8 = 10 × D4 ===
        self.shells = cluster_by_hopf(self.roots)

        # Root-to-fiber lookup
        self.root_fiber = np.zeros(self.n_roots, dtype=int)
        for fi, sh in enumerate(self.shells):
            for ri in sh:
                self.root_fiber[ri] = fi

        # === Perpendicular pairs ===
        self.perp_pairs = []
        for i in range(10):
            for j in range(i + 1, 10):
                cross = self.roots[self.shells[i]] @ self.roots[self.shells[j]].T
                if np.allclose(cross, 0, atol=1e-6):
                    self.perp_pairs.append((i, j))

        # Fiber-to-pair lookup
        self.fiber_pair = np.zeros(10, dtype=int)
        for pi, (a, b) in enumerate(self.perp_pairs):
            self.fiber_pair[a] = pi
            self.fiber_pair[b] = pi

        # === Simple roots and Coxeter element ===
        self._build_simple_roots()
        self._build_coxeter()

        # === A8 coset grading ===
        self._build_coset_grading()

        # === Adjacency (inner product structure) ===
        self._build_adjacency()

        # === Empire computation ===
        self._build_empires()

    def _build_simple_roots(self):
        """E8 simple roots (standard basis)."""
        self.simple_roots = np.zeros((8, 8))
        self.simple_roots[0] = [1, -1, 0, 0, 0, 0, 0, 0]
        self.simple_roots[1] = [0, 1, -1, 0, 0, 0, 0, 0]
        self.simple_roots[2] = [0, 0, 1, -1, 0, 0, 0, 0]
        self.simple_roots[3] = [0, 0, 0, 1, -1, 0, 0, 0]
        self.simple_roots[4] = [0, 0, 0, 0, 1, -1, 0, 0]
        self.simple_roots[5] = [0, 0, 0, 0, 0, 1, -1, 0]
        self.simple_roots[6] = [0, 0, 0, 0, 0, 1, 1, 0]
        self.simple_roots[7] = [-0.5] * 8

        # Cartan matrix
        self.cartan = np.array([
            [round(2 * (self.simple_roots[i] @ self.simple_roots[j]) /
                   (self.simple_roots[j] @ self.simple_roots[j]))
             for j in range(8)] for i in range(8)
        ])
        self._cartan_inv = np.linalg.inv(self.cartan.astype(float))

    def _build_coxeter(self):
        """Build the Coxeter element C = s1 s2 ... s8."""
        self.coxeter = np.eye(8)
        for i in range(8):
            S_i = np.eye(8) - np.outer(self.simple_roots[i], self.simple_roots[i])
            self.coxeter = S_i @ self.coxeter

        # Precompute: for each root, where does C send it?
        self.coxeter_perm = np.zeros(self.n_roots, dtype=int)
        for i in range(self.n_roots):
            v = self.coxeter @ self.roots[i]
            _, idx = self._tree.query(v)
            self.coxeter_perm[i] = idx

        # Precompute all 8 Coxeter orbits
        self.coxeter_orbits = []
        visited = set()
        for seed in range(self.n_roots):
            if seed in visited:
                continue
            orbit = [seed]
            idx = seed
            for _ in range(29):
                idx = self.coxeter_perm[idx]
                orbit.append(idx)
            visited.update(orbit)
            self.coxeter_orbits.append(orbit)

    def _build_coset_grading(self):
        """A8 coset grading: coefficient of α₅ (Coxeter label 3) mod 3."""
        self.root_coset = np.zeros(self.n_roots, dtype=int)
        for i in range(self.n_roots):
            n = np.array([self.roots[i] @ self.simple_roots[j] for j in range(8)])
            c = self._cartan_inv @ n
            self.root_coset[i] = round(c[5]) % 3

    def _build_adjacency(self):
        """Build adjacency from inner products.

        Two E8 roots are 'adjacent' if their inner product is ±1
        (the nearest non-antipodal neighbors in the root system).
        """
        self.adjacency = {i: [] for i in range(self.n_roots)}
        self.inner_products = {}

        for i in range(self.n_roots):
            for j in range(i + 1, self.n_roots):
                ip = round(float(self.roots[i] @ self.roots[j]))
                if ip == 1 or ip == -1:
                    self.adjacency[i].append(j)
                    self.adjacency[j].append(i)
                    self.inner_products[(i, j)] = ip
                    self.inner_products[(j, i)] = ip

    def _build_empires(self):
        """Compute the empire of each root.

        The empire of root r is the set of all roots whose existence is
        FORCED by r — roots that must appear in any valid quasicrystal
        containing r. In the E8 root system, the empire of r includes:

        1. All roots with inner product 1 with r (nearest neighbors)
        2. All roots forced by the cut-window constraint

        For the root system (not the full quasicrystal), the empire is
        defined by the perpendicular-space neighborhood: roots whose
        perp projections are close to r's perp projection.

        Simplified version: empire = all roots reachable within a certain
        inner-product distance, weighted by the savings metric.
        """
        # For the root system, the "empire" of root i is:
        # - Its neighbors (IP = ±1): local empire
        # - Extended empire: neighbors of neighbors, etc.
        # The savings between two roots = size of empire intersection.

        self.empire = {}
        for i in range(self.n_roots):
            # Level 1: direct neighbors (IP = ±1)
            level1 = set(self.adjacency[i])
            # Level 2: neighbors of neighbors
            level2 = set()
            for j in level1:
                level2.update(self.adjacency[j])
            level2 -= {i}
            level2 -= level1

            self.empire[i] = {
                'local': level1,           # forced neighbors
                'extended': level2,         # second-shell forced
                'full': level1 | level2,    # complete empire
            }

    def savings(self, root_i, root_j):
        """Compute the savings between two roots.

        Savings = |Empire(i) ∩ Empire(j)| = number of empire tiles
        that are preserved when moving from i to j.

        High savings = small change = preferred by least-change principle.
        """
        return len(self.empire[root_i]['full'] & self.empire[root_j]['full'])

    def neighbors_with_savings(self, root_idx):
        """Get all neighbors of a root with their savings values.

        Returns list of (neighbor_idx, savings_value) sorted by savings (descending).
        """
        result = []
        for j in self.adjacency[root_idx]:
            s = self.savings(root_idx, j)
            result.append((j, s))
        return sorted(result, key=lambda x: -x[1])

    def step_probabilities(self, root_idx, exponent=28):
        """Compute transition probabilities from a root to its neighbors.

        Uses the savings-weighted probability from the FIG Dynamics code:
          P(i → j) ∝ (savings(i,j) + 1)^exponent

        Higher exponent = more deterministic (strongly prefer high savings).

        Returns list of (neighbor_idx, probability).
        """
        neighbors = self.neighbors_with_savings(root_idx)
        if not neighbors:
            return []

        weights = np.array([(s + 1) ** exponent for _, s in neighbors], dtype=float)
        probs = weights / weights.sum()

        return [(idx, float(p)) for (idx, _), p in zip(neighbors, probs)]

    def root_quantum_numbers(self, root_idx):
        """Get the three quantum numbers of a root.

        Returns dict with:
          fiber: Hopf cluster index (0-9)
          pair: perpendicular pair index (0-4)
          coset: A8 coset label (0, 1, 2)
          chirality_sign: which half of the perp pair (+1 or -1)
        """
        f = self.root_fiber[root_idx]
        p = self.fiber_pair[f]
        c = self.root_coset[root_idx]

        # Chirality sign: by convention, the lower-indexed fiber in a pair is +
        pair_a, pair_b = self.perp_pairs[p]
        sign = +1 if f == pair_a else -1

        return {
            'fiber': int(f),
            'pair': int(p),
            'coset': int(c),
            'chirality_sign': int(sign),
        }

    def conformal_project_fiber(self, fiber_idx, normal_idx=0):
        """Conformally project a D4 fiber to 3D, producing a cuboctahedron.

        Args:
            fiber_idx: which Hopf cluster (0-9)
            normal_idx: which conformal normal (0-3); 0 and 3 give cuboctahedra

        Returns:
            (selected_root_indices, coords_3d) where coords_3d is (12, 3)
        """
        d4_roots = self.roots[self.shells[fiber_idx]]
        _, _, Vt = np.linalg.svd(d4_roots, full_matrices=False)
        basis_4d = Vt[:4]
        d4_4d = d4_roots @ basis_4d.T

        n = np.zeros(4)
        n[normal_idx] = 1.0
        dots = d4_4d @ n
        selected_local = np.where(np.abs(dots) < 1e-6)[0]

        proj_axes = [k for k in range(4) if k != normal_idx]
        coords_3d = d4_4d[selected_local][:, proj_axes]
        selected_global = [self.shells[fiber_idx][i] for i in selected_local]

        return selected_global, coords_3d

    def export_state(self):
        """Export the gameboard state as a JSON-serializable dict.

        For use by Three.js visualization.
        """
        return {
            'n_roots': self.n_roots,
            'roots_8d': self.roots.tolist(),
            'fibers': [list(sh) for sh in self.shells],
            'perp_pairs': self.perp_pairs,
            'root_fiber': self.root_fiber.tolist(),
            'root_coset': self.root_coset.tolist(),
            'adjacency': {str(k): v for k, v in self.adjacency.items()},
            'coxeter_orbits': self.coxeter_orbits,
            'coxeter_perm': self.coxeter_perm.tolist(),
        }


# === Quick self-test ===
if __name__ == '__main__':
    print("Building E8 Gameboard...")
    board = E8Gameboard()

    print(f"  Roots: {board.n_roots}")
    print(f"  Hopf fibers: {len(board.shells)} × {len(board.shells[0])}")
    print(f"  Perp pairs: {board.perp_pairs}")
    print(f"  Coxeter orbits: {len(board.coxeter_orbits)} × {len(board.coxeter_orbits[0])}")

    # Coset check
    counts = [np.sum(board.root_coset == i) for i in range(3)]
    print(f"  A8 cosets: L0={counts[0]}, L1={counts[1]}, L2={counts[2]}")

    # Adjacency check
    n_edges = sum(len(v) for v in board.adjacency.values()) // 2
    avg_degree = np.mean([len(v) for v in board.adjacency.values()])
    print(f"  Edges: {n_edges}, avg degree: {avg_degree:.1f}")

    # Empire check
    emp = board.empire[0]
    print(f"  Root 0 empire: {len(emp['local'])} local, {len(emp['extended'])} extended")

    # Savings check
    nws = board.neighbors_with_savings(0)
    print(f"  Root 0 neighbors: {len(nws)}, savings range: {nws[-1][1]}-{nws[0][1]}")

    # Quantum numbers
    qn = board.root_quantum_numbers(0)
    print(f"  Root 0: fiber={qn['fiber']}, pair={qn['pair']}, "
          f"coset={qn['coset']}, chirality={'+' if qn['chirality_sign']>0 else '-'}")

    # Transition probabilities
    probs = board.step_probabilities(0, exponent=28)
    print(f"  Root 0 top transition: root {probs[0][0]} with P={probs[0][1]:.4f}")

    print("\nE8 Gameboard ready.")
