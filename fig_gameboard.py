"""
FIG Gameboard: The 3D Fibonacci Icosagrid as a game board for cycle clocks.

The FIG is the physical arena where cycle clocks walk. It is the 3D slice
of the 4D Elser-Sloane quasicrystal (ESQC), which is the cut-and-project
of the 8D E8 lattice.

Each FIG vertex carries:
  - 3D position (for visualization and spatial adjacency)
  - 4D ESQC coordinates (for perpendicular-space empire computation)
  - 8D E8 lattice coordinates (for algebraic structure)
  - Empire: the set of forced vertices (determined by perp-space proximity)
  - Savings to each neighbor: |Empire(i) ∩ Empire(j)|

The center emperor (perp radius = 0) has the largest empire.
Vertices near the QC window boundary have the smallest empires.
This gradient drives the least-change dynamics.

Chain: E8 → H4fold → ESQC (4D) → 3D slice → FIG (287 vertices)
"""

import numpy as np
from itertools import product as iprod
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist, squareform
import sys, os, json

_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CCT-StandardModel')
if _repo not in sys.path:
    sys.path.insert(0, _repo)
from e8_utils import build_e8_roots


class FIGGameboard:
    """The Fibonacci Icosagrid as a game board.

    Attributes:
        n_vertices: number of FIG vertices (~287)
        pos_3d: (N, 3) array of 3D positions
        pos_4d: (N, 4) array of 4D ESQC coordinates
        pos_perp: (N, 4) array of perpendicular-space coordinates
        pos_8d: (N, 8) array of 8D E8 lattice coordinates
        adjacency: dict mapping vertex index to list of neighbor indices
        empire: dict mapping vertex index to set of empire vertex indices
        savings_matrix: (N, N) sparse savings values (only for edges)
        perp_radius: (N,) array of perpendicular-space radii
    """

    # Default parameters
    SLAB_THICKNESS = 1.0      # 4D→3D slicing slab half-width
    MAX_NORM_SQ = 4           # E8 lattice generation bound

    def __init__(self):
        """Build the FIG gameboard with proper empire computation."""

        self._build_pipeline()
        self._build_adjacency()
        self._build_empires()
        self._build_savings()

    def _build_pipeline(self):
        """E8 → H4fold → ESQC → 3D slice → FIG."""
        # E8 roots (for projection matrices)
        E8 = build_e8_roots()
        Phi = (1 + np.sqrt(5)) / 2
        phi = Phi - 1
        phi_sq = phi ** 2

        # Moxness H4fold matrix
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

        # Perpendicular projections of E8 roots (for cut window)
        perp_roots = (Om @ E8.T).T

        # Generate E8 lattice points
        points = []
        r = 2
        coords = list(range(-r, r + 1))
        for v in iprod(coords, repeat=8):
            v = np.array(v, dtype=float)
            if np.sum(v ** 2) <= self.MAX_NORM_SQ and np.sum(v) % 2 == 0:
                points.append(v)
        half_coords = [x + 0.5 for x in range(-r, r)]
        for v in iprod(half_coords, repeat=8):
            v = np.array(v, dtype=float)
            if np.sum(v ** 2) <= self.MAX_NORM_SQ and round(np.sum(v)) % 2 == 0:
                points.append(v)
        lattice = np.array(points)

        # Project to parallel and perpendicular spaces
        par_all = (Pi @ lattice.T).T
        perp_all = (Om @ lattice.T).T

        # Cut window: convex hull of root perpendicular projections
        hull = ConvexHull(perp_roots)
        A = hull.equations[:, :-1]
        b = hull.equations[:, -1]
        in_window = np.all(A @ perp_all.T + b[:, None] <= 1e-10, axis=0)

        esqc_par = par_all[in_window]
        esqc_perp = perp_all[in_window]
        esqc_8d = lattice[in_window]

        # 3D slice
        eta = np.array([1, -1, 1, 1]) / 2
        eta = eta / np.linalg.norm(eta)

        basis = []
        for e in np.eye(4):
            v = e - np.dot(e, eta) * eta
            for bb in basis:
                v -= np.dot(v, bb) * bb
            n = np.linalg.norm(v)
            if n > 1e-10:
                basis.append(v / n)
            if len(basis) == 3:
                break
        self._basis_3d = np.array(basis)
        self._eta = eta

        heights = esqc_par @ eta
        in_slab = np.abs(heights) < self.SLAB_THICKNESS

        fig_3d = esqc_par[in_slab] @ self._basis_3d.T
        fig_4d = esqc_par[in_slab]
        fig_perp = esqc_perp[in_slab]
        fig_8d = esqc_8d[in_slab]

        # Deduplicate
        _, ui = np.unique(np.round(fig_3d, 8), axis=0, return_index=True)
        ui = np.sort(ui)

        self.pos_3d = fig_3d[ui]
        self.pos_4d = fig_4d[ui]
        self.pos_perp = fig_perp[ui]
        self.pos_8d = fig_8d[ui]
        self.n_vertices = len(self.pos_3d)

        # Perpendicular-space radius (determines empire size)
        self.perp_radius = np.linalg.norm(self.pos_perp, axis=1)

        # 3D distance matrix
        self._D3 = squareform(pdist(self.pos_3d))
        # Perp distance matrix
        self._Dp = squareform(pdist(self.pos_perp))

        # Spatial index for 3D lookups
        self._tree_3d = cKDTree(self.pos_3d)

        # Find the origin vertex (center emperor)
        self.origin_idx = int(np.argmin(self.perp_radius))

    def _build_adjacency(self):
        """Build adjacency using adaptive shell-aware thresholds.

        For each vertex, find its nearest neighbors by detecting the first
        significant gap in the sorted distance distribution. This respects
        the natural tetrahedral edge structure of the FIG rather than
        imposing a uniform distance cutoff.

        Guarantees at least 4 neighbors per vertex, stops at the first
        gap > 15% of the previous distance (or at max_neighbors=12).
        """
        max_neighbors = 12
        self.adjacency = {i: [] for i in range(self.n_vertices)}

        for i in range(self.n_vertices):
            di = self._D3[i].copy()
            di[i] = np.inf
            sorted_idx = np.argsort(di)
            sorted_d = di[sorted_idx]

            # Collect neighbors until a significant gap
            neighbors = [sorted_idx[0]]
            for k in range(1, min(max_neighbors, self.n_vertices - 1)):
                gap_ratio = (sorted_d[k] - sorted_d[k - 1]) / max(sorted_d[k - 1], 1e-10)
                if gap_ratio > 0.15 and k >= 4:
                    break
                neighbors.append(sorted_idx[k])

            for j in neighbors:
                if j not in self.adjacency[i]:
                    self.adjacency[i].append(j)
                if i not in self.adjacency[j]:
                    self.adjacency[j].append(i)

        self.degrees = np.array([len(self.adjacency[i]) for i in range(self.n_vertices)])

    def _build_empires(self):
        """Build empire using the convex hull Minkowski intersection method.

        The empire of vertex i is the set of all FIG vertices j such that
        the perpendicular-space displacement (p_j - p_i) falls inside the
        QC cut window. This is the mathematically correct empire definition
        from the Non-Local Game of Life paper.

        Geometrically: the empire window of vertex i is the intersection of
        the QC window W with W shifted to center on p_i. Vertices closer to
        the center of W (small perp radius) have larger empire windows and
        therefore larger empires.

        The center emperor (perp radius ≈ 0) has the full QC window as its
        empire window, forcing nearly every tile in the quasicrystal.
        """
        # QC window is already computed as the convex hull of E8 root perp projections.
        # Its half-plane inequalities are stored from the pipeline build.
        # Recompute them here from the stored perp data.
        E8 = build_e8_roots()
        Phi = (1 + np.sqrt(5)) / 2
        phi_val = Phi - 1
        phi_sq = phi_val ** 2
        H4fold = np.array([
            [Phi, 0, 0, 0, phi_sq, 0, 0, 0],
            [0, phi_val, 1, 0, 0, -phi_val, 1, 0],
            [0, 1, 0, phi_val, 0, 1, 0, -phi_val],
            [0, 0, phi_val, 1, 0, 0, -phi_val, 1],
            [phi_sq, 0, 0, 0, Phi, 0, 0, 0],
            [0, -phi_val, 1, 0, 0, phi_val, 1, 0],
            [0, 1, 0, -phi_val, 0, 1, 0, phi_val],
            [0, 0, -phi_val, 1, 0, 0, phi_val, 1],
        ])
        Om = H4fold[4:, :]
        perp_roots = (Om @ E8.T).T
        qc_hull = ConvexHull(perp_roots)
        self._qc_A = qc_hull.equations[:, :-1]
        self._qc_b = qc_hull.equations[:, -1]

        # For each vertex i, empire = {j : (p_j - p_i) is inside QC window}
        self.empire = {}
        for i in range(self.n_vertices):
            shifted = self.pos_perp - self.pos_perp[i]  # (N, 4)
            inside = np.all(self._qc_A @ shifted.T + self._qc_b[:, None] <= 1e-8, axis=0)
            inside[i] = False
            self.empire[i] = set(np.where(inside)[0])

        self.empire_sizes = np.array([len(self.empire[i]) for i in range(self.n_vertices)])

    def _build_savings(self):
        """Precompute savings for all adjacent pairs.

        savings[i][j] = |Empire(i) ∩ Empire(j)|
        Only computed for adjacent pairs (edges).
        """
        self.savings_cache = {}
        for i in range(self.n_vertices):
            for j in self.adjacency[i]:
                if (i, j) not in self.savings_cache:
                    s = len(self.empire[i] & self.empire[j])
                    self.savings_cache[(i, j)] = s
                    self.savings_cache[(j, i)] = s

    def savings(self, i, j):
        """Get precomputed savings between vertices i and j."""
        return self.savings_cache.get((i, j), 0)

    def neighbors_with_savings(self, vertex_idx):
        """Get neighbors sorted by savings (descending)."""
        result = []
        for j in self.adjacency[vertex_idx]:
            result.append((j, self.savings(vertex_idx, j)))
        return sorted(result, key=lambda x: -x[1])

    def step_probabilities(self, vertex_idx, exponent=28.0):
        """Compute savings-weighted transition probabilities.

        P(i → j) ∝ (savings(i,j) + 1)^exponent
        """
        neighbors = self.adjacency[vertex_idx]
        if not neighbors:
            return []

        savings = np.array([self.savings(vertex_idx, j) for j in neighbors], dtype=float)
        weights = (savings + 1) ** exponent
        probs = weights / weights.sum()
        return list(zip(neighbors, probs.tolist()))

    def vertex_info(self, idx):
        """Get complete information about a vertex."""
        return {
            'idx': idx,
            'pos_3d': self.pos_3d[idx].tolist(),
            'pos_8d': self.pos_8d[idx].tolist(),
            'perp_radius': float(self.perp_radius[idx]),
            'degree': int(self.degrees[idx]),
            'empire_size': int(self.empire_sizes[idx]),
            'is_origin': idx == self.origin_idx,
        }

    def export_for_threejs(self, filepath=None):
        """Export the complete gameboard state for Three.js visualization."""
        data = {
            'n_vertices': self.n_vertices,
            'positions_3d': self.pos_3d.tolist(),
            'perp_radii': self.perp_radius.tolist(),
            'empire_sizes': self.empire_sizes.tolist(),
            'degrees': self.degrees.tolist(),
            'origin_idx': self.origin_idx,
            'edges': [],
            'edge_savings': [],
        }

        # Edges with savings
        seen = set()
        for i in range(self.n_vertices):
            for j in self.adjacency[i]:
                if (i, j) not in seen:
                    data['edges'].append([i, j])
                    data['edge_savings'].append(self.savings(i, j))
                    seen.add((i, j))
                    seen.add((j, i))

        if filepath:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super().default(obj)
            with open(filepath, 'w') as f:
                json.dump(data, f, cls=NpEncoder)

        return data


# === Self-test ===
if __name__ == '__main__':
    print("Building FIG Gameboard...")
    board = FIGGameboard()

    print(f"\n=== FIG GAMEBOARD ===")
    print(f"  Vertices: {board.n_vertices}")
    print(f"  Edges: {len(board.savings_cache)//2}")
    print(f"  Degree: {board.degrees.min()}-{board.degrees.max()} "
          f"(mean {board.degrees.mean():.1f})")
    print(f"  Empire: {board.empire_sizes.min()}-{board.empire_sizes.max()} "
          f"(mean {board.empire_sizes.mean():.0f})")

    # Savings statistics
    all_savings = list(set(board.savings_cache.values()))
    print(f"  Savings: {min(all_savings)}-{max(all_savings)} "
          f"(mean {np.mean(list(board.savings_cache.values())):.1f})")

    # Center emperor
    o = board.origin_idx
    info = board.vertex_info(o)
    print(f"\n=== CENTER EMPEROR (vertex {o}) ===")
    print(f"  Perp radius: {info['perp_radius']:.4f}")
    print(f"  Degree: {info['degree']}")
    print(f"  Empire size: {info['empire_size']}")

    # Neighbors of origin, sorted by savings
    nws = board.neighbors_with_savings(o)
    print(f"\n  Top 5 neighbors by savings:")
    for j, s in nws[:5]:
        ji = board.vertex_info(j)
        print(f"    → v{j}: savings={s}, d3={board._D3[o,j]:.3f}, "
              f"perp_r={ji['perp_radius']:.3f}, empire={ji['empire_size']}")

    print(f"\n  Bottom 5 neighbors by savings:")
    for j, s in nws[-5:]:
        ji = board.vertex_info(j)
        print(f"    → v{j}: savings={s}, d3={board._D3[o,j]:.3f}, "
              f"perp_r={ji['perp_radius']:.3f}, empire={ji['empire_size']}")

    # Transition probabilities from origin
    probs = board.step_probabilities(o, exponent=10)
    probs_sorted = sorted(probs, key=lambda x: -x[1])
    print(f"\n  Transition probabilities (exponent=10):")
    print(f"    Top: v{probs_sorted[0][0]} P={probs_sorted[0][1]:.4f}")
    print(f"    Bottom: v{probs_sorted[-1][0]} P={probs_sorted[-1][1]:.6f}")
    print(f"    Ratio top/bottom: {probs_sorted[0][1]/probs_sorted[-1][1]:.0f}×")

    # Compare with exponent=28
    probs28 = board.step_probabilities(o, exponent=28)
    probs28_sorted = sorted(probs28, key=lambda x: -x[1])
    print(f"\n  With exponent=28:")
    print(f"    Top: v{probs28_sorted[0][0]} P={probs28_sorted[0][1]:.6f}")
    print(f"    Ratio top/bottom: {probs28_sorted[0][1]/max(probs28_sorted[-1][1],1e-30):.0f}×")

    # Export for Three.js
    board.export_for_threejs('fig_gameboard.json')
    print(f"\n  Exported to fig_gameboard.json")

    print("\nFIG Gameboard ready.")
