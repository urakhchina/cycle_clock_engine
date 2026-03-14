"""
Fibonacci chain: 1D quasicrystal on the Fibonacci lattice.

Port of the FibonacciChain object from the 2022 Mathematica simulation
(FIG Dynamics Simulation - v6.wl). The Fibonacci chain is the 1D analog
of the Fibonacci Icosagrid — vertices sit at positions n·φ projected
from the 2D integer lattice through the golden-ratio cut-and-project scheme.
"""

import numpy as np
from math import sqrt, floor, ceil


PHI = (1 + sqrt(5)) / 2
phi = PHI - 1  # 0.618...

# Projection from Z^2 onto the physical line
PROJECTION_VECTOR = np.array([
    sqrt((5 + sqrt(5)) / 10),
    sqrt(2 / (5 + sqrt(5)))
])

# Perpendicular-space slab half-width
SLAB_THICKNESS = sqrt(1 + 2 / sqrt(5))

# Spacing between proxy lattice sites
PROXY_SPACING = sqrt(5 - 2 * sqrt(5))

# Embedding ratio: physical-space projection scale
EMBEDDING_RATIO = sqrt(2 / (5 + sqrt(5)))


class FibonacciChain:
    """1D Fibonacci quasicrystal via cut-and-project from Z^2."""

    TOL = 1e-8

    def lattice_vertex(self, n):
        """Integer lattice point for index n: (round(n*phi), round(n*phi^2))."""
        return (round(n * phi), round(n * phi**2))

    def vertex(self, n):
        """Physical-space position of Fibonacci vertex n.
        Projects lattice_vertex(n) onto the physical line via PROJECTION_VECTOR."""
        lv = self.lattice_vertex(n)
        return lv[0] * PROJECTION_VECTOR[0] + lv[1] * PROJECTION_VECTOR[1]

    def proxy_index(self, x):
        """Continuous proxy index for physical-space coordinate x.
        Maps x to the approximate Fibonacci index via the embedding ratio."""
        return x * EMBEDDING_RATIO / PROXY_SPACING if PROXY_SPACING != 0 else 0

    def ceiling_index(self, x):
        """Smallest Fibonacci index n such that vertex(n) >= x."""
        # Start from proxy estimate, search upward
        n = ceil(self.proxy_index(x))
        # Adjust: search nearby to find the true ceiling
        while self.vertex(n) < x - self.TOL:
            n += 1
        # Back up if we overshot
        while n > 0 and self.vertex(n - 1) >= x - self.TOL:
            n -= 1
        return n

    def floor_index(self, x):
        """Largest Fibonacci index n such that vertex(n) <= x."""
        n = floor(self.proxy_index(x))
        # Adjust: search nearby to find the true floor
        while self.vertex(n) > x + self.TOL:
            n -= 1
        while self.vertex(n + 1) <= x + self.TOL:
            n += 1
        return n

    def interval_vertices(self, lo, hi):
        """All Fibonacci vertex positions in the interval [lo, hi].
        Returns sorted numpy array of physical-space coordinates."""
        n_lo = self.ceiling_index(lo)
        n_hi = self.floor_index(hi)
        if n_lo > n_hi:
            return np.array([])
        return np.array([self.vertex(n) for n in range(n_lo, n_hi + 1)])

    def interval_indices(self, lo, hi):
        """All Fibonacci indices n such that vertex(n) is in [lo, hi]."""
        n_lo = self.ceiling_index(lo)
        n_hi = self.floor_index(hi)
        if n_lo > n_hi:
            return []
        return list(range(n_lo, n_hi + 1))

    def vertex_q(self, x):
        """Is x (approximately) on the Fibonacci chain?
        Returns True if x is within TOL of some vertex(n)."""
        # Find nearest candidate
        n = round(self.proxy_index(x))
        for dn in range(-2, 3):
            if abs(self.vertex(n + dn) - x) < self.TOL:
                return True
        return False
