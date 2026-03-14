"""
Empire: segment-based empire computation.

The empire of a vertex v is the set of all SEGMENTS in the FIG that are
forced by v's position — segments that must exist in any valid quasicrystal
containing v.

A segment (i,j) is in the empire of v if BOTH vertices i and j are in
v's vertex empire (their perp-space displacements from v fall inside
the QC window).

This is stricter than vertex-only empire because a segment requires
both endpoints to be forced, not just one.
"""

import numpy as np
from scipy.spatial import cKDTree


class EmpireComputer:
    """Compute segment-based empires for all FIG vertices.

    The vertex empire uses the Minkowski intersection of the QC window
    (same as v1). The segment empire is the subset of segments whose
    both endpoints are in the vertex empire.

    Attributes:
        vertex_empire: dict mapping vertex → set of forced vertex indices
        segment_empire: dict mapping vertex → set of forced segment indices
        vertex_empire_sizes: array of vertex empire sizes
        segment_empire_sizes: array of segment empire sizes
    """

    def __init__(self, fig_builder, segment_set):
        self.fig = fig_builder
        self.segs = segment_set
        self._compute_vertex_empires()
        self._compute_segment_empires()

    def _compute_vertex_empires(self):
        """Vertex empire: Minkowski intersection of QC window."""
        self.vertex_empire = {}
        for i in range(self.fig.n_vertices):
            shifted = self.fig.pos_perp - self.fig.pos_perp[i]
            inside = self.fig.in_qc_window(shifted)
            inside[i] = False
            self.vertex_empire[i] = set(np.where(inside)[0])
        self.vertex_empire_sizes = np.array(
            [len(self.vertex_empire[i]) for i in range(self.fig.n_vertices)])

    def _compute_segment_empires(self):
        """Segment empire: segments whose BOTH endpoints are in vertex empire."""
        self.segment_empire = {}
        for v in range(self.fig.n_vertices):
            vemp = self.vertex_empire[v]
            forced_segs = set()
            for seg_idx, (i, j) in enumerate(self.segs.segments):
                if i in vemp and j in vemp:
                    forced_segs.add(seg_idx)
            # Also include segments directly connected to v
            # if their other endpoint is in the empire
            for seg_idx in self.segs.vertex_segments[v]:
                i, j = self.segs.segments[seg_idx]
                other = j if i == v else i
                if other in vemp:
                    forced_segs.add(seg_idx)
            self.segment_empire[v] = forced_segs
        self.segment_empire_sizes = np.array(
            [len(self.segment_empire[i]) for i in range(self.fig.n_vertices)])

    def dynamic_empire_intersection(self, v1, v2):
        """Compute the segment empire intersection between two vertices.

        This is the DYNAMIC computation that the old Mathematica code did
        at every step. It counts how many forced segments are shared.

        Returns:
            shared_segments: set of segment indices in both empires
            savings: len(shared_segments)
        """
        shared = self.segment_empire[v1] & self.segment_empire[v2]
        return shared, len(shared)

    def dynamic_savings_for_move(self, from_vertex, to_vertex, other_clock_vertex=None):
        """Compute segment-based savings for a candidate move.

        If other_clock_vertex is provided, also accounts for the interaction:
        segments in the other clock's empire are penalized/bonused.

        Returns:
            savings: number of surviving segments
            interaction_overlap: segments shared with other clock's empire
        """
        shared, savings = self.dynamic_empire_intersection(from_vertex, to_vertex)

        interaction_overlap = 0
        if other_clock_vertex is not None:
            other_empire = self.segment_empire[other_clock_vertex]
            interaction_overlap = len(shared & other_empire)

        return savings, interaction_overlap
