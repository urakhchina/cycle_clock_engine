"""
Segments: the fundamental geometric objects of the FIG.

A segment is an edge between two FIG vertices. The empire is defined
on segments, savings counts surviving segments, and the probability
distributions are segment-based.

This is the key upgrade from v1 (vertex-only) to v2 (segment-based),
matching the mathematical rigor of the original Mathematica code.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


class SegmentSet:
    """The complete set of segments (edges) in the FIG.

    Attributes:
        n_segments: total number of segments
        segments: list of (i, j) vertex index pairs, i < j
        segment_index: dict mapping (i, j) → segment index
        vertex_segments: dict mapping vertex → list of segment indices
        adjacency: dict mapping vertex → list of neighbor vertices
        degrees: array of vertex degrees
    """

    def __init__(self, fig_builder, max_neighbors=12, gap_threshold=0.15):
        """Build segments using adaptive shell-aware adjacency.

        For each vertex, finds neighbors by detecting the first
        significant gap in the sorted distance distribution.
        """
        self.n_vertices = fig_builder.n_vertices
        pos = fig_builder.pos_3d
        D = squareform(pdist(pos))

        # Adaptive adjacency: same algorithm as v1 but building segments
        self.segments = []
        self.segment_index = {}
        self.adjacency = {i: [] for i in range(self.n_vertices)}
        self.vertex_segments = {i: [] for i in range(self.n_vertices)}

        edges_seen = set()

        for i in range(self.n_vertices):
            di = D[i].copy()
            di[i] = np.inf
            sorted_idx = np.argsort(di)
            sorted_d = di[sorted_idx]

            neighbors = [sorted_idx[0]]
            for k in range(1, min(max_neighbors, self.n_vertices - 1)):
                gap_ratio = (sorted_d[k] - sorted_d[k - 1]) / max(sorted_d[k - 1], 1e-10)
                if gap_ratio > gap_threshold and k >= 4:
                    break
                neighbors.append(sorted_idx[k])

            for j in neighbors:
                edge = (min(i, j), max(i, j))
                if edge not in edges_seen:
                    edges_seen.add(edge)
                    seg_idx = len(self.segments)
                    self.segments.append(edge)
                    self.segment_index[edge] = seg_idx

                if j not in self.adjacency[i]:
                    self.adjacency[i].append(j)
                if i not in self.adjacency[j]:
                    self.adjacency[j].append(i)

        # Build vertex → segment mapping
        for seg_idx, (i, j) in enumerate(self.segments):
            self.vertex_segments[i].append(seg_idx)
            self.vertex_segments[j].append(seg_idx)

        self.n_segments = len(self.segments)
        self.degrees = np.array([len(self.adjacency[i]) for i in range(self.n_vertices)])

        # Store 3D midpoints and lengths for each segment
        self.midpoints = np.zeros((self.n_segments, 3))
        self.lengths = np.zeros(self.n_segments)
        for s, (i, j) in enumerate(self.segments):
            self.midpoints[s] = (pos[i] + pos[j]) / 2
            self.lengths[s] = D[i, j]

    def segments_of(self, vertex):
        """Get all segment indices connected to a vertex."""
        return self.vertex_segments[vertex]

    def segments_between(self, vertices):
        """Get all segments whose BOTH endpoints are in the vertex set."""
        vset = set(vertices)
        result = []
        for seg_idx, (i, j) in enumerate(self.segments):
            if i in vset and j in vset:
                result.append(seg_idx)
        return result

    def translate_segments(self, seg_indices, from_vertex, to_vertex, pos_3d):
        """Translate segments from one vertex position to another.

        Returns the translated segment midpoints (for spatial lookup).
        This mirrors the Mathematica VectorTranslate operation.
        """
        offset = pos_3d[to_vertex] - pos_3d[from_vertex]
        translated = self.midpoints[seg_indices] + offset
        return translated
