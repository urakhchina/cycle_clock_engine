"""
Helix builder: pentagonal helix construction and chirality operations.

Port of the pentagonal cylinder / wafer / helix algorithm from the 2022
Mathematica simulation (v6.wl lines 332-396). Constructs pentagonal helix
paths along the 12 five-fold symmetry axes of the icosahedron.

The helixes are constructed geometrically:
  - A regular pentagon perpendicular to the axis
  - Successive pentagons are shifted along the axis and rotated by a
    golden-angle offset, creating a helical path
  - Each helix vertex is the closest FIG vertex to the ideal position
"""

import numpy as np
from math import pi, sqrt
from .fig_icosagrid import FIGIcosagrid, FIVEFOLD_AXES, TET_GROUPS


PHI = (1 + sqrt(5)) / 2


def _rotation_matrix(axis, angle):
    """Rodrigues rotation matrix for rotation by `angle` about `axis`."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _perpendicular_basis(axis):
    """Return two unit vectors perpendicular to axis and each other."""
    axis = axis / np.linalg.norm(axis)
    # Pick a vector not parallel to axis
    if abs(axis[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    e1 = v - np.dot(v, axis) * axis
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


class HelixBuilder:
    """Builds pentagonal helix segments along five-fold axes."""

    def __init__(self, icosagrid=None):
        self.grid = icosagrid or FIGIcosagrid()

    def construct_helix(self, center, axis, n_wafers=12,
                        pent_radius=1.0, axial_step=None, handedness=1):
        """Construct an ideal pentagonal helix along an axis.

        Parameters:
            center: 3D center point
            axis: unit direction of the helix axis
            n_wafers: number of pentagonal layers
            pent_radius: radius of the pentagon (distance from axis)
            axial_step: step along axis per wafer (default: pent_radius * phi)
            handedness: +1 for left-handed, -1 for right-handed

        Returns array of shape (n_wafers, 3) — one vertex per wafer,
        tracing a helical path.
        """
        center = np.asarray(center, dtype=np.float64)
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

        if axial_step is None:
            axial_step = pent_radius * PHI * 0.5

        e1, e2 = _perpendicular_basis(axis)

        # Angular step: 72° (2π/5) per wafer + small golden-angle offset
        # for the helix. The offset creates the helical twist.
        angle_step = 2 * pi / 5  # 72° base rotation
        # Golden offset per wafer (creates the helix rather than a straight column)
        golden_offset = handedness * 2 * pi / (5 * PHI)

        helix = np.zeros((n_wafers, 3))
        for i in range(n_wafers):
            theta = i * (angle_step + golden_offset)
            axial_pos = (i - n_wafers // 2) * axial_step
            helix[i] = (center
                        + axial_pos * axis
                        + pent_radius * (np.cos(theta) * e1
                                         + np.sin(theta) * e2))

        return helix

    def snap_to_fig(self, helix, fig_vertices, max_snap_dist=None):
        """Snap each helix vertex to the nearest FIG vertex.

        Returns (snapped_helix, snap_distances).
        If max_snap_dist is given, vertices that can't snap are left at
        their ideal position.
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(fig_vertices)
        dists, indices = tree.query(helix)

        snapped = fig_vertices[indices].copy()
        if max_snap_dist is not None:
            too_far = dists > max_snap_dist
            snapped[too_far] = helix[too_far]

        return snapped, dists

    def build_helix_segments(self, center, axis, fig_vertices,
                              n_wafers=20, segment_length=6,
                              pent_radius=1.0, handedness=1):
        """Build helix segments for one axis direction.

        Constructs a long helix, optionally snaps to FIG vertices,
        then slices into overlapping segments of `segment_length` vertices.
        """
        helix = self.construct_helix(
            center, axis, n_wafers=n_wafers,
            pent_radius=pent_radius, handedness=handedness)

        # Snap to nearest FIG vertices if available
        if fig_vertices is not None and len(fig_vertices) > 0:
            helix, _ = self.snap_to_fig(helix, fig_vertices)

        # Slice into segments
        segments = []
        for start in range(0, len(helix) - segment_length + 1):
            segments.append(helix[start:start + segment_length].copy())

        return segments

    def build_axis_group_segments(self, center, axis_group_idx,
                                   fig_vertices=None,
                                   n_wafers=20, segment_length=6,
                                   pent_radius=1.0):
        """Build all helix segments for one tetrahedral axis group.

        For each of the 12 five-fold axes, builds helixes in both
        positive and negative axis directions.

        Returns list of segments, each an array of shape (segment_length, 3).
        """
        center = np.asarray(center, dtype=np.float64)
        segments = []

        for axis_idx in range(len(FIVEFOLD_AXES)):
            axis = FIVEFOLD_AXES[axis_idx]

            for direction in [1, -1]:
                segs = self.build_helix_segments(
                    center, direction * axis, fig_vertices,
                    n_wafers=n_wafers, segment_length=segment_length,
                    pent_radius=pent_radius, handedness=1)
                segments.extend(segs)

        return segments

    def chiral_reverse(self, segments):
        """Mirror helix segments to reverse handedness.

        For each segment:
          x = normalize(radial direction from origin)
          v = normalize(tangent along helix)
          y = normalize(v - (x·v)x)  (perpendicular component)
          z = cross(x, y)
          mirror = basis^-1 @ diag(1,1,-1) @ basis
        """
        mirrored = []
        for seg in segments:
            if len(seg) < 2:
                mirrored.append(seg.copy())
                continue

            midpoint = seg.mean(axis=0)
            x = midpoint / (np.linalg.norm(midpoint) + 1e-12)

            v = seg[-1] - seg[0]
            v = v / (np.linalg.norm(v) + 1e-12)

            y = v - np.dot(x, v) * x
            y_norm = np.linalg.norm(y)
            if y_norm < 1e-10:
                y = np.cross(x, [1, 0, 0])
                if np.linalg.norm(y) < 1e-10:
                    y = np.cross(x, [0, 1, 0])
                y = y / np.linalg.norm(y)
            else:
                y = y / y_norm

            z = np.cross(x, y)
            basis = np.column_stack([x, y, z])
            det = np.linalg.det(basis)
            if abs(det) < 1e-10:
                # Degenerate basis — just negate z-component directly
                mirrored.append(seg * np.array([1.0, 1.0, -1.0]))
                continue
            mirror = basis @ np.diag([1.0, 1.0, -1.0]) @ np.linalg.inv(basis)

            mirrored.append(seg @ mirror.T)

        return mirrored

    def segment_chain(self, segment, count):
        """Extend a segment by repeating its translation vector.

        displacement = segment[-1] - segment[0]
        Each new segment = previous + displacement.
        """
        segment = np.asarray(segment)
        displacement = segment[-1] - segment[0]
        chain = [segment]
        for i in range(1, count):
            chain.append(chain[-1] + displacement)
        return chain
