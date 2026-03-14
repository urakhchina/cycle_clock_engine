#!/usr/bin/env python3
"""
Export E8 → FIG → 20G data to JSON for Three.js scrollytelling visualization.

Runs the full computation pipeline from my_E8_FIG_projection.ipynb headlessly
and writes e8_fig_data.json with all geometry needed for the 9-section page.
"""

import numpy as np
import json
import sys
from itertools import combinations, product as iprod
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict

Phi = (1 + np.sqrt(5)) / 2       # golden ratio ~1.618
phi = Phi - 1                      # 1/Phi ~0.618
phi_sq = phi**2                    # ~0.382


def main():
    print("=" * 60)
    print("E8 → FIG → 20G  Data Export")
    print("=" * 60)

    # =================================================================
    # Cell 1: E8 Root System (240 vectors in 8D)
    # =================================================================
    print("\n[1/9] Computing E8 root system...")

    # Type A (112): permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    type_A = []
    for i, j in combinations(range(8), 2):
        for si in [+1, -1]:
            for sj in [+1, -1]:
                v = np.zeros(8)
                v[i] = si
                v[j] = sj
                type_A.append(v)
    type_A = np.array(type_A)

    # Type B (128): all (±1/2)^8 with even number of minus signs
    type_B = []
    for bits in range(256):
        signs = np.array([(bits >> k) & 1 for k in range(8)])
        num_minus = np.sum(signs)
        if num_minus % 2 == 0:
            v = np.where(signs, -0.5, +0.5)
            type_B.append(v)
    type_B = np.array(type_B)

    e8_roots = np.vstack([type_A, type_B])
    norms_sq = np.sum(e8_roots**2, axis=1)
    print(f"  Type A: {len(type_A)}, Type B: {len(type_B)}, Total: {len(e8_roots)}")
    print(f"  All squared norms = 2? {np.allclose(norms_sq, 2.0)}")

    # =================================================================
    # Cell 2: Moxness H4fold Projection Matrix
    # =================================================================
    print("\n[2/9] Building projection matrices...")

    H4fold = np.array([
        [ Phi,    0,     0,     0,    phi_sq,  0,     0,     0   ],
        [  0,    phi,    1,     0,     0,    -phi,    1,     0   ],
        [  0,     1,     0,    phi,    0,      1,     0,   -phi  ],
        [  0,     0,    phi,    1,     0,      0,   -phi,    1   ],
        [phi_sq,  0,     0,     0,    Phi,     0,     0,     0   ],
        [  0,   -phi,    1,     0,     0,     phi,    1,     0   ],
        [  0,     1,     0,   -phi,    0,      1,     0,    phi  ],
        [  0,     0,   -phi,    1,     0,      0,    phi,    1   ]
    ])

    Pi_proj = H4fold[:4, :]     # parallel space (4×8)
    Omega_proj = H4fold[4:, :]  # perpendicular space (4×8)

    parallel_4d = (Pi_proj @ e8_roots.T).T       # (240, 4)
    perp_4d     = (Omega_proj @ e8_roots.T).T    # (240, 4)

    par_norms = np.linalg.norm(parallel_4d, axis=1)
    mask_inner = par_norms < np.mean(par_norms)   # 120 inner
    mask_outer = ~mask_inner                       # 120 outer

    print(f"  Inner: {mask_inner.sum()}, Outer: {mask_outer.sum()}")

    # =================================================================
    # Cell 3: Cut Window + ESQC
    # =================================================================
    print("\n[3/9] Generating E8 lattice and computing ESQC...")
    print("  (This may take a few minutes for the lattice enumeration)")

    def generate_e8_lattice(max_norm_sq=4, coord_range=2):
        points = []
        r = coord_range
        coords = list(range(-r, r + 1))
        for v in iprod(coords, repeat=8):
            v = np.array(v, dtype=float)
            if np.sum(v**2) <= max_norm_sq and np.sum(v) % 2 == 0:
                points.append(v)
        half_coords = [x + 0.5 for x in range(-r, r)]
        for v in iprod(half_coords, repeat=8):
            v = np.array(v, dtype=float)
            if np.sum(v**2) <= max_norm_sq and round(np.sum(v)) % 2 == 0:
                points.append(v)
        return np.array(points)

    e8_lattice = generate_e8_lattice(max_norm_sq=4, coord_range=2)
    print(f"  E8 lattice points: {len(e8_lattice)}")

    parallel_all = (Pi_proj @ e8_lattice.T).T
    perp_all     = (Omega_proj @ e8_lattice.T).T

    # Cut window = convex hull of root perp projections
    root_perp_hull = ConvexHull(perp_4d)

    def points_in_hull(points, hull):
        A = hull.equations[:, :-1]
        b = hull.equations[:, -1]
        return np.all(A @ points.T + b[:, None] <= 1e-10, axis=0)

    in_window = points_in_hull(perp_all, root_perp_hull)
    esqc_4d = parallel_all[in_window]
    print(f"  Inside cut window: {in_window.sum()} / {len(e8_lattice)}")

    # =================================================================
    # Cell 4: 3D Slice of 4D ESQC → FIG
    # =================================================================
    print("\n[4/9] Slicing ESQC to get FIG...")

    eta = np.array([1, -1, 1, 1]) / 2.0
    eta = eta / np.linalg.norm(eta)

    def orthonormal_complement(normal, dim=4):
        basis = []
        candidates = np.eye(dim)
        for e in candidates:
            v = e.copy()
            v -= np.dot(v, normal) * normal
            for b in basis:
                v -= np.dot(v, b) * b
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                basis.append(v / norm)
            if len(basis) == dim - 1:
                break
        return np.array(basis)

    basis_3d = orthonormal_complement(eta, 4)
    heights = esqc_4d @ eta
    slab_thickness = 1.0
    in_slab = np.abs(heights) < slab_thickness
    fig_3d = (esqc_4d[in_slab]) @ basis_3d.T
    print(f"  FIG 3D points: {len(fig_3d)}")

    # =================================================================
    # Cell 5: FIG + 20G Core
    # =================================================================
    print("\n[5/9] Identifying 20G core...")

    dists_c5 = np.linalg.norm(fig_3d, axis=1)
    dist_sorted_c5 = np.sort(np.unique(np.round(dists_c5, 4)))

    shell_info = []
    for d in dist_sorted_c5[:10]:
        count = int(np.sum(np.abs(dists_c5 - d) < 0.01))
        shell_info.append({"r": round(float(d), 4), "count": count})

    if dist_sorted_c5[0] < 0.01:
        core_radius_c5 = dist_sorted_c5[1] if len(dist_sorted_c5) > 1 else 0
    else:
        core_radius_c5 = dist_sorted_c5[0]

    core_mask_c5 = np.abs(dists_c5 - core_radius_c5) < 0.05
    outer_mask_c5 = ~core_mask_c5 & (dists_c5 > 0.01)
    origin_mask_c5 = dists_c5 < 0.01

    print(f"  Core vertices (r ~ {core_radius_c5:.4f}): {core_mask_c5.sum()}")
    print(f"  Outer FIG: {outer_mask_c5.sum()}, Origin: {origin_mask_c5.sum()}")

    # =================================================================
    # Cell 6: C5C — 5 Cuboctahedra (with chirality)
    # =================================================================
    print("\n[6/9] Finding C5C cuboctahedra...")

    # Deduplicate FIG
    fig_rounded = np.round(fig_3d, 8)
    _, unique_idx = np.unique(fig_rounded, axis=0, return_index=True)
    fig_unique = fig_3d[np.sort(unique_idx)]
    print(f"  FIG unique points: {len(fig_unique)}")

    dists_from_origin = np.linalg.norm(fig_unique, axis=1)
    sorted_idx = np.argsort(dists_from_origin)

    n_search = min(152, len(fig_unique) - 1)
    core_idx = sorted_idx[1:n_search + 1]
    core_pts = fig_unique[core_idx]
    core_radii = dists_from_origin[core_idx]
    D_core = squareform(pdist(core_pts))

    # Find ALL regular triangles
    regularity_tol = 0.10
    all_regular_triangles = []
    for i in range(len(core_pts)):
        ri = core_radii[i]
        for j in range(i + 1, len(core_pts)):
            rj = core_radii[j]
            dij = D_core[i, j]
            m3 = (ri + rj + dij) / 3
            if max(abs(ri - m3), abs(rj - m3), abs(dij - m3)) / m3 > regularity_tol:
                continue
            for k in range(j + 1, len(core_pts)):
                rk = core_radii[k]
                dik = D_core[i, k]
                djk = D_core[j, k]
                all6 = [ri, rj, rk, dij, dik, djk]
                m6 = np.mean(all6)
                max_dev = max(abs(v - m6) / m6 for v in all6)
                if max_dev < regularity_tol:
                    all_regular_triangles.append((i, j, k, m6, max_dev))

    print(f"  Regular triangles found: {len(all_regular_triangles)}")

    # Bucket by edge length, find 4-Groups via backtracking
    buckets = defaultdict(list)
    for idx, tri in enumerate(all_regular_triangles):
        edge_len = np.round(tri[3], 2)
        buckets[edge_len].append(idx)

    centroid_tol = 0.05
    all_4groups = []
    all_4group_vsets = []

    for edge_len in sorted(buckets.keys()):
        tri_indices = buckets[edge_len]
        tris = [all_regular_triangles[i] for i in tri_indices]
        n = len(tris)
        if n < 4:
            continue

        centroids = [np.mean(core_pts[list(t[:3])], axis=0) for t in tris]
        bucket_results = []

        def bt_search(start, chosen, used_v, c_sum):
            if len(chosen) == 4:
                if np.linalg.norm(c_sum) < centroid_tol:
                    bucket_results.append(tuple(chosen))
                return
            rem = 4 - len(chosen)
            for ii in range(start, n - rem + 1):
                tv = {tris[ii][0], tris[ii][1], tris[ii][2]}
                if tv.isdisjoint(used_v):
                    bt_search(ii + 1, chosen + [ii], used_v | tv,
                              c_sum + centroids[ii])

        bt_search(0, [], set(), np.zeros(3))

        for local_combo in bucket_results:
            global_combo = tuple(tri_indices[lc] for lc in local_combo)
            verts = set()
            for lc in local_combo:
                verts.update(tris[lc][:3])
            all_4groups.append(global_combo)
            all_4group_vsets.append(frozenset(verts))

    print(f"  Total 4-Groups: {len(all_4groups)}")

    # Group by vertex set, keep real cuboctahedra (2+ chiralities)
    vset_to_groups = defaultdict(list)
    for gi, vset in enumerate(all_4group_vsets):
        vset_to_groups[vset].append(gi)

    real_cubos = []
    for vset, group_indices in vset_to_groups.items():
        if len(group_indices) >= 2:
            distinct = []
            seen = set()
            for gi in group_indices:
                key = all_4groups[gi]
                if key not in seen:
                    distinct.append(gi)
                    seen.add(key)
            if len(distinct) >= 2:
                real_cubos.append((vset, distinct))

    print(f"  Real cuboctahedra: {len(real_cubos)}")

    # Find 5 mutually vertex-disjoint cuboctahedra
    def find_5_disjoint(cubos, current, used_verts, start_idx):
        if len(current) == 5:
            return current
        for i in range(start_idx, len(cubos)):
            if cubos[i][0].isdisjoint(used_verts):
                res = find_5_disjoint(cubos, current + [i],
                                       used_verts | cubos[i][0], i + 1)
                if res:
                    return res
        return None

    best_5 = find_5_disjoint(real_cubos, [], set(), 0)
    print(f"  5 disjoint cubos found: {best_5 is not None}")

    def signed_vol(v0, v1, v2):
        return np.dot(v0, np.cross(v1, v2)) / 6.0

    cubo_colors = ['red', 'blue', 'green', 'orange', 'purple']
    cuboctahedra = []
    left_20g_triangles = []
    right_20g_triangles = []
    used_vertices = set()

    if best_5 is not None:
        for ci, cubo_idx in enumerate(best_5):
            vset, group_indices = real_cubos[cubo_idx]
            used_vertices.update(vset)

            group_A = all_4groups[group_indices[0]]
            group_B = all_4groups[group_indices[1]]

            set_a_tris = [all_regular_triangles[i] for i in group_A]
            set_b_tris = [all_regular_triangles[i] for i in group_B]

            sv_a = np.mean([signed_vol(core_pts[t[0]], core_pts[t[1]],
                                        core_pts[t[2]]) for t in set_a_tris])
            sv_b = np.mean([signed_vol(core_pts[t[0]], core_pts[t[1]],
                                        core_pts[t[2]]) for t in set_b_tris])

            if sv_a < sv_b:
                set_a_tris, set_b_tris = set_b_tris, set_a_tris

            cubo_verts_list = sorted(list(vset))
            cuboctahedra.append((cubo_verts_list, set_a_tris, set_b_tris))
            left_20g_triangles.extend(set_a_tris)
            right_20g_triangles.extend(set_b_tris)

        print(f"  Left-Twisted 20G: {len(left_20g_triangles)} tetrahedra")
        print(f"  Right-Twisted 20G: {len(right_20g_triangles)} tetrahedra")
    else:
        print("  WARNING: Could not find 5 disjoint cuboctahedra!")

    # =================================================================
    # Cell 7: Chirality Analysis
    # =================================================================
    print("\n[7/9] Analyzing chirality...")

    left_verts = set()
    for tri in left_20g_triangles:
        left_verts.update([tri[0], tri[1], tri[2]])
    right_verts = set()
    for tri in right_20g_triangles:
        right_verts.update([tri[0], tri[1], tri[2]])

    def get_edges(tri_list):
        edges = set()
        for tri in tri_list:
            edges.add((min(tri[0], tri[1]), max(tri[0], tri[1])))
            edges.add((min(tri[0], tri[2]), max(tri[0], tri[2])))
            edges.add((min(tri[1], tri[2]), max(tri[1], tri[2])))
        return edges

    left_edges = get_edges(left_20g_triangles)
    right_edges = get_edges(right_20g_triangles)
    shared_edges = left_edges & right_edges

    left_signed_volumes = []
    for tri in left_20g_triangles:
        sv = signed_vol(core_pts[tri[0]], core_pts[tri[1]], core_pts[tri[2]])
        left_signed_volumes.append(float(sv))

    right_signed_volumes = []
    for tri in right_20g_triangles:
        sv = signed_vol(core_pts[tri[0]], core_pts[tri[1]], core_pts[tri[2]])
        right_signed_volumes.append(float(sv))

    # Bar colors by parent cuboctahedron
    bar_colors_left = []
    bar_colors_right = []
    for ci, (verts, set_a, set_b) in enumerate(cuboctahedra):
        for _ in set_a:
            bar_colors_left.append(cubo_colors[ci % 5])
        for _ in set_b:
            bar_colors_right.append(cubo_colors[ci % 5])

    print(f"  Shared verts: {len(left_verts & right_verts)}")
    print(f"  Left edges: {len(left_edges)}, Right edges: {len(right_edges)}")
    print(f"  Shared edges: {len(shared_edges)}")

    # =================================================================
    # Cell 9: FIG-Wide 20G Scan
    # =================================================================
    print("\n[9/9] Scanning FIG for 20G clusters (this is the slowest step)...")

    tree = cKDTree(fig_unique)
    found_20Gs = []
    n_neighbors = 200

    for center_idx in range(len(fig_unique)):
        center = fig_unique[center_idx]
        k = min(n_neighbors + 1, len(fig_unique))
        nn_dists, nn_idxs = tree.query(center, k=k)
        nn_dists = nn_dists[1:]
        nn_idxs = nn_idxs[1:]
        nn_pts = fig_unique[nn_idxs] - center
        nn_radii = np.linalg.norm(nn_pts, axis=1)

        if len(nn_radii) < 60:
            continue

        n_inner = min(80, len(nn_pts))
        inner_pts = nn_pts[:n_inner]
        inner_r = nn_radii[:n_inner]
        D_nn = np.zeros((n_inner, n_inner))
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                d = np.linalg.norm(inner_pts[i] - inner_pts[j])
                D_nn[i, j] = D_nn[j, i] = d

        tol = 0.05
        reg_tris = []
        for i in range(n_inner):
            ri = inner_r[i]
            for j in range(i + 1, n_inner):
                rj = inner_r[j]
                dij = D_nn[i, j]
                m3 = (ri + rj + dij) / 3
                if max(abs(ri - m3), abs(rj - m3), abs(dij - m3)) / m3 > tol:
                    continue
                for k in range(j + 1, n_inner):
                    rk = inner_r[k]
                    dik = D_nn[i, k]
                    djk = D_nn[j, k]
                    all6 = [ri, rj, rk, dij, dik, djk]
                    m6 = np.mean(all6)
                    maxd = max(abs(v - m6) / m6 for v in all6)
                    if maxd < tol:
                        reg_tris.append((i, j, k, m6, maxd))

        if len(reg_tris) < 20:
            continue

        reg_tris.sort(key=lambda x: x[4])
        disjoint = []
        used = set()
        for t in reg_tris:
            if {t[0], t[1], t[2]}.isdisjoint(used):
                disjoint.append(t)
                used.update([t[0], t[1], t[2]])

        if len(disjoint) >= 15:
            found_20Gs.append({
                'center_idx': center_idx,
                'center': center.copy(),
                'n_regular_tets': len(disjoint),
                'n_c5c_verts': len(used),
                'nn_idxs': nn_idxs[:n_inner]
            })

        # Progress indicator
        if center_idx % 50 == 0:
            sys.stdout.write(f"\r  Scanned {center_idx}/{len(fig_unique)} vertices, "
                             f"found {len(found_20Gs)} clusters so far...")
            sys.stdout.flush()

    print(f"\r  Scan complete: {len(found_20Gs)} 20G clusters found"
          + " " * 30)

    for i, g in enumerate(found_20Gs):
        dist = np.linalg.norm(g['center'])
        print(f"    Cluster #{i+1}: |r|={dist:.4f}, "
              f"{g['n_regular_tets']} tets, {g['n_c5c_verts']} C5C verts")

    # =================================================================
    # Build JSON export
    # =================================================================
    print("\n" + "=" * 60)
    print("Exporting to e8_fig_data.json...")

    # Helper: subsample large arrays
    def subsample(arr, max_n=1000):
        if len(arr) <= max_n:
            return arr
        idx = np.random.default_rng(42).choice(len(arr), max_n, replace=False)
        return arr[np.sort(idx)]

    # Build cuboctahedra export data (used for cells 6 and 8)
    cubo_export = []
    for ci, (verts_list, set_a, set_b) in enumerate(cuboctahedra):
        # Map core_pts indices to local cubo vertex indices
        idx_map = {v: li for li, v in enumerate(verts_list)}
        cubo_verts_3d = core_pts[verts_list].tolist()

        left_tris = []
        for t in set_a:
            left_tris.append([idx_map[t[0]], idx_map[t[1]], idx_map[t[2]]])
        right_tris = []
        for t in set_b:
            right_tris.append([idx_map[t[0]], idx_map[t[1]], idx_map[t[2]]])

        cubo_export.append({
            "color": cubo_colors[ci % 5],
            "vertices": cubo_verts_3d,
            "left_triangles": left_tris,
            "right_triangles": right_tris
        })

    # Build FIG background (non-C5C vertices) for cell 6
    c5c_vertex_indices = sorted(used_vertices)
    non_c5c = np.ones(len(fig_unique), dtype=bool)
    non_c5c[sorted_idx[0]] = False  # origin
    for v in c5c_vertex_indices:
        non_c5c[core_idx[v]] = False
    fig_background = fig_unique[non_c5c]

    # Perp-space data for cell 3 (subsample outside points)
    inside_perp = perp_all[in_window][:, :3]
    outside_perp = subsample(perp_all[~in_window][:, :3], max_n=1500)
    esqc_norms = np.linalg.norm(esqc_4d, axis=1)

    # FIG dists for cell 4
    fig_dists = np.linalg.norm(fig_3d, axis=1)

    # Cluster data for cell 9
    clusters_export = []
    for g in found_20Gs:
        nn_pts_abs = fig_unique[g['nn_idxs']]
        clusters_export.append({
            "center": g['center'].tolist(),
            "n_tets": g['n_regular_tets'],
            "n_c5c": g['n_c5c_verts'],
            "neighbor_pts": nn_pts_abs.tolist()
        })

    data = {
        "cell1_e8_roots": {
            "type_A": type_A[:, :3].tolist(),
            "type_B": type_B[:, :3].tolist()
        },
        "cell2_projections": {
            "parallel_inner": parallel_4d[mask_inner][:, :3].tolist(),
            "parallel_outer": parallel_4d[mask_outer][:, :3].tolist(),
            "perp_inner": perp_4d[mask_inner][:, :3].tolist(),
            "perp_outer": perp_4d[mask_outer][:, :3].tolist()
        },
        "cell3_esqc": {
            "inside_perp": inside_perp.tolist(),
            "outside_perp": outside_perp.tolist(),
            "esqc_pts": esqc_4d[:, :3].tolist(),
            "esqc_norms": esqc_norms.tolist()
        },
        "cell4_fig": {
            "points": fig_3d.tolist(),
            "dists": fig_dists.tolist()
        },
        "cell5_core": {
            "fig_outer": fig_3d[outer_mask_c5].tolist(),
            "core_pts": fig_3d[core_mask_c5].tolist(),
            "origin": [0.0, 0.0, 0.0],
            "shell_info": shell_info
        },
        "cell6_c5c": {
            "fig_background": fig_background.tolist(),
            "cuboctahedra": cubo_export,
            "origin": [0.0, 0.0, 0.0]
        },
        "cell7_chirality": {
            "left_signed_volumes": left_signed_volumes,
            "right_signed_volumes": right_signed_volumes,
            "bar_colors_left": bar_colors_left,
            "bar_colors_right": bar_colors_right,
            "stats": {
                "shared_verts": len(left_verts & right_verts),
                "left_edges": len(left_edges),
                "right_edges": len(right_edges),
                "shared_edges": len(shared_edges)
            }
        },
        "cell8_chiral_20g": {
            "cuboctahedra": cubo_export,
            "origin": [0.0, 0.0, 0.0]
        },
        "cell9_scan": {
            "fig_pts": fig_unique.tolist(),
            "clusters": clusters_export
        }
    }

    with open('e8_fig_data.json', 'w') as f:
        json.dump(data, f)

    # Verify counts
    fsize = len(json.dumps(data)) / 1024
    print(f"\n  JSON size: {fsize:.0f} KB")
    print(f"  cell1: {len(data['cell1_e8_roots']['type_A'])} + "
          f"{len(data['cell1_e8_roots']['type_B'])} = "
          f"{len(data['cell1_e8_roots']['type_A']) + len(data['cell1_e8_roots']['type_B'])} roots")
    print(f"  cell2: {len(data['cell2_projections']['parallel_inner'])} inner + "
          f"{len(data['cell2_projections']['parallel_outer'])} outer")
    print(f"  cell3: {len(data['cell3_esqc']['esqc_pts'])} ESQC points")
    print(f"  cell4: {len(data['cell4_fig']['points'])} FIG points")
    print(f"  cell5: {len(data['cell5_core']['core_pts'])} core + "
          f"{len(data['cell5_core']['fig_outer'])} outer")
    print(f"  cell6: {len(data['cell6_c5c']['cuboctahedra'])} cuboctahedra")
    print(f"  cell7: {len(data['cell7_chirality']['left_signed_volumes'])} left + "
          f"{len(data['cell7_chirality']['right_signed_volumes'])} right volumes")
    print(f"  cell9: {len(data['cell9_scan']['clusters'])} clusters")
    print(f"\ne8_fig_data.json written successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
