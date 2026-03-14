#!/usr/bin/env python3
"""
Compute the 20G with correct icosahedral face structure and 12 pentagonal gaps.

The 60 C5C vertices cluster into 12 groups of 5 around the 12 icosahedral
vertex directions. The 20 faces each take one vertex from 3 adjacent clusters
(following icosahedral face adjacency). Both chiralities use equilateral
triangles with edge = sqrt(8), non-overlapping, with 12 pentagonal gaps.
"""

import numpy as np
import json
from collections import Counter, defaultdict

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI2 = PHI * PHI
PHI_INV2 = PHI_INV * PHI_INV
VAL_2PHI_1 = 2 * PHI - 1
SCALE = 0.7
EDGE = np.sqrt(8)
EDGE_TOL = 0.001


def generate_c5c_vertices():
    triples = [(2, 0, 2), (PHI, PHI_INV, VAL_2PHI_1), (1, PHI_INV2, PHI2)]
    verts = []
    for (a, b, c) in triples:
        for sx in [+1, -1]:
            for sy in [+1, -1]:
                for sz in [+1, -1]:
                    x, y, z = a*sx, b*sy, c*sz
                    for perm in [(x,y,z), (y,z,x), (z,x,y)]:
                        v = np.array(perm)
                        if not any(np.linalg.norm(v - e) < 1e-8 for e in verts):
                            verts.append(v)
    return np.array(verts)


def build_icosahedron():
    verts = []
    for s1 in [+1, -1]:
        for s2 in [+1, -1]:
            verts.append(np.array([0, s1, s2 * PHI]))
            verts.append(np.array([s1, s2 * PHI, 0]))
            verts.append(np.array([s2 * PHI, 0, s1]))
    verts = np.array(verts)
    edge_len = 2.0
    faces = []
    n = len(verts)
    for i in range(n):
        for j in range(i+1, n):
            if abs(np.linalg.norm(verts[i] - verts[j]) - edge_len) > 0.01:
                continue
            for k in range(j+1, n):
                if (abs(np.linalg.norm(verts[i] - verts[k]) - edge_len) < 0.01 and
                    abs(np.linalg.norm(verts[j] - verts[k]) - edge_len) < 0.01):
                    faces.append((i, j, k))
    return verts, faces


def cluster_vertices(c5c, ico_verts):
    c5c_unit = c5c / np.linalg.norm(c5c, axis=1)[:, None]
    ico_unit = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, None]
    clusters = {i: [] for i in range(12)}
    for vi, v in enumerate(c5c_unit):
        best = np.argmax(ico_unit @ v)
        clusters[best].append(vi)
    return clusters


def find_all_equilateral_cross_cluster(c5c, clusters, ico_faces):
    """Find ALL equilateral triangles (edge=√8) with one vertex per cluster,
    organized by icosahedral face."""
    face_tris = defaultdict(list)

    for fi, (a, b, c_i) in enumerate(ico_faces):
        for va in clusters[a]:
            for vb in clusters[b]:
                d_ab = np.linalg.norm(c5c[va] - c5c[vb])
                if abs(d_ab - EDGE) > EDGE_TOL:
                    continue
                for vc in clusters[c_i]:
                    d_ac = np.linalg.norm(c5c[va] - c5c[vc])
                    d_bc = np.linalg.norm(c5c[vb] - c5c[vc])
                    if abs(d_ac - EDGE) < EDGE_TOL and abs(d_bc - EDGE) < EDGE_TOL:
                        face_tris[fi].append((va, vb, vc))

    return face_tris


def find_20g_assignment(face_tris, n_faces=20):
    """Backtracking: pick one triangle per face using each vertex exactly once."""
    result = [None]

    def bt(fi, used, chosen):
        if fi == n_faces:
            result[0] = chosen[:]
            return True
        for tri in face_tris[fi]:
            va, vb, vc = tri
            if va in used or vb in used or vc in used:
                continue
            used.add(va); used.add(vb); used.add(vc)
            chosen.append(tri)
            if bt(fi + 1, used, chosen):
                return True
            chosen.pop()
            used.remove(va); used.remove(vb); used.remove(vc)
        return False

    bt(0, set(), [])
    return result[0]


def find_both_chiralities(face_tris, n_faces=20):
    """Find TWO disjoint assignments (left and right chirality)."""
    all_results = []

    def bt(fi, used, chosen):
        if fi == n_faces:
            all_results.append(chosen[:])
            return len(all_results) >= 10  # Find up to 10
        for tri in face_tris[fi]:
            va, vb, vc = tri
            if va in used or vb in used or vc in used:
                continue
            used.add(va); used.add(vb); used.add(vc)
            chosen.append(tri)
            if bt(fi + 1, used, chosen):
                return True
            chosen.pop()
            used.remove(va); used.remove(vb); used.remove(vc)
        return False

    bt(0, set(), [])
    return all_results


def verify(triangles, verts, label=""):
    def pit(p, a, b, c):
        nab = np.cross(a, b); nbc = np.cross(b, c); nca = np.cross(c, a)
        ct = (a + b + c); ct /= np.linalg.norm(ct)
        return (np.dot(p, nab)*np.dot(ct, nab) >= -1e-10 and
                np.dot(p, nbc)*np.dot(ct, nbc) >= -1e-10 and
                np.dot(p, nca)*np.dot(ct, nca) >= -1e-10)
    faces_unit = []
    for tri in triangles:
        t = verts[list(tri)]
        t_u = t / np.linalg.norm(t, axis=1)[:, None]
        faces_unit.append(t_u)
    np.random.seed(42)
    N = 20000
    pts = np.random.randn(N, 3); pts /= np.linalg.norm(pts, axis=1)[:, None]
    ic = np.zeros(N, dtype=int)
    for f in faces_unit:
        for j, p in enumerate(pts):
            if pit(p, f[0], f[1], f[2]): ic[j] += 1
    gap = np.sum(ic == 0); clean = np.sum(ic == 1); overlap = np.sum(ic >= 2)
    print(f"  {label}: gap={100*gap/N:.1f}% clean={100*clean/N:.1f}% overlap={100*overlap/N:.1f}%")
    return overlap == 0


def main():
    print("=" * 60)
    print("Computing 20G with Icosahedral Face Structure")
    print("=" * 60)

    c5c = generate_c5c_vertices()
    ico_verts, ico_faces = build_icosahedron()
    clusters = cluster_vertices(c5c, ico_verts)
    print(f"C5C: {len(c5c)} verts, Ico: {len(ico_faces)} faces")
    print(f"Clusters: {[len(v) for v in clusters.values()]}")

    # Find all equilateral cross-cluster triangles
    face_tris = find_all_equilateral_cross_cluster(c5c, clusters, ico_faces)
    for fi in range(20):
        print(f"  Face {fi}: {len(face_tris[fi])} equilateral triangles")

    # Find multiple valid assignments
    print("\nSearching for valid 20G assignments...")
    results = find_both_chiralities(face_tris)
    print(f"Found {len(results)} valid assignments")

    if len(results) < 2:
        print("ERROR: Need at least 2 assignments for chirality!")
        return

    # Pick two assignments that differ most (different signed volume sums)
    def sv_sum(tris):
        return sum(np.dot(c5c[t[0]], np.cross(c5c[t[1]], c5c[t[2]])) for t in tris)

    sv_sums = [(i, sv_sum(r)) for i, r in enumerate(results)]
    sv_sums.sort(key=lambda x: x[1])
    print("Signed volume sums:", [(i, f"{s:.2f}") for i, s in sv_sums])

    # Most positive = left, most negative = right
    left_idx = sv_sums[-1][0]
    right_idx = sv_sums[0][0]

    left_tris = results[left_idx]
    right_tris = results[right_idx]

    print(f"\nLeft (idx={left_idx}, sv={sv_sums[-1][1]:.2f}):")
    lv = set(); [lv.update(t) for t in left_tris]
    print(f"  {len(left_tris)} tris, {len(lv)} verts")
    fe = [np.linalg.norm(c5c[t[i]]-c5c[t[j]]) for t in left_tris for i,j in [(0,1),(0,2),(1,2)]]
    print(f"  Edges: {np.mean(fe):.4f} ± {np.std(fe):.6f}")
    verify(left_tris, c5c, "Left")

    print(f"\nRight (idx={right_idx}, sv={sv_sums[0][1]:.2f}):")
    rv = set(); [rv.update(t) for t in right_tris]
    print(f"  {len(right_tris)} tris, {len(rv)} verts")
    fe2 = [np.linalg.norm(c5c[t[i]]-c5c[t[j]]) for t in right_tris for i,j in [(0,1),(0,2),(1,2)]]
    print(f"  Edges: {np.mean(fe2):.4f} ± {np.std(fe2):.6f}")
    verify(right_tris, c5c, "Right")

    print(f"\nSame vertex set? {lv == rv}")

    # Export
    print("\n--- Export ---")
    edge_len = np.sqrt(8)
    adj = defaultdict(set)
    for i in range(60):
        for j in range(i+1, 60):
            if abs(np.linalg.norm(c5c[i]-c5c[j]) - edge_len) < 0.001:
                adj[i].add(j); adj[j].add(i)
    visited = set(); cubos = []
    for s in range(60):
        if s in visited: continue
        q = [s]; comp = set()
        while q:
            n = q.pop(0)
            if n in visited: continue
            visited.add(n); comp.add(n)
            for nb in adj[n]:
                if nb not in visited: q.append(nb)
        cubos.append(sorted(comp))

    v_to_cubo = {}
    for ci, comp in enumerate(cubos):
        for v in comp: v_to_cubo[v] = ci
    cnames = ['red', 'blue', 'green', 'orange', 'purple']

    export = {
        "vertices": (c5c * SCALE).tolist(),
        "origin": [0.0, 0.0, 0.0],
        "left_triangles": [list(t) for t in left_tris],
        "right_triangles": [list(t) for t in right_tris],
        "left_colors": [cnames[v_to_cubo[t[0]] % 5] for t in left_tris],
        "right_colors": [cnames[v_to_cubo[t[0]] % 5] for t in right_tris],
        "cuboctahedra": []
    }
    for ci, comp in enumerate(cubos):
        export["cuboctahedra"].append({
            "color": cnames[ci % 5],
            "vertex_indices": comp,
            "vertices": (c5c[comp] * SCALE).tolist(),
            "left_triangles": [list(t) for t in left_tris if any(v in comp for v in t)],
            "right_triangles": [list(t) for t in right_tris if any(v in comp for v in t)],
        })

    with open('perfect_20g.json', 'w') as f:
        json.dump(export, f, indent=2)

    try:
        with open('e8_fig_data.json', 'r') as f:
            full = json.load(f)
        full['cell8_chiral_20g'] = export
        with open('e8_fig_data.json', 'w') as f:
            json.dump(full, f)
        print("Updated e8_fig_data.json")
    except: pass

    print("Done!")


if __name__ == '__main__':
    main()
