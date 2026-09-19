"""Optimized Python adaptation of the supplied Wild simplifier.

Usage: python wild_simplify_optimized.py --input mesh.obj --ratio 0.5
Optional acceleration: pip install numba
Improved UV packing (default): pip install xatlas
The first Numba call includes compilation time. Geometry and UV conventions,
virtual-edge threshold, area weights, and fixed-query successive mapping follow
the supplied implementation. This is not the authors' reference implementation.
Floating-point summation and tie ordering can change the collapse sequence.
The original per-face tiled atlas may exceed --texture_size (minimum 6px/tile).
"""
if __name__ == "__main__":
    print("[startup] wild_simplify_optimized v2: loading dependencies...", flush=True)

import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import heapq
import time
import warnings
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from PIL import Image


def compute_area_quadric(edge, vertices, vertex_edges, edge_faces):
    a, b = edge
    Q_area = np.zeros((4, 4), dtype=np.float64)
    one_ring_edges = vertex_edges[a] | vertex_edges[b]

    for boundary_edge in one_ring_edges:
        if boundary_edge not in edge_faces:
            continue
        if len(edge_faces[boundary_edge]) != 1:
            continue

        u, v = boundary_edge
        s = vertices[v] - vertices[u]
        t = np.cross(vertices[u], vertices[v])
        S = cross_matrix(s)

        Q_area[:3, :3] += 0.5 * (S.T @ S)
        q = -0.5 * (S @ t)
        Q_area[:3, 3] += q
        Q_area[3, :3] += q
        Q_area[3, 3] += 0.5 * np.dot(t, t)

    return Q_area


def compute_edge_cost(edge, vertices, Q, vertex_edges, edge_faces, area_cache=None):
    a, b = edge
    Q_ab = Q[a] + Q[b]
    if area_cache is None:
        area = compute_area_quadric(edge, vertices, vertex_edges, edge_faces)
    else:
        vertex_area, boundary_quadrics = area_cache
        area = vertex_area[a] + vertex_area[b]
        shared = boundary_quadrics.get(edge)
        if shared is not None:
            area = area - shared  # union: shared boundary edge counted once
    Q_total = Q_ab + area

    try:
        position = np.linalg.solve(
            Q_total[:3, :3],
            -Q_total[:3, 3]
        )
    except np.linalg.LinAlgError:
        position = (vertices[a] + vertices[b]) / 2.0

    position_homo = np.append(position, 1.0)

    cost = (
        position_homo.T
        @ Q_total
        @ position_homo
    )

    return cost, position


def point_triangle_distance_sq(point, triangle):
    closest, _ = closest_point_on_triangle(
        point,
        triangle[0],
        triangle[1],
        triangle[2]
    )
    delta = point - closest
    return np.dot(delta, delta)


def segment_segment_distance_sq(p1, q1, p2, q2):
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    eps = 1e-15

    if a <= eps and e <= eps:
        delta = p1 - p2
        return np.dot(delta, delta)

    if a <= eps:
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    else:
        c = np.dot(d1, r)

        if e <= eps:
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b

            if denom != 0.0:
                s = min(max((b * f - c * e) / denom, 0.0), 1.0)
            else:
                s = 0.0

            t = (b * s + f) / e

            if t < 0.0:
                t = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)

    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    delta = c1 - c2
    return np.dot(delta, delta)


def segment_intersects_triangle(p0, p1, triangle):
    v0, v1, v2 = triangle
    direction = p1 - p0
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    det = np.dot(edge1, h)
    eps = 1e-12

    if abs(det) < eps:
        return False

    inv_det = 1.0 / det
    s = p0 - v0
    u = inv_det * np.dot(s, h)

    if u < -eps or u > 1.0 + eps:
        return False

    q = np.cross(s, edge1)
    v = inv_det * np.dot(direction, q)

    if v < -eps or u + v > 1.0 + eps:
        return False

    t = inv_det * np.dot(edge2, q)
    return -eps <= t <= 1.0 + eps


def triangle_triangle_distance_sq(triangle_a, triangle_b):
    edges = ((0, 1), (1, 2), (2, 0))

    for i, j in edges:
        if segment_intersects_triangle(
            triangle_a[i],
            triangle_a[j],
            triangle_b
        ):
            return 0.0

    for i, j in edges:
        if segment_intersects_triangle(
            triangle_b[i],
            triangle_b[j],
            triangle_a
        ):
            return 0.0

    minimum = np.inf

    for point in triangle_a:
        minimum = min(
            minimum,
            point_triangle_distance_sq(point, triangle_b)
        )

    for point in triangle_b:
        minimum = min(
            minimum,
            point_triangle_distance_sq(point, triangle_a)
        )

    for a0, a1 in edges:
        for b0, b1 in edges:
            minimum = min(
                minimum,
                segment_segment_distance_sq(
                    triangle_a[a0],
                    triangle_a[a1],
                    triangle_b[b0],
                    triangle_b[b1]
                )
            )

    return minimum


def closest_vertex_pair(face_a, face_b, vertices):
    best_a, best_b = -1, -1
    best_distance = np.inf
    for a in face_a:
        for b in face_b:
            if a == b:
                continue
            dx = vertices[a, 0] - vertices[b, 0]
            dy = vertices[a, 1] - vertices[b, 1]
            dz = vertices[a, 2] - vertices[b, 2]
            distance = dx * dx + dy * dy + dz * dz
            if distance < best_distance:
                best_distance = distance
                best_a, best_b = min(a, b), max(a, b)
    if best_a < 0:
        return None
    return int(best_a), int(best_b)


def virtual_edge_batch(i, candidates, triangles, faces, vertices, threshold_sq):
    # Compile the entire candidate loop, not just individual distance calls.
    pairs = np.empty((len(candidates), 2), dtype=np.int64)
    count = 0
    for j in candidates:
        distance_sq = triangle_triangle_distance_sq(triangles[i], triangles[j])
        if distance_sq >= threshold_sq:
            continue
        edge = closest_vertex_pair(faces[i], faces[j], vertices)
        if edge is not None:
            pairs[count, 0], pairs[count, 1] = edge
            count += 1
    return pairs[:count]


def connected_components(vertices, faces):
    parent = np.arange(len(vertices), dtype=np.int64)
    rank = np.zeros(len(vertices), dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra == rb:
            return

        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for face in faces:
        i, j, k = face
        union(i, j)
        union(j, k)
        union(k, i)

    face_components = np.empty(len(faces), dtype=np.int64)

    for fi, face in enumerate(faces):
        face_components[fi] = find(int(face[0]))

    return face_components


def build_virtual_edges(vertices, faces, radius):
    if radius <= 0.0 or len(faces) == 0:
        print("[virtual edges] skipped (radius=0 or empty mesh)", flush=True)
        return set()

    started = time.perf_counter()
    print("[virtual edges] finding connected components...", flush=True)
    face_components = connected_components(vertices, faces)
    component_count = len(np.unique(face_components))
    print(f"[virtual edges] components={component_count:,}, "
          f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if component_count == 1:
        print("[virtual edges] single component: no virtual edges needed", flush=True)
        return set()
    triangles = vertices[faces]
    bounds_min = triangles.min(axis=1)
    bounds_max = triangles.max(axis=1)
    centroids = np.mean(triangles, axis=1)
    triangle_radii = np.max(
        np.linalg.norm(triangles - centroids[:, None, :], axis=2), axis=1)
    maximum_radius = np.max(triangle_radii)
    tree = cKDTree(centroids)
    virtual_edges = set()
    threshold_sq = (2.0 * radius) ** 2

    if NUMBA_AVAILABLE:
        print(
            "[virtual edges] preparing Numba kernel (first run compiles)...", flush=True)
        virtual_edge_batch(0, np.empty(0, dtype=np.int64), triangles,
                           faces, vertices, threshold_sq)
        print("[virtual edges] Numba kernel ready", flush=True)
    else:
        print(
            "[virtual edges] Python fallback; install numba for acceleration", flush=True)

    checked = 0
    broad_candidates = 0
    last_report = time.perf_counter()
    print(f"[virtual edges] scanning 0/{len(faces):,} faces", flush=True)

    def report(completed, force=False):
        nonlocal last_report
        now = time.perf_counter()
        if force or now - last_report >= 2.0:
            print(f"[virtual edges] faces={completed:,}/{len(faces):,} "
                  f"({100.0*completed/len(faces):.1f}%) "
                  f"broad={broad_candidates:,} checked={checked:,} "
                  f"edges={len(virtual_edges):,} elapsed={now-started:.1f}s",
                  flush=True)
            last_report = now

    for i in range(len(faces)):
        search_radius = 2.0 * radius + triangle_radii[i] + maximum_radius
        candidates = np.asarray(tree.query_ball_point(centroids[i], search_radius),
                                dtype=np.int64)
        candidates = candidates[(candidates > i) &
                                (face_components[candidates] != face_components[i])]
        broad_candidates += len(candidates)
        if len(candidates):
            gap = np.maximum(0.0, np.maximum(bounds_min[candidates] - bounds_max[i],
                                             bounds_min[i] - bounds_max[candidates]))
            candidates = candidates[np.einsum(
                "ij,ij->i", gap, gap) < threshold_sq]
            # Bounded batches return control regularly for progress and Ctrl+C.
            batch_size = 2048 if NUMBA_AVAILABLE else 32
            for begin in range(0, len(candidates), batch_size):
                batch = candidates[begin:begin + batch_size]
                pairs = virtual_edge_batch(
                    i, batch, triangles, faces, vertices, threshold_sq)
                virtual_edges.update((int(a), int(b)) for a, b in pairs)
                checked += len(batch)
                report(i)
        report(i + 1)
    report(len(faces), force=True)
    return virtual_edges


def simplify(vertices, faces, target_faces, virtual_radius, track_history):

    # The Q matrix from QEM for each vertex
    Q = np.zeros((len(vertices), 4, 4), dtype=np.float64)
    # Holds a list of face indices for each vertex
    vertex_faces = [set() for _ in range(len(vertices))]
    # Holds a list of edge for each vertex
    vertex_edges = [set() for _ in range(len(vertices))]
    # Dictionary with edge tuple key and set of face indices as value
    edge_faces = defaultdict(set)
    # Version dict for preventing using old versions from heap pop
    edge_version = defaultdict(int)

    vertex_active = np.ones(len(vertices), dtype=bool)
    face_active = np.ones(len(faces), dtype=bool)

    init_started = time.perf_counter()
    print("[initialize] building mesh adjacency...", flush=True)
    for f_idx, face in enumerate(faces):
        if f_idx and f_idx % 25000 == 0:
            print(f"[initialize] faces={f_idx:,}/{len(faces):,}, "
                  f"elapsed={time.perf_counter()-init_started:.1f}s", flush=True)

        i, j, k = face

        vertex_faces[i].add(f_idx)
        vertex_faces[j].add(f_idx)
        vertex_faces[k].add(f_idx)

        edges = [
            tuple(sorted((i, j))),
            tuple(sorted((j, k))),
            tuple(sorted((k, i)))
        ]

        for edge in edges:
            a, b = edge
            vertex_edges[a].add(edge)
            vertex_edges[b].add(edge)
            edge_faces[edge].add(f_idx)

    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.where(lengths > 0, lengths, 1.0)[:, None]
    planes = np.column_stack(
        (normals, -np.einsum("ij,ij->i", normals, triangles[:, 0])))
    quadrics = np.einsum("ni,nj,n->nij", planes, planes, lengths / 6.0)
    for corner in range(3):
        np.add.at(Q, faces[:, corner], quadrics)
    del triangles, normals, lengths, planes, quadrics

    print(
        f"[virtual edges] faces={len(faces):,}, radius={virtual_radius:g}", flush=True)
    physical_edges = set(edge_faces.keys())
    virtual_edges = build_virtual_edges(
        vertices,
        faces,
        virtual_radius
    )

    print(f"[virtual edges] candidates={len(virtual_edges):,}", flush=True)
    for edge in virtual_edges:
        if edge in physical_edges:
            continue
        a, b = edge
        vertex_edges[a].add(edge)
        vertex_edges[b].add(edge)
        edge_faces[edge]

    vertex_area = np.zeros_like(Q)
    boundary_quadrics = {}

    def refresh_area(vertex_ids):
        # Rebuild current boundary contributions, never accumulate old area terms.
        touched_edges = set()
        for v in vertex_ids:
            touched_edges.update(vertex_edges[v])
        for e in touched_edges:
            if len(edge_faces[e]) == 1:
                u, v = e
                s = vertices[v] - vertices[u]
                t = np.cross(vertices[u], vertices[v])
                S = cross_matrix(s)
                q = np.empty((4, 4), dtype=np.float64)
                q[:3, :3] = 0.5 * (S.T @ S)
                q[:3, 3] = q[3, :3] = -0.5 * (S @ t)
                q[3, 3] = 0.5 * np.dot(t, t)
                boundary_quadrics[e] = q
            else:
                boundary_quadrics.pop(e, None)
        for v in vertex_ids:
            vertex_area[v].fill(0.0)
            for e in vertex_edges[v]:
                q = boundary_quadrics.get(e)
                if q is not None:
                    vertex_area[v] += q

    refresh_area(range(len(vertices)))
    area_cache = (vertex_area, boundary_quadrics)
    started = time.perf_counter()
    last_report = started
    collapse_count = 0

    # Min heap for getting the lowest cost edge
    heap = []

    # Min heap tie-breaker counter
    counter = 0

    print(f"[heap] initializing {len(edge_faces):,} edge costs", flush=True)
    for edge_index, edge in enumerate(list(edge_faces.keys())):
        if edge_index and edge_index % 25000 == 0:
            print(
                f"[heap] costs={edge_index:,}/{len(edge_faces):,}", flush=True)
        cost, position = compute_edge_cost(
            edge,
            vertices,
            Q,
            vertex_edges,
            edge_faces,
            area_cache
        )

        version = edge_version[edge]

        heap.append((cost, counter, edge, position, version))
        counter += 1

    heapq.heapify(heap)
    active_face_count = len(faces)
    history = []

    while active_face_count > target_faces and heap:
        cost, _, edge, position, version = heapq.heappop(heap)
        a, b = edge

        if not vertex_active[a] or not vertex_active[b]:
            continue
        # Edge could still have been popped from heap even though,
        # it is already deleted in edge_faces
        if edge not in edge_faces:
            continue
        # Skip if the version is old
        if version != edge_version[edge]:
            continue

        old_face_ids = {
            fi for fi in (vertex_faces[a] | vertex_faces[b])
            if face_active[fi]
        }
        old_edges = set(vertex_edges[a]) | set(vertex_edges[b])
        affected_vertices = {a, b}

        for fi in old_face_ids:
            affected_vertices.update(int(v) for v in faces[fi])

        for e in old_edges:
            affected_vertices.update(e)

        if track_history:
            pre_face_ids = np.array(
                sorted(old_face_ids),
                dtype=np.int64
            )
            if len(pre_face_ids) > 0:
                pre_triangles = vertices[faces[pre_face_ids]].astype(
                    np.float32,
                    copy=True
                )
            else:
                pre_triangles = np.empty((0, 3, 3), dtype=np.float32)

        for fi in old_face_ids:
            face = faces[fi]

            for v in face:
                vertex_faces[v].discard(fi)

            i, j, k = face
            local_edges = [
                tuple(sorted((i, j))),
                tuple(sorted((j, k))),
                tuple(sorted((k, i))),
            ]

            for e in local_edges:
                if e in edge_faces:
                    edge_faces[e].discard(fi)

        for e in old_edges:
            boundary_quadrics.pop(e, None)
            edge_version[e] += 1

            if e in edge_faces:
                del edge_faces[e]

            u, v = e
            vertex_edges[u].discard(e)
            vertex_edges[v].discard(e)

        vertices[a] = position
        Q[a] += Q[b]
        vertex_active[b] = False

        # Used list to copy face indices, since it could be modified
        # throughout the for loop
        for fi in list(old_face_ids):
            if not face_active[fi]:
                continue

            face = faces[fi].copy()

            # (a,b,c) -> (a,a,c) degenerate face
            if a in face and b in face:
                face_active[fi] = False
                active_face_count -= 1
                continue

            # b -> a
            faces[fi][faces[fi] == b] = a

            if len(set(int(v) for v in faces[fi])) < 3:
                face_active[fi] = False
                active_face_count -= 1

        unique_faces = {}

        for fi in sorted(old_face_ids):
            if not face_active[fi]:
                continue

            key = tuple(sorted(int(v) for v in faces[fi]))

            if key in unique_faces:
                face_active[fi] = False
                active_face_count -= 1
            else:
                unique_faces[key] = fi

        for fi in old_face_ids:
            if not face_active[fi]:
                continue

            i, j, k = faces[fi]

            vertex_faces[i].add(fi)
            vertex_faces[j].add(fi)
            vertex_faces[k].add(fi)

            new_edges = [
                tuple(sorted((i, j))),
                tuple(sorted((j, k))),
                tuple(sorted((k, i))),
            ]

            for e in new_edges:
                u, v = e
                edge_faces[e].add(fi)
                vertex_edges[u].add(e)
                vertex_edges[v].add(e)

        for old_edge in old_edges:
            u, v = old_edge

            if u == b:
                u = a
            if v == b:
                v = a
            if u == v:
                continue

            new_edge = tuple(sorted((u, v)))
            x, y = new_edge
            edge_faces[new_edge]
            vertex_edges[x].add(new_edge)
            vertex_edges[y].add(new_edge)

        vertex_faces[b].clear()
        vertex_edges[b].clear()

        if track_history:
            history.append((
                pre_face_ids,
                pre_triangles,
                np.array(sorted(vertex_faces[a]), dtype=np.int64)
            ))

        refresh_area(affected_vertices)
        affected_edges = set()

        for v in affected_vertices:
            if v < 0 or v >= len(vertices):
                continue
            if not vertex_active[v]:
                continue
            affected_edges.update(vertex_edges[v])

        for new_edge in affected_edges:
            u, v = new_edge

            if not vertex_active[u] or not vertex_active[v]:
                continue
            if new_edge not in edge_faces:
                continue

            new_cost, new_position = compute_edge_cost(
                new_edge,
                vertices,
                Q,
                vertex_edges,
                edge_faces,
                area_cache
            )

            edge_version[new_edge] += 1
            new_version = edge_version[new_edge]

            heapq.heappush(
                heap,
                (
                    new_cost,
                    counter,
                    new_edge,
                    new_position,
                    new_version
                )
            )
            counter += 1

        collapse_count += 1
        # Lazy invalidation needs bounded storage on large meshes.
        if len(heap) > max(10000, 4 * len(edge_faces)):
            heap = [item for item in heap
                    if item[2] in edge_faces
                    and item[4] == edge_version[item[2]]
                    and vertex_active[item[2][0]] and vertex_active[item[2][1]]]
            heapq.heapify(heap)
        if collapse_count % 1000 == 0:
            now = time.perf_counter()
            if now - last_report >= 5.0:
                print(f"[simplify] faces={active_face_count:,} target={target_faces:,} "
                      f"collapses={collapse_count:,} elapsed={now-started:.1f}s", flush=True)
                last_report = now

    active_vertex_indices = np.where(vertex_active)[0]

    old_to_new = -np.ones(len(vertices), dtype=np.int64)
    old_to_new[active_vertex_indices] = np.arange(
        len(active_vertex_indices)
    )

    new_vertices = vertices[active_vertex_indices]

    active_face_indices = np.where(face_active)[0]
    new_faces = old_to_new[faces[active_face_indices]]

    return new_vertices, new_faces, active_face_indices, history


def cross_matrix(v):
    x, y, z = v

    matrix = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ], dtype=np.float64)

    return matrix


def closest_point_on_triangle(point, a, b, c):
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), np.array([1.0, 0.0, 0.0])

    bp = point - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)

    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), np.array([0.0, 1.0, 0.0])

    vc = d1 * d4 - d3 * d2

    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1.0 - v, v, 0.0])

    cp = point - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)

    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), np.array([0.0, 0.0, 1.0])

    vb = d5 * d2 - d1 * d6

    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1.0 - w, 0.0, w])

    va = d3 * d6 - d5 * d4

    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = c - b
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * bc, np.array([0.0, 1.0 - w, w])

    denom = va + vb + vc

    if abs(denom) < 1e-15:
        distances = [
            np.dot(point - a, point - a),
            np.dot(point - b, point - b),
            np.dot(point - c, point - c)
        ]
        index = int(np.argmin(np.asarray(distances)))
        bary = np.zeros(3, dtype=np.float64)
        bary[index] = 1.0
        triangle = (a, b, c)
        return triangle[index].copy(), bary

    inv_denom = 1.0 / denom
    v = vb * inv_denom
    w = vc * inv_denom
    u = 1.0 - v - w
    closest = u * a + v * b + w * c
    return closest, np.array([u, v, w])


def build_grid_atlas_samples(vertices, faces, active_face_indices, texture_size):
    """Legacy per-face tiled atlas kept as a fallback/debugging mode."""
    face_count = len(faces)
    grid_size = max(1, int(np.ceil(np.sqrt(face_count))))
    tile_size = max(6, texture_size // grid_size)
    atlas_size = tile_size * grid_size
    margin = max(1, min(2, tile_size // 4))
    span = tile_size - 2 * margin - 1
    yy, xx = np.mgrid[margin:tile_size-margin, margin:tile_size-margin]
    beta = (xx - margin) / span
    gamma = (yy - margin) / span
    alpha = 1.0 - beta - gamma
    valid = alpha >= -1e-12
    template_bary = np.column_stack((alpha[valid], beta[valid], gamma[valid]))
    template_pixels = np.column_stack((yy[valid], xx[valid]))
    mask = np.zeros((tile_size, tile_size), dtype=bool)
    mask[yy[valid], xx[valid]] = True
    origins = np.column_stack((np.arange(face_count) // grid_size,
                               np.arange(face_count) % grid_size)) * tile_size
    uv_template = np.array([[margin+.5, margin+.5],
                            [margin+span+.5, margin+.5],
                            [margin+.5, margin+span+.5]])
    triangle_uvs = (origins[:, None, ::-1] + uv_template) / atlas_size
    triangle_uvs[:, :, 1] = 1.0 - triangle_uvs[:, :, 1]
    samples_per_face = len(template_bary)
    points = np.empty((face_count * samples_per_face, 3), dtype=np.float64)
    for begin in range(0, face_count, 4096):
        end = min(begin + 4096, face_count)
        points[begin*samples_per_face:end*samples_per_face] = np.einsum(
            "sj,fjk->fsk", template_bary, vertices[faces[begin:end]]).reshape(-1, 3)
    barycentrics = np.tile(template_bary, (face_count, 1))
    face_ids = np.repeat(active_face_indices, samples_per_face)
    pixels = (origins[:, None, :] + template_pixels).reshape(-1, 2)
    masks = [(int(y), int(x), mask) for y, x in origins]

    uv_vertices = triangle_uvs.reshape(-1, 2).copy()
    face_uv_indices = np.arange(face_count * 3, dtype=np.int64).reshape(-1, 3)

    print(
        f"[atlas:grid] {atlas_size}x{atlas_size}, {len(points):,} samples", flush=True)
    return {
        "atlas_size": atlas_size,
        "triangle_uvs": triangle_uvs,
        "uv_vertices": uv_vertices,
        "face_uv_indices": face_uv_indices,
        "points": points,
        "barycentrics": barycentrics,
        "face_ids": face_ids,
        "pixels": pixels,
        "tile_size": tile_size,
        "masks": masks,
        "valid_mask": None,
    }


def barycentric_2d(points_uv, triangle_uv):
    """Return barycentric coordinates for 2D points with respect to a UV triangle."""
    a, b, c = triangle_uv
    v0 = b - a
    v1 = c - a
    v2 = points_uv - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return np.empty((0, 3), dtype=np.float64)
    d20 = v2 @ v0
    d21 = v2 @ v1
    beta = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    alpha = 1.0 - beta - gamma
    return np.column_stack((alpha, beta, gamma))


def rasterize_uv_triangle(triangle_uv, texture_size):
    """Rasterize texel centers covered by one normalized UV triangle."""
    u_min = max(0.0, float(np.min(triangle_uv[:, 0])))
    u_max = min(1.0, float(np.max(triangle_uv[:, 0])))
    v_min = max(0.0, float(np.min(triangle_uv[:, 1])))
    v_max = min(1.0, float(np.max(triangle_uv[:, 1])))

    x0 = max(0, int(np.floor(u_min * texture_size - 0.5)))
    x1 = min(texture_size - 1, int(np.ceil(u_max * texture_size - 0.5)))
    y0 = max(0, int(np.floor((1.0 - v_max) * texture_size - 0.5)))
    y1 = min(texture_size - 1, int(np.ceil((1.0 - v_min) * texture_size - 0.5)))

    if x1 < x0 or y1 < y0:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.float64))

    xs = np.arange(x0, x1 + 1, dtype=np.int32)
    ys = np.arange(y0, y1 + 1, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    sample_uv = np.column_stack((
        (xx.ravel().astype(np.float64) + 0.5) / texture_size,
        1.0 - (yy.ravel().astype(np.float64) + 0.5) / texture_size,
    ))
    bary = barycentric_2d(sample_uv, triangle_uv)
    if len(bary) == 0:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.float64))

    eps = 1e-8
    inside = np.all(bary >= -eps, axis=1) & np.all(bary <= 1.0 + eps, axis=1)
    if not np.any(inside):
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.float64))

    return yy.ravel()[inside], xx.ravel()[inside], bary[inside]


def build_xatlas_samples(vertices, faces, active_face_indices, texture_size, padding):
    """Create a charted UV atlas with xatlas and rasterize its texels.

    Unlike the old per-face atlas, adjacent triangles share UV vertices inside
    a chart, so Unreal does not split every triangle corner into a separate
    render vertex. Chart area is packed according to surface geometry instead
    of assigning the same tiny tile to every face.
    """
    try:
        import xatlas
    except ImportError as exc:
        raise RuntimeError(
            "xatlas is required for the improved atlas. Install it with: pip install xatlas\n"
            "Or run with --atlas grid to use the old per-face atlas."
        ) from exc

    print(f"[atlas:xatlas] generating charts at {texture_size}x{texture_size} "
          f"with padding={padding}", flush=True)

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.uint32)
    )
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    if hasattr(pack_options, "resolution"):
        pack_options.resolution = int(texture_size)
    if hasattr(pack_options, "padding"):
        pack_options.padding = int(padding)
    try:
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
    except TypeError:
        # Compatibility with xatlas bindings that only accept positional args.
        atlas.generate(chart_options, pack_options)

    vmapping, atlas_indices, uv_vertices = atlas[0]
    vmapping = np.asarray(vmapping, dtype=np.int64)
    atlas_indices = np.asarray(atlas_indices, dtype=np.int64).reshape(-1, 3)
    uv_vertices = np.asarray(uv_vertices, dtype=np.float64)

    if len(atlas_indices) != len(faces):
        raise RuntimeError(
            f"xatlas returned {len(atlas_indices)} triangles for {len(faces)} input triangles."
        )

    # Match xatlas output triangles back to the simplified input face IDs.
    # This avoids relying on xatlas preserving face order.
    face_lookup = defaultdict(list)
    for fi, face in enumerate(faces):
        face_lookup[tuple(sorted(int(v) for v in face))].append(fi)

    triangle_uvs = np.empty((len(faces), 3, 2), dtype=np.float64)
    face_uv_indices = np.empty((len(faces), 3), dtype=np.int64)

    for atlas_fi, uv_face in enumerate(atlas_indices):
        geom_face = vmapping[uv_face]
        key = tuple(sorted(int(v) for v in geom_face))
        candidates = face_lookup.get(key)
        if not candidates:
            raise RuntimeError(
                "Could not map an xatlas triangle back to the simplified mesh."
            )
        source_fi = candidates.pop()
        source_face = faces[source_fi]

        for source_corner, vertex_id in enumerate(source_face):
            matches = np.where(geom_face == int(vertex_id))[0]
            if len(matches) != 1:
                raise RuntimeError("Ambiguous xatlas corner mapping encountered.")
            atlas_corner = int(matches[0])
            uv_index = int(uv_face[atlas_corner])
            face_uv_indices[source_fi, source_corner] = uv_index
            triangle_uvs[source_fi, source_corner] = uv_vertices[uv_index]

    if not np.all(np.isfinite(triangle_uvs)):
        raise RuntimeError("xatlas produced non-finite UV coordinates.")

    # Some bindings may expose pixel-space UVs. Normalize if necessary.
    uv_max = float(np.max(uv_vertices)) if len(uv_vertices) else 1.0
    if uv_max > 1.5:
        uv_vertices = uv_vertices / float(texture_size)
        triangle_uvs = triangle_uvs / float(texture_size)

    point_parts = []
    bary_parts = []
    face_id_parts = []
    pixel_parts = []
    valid_mask = np.zeros((texture_size, texture_size), dtype=bool)

    print(f"[atlas:xatlas] rasterizing {len(faces):,} UV triangles...", flush=True)
    total_samples = 0
    for fi, triangle_uv in enumerate(triangle_uvs):
        rows, cols, bary = rasterize_uv_triangle(triangle_uv, texture_size)
        if len(rows) == 0:
            continue

        points = bary @ vertices[faces[fi]]
        point_parts.append(points.astype(np.float64, copy=False))
        bary_parts.append(bary.astype(np.float64, copy=False))
        face_id_parts.append(np.full(len(rows), active_face_indices[fi], dtype=np.int64))
        pixel_parts.append(np.column_stack((rows, cols)).astype(np.int32, copy=False))
        valid_mask[rows, cols] = True
        total_samples += len(rows)

        if fi and fi % 5000 == 0:
            print(f"[atlas:xatlas] faces={fi:,}/{len(faces):,}, "
                  f"samples={total_samples:,}", flush=True)

    if not point_parts:
        raise RuntimeError("xatlas rasterization produced no texture samples.")

    points = np.concatenate(point_parts, axis=0)
    barycentrics = np.concatenate(bary_parts, axis=0)
    face_ids = np.concatenate(face_id_parts, axis=0)
    pixels = np.concatenate(pixel_parts, axis=0)

    print(f"[atlas:xatlas] {texture_size}x{texture_size}, "
          f"uv_vertices={len(uv_vertices):,}, samples={len(points):,}", flush=True)

    return {
        "atlas_size": texture_size,
        "triangle_uvs": triangle_uvs,
        "uv_vertices": uv_vertices,
        "face_uv_indices": face_uv_indices,
        "points": points,
        "barycentrics": barycentrics,
        "face_ids": face_ids,
        "pixels": pixels,
        "tile_size": None,
        "masks": None,
        "valid_mask": valid_mask,
    }


def project_samples(points, triangles):
    selected = np.empty(len(points), dtype=np.int64)
    barycentrics = np.empty((len(points), 3), dtype=np.float64)
    for i in range(len(points)):
        best_distance = np.inf
        for j in range(len(triangles)):
            closest, bary = closest_point_on_triangle(
                points[i], triangles[j, 0], triangles[j, 1], triangles[j, 2])
            delta = points[i] - closest
            distance = np.dot(delta, delta)
            if distance < best_distance:
                best_distance = distance
                selected[i] = j
                barycentrics[i] = bary
    return selected, barycentrics


def successive_map(points, barycentrics, face_ids, history):
    current_faces = face_ids.copy()
    current_barycentrics = barycentrics.copy()
    face_samples = defaultdict(list)

    for sample_index, face_id in enumerate(current_faces):
        face_samples[int(face_id)].append(sample_index)

    for pre_face_ids, pre_triangles, post_face_ids in reversed(history):
        affected_samples = []

        for face_id in post_face_ids:
            sample_indices = face_samples.pop(int(face_id), None)

            if sample_indices:
                affected_samples.extend(sample_indices)

        if not affected_samples or len(pre_face_ids) == 0:
            continue

        reassigned = defaultdict(list)

        sample_indices = np.asarray(affected_samples, dtype=np.int64)
        selected, bary = project_samples(
            points[sample_indices], pre_triangles.astype(np.float64))
        chosen_faces = pre_face_ids[selected]
        current_faces[sample_indices] = chosen_faces
        current_barycentrics[sample_indices] = bary
        for sample_index, best_face in zip(sample_indices, chosen_faces):
            reassigned[int(best_face)].append(int(sample_index))

        for face_id, sample_indices in reassigned.items():
            face_samples[face_id].extend(sample_indices)

    return current_faces, current_barycentrics


def sample_texture_bilinear(texture, uvs):
    if texture.ndim == 2:
        texture = np.repeat(texture[:, :, None], 3, axis=2)
    if texture.shape[2] == 4:
        texture = texture[:, :, :3]

    height, width = texture.shape[:2]
    u = np.clip(uvs[:, 0], 0.0, 1.0)
    v = np.clip(uvs[:, 1], 0.0, 1.0)
    x = u * (width - 1)
    y = (1.0 - v) * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]

    c00 = texture[y0, x0].astype(np.float64)
    c10 = texture[y0, x1].astype(np.float64)
    c01 = texture[y1, x0].astype(np.float64)
    c11 = texture[y1, x1].astype(np.float64)

    top = c00 * (1.0 - wx) + c10 * wx
    bottom = c01 * (1.0 - wx) + c11 * wx
    result = top * (1.0 - wy) + bottom * wy
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def pad_atlas(atlas, valid_mask, padding):
    """Dilate valid texels into xatlas chart gutters to reduce mip bleeding."""
    if padding <= 0 or valid_mask is None or np.all(valid_mask):
        return atlas
    missing = ~valid_mask
    distances, nearest = distance_transform_edt(
        missing,
        return_distances=True,
        return_indices=True
    )
    fill_mask = missing & (distances <= float(padding))
    if np.any(fill_mask):
        atlas[fill_mask] = atlas[
            nearest[0][fill_mask],
            nearest[1][fill_mask]
        ]
    return atlas


def bake_texture(
    vertices,
    faces,
    active_face_indices,
    history,
    original_triangle_uvs,
    original_material_ids,
    original_textures,
    texture_size,
    atlas_mode,
    texture_padding
):
    if atlas_mode == "xatlas":
        atlas_data = build_xatlas_samples(
            vertices,
            faces,
            active_face_indices,
            texture_size,
            texture_padding
        )
    else:
        atlas_data = build_grid_atlas_samples(
            vertices,
            faces,
            active_face_indices,
            texture_size
        )

    atlas_size = atlas_data["atlas_size"]
    triangle_uvs = atlas_data["triangle_uvs"]
    uv_vertices = atlas_data["uv_vertices"]
    face_uv_indices = atlas_data["face_uv_indices"]
    points = atlas_data["points"]
    barycentrics = atlas_data["barycentrics"]
    face_ids = atlas_data["face_ids"]
    pixels = atlas_data["pixels"]

    print("[texture] tracing collapse history", flush=True)
    mapped_faces, mapped_barycentrics = successive_map(
        points,
        barycentrics,
        face_ids,
        history
    )

    uv_triangles = original_triangle_uvs[mapped_faces]
    source_uvs = np.einsum(
        "ni,nij->nj",
        mapped_barycentrics,
        uv_triangles
    )
    material_ids = original_material_ids[mapped_faces]
    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)

    for material_id in np.unique(material_ids):
        indices = np.where(material_ids == material_id)[0]

        if len(original_textures) == 1:
            texture_index = 0
        else:
            texture_index = int(material_id)

        if texture_index < 0 or texture_index >= len(original_textures):
            texture_index = 0

        colors = sample_texture_bilinear(
            original_textures[texture_index],
            source_uvs[indices]
        )
        atlas[
            pixels[indices, 0],
            pixels[indices, 1]
        ] = colors

    if atlas_mode == "grid":
        # Preserve the old per-tile border fill in legacy mode.
        nearest = None
        tile_size = atlas_data["tile_size"]
        for top, left, mask in atlas_data["masks"]:
            if not np.any(mask):
                continue
            tile = atlas[top:top + tile_size, left:left + tile_size]
            missing = ~mask
            if not np.any(missing):
                continue
            if nearest is None:
                nearest = distance_transform_edt(
                    missing, return_distances=False, return_indices=True)
            filled = tile.copy()
            filled[missing] = tile[nearest[0][missing], nearest[1][missing]]
            atlas[top:top + tile_size, left:left + tile_size] = filled
    else:
        atlas = pad_atlas(atlas, atlas_data["valid_mask"], texture_padding)

    return triangle_uvs, uv_vertices, face_uv_indices, atlas


def write_textured_obj(
    path,
    vertices,
    faces,
    triangle_uvs,
    texture,
    uv_vertices=None,
    face_uv_indices=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = path.with_suffix(".mtl")
    texture_path = path.with_name(f"{path.stem}_texture.png")

    Image.fromarray(texture).save(texture_path)

    with open(mtl_path, "w", encoding="utf-8") as file:
        file.write("newmtl material0\n")
        file.write("Ka 1.000000 1.000000 1.000000\n")
        file.write("Kd 1.000000 1.000000 1.000000\n")
        file.write("Ks 0.000000 0.000000 0.000000\n")
        file.write(f"map_Kd {texture_path.name}\n")

    if uv_vertices is None or face_uv_indices is None:
        uv_vertices = triangle_uvs.reshape(-1, 2)
        face_uv_indices = np.arange(len(faces) * 3, dtype=np.int64).reshape(-1, 3)

    with open(path, "w", encoding="utf-8") as file:
        file.write(f"mtllib {mtl_path.name}\n")

        for vertex in vertices:
            file.write(
                f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n"
            )

        for uv in uv_vertices:
            file.write(f"vt {uv[0]:.17g} {uv[1]:.17g}\n")

        file.write("usemtl material0\n")

        for fi, face in enumerate(faces):
            uv_face = face_uv_indices[fi]
            file.write(
                "f "
                f"{face[0] + 1}/{uv_face[0] + 1} "
                f"{face[1] + 1}/{uv_face[1] + 1} "
                f"{face[2] + 1}/{uv_face[2] + 1}\n"
            )



# Optional Numba acceleration for numeric nested loops. First run includes JIT time.
try:
    from numba import njit
except ImportError:
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True
    for _name in ("closest_point_on_triangle", "point_triangle_distance_sq",
                  "segment_segment_distance_sq", "segment_intersects_triangle",
                  "triangle_triangle_distance_sq", "closest_vertex_pair",
                  "virtual_edge_batch", "project_samples"):
        globals()[_name] = njit(cache=True)(globals()[_name])


def main():
    print("[startup] importing Open3D...", flush=True)
    import open3d as o3d
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Input mesh path. e.g) data/mesh.ply")
    parser.add_argument(
        "--output", help="Output mesh path. e.g) results/mesh_simplified.ply")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--virtual_radius", type=float)
    parser.add_argument("--texture")
    parser.add_argument("--texture_size", type=int, default=4096,
                        help="Baked atlas resolution. Default: 4096")
    parser.add_argument("--atlas", choices=("xatlas", "grid"), default="xatlas",
                        help="UV atlas mode. xatlas is recommended; grid is the old per-face layout.")
    parser.add_argument("--texture_padding", type=int, default=8,
                        help="Chart gutter in texels for xatlas and post-bake dilation. Default: 8")
    args = parser.parse_args()
    run_started = time.perf_counter()
    if args.texture_size < 1:
        parser.error("--texture_size must be positive")
    if args.texture_padding < 0:
        parser.error("--texture_padding must be non-negative")
    if args.virtual_radius is not None and args.virtual_radius < 0:
        parser.error("--virtual_radius must be non-negative")
    if not NUMBA_AVAILABLE:
        warnings.warn(
            "Numba is not installed; use pip install numba for faster distance and texture mapping loops.")

    input_path = Path(args.input)

    if args.output is None:
        output_path = input_path.with_stem(
            f"{input_path.stem}_simplified"
        )
    else:
        output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading {input_path}", flush=True)
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    print(
        f"[load] vertices={len(mesh.vertices):,}, faces={len(mesh.triangles):,}", flush=True)

    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.triangles).copy()

    if not 0.0 < args.ratio <= 1.0:
        parser.error(
            "--ratio must be greater than 0 and less than or equal to 1")

    target_faces = max(1, int(len(faces) * args.ratio))

    original_triangle_uvs = np.asarray(mesh.triangle_uvs).copy()
    original_material_ids = np.asarray(mesh.triangle_material_ids).copy()

    if args.texture is not None:
        original_textures = [
            np.asarray(Image.open(args.texture).convert("RGB"))
        ]
    elif mesh.has_textures():
        original_textures = [
            np.asarray(texture).copy()
            for texture in mesh.textures
        ]
    else:
        original_textures = []

    has_texture = (
        len(original_triangle_uvs) == 3 * len(faces)
        and len(original_textures) > 0
    )

    if has_texture:
        original_triangle_uvs = original_triangle_uvs.reshape(-1, 3, 2)
        if len(original_material_ids) != len(faces):
            original_material_ids = np.zeros(len(faces), dtype=np.int32)

    if len(faces) > 0:
        bbox_diagonal = np.linalg.norm(
            np.max(vertices, axis=0) - np.min(vertices, axis=0)
        )
    else:
        bbox_diagonal = 0.0

    if args.virtual_radius is None:
        virtual_radius = 0.0025 * bbox_diagonal
    else:
        virtual_radius = args.virtual_radius

    simplified_vertices, simplified_faces, active_face_indices, history = simplify(
        vertices,
        faces,
        target_faces,
        virtual_radius,
        has_texture
    )

    if has_texture:
        if output_path.suffix.lower() != ".obj":
            output_path = output_path.with_suffix(".obj")

        (simplified_triangle_uvs,
         simplified_uv_vertices,
         simplified_face_uv_indices,
         simplified_texture) = bake_texture(
            simplified_vertices,
            simplified_faces,
            active_face_indices,
            history,
            original_triangle_uvs,
            original_material_ids,
            original_textures,
            args.texture_size,
            args.atlas,
            args.texture_padding
        )

        write_textured_obj(
            output_path,
            simplified_vertices,
            simplified_faces,
            simplified_triangle_uvs,
            simplified_texture,
            simplified_uv_vertices,
            simplified_face_uv_indices
        )
    else:
        simplified_mesh = o3d.geometry.TriangleMesh()
        simplified_mesh.vertices = o3d.utility.Vector3dVector(
            simplified_vertices
        )
        simplified_mesh.triangles = o3d.utility.Vector3iVector(
            simplified_faces
        )

        simplified_mesh.compute_vertex_normals()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        o3d.io.write_triangle_mesh(
            str(output_path),
            simplified_mesh
        )

    print(
        f"[done] faces={len(simplified_faces):,}, elapsed={time.perf_counter()-run_started:.1f}s, output={output_path}", flush=True)


if __name__ == "__main__":
    main()
