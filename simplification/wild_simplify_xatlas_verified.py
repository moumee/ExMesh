"""Audited Python Wild adaptation with optional Sobel face weights.

Wild baseline: memory plane Q + memoryless boundary area Q + virtual edges
and fixed-query successive mapping. Uses a rank-aware midpoint-anchored solver.
This is a Python/xatlas adaptation, NOT the authors' reference implementation.
The original supplied scripts remain unchanged for historical reproducibility.
"""
if __name__ == "__main__":
    print("[startup] wild_simplify_xatlas_verified: loading dependencies...", flush=True)

import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import heapq
import time
import warnings
import json
from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from PIL import Image


def minimize_quadric(quadric, midpoint, rcond=1e-10):
    """Minimize a quadric while keeping nullspace directions at the midpoint."""
    a = (quadric[:3, :3] + quadric[:3, :3].T) * .5
    b = quadric[:3, 3]
    if not np.all(np.isfinite(quadric)):
        raise ValueError('Non-finite quadric')
    values, vectors = np.linalg.eigh(a)
    scale = max(float(values[-1]), 0.)
    if values[0] < -max(scale * 1e-7, 1e-20):
        raise ValueError('Quadric is materially indefinite')
    valid = values > scale * rcond if scale > 0 else np.zeros(3, bool)
    if np.all(valid):
        # Keep the existing full-rank solve. Only the deficient branch changes.
        position = np.linalg.solve(a, -b)
    else:
        position = midpoint.copy()
        if np.any(valid):
            basis = vectors[:, valid]
            position -= basis @ ((basis.T @ (a @ midpoint + b)) / values[valid])
    h = np.r_[position, 1.]
    # A squared-error cost can only become slightly negative by cancellation.
    return max(float(h @ quadric @ h), 0.), position


def compute_area_quadric(edge, vertices, vertex_edges, edge_faces):
    a, b = edge
    Q_area = np.zeros((4, 4), dtype=np.float64)
    one_ring_edges = vertex_edges[a] | vertex_edges[b]

    for boundary_edge in sorted(one_ring_edges):
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

    return minimize_quadric(Q_total, (vertices[a] + vertices[b]) / 2.)


def minimize_quadrics(quadrics, midpoints, rcond=1e-10):
    """minimize_quadric for a stack of quadrics; returns (costs, positions).

    One batched eigendecomposition and solve replace a pair of LAPACK calls
    per edge. Rank-deficient rows are passed to minimize_quadric itself.
    """
    count = len(quadrics)
    costs = np.empty(count, dtype=np.float64)
    positions = np.empty((count, 3), dtype=np.float64)
    if count == 0:
        return costs, positions
    if not np.all(np.isfinite(quadrics)):
        raise ValueError('Non-finite quadric')
    upper = quadrics[:, :3, :3]
    a = (upper + upper.transpose(0, 2, 1)) * .5
    values = np.linalg.eigh(a)[0]
    scale = np.maximum(values[:, 2], 0.)
    if np.any(values[:, 0] < -np.maximum(scale * 1e-7, 1e-20)):
        raise ValueError('Quadric is materially indefinite')
    full_rank = (scale > 0) & np.all(values > (scale * rcond)[:, None], axis=1)
    if full_rank.all():
        rows = slice(None)
    else:
        rows = np.flatnonzero(full_rank)
        for row in np.flatnonzero(~full_rank):
            costs[row], positions[row] = minimize_quadric(quadrics[row], midpoints[row], rcond)
        a, quadrics = a[rows], quadrics[rows]
    if len(a):
        solved = np.linalg.solve(a, -quadrics[:, :3, 3, None])[:, :, 0]
        h = np.ones((len(a), 4), dtype=np.float64)
        h[:, :3] = solved
        value = (h[:, None, :] @ quadrics @ h[:, :, None])[:, 0, 0]
        costs[rows] = np.maximum(value, 0.)
        positions[rows] = solved
    return costs, positions


def compute_edge_costs(edges, vertices, Q, vertex_edges, edge_faces, area_cache=None):
    """compute_edge_cost for a list of edges; returns (costs, positions)."""
    ends = np.array(edges, dtype=np.int64).reshape(-1, 2)
    a, b = ends[:, 0], ends[:, 1]
    if area_cache is None:
        area = np.zeros((len(edges), 4, 4), dtype=np.float64)
        for row, edge in enumerate(edges):
            area[row] = compute_area_quadric(edge, vertices, vertex_edges, edge_faces)
    else:
        vertex_area, boundary_quadrics = area_cache
        area = vertex_area[a] + vertex_area[b]
        if boundary_quadrics:
            for row, edge in enumerate(edges):
                shared = boundary_quadrics.get(edge)
                if shared is not None:
                    area[row] -= shared  # union: shared boundary edge counted once
    return minimize_quadrics(Q[a] + Q[b] + area, (vertices[a] + vertices[b]) / 2.)


def closest_point_barycentric(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """Closest point of triangle abc to p: (squared distance, u, v, w).

    Scalar arithmetic only, so Numba compiles it without array temporaries.
    closest_points_barycentric is the same sequence of operations on arrays.
    """
    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    apx, apy, apz = px - ax, py - ay, pz - az
    nx = aby * acz - abz * acy
    ny = abz * acx - abx * acz
    nz = abx * acy - aby * acx

    if nx * nx + ny * ny + nz * nz < 1e-20:
        # Triangle이 거의 선/점이면 세 edge 중 가장 가까운 점을 사용
        best = apx * apx + apy * apy + apz * apz
        u, v, w = 1.0, 0.0, 0.0
        length = abx * abx + aby * aby + abz * abz
        if length > 1e-20:
            t = min(max((apx * abx + apy * aby + apz * abz) / length, 0.0), 1.0)
            dx, dy, dz = px - (ax + t * abx), py - (ay + t * aby), pz - (az + t * abz)
            distance = dx * dx + dy * dy + dz * dz
            if distance < best:
                best = distance
                u, v, w = 1.0 - t, t, 0.0
        length = acx * acx + acy * acy + acz * acz
        if length > 1e-20:
            t = min(max((apx * acx + apy * acy + apz * acz) / length, 0.0), 1.0)
            dx, dy, dz = px - (ax + t * acx), py - (ay + t * acy), pz - (az + t * acz)
            distance = dx * dx + dy * dy + dz * dz
            if distance < best:
                best = distance
                u, v, w = 1.0 - t, 0.0, t
        bcx, bcy, bcz = cx - bx, cy - by, cz - bz
        length = bcx * bcx + bcy * bcy + bcz * bcz
        if length > 1e-20:
            t = ((px - bx) * bcx + (py - by) * bcy + (pz - bz) * bcz) / length
            t = min(max(t, 0.0), 1.0)
            dx, dy, dz = px - (bx + t * bcx), py - (by + t * bcy), pz - (bz + t * bcz)
            distance = dx * dx + dy * dy + dz * dz
            if distance < best:
                best = distance
                u, v, w = 0.0, 1.0 - t, t
        return best, u, v, w

    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return apx * apx + apy * apy + apz * apz, 1.0, 0.0, 0.0

    bpx, bpy, bpz = px - bx, py - by, pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bpx * bpx + bpy * bpy + bpz * bpz, 0.0, 1.0, 0.0

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        dx, dy, dz = px - (ax + v * abx), py - (ay + v * aby), pz - (az + v * abz)
        return dx * dx + dy * dy + dz * dz, 1.0 - v, v, 0.0

    cpx, cpy, cpz = px - cx, py - cy, pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cpx * cpx + cpy * cpy + cpz * cpz, 0.0, 0.0, 1.0

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        dx, dy, dz = px - (ax + w * acx), py - (ay + w * acy), pz - (az + w * acz)
        return dx * dx + dy * dy + dz * dz, 1.0 - w, 0.0, w

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        dx = px - (bx + w * (cx - bx))
        dy = py - (by + w * (cy - by))
        dz = pz - (bz + w * (cz - bz))
        return dx * dx + dy * dy + dz * dz, 0.0, 1.0 - w, w

    denom = va + vb + vc
    if abs(denom) < 1e-15:
        to_a = apx * apx + apy * apy + apz * apz
        to_b = bpx * bpx + bpy * bpy + bpz * bpz
        to_c = cpx * cpx + cpy * cpy + cpz * cpz
        if to_a <= to_b and to_a <= to_c:
            return to_a, 1.0, 0.0, 0.0
        if to_b <= to_c:
            return to_b, 0.0, 1.0, 0.0
        return to_c, 0.0, 0.0, 1.0

    inv_denom = 1.0 / denom
    v = vb * inv_denom
    w = vc * inv_denom
    u = 1.0 - v - w
    dx = px - (u * ax + v * bx + w * cx)
    dy = py - (u * ay + v * by + w * cy)
    dz = pz - (u * az + v * bz + w * cz)
    return dx * dx + dy * dy + dz * dz, u, v, w


def closest_points_barycentric(points, triangles):
    """closest_point_barycentric for every (point, triangle) pair, with NumPy.

    points (m, 3), triangles (k, 3, 3) -> squared distances (m, k) and
    barycentrics (m, k, 3). The branches are applied from last to first, so
    each pair ends with the first branch the scalar version would take.
    """
    px, py, pz = (points[:, None, axis] for axis in range(3))
    ax, ay, az = (triangles[None, :, 0, axis] for axis in range(3))
    bx, by, bz = (triangles[None, :, 1, axis] for axis in range(3))
    cx, cy, cz = (triangles[None, :, 2, axis] for axis in range(3))
    zero = np.zeros(np.broadcast_shapes(px.shape, ax.shape), dtype=np.float64)
    one = zero + 1.0

    def squared(dx, dy, dz):
        return dx * dx + dy * dy + dz * dz

    def choose(mask, distance, u, v, w):
        nonlocal best, bu, bv, bw
        best = np.where(mask, distance, best)
        bu, bv, bw = np.where(mask, u, bu), np.where(mask, v, bv), np.where(mask, w, bw)

    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    bcx, bcy, bcz = cx - bx, cy - by, cz - bz
    apx, apy, apz = px - ax, py - ay, pz - az
    bpx, bpy, bpz = px - bx, py - by, pz - bz
    cpx, cpy, cpz = px - cx, py - cy, pz - cz
    to_a, to_b, to_c = squared(apx, apy, apz), squared(bpx, bpy, bpz), squared(cpx, cpy, cpz)
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_denom = 1.0 / denom
        v = vb * inv_denom
        w = vc * inv_denom
        u = 1.0 - v - w
        best = squared(px - (u * ax + v * bx + w * cx),
                       py - (u * ay + v * by + w * cy),
                       pz - (u * az + v * bz + w * cz))
        bu, bv, bw = u + zero, v + zero, w + zero

        flat = np.abs(denom) < 1e-15
        near_a = (to_a <= to_b) & (to_a <= to_c)
        choose(flat, to_c, zero, zero, one)
        choose(flat & ~near_a & (to_b <= to_c), to_b, zero, one, zero)
        choose(flat & near_a, to_a, one, zero, zero)

        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        choose((va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0),
               squared(px - (bx + w * bcx), py - (by + w * bcy), pz - (bz + w * bcz)),
               zero, 1.0 - w, w)
        w = d2 / (d2 - d6)
        choose((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),
               squared(px - (ax + w * acx), py - (ay + w * acy), pz - (az + w * acz)),
               1.0 - w, zero, w)
        choose((d6 >= 0.0) & (d5 <= d6), to_c, zero, zero, one)
        v = d1 / (d1 - d3)
        choose((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),
               squared(px - (ax + v * abx), py - (ay + v * aby), pz - (az + v * abz)),
               1.0 - v, v, zero)
        choose((d3 >= 0.0) & (d4 <= d3), to_b, zero, one, zero)
        choose((d1 <= 0.0) & (d2 <= 0.0), to_a, one, zero, zero)

        nx = aby * acz - abz * acy
        ny = abz * acx - abx * acz
        nz = abx * acy - aby * acx
        sliver = np.broadcast_to(squared(nx, ny, nz) < 1e-20, zero.shape)
        if np.any(sliver):
            # Triangle이 거의 선/점이면 세 edge 중 가장 가까운 점을 사용
            def on_edge(ox, oy, oz, ex, ey, ez, rx, ry, rz):
                """Clamped projection on the edge o + t * e, where r = p - o."""
                length = squared(ex, ey, ez)
                t = np.minimum(np.maximum((rx * ex + ry * ey + rz * ez) / length, 0.0), 1.0)
                distance = squared(px - (ox + t * ex), py - (oy + t * ey), pz - (oz + t * ez))
                return sliver & (length > 1e-20) & (distance < best), distance, t

            choose(sliver, to_a, one, zero, zero)
            closer, distance, t = on_edge(ax, ay, az, abx, aby, abz, apx, apy, apz)
            choose(closer, distance, 1.0 - t, t, zero)
            closer, distance, t = on_edge(ax, ay, az, acx, acy, acz, apx, apy, apz)
            choose(closer, distance, 1.0 - t, zero, t)
            closer, distance, t = on_edge(bx, by, bz, bcx, bcy, bcz, bpx, bpy, bpz)
            choose(closer, distance, zero, 1.0 - t, t)
    return best, np.stack((bu, bv, bw), axis=-1)


def point_triangle_distance_sq(point, triangle):
    return closest_point_barycentric(
        point[0], point[1], point[2],
        triangle[0, 0], triangle[0, 1], triangle[0, 2],
        triangle[1, 0], triangle[1, 1], triangle[1, 2],
        triangle[2, 0], triangle[2, 1], triangle[2, 2])[0]


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
    """Component label of every face (faces that share a vertex are connected)."""
    if len(faces) == 0:
        return np.empty(0, dtype=np.int64)
    rows = np.concatenate((faces[:, 0], faces[:, 1]))
    cols = np.concatenate((faces[:, 1], faces[:, 2]))
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                       shape=(len(vertices), len(vertices)))
    labels = csgraph.connected_components(graph, directed=False)[1]
    return labels[faces[:, 0]].astype(np.int64)


def near_other_component(bounds_min, bounds_max, face_components, threshold_sq):
    """Faces whose box is closer than the threshold to the box of another component.

    A face box lies inside its component box, so a pair of faces that passes
    the box-gap test of build_virtual_edges always has both faces marked here.
    The other faces cannot get a virtual edge and skip the neighbor query.
    """
    labels, component = np.unique(face_components, return_inverse=True)
    near = np.zeros(len(component), dtype=bool)
    if len(labels) > 128:
        near[:] = True  # one pass per component would cost more than it saves
        return near
    for index in range(len(labels)):
        members = component == index
        low = bounds_min[members].min(axis=0)
        high = bounds_max[members].max(axis=0)
        gap = np.maximum(0.0, np.maximum(low - bounds_max, bounds_min - high))
        near |= ~members & (np.einsum("ij,ij->i", gap, gap) < threshold_sq)
    return near


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

    scan = np.flatnonzero(near_other_component(
        bounds_min, bounds_max, face_components, threshold_sq))
    print(f"[virtual edges] {len(scan):,} faces lie near another component", flush=True)
    if len(scan) == 0:
        return virtual_edges

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

    for i in scan.tolist():
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


def fibonacci_sphere(count):
    """Near-uniform unit directions on the full sphere."""
    i = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def face_visibility(vertices, faces, directions=128):
    """Per-face exterior visibility in [0, 1] (visibility-guided QEM).

    Cosine-weighted fraction of sphere directions from which a face is not
    occluded by the mesh itself, i.e. 1 - ambient occlusion, following
    Zhang & Turk, "Visibility-Guided Simplification" (IEEE Vis 2002). Rays
    start just above the face on the side facing each direction; the larger
    of the two sides is kept because reconstructed surfaces are often open,
    one-sided sheets. Zero-area faces get 1 (they add no quadric anyway).
    """
    import open3d as o3d
    triangles = vertices[faces]
    centroids = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid, None]
    offset = 1e-4 * np.linalg.norm(np.ptp(vertices, axis=0))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                        o3d.core.Tensor(faces.astype(np.uint32)))
    seen = np.zeros((2, len(faces)))
    total = np.zeros((2, len(faces)))
    started = time.perf_counter()
    for direction in fibonacci_sphere(directions):
        cosine = normals @ direction
        side = np.where(cosine >= 0, 1.0, -1.0)
        origins = centroids + (side * offset)[:, None] * normals
        rays = np.hstack((origins, np.broadcast_to(direction, origins.shape))).astype(np.float32)
        escaped = np.isinf(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy())
        index = (cosine < 0).astype(np.int64)
        weight = np.abs(cosine)
        np.add.at(total, (index, np.arange(len(faces))), weight)
        np.add.at(seen, (index, np.arange(len(faces))), weight * escaped)
    visibility = np.max(np.divide(seen, total, out=np.zeros_like(seen), where=total > 0), axis=0)
    visibility[~valid] = 1.0
    print(f"[visibility] {directions} directions, mean={visibility.mean():.3f}, "
          f"faces<0.05: {np.mean(visibility < 0.05):.1%}, "
          f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return visibility


def photometric_face_weights(vertices, faces, density, lam=1.0):
    """Face weights for the multi-view photometric parallax quadric.

    density = W_f / A_f from photometric_quadric_weights.py, where
    W_f = sum over capture views and covered pixels of ||grad I . J n_f||^2.
    Weight 1 + lam * density / mean, mean = sum W / sum A, so lam = 1 gives the
    photometric term the same total energy as the geometric area term.
    """
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    mean = np.sum(density * area) / np.sum(area)
    if not mean > 0:
        raise ValueError('Photometric energy is zero; check cameras and mesh alignment')
    return 1.0 + lam * density / mean


def collapse_normal_diagnostics(vertices, local_faces, a, b, position):
    """Count inversions and numerical degeneracies among surviving local faces.

    Removed and already zero-area faces do not constrain placement. This local
    orientation check does not guarantee intersection-free output or bound
    cumulative normal changes over multiple collapses.
    """
    flips = 0
    degeneracies = 0
    px, py, pz = position[0], position[1], position[2]
    for index in range(len(local_faces)):
        i, j, k = local_faces[index, 0], local_faces[index, 1], local_faces[index, 2]
        move_i = i == a or i == b
        move_j = j == a or j == b
        move_k = k == a or k == b
        if (i == a or j == a or k == a) and (i == b or j == b or k == b):
            continue
        x0, y0, z0 = vertices[i, 0], vertices[i, 1], vertices[i, 2]
        x1, y1, z1 = vertices[j, 0], vertices[j, 1], vertices[j, 2]
        x2, y2, z2 = vertices[k, 0], vertices[k, 1], vertices[k, 2]
        ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
        vx, vy, vz = x2 - x0, y2 - y0, z2 - z0
        old_x, old_y, old_z = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        old_sq = old_x * old_x + old_y * old_y + old_z * old_z
        if old_sq == 0:
            continue
        if move_i:
            x0, y0, z0 = px, py, pz
        if move_j:
            x1, y1, z1 = px, py, pz
        if move_k:
            x2, y2, z2 = px, py, pz
        ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
        vx, vy, vz = x2 - x0, y2 - y0, z2 - z0
        new_x, new_y, new_z = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        if new_x * new_x + new_y * new_y + new_z * new_z <= old_sq * 1e-24:
            degeneracies += 1
        elif old_x * new_x + old_y * new_y + old_z * new_z <= 0:
            flips += 1
    return flips, degeneracies


def simplify(vertices, faces, target_faces, virtual_radius, track_history, face_weights,
             *, normal_guard=False, diagnostics=None, reuse_costs=True):
    """Collapse edges until target_faces remain.

    reuse_costs=False recomputes the cost of every edge around a collapse, as
    the audited reference loop does. The default keeps the cost of edges whose
    inputs (endpoint quadrics, positions, boundary terms) did not change; both
    produce the same collapse sequence.
    """

    stats = dict(checked=0, rejected=0, accepted=0, accepted_flip_events=0,
                 accepted_degenerate_events=0)

    # The Q matrix from QEM for each vertex
    Q = np.zeros((len(vertices), 4, 4), dtype=np.float64)
    # Holds a list of face indices for each vertex
    vertex_faces = [set() for _ in range(len(vertices))]
    # Holds a list of edge for each vertex
    vertex_edges = [set() for _ in range(len(vertices))]
    # Dictionary with edge tuple key and set of face indices as value
    edge_faces = defaultdict(set)
    # Version dict for preventing using old versions from heap pop.
    # It changes when an edge is removed or gets a new cost.
    edge_version = defaultdict(int)

    # Plain lists: they are read several times per edge in the collapse loop.
    vertex_active = [True] * len(vertices)
    face_active = [True] * len(faces)

    init_started = time.perf_counter()
    print("[initialize] building mesh adjacency...", flush=True)
    for f_idx, (i, j, k) in enumerate(faces.tolist()):
        if f_idx and f_idx % 25000 == 0:
            print(f"[initialize] faces={f_idx:,}/{len(faces):,}, "
                  f"elapsed={time.perf_counter()-init_started:.1f}s", flush=True)

        vertex_faces[i].add(f_idx)
        vertex_faces[j].add(f_idx)
        vertex_faces[k].add(f_idx)

        for edge in ((i, j) if i < j else (j, i),
                     (j, k) if j < k else (k, j),
                     (k, i) if k < i else (i, k)):
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
    quadrics = np.einsum(
        "ni,nj,n->nij",
        planes,
        planes,
        (lengths / 6.0) * face_weights,
    )
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
    # Vertices whose vertex_area currently holds boundary terms.
    area_vertices = set()

    def refresh_area(vertex_ids):
        """Rebuild current boundary contributions, never accumulate old area terms.

        Returns the vertices with a changed boundary term: a new area quadric,
        or an incident edge whose boundary quadric appeared, moved or vanished.
        """
        changed = set()
        touched_edges = set()
        for v in vertex_ids:
            touched_edges.update(vertex_edges[v])
        for e in touched_edges:
            if len(edge_faces[e]) == 1:
                # A collapse drops the quadrics of every edge at the vertex it
                # moves, so a quadric that is still here has unmoved endpoints.
                if e in boundary_quadrics:
                    continue
                u, v = e
                s = vertices[v] - vertices[u]
                t = np.cross(vertices[u], vertices[v])
                S = cross_matrix(s)
                q = np.empty((4, 4), dtype=np.float64)
                q[:3, :3] = 0.5 * (S.T @ S)
                q[:3, 3] = q[3, :3] = -0.5 * (S @ t)
                q[3, 3] = 0.5 * np.dot(t, t)
                boundary_quadrics[e] = q
                changed.update(e)
            elif boundary_quadrics.pop(e, None) is not None:
                changed.update(e)
        if not boundary_quadrics and not area_vertices:
            return changed  # closed surface: all terms are zero
        for v in vertex_ids:
            terms = [e for e in vertex_edges[v] if e in boundary_quadrics]
            if not terms and v not in area_vertices:
                continue
            total = np.zeros((4, 4), dtype=np.float64)
            for e in sorted(terms):
                total += boundary_quadrics[e]
            if not np.array_equal(total, vertex_area[v]):
                vertex_area[v] = total
                changed.add(v)
            if terms:
                area_vertices.add(v)
            else:
                area_vertices.discard(v)
        return changed

    refresh_area(range(len(vertices)))
    area_cache = (vertex_area, boundary_quadrics)
    started = time.perf_counter()
    last_report = started
    collapse_count = 0

    # Min heap for getting the lowest cost edge
    heap = []

    # Min heap tie-breaker counter
    counter = 0

    # Current (cost, position) of every edge, valid while its inputs are unchanged.
    edge_cost = {}
    # Tie-breaker each queued edge has in the reference loop, which pushes every
    # edge around a collapse again. An edge that keeps its cost only gets its
    # number updated here; its heap entry moves when it reaches the top.
    edge_order = {}

    print(f"[heap] initializing {len(edge_faces):,} edge costs", flush=True)
    initial_edges = sorted(edge_faces)
    for begin in range(0, len(initial_edges), 25000):
        if begin:
            print(
                f"[heap] costs={begin:,}/{len(edge_faces):,}", flush=True)
        chunk = initial_edges[begin:begin + 25000]
        costs, positions = compute_edge_costs(
            chunk,
            vertices,
            Q,
            vertex_edges,
            edge_faces,
            area_cache
        )
        for edge, cost, position in zip(chunk, costs.tolist(), positions):
            edge_cost[edge] = (cost, position)
            edge_order[edge] = counter
            heap.append((cost, counter, edge, position, edge_version[edge]))
            counter += 1
    del initial_edges

    heapq.heapify(heap)
    active_face_count = len(faces)
    history = []

    while active_face_count > target_faces and heap:
        cost, order, edge, position, version = heapq.heappop(heap)
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
        # Requeued since this entry was pushed: it belongs further back.
        if order != edge_order[edge]:
            heapq.heappush(heap, (cost, edge_order[edge], edge, position, version))
            continue

        old_face_ids = sorted(
            fi for fi in (vertex_faces[a] | vertex_faces[b])
            if face_active[fi]
        )
        pre_face_ids = np.array(old_face_ids, dtype=np.int64)
        pre_faces = faces[pre_face_ids]
        if normal_guard or diagnostics is not None:
            flip_count, degenerate_count = collapse_normal_diagnostics(
                vertices, pre_faces, a, b, position)
            stats['checked'] += 1
            if normal_guard and (flip_count or degenerate_count):
                # Neighboring collapses invalidate/requeue this edge normally.
                stats['rejected'] += 1
                del edge_order[edge]
                continue
            stats['accepted'] += 1
            stats['accepted_flip_events'] += int(flip_count)
            stats['accepted_degenerate_events'] += int(degenerate_count)
        old_edges = vertex_edges[a] | vertex_edges[b]
        affected_vertices = {a, b}
        # Vertex ids of the old faces; updated in place as the faces change.
        face_rows = pre_faces.tolist()

        for row in face_rows:
            affected_vertices.update(row)

        for e in old_edges:
            affected_vertices.update(e)

        if track_history:
            pre_triangles = vertices[pre_faces].astype(np.float32)

        for fi, (i, j, k) in zip(old_face_ids, face_rows):
            vertex_faces[i].discard(fi)
            vertex_faces[j].discard(fi)
            vertex_faces[k].discard(fi)

            for e in ((i, j) if i < j else (j, i),
                      (j, k) if j < k else (k, j),
                      (k, i) if k < i else (i, k)):
                if e in edge_faces:
                    edge_faces[e].discard(fi)

        for e in old_edges:
            boundary_quadrics.pop(e, None)
            edge_cost.pop(e, None)
            edge_order.pop(e, None)
            edge_version[e] += 1

            if e in edge_faces:
                del edge_faces[e]

            u, v = e
            vertex_edges[u].discard(e)
            vertex_edges[v].discard(e)

        vertices[a] = position
        Q[a] += Q[b]
        vertex_active[b] = False

        unique_faces = set()
        surviving = []

        for fi, row in zip(old_face_ids, face_rows):
            # (a,b,c) -> (a,a,c) degenerate face
            if a in row and b in row:
                face_active[fi] = False
                active_face_count -= 1
                continue

            # b -> a
            if b in row:
                row[:] = [a if v == b else v for v in row]
                faces[fi] = row

            key = tuple(sorted(row))
            if key[0] == key[1] or key[1] == key[2] or key in unique_faces:
                face_active[fi] = False
                active_face_count -= 1
            else:
                unique_faces.add(key)
                surviving.append((fi, row))

        for fi, (i, j, k) in surviving:
            vertex_faces[i].add(fi)
            vertex_faces[j].add(fi)
            vertex_faces[k].add(fi)

            for e in ((i, j) if i < j else (j, i),
                      (j, k) if j < k else (k, j),
                      (k, i) if k < i else (i, k)):
                u, v = e
                edge_faces[e].add(fi)
                vertex_edges[u].add(e)
                vertex_edges[v].add(e)

        for u, v in old_edges:
            if u == b:
                u = a
            if v == b:
                v = a
            if u == v:
                continue

            new_edge = (u, v) if u < v else (v, u)
            edge_faces[new_edge]
            vertex_edges[u].add(new_edge)
            vertex_edges[v].add(new_edge)

        vertex_faces[b].clear()
        vertex_edges[b].clear()

        if track_history:
            history.append((
                pre_face_ids,
                pre_triangles,
                np.array(sorted(vertex_faces[a]), dtype=np.int64)
            ))

        changed_vertices = refresh_area(affected_vertices)
        affected_edges = set()

        for v in affected_vertices:
            if vertex_active[v]:
                affected_edges.update(vertex_edges[v])

        requeued = []
        stale = []

        for new_edge in sorted(affected_edges):
            u, v = new_edge

            if not vertex_active[u] or not vertex_active[v]:
                continue
            if new_edge not in edge_faces:
                continue

            requeued.append(new_edge)
            # Edges at the collapsed vertex are new and have no stored cost.
            if (not reuse_costs or new_edge not in edge_cost
                    or u in changed_vertices or v in changed_vertices):
                stale.append(new_edge)

        if stale:
            costs, positions = compute_edge_costs(
                stale,
                vertices,
                Q,
                vertex_edges,
                edge_faces,
                area_cache
            )
            for new_edge, new_cost, new_position in zip(stale, costs.tolist(), positions):
                edge_cost[new_edge] = (new_cost, new_position)
                edge_version[new_edge] += 1
                # The entry with the old cost is void; a new one is pushed below.
                edge_order.pop(new_edge, None)

        for new_edge in requeued:
            if new_edge not in edge_order:
                new_cost, new_position = edge_cost[new_edge]
                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        counter,
                        new_edge,
                        new_position,
                        edge_version[new_edge]
                    )
                )
            edge_order[new_edge] = counter
            counter += 1

        collapse_count += 1
        # Lazy invalidation needs bounded storage on large meshes.
        if len(heap) > max(10000, 4 * len(edge_faces)):
            # Most stale entries fail the version test, so it goes first.
            heap = [item for item in heap
                    if item[4] == edge_version[item[2]]
                    and item[2] in edge_faces
                    and vertex_active[item[2][0]] and vertex_active[item[2][1]]]
            heapq.heapify(heap)
        if collapse_count % 1000 == 0:
            now = time.perf_counter()
            if now - last_report >= 5.0:
                print(f"[simplify] faces={active_face_count:,} target={target_faces:,} "
                      f"collapses={collapse_count:,} elapsed={now-started:.1f}s", flush=True)
                last_report = now

    active_vertex_indices = np.flatnonzero(vertex_active)

    old_to_new = -np.ones(len(vertices), dtype=np.int64)
    old_to_new[active_vertex_indices] = np.arange(
        len(active_vertex_indices)
    )

    new_vertices = vertices[active_vertex_indices]

    active_face_indices = np.flatnonzero(face_active)
    new_faces = old_to_new[faces[active_face_indices]]

    if diagnostics is not None:
        diagnostics.update(stats, target_faces=int(target_faces), actual_faces=int(len(new_faces)))
    if normal_guard:
        print(f"[normal guard] {stats}", flush=True)

    return new_vertices, new_faces, active_face_indices, history


def cross_matrix(v):
    x, y, z = v

    matrix = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ], dtype=np.float64)

    return matrix


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
    # Written out (no BLAS) so rasterize_uv_triangles computes the same bits.
    d00 = v0[0] * v0[0] + v0[1] * v0[1]
    d01 = v0[0] * v1[0] + v0[1] * v1[1]
    d11 = v1[0] * v1[0] + v1[1] * v1[1]
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return np.empty((0, 3), dtype=np.float64)
    d20 = v2[:, 0] * v0[0] + v2[:, 1] * v0[1]
    d21 = v2[:, 0] * v1[0] + v2[:, 1] * v1[1]
    beta = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    alpha = 1.0 - beta - gamma
    return np.column_stack((alpha, beta, gamma))


def rasterize_uv_triangle(triangle_uv, texture_size, supersample=1):
    """Rasterize texel centers covered by one normalized UV triangle.

    supersample > 1 returns a stratified KxK sub-texel grid per texel instead
    of the center; rows/cols then repeat once per covered sub-sample.
    """
    u_min = max(0.0, float(np.min(triangle_uv[:, 0])))
    u_max = min(1.0, float(np.max(triangle_uv[:, 0])))
    v_min = max(0.0, float(np.min(triangle_uv[:, 1])))
    v_max = min(1.0, float(np.max(triangle_uv[:, 1])))

    # Sub-samples reach half a texel further than centers; widen the box.
    reach = 0.5 if supersample == 1 else 1.0
    x0 = max(0, int(np.floor(u_min * texture_size - reach)))
    x1 = min(texture_size - 1, int(np.ceil(u_max * texture_size - 1.0 + reach)))
    y0 = max(0, int(np.floor((1.0 - v_max) * texture_size - reach)))
    y1 = min(texture_size - 1, int(np.ceil((1.0 - v_min) * texture_size - 1.0 + reach)))

    if x1 < x0 or y1 < y0:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.float64))

    xs = np.arange(x0, x1 + 1, dtype=np.int32)
    ys = np.arange(y0, y1 + 1, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    if supersample == 1:
        sample_uv = np.column_stack((
            (xx.ravel().astype(np.float64) + 0.5) / texture_size,
            1.0 - (yy.ravel().astype(np.float64) + 0.5) / texture_size,
        ))
    else:
        offsets = (np.arange(supersample, dtype=np.float64) + 0.5) / supersample
        oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
        yy = np.repeat(yy.ravel(), supersample * supersample)
        xx = np.repeat(xx.ravel(), supersample * supersample)
        count = len(xx) // (supersample * supersample)
        sample_uv = np.column_stack((
            (xx.astype(np.float64) + np.tile(ox.ravel(), count)) / texture_size,
            1.0 - (yy.astype(np.float64) + np.tile(oy.ravel(), count)) / texture_size,
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


def rasterize_uv_triangles(triangle_uvs, corners, texture_size, supersample,
                           pixels, barycentrics, points):
    """rasterize_uv_triangle for every triangle in one compiled loop.

    Returns the sample count per triangle. Samples are written in triangle
    order while they fit, so the caller runs it once with empty outputs to
    size them and once more to fill them. corners holds the 3D triangle
    corners; points receives the 3D position of every sample.
    """
    counts = np.zeros(len(triangle_uvs), dtype=np.int64)
    capacity = len(pixels)
    reach = 0.5 if supersample == 1 else 1.0
    eps = 1e-8
    total = 0
    for face in range(len(triangle_uvs)):
        au, av = triangle_uvs[face, 0, 0], triangle_uvs[face, 0, 1]
        bu, bv = triangle_uvs[face, 1, 0], triangle_uvs[face, 1, 1]
        cu, cv = triangle_uvs[face, 2, 0], triangle_uvs[face, 2, 1]
        u_min = max(0.0, min(au, bu, cu))
        u_max = min(1.0, max(au, bu, cu))
        v_min = max(0.0, min(av, bv, cv))
        v_max = min(1.0, max(av, bv, cv))
        x0 = max(0, int(np.floor(u_min * texture_size - reach)))
        x1 = min(texture_size - 1, int(np.ceil(u_max * texture_size - 1.0 + reach)))
        y0 = max(0, int(np.floor((1.0 - v_max) * texture_size - reach)))
        y1 = min(texture_size - 1, int(np.ceil((1.0 - v_min) * texture_size - 1.0 + reach)))
        v0u, v0v = bu - au, bv - av
        v1u, v1v = cu - au, cv - av
        d00 = v0u * v0u + v0v * v0v
        d01 = v0u * v1u + v0v * v1v
        d11 = v1u * v1u + v1v * v1v
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-20:
            continue
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                for sub_y in range(supersample):
                    for sub_x in range(supersample):
                        if supersample == 1:
                            su = (x + 0.5) / texture_size
                            sv = 1.0 - (y + 0.5) / texture_size
                        else:
                            su = (x + (sub_x + 0.5) / supersample) / texture_size
                            sv = 1.0 - (y + (sub_y + 0.5) / supersample) / texture_size
                        v2u, v2v = su - au, sv - av
                        d20 = v2u * v0u + v2v * v0v
                        d21 = v2u * v1u + v2v * v1v
                        beta = (d11 * d20 - d01 * d21) / denom
                        gamma = (d00 * d21 - d01 * d20) / denom
                        alpha = 1.0 - beta - gamma
                        if not (alpha >= -eps and beta >= -eps and gamma >= -eps
                                and alpha <= 1.0 + eps and beta <= 1.0 + eps
                                and gamma <= 1.0 + eps):
                            continue
                        if total < capacity:
                            pixels[total, 0] = y
                            pixels[total, 1] = x
                            barycentrics[total, 0] = alpha
                            barycentrics[total, 1] = beta
                            barycentrics[total, 2] = gamma
                            for axis in range(3):
                                points[total, axis] = (alpha * corners[face, 0, axis]
                                                       + beta * corners[face, 1, axis]
                                                       + gamma * corners[face, 2, axis])
                        total += 1
                        counts[face] += 1
    return counts


def match_atlas_faces(faces, vmapping, atlas_indices):
    """UV vertex index of every corner of every input face.

    xatlas normally returns the triangles in input order with the corners
    unchanged; then the atlas indices are the answer. Otherwise (or when two
    faces share a vertex set, where the pairing below is order dependent) the
    triangles are matched one by one through their vertex ids.
    """
    ordered = np.sort(faces, axis=1)
    if (np.array_equal(vmapping[atlas_indices], faces)
            and np.all(ordered[:, 0] != ordered[:, 1]) and np.all(ordered[:, 1] != ordered[:, 2])
            and len(np.unique(ordered, axis=0)) == len(faces)):
        return atlas_indices.copy()

    # Match xatlas output triangles back to the simplified input face IDs.
    # This avoids relying on xatlas preserving face order.
    face_lookup = defaultdict(list)
    for fi, face in enumerate(faces):
        face_lookup[tuple(sorted(int(v) for v in face))].append(fi)

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
                raise RuntimeError(
                    "Ambiguous xatlas corner mapping encountered.")
            face_uv_indices[source_fi, source_corner] = int(uv_face[int(matches[0])])

    return face_uv_indices


def build_xatlas_samples(vertices, faces, active_face_indices, texture_size, padding, supersample=1):
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

    face_uv_indices = match_atlas_faces(faces, vmapping, atlas_indices)
    triangle_uvs = uv_vertices[face_uv_indices]

    if not np.all(np.isfinite(triangle_uvs)):
        raise RuntimeError("xatlas produced non-finite UV coordinates.")

    # Some bindings may expose pixel-space UVs. Normalize if necessary.
    uv_max = float(np.max(uv_vertices)) if len(uv_vertices) else 1.0
    if uv_max > 1.5:
        uv_vertices = uv_vertices / float(texture_size)
        triangle_uvs = triangle_uvs / float(texture_size)

    print(
        f"[atlas:xatlas] rasterizing {len(faces):,} UV triangles...", flush=True)
    corners = np.ascontiguousarray(vertices[faces], dtype=np.float64)
    pixels = np.empty((0, 2), dtype=np.int32)
    barycentrics = points = np.empty((0, 3), dtype=np.float64)
    if NUMBA_AVAILABLE:
        triangle_uvs = np.ascontiguousarray(triangle_uvs)
        # Counting pass with empty outputs, then the pass that fills them.
        counts = rasterize_uv_triangles(triangle_uvs, corners, texture_size, supersample,
                                        pixels, barycentrics, points)
        pixels = np.empty((int(counts.sum()), 2), dtype=np.int32)
        barycentrics = np.empty((len(pixels), 3), dtype=np.float64)
        points = np.empty((len(pixels), 3), dtype=np.float64)
        rasterize_uv_triangles(triangle_uvs, corners, texture_size, supersample,
                               pixels, barycentrics, points)
    else:
        point_parts = []
        bary_parts = []
        pixel_parts = []
        counts = np.zeros(len(faces), dtype=np.int64)
        for fi, triangle_uv in enumerate(triangle_uvs):
            rows, cols, bary = rasterize_uv_triangle(triangle_uv, texture_size, supersample)
            counts[fi] = len(rows)
            if len(rows) == 0:
                continue

            point_parts.append(bary[:, 0:1] * corners[fi, 0] + bary[:, 1:2] * corners[fi, 1]
                               + bary[:, 2:3] * corners[fi, 2])
            bary_parts.append(bary)
            pixel_parts.append(np.column_stack(
                (rows, cols)).astype(np.int32, copy=False))

            if fi and fi % 5000 == 0:
                print(f"[atlas:xatlas] faces={fi:,}/{len(faces):,}, "
                      f"samples={int(counts.sum()):,}", flush=True)

        if point_parts:
            points = np.concatenate(point_parts, axis=0)
            barycentrics = np.concatenate(bary_parts, axis=0)
            pixels = np.concatenate(pixel_parts, axis=0)

    if len(pixels) == 0:
        raise RuntimeError("xatlas rasterization produced no texture samples.")

    face_ids = np.repeat(np.asarray(active_face_indices, dtype=np.int64), counts)
    valid_mask = np.zeros((texture_size, texture_size), dtype=bool)
    valid_mask[pixels[:, 0], pixels[:, 1]] = True

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


def trace_history(points, faces, barycentrics, pre_start, pre_faces, pre_triangles,
                  post_start, post_faces, head, link):
    """Compiled loop of successive_map; updates faces and barycentrics in place.

    head[face] and link[sample] are linked lists of the samples on each face.
    Step i of the history owns pre_faces[pre_start[i]:pre_start[i + 1]] (with
    their triangles) and post_faces[post_start[i]:post_start[i + 1]].
    """
    for sample in range(len(points)):
        face = faces[sample]
        link[sample] = head[face]
        head[face] = sample

    for step in range(len(pre_start) - 2, -1, -1):
        chain = -1
        for index in range(post_start[step], post_start[step + 1]):
            face = post_faces[index]
            sample = head[face]
            head[face] = -1
            while sample >= 0:
                following = link[sample]
                link[sample] = chain
                chain = sample
                sample = following

        first, last = pre_start[step], pre_start[step + 1]
        if first == last:
            continue

        while chain >= 0:
            sample = chain
            chain = link[sample]
            px, py, pz = points[sample, 0], points[sample, 1], points[sample, 2]
            best_distance = np.inf
            for index in range(first, last):
                distance, u, v, w = closest_point_barycentric(
                    px, py, pz,
                    np.float64(pre_triangles[index, 0, 0]),
                    np.float64(pre_triangles[index, 0, 1]),
                    np.float64(pre_triangles[index, 0, 2]),
                    np.float64(pre_triangles[index, 1, 0]),
                    np.float64(pre_triangles[index, 1, 1]),
                    np.float64(pre_triangles[index, 1, 2]),
                    np.float64(pre_triangles[index, 2, 0]),
                    np.float64(pre_triangles[index, 2, 1]),
                    np.float64(pre_triangles[index, 2, 2]))
                if distance < best_distance:
                    best_distance = distance
                    faces[sample] = pre_faces[index]
                    barycentrics[sample, 0] = u
                    barycentrics[sample, 1] = v
                    barycentrics[sample, 2] = w
            face = faces[sample]
            link[sample] = head[face]
            head[face] = sample


def successive_map(points, barycentrics, face_ids, history):
    current_faces = face_ids.astype(np.int64)
    current_barycentrics = barycentrics.astype(np.float64)

    if not history or len(points) == 0:
        return current_faces, current_barycentrics

    if NUMBA_AVAILABLE:
        pre_start = np.zeros(len(history) + 1, dtype=np.int64)
        post_start = np.zeros(len(history) + 1, dtype=np.int64)
        np.cumsum([len(step[0]) for step in history], out=pre_start[1:])
        np.cumsum([len(step[2]) for step in history], out=post_start[1:])
        pre_faces = np.concatenate([step[0] for step in history]).astype(np.int64, copy=False)
        post_faces = np.concatenate([step[2] for step in history]).astype(np.int64, copy=False)
        pre_triangles = np.concatenate([step[1] for step in history])
        face_count = 1 + max(int(current_faces.max()), int(pre_faces.max(initial=0)),
                             int(post_faces.max(initial=0)))
        trace_history(
            np.ascontiguousarray(points, dtype=np.float64), current_faces,
            current_barycentrics, pre_start, pre_faces, pre_triangles,
            post_start, post_faces, np.full(face_count, -1, dtype=np.int64),
            np.empty(len(points), dtype=np.int64))
        return current_faces, current_barycentrics

    # NumPy fallback: the same walk, one vectorized projection per history step.
    face_samples = defaultdict(list)

    for sample_index, face_id in enumerate(current_faces.tolist()):
        face_samples[face_id].append(sample_index)

    for pre_face_ids, pre_triangles, post_face_ids in reversed(history):
        affected_samples = []

        for face_id in post_face_ids.tolist():
            sample_indices = face_samples.pop(face_id, None)

            if sample_indices:
                affected_samples.extend(sample_indices)

        if not affected_samples or len(pre_face_ids) == 0:
            continue

        sample_indices = np.asarray(affected_samples, dtype=np.int64)
        distances, bary = closest_points_barycentric(
            points[sample_indices], pre_triangles.astype(np.float64))
        # First strictly smaller distance wins, as in trace_history.
        selected = np.argmin(np.where(distances < np.inf, distances, np.inf), axis=1)
        chosen_faces = pre_face_ids[selected]
        current_faces[sample_indices] = chosen_faces
        current_barycentrics[sample_indices] = bary[np.arange(len(selected)), selected]
        for sample_index, best_face in zip(affected_samples, chosen_faces.tolist()):
            face_samples[best_face].append(sample_index)

    return current_faces, current_barycentrics


def sample_texture_bilinear(texture, uvs, as_float=False):
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
    if as_float:
        return np.clip(result, 0.0, 255.0)
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
    texture_padding,
    supersample=1
):
    if atlas_mode == "xatlas":
        atlas_data = build_xatlas_samples(
            vertices,
            faces,
            active_face_indices,
            texture_size,
            texture_padding,
            supersample
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
    if supersample > 1:
        # Box filter: each texel averages every sub-sample covering it, across
        # all triangles, so chart-border texels get coverage-weighted colors.
        color_sum = np.zeros((atlas_size, atlas_size, 3), dtype=np.float64)
        sample_count = np.zeros((atlas_size, atlas_size), dtype=np.int64)

    for material_id in np.unique(material_ids):
        indices = np.where(material_ids == material_id)[0]

        if len(original_textures) == 1:
            texture_index = 0
        else:
            texture_index = int(material_id)

        if texture_index < 0 or texture_index >= len(original_textures):
            texture_index = 0

        if supersample > 1:
            rows, cols = pixels[indices, 0], pixels[indices, 1]
            np.add.at(color_sum, (rows, cols), sample_texture_bilinear(
                original_textures[texture_index], source_uvs[indices], as_float=True))
            np.add.at(sample_count, (rows, cols), 1)
            continue

        colors = sample_texture_bilinear(
            original_textures[texture_index],
            source_uvs[indices]
        )
        atlas[
            pixels[indices, 0],
            pixels[indices, 1]
        ] = colors

    if supersample > 1:
        covered = sample_count > 0
        atlas[covered] = np.clip(np.rint(color_sum[covered] / sample_count[covered, None]),
                                 0, 255).astype(np.uint8)

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
        face_uv_indices = np.arange(
            len(faces) * 3, dtype=np.int64).reshape(-1, 3)

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
    for _name in ("closest_point_barycentric", "point_triangle_distance_sq",
                  "segment_segment_distance_sq", "segment_intersects_triangle",
                  "triangle_triangle_distance_sq", "closest_vertex_pair",
                  "virtual_edge_batch", "trace_history", "collapse_normal_diagnostics",
                  "rasterize_uv_triangles"):
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
    parser.add_argument("--texture_size", type=int, default=1024,
                        help="Baked atlas resolution. Default: 1024")
    parser.add_argument("--atlas", choices=("xatlas", "grid"), default="xatlas",
                        help="UV atlas mode. xatlas is recommended; grid is the old per-face layout.")
    parser.add_argument("--texture_padding", type=int, default=2,
                        help="Chart gutter in texels for xatlas and post-bake dilation. Default: 2")
    parser.add_argument(
        "--face_importance",
        help="Sobel face importance npy, required only for --method sobel",
    )
    parser.add_argument(
        "--importance_lambda",
        type=float,
        default=4.0,
        help="Sobel face importance weight. Default: 4.0. 0 uses the unweighted Wild-Mesh simplification cost.",
    )
    parser.add_argument('--method', choices=('wild','sobel','guarded','photometric'), default='wild')
    parser.add_argument("--face_energy",
                        help="Photometric energy density npy (photometric_quadric_weights.py), required for --method photometric")
    parser.add_argument("--photometric_lambda", type=float, default=1.0,
                        help="Photometric term weight relative to the geometric term. Default: 1.0 (equal energy)")
    parser.add_argument("--visibility", action="store_true",
                        help="Scale face quadrics by exterior visibility (combines with any --method).")
    parser.add_argument("--visibility_floor", type=float, default=0.05,
                        help="Weight of fully occluded faces: floor + (1-floor)*visibility. Default: 0.05")
    parser.add_argument("--visibility_directions", type=int, default=128)
    parser.add_argument("--texture_supersample", type=int, default=1,
                        help="KxK sub-texel samples per atlas texel, box-filtered. 1 = legacy center sampling.")
    args = parser.parse_args()
    run_started = time.perf_counter()
    if args.texture_size < 1:
        parser.error("--texture_size must be positive")
    if args.texture_padding < 0:
        parser.error("--texture_padding must be non-negative")
    if args.virtual_radius is not None and args.virtual_radius < 0:
        parser.error("--virtual_radius must be non-negative")
    if not 0.0 <= args.visibility_floor <= 1.0 or args.visibility_directions < 1:
        parser.error("--visibility_floor must be in [0, 1] and --visibility_directions positive")
    if args.texture_supersample < 1 or (args.texture_supersample > 1 and args.atlas != "xatlas"):
        parser.error("--texture_supersample must be >= 1 and needs --atlas xatlas when > 1")
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
    from obj_geometry_topology import restore_obj_position_topology
    vertices, faces, topology_stats = restore_obj_position_topology(input_path, vertices, faces)
    output_path.with_suffix('.input_topology.json').write_text(json.dumps(topology_stats, indent=2))

    face_weights = np.ones(len(faces), dtype=np.float64)
    if args.method == 'sobel':
        if not args.face_importance:
            parser.error('--method sobel requires --face_importance')
        importance = np.load(args.face_importance, allow_pickle=False).astype(np.float64)
        if importance.shape != (len(faces),) or not np.all(np.isfinite(importance)) or np.any((importance<0)|(importance>1)):
            raise ValueError('Sobel scores must be finite [0,1], one per original face')
        if not np.isfinite(args.importance_lambda) or args.importance_lambda < 0:
            raise ValueError('Invalid Sobel lambda')
        face_weights += args.importance_lambda * importance
    if args.method == 'photometric':
        if not args.face_energy:
            parser.error('--method photometric requires --face_energy')
        density = np.load(args.face_energy, allow_pickle=False).astype(np.float64)
        if density.shape != (len(faces),) or not np.all(np.isfinite(density)) or np.any(density < 0):
            raise ValueError('Photometric energy must be finite, non-negative, one per original face')
        if not np.isfinite(args.photometric_lambda) or args.photometric_lambda < 0:
            raise ValueError('Invalid photometric lambda')
        face_weights = photometric_face_weights(vertices, faces, density, args.photometric_lambda)
    if args.visibility:
        visibility = face_visibility(vertices, faces, args.visibility_directions)
        np.save(output_path.with_suffix('.visibility.npy'), visibility.astype(np.float32))
        face_weights *= args.visibility_floor + (1.0 - args.visibility_floor) * visibility

    if not 0.0 < args.ratio <= 1.0:
        parser.error(
            "--ratio must be greater than 0 and less than or equal to 1")

    target_faces = max(1, int(len(faces) * args.ratio))

    original_triangle_uvs = np.asarray(mesh.triangle_uvs).copy()
    original_material_ids = np.asarray(mesh.triangle_material_ids).copy()

    if args.texture is not None:
        texture_path = Path(args.texture)

        if not texture_path.exists():
            raise FileNotFoundError(
                f"Texture not found: {texture_path}"
            )

        print(
            f"[texture] using specified texture: {texture_path}",
            flush=True
        )

        original_textures = [
            np.asarray(
                Image.open(texture_path).convert("RGB")
            )
        ]

    elif mesh.has_textures():
        print(
            "[texture] loaded from OBJ/MTL",
            flush=True
        )

        original_textures = [
            np.asarray(texture).copy()
            for texture in mesh.textures
        ]

    else:
        # OBJ와 같은 이름의 PNG 자동 탐색
        auto_texture_path = input_path.with_suffix(".png")

        if auto_texture_path.exists():
            print(
                f"[texture] auto-detected: {auto_texture_path}",
                flush=True
            )

            original_textures = [
                np.asarray(
                    Image.open(auto_texture_path).convert("RGB")
                )
            ]

        else:
            print(
                f"[texture] no texture found "
                f"(checked {auto_texture_path})",
                flush=True
            )

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

    diagnostics = {}
    simplify_started = time.perf_counter()
    simplified_vertices, simplified_faces, active_face_indices, history = simplify(
        vertices,
        faces,
        target_faces,
        virtual_radius,
        has_texture,
        face_weights,
        normal_guard=args.method == 'guarded', diagnostics=diagnostics
    )
    simplify_seconds = time.perf_counter() - simplify_started

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
            args.texture_padding,
            args.texture_supersample
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

    output_path.with_suffix('.statistics.json').write_text(json.dumps(dict(
        method=args.method, visibility=args.visibility, visibility_floor=args.visibility_floor,
        texture_supersample=args.texture_supersample, simplification_seconds=simplify_seconds,
        total_seconds=time.perf_counter()-run_started, **diagnostics), indent=2))
    print(
        f"[done] faces={len(simplified_faces):,}, elapsed={time.perf_counter()-run_started:.1f}s, output={output_path}", flush=True)


if __name__ == "__main__":
    main()
