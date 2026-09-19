import open3d as o3d
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import heapq
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True,
                    help="Input mesh path. e.g) data/mesh.ply")
parser.add_argument(
    "--output", help="Output mesh path. e.g) results/mesh_simplified.ply")
parser.add_argument("--ratio", type=float, default=0.5)
parser.add_argument("--virtual_radius", type=float)
parser.add_argument("--texture")
parser.add_argument("--texture_size", type=int, default=1024)
args = parser.parse_args()

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

mesh = o3d.io.read_triangle_mesh(str(input_path))

vertices = np.asarray(mesh.vertices).copy()
faces = np.asarray(mesh.triangles).copy()

if not 0.0 < args.ratio <= 1.0:
    parser.error("--ratio must be greater than 0 and less than or equal to 1")

target_faces = max(1, int(len(faces) * args.ratio))

original_vertices = vertices.copy()
original_faces = faces.copy()
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


def compute_edge_cost(edge, vertices, Q, vertex_edges, edge_faces):
    a, b = edge
    Q_ab = Q[a] + Q[b]
    Q_total = Q_ab + compute_area_quadric(
        edge,
        vertices,
        vertex_edges,
        edge_faces
    )

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
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)

        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b

            if denom != 0.0:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            t = (b * s + f) / e

            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

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
    best_edge = None
    best_distance = np.inf

    for a in face_a:
        for b in face_b:
            if a == b:
                continue

            delta = vertices[a] - vertices[b]
            distance = np.dot(delta, delta)

            if distance < best_distance:
                best_distance = distance
                best_edge = tuple(sorted((int(a), int(b))))

    return best_edge


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
        return set()

    face_components = connected_components(vertices, faces)
    triangles = vertices[faces]
    centroids = np.mean(triangles, axis=1)
    triangle_radii = np.max(
        np.linalg.norm(triangles - centroids[:, None, :], axis=2),
        axis=1
    )
    maximum_radius = np.max(triangle_radii)
    tree = cKDTree(centroids)
    virtual_edges = set()
    threshold_sq = (2.0 * radius) ** 2

    for i in range(len(faces)):
        search_radius = 2.0 * radius + triangle_radii[i] + maximum_radius
        candidates = tree.query_ball_point(centroids[i], search_radius)

        for j in candidates:
            if j <= i:
                continue
            if face_components[i] == face_components[j]:
                continue

            distance_sq = triangle_triangle_distance_sq(
                triangles[i],
                triangles[j]
            )

            if distance_sq >= threshold_sq:
                continue

            edge = closest_vertex_pair(
                faces[i],
                faces[j],
                vertices
            )

            if edge is not None:
                virtual_edges.add(edge)

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

    for f_idx, face in enumerate(faces):

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

        v0, v1, v2 = vertices[face]
        normal = np.cross(v1 - v0, v2 - v0)
        normal_length = np.linalg.norm(normal)
        if normal_length == 0:
            continue

        # The norm of the cross product is (2 * area)
        face_area = normal_length / 2.0
        normal = normal / normal_length

        d = -np.dot(normal, v0)
        plane = np.append(normal, d)

        K = np.outer(plane, plane)
        quadric = (face_area / 3.0) * K
        Q[i] += quadric
        Q[j] += quadric
        Q[k] += quadric

    physical_edges = set(edge_faces.keys())
    virtual_edges = build_virtual_edges(
        vertices,
        faces,
        virtual_radius
    )

    for edge in virtual_edges:
        if edge in physical_edges:
            continue
        a, b = edge
        vertex_edges[a].add(edge)
        vertex_edges[b].add(edge)
        edge_faces[edge]

    # Min heap for getting the lowest cost edge
    heap = []

    # Min heap tie-breaker counter
    counter = 0

    for edge in list(edge_faces.keys()):
        cost, position = compute_edge_cost(
            edge,
            vertices,
            Q,
            vertex_edges,
            edge_faces
        )

        version = edge_version[edge]

        heapq.heappush(
            heap,
            (cost, counter, edge, position, version)
        )
        counter += 1

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
                edge_faces
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
        index = int(np.argmin(distances))
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


def build_atlas_samples(vertices, faces, active_face_indices, texture_size):
    face_count = len(faces)
    grid_size = int(np.ceil(np.sqrt(face_count)))
    tile_size = max(6, texture_size // max(grid_size, 1))
    atlas_size = tile_size * max(grid_size, 1)
    margin = max(1, min(2, tile_size // 4))
    triangle_uvs = np.zeros((face_count, 3, 2), dtype=np.float64)
    points = []
    barycentrics = []
    face_ids = []
    pixels = []
    masks = []

    for fi, face in enumerate(faces):
        row = fi // grid_size
        col = fi % grid_size
        left = col * tile_size
        top = row * tile_size
        x0 = left + margin
        y0 = top + margin
        x1 = left + tile_size - margin - 1
        y1 = y0
        x2 = x0
        y2 = top + tile_size - margin - 1

        p0 = np.array([x0 + 0.5, y0 + 0.5])
        p1 = np.array([x1 + 0.5, y1 + 0.5])
        p2 = np.array([x2 + 0.5, y2 + 0.5])

        triangle_uvs[fi, 0] = [
            p0[0] / atlas_size,
            1.0 - p0[1] / atlas_size
        ]
        triangle_uvs[fi, 1] = [
            p1[0] / atlas_size,
            1.0 - p1[1] / atlas_size
        ]
        triangle_uvs[fi, 2] = [
            p2[0] / atlas_size,
            1.0 - p2[1] / atlas_size
        ]

        width = p1[0] - p0[0]
        height = p2[1] - p0[1]
        mask = np.zeros((tile_size, tile_size), dtype=bool)

        for y in range(y0, y2 + 1):
            for x in range(x0, x1 + 1):
                px = x + 0.5
                py = y + 0.5
                beta = (px - p0[0]) / width
                gamma = (py - p0[1]) / height
                alpha = 1.0 - beta - gamma

                if alpha < -1e-12 or beta < -1e-12 or gamma < -1e-12:
                    continue

                bary = np.array([alpha, beta, gamma], dtype=np.float64)
                point = (
                    bary[0] * vertices[face[0]]
                    + bary[1] * vertices[face[1]]
                    + bary[2] * vertices[face[2]]
                )

                points.append(point)
                barycentrics.append(bary)
                face_ids.append(int(active_face_indices[fi]))
                pixels.append((y, x))
                mask[y - top, x - left] = True

        masks.append((top, left, mask))

    return (
        atlas_size,
        tile_size,
        triangle_uvs,
        np.asarray(points, dtype=np.float64),
        np.asarray(barycentrics, dtype=np.float64),
        np.asarray(face_ids, dtype=np.int64),
        np.asarray(pixels, dtype=np.int64),
        masks
    )


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

        for sample_index in affected_samples:
            point = points[sample_index]
            best_distance = np.inf
            best_face = -1
            best_barycentric = None

            for local_index, triangle in enumerate(pre_triangles):
                closest, barycentric = closest_point_on_triangle(
                    point,
                    triangle[0],
                    triangle[1],
                    triangle[2]
                )
                delta = point - closest
                distance = np.dot(delta, delta)

                if distance < best_distance:
                    best_distance = distance
                    best_face = int(pre_face_ids[local_index])
                    best_barycentric = barycentric

            current_faces[sample_index] = best_face
            current_barycentrics[sample_index] = best_barycentric
            reassigned[best_face].append(sample_index)

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


def bake_texture(
    vertices,
    faces,
    active_face_indices,
    history,
    original_triangle_uvs,
    original_material_ids,
    original_textures,
    texture_size
):
    (
        atlas_size,
        tile_size,
        triangle_uvs,
        points,
        barycentrics,
        face_ids,
        pixels,
        masks
    ) = build_atlas_samples(
        vertices,
        faces,
        active_face_indices,
        texture_size
    )

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

    for top, left, mask in masks:
        if not np.any(mask):
            continue

        tile = atlas[
            top:top + tile_size,
            left:left + tile_size
        ]
        missing = ~mask

        if not np.any(missing):
            continue

        _, nearest = distance_transform_edt(
            missing,
            return_indices=True
        )
        filled = tile.copy()
        filled[missing] = tile[
            nearest[0][missing],
            nearest[1][missing]
        ]
        atlas[
            top:top + tile_size,
            left:left + tile_size
        ] = filled

    return triangle_uvs, atlas


def write_textured_obj(path, vertices, faces, triangle_uvs, texture):
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

    with open(path, "w", encoding="utf-8") as file:
        file.write(f"mtllib {mtl_path.name}\n")

        for vertex in vertices:
            file.write(
                f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n"
            )

        flattened_uvs = triangle_uvs.reshape(-1, 2)

        for uv in flattened_uvs:
            file.write(f"vt {uv[0]:.17g} {uv[1]:.17g}\n")

        file.write("usemtl material0\n")

        for fi, face in enumerate(faces):
            uv_base = 3 * fi + 1
            file.write(
                "f "
                f"{face[0] + 1}/{uv_base} "
                f"{face[1] + 1}/{uv_base + 1} "
                f"{face[2] + 1}/{uv_base + 2}\n"
            )


if len(faces) > 0:
    bbox_diagonal = np.linalg.norm(
        np.max(vertices, axis=0) - np.min(vertices, axis=0)
    )
else:
    bbox_diagonal = 0.0

if args.virtual_radius is None:
    virtual_radius = 0.01 * bbox_diagonal
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

    simplified_triangle_uvs, simplified_texture = bake_texture(
        simplified_vertices,
        simplified_faces,
        active_face_indices,
        history,
        original_triangle_uvs,
        original_material_ids,
        original_textures,
        args.texture_size
    )

    write_textured_obj(
        output_path,
        simplified_vertices,
        simplified_faces,
        simplified_triangle_uvs,
        simplified_texture
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
