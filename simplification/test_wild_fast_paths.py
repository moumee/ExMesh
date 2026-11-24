"""The fast paths of wild_simplify_xatlas_verified must reproduce the reference ones.

Batched edge costs, cost reuse between collapses, the scalar/array closest-point
pair, the compiled history walk and the compiled UV rasterizer are each compared
with the plain implementation they replace.
"""
import unittest
from collections import defaultdict
from unittest.mock import patch

import numpy as np

import wild_simplify_xatlas_verified as wild


def grid(nx, ny, height):
    """Open height-field grid; height(x, y) gives z."""
    xs, ys = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    vertices = np.column_stack((xs.ravel(), ys.ravel(), height(xs, ys).ravel()))
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            faces += [[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]]
    return vertices, np.array(faces, dtype=np.int64)


def torus(nu, nv):
    """Closed surface without boundary edges."""
    u, v = np.meshgrid(np.arange(nu) * 2 * np.pi / nu, np.arange(nv) * 2 * np.pi / nv)
    ring = 3 + np.cos(v) * (1 + 0.2 * np.sin(3 * u))
    vertices = np.column_stack(((ring * np.cos(u)).ravel(), (ring * np.sin(u)).ravel(),
                                np.sin(v).ravel()))
    faces = []
    for j in range(nv):
        for i in range(nu):
            a, b = j * nu + i, j * nu + (i + 1) % nu
            c, d = ((j + 1) % nv) * nu + i, ((j + 1) % nv) * nu + (i + 1) % nu
            faces += [[a, b, c], [b, d, c]]
    return vertices, np.array(faces, dtype=np.int64)


def wavy(x, y):
    return 0.3 * np.sin(1.3 * x) * np.cos(0.7 * y)


def seam_split(nx, ny):
    """Grid cut along a column; both sides keep their own copy of the seam vertices."""
    vertices, faces = grid(nx, ny, wavy)
    cut = nx // 2
    copies = {y * nx + cut: len(vertices) + y for y in range(ny)}
    right = np.all(faces % nx >= cut, axis=1)
    faces[right] = np.vectorize(lambda index: copies.get(index, index))(faces[right])
    return np.vstack((vertices, vertices[list(copies)])), faces


def adjacency(vertex_count, faces):
    vertex_edges = [set() for _ in range(vertex_count)]
    edge_faces = defaultdict(set)
    for index, (i, j, k) in enumerate(faces.tolist()):
        for edge in (tuple(sorted((i, j))), tuple(sorted((j, k))), tuple(sorted((k, i)))):
            edge_faces[edge].add(index)
            vertex_edges[edge[0]].add(edge)
            vertex_edges[edge[1]].add(edge)
    return vertex_edges, edge_faces


def triangle_cases(rng, count):
    """Random points and triangles, with slivers, points and on-surface queries."""
    triangles = rng.normal(size=(count, 3, 3))
    points = rng.normal(size=(count, 3))
    k = count // 8
    along = rng.random((k, 1)) * 3 - 1
    triangles[:k, 2] = triangles[:k, 0] + (triangles[:k, 1] - triangles[:k, 0]) * along
    triangles[k:2 * k, 1] = triangles[k:2 * k, 0]
    triangles[2 * k:3 * k, 1:] = triangles[2 * k:3 * k, :1]
    weights = rng.random((k, 3))
    weights /= weights.sum(axis=1, keepdims=True)
    points[3 * k:4 * k] = np.einsum("ni,nij->nj", weights, triangles[3 * k:4 * k])
    points[4 * k:5 * k] = triangles[4 * k:5 * k, 0]
    points[5 * k:6 * k] = (triangles[5 * k:6 * k, 0] + triangles[5 * k:6 * k, 1]) / 2
    return points, triangles


class EdgeCostTests(unittest.TestCase):
    def test_batched_minimizer_matches_single_quadrics(self):
        rng = np.random.default_rng(0)
        quadrics = []
        for rank in (6, 6, 6, 2, 1, 0):  # rows of plane equations; fewer than 3 is rank deficient
            planes = rng.normal(size=(40, rank, 4))
            quadrics.append(np.einsum("nki,nkj->nij", planes, planes))
        quadrics = np.concatenate(quadrics)
        midpoints = rng.normal(size=(len(quadrics), 3))
        costs, positions = wild.minimize_quadrics(quadrics, midpoints)
        for quadric, midpoint, cost, position in zip(quadrics, midpoints, costs, positions):
            expected_cost, expected_position = wild.minimize_quadric(quadric, midpoint)
            np.testing.assert_allclose(position, expected_position, rtol=1e-9, atol=1e-9)
            self.assertAlmostEqual(cost, expected_cost, delta=1e-9 * (1 + expected_cost))
        self.assertEqual(wild.minimize_quadrics(quadrics[:0], midpoints[:0])[1].shape, (0, 3))

    def test_batched_minimizer_rejects_what_the_single_one_rejects(self):
        bad = np.zeros((2, 4, 4))
        bad[1, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            wild.minimize_quadrics(bad, np.zeros((2, 3)))
        indefinite = np.zeros((1, 4, 4))
        indefinite[0, 0, 0], indefinite[0, 1, 1] = 1., -1.
        with self.assertRaises(ValueError):
            wild.minimize_quadrics(indefinite, np.zeros((1, 3)))

    def test_batched_edge_costs_match_single_edges(self):
        rng = np.random.default_rng(1)
        vertices, faces = grid(5, 4, wavy)
        vertex_edges, edge_faces = adjacency(len(vertices), faces)
        planes = rng.normal(size=(len(vertices), 5, 4))
        Q = np.einsum("nki,nkj->nij", planes, planes)
        edges = sorted(edge_faces)
        costs, positions = wild.compute_edge_costs(edges, vertices, Q, vertex_edges, edge_faces)
        for edge, cost, position in zip(edges, costs, positions):
            expected_cost, expected_position = wild.compute_edge_cost(
                edge, vertices, Q, vertex_edges, edge_faces)
            np.testing.assert_allclose(position, expected_position, rtol=1e-9, atol=1e-9)
            self.assertAlmostEqual(cost, expected_cost, delta=1e-9 * (1 + expected_cost))


class CostReuseTests(unittest.TestCase):
    """reuse_costs=False is the reference loop: every edge around a collapse is recomputed."""

    def assert_same_run(self, vertices, faces, target, radius, guard):
        runs = []
        for reuse in (False, True):
            stats = {}
            result = wild.simplify(vertices.copy(), faces.copy(), target, radius, True,
                                   np.ones(len(faces)), normal_guard=guard,
                                   diagnostics=stats, reuse_costs=reuse)
            runs.append((result, stats))
        (reference, reference_stats), (reused, reused_stats) = runs
        self.assertEqual(reference_stats, reused_stats)
        for expected, actual in zip(reference[:3], reused[:3]):
            self.assertTrue(np.array_equal(expected, actual))
        self.assertEqual(len(reference[3]), len(reused[3]))
        for expected_step, actual_step in zip(reference[3], reused[3]):
            for expected, actual in zip(expected_step, actual_step):
                self.assertTrue(np.array_equal(expected, actual))
        self.assertLess(len(reused[1]), len(faces))
        return reused_stats

    def test_guard_rejections_requeue_like_the_reference(self):
        # Rejected edges leave the queue and must come back exactly as in the reference loop.
        # Sliding the vertices of a flat sheet sideways makes many collapses fold a face.
        vertices, faces = grid(10, 10, lambda x, y: 0 * x)
        vertices[:, :2] += np.random.default_rng(1).normal(scale=0.3, size=(len(vertices), 2))
        stats = self.assert_same_run(vertices, faces, len(faces) // 10, 0.0, True)
        self.assertGreater(stats["rejected"], 10)

    def test_open_surface_with_boundary_terms(self):
        vertices, faces = grid(9, 8, wavy)
        for guard in (False, True):
            self.assert_same_run(vertices, faces, len(faces) // 5, 0.0, guard)

    def test_flat_surface_with_equal_costs_and_singular_quadrics(self):
        # Every cost is zero, so the order depends only on the tie-breakers.
        vertices, faces = grid(8, 8, lambda x, y: 0 * x)
        for guard in (False, True):
            self.assert_same_run(vertices, faces, len(faces) // 4, 0.0, guard)

    def test_closed_surface(self):
        vertices, faces = torus(12, 8)
        for guard in (False, True):
            self.assert_same_run(vertices, faces, len(faces) // 5, 0.0, guard)

    def test_close_components_joined_by_virtual_edges(self):
        vertices, faces = grid(6, 6, wavy)
        both = np.vstack((vertices, vertices + [0.2, 0.1, 0.05]))
        both_faces = np.vstack((faces, faces + len(vertices)))
        self.assertTrue(wild.build_virtual_edges(both, both_faces, 0.05))
        self.assert_same_run(both, both_faces, len(both_faces) // 5, 0.05, False)

    def test_duplicate_and_degenerate_input_faces(self):
        vertices, faces = grid(7, 6, wavy)
        faces = np.vstack((faces, faces[[3, 10, 10]], [[0, 0, 1], [8, 9, 9]]))
        self.assert_same_run(vertices, faces, len(faces) // 4, 0.0, False)

    def test_seam_closed_by_virtual_edges(self):
        # Zero-length virtual edges merge the two copies of the seam, so boundary
        # edges turn into interior ones and their area terms disappear.
        vertices, faces = seam_split(8, 7)
        self.assertEqual(len(set(wild.connected_components(vertices, faces).tolist())), 2)
        self.assert_same_run(vertices, faces, len(faces) // 3, 0.02, False)


class CollapseTests(unittest.TestCase):
    def test_output_has_no_degenerate_or_duplicate_faces(self):
        vertices, faces = torus(12, 8)
        for target in (60, 20, 6):
            new_vertices, new_faces, active, _ = wild.simplify(
                vertices.copy(), faces.copy(), target, 0.0, False, np.ones(len(faces)))
            self.assertLessEqual(len(new_faces), target)
            self.assertEqual(len(active), len(new_faces))
            self.assertTrue(np.all((new_faces >= 0) & (new_faces < len(new_vertices))))
            ordered = np.sort(new_faces, axis=1)
            self.assertTrue(np.all(ordered[:, 0] < ordered[:, 1]) and np.all(ordered[:, 1] < ordered[:, 2]))
            self.assertEqual(len(np.unique(ordered, axis=0)), len(new_faces))
            self.assertEqual(len(np.unique(new_faces)), len(new_vertices))


class ClosestPointTests(unittest.TestCase):
    def test_array_version_equals_scalar_version_bit_for_bit(self):
        points, triangles = triangle_cases(np.random.default_rng(2), 160)
        distances, barycentrics = wild.closest_points_barycentric(points[:40], triangles)
        for i in range(40):
            for j in range(len(triangles)):
                expected = wild.closest_point_barycentric(*points[i], *triangles[j].ravel())
                self.assertEqual((distances[i, j], *barycentrics[i, j]), expected)

    def test_result_is_the_closest_point_of_the_triangle(self):
        points, triangles = triangle_cases(np.random.default_rng(3), 240)
        steps = np.linspace(0, 1, 41)
        grid_u, grid_v = np.meshgrid(steps, steps)
        inside = grid_u + grid_v <= 1 + 1e-12
        weights = np.column_stack((1 - grid_u[inside] - grid_v[inside], grid_u[inside], grid_v[inside]))
        for point, triangle in zip(points, triangles):
            distance, u, v, w = wild.closest_point_barycentric(*point, *triangle.ravel())
            self.assertAlmostEqual(u + v + w, 1.0, places=12)
            self.assertGreaterEqual(min(u, v, w), -1e-12)
            closest = u * triangle[0] + v * triangle[1] + w * triangle[2]
            self.assertAlmostEqual(distance, np.sum((point - closest) ** 2), places=10)
            sampled = np.min(np.sum((weights @ triangle - point) ** 2, axis=1))
            self.assertLessEqual(distance, sampled + 1e-9)

    def test_point_triangle_distance(self):
        triangle = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0]])
        self.assertAlmostEqual(wild.point_triangle_distance_sq(np.array([.5, .5, 3.]), triangle), 9.)
        self.assertAlmostEqual(wild.point_triangle_distance_sq(np.array([3., 0, 0]), triangle), 1.)
        self.assertAlmostEqual(wild.point_triangle_distance_sq(np.array([2., 2, 0]), triangle), 2.)


class HistoryWalkTests(unittest.TestCase):
    def test_compiled_walk_equals_numpy_walk(self):
        rng = np.random.default_rng(4)
        vertices, faces = grid(10, 9, wavy)
        new_vertices, new_faces, active, history = wild.simplify(
            vertices.copy(), faces.copy(), len(faces) // 6, 0.0, True, np.ones(len(faces)))
        samples = 40
        weights = rng.random((len(new_faces), samples, 3))
        weights /= weights.sum(axis=2, keepdims=True)
        points = np.einsum("fsi,fij->fsj", weights, new_vertices[new_faces]).reshape(-1, 3)
        barycentrics = weights.reshape(-1, 3)
        face_ids = np.repeat(active, samples)
        mapped_faces, mapped_barycentrics = wild.successive_map(points, barycentrics, face_ids, history)
        with patch.object(wild, "NUMBA_AVAILABLE", False):
            numpy_faces, numpy_barycentrics = wild.successive_map(points, barycentrics, face_ids, history)
        self.assertTrue(np.array_equal(mapped_faces, numpy_faces))
        self.assertTrue(np.array_equal(mapped_barycentrics, numpy_barycentrics))
        # Samples end on original faces, at a point of the original surface near the query.
        self.assertTrue(np.all((mapped_faces >= 0) & (mapped_faces < len(faces))))
        np.testing.assert_allclose(mapped_barycentrics.sum(axis=1), 1.0, atol=1e-12)
        self.assertFalse(np.array_equal(mapped_faces, face_ids))
        landed = np.einsum("ni,nij->nj", mapped_barycentrics, vertices[faces[mapped_faces]])
        self.assertLess(np.max(np.linalg.norm(landed - points, axis=1)), 1.0)

    def test_empty_history_returns_the_input(self):
        faces, barycentrics = wild.successive_map(
            np.zeros((2, 3)), np.array([[1., 0, 0], [0, 1., 0]]), np.array([4, 7]), [])
        np.testing.assert_array_equal(faces, [4, 7])
        np.testing.assert_array_equal(barycentrics, [[1., 0, 0], [0, 1., 0]])


class AtlasTests(unittest.TestCase):
    def test_compiled_rasterizer_equals_per_triangle_rasterizer(self):
        rng = np.random.default_rng(5)
        triangle_uvs = rng.random((14, 3, 2)) * 1.2 - 0.1   # some reach outside the atlas
        triangle_uvs[3, 2] = triangle_uvs[3, 0]             # zero area
        triangle_uvs[4] = [[0, 0], [1, 0], [0, 1]]
        triangle_uvs[5] = [[1, 0], [1, 1], [0, 1]]
        corners = rng.normal(size=(14, 3, 3))
        for size, supersample in ((24, 1), (16, 3)):
            expected = [wild.rasterize_uv_triangle(triangle, size, supersample)
                        for triangle in triangle_uvs]
            counts = wild.rasterize_uv_triangles(
                triangle_uvs, corners, size, supersample,
                np.empty((0, 2), np.int32), np.empty((0, 3)), np.empty((0, 3)))
            np.testing.assert_array_equal(counts, [len(rows) for rows, _, _ in expected])
            pixels = np.empty((counts.sum(), 2), np.int32)
            barycentrics = np.empty((counts.sum(), 3))
            points = np.empty((counts.sum(), 3))
            wild.rasterize_uv_triangles(triangle_uvs, corners, size, supersample,
                                        pixels, barycentrics, points)
            rows = np.concatenate([rows for rows, _, _ in expected])
            cols = np.concatenate([cols for _, cols, _ in expected])
            weights = np.concatenate([bary for _, _, bary in expected])
            self.assertTrue(np.array_equal(pixels, np.column_stack((rows, cols))))
            self.assertTrue(np.array_equal(barycentrics, weights))
            owner = np.repeat(np.arange(len(counts)), counts)
            np.testing.assert_allclose(
                points, np.einsum("ni,nij->nj", weights, corners[owner]), atol=1e-12)

    def test_atlas_faces_are_matched_in_any_order(self):
        rng = np.random.default_rng(6)
        faces = np.array([[0, 1, 2], [2, 1, 3], [3, 1, 4], [4, 5, 3], [0, 2, 5]])
        # UV vertices 0..5 map to themselves; 6..8 are seam copies of 1, 2 and 3.
        vmapping = np.array([0, 1, 2, 3, 4, 5, 1, 2, 3])
        atlas_indices = np.array([[0, 1, 2], [7, 6, 3], [8, 6, 4], [4, 5, 8], [0, 7, 5]])
        np.testing.assert_array_equal(
            wild.match_atlas_faces(faces, vmapping, atlas_indices), atlas_indices)
        order = rng.permutation(len(faces))
        shuffled = np.array([np.roll(atlas_indices[index], index) for index in order])
        np.testing.assert_array_equal(
            wild.match_atlas_faces(faces, vmapping, shuffled), atlas_indices)
        # Two faces with the same vertices: the last of them takes the first atlas triangle.
        doubled = np.vstack((faces, faces[:1]))
        matched = wild.match_atlas_faces(doubled, vmapping, np.vstack((atlas_indices, [[0, 6, 7]])))
        np.testing.assert_array_equal(matched[[0, 5]], [[0, 6, 7], [0, 1, 2]])
        np.testing.assert_array_equal(matched[1:5], atlas_indices[1:])


class VirtualEdgeScanTests(unittest.TestCase):
    def test_face_components(self):
        faces = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 5], [9, 10, 11]])
        labels = wild.connected_components(np.zeros((12, 3)), faces)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertEqual(len(set(labels.tolist())), 3)

    def test_prefilter_keeps_both_faces_of_every_candidate_pair(self):
        rng = np.random.default_rng(7)
        components = rng.integers(0, 5, size=300)
        # Components 0 and 1 overlap in x; the other three are far away from everything.
        origins = np.array([[0., 0, 0], [3, 0, 0], [40, 0, 0], [0, 40, 0], [0, 0, 40]])
        centers = origins[components] + rng.random((300, 3)) * 4
        bounds_min = centers - rng.random((300, 3)) * 0.4
        bounds_max = centers + rng.random((300, 3)) * 0.4
        threshold_sq = 0.5 ** 2
        near = wild.near_other_component(bounds_min, bounds_max, components, threshold_sq)
        gap = np.maximum(0.0, np.maximum(bounds_min[:, None] - bounds_max[None],
                                         bounds_min[None] - bounds_max[:, None]))
        candidate = (np.einsum("ijk,ijk->ij", gap, gap) < threshold_sq) \
            & (components[:, None] != components[None])
        self.assertTrue(np.any(candidate))
        self.assertTrue(np.all(near[np.any(candidate, axis=1)]))
        self.assertFalse(np.any(near[components >= 2]))
        self.assertFalse(np.all(near[components < 2]))


if __name__ == "__main__":
    unittest.main()
