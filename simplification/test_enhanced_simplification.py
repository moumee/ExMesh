import unittest

import numpy as np

import wild_simplify_xatlas_verified as wild


class SupersampleTests(unittest.TestCase):
    def test_single_sample_is_legacy_texel_center(self):
        triangle = np.array([[0.1, 0.1], [0.8, 0.15], [0.3, 0.9]])
        rows, cols, bary = wild.rasterize_uv_triangle(triangle, 32, 1)
        uv = np.column_stack(((cols + 0.5) / 32, 1 - (rows + 0.5) / 32))
        np.testing.assert_allclose(bary @ triangle, uv, atol=1e-12)

    def test_supersampling_covers_texels_with_k_squared_samples(self):
        square = [np.array([[0, 0], [1, 0], [0, 1]]), np.array([[1, 0], [1, 1], [0, 1]])]
        counts = np.zeros((8, 8), int)
        for triangle in square:
            rows, cols, _ = wild.rasterize_uv_triangle(triangle.astype(float), 8, 3)
            np.add.at(counts, (rows, cols), 1)
        # Every texel is covered; diagonal texels may count shared-edge samples twice.
        self.assertTrue(np.all(counts >= 9))
        self.assertEqual(int(np.median(counts)), 9)


class PhotometricWeightTests(unittest.TestCase):
    def test_area_weighted_mean_of_extra_weight_equals_lambda(self):
        rng = np.random.default_rng(0)
        vertices = rng.normal(size=(30, 3))
        faces = rng.integers(0, 30, size=(40, 3))
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
        density = rng.random(len(faces)) * 1e9
        tri = vertices[faces]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
        for lam in (0.5, 1.0, 4.0):
            w = wild.photometric_face_weights(vertices, faces, density, lam)
            self.assertAlmostEqual(np.sum((w - 1) * area) / area.sum(), lam, places=10)

    def test_zero_energy_is_rejected(self):
        vertices = np.eye(3)
        with self.assertRaises(ValueError):
            wild.photometric_face_weights(vertices, np.array([[0, 1, 2]]), np.zeros(1))


class VisibilityTests(unittest.TestCase):
    def test_enclosed_face_is_invisible_and_hull_is_visible(self):
        # Closed unit cube (outward faces) with a small triangle sealed inside it.
        v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        faces = [f for a, b, c, d in quads for f in ((a, b, c), (a, c, d))]
        inner = len(v)
        v = np.vstack([v, [[0.4, 0.4, 0.5], [0.6, 0.4, 0.5], [0.5, 0.6, 0.5]]])
        faces.append((inner, inner + 1, inner + 2))
        visibility = wild.face_visibility(v, np.array(faces), 64)
        self.assertLess(visibility[-1], 1e-6)
        self.assertGreater(visibility[:-1].min(), 0.99)

    def test_fibonacci_directions_are_unit_and_balanced(self):
        d = wild.fibonacci_sphere(256)
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
        self.assertLess(np.linalg.norm(d.mean(0)), 0.02)


class EdgePaddingTests(unittest.TestCase):
    def test_uncovered_texels_copy_nearest_chart_texel_and_receive_no_gradient(self):
        try:
            import torch
            from appearance_refine import edge_pad_index, padded
        except ImportError as error:
            self.skipTest(str(error))
        covered = torch.zeros(6, 6, dtype=torch.bool)
        covered[1:3, 1:4] = True
        texture = torch.arange(6 * 6 * 3, dtype=torch.float32).reshape(1, 6, 6, 3).requires_grad_(True)
        out = padded(texture, edge_pad_index(covered))
        self.assertTrue(torch.equal(out[0][covered], texture[0][covered]))
        self.assertTrue(torch.equal(out[0, 0, 0], texture[0, 1, 1]))
        self.assertTrue(torch.equal(out[0, 5, 3], texture[0, 2, 3]))
        out.sum().backward()
        self.assertTrue(torch.all(texture.grad[0][~covered] == 0))
        self.assertTrue(torch.all(texture.grad[0][covered] >= 1))


class CameraConventionTests(unittest.TestCase):
    def test_nvdiffrast_matrix_matches_cpu_pinhole(self):
        """photometric_quadric_weights.camera_mvp must project like evaluate_mesh_preservation.render_cpu."""
        try:
            from photometric_quadric_weights import camera_mvp
        except ImportError as error:  # torch missing in the CPU-only env
            self.skipTest(str(error))
        rng = np.random.default_rng(1)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.linalg.det(q))
        camera = dict(rotation=q.tolist(), position=[0.3, -0.2, -2.5], fx=900.0, fy=880.0, width=640, height=480)
        mvp, _ = camera_mvp(camera, 0.1, 10.0)
        points = np.asarray(camera["position"]) + (q @ np.array([[0.1, -0.2, 2.0], [-0.3, 0.25, 3.0]]).T).T
        clip = np.c_[points, np.ones(2)] @ mvp.T
        # nvdiffrast: pixel column = (x_ndc + 1) W / 2, row = (y_ndc + 1) H / 2 (row 0 = image top here).
        cols = (clip[:, 0] / clip[:, 3] + 1) * camera["width"] / 2
        rows = (clip[:, 1] / clip[:, 3] + 1) * camera["height"] / 2
        local = (q.T @ (points - camera["position"]).T).T
        np.testing.assert_allclose(cols, camera["fx"] * local[:, 0] / local[:, 2] + camera["width"] / 2, atol=1e-9)
        np.testing.assert_allclose(rows, camera["fy"] * local[:, 1] / local[:, 2] + camera["height"] / 2, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
