import unittest
from unittest.mock import patch
import numpy as np
import wild_simplify_xatlas_verified as wild


class NormalGuardTests(unittest.TestCase):
    def setUp(self):
        self.v = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [-1., 0, 0]])
        self.f = np.array([[0, 1, 2]], dtype=np.int64)

    def test_detects_flip_and_zero_area(self):
        check = wild.collapse_normal_diagnostics
        self.assertEqual(check(self.v, self.f, 0, 3, np.array([2., 2, 0])), (1, 0))
        self.assertEqual(check(self.v, self.f, 0, 3, np.array([.5, .5, 0])), (0, 1))
        self.assertEqual(check(self.v, self.f, 0, 3, np.array([.1, .1, 0])), (0, 0))

    def test_removed_faces_do_not_constrain_placement(self):
        self.assertEqual(wild.collapse_normal_diagnostics(
            self.v, self.f, 0, 1, np.array([2., 2, 0])), (0, 0))

    def test_winding_scale_and_translation_do_not_change_decision(self):
        for scale in (1e-4, 1., 1e4):
            shift = np.array([3., -2., 5.]) * scale
            self.assertEqual(wild.collapse_normal_diagnostics(
                self.v*scale+shift, self.f[:, ::-1].copy(), 0, 3,
                np.array([2., 2, 0])*scale+shift), (1, 0))

    def test_guard_rejects_actual_invalid_heap_candidate(self):
        # Force a virtual edge whose proposed position would invert a face.
        v = np.vstack([self.v[:3], [[2., 2, .01], [3., 2, .01], [2., 3, .01]]])
        f = np.array([[0, 1, 2], [3, 4, 5]])
        def cost(edge, vertices, *unused):
            if edge == (0, 3):
                return -1., np.array([2., 2, .01])
            return 1., (vertices[edge[0]]+vertices[edge[1]])/2
        for guard in (False, True):
            stats = {}
            with patch.object(wild, 'build_virtual_edges', return_value={(0, 3)}), \
                    patch.object(wild, 'compute_edge_cost', side_effect=cost):
                result = wild.simplify(v.copy(), f.copy(), 1, .1, True, np.ones(2),
                                       normal_guard=guard, diagnostics=stats)
            self.assertLessEqual(len(result[1]), 1)
            if guard:
                self.assertGreater(stats['rejected'], 0)
                self.assertEqual(stats['accepted_flip_events'], 0)
            else:
                self.assertGreater(stats['accepted_flip_events'], 0)


if __name__ == '__main__':
    unittest.main()
