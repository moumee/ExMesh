import unittest
import tempfile
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
import numpy as np
import wild_simplify_xatlas_verified as wild
from wild_simplify_xatlas_verified import minimize_quadric
from obj_geometry_topology import restore_obj_position_topology
from moumee.moumee_img_utils import compute_face_importance


class WildAuditTests(unittest.TestCase):
    def test_obj_restores_position_ids_but_keeps_coincident_authored_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'m.obj'
            p.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 0\nv 0 -1 0\nf 1/1 2/2 3/3\nf 4/4 5/5 2/6\n')
            loaded=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,0],[0,-1,0],[1,0,0]])
            f=np.arange(6).reshape(-1,3)
            v,ff,stats=restore_obj_position_topology(p,loaded,f)
            np.testing.assert_array_equal(ff,[[0,1,2],[3,4,1]])
            self.assertEqual(len(v),5)
            self.assertNotEqual(ff[0,0],ff[1,0])
            self.assertEqual(stats['removed_uv_splits'],1)
            with self.assertRaises(ValueError):restore_obj_position_topology(p,loaded,f[::-1])

    def test_rank_two_preserves_nullspace_midpoint(self):
        q=np.diag([2.,3.,0.,0.]);q[0,3]=q[3,0]=-4;q[1,3]=q[3,1]=-9;q[3,3]=35
        cost,p=minimize_quadric(q,np.array([9.,8.,7.]))
        np.testing.assert_allclose(p,[2,3,7]);self.assertAlmostEqual(cost,0.)

    def test_nearly_singular_does_not_make_faraway_vertex(self):
        q=np.diag([1.,1.,1e-18,1.]);q[2,3]=q[3,2]=1e-10
        _,p=minimize_quadric(q,np.array([0.,0.,.5]))
        self.assertEqual(p[2],.5)
        self.assertEqual(np.linalg.solve(q[:3,:3],-q[:3,3])[2],-1e8)

    def test_area_quadric_matches_cross_product_area(self):
        v=np.array([[.2,.4,.1],[1.4,.2,.7],[.3,1.,.1]])
        ve=[set() for _ in v];ef=defaultdict(set)
        for e in [(0,1),(0,2),(1,2)]:
            ef[e]={0}
            for i in e:ve[i].add(e)
        q=wild.compute_area_quadric((0,1),v,ve,ef);p=np.array([.7,.4,.3]);h=np.r_[p,1]
        expected=sum(.5*np.linalg.norm(np.cross(v[b]-v[a],p-v[a]))**2 for a,b in ef)
        self.assertAlmostEqual(h@q@h,expected,places=12)

    def test_cached_area_equals_recomputed_after_every_collapse(self):
        v=np.array([[x,y,.12*np.sin(x+y)] for y in range(4) for x in range(4)])
        f=[]
        for y in range(3):
            for x in range(3):
                a=y*4+x;f += [[a,a+1,a+4],[a+1,a+5,a+4]]
        calls=[];original=wild.compute_edge_cost
        def checking(edge,vertices,q,ve,ef,cache=None):
            if cache is not None:
                va,bq=cache;area=va[edge[0]]+va[edge[1]]-bq.get(edge,np.zeros((4,4)))
                np.testing.assert_allclose(area,wild.compute_area_quadric(edge,vertices,ve,ef),atol=1e-12)
            calls.append(edge)
            return original(edge,vertices,q,ve,ef,cache)
        with patch.object(wild,'compute_edge_cost',side_effect=checking):
            result=wild.simplify(v.copy(),np.array(f),6,0,True,np.ones(len(f)))
        self.assertGreater(len(calls),60)
        self.assertLessEqual(len(result[1]),6)
        self.assertTrue(np.all(np.isfinite(result[0])))
        self.assertTrue(np.all(result[1]>=0));self.assertLess(result[1].max(),len(result[0]))

    def test_virtual_edges_between_close_components(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.05],[1,0,.05],[0,1,.05]],float)
        f=np.array([[0,1,2],[3,4,5]])
        self.assertEqual(wild.build_virtual_edges(v,f,.01),set())
        result=wild.build_virtual_edges(v,f,.03)
        self.assertEqual(result,{(0,3)})

    def test_successive_map_reuses_original_final_query(self):
        # Later split projects to z=1, earlier split must still query original z=2.
        # Two disconnected pre surfaces z=0 and z=1.8 make composed query differ.
        tri=np.array([[0.,0,0],[2,0,0],[0,2,0]])
        history=[(np.array([0,1]),np.array([tri,tri+[0,0,1.8]]),np.array([2])),
                 (np.array([2]),np.array([tri+[0,0,1.]]),np.array([3]))]
        ids,bary=wild.successive_map(np.array([[.4,.4,2.]]),np.array([[.6,.2,.2]]),np.array([3]),history)
        self.assertEqual(ids[0],1);np.testing.assert_allclose(bary,[[.6,.2,.2]])


class SobelScoreTests(unittest.TestCase):
    def test_visible_views_are_equally_weighted_and_background_is_excluded(self):
        # Face 0 covers four pixels in the first view and one in the second.
        # Its result must be the mean of the two view means, not a pooled mean.
        magnitudes = [np.array([[2, 2, 2, 2, 6, 999]], dtype=np.float32),
                      np.array([[6, 999]], dtype=np.float32)]
        ids = [np.array([[0, 0, 0, 0, 1, -1]]), np.array([[0, -1]])]
        score, q = compute_face_importance(magnitudes, ids, 3)
        self.assertEqual(q, 6)
        np.testing.assert_allclose(score, [2/3, 1, 0], rtol=1e-7)
        self.assertEqual(score.dtype, np.float32)

    def test_wrong_face_ids_and_resolution_are_rejected(self):
        magnitude = [np.ones((2, 2), dtype=np.float32)]
        with self.assertRaises(ValueError):
            compute_face_importance(magnitude, [np.full((2, 2), 3)], 3)
        with self.assertRaises(ValueError):
            compute_face_importance(magnitude, [np.zeros((1, 2), dtype=int)], 3)

    def test_no_visible_pixels_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_face_importance([np.ones((2, 2), dtype=np.float32)],
                                    [np.full((2, 2), -1)], 3)


if __name__=='__main__':unittest.main()
