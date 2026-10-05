"""Analytical checks: python -m unittest discover -s simplification -p test_mesh_preservation.py."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from evaluate_mesh_preservation import load_mesh, geometry_pair, sample_surface, perturb_cameras, paired_summary


class PreservationTests(unittest.TestCase):
    def test_parallel_surfaces_have_known_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            def square(name,z):
                p=Path(directory)/name
                p.write_text(f'v -10 -10 {z}\nv 10 -10 {z}\nv 10 10 {z}\nv -10 10 {z}\nf 1 2 3\nf 1 3 4\n')
                return load_mesh(p)
            a=square('a.obj',0); b=square('b.obj',.25)
            m=geometry_pair(a,b,2000,7)
            for key in ('mean_surface_distance','rms_surface_distance','sampled_hausdorff','hd95'):
                self.assertAlmostEqual(m[key],.25,places=5)
            self.assertLess(geometry_pair(a,a,2000,7)['sampled_hausdorff'],1e-5)
            np.testing.assert_array_equal(sample_surface(a,20,3),sample_surface(a,20,3))

    def test_camera_basis_is_rigid_and_pose_changes(self):
        c=dict(rotation=np.eye(3).tolist(),position=[0,0,-3],width=800,height=600,fx=900,fy=900,img_name='a')
        views=perturb_cameras([c],np.zeros(3),5,2)
        self.assertEqual(len(views),2)
        for v in views:
            r=np.asarray(v['rotation'])
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(v['position']),3)
            self.assertFalse(np.allclose(v['position'],c['position']))
            self.assertEqual(v['width'],400)

    def test_lpips_improvement_direction_and_cluster_ci(self):
        rows=[]
        for source in ('a','b','c'):
            for direction in (-1,1):
                row={'source_view':source}
                for metric in ('psnr','ssim','lpips','silhouette_iou'):
                    row['base_'+metric]=1
                    row['mod_'+metric]=.8 if metric=='lpips' else 1.2
                rows.append(row)
        result=paired_summary(rows,'base',['base','mod'])['mod']
        for metric in result:
            self.assertAlmostEqual(result[metric]['improvement_mean'],.2)
            self.assertEqual(result[metric]['source_view_win_rate'],1)


if __name__=='__main__': unittest.main()
