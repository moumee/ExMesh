import unittest
import numpy as np
import open3d as o3d
from render_wireframe import wireframe


class WireTests(unittest.TestCase):
    def data(self,v,f):
        scene=o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)),o3d.core.Tensor(f.astype(np.uint32)))
        return dict(v=v,f=f,scene=scene)

    def test_hidden_edges_are_not_visible(self):
        v=np.array([[-1.,-1,2],[1,-1,2],[1,1,2],[-1,1,2]])
        f=np.array([[0,1,2],[0,2,3]],np.int64)
        camera=dict(width=80,height=80,fx=80,fy=80,rotation=np.eye(3),position=[0,0,0])
        front=wireframe(self.data(v,f),camera)
        vv=np.concatenate([v,[[-.5,-.5,3],[.5,-.5,3],[0,.5,3]]])
        ff=np.concatenate([f,[[4,5,6]]])
        both=wireframe(self.data(vv,ff),camera)
        np.testing.assert_array_equal(front,both)
        self.assertLess(front[40,40].mean(),front[30,40].mean())


if __name__=='__main__':unittest.main()
