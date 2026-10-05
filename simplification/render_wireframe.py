"""Visible triangle wireframe from ray-hit barycentrics, with pixel-width edges.

Perspective-correct screen barycentrics prevent varying width with face depth.
Only the first visible triangle contributes: hidden edges are not drawn.
"""
import numpy as np


def wireframe(data,camera,width=1.0):
    import open3d as o3d
    w,h=camera['width'],camera['height'];yy,xx=np.mgrid[:h,:w]
    rotation=np.asarray(camera['rotation']);position=np.asarray(camera['position'])
    local=np.stack([(xx+.5-w/2)/camera['fx'],(yy+.5-h/2)/camera['fy'],np.ones_like(xx)],-1)
    directions=local@rotation.T
    rays=np.concatenate([np.broadcast_to(position,directions.shape),directions],-1).astype(np.float32)
    hit=data['scene'].cast_rays(o3d.core.Tensor(rays));depth=hit['t_hit'].numpy()
    valid=np.isfinite(depth)&(depth>.01)&(depth<100)
    ids=hit['primitive_ids'].numpy()[valid];uv=hit['primitive_uvs'].numpy()[valid]
    bary=np.column_stack([1-uv.sum(axis=1),uv])
    camera_v=(data['v']-position)@rotation
    tri=camera_v[data['f']]
    z=tri[:,:,2]
    screen=np.stack([tri[:,:,0]/np.maximum(z,1e-10)*camera['fx']+w/2,
                     tri[:,:,1]/np.maximum(z,1e-10)*camera['fy']+h/2],-1)
    e1=screen[:,1]-screen[:,0];e2=screen[:,2]-screen[:,0]
    double_area=np.abs(e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0])
    opposite=np.stack([np.linalg.norm(screen[:,1]-screen[:,2],axis=1),
                       np.linalg.norm(screen[:,2]-screen[:,0],axis=1),
                       np.linalg.norm(screen[:,0]-screen[:,1],axis=1)],-1)
    altitude=double_area[:,None]/np.maximum(opposite,1e-15)
    screen_bary=bary*z[ids];screen_bary/=np.maximum(screen_bary.sum(axis=1)[:,None],1e-15)
    distance=np.min(screen_bary*altitude[ids],axis=1)
    edge_alpha=np.clip(width/2+.5-distance,0,1)
    normals=hit['primitive_normals'].numpy()[valid]
    view=-directions[valid];view/=np.linalg.norm(view,axis=1)[:,None]
    base=(.72+.20*np.abs((normals*view).sum(axis=1)))[:,None]*np.ones((1,3))
    color=base*(1-edge_alpha[:,None])+.08*edge_alpha[:,None]
    rgb=np.ones((h,w,3),np.float32);rgb[valid]=color
    return rgb


def topology_diagnostic(data):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    v,inverse=np.unique(data['v'],axis=0,return_inverse=True);f=inverse[data['f']]
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True)
    graph=coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v))).tocsr()
    components=connected_components(graph,directed=False,return_labels=False)
    tri=v[f];lengths=np.stack([np.linalg.norm(tri[:,1]-tri[:,2],axis=1),np.linalg.norm(tri[:,2]-tri[:,0],axis=1),np.linalg.norm(tri[:,0]-tri[:,1],axis=1)],-1)
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    quality=4*np.sqrt(3)*area/np.maximum((lengths**2).sum(axis=1),1e-30)
    return dict(welded_vertices=len(v),edges=len(edges),components=int(components),boundary_edges=int((counts==1).sum()),
                nonmanifold_edges=int((counts>2).sum()),degenerate_faces=int((area<=1e-15).sum()),
                triangle_quality_p5=float(np.percentile(quality,5)),sliver_faces_pct=float(100*np.mean(quality<.1)))
