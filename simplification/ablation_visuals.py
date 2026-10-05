"""Shared experiment visualization; independent of preservation cost terms."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCORE = ROOT / 'outputs/DTU/scan24/ablation/importance/s1_g13_k3.npy'


def clay(data, camera):
    """Render flat triangle shading to inspect geometry without texture."""
    import open3d as o3d
    w, h = camera['width'], camera['height']
    yy, xx = np.mgrid[:h, :w]
    local = np.stack([(xx+.5-w/2)/camera['fx'],
                      (yy+.5-h/2)/camera['fy'], np.ones_like(xx)], -1)
    directions = local @ np.asarray(camera['rotation']).T
    origin = np.broadcast_to(np.asarray(camera['position']), directions.shape)
    rays = np.concatenate([origin, directions], -1).astype(np.float32)
    hit = data['scene'].cast_rays(o3d.core.Tensor(rays))
    t = hit['t_hit'].numpy()
    valid = np.isfinite(t) & (t > .01) & (t < 100)
    normals = hit['primitive_normals'].numpy()[valid]
    view = -directions[valid]
    view /= np.linalg.norm(view, axis=1)[:, None]
    shade = .25 + .75*np.abs(np.sum(normals*view, axis=1))
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[valid] = shade[:, None]*np.array([.78, .78, .78])
    return rgb
