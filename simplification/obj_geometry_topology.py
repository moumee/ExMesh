"""Restore authored OBJ position indices without merging coincident surfaces.

Open3D may split a position vertex for different UVs. Wild's geometry must use
the OBJ v/f position topology, while texture transfer keeps per-face UV corners.
For triangular OBJ only. Fail on a face/corner order mismatch instead of silently
misaligning the original face importance or texture correspondence.
"""
from pathlib import Path
import numpy as np


def restore_obj_position_topology(path, loaded_vertices, loaded_faces):
    path=Path(path)
    if path.suffix.lower()!='.obj':
        return loaded_vertices,loaded_faces,dict(restored=False,reason='non-OBJ input')
    positions=[];triangles=[]
    with path.open(encoding='utf-8-sig') as source:
        for line in source:
            fields=line.split('#',1)[0].split()
            if not fields:continue
            if fields[0]=='v':
                if len(fields)!=4:
                    raise ValueError('Position-topology restoration requires ordinary 3D OBJ v records')
                positions.append([float(x) for x in fields[1:]])
            elif fields[0]=='f':
                if len(fields)!=4:
                    raise ValueError('Position-topology restoration requires triangular OBJ faces')
                ids=[]
                for corner in fields[1:]:
                    index=int(corner.split('/')[0])
                    if index==0:raise ValueError('OBJ vertex index cannot be zero')
                    ids.append(index-1 if index>0 else len(positions)+index)
                triangles.append(ids)
    v=np.asarray(positions,dtype=np.float64);f=np.asarray(triangles,dtype=np.int64).reshape(-1,3)
    if v.ndim!=2 or v.shape[1]!=3 or not np.all(np.isfinite(v)) or np.any(f<0) or np.any(f>=len(v)):
        raise ValueError('Invalid OBJ geometry')
    if f.shape!=loaded_faces.shape:
        raise ValueError('OBJ/Open3D face count mismatch; cannot preserve UV/importance correspondence')
    delta=np.max(np.abs(v[f]-loaded_vertices[loaded_faces])) if len(f) else 0.
    scale=max(float(np.max(np.abs(v))),1.)
    if delta > scale*1e-8:
        raise ValueError('OBJ/Open3D corner order or coordinate mismatch; refusing unsafe UV/score reassignment')
    stats=dict(restored=True,loaded_vertices=len(loaded_vertices),obj_position_vertices=len(v),
               removed_uv_splits=len(loaded_vertices)-len(v),max_corner_coordinate_difference=float(delta),
               rule='authored OBJ position IDs, never merge separate OBJ IDs by coordinate')
    print('[OBJ position topology]',stats,flush=True)
    return v,f,stats
