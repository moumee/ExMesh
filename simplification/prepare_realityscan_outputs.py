"""Apply camera-only rigid similarity and render comparable RGB/topology views."""
import json, shutil, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
from run_realityscan_compare import ROOT, OUT, sha256
from run_sobel_ablation import REF, TEX, CAM
from evaluate_mesh_preservation import load_mesh, texture_data, render_cpu, perturb_cameras
from ablation_visuals import clay
from render_wireframe import wireframe, topology_diagnostic

CONTROL=ROOT/'outputs/DTU/scan24/sobel_current'

def prepare():
    record=json.loads((OUT/'coordinate_alignment.json').read_text())
    rotation=np.array(record['rotation']);scale=record['scale'];translation=np.array(record['translation'])
    (OUT/'meshes').mkdir(exist_ok=True)
    assets={}
    names=['RS_original_fixed','RS_20000','RS_10000','RS_5000']
    names += [f'RS_same_{b}' for b in (20000,10000,5000) if (OUT/'raw'/f'RS_same_{b}.obj').exists()]
    for name in names:
        src=OUT/'raw'/f'{name}.obj';dest=OUT/'meshes'/src.name
        info=ET.fromstring('<root>'+src.with_suffix('.obj.rsInfo').read_text()+'</root>')
        assert info.find('Model').attrib['transformToModel']=='1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'
        assert info.find('ModelExport').attrib['settingsRotation']=='0 0 0'
        faces=0; vertices=0
        with src.open() as reader,dest.open('w') as writer:
            for line in reader:
                if line.startswith('v '):
                    fields=line.split();v=scale*rotation@np.array(list(map(float,fields[1:4])))+translation
                    line='v '+' '.join(f'{x:.12g}' for x in v)+'\n';vertices+=1
                elif line.startswith('f '):
                    assert len(line.split())==4,'Non-triangular output'
                    faces+=1
                writer.write(line)
        mtl=src.with_suffix('.mtl');shutil.copy2(mtl,dest.with_suffix('.mtl'))
        textures=[line.split(maxsplit=1)[1].strip() for line in mtl.read_text().splitlines() if line.startswith('map_Kd ')]
        assert len(textures)==1,textures
        texture=OUT/'meshes'/textures[0];shutil.copy2(src.parent/textures[0],texture)
        size=Image.open(texture).size
        assert size==((4096,4096) if name=='RS_original_fixed' else (1024,1024)),(name,size)
        if name!='RS_original_fixed':assert abs(faces-int(name.split('_')[-1]))<=(20 if name.startswith('RS_same_') else 2)
        assets[name]=dict(mesh=str(dest),texture=str(texture),faces=faces,vertices=vertices,
            raw_obj_sha256=sha256(src),aligned_obj_sha256=sha256(dest),texture_sha256=sha256(texture),texture_size=size)
    cameras=CONTROL/'evaluation/20000/evaluation_cameras.json'
    shutil.copy2(cameras,OUT/'evaluation_cameras.json')
    assert len(json.loads(cameras.read_text()))==98
    (OUT/'assets.json').write_text(json.dumps(assets,indent=2))
    imported=load_mesh(OUT/'raw/ExMesh_imported_reference.obj')
    original=load_mesh(REF)
    v=scale*imported['v']@rotation.T+translation
    from scipy.spatial import cKDTree
    unique,inverse=np.unique(original['v'],axis=0,return_inverse=True)
    distances,ids=cKDTree(unique).query(v)
    f1=np.sort(inverse[original['f']],axis=1)
    f2=np.sort(ids[imported['f']],axis=1)
    f1=f1[np.lexsort(f1.T[::-1])];f2=f2[np.lexsort(f2.T[::-1])]
    same_triangles=f1.shape==f2.shape and np.array_equal(f1,f2)
    assert distances.max()<1e-6 and same_triangles,'RealityScan import changed source geometry'
    (OUT/'same_input_verification.json').write_text(json.dumps(dict(
        original_faces=len(f1),imported_faces=len(f2),maximum_vertex_error=float(distances.max()),
        triangle_geometry_identical=bool(same_triangles),coordinate_fit='camera centers only, no geometry fitting'),indent=2))
    print(json.dumps({k:dict(faces=v['faces'],vertices=v['vertices'],texture=v['texture_size']) for k,v in assets.items()},indent=2),flush=True)

def render():
    assets=json.loads((OUT/'assets.json').read_text())
    ref=load_mesh(REF)
    cameras=perturb_cameras(json.loads(CAM.read_text()),(ref['v'].min(0)+ref['v'].max(0))/2,5,1)
    selected=[c for c in cameras if c['img_name'] in ('0000_orbit_+5','0024_orbit_+5','0048_orbit_+5')]
    (OUT/'visuals').mkdir(exist_ok=True)
    sources={'exmesh_original':dict(mesh=str(REF),texture=str(TEX)),**assets}
    for b in (20000,10000,5000):
        for m in ('wild','sobel'):
            p=CONTROL/'meshes'/f'{m}_{b}.obj'
            marker=json.loads(p.with_suffix('.complete.json').read_text())
            assert marker['obj_sha256']==sha256(p)
            sources[f'{m}_{b}']=dict(mesh=str(p),texture=str(p.with_name(p.stem+'_texture.png')))
    diagnostics=[]
    for name,source in sources.items():
        data=load_mesh(source['mesh']);texture_data(data,source['texture'])
        diagnostics.append(dict(method=name,faces=len(data['f']),**topology_diagnostic(data)))
        for camera in selected:
            view=camera['img_name']
            for mode in ('rgb','clay','wire'):
                path=OUT/'visuals'/f'{view}_{name}_{mode}.png'
                if path.exists():continue
                rgb=render_cpu(data,camera)[0] if mode=='rgb' else clay(data,camera) if mode=='clay' else wireframe(data,camera)
                image=Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8))
                image.save(path)
                if view=='0024_orbit_+5':image.crop((820,240,1340,850)).save(path.with_name(path.stem+'_crop.png'))
        print('[render]',name,flush=True)
    (OUT/'topology.json').write_text(json.dumps(diagnostics,indent=2))

if __name__=='__main__':
    prepare();render()
