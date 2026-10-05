"""Fit one camera-center similarity; never fit geometry to the reference mesh."""
import json, itertools, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from run_realityscan_compare import ROOT, OUT

def main():
    ns = {'xcr':'http://www.capturingreality.com/ns/xcr/1.1#'}
    cameras = {int(c['img_name']): c for c in json.loads((ROOT/'outputs/DTU/scan24/cameras.json').read_text())}
    source, target, names, rotations, focals = [], [], [], [], []
    for p in sorted((OUT/'images_grouped').glob('*.xmp')):
        tree = ET.parse(p)
        desc = tree.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
        def value(name):
            element=tree.find('.//xcr:'+name,ns)
            return element.text if element is not None else desc.attrib['{'+ns['xcr']+'}'+name]
        position = np.fromstring(value('Position'), sep=' ')
        rotation = np.fromstring(value('Rotation'), sep=' ').reshape(3,3)
        source.append(position); target.append(cameras[int(p.stem)]['position'])
        names.append(cameras[int(p.stem)]['img_name']); rotations.append(rotation)
        focals.append(float(desc.attrib['{'+ns['xcr']+'}FocalLength35mm']))
    assert len(source) == 49
    a, b = np.array(source), np.array(target)
    ac, bc = a.mean(0), b.mean(0)
    aa, bb = a-ac, b-bc
    u, s, vt = np.linalg.svd(bb.T@aa/len(a))
    sign = np.ones(3); sign[-1] = np.linalg.det(u@vt)
    rotation = (u*sign)@vt
    scale = np.sum(s*sign)/np.mean(np.sum(aa*aa,axis=1))
    translation = bc-scale*rotation@ac
    residual = np.linalg.norm(a@(scale*rotation).T+translation-b,axis=1)
    record = dict(source='RealityScan XMP camera centers',target='ExMesh cameras.json camera centers',
        fit='All 49 camera centers, proper rotation + uniform scale + translation; no mesh ICP',
        rotation=rotation.tolist(),scale=float(scale),translation=translation.tolist(),
        camera_rms=float(np.sqrt(np.mean(residual**2))),camera_max=float(residual.max()),
        camera_relative_rms=float(np.sqrt(np.mean(residual**2))/np.sqrt(np.mean(np.sum(bb*bb,axis=1)))),
        camera_residuals=dict(zip(names,residual.tolist())), focal35mm_range=[min(focals),max(focals)])
    # Region contains the original ExMesh bounds, expanded 10%, transformed to
    # native RS coordinates and enclosed by an axis-aligned RS region.
    vertices=[]
    with (ROOT/'outputs/DTU/scan24/export/simp_mod/mesh_iter_10000.obj').open() as f:
        for line in f:
            if line.startswith('v '): vertices.append(list(map(float,line.split()[1:4])))
    vertices=np.array(vertices)
    lo,hi=vertices.min(0),vertices.max(0);pad=(hi-lo)*.10
    corners=np.array(list(itertools.product(*zip(lo-pad,hi+pad))))
    native=(corners-translation)@rotation/scale
    native_lo,native_hi=native.min(0),native.max(0)
    tree=ET.parse(OUT/'auto_region.rsbox');root=tree.getroot()
    root.set('yawPitchRoll','0 0 0')
    dimensions=' '.join(map(str,native_hi-native_lo))
    dimension_element=root.find('widthHeightDepth')
    if dimension_element is None:root.set('widthHeightDepth',dimensions)
    else:dimension_element.text=dimensions
    root.find('CentreEuclid/centre').text=' '.join(map(str,(native_hi+native_lo)/2))
    tree.write(OUT/'evaluation_region.rsbox',encoding='utf-8',xml_declaration=True)
    record.update(exmesh_bounds=[lo.tolist(),hi.tolist()],rs_region_bounds=[native_lo.tolist(),native_hi.tolist()])
    (OUT/'coordinate_alignment.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:record[k] for k in ('scale','camera_rms','camera_relative_rms','camera_max','focal35mm_range')},indent=2))

if __name__=='__main__':main()
