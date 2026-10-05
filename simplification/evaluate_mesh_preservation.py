"""Reproducible, image-independent mesh preservation evaluation.

Sample-to-triangle distances approximate continuous Hausdorff distance; they
are NOT vertex-to-vertex distances or an exact continuous Hausdorff bound.
Novel renders use fixed camera perturbations and original mesh as reference.
This evaluates simplification preservation, not held-out photo reconstruction.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def load_mesh(path):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(f) or not np.isfinite(v).all():
        raise ValueError(f"Empty or nonfinite mesh: {path}")
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
    if area.sum() <= 0:
        raise ValueError(f"Mesh has no positive surface area: {path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)),
                        o3d.core.Tensor(f.astype(np.uint32)))
    return dict(mesh=mesh, v=v, f=f, area=area, scene=scene, path=str(Path(path).resolve()))


def sample_surface(data, count, seed):
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(data['f']), count, p=data['area']/data['area'].sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.column_stack((1-root, root*(1-uv[:, 1]), root*uv[:, 1]))
    return np.einsum('ni,nij->nj', bary, data['v'][data['f'][ids]]).astype(np.float32)


def distances(points, target):
    import open3d as o3d
    return target['scene'].compute_distance(o3d.core.Tensor(points)).numpy().astype(np.float64)


def geometry_pair(ref, cand, count, seed, ref_points=None):
    ref_points = sample_surface(ref, count, seed) if ref_points is None else ref_points
    a = distances(ref_points, cand)
    b = distances(sample_surface(cand, count, seed), ref)
    diag = float(np.linalg.norm(np.ptp(ref['v'], axis=0)))
    if diag <= 0:
        raise ValueError('Reference bounding box is degenerate')
    values = {
        'mean_surface_distance': float((a.mean()+b.mean())/2),
        'rms_surface_distance': float(np.sqrt((np.mean(a*a)+np.mean(b*b))/2)),
        'sampled_hausdorff': float(max(a.max(), b.max())),
        'hd95': float(max(np.percentile(a, 95), np.percentile(b, 95))),
        'reference_to_candidate_mean': float(a.mean()),
        'candidate_to_reference_mean': float(b.mean()),
    }
    values.update({k+'_pct_bbox': v/diag*100 for k, v in list(values.items())})
    return values


def perturb_cameras(cameras, center, degrees, scale):
    """Rigidly rotate camera position and basis about reference center/local up."""
    result = []
    for c in cameras:
        for angle in (-degrees, degrees):
            r = np.asarray(c['rotation'], dtype=float)
            axis = r[:, 1]/np.linalg.norm(r[:, 1])
            x,y,z = axis
            k = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            t = np.deg2rad(angle)
            rot = np.eye(3)+np.sin(t)*k+(1-np.cos(t))*(k@k)
            new = dict(c)
            new['position'] = (center+rot@(np.asarray(c['position'])-center)).tolist()
            new['rotation'] = (rot@r).tolist()
            new['width'] = max(32, round(c['width']/scale))
            new['height'] = max(32, round(c['height']/scale))
            new['fx'] = c['fx']*new['width']/c['width']
            new['fy'] = c['fy']*new['height']/c['height']
            new['img_name'] = f"{c['img_name']}_orbit_{angle:+g}"
            result.append(new)
    return result


def texture_data(data, explicit=None):
    mesh = data['mesh']
    uv = np.asarray(mesh.triangle_uvs)
    if len(uv) != 3*len(data['f']):
        raise ValueError(f"Missing per-corner UVs: {data['path']}")
    if explicit:
        tex = np.asarray(Image.open(explicit).convert('RGB'))
    elif len(mesh.textures) == 1:
        tex = np.asarray(mesh.textures[0])[:, :, :3]
    else:
        raise ValueError('Specify texture explicitly for missing/multiple materials')
    data['uv'] = uv.reshape(-1,3,2)
    data['tex'] = tex.astype(np.float32)/255


def render_cpu(data, camera, background=0):
    import open3d as o3d
    h,w = camera['height'], camera['width']
    yy,xx = np.mgrid[:h,:w]
    local = np.stack(((xx+.5-w/2)/camera['fx'], (yy+.5-h/2)/camera['fy'], np.ones((h,w))), -1)
    direction = local@np.asarray(camera['rotation']).T
    origins = np.broadcast_to(camera['position'], direction.shape)
    rays = np.concatenate((origins, direction), -1).astype(np.float32)
    hits = data['scene'].cast_rays(o3d.core.Tensor(rays))
    depth = hits['t_hit'].numpy()
    mask = np.isfinite(depth) & (depth > .01) & (depth < 100)
    image = np.full((h,w,3), background, np.float32)
    ids = hits['primitive_ids'].numpy()[mask].astype(np.int64)
    buv = hits['primitive_uvs'].numpy()[mask]
    bary = np.column_stack((1-buv.sum(1), buv))
    uv = np.clip(np.einsum('ni,nij->nj', bary, data['uv'][ids]),0,1)
    tex = data['tex']; th,tw = tex.shape[:2]
    x=uv[:,0]*(tw-1); y=(1-uv[:,1])*(th-1)
    x0=x.astype(int); y0=y.astype(int); x1=np.minimum(x0+1,tw-1); y1=np.minimum(y0+1,th-1)
    dx=(x-x0)[:,None]; dy=(y-y0)[:,None]
    image[mask]=(tex[y0,x0]*(1-dx)+tex[y0,x1]*dx)*(1-dy)+(tex[y1,x0]*(1-dx)+tex[y1,x1]*dx)*dy
    return image, mask


def paired_summary(rows, baseline, candidates, seed=20261004, rgb_only=False):
    out = {}; rng = np.random.default_rng(seed)
    # Cluster by source camera because +/- perturbations are dependent.
    groups = sorted({r['source_view'] for r in rows})
    for label in candidates:
        if label == baseline: continue
        out[label] = {}
        pairs=[('psnr',1),('ssim',1),('lpips',-1)]
        if not rgb_only: pairs.append(('silhouette_iou',1))
        for metric, sign in pairs:
            vals = np.array([np.mean([sign*(r[label+'_'+metric]-r[baseline+'_'+metric])
                         for r in rows if r['source_view']==g]) for g in groups])
            boots = np.mean(vals[rng.integers(0,len(vals),(2000,len(vals)))],axis=1)
            out[label][metric] = dict(improvement_mean=float(vals.mean()),
                cluster_bootstrap_95ci=np.percentile(boots,[2.5,97.5]).tolist(),
                source_view_win_rate=float(np.mean(vals>0)))
    return out


def lpips_against_features(model, candidate, reference_features):
    """Exact local LPIPS forward with shared, precomputed reference features."""
    import torch
    candidate_features=model.net(candidate)
    values=[layer((a-b)**2).mean((2,3),True)
            for layer,a,b in zip(model.lin,candidate_features,reference_features)]
    return torch.sum(torch.cat(values,0),0,True).mean()


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',choices=['geometry','novel'],required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--candidate',action='append',required=True,help='LABEL=OBJ, repeat for lambda sweep')
    p.add_argument('--texture',action='append',default=[],help='LABEL=PNG; reference is reserved')
    p.add_argument('--baseline_label',default='lambda0')
    p.add_argument('--face_budget_tolerance',type=int,default=0,help='Explicit allowed face-count difference, recorded in report')
    p.add_argument('--output_dir',required=True)
    p.add_argument('--cameras_json')
    p.add_argument('--evaluation_cameras_json',help='Reuse explicitly fixed novel poses instead of generating new ones')
    p.add_argument('--crop_reference',help='Optional mesh defining a shared render crop across reconstruction pipelines')
    p.add_argument('--crop_texture',help='Texture for the optional crop-reference mesh')
    p.add_argument('--samples',type=int,default=100000)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--orbit_degrees',type=float,default=5)
    p.add_argument('--resolution',type=float,default=2)
    p.add_argument('--crop_padding',type=int,default=16)
    p.add_argument('--save_renders',action='store_true')
    p.add_argument('--skip_geometry',action='store_true',help='Report only RGB and silhouette metrics for novel protocol')
    p.add_argument('--rgb_only',action='store_true',help='Omit silhouette metric from novel evaluation')
    p.add_argument('--white_background',action='store_true')
    args=p.parse_args()
    if args.protocol=='geometry' and args.skip_geometry:
        p.error('--skip_geometry requires --protocol novel')
    if args.samples<1 or args.resolution<=0 or args.crop_padding<0 or not 0<args.orbit_degrees<45:
        p.error('Invalid samples, resolution, crop padding or orbit angle')
    def mapping(items):
        result={}
        for item in items:
            label,path=item.split('=',1)
            if label in result or not label or any(c in label for c in '/\\:'):
                p.error('Labels must be unique safe filenames')
            result[label]=path
        return result
    paths=mapping(args.candidate); textures=mapping(args.texture)
    if 'reference' in paths: p.error('reference is a reserved label')
    if args.baseline_label not in paths: p.error('baseline_label is missing from candidates')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    ref=load_mesh(args.reference)
    meshes={k:load_mesh(v) for k,v in paths.items()}
    counts={k:len(v['f']) for k,v in meshes.items()}
    if args.face_budget_tolerance < 0 or max(counts.values())-min(counts.values()) > args.face_budget_tolerance:
        p.error(f'Unequal actual face budgets: {counts}')
    report=dict(protocol=args.protocol, purpose='simplification preservation, not held-out real-photo quality',
        evaluator_sha256=sha256(__file__),
        reference=dict(path=ref['path'],sha256=sha256(ref['path']),faces=len(ref['f'])),
        candidates={k:dict(path=v['path'],sha256=sha256(v['path']),faces=len(v['f']),ratio=len(v['f'])/len(ref['f'])) for k,v in meshes.items()},
        config=vars(args), geometry_definition='unsigned sample-to-triangle; area-uniform samples; HD95=max of directed P95; distances normalized by original bbox diagonal',
        hausdorff_caveat='sampled approximation, not exact continuous Hausdorff; maximum includes original vertices too',
        inference_caveat='Existing meshes may differ in fixes beyond Sobel. lambda was previously selected on this scene. Confirm on independent scenes with same-code lambda0.')
    geometry=[]
    for seed in ([] if args.skip_geometry else args.seeds):
        rp=sample_surface(ref,args.samples,seed)
        for label,mesh in meshes.items():
            print(f'[geometry] {label} seed={seed}',flush=True)
            values=geometry_pair(ref,mesh,args.samples,seed,rp)
            vertex_max=max(distances(ref['v'].astype(np.float32),mesh).max(),distances(mesh['v'].astype(np.float32),ref).max())
            values['sampled_hausdorff']=max(values['sampled_hausdorff'],float(vertex_max))
            values['sampled_hausdorff_pct_bbox']=values['sampled_hausdorff']/np.linalg.norm(np.ptp(ref['v'],axis=0))*100
            geometry.append(dict(candidate=label,seed=seed,**values))
    if geometry:
        write_csv(out/'geometry_per_seed.csv',geometry)
        report['geometry_mean']={label:{key:float(np.mean([r[key] for r in geometry if r['candidate']==label]))
            for key in geometry[0] if key not in ('candidate','seed')} for label in meshes}
    if args.protocol=='novel':
        if not args.cameras_json: p.error('novel requires cameras_json')
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
        import torch
        from utils.loss_utils import ssim
        from lpipsPyTorch import LPIPS
        device='cuda' if torch.cuda.is_available() else 'cpu'
        lp=LPIPS('vgg','0.1').to(device).eval()
        all_meshes={'reference':ref,**meshes}
        for label,mesh in all_meshes.items(): texture_data(mesh,textures.get(label))
        report['textures']={label:dict(shape=mesh['tex'].shape,sha256=hashlib.sha256(mesh['tex'].tobytes()).hexdigest()) for label,mesh in all_meshes.items()}
        source=json.loads(Path(args.cameras_json).read_text())
        cameras=(json.loads(Path(args.evaluation_cameras_json).read_text()) if args.evaluation_cameras_json else
                 perturb_cameras(source,(ref['v'].min(0)+ref['v'].max(0))/2,args.orbit_degrees,args.resolution))
        crop_ref=(load_mesh(args.crop_reference) if args.crop_reference and
                  Path(args.crop_reference).resolve()!=Path(args.reference).resolve() else ref)
        if crop_ref is not ref: texture_data(crop_ref,args.crop_texture)
        if not cameras or len({c['img_name'] for c in cameras})!=len(cameras):
            p.error('Evaluation camera names must be nonempty and unique')
        # Reject poses that coincide with any original pose.
        for c in cameras:
            if any(np.allclose(c['position'],s['position'],atol=1e-6) and np.allclose(c['rotation'],s['rotation'],atol=1e-6) for s in source):
                raise ValueError('Perturbed camera coincides with a source pose')
        (out/'evaluation_cameras.json').write_text(json.dumps(cameras,indent=2))
        rows=[]
        with torch.no_grad():
            for i,c in enumerate(cameras):
                print(f'[novel] {i+1}/{len(cameras)} {c["img_name"]}',flush=True)
                ri,rm=render_cpu(ref,c,int(args.white_background))
                crop_mask=rm if crop_ref is ref else render_cpu(crop_ref,c,int(args.white_background))[1]
                ys,xs=np.where(crop_mask)
                if not len(xs): raise ValueError(f'Empty reference view: {c["img_name"]}')
                pad=args.crop_padding; h,w=rm.shape
                bbox=(max(0,ys.min()-pad),min(h,ys.max()+pad+1),max(0,xs.min()-pad),min(w,xs.max()+pad+1))
                y0,y1,x0,x1=bbox
                def tensor(im): return torch.from_numpy(im[y0:y1,x0:x1].copy()).permute(2,0,1)[None].to(device)
                rt=tensor(ri)
                reference_features=lp.net(rt*2-1)
                row=dict(view=c['img_name'],source_view=c['img_name'].split('_orbit_')[0],reference_coverage=float(rm.mean()))
                for label,mesh in meshes.items():
                    ci,cm=render_cpu(mesh,c,int(args.white_background)); ct=tensor(ci)
                    mse=float(((ct-rt)**2).mean()); union=(cm|rm).sum()
                    row.update({label+'_psnr':float(-10*np.log10(max(mse,1e-15))),label+'_ssim':float(ssim(ct,rt)),
                        label+'_lpips':float(lpips_against_features(lp,ct*2-1,reference_features))})
                    if not args.rgb_only: row[label+'_silhouette_iou']=float((cm&rm).sum()/union)
                    if args.save_renders:
                        directory=out/'renders'/label; directory.mkdir(parents=True,exist_ok=True)
                        Image.fromarray((np.clip(ci,0,1)*255).astype(np.uint8)).save(directory/(c['img_name']+'.png'))
                if args.save_renders:
                    directory=out/'renders'/'reference';directory.mkdir(parents=True,exist_ok=True)
                    Image.fromarray((ri*255).astype(np.uint8)).save(directory/(c['img_name']+'.png'))
                rows.append(row)
        write_csv(out/'per_view_metrics.csv',rows)
        metric_names=['psnr','ssim','lpips']+([] if args.rgb_only else ['silhouette_iou'])
        report['render_mean']={label:{m:float(np.mean([r[label+'_'+m] for r in rows])) for m in metric_names} for label in meshes}
        report['paired_comparison']=paired_summary(rows,args.baseline_label,meshes,rgb_only=args.rgb_only)
        report['render_definition']='CPU pinhole raycast, bilinear texture, no antialiasing, common reference bbox crop; LPIPS input [-1,1]; identical renderer for all meshes. Do not compare absolute values to legacy nvdiffrast results.'
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('geometry_mean','render_mean','paired_comparison') if k in report},indent=2),flush=True)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__=='__main__': main()
