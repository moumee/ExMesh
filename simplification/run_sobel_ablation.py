"""Same-code, one-factor-at-a-time Sobel ablation on scan24. Resume-safe.

Run with .ablation_env/Scripts/python.exe simplification/run_sobel_ablation.py.
Face IDs are recalculated by CPU raycasting at original source cameras.
All candidates use these same IDs, original mesh/texture, ratio and atlas options.
The importance default is therefore recalculated, not silently mixed with the
historical nvdiffrast importance file.
"""
from __future__ import annotations
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from moumee.moumee_img_utils import (
    aggregate_face_importance,
    compute_sobel_magnitude,
    normalize_magnitude,
)
from evaluate_mesh_preservation import load_mesh,sha256

OUT=ROOT/'outputs/DTU/scan24/ablation'
REF=ROOT/'outputs/DTU/scan24/export/simp_mod/mesh_iter_10000.obj'
TEX=REF.with_suffix('.png')
CAM=ROOT/'outputs/DTU/scan24/cameras.json'
IMAGES=ROOT/'workdir/DTU/scan24/images'


def configurations():
    rows=[]
    def add(label,group,lam=4,sigma=1,gk=5,sk=3):
        rows.append(dict(label=label,group=group,importance_lambda=lam,sigma=sigma,gaussian_kernel=gk,sobel_kernel=sk))
    for lam in (0,1,2,4,8): add(f'lambda{lam}','lambda',lam)
    for sigma in (.5,2): add(f'sigma{sigma:g}','sigma',sigma=sigma)
    for gk in (3,9,13): add(f'gaussian{gk}','gaussian_kernel',gk=gk)
    for sk in (5,7): add(f'sobel{sk}','sobel_kernel',sk=sk)
    add('sigma2_gaussian13','interaction',sigma=2,gk=13)
    return rows


def face_maps(ref,cameras):
    import open3d as o3d
    directory=OUT/'face_ids';directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for i,c in enumerate(cameras):
        path=directory/(c['img_name']+'.npy');paths.append(path)
        if path.exists(): continue
        h,w=c['height'],c['width']; y,x=np.mgrid[:h,:w]
        local=np.stack(((x+.5-w/2)/c['fx'],(y+.5-h/2)/c['fy'],np.ones((h,w))),-1)
        directions=local@np.asarray(c['rotation']).T
        origins=np.broadcast_to(c['position'],directions.shape)
        hit=ref['scene'].cast_rays(o3d.core.Tensor(np.concatenate((origins,directions),-1).astype(np.float32)))
        depth=hit['t_hit'].numpy();ids=hit['primitive_ids'].numpy().astype(np.int32)
        ids[~np.isfinite(depth)|(depth<.01)|(depth>100)]=-1
        np.save(path,ids)
        print(f'[face ids] {i+1}/{len(cameras)}',flush=True)
    return paths


def importance(config,cameras,ids_paths,num_faces):
    name=f"s{config['sigma']:g}_g{config['gaussian_kernel']}_k{config['sobel_kernel']}"
    directory=OUT/'importance';directory.mkdir(parents=True,exist_ok=True)
    dest=directory/(name+'.npy')
    if dest.exists(): return dest
    # Store only visible pixel values. This preserves exact global P95 while
    # bounding RAM and reusing the same score samples for face aggregation.
    sizes=[int(np.sum(np.load(p,mmap_mode='r')>=0)) for p in ids_paths]
    rawpath=directory/'working_magnitudes.dat'
    raw=np.memmap(rawpath,dtype=np.float32,mode='w+',shape=(sum(sizes),))
    start=0
    for i,(c,ip) in enumerate(zip(cameras,ids_paths)):
        image=np.asarray(Image.open(IMAGES/(c['img_name']+'.png')).convert('RGB'),np.float32)/255
        ids=np.load(ip,mmap_mode='r')
        if image.shape[:2]!=ids.shape: raise ValueError('Image/camera resolution mismatch')
        mag=compute_sobel_magnitude(image,config['sigma'],config['gaussian_kernel'],config['sobel_kernel'])
        raw[start:start+sizes[i]]=mag[ids>=0];start+=sizes[i]
    raw.flush()
    q=float(np.percentile(raw,95))
    def view_samples():
        start=0
        for i,ip in enumerate(ids_paths):
            ids=np.load(ip,mmap_mode='r');valid_ids=ids[ids>=0]
            normalized=normalize_magnitude(raw[start:start+sizes[i]],q)
            start+=sizes[i]
            yield valid_ids,normalized
    score,views=aggregate_face_importance(view_samples(),num_faces)
    np.save(dest,score)
    info=dict(config={k:config[k] for k in ('sigma','gaussian_kernel','sobel_kernel')},p95=q,
        invisible_faces=int(np.sum(views==0)),mean=float(score.mean()),std=float(score.std()),sha256=sha256(dest))
    (directory/(name+'.json')).write_text(json.dumps(info,indent=2))
    del raw
    rawpath.unlink()
    print(f'[importance] {name} P95={q:.6g}',flush=True)
    return dest


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for d in ('meshes','logs'): (OUT/d).mkdir(exist_ok=True)
    configs=configurations()
    manifest=dict(reference_sha256=sha256(REF),texture_sha256=sha256(TEX),camera_sha256=sha256(CAM),
        configurations=configs,ratio=.1,texture_size=1024,texture_padding=2,
        importance_method='Original 49 photos, Open3D source-camera face IDs, RGB Gaussian+Sobel, global visible-pixel P95, equal mean over visible views',
        evaluation='Same fixed 98 +/-5-degree novel poses; original mesh render reference; PSNR SSIM LPIPS; CPU raycast renderer',
        selection_caveat='Exploratory ablation on scan24; selecting best here is validation, not independent test performance')
    manifestpath=OUT/'manifest.json'
    if manifestpath.exists() and json.loads(manifestpath.read_text())!=manifest:
        raise ValueError('Inputs or configuration changed. Use a new output directory.')
    manifestpath.write_text(json.dumps(manifest,indent=2))
    cameras=json.loads(CAM.read_text())
    ref=load_mesh(REF)
    ids_paths=face_maps(ref,cameras)
    if '--prepare_only' in sys.argv:
        for config in configs: importance(config,cameras,ids_paths,len(ref['f']))
        return
    scores={c['label']:importance(c,cameras,ids_paths,len(ref['f'])) for c in configs}
    def generate(config):
        label=config['label'];mesh=OUT/'meshes'/(label+'.obj')
        score=scores[label]
        success=OUT/'meshes'/(label+'.complete.json')
        print(f'[generate] {label}',flush=True)
        if not success.exists():
            command=[sys.executable,str(ROOT/'simplification/wild_simplify_xatlas_modified.py'),'--input',str(REF),'--texture',str(TEX),
                '--output',str(mesh),'--face_importance',str(score),'--importance_lambda',str(config['importance_lambda']),
                '--ratio','0.1','--texture_size','1024','--texture_padding','2']
            start=time.perf_counter()
            with (OUT/'logs'/(label+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            success.write_text(json.dumps(dict(command=command,elapsed_seconds=time.perf_counter()-start,obj_sha256=sha256(mesh)),indent=2))
        print(f'[generated] {label}',flush=True)
    # Separate output directories make these subprocesses independent. Limit
    # concurrency to avoid oversubscribing memory during collapse-history baking.
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(generate,configs))
    evaldir=OUT/'evaluation'/'all'
    reportpath=evaldir/'summary.json'
    if not reportpath.exists():
        command=[sys.executable,str(ROOT/'simplification/evaluate_render_quality.py'),'--protocol','novel','--skip_geometry','--rgb_only',
            '--reference',str(REF),'--texture',f'reference={TEX}',
            '--cameras_json',str(CAM),'--output_dir',str(evaldir),'--resolution','2','--face_budget_tolerance','2']
        for c in configs:
            label=c['label'];mesh=OUT/'meshes'/(label+'.obj')
            command+=['--candidate',f'{label}={mesh}','--texture',f'{label}={mesh.with_name(label+"_texture.png")}']
        print('[evaluate] all 13 candidates at 98 fixed novel poses',flush=True)
        with (OUT/'logs'/'evaluation.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    metrics=[]
    for config in configs:
        label=config['label']
        report=json.loads(reportpath.read_text());result=report['render_mean'][label]
        base=report['render_mean']['lambda0']
        row=dict(**config,faces=report['candidates'][label]['faces'],psnr=result['psnr'],ssim=result['ssim'],lpips=result['lpips'],
            delta_psnr=result['psnr']-base['psnr'],delta_ssim=result['ssim']-base['ssim'],delta_lpips=result['lpips']-base['lpips'])
        if label!='lambda0':
            for m in ('psnr','ssim','lpips'):
                paired=report['paired_comparison'][label][m]
                row[m+'_improvement_ci_low'],row[m+'_improvement_ci_high']=paired['cluster_bootstrap_95ci']
                row[m+'_group_win_rate']=paired['source_view_win_rate']
        else:
            for m in ('psnr','ssim','lpips'):
                row[m+'_improvement_ci_low']=row[m+'_improvement_ci_high']=row[m+'_group_win_rate']=0
        metrics.append(row)
        with (OUT/'ablation_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
        (OUT/'ablation_results.json').write_text(json.dumps(metrics,indent=2))
        print(f'[result] {label} PSNR={result["psnr"]:.4f} SSIM={result["ssim"]:.6f} LPIPS={result["lpips"]:.6f}',flush=True)


if __name__=='__main__':main()
