"""Reproducible frozen-setting comparison at extreme face budgets."""
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_mesh_preservation import load_mesh, texture_data, perturb_cameras, render_cpu
from run_sobel_ablation import ROOT, REF, TEX, CAM, sha256

OUT = ROOT / 'outputs/DTU/scan24/extreme_budgets'
BUDGETS = (1000, 500, 100)
SETTINGS = {'baseline': 0, 'sobel': 4}
SCORE = ROOT / 'outputs/DTU/scan24/ablation/importance/s1_g13_k3.npy'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ('meshes', 'logs', 'visuals'):
        (OUT / folder).mkdir(exist_ok=True)
    ref = load_mesh(REF)
    nfaces = len(ref['f'])
    manifest = dict(reference_sha256=sha256(REF), texture_sha256=sha256(TEX),
                    cameras_sha256=sha256(CAM), importance_sha256=sha256(SCORE),
                    budgets=BUDGETS, settings=SETTINGS, sigma=1, gaussian_kernel=13,
                    sobel_kernel=3, texture_size=1024, texture_padding=2,
                    visual_views=['0000_orbit_+5', '0024_orbit_+5', '0048_orbit_+5'],
                    visual_resolution=1, metric_resolution=2,
                    note='Frozen best setting selected at 20k on scan24; no tuning at low budgets. Same 98 nearby novel poses; original mesh renders are reference.')
    manifest_path = OUT / 'manifest.json'
    serialized = json.dumps(manifest, indent=2)
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(serialized):
        raise ValueError('Manifest changed; use a new output directory.')
    manifest_path.write_text(serialized)

    def generate(job):
        budget, label = job
        stem = f'{label}_{budget}'
        mesh = OUT / 'meshes' / (stem + '.obj')
        marker = mesh.with_suffix('.complete.json')
        if marker.exists():
            saved = json.loads(marker.read_text())
            if sha256(mesh) != saved['obj_sha256']:
                raise ValueError('Cached mesh hash mismatch')
            return
        # ceil avoids float rounding below the intended integer in int(N*ratio).
        ratio = np.nextafter(budget / nfaces, np.inf)
        command = [sys.executable, str(ROOT / 'simplification/wild_simplify_xatlas_modified.py'),
                   '--input', str(REF), '--texture', str(TEX), '--output', str(mesh),
                   '--face_importance', str(SCORE), '--importance_lambda', str(SETTINGS[label]),
                   '--ratio', str(ratio), '--texture_size', '1024', '--texture_padding', '2']
        print(f'[generate] {stem}', flush=True)
        start = time.perf_counter()
        with (OUT / 'logs' / (stem + '.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        marker.write_text(json.dumps(dict(command=command, elapsed_seconds=time.perf_counter()-start,
                                         obj_sha256=sha256(mesh), faces=len(load_mesh(mesh)['f'])), indent=2))
        print(f'[generated] {stem}', flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(generate, [(b, l) for b in BUDGETS for l in SETTINGS]))

    records = []
    for budget in BUDGETS:
        directory = OUT / 'evaluation' / str(budget)
        report_path = directory / 'summary.json'
        if not report_path.exists():
            command = [sys.executable, str(ROOT / 'simplification/evaluate_render_quality.py'),
                       '--protocol', 'novel', '--skip_geometry', '--rgb_only',
                       '--reference', str(REF), '--texture', f'reference={TEX}',
                       '--cameras_json', str(CAM), '--output_dir', str(directory),
                       '--resolution', '2', '--face_budget_tolerance', '2', '--baseline_label', 'baseline']
            for label in SETTINGS:
                mesh = OUT / 'meshes' / f'{label}_{budget}.obj'
                command += ['--candidate', f'{label}={mesh}', '--texture', f'{label}={mesh.with_name(mesh.stem+"_texture.png")}']
            print(f'[evaluate] {budget} faces, 98 poses', flush=True)
            with (OUT / 'logs' / f'evaluation_{budget}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(report_path.read_text())
        for label in SETTINGS:
            records.append(dict(target_faces=budget, method=label, actual_faces=report['candidates'][label]['faces'],
                                **report['render_mean'][label]))
    prior = json.loads((ROOT / 'outputs/DTU/scan24/ablation/evaluation/all/summary.json').read_text())
    for label, old in [('baseline', 'lambda0'), ('sobel', 'gaussian13')]:
        records.insert(0, dict(target_faces=20785, method=label, actual_faces=prior['candidates'][old]['faces'],
                               **{k:prior['render_mean'][old][k] for k in ('psnr','ssim','lpips')}))
    records.sort(key=lambda r: (-r['target_faces'], r['method']))
    (OUT / 'results.json').write_text(json.dumps(records, indent=2))
    dest = ROOT / 'deliverables/Sobel_extreme_budget_results.csv'
    with dest.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

    texture_data(ref, TEX)
    cameras = perturb_cameras(json.loads(CAM.read_text()), (ref['v'].min(0)+ref['v'].max(0))/2, 5, 1)
    selected = [c for c in cameras if c['img_name'] in manifest['visual_views']]
    meshes = {'reference': ref}
    for b in (20785, *BUDGETS):
        for label in SETTINGS:
            meshpath = (ROOT / 'outputs/DTU/scan24/ablation/meshes' / ('lambda0.obj' if label=='baseline' else 'gaussian13.obj')) if b==20785 else OUT/'meshes'/f'{label}_{b}.obj'
            mesh = load_mesh(meshpath)
            texture_data(mesh, meshpath.with_name(meshpath.stem+'_texture.png'))
            meshes[f'{label}_{b}'] = mesh
    for c in selected:
        for label, mesh in meshes.items():
            dest = OUT / 'visuals' / f'{c["img_name"]}_{label}.png'
            if not dest.exists():
                rgb, _ = render_cpu(mesh, c)
                Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(dest)
        print(f'[visual] {c["img_name"]}', flush=True)
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    main()
