"""Fixed 90-degree collapse filter vs Wild and Sobel on scan24."""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from evaluate_mesh_preservation import load_mesh, sha256

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'outputs/DTU/scan24/normal_guard'
REF = ROOT/'outputs/DTU/scan24/export/simp_mod/mesh_iter_10000.obj'
TEX = REF.with_suffix('.png')
CAM = ROOT/'outputs/DTU/scan24/cameras.json'
OLD = ROOT/'outputs/DTU/scan24/sobel_current'
BUDGETS = (20000, 10000, 5000)
METHODS = ('wild', 'sobel', 'guarded')
SIMPLIFIER = ROOT/'simplification/wild_simplify_xatlas_verified.py'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def paths(method, budget):
    directory = OUT if method == 'guarded' else OLD
    p = directory/'meshes'/f'{method}_{budget}.obj'
    return p, p.with_name(p.stem+'_texture.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    for name in ('meshes', 'logs', 'evaluation', 'visuals'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    controls = {}
    for method in ('wild', 'sobel'):
        for budget in BUDGETS:
            mesh, texture = paths(method, budget)
            meta = json.loads(mesh.with_suffix('.complete.json').read_text())
            checks = dict(obj_sha256=sha256(mesh), texture_sha256=sha256(texture))
            if any(meta[key] != value for key, value in checks.items()):
                raise ValueError('Historical control hash mismatch')
            controls[f'{method}_{budget}'] = checks
    sources = [SIMPLIFIER, Path(__file__).resolve(),
               ROOT/'simplification/evaluate_mesh_preservation.py', ROOT/'simplification/obj_geometry_topology.py']
    manifest = dict(reference_sha256=sha256(REF), texture_sha256=sha256(TEX), cameras_sha256=sha256(CAM),
                    sources={p.name: sha256(p) for p in sources}, controls=controls,
                    budgets=BUDGETS, max_normal_change_degrees=90, min_relative_area=1e-12,
                    texture_size=1024, padding=2, geometry_samples=100000, seeds=[0, 1, 2],
                    novel_poses=98, resolution=2, timing='Concurrent generation, not isolated benchmarks',
                    interpretation='Local orientation filter. Single-scene source-mesh preservation.')
    manifest_path = OUT/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError('Input or code changed; use a new experiment directory')
    save(manifest_path, manifest)
    nfaces = len(load_mesh(REF)['f'])

    def generate(job):
        method, budget = job
        output = OUT/'meshes'/f'{method}_{budget}.obj'
        marker = output.with_suffix('.complete.json')
        if marker.exists():
            result = json.loads(marker.read_text())
            if (sha256(output) != result['obj_sha256'] or
                    sha256(output.with_name(output.stem+'_texture.png')) != result['texture_sha256']):
                raise ValueError('Cached result hash mismatch')
        else:
            command = [sys.executable, str(SIMPLIFIER), '--input', str(REF), '--texture', str(TEX),
                       '--output', str(output), '--ratio', str(np.nextafter(budget/nfaces, np.inf)),
                       '--method', method, '--texture_size', '1024', '--texture_padding', '2']
            print(f'[generate] {method} {budget}', flush=True)
            start = time.perf_counter()
            with (OUT/'logs'/f'{method}_{budget}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = dict(command=command, seconds=time.perf_counter()-start,
                          obj_sha256=sha256(output), texture_sha256=sha256(output.with_name(output.stem+'_texture.png')),
                          faces=len(load_mesh(output)['f']))
            save(marker, result)
        if abs(result['faces']-budget) > 2:
            raise ValueError(f'Target not reached: {method}, {budget}, actual={result["faces"]}')
        if method == 'wild' and any(result[k] != controls[f'wild_{budget}'][k]
                                   for k in ('obj_sha256', 'texture_sha256')):
            raise ValueError('Fresh Wild no longer matches control; stop comparison')
        print(f'[generated] {method} {budget}', flush=True)

    jobs = [('wild', 20000)] + [('guarded', b) for b in BUDGETS]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(generate, jobs))
    records = []
    for budget in BUDGETS:
        output = OUT/'evaluation'/str(budget)
        if not (output/'summary.json').exists():
            command = [sys.executable, str(ROOT/'simplification/evaluate_mesh_preservation.py'),
                       '--protocol', 'novel', '--reference', str(REF), '--texture', f'reference={TEX}',
                       '--cameras_json', str(CAM), '--output_dir', str(output),
                       '--baseline_label', 'wild', '--face_budget_tolerance', '2', '--resolution', '2']
            for method in METHODS:
                mesh, texture = paths(method, budget)
                command += ['--candidate', f'{method}={mesh}', '--texture', f'{method}={texture}']
            print(f'[evaluate] {budget}', flush=True)
            with (OUT/'logs'/f'evaluation_{budget}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((output/'summary.json').read_text())
        for method in METHODS:
            records.append(dict(target_faces=budget, method=method, faces=report['candidates'][method]['faces'],
                                **report['render_mean'][method], **report['geometry_mean'][method]))
    save(OUT/'results.json', records)
    with (ROOT/'deliverables/NormalGuard_비교결과.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print('[complete] normal guard comparison', flush=True)


if __name__ == '__main__':
    main()
