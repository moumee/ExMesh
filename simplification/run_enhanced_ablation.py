"""Ablation of simplification enhancements beyond the Sobel weight.

Axes (each run uses the verified Wild simplifier's own functions):
  cost   : wild | sobel (lambda 4) | vis (visibility-guided) | sobel_vis
  bake   : ss1 (legacy texel-center transfer) | ssK (KxK supersampled transfer)
  refine : none | tex | full  (appearance_refine.py, nvdiffrast, separate env)
One simplification per (cost, budget) is cached with its collapse history so
every bake/refine variant shares the identical collapse sequence.

  python run_enhanced_ablation.py --scene scan24 --stage simplify|bake|refine|evaluate|engine
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import wild_simplify_xatlas_verified as wild  # noqa: E402
from obj_geometry_topology import restore_obj_position_topology  # noqa: E402
from evaluate_mesh_preservation import sha256  # noqa: E402

SCENES = {
    "scan24": dict(reference=ROOT / "outputs/DTU/scan24/export/simp_mod/mesh_iter_10000.obj",
                   # Same score file as the stored Sobel lambda-4 control (sigma1, G13, K3).
                   importance=ROOT / "outputs/DTU/scan24/ablation/importance/s1_g13_k3.npy",
                   cameras=ROOT / "outputs/DTU/scan24/cameras.json",
                   eval_cameras=ROOT / "outputs/DTU/scan24/sobel_current/evaluation_cameras.json",
                   controls=ROOT / "outputs/DTU/scan24/sobel_current/meshes"),
    "scan83": dict(reference=ROOT / "outputs/DTU/scan83/export/mesh_iter_10000.obj",
                   importance=ROOT / "outputs/DTU/scan83/face_importance.npy",
                   cameras=ROOT / "outputs/DTU/scan83/cameras.json", eval_cameras=None, controls=None),
}
EXMESH_PY = Path(r"C:/Users/zzuzz/miniconda3/envs/exmesh/python.exe")
GPU_RUN = Path(r"D:/gamecontents_capstone/third_party/gpu_run.py")
LAMBDA = 4.0
VIS_FLOOR = 0.05


def load_reference(path):
    """Same loading path as wild_simplify_xatlas_verified.main()."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.triangles).copy()
    vertices, faces, _ = restore_obj_position_topology(path, vertices, faces)
    uvs = np.asarray(mesh.triangle_uvs).copy().reshape(-1, 3, 2)
    materials = np.zeros(len(faces), dtype=np.int32)
    texture = np.asarray(Image.open(Path(path).with_suffix(".png")).convert("RGB"))
    return vertices, faces, uvs, materials, [texture]


def face_weights(cost, vertices, faces, scene, out):
    weights = np.ones(len(faces))
    if "sobel" in cost:
        importance = np.load(SCENES[scene]["importance"]).astype(np.float64)
        assert importance.shape == (len(faces),)
        weights += LAMBDA * importance
    if "ppq" in cost:
        # Multi-view photometric parallax quadric (photometric_quadric_weights.py), lambda 1.
        weights = wild.photometric_face_weights(vertices, faces, np.load(out / "photometric_energy.npy"), 1.0)
    if "vis" in cost:
        cache = out / "visibility.npy"
        if not cache.exists():
            np.save(cache, wild.face_visibility(vertices, faces, 128))
        weights *= VIS_FLOOR + (1 - VIS_FLOOR) * np.load(cache)
    return weights


def simplify_job(scene, cost, budget, out):
    cache = out / "cache" / f"{cost}_{budget}.pkl"
    if cache.exists():
        return
    vertices, faces, *_ = load_reference(SCENES[scene]["reference"])
    weights = face_weights(cost, vertices, faces, scene, out)
    diagnostics = {}
    started = time.perf_counter()
    result = wild.simplify(vertices.copy(), faces.copy(), budget, 0.0025 * np.linalg.norm(np.ptp(vertices, 0)),
                           True, weights, diagnostics=diagnostics)
    seconds = time.perf_counter() - started
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(".tmp").open("wb") as f:
        pickle.dump(dict(result=result, seconds=seconds, diagnostics=diagnostics), f, protocol=5)
    cache.with_suffix(".tmp").replace(cache)


def bake_job(scene, cost, budget, supersample, out):
    mesh_path = out / "meshes" / f"{cost}_ss{supersample}_{budget}.obj"
    marker = mesh_path.with_suffix(".complete.json")
    if marker.exists():
        return
    vertices, faces, uvs, materials, textures = load_reference(SCENES[scene]["reference"])
    with (out / "cache" / f"{cost}_{budget}.pkl").open("rb") as f:
        cached = pickle.load(f)
    new_vertices, new_faces, active, history = cached["result"]
    started = time.perf_counter()
    tri_uvs, uv_vertices, uv_indices, atlas = wild.bake_texture(
        new_vertices, new_faces, active, history, uvs, materials, textures, 1024, "xatlas", 2, supersample)
    bake_seconds = time.perf_counter() - started
    wild.write_textured_obj(mesh_path, new_vertices, new_faces, tri_uvs, atlas, uv_vertices, uv_indices)
    marker.write_text(json.dumps(dict(
        cost=cost, budget=budget, supersample=supersample, faces=int(len(new_faces)),
        simplify_seconds=cached["seconds"], bake_seconds=bake_seconds, diagnostics=cached["diagnostics"],
        obj_sha256=sha256(mesh_path), texture_sha256=sha256(mesh_path.with_name(mesh_path.stem + "_texture.png"))),
        indent=2))


def refine_job(scene, name, mode, out, extra):
    source = out / "meshes" / f"{name}.obj"
    target = out / "meshes" / f"{name}_{mode}.obj"
    if target.with_suffix(".complete.json").exists():
        return
    reference = SCENES[scene]["reference"]
    command = [sys.executable, str(GPU_RUN), "--label", f"refine-{scene}-{name}-{mode}", "--",
               str(EXMESH_PY), str(HERE / "appearance_refine.py"), "--reference", str(reference),
               "--reference_texture", str(reference.with_suffix(".png")), "--input", str(source),
               "--output", str(target), "--mode", mode] + extra
    with (out / "logs" / f"refine_{name}_{mode}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def orbit_cameras(scene, out):
    path = out / "orbit_cameras.json"
    if not path.exists():
        subprocess.run([sys.executable, str(HERE / "make_orbit_cameras.py"), "--mesh",
                        str(SCENES[scene]["reference"]), "--output", str(path)], check=True)
    return path


def mesh_path(out, label, budget):
    if label.endswith(("_tex", "_full")):
        return out / "meshes" / (label.replace("_tex", f"_{budget}_tex").replace("_full", f"_{budget}_full") + ".obj")
    return out / "meshes" / f"{label}_{budget}.obj"


def evaluate(scene, budget, labels, out, tag, baseline, protocol="dtu"):
    target = out / "evaluation" / f"{tag}_{protocol}_{budget}"
    if (target / "summary.json").exists():
        return json.loads((target / "summary.json").read_text())
    reference = SCENES[scene]["reference"]
    command = [sys.executable, str(HERE / "evaluate_mesh_preservation.py"), "--protocol", "novel",
               "--reference", str(reference), "--texture", f"reference={reference.with_suffix('.png')}",
               "--cameras_json", str(SCENES[scene]["cameras"]), "--output_dir", str(target),
               "--baseline_label", baseline, "--face_budget_tolerance", "3", "--resolution", "2"]
    if protocol == "orbit":
        command += ["--evaluation_cameras_json", str(orbit_cameras(scene, out))]
    elif SCENES[scene]["eval_cameras"]:
        command += ["--evaluation_cameras_json", str(SCENES[scene]["eval_cameras"])]
    for label in labels:
        mesh = mesh_path(out, label, budget)
        command += ["--candidate", f"{label}={mesh}", "--texture", f"{label}={mesh.with_name(mesh.stem + '_texture.png')}"]
    target.mkdir(parents=True, exist_ok=True)
    with (out / "logs" / f"evaluate_{tag}_{protocol}_{budget}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((target / "summary.json").read_text())


def evaluate_engine(scene, budget, labels, out, tag, protocol):
    """Secondary engine-like render metrics at the poses of the matching CPU evaluation."""
    target = out / "evaluation" / f"engine_{protocol}_{budget}"
    if (target / "summary.json").exists():
        return json.loads((target / "summary.json").read_text())
    reference = SCENES[scene]["reference"]
    command = [sys.executable, str(GPU_RUN), "--label", f"engine-{scene}-{protocol}-{budget}", "--",
               str(EXMESH_PY), str(HERE / "evaluate_engine_like.py"), "--reference", str(reference),
               "--reference_texture", str(reference.with_suffix(".png")), "--output_dir", str(target),
               "--cameras_json", str(out / "evaluation" / f"{tag}_{protocol}_{budget}" / "evaluation_cameras.json")]
    for label in labels:
        command += ["--candidate", f"{label}={mesh_path(out, label, budget)}"]
    with (out / "logs" / f"engine_{protocol}_{budget}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((target / "summary.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--stage", choices=("simplify", "bake", "refine", "evaluate", "engine"), required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20000, 10000, 5000])
    parser.add_argument("--costs", nargs="+", default=["wild", "sobel", "vis", "sobel_vis"])
    parser.add_argument("--supersample", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--refine_names", nargs="*", default=[], help="mesh stems without budget, e.g. wild_ss3")
    parser.add_argument("--refine_modes", nargs="*", default=["tex", "full"])
    parser.add_argument("--refine_args", default="")
    parser.add_argument("--labels", nargs="*", default=[], help="evaluate: candidate labels without budget")
    parser.add_argument("--baseline", default="wild_ss1")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--protocol", choices=("dtu", "orbit"), default="dtu",
                        help="dtu: 98 poses at +/-5 deg of the capture cameras; orbit: 60 fixed full-sphere poses")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = ROOT / "outputs/DTU" / args.scene / "enhanced"
    for name in ("cache", "meshes", "logs", "evaluation"):
        (out / name).mkdir(parents=True, exist_ok=True)

    if args.stage == "simplify":
        # One process per job: simplify() is single-threaded Python.
        jobs = [(c, b) for c in args.costs for b in args.budgets
                if not (out / "cache" / f"{c}_{b}.pkl").exists()]
        if any("vis" in c for c, _ in jobs):
            face_weights("vis", *load_reference(SCENES[args.scene]["reference"])[:2], args.scene, out)

        def run(job):
            cost, budget = job
            with (out / "logs" / f"simplify_{cost}_{budget}.log").open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, __file__, "--scene", args.scene, "--stage", "simplify",
                                "--costs", cost, "--budgets", str(budget), "--workers", "0"],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"[simplified] {cost} {budget}", flush=True)
        if args.workers == 0:
            for cost, budget in jobs:
                simplify_job(args.scene, cost, budget, out)
        else:
            with ThreadPoolExecutor(args.workers) as pool:
                list(pool.map(run, jobs))
    elif args.stage == "bake":
        jobs = [(c, b, k) for c in args.costs for b in args.budgets for k in args.supersample]

        def run(job):
            cost, budget, k = job
            started = time.perf_counter()
            bake_job(args.scene, cost, budget, k, out)
            print(f"[baked] {cost} ss{k} {budget} {time.perf_counter()-started:.0f}s", flush=True)
        with ThreadPoolExecutor(max(1, args.workers)) as pool:
            list(pool.map(run, jobs))
    elif args.stage == "refine":
        for name in args.refine_names:
            for budget in args.budgets:
                for mode in args.refine_modes:
                    refine_job(args.scene, f"{name}_{budget}", mode, out, args.refine_args.split())
                    print(f"[refined] {name} {budget} {mode}", flush=True)
    elif args.stage == "engine":
        for budget in args.budgets:
            report = evaluate_engine(args.scene, budget, args.labels, out, args.tag, args.protocol)
            for label in args.labels:
                r = report["render_mean"][label]
                print(f"{budget:>6} {label:<24} PSNR={r['psnr']:.3f} SSIM={r['ssim']:.4f} LPIPS={r['lpips']:.4f}")
    else:
        rows = []
        for budget in args.budgets:
            report = evaluate(args.scene, budget, args.labels, out, args.tag, args.baseline, args.protocol)
            for label in args.labels:
                rows.append(dict(budget=budget, label=label, faces=report["candidates"][label]["faces"],
                                 **report["render_mean"][label], **report["geometry_mean"][label],
                                 paired=report["paired_comparison"].get(label)))
        (out / f"results_{args.tag}_{args.protocol}.json").write_text(json.dumps(rows, indent=2))
        for r in rows:
            print(f"{r['budget']:>6} {r['label']:<24} faces={r['faces']:>6} PSNR={r['psnr']:.3f} "
                  f"SSIM={r['ssim']:.4f} LPIPS={r['lpips']:.4f} mean_dist%={r['mean_surface_distance_pct_bbox']:.5f} "
                  f"HD95%={r['hd95_pct_bbox']:.5f}")


if __name__ == "__main__":
    main()
