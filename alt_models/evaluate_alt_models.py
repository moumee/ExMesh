"""Compare reconstruction sources (ExMesh vs image-to-3D models) on a DTU scene.

align   : similarity transform (scale, rotation, translation) of a generated mesh
          into ExMesh's normalized scene frame, fitted to the DTU GT scan (random
          rotation search + trimmed scaled ICP). This is ORACLE alignment that
          uses ground truth, so it favours the generative models.
chamfer : the repo's DTU protocol (mask culling + scripts/eval_dtu/eval.py), mm.
photo   : render the aligned mesh at every capture camera and compare with the
          real photo inside the DTU mask (PSNR over mask pixels, SSIM/LPIPS on
          the masked crop, silhouette IoU). ExMesh was optimized on these photos;
          image-to-3D models saw one view. This measures fidelity to the real
          object, not held-out generalization.

  python evaluate_alt_models.py --scene scan83 --model triposr --stage align|chamfer|photo
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "simplification"))
from evaluate_mesh_preservation import load_mesh, render_cpu, sample_surface, texture_data  # noqa: E402

INPUT_VIEW = {"scan83": 51, "scan24": 23}


def scene_paths(scene):
    data = ROOT / "workdir/DTU" / scene
    return dict(data=data, cameras_npz=data / "cameras.npz", cameras_json=ROOT / "outputs/DTU" / scene / "cameras.json",
                gt=ROOT / f"workdir/DTU/Points/stl/stl{int(scene[4:]):03d}_total.ply",
                exmesh=ROOT / "outputs/DTU" / scene / ("export/simp_mod/mesh_iter_10000.obj" if scene == "scan24" else "export/mesh_iter_10000.obj"))


def model_dir(scene, model):
    return ROOT / "outputs/alt_models" / scene / model


def projection_matrices(npz_path):
    cams = np.load(npz_path)
    count = len([k for k in cams.files if k.startswith("world_mat_") and "inv" not in k])
    scale = cams["scale_mat_0"].astype(np.float64)
    return [(cams[f"world_mat_{i}"] @ cams[f"scale_mat_{i}"])[:3, :4] for i in range(count)], scale


def mask_cull(points, scene):
    """Keep points inside every dilated DTU mask they project into (eval.py's culling rule)."""
    from skimage.morphology import binary_dilation, disk
    projections, _ = projection_matrices(scene_paths(scene)["cameras_npz"])
    keep = np.ones(len(points), bool)
    homogeneous = np.c_[points, np.ones(len(points))]
    for i, matrix in enumerate(projections):
        mask = np.asarray(Image.open(scene_paths(scene)["data"] / f"mask/{i:03d}.png").convert("L")) > 0
        mask = binary_dilation(mask, disk(24))
        h, w = mask.shape
        uvw = homogeneous @ matrix.T
        u, v = uvw[:, 0] / uvw[:, 2], uvw[:, 1] / uvw[:, 2]
        inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1) & (uvw[:, 2] > 0)
        hit = np.zeros(len(points), bool)
        hit[inside] = mask[np.rint(v[inside]).astype(int), np.rint(u[inside]).astype(int)]
        keep &= hit | ~inside
    return keep


def gt_points(scene, count=60000, seed=0):
    import open3d as o3d
    cache = ROOT / "outputs/alt_models" / scene / "gt_object_points.npy"
    if cache.exists():
        return np.load(cache)
    _, scale = projection_matrices(scene_paths(scene)["cameras_npz"])
    cloud = o3d.io.read_point_cloud(str(scene_paths(scene)["gt"])).voxel_down_sample(1.0)
    points = (np.asarray(cloud.points) - scale[:3, 3]) / scale[0, 0]
    points = points[mask_cull(points, scene)]
    points = points[np.random.default_rng(seed).choice(len(points), min(count, len(points)), replace=False)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, points)
    return points


def random_rotations(count, rng):
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def umeyama(source, target):
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(xt.T @ xs / len(source))
    sign = np.eye(3)
    sign[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / xs.var(0).sum()
    return scale, rotation, mu_t - scale * rotation @ mu_s


def trimmed(values, keep=0.9):
    values = np.sort(values)
    return values[:max(1, int(len(values) * keep))].mean()


def icp(source, target, scale, rotation, translation, iterations=40):
    """Scaled ICP with GT->mesh correspondences: every GT point must be explained."""
    target_tree = cKDTree(target)
    for _ in range(iterations):
        moved = scale * source @ rotation.T + translation
        distance, index = cKDTree(moved).query(target)
        keep = distance <= np.quantile(distance, 0.9)
        scale, rotation, translation = umeyama(source[index[keep]], target[keep])
    moved = scale * source @ rotation.T + translation
    objective = trimmed(cKDTree(moved).query(target)[0]) + trimmed(target_tree.query(moved)[0], 0.5)
    return objective, scale, rotation, translation


def align(scene, model, mesh_path):
    out = model_dir(scene, model)
    data = load_mesh(mesh_path)
    rng = np.random.default_rng(0)
    source = sample_surface(data, 20000, 0).astype(np.float64)
    target = gt_points(scene)
    target_small = target[rng.choice(len(target), 4000, replace=False)]
    source_small = source[rng.choice(len(source), 4000, replace=False)]
    base_scale = np.sqrt(((target - target.mean(0)) ** 2).sum(1).mean() / ((source - source.mean(0)) ** 2).sum(1).mean())
    started = time.perf_counter()
    candidates = []
    target_tree = cKDTree(target_small)
    for rotation in random_rotations(3000, rng):
        translation = target.mean(0) - base_scale * rotation @ source.mean(0)
        moved = base_scale * source_small @ rotation.T + translation
        score = trimmed(cKDTree(moved).query(target_small)[0]) + trimmed(target_tree.query(moved)[0], 0.5)
        candidates.append((score, rotation, translation))
    candidates.sort(key=lambda c: c[0])
    best = min((icp(source_small, target_small, base_scale, r, t, 30) for _, r, t in candidates[:24]),
               key=lambda c: c[0])
    objective, scale, rotation, translation = icp(source, target, *best[1:], iterations=40)
    write_transformed_obj(mesh_path, out / "aligned.obj", scale, rotation, translation)
    info = dict(source=str(mesh_path), objective=objective, scale=scale, rotation=rotation.tolist(),
                translation=translation.tolist(), seconds=time.perf_counter() - started,
                method="3000 random rotations, top-24 trimmed scaled ICP on 4k points, final ICP on 20k/60k points; "
                       "GT points culled by dilated DTU masks; oracle alignment")
    (out / "alignment.json").write_text(json.dumps(info, indent=2))
    print(f"[align] {scene} {model}: objective={objective:.5f} scale={scale:.4f} ({info['seconds']:.0f}s)", flush=True)


def write_transformed_obj(source, target, scale, rotation, translation):
    """Apply the similarity to `v` lines only; UVs, faces and materials are untouched."""
    source, target = Path(source), Path(target)
    lines = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            p = scale * rotation @ np.array([float(x) for x in parts[1:4]]) + translation
            line = " ".join(["v", *(f"{x:.9g}" for x in p), *parts[4:]])
        elif line.startswith("vn "):
            parts = line.split()
            n = rotation @ np.array([float(x) for x in parts[1:4]])
            line = " ".join(["vn", *(f"{x:.6g}" for x in n)])
        elif line.startswith("mtllib"):
            mtl = source.parent / line.split(maxsplit=1)[1].strip()
            if mtl.exists() and mtl.resolve() != (target.parent / mtl.name).resolve():
                shutil.copy2(mtl, target.parent / mtl.name)
                for entry in mtl.read_text(errors="ignore").splitlines():
                    parts = entry.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].lower() == "map_kd" and (source.parent / parts[1]).exists():
                        if (source.parent / parts[1]).resolve() != (target.parent / parts[1]).resolve():
                            shutil.copy2(source.parent / parts[1], target.parent / parts[1])
        lines.append(line)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chamfer(scene, model, mesh_path, python):
    out = model_dir(scene, model) / "dtu_eval"
    out.mkdir(parents=True, exist_ok=True)
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    ply = out / "mesh_normalized.ply"
    o3d.io.write_triangle_mesh(str(ply), o3d.geometry.TriangleMesh(mesh.vertices, mesh.triangles))
    scan = scene[4:]
    with (out / "eval.log").open("w", encoding="utf-8") as log:
        subprocess.run([python, "scripts/eval_dtu/evaluate_single_scene.py", "--input_mesh", str(ply), "--scan_id", scan,
                        "--output_dir", str(out), "--mask_dir", "workdir/DTU", "--DTU", "workdir/DTU"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([python, "scripts/eval_dtu/eval.py", "--data", str(out / "culled_mesh.ply"), "--scan", scan,
                        "--mode", "mesh", "--dataset_dir", "workdir/DTU", "--vis_out_dir", str(out)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((out / "results.json").read_text())
    print(f"[chamfer] {scene} {model}: {result}", flush=True)
    return result


def photo(scene, model, mesh_path, resolution=2, label=None):
    import torch
    from utils.loss_utils import ssim
    from lpipsPyTorch import LPIPS
    label = label or model
    out = model_dir(scene, model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips = LPIPS("vgg", "0.1").to(device).eval()
    data = load_mesh(mesh_path)
    texture = texture_for(mesh_path)
    textured = texture is not None and len(np.asarray(data["mesh"].triangle_uvs)) == 3 * len(data["f"])
    if textured:
        texture_data(data, texture)
    cameras = json.loads(scene_paths(scene)["cameras_json"].read_text())
    rows = []
    for camera in cameras:
        name = camera["img_name"]
        # DTU images are principal-point-centered crops of the 1600x1200 frames and carry the
        # object mask as alpha; the separate mask/*.png files are uncropped (resizing them misaligns).
        rgba = Image.open(scene_paths(scene)["data"] / "images" / f"{name}.png").convert("RGBA")
        size = (max(32, round(camera["width"] / resolution)), max(32, round(camera["height"] / resolution)))
        cam = dict(camera, width=size[0], height=size[1], fx=camera["fx"] * size[0] / camera["width"],
                   fy=camera["fy"] * size[1] / camera["height"])
        target = np.asarray(rgba.convert("RGB").resize(size, Image.BOX), np.float32) / 255
        mask = np.asarray(rgba.getchannel("A").resize(size, Image.NEAREST)) > 127
        if textured:
            image, alpha = render_cpu(data, cam)
        else:
            image, alpha = None, render_cpu_mask(data, cam)
        union = (alpha | mask).sum()
        row = dict(view=name, input_view=int(name) == INPUT_VIEW[scene], silhouette_iou=float((alpha & mask).sum() / union))
        if textured:
            ys, xs = np.where(mask)
            y0, y1, x0, x1 = max(0, ys.min() - 16), ys.max() + 17, max(0, xs.min() - 16), xs.max() + 17
            a = torch.from_numpy((image * mask[..., None])[y0:y1, x0:x1].copy()).permute(2, 0, 1)[None].to(device)
            b = torch.from_numpy((target * mask[..., None])[y0:y1, x0:x1].copy()).permute(2, 0, 1)[None].to(device)
            mse = float(((image - target) ** 2)[mask].mean())
            with torch.no_grad():
                row.update(psnr=float(-10 * np.log10(max(mse, 1e-12))), ssim=float(ssim(a, b)),
                           lpips=float(lpips(a * 2 - 1, b * 2 - 1)))
        rows.append(row)
    keys = [k for k in ("psnr", "ssim", "lpips", "silhouette_iou") if k in rows[0]]
    summary = dict(label=label, mesh=str(mesh_path), faces=int(len(data["f"])), textured=textured,
                   resolution_divisor=resolution, views=len(rows),
                   all_views={k: float(np.mean([r[k] for r in rows])) for k in keys},
                   excluding_input_view={k: float(np.mean([r[k] for r in rows if not r["input_view"]])) for k in keys},
                   input_view={k: float(np.mean([r[k] for r in rows if r["input_view"]])) for k in keys},
                   definition="PSNR over DTU-mask pixels; SSIM/LPIPS on mask-multiplied bbox crop; "
                              "IoU of rendered alpha vs undilated DTU mask; CPU raycast, bilinear texture")
    (out / f"photo_{label}.json").write_text(json.dumps(dict(summary=summary, per_view=rows), indent=2))
    print(f"[photo] {scene} {label}: {summary['excluding_input_view']}", flush=True)
    return summary


def texture_for(obj_path):
    """map_Kd of the OBJ's MTL (Open3D also reports an empty default texture slot)."""
    obj_path = Path(obj_path)
    for line in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("mtllib"):
            mtl = obj_path.parent / line.split(maxsplit=1)[1].strip()
            for entry in mtl.read_text(errors="ignore").splitlines() if mtl.exists() else []:
                parts = entry.split(maxsplit=1)
                if len(parts) == 2 and parts[0].lower() == "map_kd":
                    return obj_path.parent / parts[1].strip()
    return None


def render_cpu_mask(data, camera):
    import open3d as o3d
    h, w = camera["height"], camera["width"]
    yy, xx = np.mgrid[:h, :w]
    local = np.stack(((xx + .5 - w / 2) / camera["fx"], (yy + .5 - h / 2) / camera["fy"], np.ones((h, w))), -1)
    direction = local @ np.asarray(camera["rotation"]).T
    rays = np.concatenate((np.broadcast_to(camera["position"], direction.shape), direction), -1).astype(np.float32)
    depth = data["scene"].cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return np.isfinite(depth) & (depth > .01) & (depth < 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=INPUT_VIEW)
    parser.add_argument("--model", required=True, help="exmesh | triposr | hunyuan3d2 | <label>")
    parser.add_argument("--stage", required=True, choices=("align", "chamfer", "photo"))
    parser.add_argument("--mesh", help="Override mesh path (default: model output, aligned.obj after align)")
    parser.add_argument("--label")
    parser.add_argument("--python", default=sys.executable, help="Interpreter with torch+trimesh for DTU culling")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    directory = model_dir(args.scene, args.model)
    directory.mkdir(parents=True, exist_ok=True)
    if args.model == "exmesh":
        default = scene_paths(args.scene)["exmesh"]
    else:
        default = directory / "aligned.obj" if args.stage != "align" else directory / "mesh.obj"
    mesh = Path(args.mesh) if args.mesh else default
    if args.stage == "align":
        align(args.scene, args.model, mesh)
    elif args.stage == "chamfer":
        chamfer(args.scene, args.model, mesh, args.python)
    else:
        photo(args.scene, args.model, mesh, label=args.label)


if __name__ == "__main__":
    main()
