"""Compare appearance preservation of two simplified textured OBJ meshes.

The script renders a reference OBJ, a baseline simplified OBJ, and a modified
simplified OBJ from the SAME ExMesh/DTU camera views, then reports PSNR, SSIM,
and LPIPS relative to the reference render.

Place this file under the ExMesh repository root or ExMesh/simplification/.
It uses ExMesh's camera loading conventions and nvdiffrast.

Example (Windows CMD):

python simplification\\evaluate_render_quality.py ^
  --source_path workdir\\DTU\\scan24 ^
  --reference outputs\\DTU\\scan24\\export\\mesh_iter_10000.obj ^
  --baseline outputs\\DTU\\scan24\\simplification\\mesh_iter_10000_simplified_0_1.obj ^
  --modified outputs\\DTU\\scan24\\simplification\\mesh_iter_10000_mod_simplified_0_1.obj ^
  --output_dir outputs\\DTU\\scan24\\render_eval ^
  --resolution 1 ^
  --save_renders

Higher PSNR/SSIM is better. Lower LPIPS is better.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# The fair protocol is independent of ExMesh's CUDA-only training imports.
# Dispatch before importing nvdiffrast so geometry and CPU renders also work on
# evaluation machines without the reconstruction environment.
if __name__ == "__main__" and any(arg == "--protocol" or arg.startswith("--protocol=") for arg in sys.argv):
    from evaluate_mesh_preservation import main as preservation_main
    preservation_main()
    raise SystemExit(0)

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import nvdiffrast.torch as dr


# -----------------------------------------------------------------------------
# Make ExMesh imports work whether this file lives in ExMesh/ or
# ExMesh/simplification/.
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "scene").is_dir():
    ROOT_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "scene").is_dir():
    ROOT_DIR = SCRIPT_DIR.parent
else:
    raise RuntimeError(
        "Could not find the ExMesh root. Put this script in ExMesh/ or "
        "ExMesh/simplification/."
    )

sys.path.insert(0, str(ROOT_DIR))

from scene.dataset_readers import sceneLoadTypeCallbacks  # noqa: E402
from utils.camera_utils import cameraList_from_camInfos  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402
from lpipsPyTorch import LPIPS  # noqa: E402


# -----------------------------------------------------------------------------
# OBJ + texture loading
# -----------------------------------------------------------------------------
def _obj_index(raw: str, count: int) -> int:
    """Convert OBJ 1-based / negative index to Python 0-based index."""
    idx = int(raw)
    if idx > 0:
        return idx - 1
    if idx < 0:
        return count + idx
    raise ValueError("OBJ index 0 is invalid")


def _read_map_kd(obj_path: Path, mtllib_names: list[str]) -> Path | None:
    """Return the first existing map_Kd path referenced by the OBJ's MTL."""
    for mtllib in mtllib_names:
        mtl_path = obj_path.parent / mtllib
        if not mtl_path.exists():
            continue
        with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split(maxsplit=1)
                if len(parts) == 2 and parts[0].lower() == "map_kd":
                    tex_path = mtl_path.parent / parts[1].strip()
                    if tex_path.exists():
                        return tex_path
    return None


def _find_texture(obj_path: Path, explicit: str | None, mtllib_names: list[str]) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Texture not found: {path}")
        return path

    # Prefer the texture actually referenced by the MTL.
    mtl_texture = _read_map_kd(obj_path, mtllib_names)
    if mtl_texture is not None:
        return mtl_texture

    # Useful fallbacks for the user's ExMesh exporter and Wild simplifier.
    candidates = [
        obj_path.with_suffix(".png"),
        obj_path.with_name(f"{obj_path.stem}_texture.png"),
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find texture for {obj_path}. Checked the MTL, "
        f"{candidates[0].name}, and {candidates[1].name}. "
        "Use the corresponding --*_texture option if needed."
    )


class TexturedOBJ:
    """Minimal textured triangle mesh adapter for nvdiffrast rendering."""

    def __init__(self, obj_path: str, texture_path: str | None, device: str = "cuda"):
        path = Path(obj_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"OBJ not found: {path}")

        vertices: list[list[float]] = []
        uvs: list[list[float]] = []
        faces: list[list[int]] = []
        uv_faces: list[list[int]] = []
        mtllib_names: list[str] = []

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                tag = parts[0]

                if tag == "v" and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

                elif tag == "vt" and len(parts) >= 3:
                    uvs.append([float(parts[1]), float(parts[2])])

                elif tag == "mtllib" and len(parts) >= 2:
                    mtllib_names.extend(parts[1:])

                elif tag == "f" and len(parts) >= 4:
                    # Fan triangulation also handles an accidental polygon face.
                    corners: list[tuple[int, int]] = []
                    for token in parts[1:]:
                        fields = token.split("/")
                        vi = _obj_index(fields[0], len(vertices))
                        if len(fields) < 2 or fields[1] == "":
                            raise ValueError(
                                f"OBJ has a face without UVs: {path}\nLine: {stripped}"
                            )
                        ti = _obj_index(fields[1], len(uvs))
                        corners.append((vi, ti))

                    for k in range(1, len(corners) - 1):
                        tri = [corners[0], corners[k], corners[k + 1]]
                        faces.append([tri[0][0], tri[1][0], tri[2][0]])
                        uv_faces.append([tri[0][1], tri[1][1], tri[2][1]])

        if not vertices or not faces or not uvs:
            raise ValueError(
                f"Failed to load a textured triangle mesh from {path}: "
                f"vertices={len(vertices)}, faces={len(faces)}, uvs={len(uvs)}"
            )

        texture_file = _find_texture(path, texture_path, mtllib_names)
        texture_np = np.asarray(Image.open(texture_file).convert("RGB"), dtype=np.float32) / 255.0

        vertices_np = np.asarray(vertices, dtype=np.float32)
        faces_np = np.asarray(faces, dtype=np.int32)
        uvs_np = np.asarray(uvs, dtype=np.float32)
        uv_faces_np = np.asarray(uv_faces, dtype=np.int32)

        # OBJ uses V=0 at the bottom in the convention used by the user's exporter
        # and Wild texture baker. ExMesh's renderer feeds UVs directly to
        # torch.grid_sample, whose V=-1/0 side is the image top, so convert back to
        # ExMesh/image-space V before rendering.
        uvs_np[:, 1] = 1.0 - uvs_np[:, 1]

        self.vertices = torch.from_numpy(vertices_np).to(device)
        self.faces = torch.from_numpy(faces_np).to(device)
        self.uvs = torch.from_numpy(uvs_np).to(device)
        self.uv_indices = torch.from_numpy(uv_faces_np).to(device)
        self.texture = torch.from_numpy(texture_np).permute(2, 0, 1).contiguous().to(device)

        self.path = path
        self.texture_path = texture_file

        print(
            f"[mesh] {path.name}: vertices={len(vertices_np):,}, "
            f"faces={len(faces_np):,}, uvs={len(uvs_np):,}, "
            f"texture={texture_file.name}",
            flush=True,
        )


# -----------------------------------------------------------------------------
# Rendering: same camera/projection convention as ExMesh renderer, but only RGB
# and alpha are needed here. One RasterizeCudaContext is reused for all views.
# -----------------------------------------------------------------------------
@torch.no_grad()
def render_textured(camera, mesh: TexturedOBJ, bg_color: torch.Tensor, glctx):
    device = mesh.vertices.device
    H = int(camera.image_height)
    W = int(camera.image_width)

    view = camera.world_view_transform.to(device)
    proj = camera.projection_matrix.to(device)

    # ExMesh camera projection is D3D/GS NDC z in [0,1]. nvdiffrast expects
    # OpenGL NDC z in [-1,1]. This is the same conversion used by ExMesh.
    proj_glndc = proj.clone()
    proj_glndc[2, :] = proj_glndc[2, :] * 2.0 - proj_glndc[3, :]

    ones = torch.ones(
        (mesh.vertices.shape[0], 1),
        device=device,
        dtype=mesh.vertices.dtype,
    )
    v4 = torch.cat([mesh.vertices, ones], dim=1)
    mvp = proj_glndc @ view
    pos_clip = (v4 @ mvp.t()).unsqueeze(0).contiguous()

    faces = mesh.faces.to(torch.int32).contiguous()
    uv_indices = mesh.uv_indices.to(torch.int32).contiguous()

    rast, _ = dr.rasterize(glctx, pos_clip, faces, resolution=(H, W))

    uv_attr = mesh.uvs.unsqueeze(0).contiguous()
    uv_map, _ = dr.interpolate(uv_attr, rast, uv_indices)

    grid = uv_map.clamp(0.0, 1.0) * 2.0 - 1.0
    tex = mesh.texture.unsqueeze(0).clamp(0.0, 1.0)
    rgb = F.grid_sample(
        tex,
        grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    )[0]

    alpha = rast[..., 3:].clamp(0.0, 1.0)
    alpha = dr.antialias(alpha, rast, pos_clip, faces)
    alpha = alpha.squeeze(0).squeeze(-1).unsqueeze(0)

    image = rgb * alpha + bg_color.view(3, 1, 1) * (1.0 - alpha)
    return image.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# Camera loading
# -----------------------------------------------------------------------------
def load_exmesh_cameras(source_path: str, images: str, resolution: int, data_device: str):
    source = Path(source_path).resolve()
    if not (source / "sparse").exists():
        raise RuntimeError(
            f"This script currently expects an ExMesh/Colmap scene with a sparse/ folder: {source}"
        )

    # eval=False matches the default ExMesh DTU training setup: all available
    # cameras are placed in the training camera set.
    scene_info = sceneLoadTypeCallbacks["Colmap"](str(source), images, False)

    cam_args = SimpleNamespace(
        resolution=resolution,
        data_device=data_device,
    )
    cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, cam_args)
    cameras.sort(key=lambda c: c.image_name)
    return cameras


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def bbox_from_alpha(alpha: torch.Tensor, padding: int = 16):
    """Bounding box of reference foreground. Returns y0,y1,x0,x1 (exclusive end)."""
    mask = alpha[0] > 1e-3
    ys, xs = torch.where(mask)
    H, W = alpha.shape[-2:]
    if len(xs) == 0:
        return 0, H, 0, W

    y0 = max(0, int(ys.min().item()) - padding)
    y1 = min(H, int(ys.max().item()) + 1 + padding)
    x0 = max(0, int(xs.min().item()) - padding)
    x1 = min(W, int(xs.max().item()) + 1 + padding)
    return y0, y1, x0, x1


def crop_chw(image: torch.Tensor, bbox):
    y0, y1, x0, x1 = bbox
    return image[:, y0:y1, x0:x1]


@torch.no_grad()
def metrics_for_pair(candidate: torch.Tensor, reference: torch.Tensor, lpips_model):
    # ExMesh metric utilities expect BCHW.
    cand = candidate.unsqueeze(0)
    ref = reference.unsqueeze(0)

    p = float(psnr(cand, ref).mean().item())
    s = float(ssim(cand, ref).mean().item())
    # This implementation expects RGB in [-1, 1], not [0, 1].
    l = float(lpips_model(cand * 2.0 - 1.0, ref * 2.0 - 1.0).mean().item())
    return {"psnr": p, "ssim": s, "lpips": l}


def mean_metrics(rows, prefix: str):
    return {
        "psnr": float(np.mean([r[f"{prefix}_psnr"] for r in rows])),
        "ssim": float(np.mean([r[f"{prefix}_ssim"] for r in rows])),
        "lpips": float(np.mean([r[f"{prefix}_lpips"] for r in rows])),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render reference/baseline/modified textured OBJ meshes from the same "
            "ExMesh cameras and compare PSNR, SSIM, and LPIPS."
        )
    )
    parser.add_argument("--source_path", "-s", required=True,
                        help="ExMesh scene path, e.g. workdir/DTU/scan24")
    parser.add_argument("--reference", required=True,
                        help="Original/reference textured OBJ")
    parser.add_argument("--baseline", required=True,
                        help="Baseline simplified textured OBJ (e.g. lambda=0)")
    parser.add_argument("--modified", required=True,
                        help="Modified simplified textured OBJ (e.g. lambda=1)")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--reference_texture")
    parser.add_argument("--baseline_texture")
    parser.add_argument("--modified_texture")

    parser.add_argument("--images", default="images",
                        help="Image directory name inside the Colmap scene. Default: images")
    parser.add_argument("--resolution", type=int, default=1,
                        help="ExMesh camera resolution factor. Use 1 for original resolution.")
    parser.add_argument("--max_views", type=int, default=0,
                        help="0 = all cameras; positive value = first N cameras (quick test)")
    parser.add_argument("--crop_padding", type=int, default=16,
                        help="Pixels around reference silhouette for metric crop. Default: 16")
    parser.add_argument("--white_background", action="store_true",
                        help="Use white instead of black render background")
    parser.add_argument("--save_renders", action="store_true",
                        help="Save reference/baseline/modified rendered images")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this ExMesh/nvdiffrast evaluation script.")

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[load] cameras", flush=True)
    cameras = load_exmesh_cameras(
        args.source_path,
        args.images,
        args.resolution,
        device,
    )
    if args.max_views > 0:
        cameras = cameras[:args.max_views]
    if len(cameras) == 0:
        raise RuntimeError("No cameras were loaded.")
    print(f"[load] cameras={len(cameras)}", flush=True)

    print("[load] meshes", flush=True)
    reference = TexturedOBJ(args.reference, args.reference_texture, device)
    baseline = TexturedOBJ(args.baseline, args.baseline_texture, device)
    modified = TexturedOBJ(args.modified, args.modified_texture, device)

    bg_value = 1.0 if args.white_background else 0.0
    background = torch.full((3,), bg_value, dtype=torch.float32, device=device)

    # Match the ExMesh repository's LPIPS metric implementation (VGG).
    print("[metric] loading LPIPS VGG", flush=True)
    lpips_model = LPIPS("vgg", "0.1").to(device).eval()

    glctx = dr.RasterizeCudaContext()

    render_root = output_dir / "renders"
    if args.save_renders:
        for name in ("reference", "baseline", "modified"):
            (render_root / name).mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, camera in enumerate(cameras):
        name = str(camera.image_name)
        print(f"[render] {idx + 1}/{len(cameras)} {name}", flush=True)

        ref_img, ref_alpha = render_textured(camera, reference, background, glctx)
        base_img, _ = render_textured(camera, baseline, background, glctx)
        mod_img, _ = render_textured(camera, modified, background, glctx)

        # Evaluate on a crop around the reference silhouette so empty background
        # does not dominate the metric. Silhouette errors still count because the
        # same crop is used for all three images.
        bbox = bbox_from_alpha(ref_alpha, args.crop_padding)
        ref_crop = crop_chw(ref_img, bbox)
        base_crop = crop_chw(base_img, bbox)
        mod_crop = crop_chw(mod_img, bbox)

        base_m = metrics_for_pair(base_crop, ref_crop, lpips_model)
        mod_m = metrics_for_pair(mod_crop, ref_crop, lpips_model)

        rows.append({
            "view": name,
            "baseline_psnr": base_m["psnr"],
            "baseline_ssim": base_m["ssim"],
            "baseline_lpips": base_m["lpips"],
            "modified_psnr": mod_m["psnr"],
            "modified_ssim": mod_m["ssim"],
            "modified_lpips": mod_m["lpips"],
        })

        if args.save_renders:
            safe_name = name.replace("/", "_").replace("\\", "_")
            if not Path(safe_name).suffix:
                safe_name += ".png"
            torchvision.utils.save_image(ref_img.cpu(), render_root / "reference" / safe_name)
            torchvision.utils.save_image(base_img.cpu(), render_root / "baseline" / safe_name)
            torchvision.utils.save_image(mod_img.cpu(), render_root / "modified" / safe_name)

    baseline_mean = mean_metrics(rows, "baseline")
    modified_mean = mean_metrics(rows, "modified")

    summary = {
        "reference": str(Path(args.reference).resolve()),
        "baseline_path": str(Path(args.baseline).resolve()),
        "modified_path": str(Path(args.modified).resolve()),
        "source_path": str(Path(args.source_path).resolve()),
        "num_views": len(rows),
        "protocol": "legacy_source_views",
        "purpose": "appearance preservation relative to original mesh render; not held-out photo quality",
        "lpips_input_range": "[-1, 1]; historical runs using [0, 1] are not directly comparable",
        "metric_region": f"reference silhouette bounding box + {args.crop_padding}px padding",
        "baseline": baseline_mean,
        "modified": modified_mean,
        "delta_modified_minus_baseline": {
            "psnr": modified_mean["psnr"] - baseline_mean["psnr"],
            "ssim": modified_mean["ssim"] - baseline_mean["ssim"],
            # Negative delta is an improvement for LPIPS.
            "lpips": modified_mean["lpips"] - baseline_mean["lpips"],
        },
    }

    csv_path = output_dir / "per_view_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ Render Quality ================")
    print(f"Views: {len(rows)}")
    print("                    PSNR ↑        SSIM ↑        LPIPS ↓")
    print(
        "Baseline        "
        f"{baseline_mean['psnr']:10.4f}   "
        f"{baseline_mean['ssim']:10.6f}   "
        f"{baseline_mean['lpips']:10.6f}"
    )
    print(
        "Modified        "
        f"{modified_mean['psnr']:10.4f}   "
        f"{modified_mean['ssim']:10.6f}   "
        f"{modified_mean['lpips']:10.6f}"
    )
    print(
        "Delta (M-B)     "
        f"{summary['delta_modified_minus_baseline']['psnr']:10.4f}   "
        f"{summary['delta_modified_minus_baseline']['ssim']:10.6f}   "
        f"{summary['delta_modified_minus_baseline']['lpips']:10.6f}"
    )
    print("================================================")
    print(f"Per-view CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
