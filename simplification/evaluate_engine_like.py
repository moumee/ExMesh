"""Engine-like render evaluation, secondary to evaluate_mesh_preservation.py.

evaluate_mesh_preservation.py ray-casts with bilinear texture lookups and no
mipmaps or antialiasing, so its reference images keep the aliasing of the
2560^2 original atlas. A game engine instead shows trilinear-mipmapped,
antialiased textures. This script re-renders the same meshes that way:
  - nvdiffrast, trilinear mipmaps, every atlas edge-padded as an import pipeline should
  - 2x2 supersampling with a box filter (stand-in for MSAA/TAA)
  - the exact camera poses an evaluate_mesh_preservation run saved, the same
    reference-bbox crop (16 px) and the same PSNR / SSIM / LPIPS code
so only the texture filtering differs between the two evaluations.

  python evaluate_engine_like.py --reference orig.obj --cameras_json <cpu_eval_dir>/evaluation_cameras.json \
      --candidate wild=wild.obj --candidate ppq=ppq.obj --output_dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from appearance_refine import TexturedMesh, edge_padded, max_mip_level  # noqa: E402
from evaluate_mesh_preservation import lpips_against_features  # noqa: E402
from photometric_quadric_weights import camera_mvp  # noqa: E402


def load(dr, glctx, path, texture=None):
    mesh = TexturedMesh(path, texture)
    mesh.texture = edge_padded(dr, glctx, mesh)
    return mesh


def render(dr, glctx, mesh, camera, near, far, supersample):
    """(3, H, W) color in [0, 1] and (H, W) coverage at the camera's own resolution."""
    mvp, _ = camera_mvp(camera, near, far)  # resolution-independent: fx / width is unchanged by supersampling
    clip = (torch.nn.functional.pad(mesh.pos, (0, 1), value=1.0)
            @ torch.tensor(mvp, dtype=torch.float32, device=mesh.pos.device).T)[None].contiguous()
    size = (camera["height"] * supersample, camera["width"] * supersample)
    rast, rast_db = dr.rasterize(glctx, clip, mesh.faces, size)
    texc, texd = dr.interpolate(mesh.uv[None], rast, mesh.uv_faces, rast_db=rast_db, diff_attrs="all")
    color = dr.texture(mesh.texture, texc, texd, filter_mode="linear-mipmap-linear",
                       max_mip_level=max_mip_level(mesh.texture.shape[1:3]))
    color = torch.where(rast[..., 3:] > 0, color, torch.zeros_like(color))
    color = dr.antialias(color.contiguous(), rast, clip, mesh.faces)[0].permute(2, 0, 1)
    color = torch.nn.functional.avg_pool2d(color[None], supersample)[0].clamp(0, 1)
    mask = torch.nn.functional.max_pool2d((rast[..., 3] > 0).float()[None], supersample)[0, 0] > 0
    return color, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference_texture")
    parser.add_argument("--candidate", action="append", required=True, help="LABEL=OBJ (texture from its MTL)")
    parser.add_argument("--cameras_json", required=True, help="evaluation_cameras.json of a CPU evaluation")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--crop_padding", type=int, default=16)
    args = parser.parse_args()

    import nvdiffrast.torch as dr
    from lpipsPyTorch import LPIPS
    from utils.loss_utils import ssim
    glctx = dr.RasterizeCudaContext()
    reference = load(dr, glctx, args.reference, args.reference_texture)
    candidates = {}
    for item in args.candidate:
        label, path = item.split("=", 1)
        candidates[label] = load(dr, glctx, path)
    lp = LPIPS("vgg", "0.1").cuda().eval()
    center = reference.pos.mean(0).cpu().numpy()
    diagonal = float((reference.pos.max(0).values - reference.pos.min(0).values).norm())
    rows = []
    with torch.no_grad():
        for camera in json.loads(Path(args.cameras_json).read_text()):
            depth = float(np.linalg.norm(np.asarray(camera["position"]) - center))
            near, far = max(1e-3, (depth - diagonal) * 0.5), depth + 2 * diagonal
            reference_image, mask = render(dr, glctx, reference, camera, near, far, args.supersample)
            ys, xs = torch.where(mask)
            if not len(xs):
                raise ValueError(f"Empty reference view: {camera['img_name']}")
            pad, (h, w) = args.crop_padding, mask.shape
            y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
            x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
            rt = reference_image[None, :, y0:y1, x0:x1]
            reference_features = lp.net(rt * 2 - 1)
            row = dict(view=camera["img_name"], source_view=camera["img_name"].split("_orbit_")[0])
            for label, mesh in candidates.items():
                ct = render(dr, glctx, mesh, camera, near, far, args.supersample)[0][None, :, y0:y1, x0:x1]
                row.update({f"{label}_psnr": float(-10 * torch.log10(((ct - rt) ** 2).mean().clamp(min=1e-15))),
                            f"{label}_ssim": float(ssim(ct, rt)),
                            f"{label}_lpips": float(lpips_against_features(lp, ct * 2 - 1, reference_features))})
            rows.append(row)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_view_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(render_definition=__doc__.split("\n\n")[1].strip(), supersample=args.supersample,
                   cameras_json=str(Path(args.cameras_json).resolve()), views=len(rows),
                   candidates={label: str(Path(item.split("=", 1)[1]).resolve())
                               for label, item in zip(candidates, args.candidate)},
                   render_mean={label: {m: float(np.mean([r[f"{label}_{m}"] for r in rows]))
                                        for m in ("psnr", "ssim", "lpips")} for label in candidates})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["render_mean"], indent=2))


if __name__ == "__main__":
    main()
