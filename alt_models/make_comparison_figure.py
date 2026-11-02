"""Photo | ExMesh | image-to-3D models, rendered at the same capture cameras.

Untextured meshes are shown with a simple normal shading. Usage:
  python make_comparison_figure.py --scene scan83 --views 0010 0030 --output fig.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "simplification"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_mesh_preservation import load_mesh, render_cpu, texture_data  # noqa: E402
from evaluate_alt_models import scene_paths, texture_for  # noqa: E402

COLUMNS = [("exmesh", "ExMesh"), ("triposr", "TripoSR"), ("hunyuan3d2", "Hunyuan3D-2")]


def shaded(data, camera):
    import open3d as o3d
    h, w = camera["height"], camera["width"]
    yy, xx = np.mgrid[:h, :w]
    local = np.stack(((xx + .5 - w / 2) / camera["fx"], (yy + .5 - h / 2) / camera["fy"], np.ones((h, w))), -1)
    direction = local @ np.asarray(camera["rotation"]).T
    rays = np.concatenate((np.broadcast_to(camera["position"], direction.shape), direction), -1).astype(np.float32)
    hits = data["scene"].cast_rays(o3d.core.Tensor(rays))
    mask = np.isfinite(hits["t_hit"].numpy())
    normals = hits["primitive_normals"].numpy()
    view = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    lambert = np.abs((normals * view).sum(-1))
    image = np.zeros((h, w, 3), np.float32)
    image[mask] = (0.15 + 0.75 * lambert[mask])[:, None] * np.array([0.85, 0.85, 0.9])
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()
    meshes = {}
    for key, _ in COLUMNS:
        path = scene_paths(args.scene)["exmesh"] if key == "exmesh" else ROOT / "outputs/alt_models" / args.scene / key / "aligned.obj"
        if path.exists():
            data = load_mesh(path)
            texture = texture_for(path)
            data["textured"] = texture is not None
            if texture is not None:
                texture_data(data, texture)
            meshes[key] = data
    cameras = {c["img_name"]: c for c in json.loads(scene_paths(args.scene)["cameras_json"].read_text())}
    rows = []
    for name in args.views:
        c = dict(cameras[name])
        c.update(width=c["width"] // args.scale, height=c["height"] // args.scale,
                 fx=c["fx"] / args.scale, fy=c["fy"] / args.scale)
        rgba = Image.open(scene_paths(args.scene)["data"] / "images" / f"{name}.png").convert("RGBA")
        photo = np.asarray(rgba.convert("RGB").resize((c["width"], c["height"]), Image.BOX), np.float32) / 255
        tiles = [photo]
        for key, _ in COLUMNS:
            data = meshes.get(key)
            if data is None:
                tiles.append(np.zeros_like(photo))
            elif data["textured"]:
                tiles.append(render_cpu(data, c)[0])
            else:
                tiles.append(shaded(data, c))
        rows.append(np.concatenate(tiles, 1))
    image = Image.fromarray((np.clip(np.concatenate(rows, 0), 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    width = rows[0].shape[1] // (len(COLUMNS) + 1)
    for i, label in enumerate(["Photo"] + [label for _, label in COLUMNS]):
        draw.text((i * width + 6, 4), label, fill=(255, 255, 0))
    image.save(args.output)
    print(f"[figure] {args.output}")


if __name__ == "__main__":
    main()
