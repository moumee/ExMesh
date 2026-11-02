"""Side-by-side: original render | Sobel score | photometric quadric weight, per face.

  python visualize_face_weights.py --mesh orig.obj --texture orig.png --cameras cameras.json \
      --sobel sobel.npy --energy photometric_energy.npy --views 0010 0030 --output fig.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_mesh_preservation import load_mesh, render_cpu, texture_data  # noqa: E402
import wild_simplify_xatlas_verified as wild  # noqa: E402


def colormap(values):
    """Perceptually ordered dark-blue -> yellow ramp for values in [0, 1]."""
    stops = np.array([[0.05, 0.03, 0.25], [0.23, 0.32, 0.55], [0.13, 0.57, 0.55], [0.47, 0.82, 0.32], [0.99, 0.91, 0.14]])
    x = np.clip(values, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    t = (x - i)[:, None]
    return stops[i] * (1 - t) + stops[i + 1] * t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--sobel", required=True)
    parser.add_argument("--energy", required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()
    import open3d as o3d
    data = load_mesh(args.mesh)
    texture_data(data, args.texture)
    sobel = np.load(args.sobel)
    weight = wild.photometric_face_weights(data["v"], data["f"], np.load(args.energy), 1.0)
    # Same display range for both: Sobel weight 1 + 4 s spans [1, 5]; photometric shown on log scale.
    sobel_color = colormap(sobel)
    photo_color = colormap(np.log10(weight) / np.log10(np.percentile(weight, 99.5)))
    cameras = {c["img_name"]: c for c in json.loads(Path(args.cameras).read_text())}
    rows = []
    for name in args.views:
        c = dict(cameras[name])
        s = args.scale
        c.update(width=c["width"] // s, height=c["height"] // s, fx=c["fx"] / s, fy=c["fy"] / s)
        image, mask = render_cpu(data, c)
        h, w = c["height"], c["width"]
        yy, xx = np.mgrid[:h, :w]
        local = np.stack(((xx + .5 - w / 2) / c["fx"], (yy + .5 - h / 2) / c["fy"], np.ones((h, w))), -1)
        rays = np.concatenate((np.broadcast_to(c["position"], local.shape), local @ np.asarray(c["rotation"]).T), -1)
        ids = data["scene"].cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["primitive_ids"].numpy().astype(np.int64)
        tiles = [image]
        for colors in (sobel_color, photo_color):
            tile = np.zeros((h, w, 3))
            tile[mask] = colors[ids[mask]]
            tiles.append(tile)
        rows.append(np.concatenate(tiles, 1))
    Image.fromarray((np.clip(np.concatenate(rows, 0), 0, 1) * 255).astype(np.uint8)).save(args.output)
    print(f"[figure] {args.output}")


if __name__ == "__main__":
    main()
