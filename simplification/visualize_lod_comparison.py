"""Reference | Wild | Sobel | PPQ renders (+ x4 error maps) from one CPU evaluation.

Views are not hand-picked: the script shows the view with the median and the
largest PSNR gain of --gain_label over --baseline (from per_view_metrics.csv).
  python visualize_lod_comparison.py --scene scan24 --budget 5000 --output fig.png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_mesh_preservation import load_mesh, render_cpu, texture_data  # noqa: E402
from run_enhanced_ablation import SCENES, mesh_path  # noqa: E402

LABELS = [("wild_ss1", "Wild"), ("sobel_ss1", "Sobel"), ("ppq_ss1", "PPQ")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--protocol", default="dtu")
    parser.add_argument("--baseline", default="wild_ss1")
    parser.add_argument("--gain_label", default="ppq_ss1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = HERE.parent / "outputs/DTU" / args.scene / "enhanced"
    evaluation = out / "evaluation" / f"main_{args.protocol}_{args.budget}"
    with (evaluation / "per_view_metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gains = np.array([float(r[f"{args.gain_label}_psnr"]) - float(r[f"{args.baseline}_psnr"]) for r in rows])
    order = np.argsort(gains)
    picks = [("median", order[len(order) // 2]), ("max", order[-1])]
    cameras = {c["img_name"]: c for c in json.loads((evaluation / "evaluation_cameras.json").read_text())}
    reference_path = SCENES[args.scene]["reference"]
    reference = load_mesh(reference_path)
    texture_data(reference, reference_path.with_suffix(".png"))
    meshes = []
    for label, name in LABELS:
        path = mesh_path(out, label, args.budget)
        data = load_mesh(path)
        texture_data(data, path.with_name(path.stem + "_texture.png"))
        meshes.append((label, name, data))
    blocks = []
    for kind, index in picks:
        camera = cameras[rows[index]["view"]]
        image, mask = render_cpu(reference, camera)
        ys, xs = np.where(mask)
        crop = (slice(max(0, ys.min() - 16), ys.max() + 17), slice(max(0, xs.min() - 16), xs.max() + 17))
        top, bottom, captions = [image[crop]], [np.zeros_like(image[crop])], [f"Reference ({kind} gain view)"]
        for label, name, data in meshes:
            rendered = render_cpu(data, camera)[0][crop]
            top.append(rendered)
            bottom.append(np.clip(np.abs(rendered - image[crop]) * 4, 0, 1))
            captions.append(f"{name} {float(rows[index][f'{label}_psnr']):.2f} dB")
        block = np.concatenate((np.concatenate(top, 1), np.concatenate(bottom, 1)), 0)
        blocks.append((block, captions, top[0].shape[1]))
    width = max(b.shape[1] for b, _, _ in blocks)
    canvas = np.concatenate([np.pad(b, ((0, 0), (0, width - b.shape[1]), (0, 0))) for b, _, _ in blocks], 0)
    picture = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    draw, y = ImageDraw.Draw(picture), 0
    for block, captions, tile in blocks:
        for i, caption in enumerate(captions):
            draw.text((i * tile + 6, y + 4), caption, fill=(255, 255, 0))
        y += block.shape[0]
    picture.save(args.output)
    print(f"[figure] {args.output}: views {[rows[i]['view'] for _, i in picks]} gains {[round(gains[i], 3) for _, i in picks]}")


if __name__ == "__main__":
    main()
