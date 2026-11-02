"""Build single-image RGBA inputs for image-to-3D models from DTU views.

For each scene/view: alpha = DTU object mask, crop to mask bbox + margin,
pad to square, resize. Writes <out>/<scene>_<view>.png and a JSON sidecar with
the crop and the COLMAP camera of that view (used later for alignment).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", required=True,
                        help="scene:view, e.g. scan83:51 (repeatable)")
    parser.add_argument("--out", default=str(ROOT / "outputs/alt_models/inputs"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=0.08)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for item in args.item:
        scene, view = item.split(":")
        view = int(view)
        image_path = ROOT / f"workdir/DTU/{scene}/images/{view:04d}.png"
        mask_path = ROOT / f"workdir/DTU/{scene}/mask/{view:03d}.png"
        image = Image.open(image_path)
        rgb = np.asarray(image.convert("RGB"))
        # DTU images are centered crops of the 1600x1200 masks' frame; use the aligned alpha.
        if image.mode == "RGBA":
            mask = np.asarray(image.getchannel("A")) > 127
        else:
            mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        ys, xs = np.where(mask)
        h, w = mask.shape
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        side = int(max(y1 - y0, x1 - x0) * (1 + 2 * args.margin))
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        top, left = cy - side // 2, cx - side // 2
        canvas = np.zeros((side, side, 4), np.uint8)
        sy0, sx0 = max(top, 0), max(left, 0)
        sy1, sx1 = min(top + side, h), min(left + side, w)
        canvas[sy0 - top:sy1 - top, sx0 - left:sx1 - left, :3] = rgb[sy0:sy1, sx0:sx1]
        canvas[sy0 - top:sy1 - top, sx0 - left:sx1 - left, 3] = mask[sy0:sy1, sx0:sx1] * 255
        name = f"{scene}_view{view:03d}"
        # Unmasked crop: for models whose official preprocessing removes the background.
        Image.fromarray(canvas[..., :3]).resize((args.size, args.size), Image.LANCZOS).save(out / f"{name}_rgb.png")
        canvas[canvas[..., 3] == 0, :3] = 0
        Image.fromarray(canvas).resize((args.size, args.size), Image.LANCZOS).save(out / f"{name}.png")
        touches = bool(y0 == 0 or x0 == 0 or y1 == h or x1 == w)
        meta = dict(scene=scene, view=view, image=str(image_path), mask=str(mask_path),
                    crop_top=int(top), crop_left=int(left), crop_side=side, size=args.size,
                    object_touches_image_border=touches)
        (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        print(f"[input] {name}: crop side {side}px, touches border={touches}")


if __name__ == "__main__":
    main()
