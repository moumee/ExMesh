"""Quick single-photo mesh generation with TripoSR (MIT), for the plugin's quick mode.

Writes the same textured-OBJ triple as the ExMesh LOD tools, so the result can go
straight into wild_simplify_xatlas_verified.py (--method wild) and appearance_refine.py:

  <stem>.obj, <stem>.mtl, <stem>_texture.png   mesh + material + edge-padded texture
  <stem>_input.png                              the preprocessed model input (preview)
  <stem>.complete.json                          written last; marks success

Run in the TripoSR environment (not the exmesh one):
  python triposr_generate.py --repo <TripoSR clone> --image photo.jpg --output out/mesh.obj
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def edge_pad(rgba):
    """Copy the nearest covered texel into every texel outside the UV charts (alpha == 0)."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    covered = rgba[..., 3] > 0
    if not covered.any():
        return rgba[..., :3]
    _, (rows, cols) = distance_transform_edt(~covered, return_indices=True)
    return rgba[rows, cols, :3]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="TripoSR clone with quick_generation/triposr.diff applied")
    parser.add_argument("--image", required=True, help="Input photo. RGBA with a real alpha skips background removal")
    parser.add_argument("--output", required=True, help="Output OBJ path; .mtl and _texture.png are written beside it")
    parser.add_argument("--weights", default="stabilityai/TripoSR",
                        help="Hugging Face repo id or a local folder with config.yaml and model.ckpt")
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--mc_resolution", type=int, default=256, help="Marching cubes grid; higher = more faces")
    parser.add_argument("--foreground_ratio", type=float, default=0.85)
    parser.add_argument("--up", choices=("z", "y"), default="z",
                        help="Up axis of the written mesh. z: TripoSR native (photo side faces +X). "
                             "y: rotated for Y-up tools (photo side faces +Z)")
    parser.add_argument("--height", type=float,
                        help="Scale uniformly so the mesh height along the up axis equals this value "
                             "(e.g. 100 for 1 m in UE units). Default: TripoSR native units (about 1)")
    args = parser.parse_args()
    if args.texture_size < 64 or args.mc_resolution < 32 or not 0 < args.foreground_ratio <= 1:
        parser.error("--texture_size >= 64, --mc_resolution >= 32 and 0 < --foreground_ratio <= 1 are required")
    if args.height is not None and not args.height > 0:
        parser.error("--height must be positive")

    started = time.perf_counter()
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import numpy as np
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("TripoSR quick mode needs a CUDA GPU")
    from tsr.bake_texture import bake_texture
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    output = Path(args.output).with_suffix(".obj")
    output.parent.mkdir(parents=True, exist_ok=True)
    done = output.with_suffix(".complete.json")
    done.unlink(missing_ok=True)
    timing = {}

    t = time.perf_counter()
    model = TSR.from_pretrained(args.weights, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to("cuda:0")
    timing["load_s"] = time.perf_counter() - t

    t = time.perf_counter()
    photo = Image.open(args.image)
    alpha = np.asarray(photo.getchannel("A")) if photo.mode in ("RGBA", "LA") else None
    if alpha is not None and (alpha < 255).any():
        background = "input alpha"
        image = photo.convert("RGBA")
    else:
        import rembg
        background = "rembg u2net"
        image = remove_background(photo.convert("RGB"), rembg.new_session())
    image = resize_foreground(image, args.foreground_ratio)
    rgba = np.asarray(image, np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:4] + (1 - rgba[..., 3:4]) * 0.5
    image = Image.fromarray((rgb * 255.0).astype(np.uint8))
    image.save(output.with_name(output.stem + "_input.png"))
    timing["preprocess_s"] = time.perf_counter() - t

    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    with torch.no_grad():
        scene_codes = model([image], device="cuda:0")
    mesh = model.extract_mesh(scene_codes, False, resolution=args.mc_resolution)[0]
    if len(mesh.faces) == 0:
        raise RuntimeError("TripoSR produced an empty mesh; check the background removal of the input")
    timing["inference_s"] = time.perf_counter() - t

    t = time.perf_counter()
    bake = bake_texture(mesh, model, scene_codes[0], args.texture_size)
    texture = edge_pad(bake["colors"])
    timing["texture_s"] = time.perf_counter() - t
    peak_vram = torch.cuda.max_memory_allocated() / 2**30

    # Keep positions welded: xatlas splits vertices along UV seams (vmapping), which
    # would cut the surface open for the simplifier. Positions use the original
    # vertex ids; only UVs use the split ids.
    positions = np.asarray(mesh.vertices, np.float64)
    normals = np.asarray(mesh.vertex_normals, np.float64)
    if args.up == "y":
        to_y_up = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], np.float64)  # (x, y, z) -> (y, z, x)
        positions, normals = positions @ to_y_up.T, normals @ to_y_up.T
    up = 2 if args.up == "z" else 1
    scale = 1.0
    if args.height is not None:
        scale = args.height / float(np.ptp(positions[:, up]))
        positions = positions * scale
    # Open3D (inside the simplifier) reads OBJ positions as float32 and checks them
    # against the text to 1e-8; write float32 values with 9 significant digits.
    positions = positions.astype(np.float32)
    corner_uv = bake["indices"].astype(np.int64)
    corner_pos = bake["vmapping"].astype(np.int64)[corner_uv]

    texture_path = output.with_name(output.stem + "_texture.png")
    mtl_path = output.with_suffix(".mtl")
    Image.fromarray((np.clip(texture, 0, 1) * 255.0 + 0.5).astype(np.uint8)).transpose(
        Image.FLIP_TOP_BOTTOM).save(texture_path)
    mtl_path.write_text("newmtl material0\nKa 1.000000 1.000000 1.000000\nKd 1.000000 1.000000 1.000000\n"
                        f"Ks 0.000000 0.000000 0.000000\nmap_Kd {texture_path.name}\n", encoding="utf-8")
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"mtllib {mtl_path.name}\nusemtl material0\n")
        f.writelines(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in positions)
        f.writelines(f"vt {u:.9g} {v:.9g}\n" for u, v in bake["uvs"])
        f.writelines(f"vn {x:.5g} {y:.5g} {z:.5g}\n" for x, y, z in normals)
        for p, q in zip(corner_pos + 1, corner_uv + 1):
            f.write(f"f {p[0]}/{q[0]}/{p[0]} {p[1]}/{q[1]}/{p[1]} {p[2]}/{q[2]}/{p[2]}\n")

    report = dict(model="TripoSR", weights=args.weights, image=str(Path(args.image).resolve()),
                  output=str(output.resolve()), background=background,
                  settings=dict(texture_size=args.texture_size, mc_resolution=args.mc_resolution,
                                foreground_ratio=args.foreground_ratio, up=args.up, height=args.height),
                  scale=scale, vertices=int(len(positions)), faces=int(len(corner_pos)),
                  bbox_min=positions.min(0).astype(float).tolist(), bbox_max=positions.max(0).astype(float).tolist(),
                  timing=timing, seconds=time.perf_counter() - started, peak_vram_gib=peak_vram,
                  gpu=torch.cuda.get_device_name(0), u2net_home=os.environ.get("U2NET_HOME"))
    done.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
