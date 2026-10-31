"""TripoSR image-to-3D with the official run.py defaults + textured OBJ export.

Same steps as TripoSR/run.py (rembg u2net, foreground ratio 0.85, marching
cubes 256, bake 2048 texture) with stage timing / peak VRAM in run_info.json.
Run in the TripoSR env:  python run_triposr.py --repo <TripoSR clone> --image X --out DIR
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mc_resolution", type=int, default=256)
    parser.add_argument("--texture_resolution", type=int, default=2048)
    parser.add_argument("--foreground_ratio", type=float, default=0.85)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)

    import numpy as np
    import rembg
    import torch
    import xatlas
    from PIL import Image
    from tsr.bake_texture import bake_texture
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timing = {}
    t = time.perf_counter()
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to("cuda:0")
    timing["load_s"] = time.perf_counter() - t

    t = time.perf_counter()
    image = remove_background(Image.open(args.image), rembg.new_session())
    image = resize_foreground(image, args.foreground_ratio)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    image.save(out / "input_processed.png")
    timing["preprocess_s"] = time.perf_counter() - t

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    with torch.no_grad():
        scene_codes = model([image], device="cuda:0")
    meshes = model.extract_mesh(scene_codes, False, resolution=args.mc_resolution)
    torch.cuda.synchronize()
    timing["inference_s"] = time.perf_counter() - t

    t = time.perf_counter()
    bake = bake_texture(meshes[0], model, scene_codes[0], args.texture_resolution)
    torch.cuda.synchronize()
    timing["texture_s"] = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / 2**30

    mesh_path = out / "mesh.obj"
    xatlas.export(str(mesh_path), meshes[0].vertices[bake["vmapping"]], bake["indices"], bake["uvs"],
                  meshes[0].vertex_normals[bake["vmapping"]])
    Image.fromarray((bake["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(out / "texture.png")
    # xatlas.export writes geometry + UVs only; add the material referencing the baked texture.
    (out / "mesh.mtl").write_text("newmtl material0\nKd 1 1 1\nmap_Kd texture.png\n")
    lines = mesh_path.read_text().splitlines()
    mesh_path.write_text("\n".join(["mtllib mesh.mtl", "usemtl material0"] + lines) + "\n")

    commit = subprocess.run(["git", "-C", args.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    info = dict(model="TripoSR", repo="https://github.com/VAST-AI-Research/TripoSR", commit=commit,
                hf_repo="stabilityai/TripoSR", python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), command=sys.argv,
                input=str(Path(args.image).resolve()), seed=None,
                settings=dict(mc_resolution=args.mc_resolution, texture_resolution=args.texture_resolution,
                              foreground_ratio=args.foreground_ratio, chunk_size=8192, background_removal="rembg u2net"),
                timing=timing, peak_vram_gib=peak, vertices=int(len(meshes[0].vertices)),
                faces=int(len(meshes[0].faces)), textured=True,
                deviations=["torchmcubes replaced by PyMCubes CPU marching cubes", "bake_texture: query positions moved to the scene-code device (upstream device-mismatch bug)", "see patches/triposr.diff"])
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
