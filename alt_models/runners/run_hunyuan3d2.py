"""Hunyuan3D-2 (v2.0) image-to-textured-mesh, following the official gradio flow.

rembg (hy3dgen.rembg) -> DiT shape (seed 1234, 50 steps, cfg 7.5, octree 256)
-> FaceReducer (default 40k) -> Hunyuan3D-Paint turbo with CPU offload
(the repo's --low_vram_mode). Writes GLB + OBJ/MTL/PNG and run_info.json.
Run in the hunyuan3d2 env:  python run_hunyuan3d2.py --repo <clone> --image X --out DIR
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
    parser.add_argument("--model_path", default="tencent/Hunyuan3D-2")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num_chunks", type=int, default=200000, help="gradio default; lower = less VRAM")
    parser.add_argument("--flashvdm", action="store_true", help="FlashVDM decoding (repo option)")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)

    import torch
    from PIL import Image
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timing = {}

    t = time.perf_counter()
    image = Image.open(args.image)
    if image.mode == "RGB":
        image = BackgroundRemover()(image.convert("RGB"))
    image.save(out / "input_processed.png")
    timing["preprocess_s"] = time.perf_counter() - t

    t = time.perf_counter()
    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path, subfolder=args.subfolder)
    if args.flashvdm:
        shape.enable_flashvdm()
    timing["shape_load_s"] = time.perf_counter() - t
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    mesh = shape(image=image, num_inference_steps=args.steps, guidance_scale=7.5,
                 generator=torch.Generator().manual_seed(args.seed), octree_resolution=256,
                 num_chunks=args.num_chunks, output_type="trimesh")[0]
    torch.cuda.synchronize()
    timing["inference_s"] = time.perf_counter() - t
    peak_shape = torch.cuda.max_memory_allocated() / 2**30
    raw_faces = int(len(mesh.faces))
    mesh.export(out / "shape_raw.glb")
    del shape
    torch.cuda.empty_cache()

    t = time.perf_counter()
    mesh = FaceReducer()(mesh)
    timing["face_reduce_s"] = time.perf_counter() - t

    t = time.perf_counter()
    paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    paint.enable_model_cpu_offload()
    timing["texture_load_s"] = time.perf_counter() - t
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    textured = paint(mesh, image=image)
    torch.cuda.synchronize()
    timing["texture_s"] = time.perf_counter() - t
    peak_texture = torch.cuda.max_memory_allocated() / 2**30

    textured.export(out / "mesh.glb")
    textured.export(out / "mesh.obj")

    commit = subprocess.run(["git", "-C", args.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    info = dict(model="Hunyuan3D-2", repo="https://github.com/Tencent-Hunyuan/Hunyuan3D-2", commit=commit,
                hf_repo=[args.model_path, "tencent/Hunyuan3D-2"], hf_subfolders=[args.subfolder, "hunyuan3d-paint-v2-0-turbo",
                                                             "hunyuan3d-delight-v2-0"],
                python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), command=sys.argv, input=str(Path(args.image).resolve()),
                seed=args.seed, settings=dict(steps=args.steps, guidance_scale=7.5, octree_resolution=256,
                                              num_chunks=args.num_chunks, flashvdm=args.flashvdm,
                                              face_reducer="default (40000)", paint_cpu_offload=True,
                                              background_removal="hy3dgen.rembg (rembg)"),
                timing=timing, peak_vram_gib=dict(shape=peak_shape, texture=peak_texture),
                raw_shape_faces=raw_faces, vertices=int(len(textured.vertices)), faces=int(len(textured.faces)),
                textured=True,
                deviations=[f"shape model {args.model_path}/{args.subfolder} (gradio default: 2mini-turbo)",
                            "repo imported from sys.path (pip install -e . not run)",
                            "diffusers 0.32.2 / transformers 4.48.3 pinned for torch 2.5.1"])
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
