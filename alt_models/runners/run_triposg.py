"""TripoSG image-to-mesh (geometry only) with scripts/inference_triposg.py defaults.

BriaRMBG-1.4 background removal, seed 42, 50 steps, guidance 7.0, no face
reduction, fp16 flash decoder. Writes mesh.glb/mesh.obj and run_info.json.
Run in the triposg env:  python run_triposg.py --repo <clone> --image X --out DIR
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    sys.path.insert(0, str(Path(args.repo) / "scripts"))

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from briarmbg import BriaRMBG
    from image_process import prepare_image
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    import trimesh

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timing = {}
    t = time.perf_counter()
    weights = snapshot_download("VAST-AI/TripoSG")
    rmbg_weights = snapshot_download("briaai/RMBG-1.4")
    rmbg = BriaRMBG.from_pretrained(rmbg_weights).to("cuda").eval()
    pipe = TripoSGPipeline.from_pretrained(weights).to("cuda", torch.float16)
    timing["load_s"] = time.perf_counter() - t

    t = time.perf_counter()
    image = prepare_image(args.image, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg)
    image.save(out / "input_processed.png")
    timing["preprocess_s"] = time.perf_counter() - t

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    with torch.no_grad():
        outputs = pipe(image=image, generator=torch.Generator(device=pipe.device).manual_seed(args.seed),
                       num_inference_steps=50, guidance_scale=7.0).samples[0]
    torch.cuda.synchronize()
    timing["inference_s"] = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / 2**30
    mesh = trimesh.Trimesh(outputs[0].astype("float32"), outputs[1])
    mesh.export(out / "mesh.glb")
    mesh.export(out / "mesh.obj")

    commit = subprocess.run(["git", "-C", args.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    info = dict(model="TripoSG", repo="https://github.com/VAST-AI-Research/TripoSG", commit=commit,
                hf_repo=["VAST-AI/TripoSG", "briaai/RMBG-1.4"], python=platform.python_version(),
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                command=sys.argv, input=str(Path(args.image).resolve()), seed=args.seed,
                settings=dict(steps=50, guidance_scale=7.0, faces=-1, dtype="float16",
                              background_removal="BriaRMBG-1.4"),
                timing=timing, peak_vram_gib=peak, vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)),
                textured=False, deviations=["repo imported from sys.path",
                                            "diffusers 0.32.2 / transformers 4.48.3 pinned for torch 2.5.1"])
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
