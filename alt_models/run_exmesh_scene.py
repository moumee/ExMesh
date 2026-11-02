"""Run the full ExMesh pipeline for one DTU scene with the repo's own settings.

PGSR coarse mesh (PGSR/scripts/run_dtu_init.py args) -> ExMesh train
(scripts/run_dtu.py per-scene lambdas) -> OBJ export -> DTU chamfer eval.
Steps whose outputs already exist are skipped. Wall time per step is written
to outputs/DTU/<scene>/pipeline_timing.json for the model comparison.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENE_PARAMS = {24: (3000, 0.1), 37: (3000, 0.1), 40: (3000, 0.03), 55: (3000, 0.01),
                63: (3000, 0.03), 65: (3000, 0.01), 69: (1000, 0.1), 83: (3000, 0.01),
                97: (3000, 0.01), 105: (3000, 0.03), 106: (3000, 0.1), 110: (1000, 0.01),
                114: (3000, 0.03), 118: (1000, 0.01), 122: (1000, 0.03)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=int, required=True)
    parser.add_argument("--iteration", type=int, default=10000)
    args = parser.parse_args()
    py = sys.executable
    scene = f"scan{args.scan}"
    data = ROOT / "workdir/DTU" / scene
    coarse = ROOT / "outputs/coarse_meshes" / f"dtu_{scene}" / "test"
    model = ROOT / "outputs/DTU" / scene
    model.mkdir(parents=True, exist_ok=True)
    timing_path = model / "pipeline_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    smooth, depth = SCENE_PARAMS[args.scan]

    def step(name, command, cwd, done):
        if done.exists():
            print(f"[skip] {name}: {done} exists", flush=True)
            return
        print(f"[run] {name}: {' '.join(map(str, command))}", flush=True)
        started = time.perf_counter()
        with (model / f"pipeline_{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(list(map(str, command)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
        timing[name] = time.perf_counter() - started
        timing_path.write_text(json.dumps(timing, indent=2))

    step("pgsr_train", [py, "train.py", "-s", data, "-m", coarse, "--quiet", "-r2", "--ncc_scale", "0.5"],
         ROOT / "PGSR", coarse / "point_cloud")
    step("pgsr_mesh", [py, "render.py", "-m", coarse, "--quiet", "--num_cluster", "1", "--voxel_size", "0.01",
                       "--max_depth", "5.0", "--use_depth_filter"],
         ROOT / "PGSR", coarse / "mesh/tsdf_fusion_post.ply")
    if not (data / "mesh.ply").exists():
        shutil.copy2(coarse / "mesh/tsdf_fusion_post.ply", data / "mesh.ply")
    step("exmesh_train", [py, "train.py", "-s", data, "-m", model, "--lambda_smooth", smooth,
                          "--lambda_depth", depth],
         ROOT, model / "mesh" / f"auto_mesh_iter_{args.iteration}.ply")
    step("export_obj", [py, "scripts/export_obj.py", "--model_path", model, "--iteration", args.iteration],
         ROOT, model / "export" / f"mesh_iter_{args.iteration}.obj")
    step("dtu_eval", [py, "scripts/eval_dtu/evaluate_single_scene.py", "--input_mesh",
                      model / "mesh" / f"auto_mesh_iter_{args.iteration}.ply", "--scan_id", args.scan,
                      "--output_dir", model / "eval", "--mask_dir", ROOT / "workdir/DTU", "--DTU", ROOT / "workdir/DTU"],
         ROOT, model / "eval/results.json")
    # evaluate_single_scene.py shells out to a bare `python` for eval.py; if that
    # interpreter lacks sklearn only the culling succeeds, so finish with ours.
    step("dtu_chamfer", [py, "scripts/eval_dtu/eval.py", "--data", model / "eval/culled_mesh.ply", "--scan", args.scan,
                         "--mode", "mesh", "--dataset_dir", ROOT / "workdir/DTU", "--vis_out_dir", model / "eval"],
         ROOT, model / "eval/results.json")
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
