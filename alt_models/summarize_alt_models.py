"""Collect reconstruction-source comparison results into one CSV + JSON.

Reads outputs/alt_models/<scene>/<model>/{run_info.json, dtu_eval/results.json,
photo_<model>.json} and the ExMesh pipeline timing. Writes
deliverables/enhanced_simplification/reconstruction_sources.{csv,json}.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deliverables/enhanced_simplification"
MODELS = {"exmesh": "ExMesh (multi-view)", "triposr": "TripoSR", "triposg": "TripoSG",
          "hunyuan3d2": "Hunyuan3D-2 (dit-v2-0)", "hunyuan3d2_light": "Hunyuan3D-2mini-turbo (light)"}
LICENSE = {"exmesh": "repo LICENSE.md", "triposr": "MIT", "triposg": "MIT (+BriaRMBG-1.4 non-commercial if used)",
           "hunyuan3d2": "Tencent Hunyuan 3D 2.0 Community: excludes EU/UK/South Korea",
           "hunyuan3d2_light": "Tencent Hunyuan 3D 2.0 Community: excludes EU/UK/South Korea"}


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    rows = []
    for scene in ("scan83", "scan24"):
        for model, name in MODELS.items():
            directory = ROOT / "outputs/alt_models" / scene / model
            if not directory.exists():
                continue
            chamfer = load(directory / "dtu_eval/results.json")
            if model == "exmesh" and chamfer is None:
                chamfer = load(ROOT / "outputs/DTU" / scene / "eval/results.json")
            photo = load(directory / f"photo_{model}.json")
            run = load(directory / "run_info.json")
            row = dict(scene=scene, model=name, license=LICENSE[model])
            if run and model == "triposg":
                row["model"] += " (BriaRMBG input)" if run["input"].endswith("_rgb.png") else " (DTU-mask RGBA input)"
            if chamfer:
                row.update(chamfer_mm=chamfer["overall"], accuracy_mm=chamfer["mean_d2s"], completeness_mm=chamfer["mean_s2d"])
            if photo:
                summary = photo["summary"]
                row.update(faces=summary["faces"], textured=summary["textured"],
                           **{f"photo_{k}": v for k, v in summary["excluding_input_view"].items()})
            if run:
                timing = run["timing"]
                compute = sum(v for k, v in timing.items() if not k.endswith("load_s"))
                peak = run["peak_vram_gib"]
                row.update(seconds_excluding_model_load=compute,
                           peak_vram_gib=max(peak.values()) if isinstance(peak, dict) else peak)
            elif model == "exmesh":
                timing = load(ROOT / "outputs/DTU" / scene / "pipeline_timing.json")
                if timing:
                    row.update(seconds_excluding_model_load=sum(v for k, v in timing.items()
                                                                if k in ("pgsr_train", "pgsr_mesh", "exmesh_train")))
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reconstruction_sources.json").write_text(json.dumps(rows, indent=2))
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (OUT / "reconstruction_sources.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "license"})


if __name__ == "__main__":
    main()
