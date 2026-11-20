"""Image -> textured mesh with TRELLIS.2 or Pixal3D through ComfyUI's native nodes (int8 weights).

Same interface and outputs as quick_generation/triposr_generate.py, so the plugin can call it the same way:

    <comfyui env>\\python.exe comfyui_generate.py --repo <ComfyUI dir> --image a.png --output out\\mesh.obj
        [--model trellis2|pixal3d] [--resolution 1024|1536] [--up z|y] [--height 100]

Writes (in this order, .complete.json last = success marker):
    <stem>.glb            PBR result from ComfyUI (base color, metallic/roughness, normal, AO)
    <stem>.obj/.mtl       welded positions + UVs, base color as <stem>_texture.png
    <stem>_input.png      the photo that was used
    <stem>.complete.json  settings, timing, peak GPU memory (nvidia-smi, whole GPU)

Requirements: ComfyUI >= 0.34 (tested v0.46.0) with the models of the "Pixal3D & TRELLIS.2: Image to Model"
template in models/ (diffusion_models, vae, clip_vision, background_removal, geometry_estimation), and trimesh.
Only the first --image is used (the native nodes are single-view).
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "comfyui_pixal3d_trellis2.json"  # copy of templates/3d_pixal3d_trellis2_image_to_model.json
UI_ONLY = {"MarkdownNote", "Note", "Reroute"}
CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COLOR", "LOAD_3D"}
DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"
# node ids in the template
NODE_IMAGE, NODE_USE_TRELLIS2, NODE_REMOVE_BG = "122", "316", "248"
NODE_UPSAMPLE, NODE_TEXTURE_RES, NODE_DECIMATE = "94", "288", "186"
# glTF (Y up, input view toward +Z; checked by aligning to the bear dataset) -> plugin frames.
Y_UP_TO_Z_UP = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))


# --- ComfyUI server -------------------------------------------------------------------------

def http(port, path, data=None, timeout=60):
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0]) if out else -1
    except (OSError, subprocess.SubprocessError, ValueError):
        return -1


def gpu_total_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0]) if out else -1
    except (OSError, subprocess.SubprocessError, ValueError):
        return -1


def headroom_for_budget(budget_gib):
    """ComfyUI's --vram-headroom keeps that much VRAM free *including other apps* (Unreal, desktop).
    To cap ComfyUI itself at `budget_gib`, keep free = total - (VRAM in use now) - budget."""
    total, used = gpu_total_mib(), gpu_used_mib()
    if total <= 0 or used < 0:
        return None
    return max(0.5, (total - used) / 1024 - budget_gib)


class GpuMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop_flag = -1, False

    def run(self):
        while not self.stop_flag:
            self.peak = max(self.peak, gpu_used_mib())
            time.sleep(0.5)


# --- UI workflow -> API prompt ----------------------------------------------------------------

def _is_widget(spec):
    return isinstance(spec[0], list) or spec[0] in WIDGET_TYPES or spec[0] == DYNAMIC_COMBO


def _skip_control(values, spec):
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if values and (opts.get("control_after_generate") or
                   (spec[0] == "INT" and isinstance(values[0], str) and values[0] in CONTROL_VALUES)):
        values.pop(0)


def _consume(name, spec, values, inputs, linked):
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if name in linked:
        inputs[name] = linked[name]
        if _is_widget(spec) and values:
            values.pop(0)
            _skip_control(values, spec)
        return
    if spec[0] == DYNAMIC_COMBO:  # selected key, then the chosen option's inputs as "parent.child"
        options = opts.get("options") or []
        key = values.pop(0) if values else (opts.get("default") or options[0]["key"])
        inputs[name] = key
        chosen = next(o for o in options if o["key"] == key)
        for group in ("required", "optional"):
            for sub, sub_spec in ((chosen.get("inputs") or {}).get(group) or {}).items():
                _consume(f"{name}.{sub}", sub_spec, values, inputs, linked)
        return
    if _is_widget(spec) and values:
        inputs[name] = values.pop(0)
        _skip_control(values, spec)


def workflow_to_api(workflow, object_info):
    links = {l[0]: l for l in workflow["links"]}
    api = {}
    for node in workflow["nodes"]:
        if node["type"] in UI_ONLY or node.get("mode", 0) in (2, 4):
            continue
        spec = object_info.get(node["type"])
        if spec is None:
            raise RuntimeError(f"ComfyUI has no node '{node['type']}': update ComfyUI (>= 0.34)")
        linked = {i["name"]: [str(links[i["link"]][1]), links[i["link"]][2]]
                  for i in node.get("inputs", []) if i.get("link") is not None}
        order = spec.get("input_order") or {g: list(v) for g, v in spec["input"].items()}
        values, inputs = list(node.get("widgets_values") or []), {}
        for group in ("required", "optional"):
            for name in order.get(group, []):
                _consume(name, spec["input"][group][name], values, inputs, linked)
        api[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    return api


def prune_unused(api, model):
    """Drop the loader of the model that is not used, so its weights need not be installed.

    ComfyUI validates every loader's file name before running, even on the unused switch branch.
    """
    if model == "pixal3d":  # TRELLIS.2 UNet (node 40) only feeds switch 318's true branch
        api.pop("40", None)
        api["318"]["inputs"]["on_true"] = ["319", 0]
    return api


def run_comfyui(repo, python, image, args, log_path):
    port = args.port or free_port()
    input_name = f"rb_{uuid.uuid4().hex[:8]}{image.suffix.lower()}"
    shutil.copy2(image, repo / "input" / input_name)
    # PYTHONNOUSERSITE: another Python of the same version can leave packages in the shared user site,
    # which would shadow the env's own versions.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONNOUSERSITE": "1"}
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    cmd = [str(python), "-u", "main.py", "--listen", "127.0.0.1", "--port", str(port),
           "--disable-auto-launch", "--preview-method", "none"]
    if args.vram_budget and args.vram_budget > 0:
        headroom = headroom_for_budget(args.vram_budget)
        if headroom is not None:
            cmd += ["--vram-headroom", f"{headroom:.2f}"]
            args.vram_headroom_used = round(headroom, 2)
    cmd += args.comfy_args
    timing = {}
    with open(log_path, "w", encoding="utf-8") as log:
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"ComfyUI exited during startup (code {proc.returncode}), see {log_path}")
                try:
                    http(port, "/system_stats", timeout=5)
                    break
                except (urllib.error.URLError, OSError):
                    if time.time() - t0 > 600:
                        raise RuntimeError("ComfyUI did not start within 10 minutes")
                    time.sleep(1)
            timing["startup_s"] = time.time() - t0
            api = workflow_to_api(json.loads(Path(args.workflow).read_text(encoding="utf-8")),
                                  http(port, "/object_info", timeout=120))
            api[NODE_IMAGE]["inputs"]["image"] = input_name
            api[NODE_USE_TRELLIS2]["inputs"]["value"] = args.model == "trellis2"
            prune_unused(api, args.model)
            use_alpha = has_alpha(image)
            api[NODE_REMOVE_BG]["inputs"]["switch"] = not use_alpha
            if use_alpha:
                # LoadImage's MASK is 1 - alpha and the template feeds it to ImageCropToMask as is,
                # which would keep the background; invert it back to the alpha.
                api["9001"] = {"class_type": "InvertMask", "inputs": {"mask": [NODE_IMAGE, 1]}}
                api[NODE_REMOVE_BG]["inputs"]["on_false"] = ["9001", 0]
            api[NODE_UPSAMPLE]["inputs"]["target_resolution"] = args.resolution
            api[NODE_TEXTURE_RES]["inputs"]["value"] = args.texture_size
            api[NODE_DECIMATE]["inputs"]["target_face_count"] = args.faces
            prefix = f"3d/rb_{uuid.uuid4().hex[:8]}"
            api["322"]["inputs"]["filename_prefix"] = prefix
            t1 = time.time()
            try:
                pid = http(port, "/prompt", {"prompt": api, "client_id": uuid.uuid4().hex})["prompt_id"]
            except urllib.error.HTTPError as e:
                raise RuntimeError("ComfyUI rejected the workflow: " + e.read().decode("utf-8", "replace")[:2000])
            while True:
                time.sleep(2)
                hist = http(port, f"/history/{pid}", timeout=30)
                if pid in hist:
                    status = hist[pid].get("status", {})
                    break
                if proc.poll() is not None:
                    raise RuntimeError(f"ComfyUI exited while generating (code {proc.returncode}), see {log_path}")
            timing["generate_s"] = time.time() - t1
            if status.get("status_str") != "success":
                errors = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"generation failed: {json.dumps(errors)[:2000]}")
            glbs = sorted((repo / "output").glob(prefix + "*.glb"), key=lambda p: p.stat().st_mtime)
            if not glbs:
                raise RuntimeError("ComfyUI finished but wrote no GLB")
            return glbs[-1], timing
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            (repo / "input" / input_name).unlink(missing_ok=True)


def has_alpha(path):
    from PIL import Image

    with Image.open(path) as im:
        if "A" not in im.getbands():
            return False
        lo, hi = im.getchannel("A").getextrema()
        return lo < 250 and hi > 5


# --- GLB -> OBJ (welded positions, separate UVs) ----------------------------------------------

def level_rotation(vertices, faces, max_tilt_deg):
    """Rotation that sets the mesh on its nearest stable resting pose (within max_tilt_deg of the current up).

    Pixal3D builds the mesh in the photo's camera frame, so a photo taken from above gives a mesh tilted
    by the camera's elevation. Returns (3x3 rotation, tilt in degrees); identity if no pose qualifies.
    """
    import numpy as np
    import trimesh

    tm = trimesh.Trimesh(vertices, faces, process=False)
    transforms, probs = trimesh.poses.compute_stable_poses(tm, center_mass=tm.centroid, n_samples=1)
    best, best_tilt = None, max_tilt_deg
    for T, p in zip(transforms, probs):
        new_up = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])  # mesh direction that this pose puts up
        tilt = float(np.degrees(np.arccos(np.clip(new_up[2], -1.0, 1.0))))
        if p > 0.01 and tilt < best_tilt:
            best, best_tilt = new_up, tilt
    if best is None or best_tilt < 1.0:
        return np.eye(3), 0.0
    axis = np.cross(best, [0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    angle = np.radians(best_tilt)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K, best_tilt


def write_obj(glb, output, up, front, height, max_tilt):
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(glb), force="mesh", process=False)
    v = np.asarray(mesh.vertices, np.float64) @ np.asarray(Y_UP_TO_Z_UP).T  # -> Z up
    # rotate about Z so the photo side faces +X
    angle = -np.arctan2(front[1], front[0])
    c, s = np.cos(angle), np.sin(angle)
    v = v @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T
    tilt = 0.0
    if max_tilt > 0:  # stand the object on its base
        rot, tilt = level_rotation(v, np.asarray(mesh.faces), max_tilt)
        v = v @ rot.T
    v -= np.array([(v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2, v[:, 2].min()])
    scale = 1.0
    if height:
        scale = height / max(float(np.ptp(v[:, 2])), 1e-9)
        v *= scale
    if up == "y":  # Y up, photo side +Z
        v = v @ np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).T
    # Write float32 values with 9 significant digits (like triposr_generate.py): Open3D reads OBJ
    # coordinates as float32, and the Wild simplifier checks its corners against the OBJ text.
    uv = np.asarray(mesh.visual.uv, np.float32)
    faces = np.asarray(mesh.faces, np.int64)
    pos, inverse = np.unique(v.astype(np.float32), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    stem = output.with_suffix("")
    texture = stem.parent / f"{stem.name}_texture.png"
    material = mesh.visual.material
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is None:
        raise RuntimeError("GLB has no base color texture")
    image.convert("RGB").save(texture)
    with open(output.with_suffix(".mtl"), "w", encoding="utf-8") as f:
        f.write(f"newmtl material_0\nKd 1 1 1\nmap_Kd {texture.name}\n")
    with open(output, "w", encoding="utf-8") as f:
        f.write(f"mtllib {output.with_suffix('.mtl').name}\nusemtl material_0\n")
        f.write("".join(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in pos))
        f.write("".join(f"vt {a:.9g} {b:.9g}\n" for a, b in uv))
        tri_pos, tri_uv = inverse[faces] + 1, faces + 1
        f.write("".join(f"f {p[0]}/{t[0]} {p[1]}/{t[1]} {p[2]}/{t[2]}\n" for p, t in zip(tri_pos, tri_uv)))
    return {"vertices": int(len(pos)), "faces": int(len(faces)), "scale": scale, "leveled_deg": round(tilt, 1),
            "bbox_min": v.min(0).tolist(), "bbox_max": v.max(0).tolist(), "texture": texture.name}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="ComfyUI directory")
    ap.add_argument("--image", action="append", required=True, help="input photo (RGBA alpha skips BiRefNet)")
    ap.add_argument("--output", required=True, help="output OBJ path")
    ap.add_argument("--model", choices=("trellis2", "pixal3d"), default="pixal3d")
    ap.add_argument("--resolution", type=int, choices=(1024, 1536), default=1024)
    ap.add_argument("--texture_size", type=int, default=2048)
    ap.add_argument("--faces", type=int, default=200000, help="decimation target before UV unwrapping")
    ap.add_argument("--up", choices=("z", "y"), default="z")
    ap.add_argument("--height", type=float, default=None, help="scale so the mesh is this tall")
    ap.add_argument("--front", default="0,-1",
                    help="photo side of the generated mesh after Y->Z up, as 'x,y' (glTF +Z = input view -> 0,-1)")
    ap.add_argument("--vram_budget", type=float, default=4.0,
                    help="target GB of VRAM for ComfyUI (weights stream from RAM beyond it; 0 = no limit). "
                         "Measured peak with 4: about 4.4 GB (VAE decoding at resolution 1024 sets the floor)")
    ap.add_argument("--max_tilt", type=float, default=45.0,
                    help="stand the mesh on its nearest stable pose within this many degrees (0 = off)")
    ap.add_argument("--python", default=sys.executable, help="python that runs ComfyUI")
    ap.add_argument("--workflow", default=str(TEMPLATE))
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--comfy_args", nargs=argparse.REMAINDER, default=[], help="extra ComfyUI arguments")
    args = ap.parse_args()

    started = time.time()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    done = output.with_suffix(".complete.json")
    done.unlink(missing_ok=True)
    repo, image = Path(args.repo).resolve(), Path(args.image[0]).resolve()
    if len(args.image) > 1:
        print(f"note: only the first image is used ({len(args.image)} given)")
    stem = output.with_suffix("")
    shutil.copy2(image, stem.parent / f"{stem.name}_input.png")

    monitor = GpuMonitor()
    baseline = gpu_used_mib()
    monitor.start()
    try:
        glb, timing = run_comfyui(repo, args.python, image, args, stem.parent / f"{stem.name}_comfyui.log")
    finally:
        monitor.stop_flag = True
        monitor.join(timeout=5)
    shutil.copy2(glb, output.with_suffix(".glb"))
    t2 = time.time()
    front = tuple(float(x) for x in args.front.split(","))
    info = write_obj(output.with_suffix(".glb"), output, args.up, front, args.height, args.max_tilt)
    timing["export_s"] = time.time() - t2
    report = {
        "model": {"trellis2": "TRELLIS.2 (ComfyUI native, int8)", "pixal3d": "Pixal3D (ComfyUI native, int8)"}[args.model],
        "image": str(image), "output": str(output), "background": "input alpha" if has_alpha(image) else "BiRefNet",
        "settings": {"resolution": args.resolution, "texture_size": args.texture_size, "faces": args.faces,
                     "up": args.up, "height": args.height, "vram_budget_gib": args.vram_budget,
                     "vram_headroom_gib": getattr(args, "vram_headroom_used", None)},
        **info,
        "timing": {k: round(v, 1) for k, v in timing.items()},
        "seconds": round(time.time() - started, 1),
        "gpu_baseline_mib": baseline, "gpu_peak_mib": monitor.peak,
        "peak_vram_gib": round(max(monitor.peak - baseline, 0) / 1024, 2),
    }
    done.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
