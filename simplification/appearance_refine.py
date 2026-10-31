"""Appearance-driven refinement of a simplified textured OBJ (post-pass).

After Wild simplification + texture transfer, optimize the simplified mesh so
its renders match renders of the ORIGINAL mesh from random views over the full
sphere (no capture cameras, no evaluation poses):
  --mode tex   texture atlas only (an optimal multi-view re-bake)
  --mode full  texture atlas + per-vertex offsets with a Laplacian regularizer
Topology, UV layout and face order are unchanged, so the result imports into
UE5 exactly like the simplifier's output. Both atlases are edge-padded (every
texel outside the UV charts copies its nearest chart texel), so the black atlas
background never bleeds into chart borders through mip levels. Inspired by Hasselgren et al.,
"Appearance-Driven Automatic 3D Model Simplification" (EGSR 2021); here the
optimization starts from the Wild result instead of replacing simplification.

Requires CUDA + nvdiffrast (the ExMesh env).
  python appearance_refine.py --reference orig.obj --input simplified.obj --output refined.obj --mode full
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def read_obj(path):
    """Positions, UVs and triangle (position, uv) indices of a textured OBJ."""
    v, vt, f, ft, mtllib = [], [], [], [], None
    with open(path, encoding="utf-8", errors="ignore") as source:
        for line in source:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                v.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vt":
                vt.append([float(x) for x in parts[1:3]])
            elif parts[0] == "mtllib":
                mtllib = parts[1]
            elif parts[0] == "f":
                corners = [p.split("/") for p in parts[1:]]
                if len(corners) != 3 or any(len(c) < 2 or not c[1] for c in corners):
                    raise ValueError(f"{path}: only triangles with UVs are supported")
                f.append([int(c[0]) - 1 for c in corners])
                ft.append([int(c[1]) - 1 for c in corners])
    return (np.asarray(v, np.float32), np.asarray(vt, np.float32), np.asarray(f, np.int32),
            np.asarray(ft, np.int32), mtllib)


def find_texture(obj_path, mtllib, explicit=None):
    if explicit:
        return Path(explicit)
    obj_path = Path(obj_path)
    if mtllib and (obj_path.parent / mtllib).exists():
        for line in (obj_path.parent / mtllib).read_text(errors="ignore").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].lower() == "map_kd":
                return obj_path.parent / parts[1].strip()
    for candidate in (obj_path.with_suffix(".png"), obj_path.with_name(obj_path.stem + "_texture.png")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No texture for {obj_path}")


class TexturedMesh:
    def __init__(self, path, texture=None, device="cuda"):
        v, vt, f, ft, mtllib = read_obj(path)
        self.texture_path = find_texture(path, mtllib, texture)
        image = np.asarray(Image.open(self.texture_path).convert("RGB"), np.float32) / 255
        # dr.texture puts uv (0,0) at memory row 0; OBJ v=0 is the image bottom.
        self.texture = torch.from_numpy(np.ascontiguousarray(image[::-1])).to(device)[None]
        self.pos = torch.from_numpy(v).to(device)
        self.uv = torch.from_numpy(vt).to(device)
        self.faces = torch.from_numpy(f).to(device)
        self.uv_faces = torch.from_numpy(ft).to(device)


def look_at_mvp(eye, center, up, fovy, near, far):
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, true_up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    t = 1 / math.tan(fovy / 2)
    proj = np.array([[t, 0, 0, 0], [0, t, 0, 0],
                     [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]])
    return proj @ view


def random_cameras(count, center, radius, rng, fovy=math.radians(40), distance=(0.5, 1.2)):
    """Uniform full-sphere directions; distance factor < 1 gives close-ups."""
    mvps = []
    fit = radius / math.sin(fovy / 2)
    for _ in range(count):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        up = rng.normal(size=3)
        up -= direction * (up @ direction)
        eye = center + direction * fit * rng.uniform(*distance)
        dist = np.linalg.norm(eye - center)
        mvps.append(look_at_mvp(eye, center, up / np.linalg.norm(up), fovy, max(1e-3, dist - 1.5 * radius) * 0.5,
                                dist + 1.5 * radius))
    return np.stack(mvps).astype(np.float32)


def max_mip_level(shape):
    """Deepest mip level nvdiffrast can build (each level halves both sides)."""
    level, (h, w) = 0, shape
    while h % 2 == 0 and w % 2 == 0 and min(h, w) > 1:
        h, w, level = h // 2, w // 2, level + 1
    return level


def render(dr, glctx, pos, faces, uv, uv_faces, texture, mvp, resolution, background):
    clip = torch.einsum("vi,bji->bvj", torch.nn.functional.pad(pos, (0, 1), value=1.0), mvp).contiguous()
    rast, rast_db = dr.rasterize(glctx, clip, faces, (resolution, resolution))
    texc, texd = dr.interpolate(uv[None], rast, uv_faces, rast_db=rast_db, diff_attrs="all")
    color = dr.texture(texture, texc, texd, filter_mode="linear-mipmap-linear",
                       max_mip_level=max_mip_level(texture.shape[1:3]))
    color = torch.where(rast[..., 3:] > 0, color, background[:, None, None, :])
    return dr.antialias(color.contiguous(), rast, clip, faces)


def uv_coverage(dr, glctx, uv, uv_faces, shape, tile=1024):
    """Texels whose centre lies in a UV triangle (or holds a UV vertex), in dr.texture memory layout."""
    height, width = shape
    covered = torch.zeros(height, width, dtype=torch.bool, device=uv.device)
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            h, w = min(tile, height - y0), min(tile, width - x0)
            xy = torch.stack(((uv[:, 0] * width - x0) / w * 2 - 1, (uv[:, 1] * height - y0) / h * 2 - 1), 1)
            clip = torch.nn.functional.pad(xy, (0, 2), value=1.0)
            clip[:, 2] = 0
            rast, _ = dr.rasterize(glctx, clip[None].contiguous(), uv_faces, (h, w))
            covered[y0:y0 + h, x0:x0 + w] = rast[0, ..., 3] > 0
    column = (uv[:, 0] * width - 0.5).round().long().clamp(0, width - 1)
    row = (uv[:, 1] * height - 0.5).round().long().clamp(0, height - 1)
    covered[row, column] = True
    return covered


def edge_padded(dr, glctx, mesh):
    """mesh.texture with every texel outside the UV charts replaced by its nearest chart texel."""
    return padded(mesh.texture, edge_pad_index(uv_coverage(dr, glctx, mesh.uv, mesh.uv_faces, mesh.texture.shape[1:3])))


def edge_pad_index(covered):
    """Flat gather index sending every texel to its nearest covered texel (bake-style edge padding).

    Without it the black atlas background bleeds into chart borders at coarse mip levels:
    the reference renders then show dark seams that the optimization copies into the texture.
    """
    from scipy import ndimage
    rows, cols = ndimage.distance_transform_edt(~covered.cpu().numpy(), return_distances=False, return_indices=True)
    return torch.from_numpy((rows * covered.shape[1] + cols).ravel()).to(covered.device)


def padded(texture, index):
    return texture.reshape(-1, texture.shape[-1])[index].reshape(texture.shape)


def uniform_laplacian(faces, count, device):
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).long()
    edges = torch.cat([edges, edges.flip(1)])
    edges = torch.unique(edges, dim=0)
    values = torch.ones(len(edges), device=device)
    adjacency = torch.sparse_coo_tensor(edges.T, values, (count, count)).coalesce()
    degree = torch.sparse.sum(adjacency, 1).to_dense().clamp(min=1)
    return adjacency, degree


def write_output(source_obj, output, positions, texture):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    texture_path = output.with_name(output.stem + "_texture.png")
    mtl_path = output.with_suffix(".mtl")
    Image.fromarray(np.clip(np.rint(texture[::-1] * 255), 0, 255).astype(np.uint8)).save(texture_path)
    mtl_path.write_text("newmtl material0\nKa 1.000000 1.000000 1.000000\nKd 1.000000 1.000000 1.000000\n"
                        f"Ks 0.000000 0.000000 0.000000\nmap_Kd {texture_path.name}\n", encoding="utf-8")
    vertex = 0
    with open(source_obj, encoding="utf-8") as src, open(output, "w", encoding="utf-8") as dst:
        for line in src:
            if line.startswith("v "):
                x, y, z = positions[vertex]
                dst.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
                vertex += 1
            elif line.startswith("mtllib"):
                dst.write(f"mtllib {mtl_path.name}\n")
            else:
                dst.write(line)
    assert vertex == len(positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Original (pre-simplification) textured OBJ")
    parser.add_argument("--reference_texture")
    parser.add_argument("--input", required=True, help="Simplified textured OBJ to refine")
    parser.add_argument("--input_texture")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("tex", "full"), default="full")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--lr_texture", type=float, default=0.01)
    parser.add_argument("--lr_position", type=float, default=2e-5, help="Adam step, fraction of bbox diagonal")
    parser.add_argument("--laplacian", type=float, default=10.0,
                        help="Weight of mean squared uniform Laplacian of offsets (in units of 1e-3 bbox diag)")
    parser.add_argument("--validation_views", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import nvdiffrast.torch as dr
    device = "cuda"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    reference = TexturedMesh(args.reference, args.reference_texture, device)
    mesh = TexturedMesh(args.input, args.input_texture, device)
    glctx = dr.RasterizeCudaContext()

    low, high = reference.pos.min(0).values.cpu().numpy(), reference.pos.max(0).values.cpu().numpy()
    center, diagonal = (low + high) / 2, float(np.linalg.norm(high - low))
    radius = diagonal / 2

    reference.texture = edge_padded(dr, glctx, reference)
    # The optimized texels are the covered ones; gutter and background always copy their nearest one.
    pad = edge_pad_index(uv_coverage(dr, glctx, mesh.uv, mesh.uv_faces, mesh.texture.shape[1:3]))
    texture = mesh.texture.clone().requires_grad_(True)
    offset = torch.zeros_like(mesh.pos, requires_grad=True)
    groups = [dict(params=[texture], lr=args.lr_texture)]
    if args.mode == "full":
        groups.append(dict(params=[offset], lr=args.lr_position * diagonal))
        adjacency, degree = uniform_laplacian(mesh.faces, len(mesh.pos), device)
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: 0.1 ** (i / args.iterations))

    # Fixed validation views from an independent seed; reference renders cached.
    val_rng = np.random.default_rng(10_000 + args.seed)
    val_mvp = torch.from_numpy(random_cameras(args.validation_views, center, radius, val_rng)).to(device)
    val_bg = torch.zeros(args.validation_views, 3, device=device)
    with torch.no_grad():
        val_ref = torch.cat([render(dr, glctx, reference.pos, reference.faces, reference.uv, reference.uv_faces,
                                    reference.texture, val_mvp[i:i + 4], args.resolution, val_bg[i:i + 4])
                             for i in range(0, args.validation_views, 4)])

    def validate():
        with torch.no_grad():
            pos = mesh.pos + offset
            mse = torch.cat([((render(dr, glctx, pos, mesh.faces, mesh.uv, mesh.uv_faces, padded(texture, pad),
                                      val_mvp[i:i + 4], args.resolution, val_bg[i:i + 4])
                               - val_ref[i:i + 4]) ** 2).mean((1, 2, 3))
                             for i in range(0, args.validation_views, 4)])
            return float((-10 * torch.log10(mse.clamp(min=1e-12))).mean())

    history = [(0, validate())]
    print(f"[refine] {args.mode}: validation PSNR before {history[0][1]:.3f} dB", flush=True)
    for iteration in range(1, args.iterations + 1):
        mvp = torch.from_numpy(random_cameras(args.batch, center, radius, rng)).to(device)
        background = torch.rand(args.batch, 3, device=device)
        with torch.no_grad():
            target = render(dr, glctx, reference.pos, reference.faces, reference.uv, reference.uv_faces,
                            reference.texture, mvp, args.resolution, background)
        pos = mesh.pos + offset
        image = render(dr, glctx, pos, mesh.faces, mesh.uv, mesh.uv_faces, padded(texture, pad), mvp,
                       args.resolution, background)
        loss = ((image - target) ** 2).mean()
        if args.mode == "full":
            laplacian = offset - torch.sparse.mm(adjacency, offset) / degree[:, None]
            loss = loss + args.laplacian * (laplacian / (1e-3 * diagonal)).pow(2).sum(1).mean() * 1e-4
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            texture.clamp_(0, 1)
        if iteration % 500 == 0 or iteration == args.iterations:
            history.append((iteration, validate()))
            print(f"[refine] iter {iteration}: validation PSNR {history[-1][1]:.3f} dB "
                  f"max offset {offset.norm(dim=1).max().item() / diagonal:.2e} diag", flush=True)

    final_pos = (mesh.pos + offset).detach().cpu().numpy().astype(np.float64)
    write_output(args.input, args.output, final_pos, padded(texture.detach(), pad)[0].cpu().numpy())
    report = dict(mode=args.mode, input=str(Path(args.input).resolve()), reference=str(Path(args.reference).resolve()),
                  settings={k: v for k, v in vars(args).items() if k not in ("reference", "input", "output")},
                  validation_psnr=history, seconds=time.perf_counter() - started,
                  peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                  edge_padding="nearest covered texel (reference and output)",
                  max_offset_fraction_of_diagonal=float(offset.norm(dim=1).max()) / diagonal,
                  mean_offset_fraction_of_diagonal=float(offset.norm(dim=1).mean()) / diagonal)
    Path(args.output).with_suffix(".complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
