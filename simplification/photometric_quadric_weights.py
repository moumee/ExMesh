"""Multi-view photometric parallax weights for Wild QEM (uses ExMesh capture views).

Linearized rendering error of a surface offset. Displacing original face f by h
along its normal n_f moves the texture it carries (Wild transfers texture by
closest-point mapping, i.e. along the normal) by J_v n_f h pixels in capture
view v, where J_v = d(projection_v)/dX is the 2x3 camera Jacobian. To first
order the rendered color at a pixel x covered by f changes by
    dI_v(x) = -grad I_v(x) . J_v(X(x)) n_f h .
Summing squared errors over every capture view and every pixel where f is the
visible surface gives
    E = sum_f W_f h_f^2,   W_f = sum_v sum_{x in f} || grad I_v(x) J_v(X(x)) n_f ||^2
(RGB summed). Because h_f = n_f.v + d_f for Wild's memory planes, E is the
weighted plane quadric  Q = sum_f W_f p_f p_f^T. The script saves the energy
density W_f / A_f; the simplifier (--method photometric) uses face weights
1 + lambda * rho_f / rho_mean with rho_mean = sum W / sum A ("equal energy").

Compared with the Sobel weight (|grad I| averaged, P95-clipped, lambda-tuned)
this keeps only the gradient component along the screen-space displacement,
includes the view-angle factor (a face seen head-on cannot shift on screen),
uses squared energy, sums over views instead of averaging, and covers
silhouettes (object/background contrast) automatically. Pixels just behind an
occlusion boundary are skipped (that contrast belongs to the occluder), as are
pixels within the filter radius of the image frame (zero padding is not an edge).
The atlas is edge-padded before rendering: its black background would otherwise
darken chart borders at coarse mip levels and add gradients along UV seams.

  python photometric_quadric_weights.py --mesh mesh.obj --cameras cameras.json --output weights.npy
Requires CUDA + nvdiffrast (ExMesh env). --source photo uses the capture
photos (RGBA alpha as mask) instead of renders of the mesh.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from appearance_refine import TexturedMesh, edge_padded, max_mip_level


def camera_mvp(camera, near, far):
    rotation = np.asarray(camera["rotation"], dtype=np.float64)  # camera-to-world, x right / y down / z forward
    center = np.asarray(camera["position"], dtype=np.float64)
    view = np.eye(4)
    view[:3, :3] = rotation.T
    view[:3, 3] = -rotation.T @ center
    w, h = camera["width"], camera["height"]
    # nvdiffrast row i <-> y_ndc = -1 + (2i+1)/H, so image rows stay top-down.
    proj = np.array([[2 * camera["fx"] / w, 0, 0, 0], [0, 2 * camera["fy"] / h, 0, 0],
                     [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, 1, 0]])
    return proj @ view, view


def image_gradient(image, sigma=1.0):
    """Gaussian(sigma) then Sobel 3x3 / 8: d(color)/d(pixel) as (H, W, 3, 2)."""
    radius = int(np.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=image.device)
    gauss = torch.exp(-x * x / (2 * sigma * sigma))
    gauss /= gauss.sum()
    chw = image.permute(2, 0, 1)[:, None]
    chw = torch.nn.functional.conv2d(chw, gauss.view(1, 1, 1, -1), padding=(0, radius))
    chw = torch.nn.functional.conv2d(chw, gauss.view(1, 1, -1, 1), padding=(radius, 0))
    sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device) / 8
    gx = torch.nn.functional.conv2d(chw, sobel.view(1, 1, 3, 3), padding=1)[:, 0]
    gy = torch.nn.functional.conv2d(chw, sobel.T.contiguous().view(1, 1, 3, 3), padding=1)[:, 0]
    return torch.stack((gx, gy), -1).permute(1, 2, 0, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Original textured OBJ (same face order as the simplifier input)")
    parser.add_argument("--texture")
    parser.add_argument("--cameras", required=True, help="ExMesh cameras.json (capture views)")
    parser.add_argument("--images", help="Capture image folder, required for --source photo")
    parser.add_argument("--source", choices=("render", "photo"), default="render")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--occlusion_tolerance", type=float, default=0.01,
                        help="Relative depth gap that marks a nearer occluder inside the gradient stencil")
    args = parser.parse_args()

    import nvdiffrast.torch as dr
    started = time.perf_counter()
    device = "cuda"
    mesh = TexturedMesh(args.mesh, args.texture, device)
    glctx = dr.RasterizeCudaContext()
    if args.source == "render":
        mesh.texture = edge_padded(dr, glctx, mesh)
    pos = mesh.pos.double()
    tri = pos[mesh.faces.long()]
    cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    area = cross.norm(dim=1) / 2
    normals = (cross / cross.norm(dim=1, keepdim=True).clamp(min=1e-30)).float()
    energy = torch.zeros(len(mesh.faces), dtype=torch.float64, device=device)
    pixels = torch.zeros(len(mesh.faces), dtype=torch.float64, device=device)
    radius = float((mesh.pos.max(0).values - mesh.pos.min(0).values).norm())
    cameras = json.loads(Path(args.cameras).read_text())
    homogeneous = torch.nn.functional.pad(mesh.pos, (0, 1), value=1.0)
    for camera in cameras:
        depth = float(np.linalg.norm(np.asarray(camera["position"]) - mesh.pos.mean(0).cpu().numpy()))
        mvp, view = camera_mvp(camera, max(1e-3, (depth - radius) * 0.5), depth + 2 * radius)
        clip = (homogeneous @ torch.tensor(mvp, dtype=torch.float32, device=device).T)[None].contiguous()
        h, w = camera["height"], camera["width"]
        rast, rast_db = dr.rasterize(glctx, clip, mesh.faces, (h, w))
        if args.source == "render":
            texc, texd = dr.interpolate(mesh.uv[None], rast, mesh.uv_faces, rast_db=rast_db, diff_attrs="all")
            color = dr.texture(mesh.texture, texc, texd, filter_mode="linear-mipmap-linear",
                               max_mip_level=max_mip_level(mesh.texture.shape[1:3]))
            color = torch.where(rast[..., 3:] > 0, color, torch.zeros_like(color))
            image = dr.antialias(color.contiguous(), rast, clip, mesh.faces)[0]
        else:
            rgba = np.asarray(Image.open(Path(args.images) / f"{camera['img_name']}.png").convert("RGBA"), np.float32) / 255
            image = torch.from_numpy(rgba[..., :3] * rgba[..., 3:]).to(device)
        gradient = image_gradient(image, args.sigma)                      # (H, W, 3, 2)
        face = rast[0, ..., 3].long() - 1
        view_t = torch.tensor(view, dtype=torch.float32, device=device)
        position = dr.interpolate(mesh.pos[None], rast, mesh.faces)[0][0]
        depth = torch.where(face >= 0, position @ view_t[2, :3] + view_t[2, 3], torch.full_like(position[..., 0], float("inf")))
        # A pixel whose gradient stencil reaches a clearly nearer surface sits behind an
        # occlusion boundary: that contrast moves with the occluder, not with this face.
        reach = int(np.ceil(3 * args.sigma)) + 1
        nearest = -torch.nn.functional.max_pool2d(-depth[None, None], 2 * reach + 1, stride=1, padding=reach)[0, 0]
        behind = nearest < depth * (1 - args.occlusion_tolerance)
        # The filters zero-pad, so the image frame itself looks like an edge; skip that band.
        interior = torch.zeros_like(behind)
        interior[reach:-reach, reach:-reach] = True
        visible = (face >= 0) & ~behind & interior
        f = face[visible]
        point = position[visible]
        pc = point @ view_t[:3, :3].T + view_t[:3, 3]                      # camera coordinates
        nc = normals[f] @ view_t[:3, :3].T                                 # normal in camera frame
        z = pc[:, 2:3].clamp(min=1e-6)
        # J n: screen motion (pixels) per unit displacement along the face normal.
        jn = torch.cat((camera["fx"] * (nc[:, 0:1] * z - pc[:, 0:1] * nc[:, 2:3]) / z ** 2,
                        camera["fy"] * (nc[:, 1:2] * z - pc[:, 1:2] * nc[:, 2:3]) / z ** 2), 1)
        change = (gradient[visible] * jn[:, None, :]).sum(-1)              # (N, 3) color change per unit h
        energy.index_add_(0, f, (change.double() ** 2).sum(1))
        pixels.index_add_(0, f, torch.ones_like(f, dtype=torch.float64))
    density = (energy / area.clamp(min=1e-30)).cpu().numpy()
    density[area.cpu().numpy() <= 0] = 0.0
    np.save(args.output, density.astype(np.float64))
    seen = pixels.cpu().numpy() > 0
    summary = dict(mesh=str(Path(args.mesh).resolve()), cameras=str(Path(args.cameras).resolve()), source=args.source,
                   edge_padded_texture=args.source == "render",
                   sigma=args.sigma, occlusion_tolerance=args.occlusion_tolerance, views=len(cameras), faces=int(len(density)), seen_fraction=float(seen.mean()),
                   mean_density_area_weighted=float(energy.sum() / area.sum()),
                   density_percentiles={p: float(np.percentile(density[seen], p)) for p in (50, 90, 99, 99.9)},
                   seconds=time.perf_counter() - started)
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
