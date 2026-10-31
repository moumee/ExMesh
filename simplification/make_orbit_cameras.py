"""Fixed full-sphere evaluation cameras for a mesh (360-degree protocol).

Writes cameras in the cameras.json format used by evaluate_mesh_preservation.py
(position, camera-to-world rotation with x right / y down / z forward, fx, fy,
width, height, img_name). Directions are uniform on the sphere from a fixed
seed that is disjoint from appearance_refine.py's training/validation seeds,
looking at the bounding-box center from 0.6-1.0x the distance that fits the
bounding sphere in a 40-degree field of view.

  python make_orbit_cameras.py --mesh original.obj --output orbit_cameras.json
"""
import argparse
import json
import math

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    import open3d as o3d
    vertices = np.asarray(o3d.io.read_triangle_mesh(args.mesh, enable_post_processing=False).vertices)
    low, high = vertices.min(0), vertices.max(0)
    center, radius = (low + high) / 2, np.linalg.norm(high - low) / 2
    fov = math.radians(40)
    focal = args.size / 2 / math.tan(fov / 2)
    rng = np.random.default_rng(args.seed)
    cameras = []
    for index in range(args.count):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        position = center + direction * radius / math.sin(fov / 2) * rng.uniform(0.6, 1.0)
        forward = (center - position) / np.linalg.norm(center - position)
        helper = rng.normal(size=3)
        right = np.cross(forward, helper)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        cameras.append(dict(id=index, img_name=f"orbit_{index:03d}", width=args.size, height=args.size,
                            position=position.tolist(), rotation=np.column_stack((right, down, forward)).tolist(),
                            fx=focal, fy=focal))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)
    print(f"[orbit] {len(cameras)} cameras -> {args.output}")


if __name__ == "__main__":
    main()
