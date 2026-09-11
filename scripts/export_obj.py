from scene.mesh_model import MeshModel
import os
import sys
import argparse
import numpy as np
from PIL import Image


# ---------------------------------------------------------
# Add ExMesh root directory to Python module search path
# ---------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


def export_obj(mesh, export_dir, name="mesh"):
    """
    Export ExMesh mesh as:
        OBJ + MTL + PNG

    ExMesh/PyTorch texture coordinates use a different V-axis convention
    from OBJ, so V is flipped during export.
    """

    os.makedirs(export_dir, exist_ok=True)

    obj_path = os.path.join(export_dir, f"{name}.obj")
    mtl_path = os.path.join(export_dir, f"{name}.mtl")
    texture_path = os.path.join(export_dir, f"{name}_texture.png")

    # -----------------------------------------------------
    # Mesh data
    # -----------------------------------------------------
    vertices = mesh.get_vertices.detach().cpu().numpy()
    faces = mesh.get_faces.detach().cpu().numpy()
    uvs = mesh.get_uvs.detach().cpu().numpy()
    uv_indices = mesh.get_uv_indices.detach().cpu().numpy()

    print(f"Vertices:   {vertices.shape}")
    print(f"Faces:      {faces.shape}")
    print(f"UVs:        {uvs.shape}")
    print(f"UV indices: {uv_indices.shape}")

    if faces.shape != uv_indices.shape:
        raise ValueError(
            f"faces and uv_indices shape mismatch: "
            f"{faces.shape} vs {uv_indices.shape}"
        )

    # -----------------------------------------------------
    # Save learned texture
    #
    # ExMesh:
    #     (3, H, W)
    #
    # PNG/PIL:
    #     (H, W, 3)
    # -----------------------------------------------------
    texture = (
        mesh.get_texture
        .detach()
        .cpu()
        .clamp(0.0, 1.0)
        .numpy()
    )

    if texture.ndim != 3:
        raise ValueError(
            f"Unexpected texture shape: {texture.shape}"
        )

    if texture.shape[0] == 3:
        texture = np.transpose(texture, (1, 2, 0))

    texture = (texture * 255.0).round().astype(np.uint8)

    Image.fromarray(texture).save(texture_path)

    # -----------------------------------------------------
    # MTL
    # -----------------------------------------------------
    material_name = f"{name}_material"

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")
        f.write(f"map_Kd {name}_texture.png\n")

    # -----------------------------------------------------
    # OBJ
    # -----------------------------------------------------
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {name}.mtl\n")
        f.write(f"o {name}\n")

        # Geometry vertices
        for x, y, z in vertices:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")

        # UV coordinates
        #
        # IMPORTANT:
        # ExMesh/PyTorch image-space V convention
        # is opposite to OBJ's common convention.
        #
        # Therefore:
        #     v_obj = 1 - v_exmesh
        # -------------------------------------------------
        for u, v in uvs:
            v = 1.0 - v
            f.write(f"vt {u:.9f} {v:.9f}\n")

        f.write(f"usemtl {material_name}\n")

        # Faces
        #
        # OBJ indices are 1-based.
        #
        # Geometry vertex indices and UV indices must remain
        # separate because ExMesh can have UV seams.
        for face, uv_face in zip(faces, uv_indices):
            v0 = int(face[0]) + 1
            v1 = int(face[1]) + 1
            v2 = int(face[2]) + 1

            t0 = int(uv_face[0]) + 1
            t1 = int(uv_face[1]) + 1
            t2 = int(uv_face[2]) + 1

            f.write(
                f"f "
                f"{v0}/{t0} "
                f"{v1}/{t1} "
                f"{v2}/{t2}\n"
            )

    print()
    print("Export complete.")
    print(f"OBJ:     {obj_path}")
    print(f"MTL:     {mtl_path}")
    print(f"Texture: {texture_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained ExMesh model as OBJ + MTL + PNG."
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Example: outputs/DTU/scan24"
    )

    parser.add_argument(
        "--iteration",
        type=int,
        default=10000,
        help="Checkpoint iteration to export"
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Resolve model path
    # -----------------------------------------------------
    if os.path.isabs(args.model_path):
        model_path = args.model_path
    else:
        model_path = os.path.join(ROOT_DIR, args.model_path)

    model_path = os.path.normpath(model_path)

    checkpoint_dir = os.path.join(
        model_path,
        "point_cloud",
        f"iteration_{args.iteration}"
    )

    checkpoint_file = os.path.join(
        checkpoint_dir,
        "model_state_dict.pt"
    )

    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_file}"
        )

    export_dir = os.path.join(
        model_path,
        "export"
    )

    print(f"Loading checkpoint:")
    print(checkpoint_dir)
    print()

    mesh = MeshModel()
    mesh.load(checkpoint_dir)

    export_obj(
        mesh,
        export_dir,
        name=f"mesh_iter_{args.iteration}"
    )


if __name__ == "__main__":
    main()
