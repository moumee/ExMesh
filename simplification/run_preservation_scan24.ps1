# Run from the repository root. Open3D, numpy, Pillow, torch, torchvision,
# scipy and the local LPIPS implementation are required. nvdiffrast is not.
$ErrorActionPreference = 'Stop'
$meshRoot = 'outputs/DTU/scan24/export/simp_mod'
python simplification/evaluate_render_quality.py `
  --protocol novel `
  --reference "$meshRoot/mesh_iter_10000.obj" `
  --candidate "lambda0=$meshRoot/mesh_iter_10000_simplified_0_1.obj" `
  --candidate "lambda1=$meshRoot/mesh_iter_10000_mod_1_simplified_0_1.obj" `
  --candidate "lambda4=$meshRoot/mesh_iter_10000_mod_4_simplified_0_1.obj" `
  --candidate "lambda8=$meshRoot/mesh_iter_10000_mod_8_simplified_0_1.obj" `
  --texture "reference=$meshRoot/mesh_iter_10000.png" `
  --texture "lambda0=$meshRoot/mesh_iter_10000_simplified_0_1_texture.png" `
  --texture "lambda1=$meshRoot/mesh_iter_10000_mod_1_simplified_0_1_texture.png" `
  --texture "lambda4=$meshRoot/mesh_iter_10000_mod_4_simplified_0_1_texture.png" `
  --texture "lambda8=$meshRoot/mesh_iter_10000_mod_8_simplified_0_1_texture.png" `
  --cameras_json outputs/DTU/scan24/cameras.json `
  --output_dir outputs/DTU/scan24/preservation_eval `
  --face_budget_tolerance 1 --samples 100000 --seeds 0 1 2 `
  --orbit_degrees 5 --resolution 2 --save_renders
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE" }
