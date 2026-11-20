# 빠른 생성 모드 (Pixal3D / TripoSR)

사진 1장으로 텍스처 메시를 만드는 플러그인 부가 모드입니다. 정확한 복원은 ExMesh를 쓰고, 이 모드는 미리보기나 사진이 부족할 때 씁니다. 플러그인 퀵 모드는 ComfyUI가 설치돼 있으면 Pixal3D를, 없으면 TripoSR를 씁니다.

두 스크립트는 인자와 출력이 같습니다.

- 출력: `mesh.obj`, `mesh.mtl`, `mesh_texture.png`, `mesh_input.png`, `mesh.complete.json`(마지막에 기록)
- 축: 기본은 Z가 위, 사진 쪽이 +X입니다. `--up y`면 Y가 위, 사진 쪽이 +Z입니다.
- 크기: `--height 100`이면 높이가 100(UE 기준 1 m)이 되도록 맞춥니다.
- LOD: `wild_simplify_xatlas_verified.py --method wild --texture out\mesh_texture.png`, 이어서 `appearance_refine.py`를 그대로 쓸 수 있습니다.

## Pixal3D (기본, `comfyui_generate.py`)

ComfyUI를 백그라운드 서버로 띄워 "Pixal3D & TRELLIS.2: Image to Model" 워크플로를 실행하고, 결과 GLB를 OBJ로 바꿉니다.

- 워크플로 파일: `comfyui_pixal3d_trellis2.json`
  - 출처: [Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates)의 `3d_pixal3d_trellis2_image_to_model.json`, MIT
- 노드: ComfyUI 기본 노드를 씁니다. ComfyUI 0.34 이상이 필요하고, v0.46.0에서 검증했습니다.
- 가중치: Comfy-Org의 int8 가중치를 씁니다. Pixal3D와 TRELLIS.2 모두 MIT입니다.

### 설치

`Plugins/ReconstructionBridge/Setup/setup_pixal3d.bat`가 다음을 설치합니다.
- ComfyUI v0.46.0: `ThirdParty/ComfyUI`
- conda 환경 `comfyui`: Python 3.13, PyTorch cu130
- Pixal3D 가중치 약 10 GB
- `setup_pixal3d.bat trellis2`로 실행하면 `--model trellis2`용 TRELLIS.2 가중치 5.3 GB도 받습니다.

수동으로 설치하려면 이렇게 합니다.

```bat
git clone --branch v0.46.0 https://github.com/Comfy-Org/ComfyUI
conda create -n comfyui python=3.13 -y
set PYTHONNOUSERSITE=1
<env>\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
<env>\python.exe -m pip install -r ComfyUI\requirements.txt trimesh
```

가중치는 Hugging Face 저장소와 같은 경로로 `ComfyUI\models\` 아래에 받습니다.
- `Comfy-Org/Pixal3D`
  - `diffusion_models/pixal3d_int8_convrot.safetensors`
  - `vae/trellis_2_shape_vae_bf16.safetensors`
  - `vae/trellis_2_texture_vae_bf16.safetensors`
  - `clip_vision/dino_v3_L_naf_fp32.safetensors`
- `Comfy-Org/BiRefNet`: `background_removal/birefnet.safetensors`
- `Comfy-Org/MoGe`: `geometry_estimation/moge_2_vitl_normal_fp16.safetensors`
- `--model trellis2`를 쓸 때만: `Comfy-Org/TRELLIS.2`의 `diffusion_models/trellis_2_int8_convrot.safetensors`
  - Pixal3D로 생성할 때는 스크립트가 TRELLIS.2 로더를 워크플로에서 빼므로 이 파일이 없어도 됩니다.

### 실행

```bat
<env>\python.exe comfyui_generate.py --repo <ComfyUI> --image photo.png --output out\mesh.obj [--height 100]
```

- `--model`: `pixal3d`(기본) 또는 `trellis2`
- `--resolution`: `1024`(기본) 또는 `1536`. Pixal3D는 두 해상도의 형상 정확도가 같았고, 1024가 더 빠릅니다.
- `--vram_budget`: ComfyUI가 쓸 VRAM 목표(GB)입니다.
  - 기본 4이고, 이때 실제 최대 사용량은 약 4.4 GB입니다. 넘치는 가중치는 RAM에서 올려 쓰고, 속도와 결과는 같습니다.
  - `0`이면 남는 VRAM을 다 씁니다(약 10 GB).
  - 4보다 낮춰도 해상도 1024의 VAE 디코딩 때문에 약 4.5 GB가 바닥입니다.
- 배경: 입력이 RGBA면 알파를 마스크로 쓰고, 아니면 BiRefNet으로 배경을 지웁니다.
- 기울기 보정: Pixal3D는 사진 카메라 기준으로 메시를 만들어서, 위에서 찍은 사진이면 그만큼 기울어집니다.
  - 가장 가까운 안정 자세로 바닥에 세웁니다(`--max_tilt 45`, `0`이면 끔).
  - 곰 사진에서는 기울기가 31°에서 7.5°로 줄었습니다.
- 검증(RTX 4070, 곰 사진 1장)
  - 생성 50초, 서버 시작과 변환을 포함하면 약 65초입니다. 결과는 20만 면입니다.
  - 언리얼 에디터를 켠 상태에서도 시간이 같았습니다.
  - 실루엣 IoU: TripoSR 0.795, Pixal3D 0.858, 120장 visual hull 0.837
- 플러그인 퀵 모드는 `--faces 100000`으로 부릅니다. TripoSR와 비슷한 면 수라 단순화와 정제가 오래 걸리지 않습니다.
  - 생성 + 10% 단순화 + 정제까지 약 4.5분 걸렸습니다(언리얼 에디터 켠 상태).

주의할 점이 두 가지 있습니다.
- **공식 워크플로의 "배경 제거 끄기" 경로:** `LoadImage`의 MASK(1 − 알파)를 그대로 써서 물체가 지워집니다. 스크립트가 `InvertMask` 노드를 끼워 고칩니다.
- **사용자 site-packages 공유:** 같은 버전의 다른 Python이 사용자 site-packages를 공유하면 패키지가 섞입니다. 스크립트는 ComfyUI를 `PYTHONNOUSERSITE=1`로 실행합니다. pip도 같은 설정으로 실행하세요.

## TripoSR (`triposr_generate.py`)

### 설치 (Windows, 별도 conda 환경)

```bat
conda create -p <envs>\triposr python=3.10 -y
<env>\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/VAST-AI-Research/TripoSR   :: 검증 커밋 107cefd
<env>\python.exe -m pip install omegaconf==2.3.0 Pillow==10.1.0 einops==0.7.0 transformers==4.35.0 trimesh==4.0.5 rembg huggingface-hub xatlas==0.0.9 moderngl==5.10.0 PyMCubes onnxruntime scipy
cd TripoSR && git apply <ExMesh>\quick_generation\triposr.diff
```

`triposr.diff`는 두 가지를 고칩니다.
- torchmcubes(Windows에서 CUDA 빌드 필요)가 없으면 PyMCubes로 대체
- `bake_texture`의 CPU/GPU 텐서 불일치 버그

### 실행

```bat
set HF_HOME=<가중치 캐시>
set U2NET_HOME=<배경 제거 모델 폴더>
<env>\python.exe triposr_generate.py --repo <TripoSR> --image photo.jpg --output out\mesh.obj
```

- 검증(RTX 4070, DTU scan83 사진 1장): 생성 약 40초·VRAM 2.4 GB, 9.4만 면 → 10% 단순화 약 80초, 정제 약 30초
