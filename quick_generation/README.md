# 빠른 생성 모드 (TripoSR)

사진 1장으로 텍스처 메시를 만드는 플러그인 부가 모드입니다. 정확한 복원은 ExMesh를 쓰고, 이 모드는 미리보기나 사진이 부족할 때 씁니다.

## 설치 (Windows, 별도 conda 환경)

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

## 실행

```bat
set HF_HOME=<가중치 캐시>
set U2NET_HOME=<배경 제거 모델 폴더>
<env>\python.exe triposr_generate.py --repo <TripoSR> --image photo.jpg --output out\mesh.obj
```

- 출력: `mesh.obj`, `mesh.mtl`, `mesh_texture.png`, `mesh_input.png`, `mesh.complete.json`(마지막에 기록)
- 축: 기본은 Z가 위, 사진 쪽이 +X입니다. `--up y`면 Y가 위, 사진 쪽이 +Z입니다.
- 크기: `--height 100`이면 높이가 100(UE 기준 1 m)이 되도록 맞춥니다.
- LOD: `wild_simplify_xatlas_verified.py --method wild --texture out\mesh_texture.png`, 이어서 `appearance_refine.py`를 그대로 쓸 수 있습니다.
- 검증(RTX 4070, DTU scan83 사진 1장): 생성 약 40초·VRAM 2.4 GB, 9.4만 면 → 10% 단순화 약 80초, 정제 약 30초
