# 대체 복원 모델(Tripo/Hunyuan) 비교 실험: 설치·실행·평가

ExMesh 대신 단일 이미지 기반 3D 생성 모델을 쓰면 어떤지 비교한 실험의 재현 안내입니다. 결과와 결론은
[`simplification/ENHANCED_SIMPLIFICATION_GUIDE.md`](../simplification/ENHANCED_SIMPLIFICATION_GUIDE.md) 5장에 정리했습니다.
이 폴더의 코드는 **비교 실험용**이며 UE5 플러그인 배포 대상이 아닙니다.

- **Meshy**: 오픈소스가 아닌 유료 API라 실행하지 않았습니다. 외부 서버로 이미지를 보내야 하므로 연구용 비교에도 쓰지 않았습니다.
- **Stable Fast 3D**: HF 접근 승인(gated)이 필요해 제외했습니다.
- **TRELLIS**: Linux 전용 의존성(spconv, kaolin, flash-attn 등) 때문에 Windows에서는 시도하지 않았습니다.

## 1. 공통 준비

- GPU: RTX 4070 12 GB, CUDA Toolkit 12.1, VS 2022 Community(MSVC 14.36 툴셋)
- 저장소·가상환경·가중치는 ExMesh 저장소 밖 `D:\gamecontents_capstone\third_party\`에 둡니다.
  - Hugging Face 가중치는 약 25 GB입니다(TripoSR, Hunyuan3D-2, Hunyuan3D-2mini).
  - 환경 변수: `HF_HOME=D:\gamecontents_capstone\third_party\hf_cache`, `PIP_CACHE_DIR=...\pip_cache`
- GPU 작업이 겹치지 않도록 모든 GPU 명령은 `third_party\gpu_run.py --label 이름 -- <명령>`으로 감쌌습니다. 파일 잠금 방식입니다.
- 입력 이미지: `python alt_models/prepare_inputs.py --item scan83:51 --item scan24:23`
  - 마스크 bbox로 정사각형 크롭을 만들고 1024²로 리사이즈합니다.
  - `_rgb.png`는 배경을 포함한 원본이고, 각 모델의 기본 배경 제거 입력으로 씁니다. `.png`는 RGBA(DTU 마스크 알파)입니다.
  - 주의: DTU `images/*.png`는 1600×1200 원본을 (46, 38) 픽셀 오프셋으로 잘라 주점을 중앙에 맞춘 이미지입니다. 따라서 `mask/*.png`(원본 크기)를 **리사이즈하면 어긋납니다**. 이미지 알파 채널을 쓰세요.
  - 실험에 사용한 TripoSR/Hunyuan 입력(`outputs/alt_models/inputs/`)은 이 수정 전에 만든 것입니다. 모델이 자체 배경 제거를 하므로 결과에는 영향이 없습니다.

## 2. 모델별 설치

각 모델마다 별도 conda 환경을 씁니다(Python 3.10, PyTorch 2.5.1+cu121).

```bat
set PIP_CACHE_DIR=D:\gamecontents_capstone\third_party\pip_cache
conda create -p D:\gamecontents_capstone\third_party\envs\<model> python=3.10 -y
<env>\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### TripoSR (MIT)
```bat
git clone https://github.com/VAST-AI-Research/TripoSR   :: 사용 커밋 107cefd
pip install omegaconf==2.3.0 Pillow==10.1.0 einops==0.7.0 transformers==4.35.0 trimesh==4.0.5 rembg huggingface-hub "imageio[ffmpeg]" gradio xatlas==0.0.9 moderngl==5.10.0 PyMCubes onnxruntime
cd TripoSR && git apply ..\..\ExMesh\quick_generation\triposr.diff
```
패치는 두 가지입니다(`quick_generation/triposr.diff`).
1. torchmcubes는 Windows에서 CUDA 빌드가 필요하므로 PyMCubes로 대체했습니다. 축 순서도 맞췄습니다.
2. `bake_texture`의 원본 버그를 고쳤습니다. 위치 텐서가 CPU에 남아 있어 device 불일치가 나던 문제입니다.

### Hunyuan3D-2 (Tencent Hunyuan 3D 2.0 Community License — **EU·영국·한국 제외**)
```bat
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2   :: 사용 커밋 f8db630
pip install "diffusers==0.32.2" "transformers==4.48.3" "huggingface_hub==0.27.1" "accelerate==1.3.0" einops opencv-python omegaconf tqdm trimesh pymeshlab pygltflib xatlas rembg onnxruntime ninja pybind11
:: custom_rasterizer CUDA 확장 빌드 (x64 Native Tools / vcvars64 -vcvars_ver=14.36, DISTUTILS_USE_SDK=1, TORCH_CUDA_ARCH_LIST=8.9)
cd Hunyuan3D-2\hy3dgen\texgen\custom_rasterizer && <env>\python.exe setup.py install
```
- `pip install -e .`는 실행하지 않았습니다. 러너가 저장소를 `sys.path`로 불러옵니다.
- `differentiable_renderer`는 순수 Python 대체 구현을 그대로 사용했습니다.
- diffusers/transformers는 torch 2.5.1과 호환되는 버전으로 고정했습니다. 최신 diffusers는 `torch.accelerator`가 필요합니다.
- **라이선스**: Section 1(l)·5(c)에 따라 대한민국에서 모델이나 그 출력을 사용하는 것은 라이선스 범위 밖입니다. 실제 플러그인에는 넣지 마세요.

## 3. 실행

```bat
:: 출력: outputs/alt_models/<scene>/<model>/{mesh.obj, *.png, run_info.json(시간·VRAM·버전)}
python alt_models/runners/run_triposr.py   --repo <TripoSR>     --image outputs/alt_models/inputs/scan83_view051_rgb.png --out outputs/alt_models/scan83/triposr
python alt_models/runners/run_hunyuan3d2.py --repo <Hunyuan3D-2> --image ...  --out ...   :: 기본: dit-v2-0, 50 step
python alt_models/runners/run_hunyuan3d2.py ... --model_path tencent/Hunyuan3D-2mini --subfolder hunyuan3d-dit-v2-mini-turbo --steps 5 --num_chunks 8000 --flashvdm   :: 경량(4 GB)
```
- 각 모델의 공식 기본값을 따랐습니다. 다른 점은 `run_info.json`의 `deviations`에 기록되어 있습니다.

## 4. 평가 (`evaluate_alt_models.py`, ExMesh 환경)

```bat
python alt_models/evaluate_alt_models.py --scene scan83 --model triposr --stage align    :: GT 기반 유사변환 정렬
python alt_models/evaluate_alt_models.py --scene scan83 --model triposr --stage chamfer  :: DTU 공식 프로토콜(mm)
python alt_models/evaluate_alt_models.py --scene scan83 --model triposr --stage photo    :: 실제 사진과 비교
python alt_models/summarize_alt_models.py      :: deliverables/enhanced_simplification/reconstruction_sources.csv
python alt_models/make_comparison_figure.py --scene scan83 --views 0010 0040 --output fig.png
```
- **정렬(align)**:
  - 생성 모델 출력은 크기·자세가 임의라서 DTU GT 점군(마스크로 물체만 남김)에 맞춥니다.
  - 무작위 회전 3000개를 시도한 뒤, 상위 24개에 대해 trimmed scaled ICP로 유사변환(스케일·회전·이동)을 구합니다.
  - GT를 사용하는 **오라클 정렬이므로 생성 모델에 유리한 조건**입니다.
- **chamfer**: 저장소의 DTU 평가(마스크 컬링 + `eval.py`)를 그대로 씁니다. ExMesh와 같은 정규화 좌표계에서 수행합니다.
- **photo**:
  - 모든 촬영 카메라에서 렌더링하고, 이미지 알파(물체 마스크) 영역의 PSNR과 마스크 bbox 크롭의 SSIM·LPIPS를 구합니다.
  - 실루엣 IoU도 함께 계산합니다. 해상도는 절반입니다.
  - 생성 모델은 입력으로 쓴 시점을 빼고 평균합니다.
  - ExMesh는 이 사진들로 학습했으므로 일반화 성능이 아니라 **실제 물체에 대한 충실도** 비교입니다.
- ExMesh scan83 전체 파이프라인: `python alt_models/run_exmesh_scene.py --scan 83`
  - PGSR → ExMesh → OBJ export → DTU 평가 순서로 진행합니다. `pipeline_timing.json`에 단계별 시간이 남습니다.
