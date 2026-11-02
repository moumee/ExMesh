# UE5 플러그인 연동 명세 (Codex 작업용)

이 문서는 UE5 플러그인 저장소에서 Codex에게 그대로 넘기는 작업 명세입니다. 플러그인 저장소의 `AGENTS.md`에 붙여 넣거나, 작업 지시와 함께 첨부하세요.
배경, 실험 결과, 근거는 [ENHANCED_SIMPLIFICATION_GUIDE.md](ENHANCED_SIMPLIFICATION_GUIDE.md)에 있습니다. 기존 Sobel 옵션은 [SOBEL_PLUGIN_MIGRATION.md](SOBEL_PLUGIN_MIGRATION.md)에 있습니다. 이 문서와 두 문서가 다르면 **이 문서와 코드의 argparse가 우선**입니다.

## 1. 작업 목표

플러그인의 LOD 생성 단계를 다음 파이프라인으로 바꿉니다. 각 단계는 Python 스크립트를 **별도 프로세스로 실행**합니다.

```
ExMesh 학습 결과(model_dir)
  ├─ [1] photometric_quadric_weights.py   GPU, 약 2초     → photometric_energy.npy
  ├─ [2] wild_simplify_xatlas_verified.py CPU, 약 3~4분   → lod.obj / lod.mtl / lod_texture.png
  └─ [3] appearance_refine.py (권장)      GPU, 약 30초    → lod_refined.obj / .mtl / _texture.png
UE5 임포트: [3]의 결과. [3]을 끄면 [2]의 결과
```

- 기본 경로는 `--method photometric`(PPQ)입니다. 기존 `wild`와 `sobel`은 그대로 지원하고 옵션으로 남깁니다.
- 생성 모델(TripoSR, TripoSG, Hunyuan3D-2)은 플러그인에 넣지 않습니다. `alt_models/`는 비교 실험 코드이고 배포 대상이 아닙니다. Hunyuan3D-2는 라이선스가 한국을 제외합니다.

## 2. 배포 파일과 환경

- 아래 4개 파일을 **같은 폴더**에 둡니다. `photometric_quadric_weights.py`가 `appearance_refine.py`를 import합니다.
  - `simplification/wild_simplify_xatlas_verified.py`
  - `simplification/obj_geometry_topology.py`
  - `simplification/photometric_quadric_weights.py`
  - `simplification/appearance_refine.py`
- Python 환경: ExMesh의 `exmesh` conda 환경 하나로 세 단계가 모두 돕니다.
  - torch 2.5.1+cu121, nvdiffrast 0.4.0, numpy, scipy, pillow, open3d, xatlas, numba(선택)
- [1]과 [3]은 CUDA GPU가 필요합니다. [2]는 CPU만 씁니다.
- Python 실행 파일 경로는 플러그인 설정으로 받습니다. 시스템 `python`에 의존하지 마세요. PATH의 다른 파이썬이 잡혀 실패한 사례가 있습니다.

## 3. 입력

ExMesh 학습과 export가 끝난 `model_dir`에서 다음 파일을 씁니다.

| 파일 | 만드는 곳 | 용도 |
|---|---|---|
| `<model_dir>/export/mesh_iter_<N>.obj` | `scripts/export_obj.py --iteration N` | 원본 메시 |
| `<model_dir>/export/mesh_iter_<N>.mtl`, `.png` | 같음 | 원본 텍스처(MTL의 `map_Kd`) |
| `<model_dir>/cameras.json` | `train.py`가 학습 시작 시 저장 | 촬영 카메라. [1]에 필요 |

`cameras.json`이 없으면(예: 외부에서 가져온 메시) [1]을 건너뛰고 [2]를 `--method wild`로 실행합니다(7절).

## 4. 명령

모든 경로는 **절대 경로**로 넘깁니다. 인자는 문자열 하나로 합치지 말고 인자 배열로 넘기세요. 경로에 공백이 있을 수 있습니다.

### [1] PPQ 에너지

```
<python> <dir>/photometric_quadric_weights.py
  --mesh    <model_dir>/export/mesh_iter_<N>.obj
  --cameras <model_dir>/cameras.json
  --output  <work>/photometric_energy.npy
```

- 출력: `photometric_energy.npy`(float64, 길이 = 원본 면 수)와 요약 `photometric_energy.json`
- 텍스처는 MTL의 `map_Kd`에서 찾습니다. 다른 텍스처를 쓰려면 `--texture <png>`를 줍니다.
- 나머지 옵션(`--source`, `--sigma`, `--occlusion_tolerance`)은 기본값을 씁니다. UI에 노출하지 마세요.

### [2] 단순화

```
<python> <dir>/wild_simplify_xatlas_verified.py
  --input   <model_dir>/export/mesh_iter_<N>.obj
  --texture <model_dir>/export/mesh_iter_<N>.png
  --output  <work>/lod.obj
  --ratio   <r>
  --method  photometric
  --face_energy <work>/photometric_energy.npy
```

- `--ratio`: 원본 **면** 수 대비 목표 비율, `0 < r ≤ 1`. 목표 면 수는 `max(1, int(원본 면 수 × r))`이고 실제 결과는 조금 다를 수 있습니다. 면 수를 직접 받는 옵션은 없습니다.
- `--method` 값:

| 값 | 추가 인자 | 용도 |
|---|---|---|
| `photometric` | `--face_energy <npy>` 필수 | **기본으로 쓸 값** |
| `wild` | 없음 | 카메라가 없을 때 대체 경로. 생략 시 기본값 |
| `sobel` | `--face_importance <npy>` 필수, `--importance_lambda` 기본 4.0 | 이전 방식. 호환용 |
| `guarded` | 없음 | 실험용. 플러그인에 노출하지 않음 |

- 플러그인에서 그대로 두는 기존 옵션: `--texture_size 1024`, `--atlas xatlas`, `--texture_padding 2`, `--virtual_radius`(생략)
- 쓰지 않는 옵션: `--photometric_lambda`(1.0 고정), `--texture_supersample`, `--visibility`. 평가에서 기본값으로 쓰지 않기로 했습니다.
- 출력: `lod.obj`, `lod.mtl`, `lod_texture.png` + 진단용 `lod.statistics.json`, `lod.input_topology.json`

### [3] 외관 정제 (권장, 끌 수 있게)

```
<python> <dir>/appearance_refine.py
  --reference <model_dir>/export/mesh_iter_<N>.obj
  --input     <work>/lod.obj
  --output    <work>/lod_refined.obj
  --mode      full
```

- `--mode full`: 텍스처와 정점 위치를 함께 정제합니다. `tex`: 텍스처만. UI에는 끔 / `tex` / `full` 세 가지를 두고 기본은 `full`로 합니다.
- 위상, UV, 면 순서는 바뀌지 않습니다. 그래서 UE5 임포트 방식이 [2]와 같습니다.
- 출력: `lod_refined.obj`, `lod_refined.mtl`, `lod_refined_texture.png` + 완료 표시 `lod_refined.complete.json`(**마지막에** 씀)
- 출력 텍스처는 UV 차트 밖이 채워져(edge padding) 있습니다. UE5 mip에서 차트 경계가 검게 번지지 않습니다.

## 5. 반드시 지킬 조건

1. **같은 OBJ**: [1]의 `--mesh`와 [2]의 `--input`은 바이트 단위로 같은 파일이어야 합니다. 에너지는 원본 면 순서에 대응합니다.
   - 면 재정렬, 삭제, 재삼각분할, 좌표 변환(UE 단위·축 변환 포함)은 [1] **전에** 하거나 하지 마세요. UE 좌표 변환은 임포트 단계에서 하세요.
   - 길이나 값이 맞지 않으면 [2]가 `ValueError`로 종료합니다(`Photometric energy must be finite, non-negative, one per original face`).
2. **같은 원본**: [3]의 `--reference`는 [2]의 `--input`과 같은 원본 OBJ입니다.
3. **세 파일은 한 묶음**: `<stem>.obj`, `<stem>.mtl`, `<stem>_texture.png`를 함께 옮기고 이름을 바꾸지 마세요. OBJ의 `mtllib`와 MTL의 `map_Kd`가 파일 이름을 가리킵니다.
4. **캐시 키**: 에너지 `.npy`는 원본 OBJ와 `cameras.json`에서만 정해집니다. 같은 원본으로 여러 `--ratio`를 만들 때는 [1]을 한 번만 실행하고 재사용하세요. 원본 OBJ가 바뀌면 다시 만드세요.
5. **GPU 동시 실행**: [1]과 [3]은 GPU를 씁니다. 여러 LOD를 만들 때 [3]은 한 번에 하나씩 실행하세요. 12 GB GPU에서 동시 실행으로 VRAM이 넘치면 Windows가 공유 메모리로 페이징해 속도가 크게 떨어집니다. [2]는 CPU라 병렬로 돌려도 됩니다.

## 6. 성공 판정

로그 문자열로 판정하지 마세요. 로그 형식은 바뀔 수 있습니다. 아래 조건만 봅니다.

| 단계 | 성공 조건 |
|---|---|
| [1] | 종료 코드 0, `<out>.npy`와 `<out>.json` 존재 |
| [2] | 종료 코드 0, `<stem>.obj`, `<stem>.mtl`, `<stem>_texture.png` 존재 |
| [3] | 종료 코드 0, `<stem>.complete.json` 존재(마지막에 쓰므로 완료 표시로 씀), 세 파일 존재 |

- `<stem>.input_topology.json`은 입력을 읽은 직후 생깁니다. 완료 판정에 쓰지 마세요.
- 실패하면 stderr 끝부분을 사용자에게 보여 주세요. Python 예외 메시지가 원인을 직접 말합니다.
- 타임아웃은 넉넉하게 잡으세요. 기준 측정(RTX 4070, 원본 약 21만 면): [1] 약 2초, [2] 약 3~4분, [3] 약 30초. 첫 실행에는 Numba JIT와 CUDA 초기화 시간이 더해집니다. 원본이 더 크면 [2]가 비례해서 늘어납니다.
- 진행 상황 표시가 필요하면 stdout을 줄 단위로 읽어 그대로 로그 창에 흘리세요. 파싱해서 분기하지는 마세요.

## 7. 대체 경로

| 상황 | 처리 |
|---|---|
| `cameras.json` 없음 | [1] 생략, [2]를 `--method wild`로 실행, [3]은 그대로 가능 |
| CUDA GPU 없음 | [1]과 [3] 생략, [2]를 `--method wild`로 실행 |
| [3]을 끈 상태 | [2]의 텍스처는 차트 둘레 1~2 텍셀만 채워지고 나머지는 검은색입니다. UE5 mip에서 경계가 어두워질 수 있으므로, 임포트 전에 edge padding을 하거나 [3]을 `--mode tex`로라도 실행하세요 |
| [1] 실패 | 사용자에게 알리고 `--method wild`로 진행할지 선택하게 하세요. 조용히 바꾸지 마세요 |

## 8. 의사 코드

```
energy = work / "photometric_energy.npy"
if cameras_json.exists() and gpu_available:
    run([py, D/"photometric_quadric_weights.py",
         "--mesh", src_obj, "--cameras", cameras_json, "--output", energy])
    require(exit == 0 and energy.exists() and energy.with_suffix(".json").exists())
    method_args = ["--method", "photometric", "--face_energy", energy]
else:
    method_args = ["--method", "wild"]

for ratio in lod_ratios:                       # 예: [0.1, 0.05, 0.025]
    lod = work / f"lod_{ratio}.obj"
    run([py, D/"wild_simplify_xatlas_verified.py",
         "--input", src_obj, "--texture", src_png, "--output", lod,
         "--ratio", str(ratio), *method_args])
    require(exit == 0 and triple_exists(lod))   # .obj, .mtl, _texture.png

    final = lod
    if refine_mode != "off" and gpu_available:  # 한 번에 하나씩
        final = work / f"lod_{ratio}_refined.obj"
        run([py, D/"appearance_refine.py",
             "--reference", src_obj, "--input", lod, "--output", final,
             "--mode", refine_mode])
        require(exit == 0 and final.with_suffix(".complete.json").exists()
                and triple_exists(final))
    import_to_ue(final)
```

## 9. 완료 확인 체크리스트

- [ ] 샘플 `model_dir` 하나로 [1]→[2]→[3]이 플러그인 안에서 끝나고 UE5에 임포트된다.
- [ ] `cameras.json`을 치운 상태에서 `wild` 경로로 끝까지 돈다.
- [ ] 경로에 공백과 한글이 있는 폴더에서도 돈다.
- [ ] 단계 하나를 일부러 실패시키면(예: 잘못된 `--face_energy`) 플러그인이 멈추지 않고 오류 메시지를 보여 준다.
- [ ] 정제 결과를 UE5에서 멀리서 봤을 때 UV 경계에 검은 선이 없다.
- [ ] 같은 원본으로 LOD를 여러 개 만들 때 [1]이 한 번만 실행된다.
