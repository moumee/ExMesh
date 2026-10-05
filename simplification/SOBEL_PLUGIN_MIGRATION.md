# modified → verified: Unreal 플러그인 연동 안내

현재 실행 파일은 `wild_simplify_xatlas_verified.py`다. 가장 중요한 변경은 Sobel 실행 시 `--method sobel`을 반드시 지정하는 것이다. 기본 method는 `wild`이며, 이 모드에서는 `--face_importance`와 `--importance_lambda`를 전달해도 Sobel 가중치를 사용하지 않는다.

## 명령줄 옵션

| 옵션 | modified | verified / 연동 시 주의 |
|---|---|---|
| `--method` | 없음 | 추가됨. `wild` 또는 `sobel`, 기본 `wild`. Sobel은 반드시 명시 |
| `--face_importance` | 항상 필수 | Sobel 모드에서만 필수. 원본 face당 점수 하나인 `.npy` |
| `--importance_lambda` | 기본 1.0 | 기본 4.0으로 변경. 생략하면 현재 설정 λ4 적용 |
| `--input` | 필수 | 동일 |
| `--output` | 생략 시 입력 이름에 `_simplified` | 동일. 플러그인은 출력 경로를 명시 권장 |
| `--ratio` | 기본 0.5 | 동일. 정점 비율이 아니라 원본 face 수에 대한 비율 |
| `--virtual_radius` | 생략 시 bbox 대각선 × 0.0025 | 동일. 명시한 값은 입력 메시 좌표 단위. `0`은 virtual edges 비활성화 |
| `--texture` | 텍스처 경로, 생략 시 OBJ/MTL 또는 같은 이름 PNG 탐색 | 동일 |
| `--texture_size` | 기본 1024 | 동일 |
| `--atlas` | `xatlas` 또는 `grid`, 기본 `xatlas` | 동일 |
| `--texture_padding` | 기본 2 | 동일 |

`--ratio`는 `0 < ratio ≤ 1`. 목표 face 수는 `max(1, int(원본 face 수 × ratio))`다. 정수 face 수를 직접 받는 `--target_faces` 옵션은 없으며, 실제 최종 face 수는 목표와 조금 다를 수 있다. λ의 기본값은 4이므로 생략할 수 있고, 명시한 값이 있으면 그 값을 사용한다. `--method sobel`과 점수 파일은 여전히 명시해야 한다.

## Sobel 실행 예시

```powershell
python simplification/wild_simplify_xatlas_verified.py --input "original.obj" --texture "original.png" --output "result.obj" --ratio 0.1 --method sobel --face_importance "importance.npy" --importance_lambda 4 --atlas xatlas --texture_size 1024 --texture_padding 2
```

Wild 대조군은 `--method wild`를 사용하고 `--face_importance`, `--importance_lambda`를 생략한다. 상대 경로는 프로세스의 작업 폴더를 기준으로 해석되므로, UE 플러그인에서는 Python 실행 파일 및 입력·출력 경로를 절대 경로로 전달하고 공백이 있는 경로를 별도 인자 또는 적절한 따옴표로 처리한다.

## 배포 파일과 내부 변경

- `wild_simplify_xatlas_verified.py`와 `obj_geometry_topology.py`를 같은 폴더에 배포한다. QEM solver는 verified 내부의 `minimize_quadric()`로 통합됐으므로 별도 `wild_quadric_solver.py`는 필요 없다.
- NumPy, SciPy, Pillow, Open3D 의존성은 기존과 같다. `--atlas xatlas`에는 xatlas가 필요하다. Numba는 선택 가속이며 첫 실행 JIT 시간이 포함될 수 있다.
- 삼각형 OBJ를 읽을 때 원래 `v/f` 정점 인덱스의 연결 관계를 복원한다. 같은 좌표라도 원본 정점 ID가 다른 표면은 합치지 않는다. 일반 3D `v x y z`와 삼각형 `f`를 지원하며, 비삼각형 또는 추가 성분이 있는 OBJ 정점 레코드는 복원 단계에서 거부한다. 다른 확장자는 이 복원을 생략한다.
- face 수와 corner 좌표·순서가 로더 결과와 일치하는지 검증한다. UV와 중요도는 face 순서에 대응하므로, 점수를 만든 뒤 face 재정렬·삭제·재삼각분할을 하면 안 된다.
- 특이하거나 거의 특이한 QEM 행렬의 위치 계산 및 edge 순회 순서가 바뀌었다. 같은 옵션과 데이터라도 modified와 결과 메시·UV·텍스처가 완전히 같을 것을 기대하면 안 된다. 새 구현 기준으로 캐시 키/버전을 구분한다.
- Python 함수를 직접 호출하는 경우 `simplify(vertices, faces, target_faces, virtual_radius, track_history, face_weights)`와 4개 반환값은 유지됐다. 다만 OBJ 연결 구조 복원은 CLI의 `main()`에서 수행하므로, `simplify()`를 직접 호출하면 입력 단계에서 `restore_obj_position_topology()`도 적용해야 한다. `face_weights = 1 + 4 × importance`는 호출자가 구성한다.

## 점수와 출력 파일

verified는 사진에서 Sobel 점수를 생성하지 않고 이미 계산된 `.npy`를 입력으로 사용한다. Gaussian σ/커널·Sobel 커널은 verified의 argparse 옵션이 아니다. 현재 선택은 σ1·Gaussian13×13·Sobel3×3이며 점수 생성 단계에서 적용한다. 점수는 원본 face 순서로 길이 `(원본 face 수,)`, 모든 값이 유한하고 `[0, 1]` 범위여야 한다. 새로운 메시에는 새 점수가 필요하다.

`moumee/moumee_img_utils.py`도 업데이트해서 전달한다. 기존 `compute_sobel_magnitude(img)` 호출은 그대로 가능하며 기본값은 현재 선택한 σ1·Gaussian13×13·Sobel3×3이다. 설정을 명시하는 호출은 다음과 같다. 이 점수 생성 단계에는 OpenCV(`cv2`)도 필요하다.

```python
compute_sobel_magnitude(img, sigma=1.0, gaussian_kernel=13, sobel_kernel=3)
```

ExMesh의 기존 `train.py` 점수 생성 경로에 통합했다. renderer의 `face_id`와 `compute_sobel_magnitude(image_np)` 결과를 기존 `moumee_img_utils.py`의 `compute_face_importance()`에 전달한다. 이 함수는 기존 P95 정규화 → 시점별 face 평균 → 가시 시점 평균을 수행한다. `<model_path>/face_importance.npy` 저장 경로와 float32 출력 형식은 유지됐다. 마지막 iteration의 topology 변경까지 반영한 체크포인트를 먼저 저장하므로 `scripts/export_obj.py`에서 같은 최종 iteration을 export해서 사용한다. 친구는 변경된 train.py와 moumee_img_utils.py를 함께 업데이트한다. 함께 사용하는 renderer의 face_id 반환 구현도 유지한다. 별도 점수 생성 모듈은 필요 없다.

기존 점수 파일을 입력으로 실행만 할 때는 moumee와 점수 생성 코드는 단순화 런타임에 필요 없다. 기본값 변경은 이미 저장한 `.npy`를 수정하지 않으므로 새 설정의 점수는 다시 생성해야 한다. 점수의 face 순서는 함께 export한 동일 최종 메시와 일치해야 한다.

기존 `run_sobel_ablation.py`의 자동 실행은 modified와 과거 13조건을 사용한다. 이 파일은 과거 필터 비교의 재현용이고 배포에는 필요 없다. 점수 집계는 별도 구현을 제거하고 원본 moumee_img_utils의 normalize_magnitude/aggregate_face_importance를 공유한다. ablation의 Open3D CPU raycast face ID와 train의 nvdiffrast face ID는 서로 다른 가시성 구현이므로, 두 경로의 점수 자체가 비트 단위로 같다는 보장은 없다. 같은 magnitude/face ID 입력에 대한 집계는 기존 train과 정확히 일치하는 것을 검증했다.

텍스처가 있는 결과의 출력 형식은 기존과 같은 `<stem>.obj`, `<stem>.mtl`, `<stem>_texture.png`다. 지정한 출력 확장자가 OBJ가 아니어도 텍스처 처리 시 OBJ로 바뀐다. 원본 UV가 없으면 텍스처 경로만으로 UV를 생성·전달하는 경로에 진입하지 않는다. 세 파일은 함께 유지해야 한다.

추가로 `<stem>.input_topology.json`이 입력 topology 진단용으로 생성된다. OBJ/MTL/PNG 가져오기 대상에 이 JSON을 포함할 필요는 없다. 이 JSON은 입력을 읽은 직후 생성되므로, 존재만으로 작업 완료를 판정하면 안 된다. subprocess 종료 코드와 최종 메시·필요한 텍스처 출력을 확인한다. 시작 로그 문자열과 OBJ 연결 구조 로그가 추가·변경됐으므로 특정 로그 한 줄만으로 성공을 판정하지 않는다.

`verified`는 검토·수정한 Python/xatlas 구현이라는 파일명이며 저자 reference 구현과 완전히 동일함을 보장하는 이름은 아니다. 이전 결과와의 재현·비교에는 modified를 보존하고 사용한 코드와 옵션을 기록한다.

발표자료와 로컬 생성 작업의 Git 제외 범위는 저장소 루트의 `/deliverables/`, `/.presentation_build/`, `/.audit_build/` 및 `*.pptx`다. 단순화·점수 생성에 필요한 실제 코드는 제외하지 않는다.
