# Sobel 중요도 기반 단순화 평가

## 실행

저장된 scan24 결과 네 개를 평가하려면 저장소 루트에서 실행한다.

```powershell
./simplification/run_preservation_scan24.ps1
python -m unittest discover -s simplification -p test_mesh_preservation.py
```

기존 `evaluate_render_quality.py`에 `--protocol geometry` 또는 `--protocol novel`을
전달하면 새로운 평가로 진입한다. `--candidate LABEL=OBJ`를 반복하여 여러 결과를
평가한다. 텍스처는 `--texture LABEL=PNG`로 지정한다. `reference`는 예약된 label이다.
geometry 모드에는 CUDA, 카메라, 텍스처가 필요 없다. novel 모드는 Open3D CPU raycast와
PyTorch 이미지 지표를 사용하며 nvdiffrast나 ExMesh 학습 모듈을 불러오지 않는다.
Open3D, numpy, Pillow, torch, torchvision, scipy가 필요하다. 최초 LPIPS 실행에는
사전학습 가중치 다운로드가 필요할 수 있다.

## 평가 대상과 독립성

모든 지표의 기준은 단순화 이전 원본 메시이다. 실제 사진에 대한 재구성 정확도와는
구분한다. 입력 사진이나 Sobel 점수를 평가에 사용하지 않는다. 원래 카메라의 로컬
up 축을 기준으로 원본 bounding box 중심 주변에서 위치와 회전을 ±5도 함께 회전시켜
49개 카메라에서 98개 새 pose를 생성한다. 원래 pose와 일치하면 실행을 중단한다.
생성 카메라는 JSON으로 저장한다. 기존 카메라 주변의 제한적인 시점 변화이며,
완전히 다른 장면이나 held-out 실제 사진에 대한 일반화를 증명하지 않는다.

## 기하 지표

각 방향에 면적 비례 표면 샘플 100,000개를 사용한다. 타깃은 샘플 포인트 클라우드가
아니라 실제 삼각형 표면이다. Open3D unsigned point-to-triangle 거리를 사용하므로
watertight 조건이 필요 없다. 같은 seed의 reference samples를 모든 후보에 공유한다.

- 평균 표면 거리: 양방향 거리 평균의 산술 평균. 단위는 메시 좌표계 길이.
- RMS 표면 거리: 양방향 평균 제곱 거리를 평균한 뒤 제곱근.
- sampled Hausdorff: 양방향 샘플과 원본 vertex의 point-to-triangle 최대 거리.
  연속 표면 Hausdorff의 정확값이나 보장된 상한이 아니다.
- HD95: 양방향 거리의 각 95백분위수 중 큰 값. 최대값과 함께 보고한다.

seed 0, 1, 2의 값을 각각 저장하고 평균을 보고한다. 원본 bounding box 대각선으로
나눈 % 값도 제공한다. 메시를 각각 따로 정규화하거나 정렬하지 않는다. 알려진
물리 단위가 없으므로 mm로 표기하지 않는다. HD95와 Hausdorff는 텍스처 손실을
측정하지 않으므로 RGB 지표를 대체하지 않는다.

## 외관 지표

동일 pinhole raycast와 bilinear texture sampling으로 모든 메시를 렌더링한다.
원본 실루엣 bounding box와 공통 padding을 PSNR, SSIM, LPIPS에 적용한다.
silhouette IoU는 전체 화면에서 계산해 crop 밖 돌출도 포함한다. CPU renderer는
antialiasing을 하지 않으며 기존 nvdiffrast와 절대 점수를 비교하지 않는다.
LPIPS 입력은 [-1,1]이다. 과거 평가 코드의 [0,1] 입력도 수정했으므로 기존 발표의
LPIPS 값과 새 값은 직접 비교하지 않는다.

각 시점의 paired difference를 저장한다. ±5도 시점은 독립적이지 않으므로 원래
카메라별로 평균을 내고, 49개 그룹을 2,000회 bootstrap하여 조건부 95% 구간과
승률을 계산한다. LPIPS improvement는 baseline - candidate, 나머지는 candidate -
baseline이므로 모두 양수가 개선이다. 이 구간은 해당 장면/카메라 집합에 대한 것이며
장면 간 일반화, 다중 비교 보정, 학습 seed 변동까지 포함하지 않는다.

## 비교 조건과 해석

실제 face 수는 baseline/λ4/λ8 20,784개, λ1 20,785개이다. 기본적으로 동일 face 수를
요구하며 이번 실행은 명시적으로 허용 오차 1개를 설정했다. OBJ SHA256, texture hash,
설정 및 카메라 목록을 저장한다. 입력 파일과 저장된 원본 발표자료는 덮어쓰지 않는다.

기존 출력물은 Sobel 이외의 코드 수정 차이가 섞였을 가능성이 있다. 따라서 결과는
기존 출력물의 실제 외관 보존 개선을 보여주며 Sobel만의 인과적 효과를 확정하지 않는다.
다음 확증 실험에서는 동일 modified 코드로 λ0/λ4를 다시 생성하고 동일 texture 및
atlas 조건을 사용한다. face_importance.npy와 OBJ face 순서의 일치도 확인한다.
λ4는 이 장면의 과거 결과로 선택했으므로 다른 장면에서는 λ4를 고정하고 평가한다.
전체 재구성 일반화는 PGSR 초기화, ExMesh 학습, texture 생성, Sobel 계산 및 파라미터
선택에서 test 사진을 제외한 별도 재학습을 필요로 한다.

## 이번 실행 결과

새 시점 평균: λ0 PSNR 29.2567, SSIM 0.903049, LPIPS 0.078543.
λ4 PSNR 29.6691, SSIM 0.908082, LPIPS 0.076188.
차이는 +0.4123 dB, +0.005033, LPIPS 약 3.00% 감소이다.
PSNR/SSIM source-camera group 승률 49/49, LPIPS 43/49이다.

기하 평균 거리: 0.010135%에서 0.010202% (원본 bbox 대각선 대비).
HD95: 0.028498%에서 0.028519%. sampled Hausdorff: 0.272330%에서 0.398176%.
silhouette IoU: 0.999499에서 0.999439. 외관 개선과 기하 지표의 trade-off가 존재한다.
결론은 'scan24의 새 시점에서 기존 λ4 출력물의 외관 보존이 개선되었으나,
기하 및 실루엣 개선은 확인되지 않았다'이다.

## 참고

- https://arxiv.org/html/2409.15458v1
- https://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html
