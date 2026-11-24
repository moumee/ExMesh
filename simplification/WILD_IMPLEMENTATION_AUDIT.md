# Wild Python 구현 검토와 공통 수정

기준 문헌: Liu, Zhang, Yuksel, *Simplifying Textured Triangle Meshes in the Wild*, TOG 2025.
[저자 제공 논문](https://www.cemyuksel.com/research/papers/simplifying_meshes_in_the_wild.pdf)

원본 `wild_simplify_xatlas.py`, `wild_simplify_xatlas_modified.py`를 보존하고,
검토한 공통 구현을 `wild_simplify_xatlas_verified.py`에 별도로 작성했다.
이 이름은 아래 항목을 검토했다는 뜻이다. 저자 reference implementation과 일치함을 보장하지 않는다.
저자의 공개 연구 페이지에서 해당 논문의 reference code 링크를 확인하지 못해, 논문 수식과 작은 분석적 사례로 검증했다.

## 논문 대조 결과

| 항목 | 논문 | 검토 결과 |
|:---|:---|:---|
| Vertex plane Q | Appendix A.1, 면적/3 가중치 | cross 길이/6 사용. 해당 수식과 일치 |
| Q 누적 | Section 4.2, Appendix A.2 | collapse에서 Q[a]+=Q[b], 원본 평면 정보를 누적 |
| Boundary area Q | Appendix A.3 Eq.15–17 | 0.5 곱의 cross matrix 수식과 일치. 분석적 cross-product 면적 비용으로 검증 |
| Area Q 갱신 | Section 4.2 | 현재 boundary에서 다시 계산, 누적하지 않음. 여러 collapse의 모든 비용 갱신에서 cache와 직접 계산을 대조 |
| Virtual edges | Section 4.1 | 다른 component의 triangle 거리가 2r보다 작을 때 가장 가까운 vertex pair 연결. 작은 분리 면 사례 검증 |
| Collapse 대상 | Section 4.1 | physical/virtual 1-simplex 모두 처리하고, face가 없는 1-simplex도 유지 |
| Successive texture map | Section 4.3.1 | 역순 collapse history로 local one-ring 선택, query는 최종 표면의 원래 위치를 계속 사용. 이 고정 query는 논문의 의도 |
| QEM 최소화 | Appendix A.2 Eq.14 | 기존은 solve 실패 때 midpoint만 사용. 새 공통 구현은 rank가 부족하거나 거의 특이하면 pseudoinverse 사용 |
| UV packing | Section 4.3.1 | 논문의 mesh color texture 대신 xatlas 사용. 논문은 다른 injective mapping도 허용하지만, 동일 reference implementation은 아님 |
| OBJ 입력 topology | Geometry/UV correspondence를 분리하는 Section4.3 | Open3D의 UV 분리 정점을 geometry로 그대로 사용하던 문제 확인. 원본 OBJ의 v/f 위치 인덱스로 복원 |

현재 검토에서 핵심 수식의 명백한 이식 오류는 확인하지 못했다. 다만 **원본 OBJ 위치 연결 정보를 loader의 UV 분리 topology로 바꾸던 입력 표현 문제**를 확인했다. 원본 파일은 v104,706개·vt122,923개인데, Open3D는 geometry vertex122,923개를 반환한다. 원래 위치 인덱스는1component인데 loader 결과는1,507components다. 원래 파일의 각 face/corner 좌표와 loader의 좌표 차이는 최대5e-10으로, 위치/face 순서는 사실상 같으면서 연결 정보만 달랐다. 이는 virtual edge 수식 자체의 오류와 구별해야 한다.

`obj_geometry_topology.py`가 원래 OBJ v/f 위치 인덱스를 복원한다. **좌표가 같다는 이유로 다른 원본 위치 ID를 합치지 않는다.** 우연히 겹친 별개 표면의 연결을 바꾸지 않기 위해서다. Per-face UV/material/importance 순서는 유지하고, 복원 전에 face 수 및 corner 위치 일치를 검증한다. 불일치하면 중단한다. 이 수정은 Wild·Sobel 모두에 적용했다. 현재 지원 범위는 일반3D 위치를 사용하는 삼각형 OBJ이며, 다른 파일 형식은 기존 loader geometry를 그대로 사용한다.

UV seam으로 분리된 표현도 Wild가 처리할 수 있는 유효한 simplicial complex이다. 그러나 해당 scan24 OBJ의 원래 연결 구조와 달라 비용/가상 edge/결과가 UV 분할에 의존한다. 따라서 원본 geometry topology를 반영하려는 이번 구현에서 복원했다. Nonmanifold 출력 그 자체는 오류 증거가 아니다.

## 공통 수치 개선

`wild_simplify_xatlas_verified.py`의 `minimize_quadric()`는 symmetric A를 eigendecomposition해서 상대 eigenvalue 1e-10 이하 방향을 nullspace로 취급한다. 별도 solver 파일은 이 함수로 통합했다.
Full-rank에서는 기존 solve를 사용한다. Rank가 부족하면 `x=m-A⁺(Am+b)`로 midpoint m에 가장 가까운 최소해를 택한다.
대칭 PSD 행렬에서는 SVD pseudoinverse와 같은 해이며, 논문 Appendix A.2가 제안하는 안정적인 해법에 해당한다.
Nullspace를 원점으로 당기지 않고 midpoint에 유지한다. 제곱 오차의 작은 음수 반올림은 0으로 제한한다.
정렬한 edge 순회로 비용 합산과 동률 처리 순서를 고정했다. 모든 비교 방법에 같은 변경을 적용했다.
작은 nearly-singular 사례에서 기존 solve가 z=-1e8을 만드는 반면 새 solver는 midpoint의 z=0.5를 유지한다.
이는 기존 scan24 결과가 그 문제 때문에 실패했다는 증거는 아니다.

## 현재 구현과 재현

미사용 원본 구조 보존 확장은 제거했다. 현재 verified 구현은 Wild와 Sobel만 지원한다.
현재 출력은 `outputs/DTU/scan24/sobel_current/meshes`에 있고, 원본207,856 faces에서 목표20k/10k/5k로 독립 단순화했다. Sobelλ4·σ1·Gaussian13·Sobel3, xatlas1024/padding2.

검증: `.ablation_env/Scripts/python.exe -m unittest discover -s simplification -p "test_wild_*.py" -v`.
세부 실행·자료 경로: `deliverables/Sobel_통합실험_스크립트와_파일경로.md`.

## 속도 최적화 (2026-11)

`wild_simplify_xatlas_verified.py`의 실행 시간을 줄였다. 단순화 결과(정점, face, collapse 순서)는 바꾸지 않았다.

| 99,998 faces (Pixal3D), ratio 0.1 | 이전 | 현재 |
|:---|---:|---:|
| Numba 있음 | 98초 | 14초 |
| Numba 없음 | 1,247초 | 30초 |

- **Edge 비용 재사용.** 이전에는 collapse마다 주변 edge 약 36개의 비용을 edge마다 `eigh`와 `solve`로 다시 계산했다. 이제 입력(양 끝 정점의 Q, 위치, boundary 항)이 바뀐 edge만(평균 약 7개) `compute_edge_costs()`/`minimize_quadrics()`로 한 번에 계산한다. Rank가 부족한 행은 기존 `minimize_quadric()`을 그대로 쓴다.
- **Heap.** 비용이 그대로인 edge는 heap에 다시 넣지 않고 동률 처리 번호만 갱신하며, 그 entry가 맨 위에 왔을 때 새 번호로 옮긴다. 번호는 커지기만 하므로 pop 순서는 매번 다시 넣는 방식과 같다.
- **기준 루프.** `simplify(..., reuse_costs=False)`는 주변 edge를 모두 다시 계산하고 다시 넣는 기존 방식이다. 기본값(`True`)과 collapse 순서가 같은지 테스트한다.
- **Texture 역추적.** Collapse history 전체를 Numba kernel 하나(`trace_history`)로 처리한다. Numba가 없으면 history step마다 NumPy로 벡터화한 `closest_points_barycentric()`을 쓴다. 이전에는 이 구간이 순수 Python 이중 루프라서 Numba 없이 20분 가까이 걸렸다.
- **Closest point.** 배열 임시값과 BLAS 호출 없이 scalar 산술로 다시 썼다(`closest_point_barycentric`). Numba 경로와 NumPy 경로가 같은 연산 순서를 쓰므로 Numba 유무와 관계없이 같은 texture가 나온다.
- **그 밖.** UV rasterize kernel(`rasterize_uv_triangles`), xatlas가 face 순서를 유지한 경우의 빠른 대응(`match_atlas_faces`), 다른 component의 bounding box 근처 face만 보는 virtual edge 탐색(`near_other_component`), SciPy connected components.

검증한 내용:

- 원본 구현과 `simplify()` 출력(정점, face, history)이 랜덤 메시 60개(boundary, 평면 동률, 닫힌 표면, virtual edge, seam, 중복·퇴화 face, normal guard, face weight)에서 bit 단위로 같다.
- 실제 메시 두 개(Pixal3D 99,998 faces, TripoSR 103,252 faces)에서 wild/guarded/photometric/visibility, ratio 0.1–0.5의 출력 OBJ가 이전 구현과 byte 단위로 같다.
- Texture는 1024² texel 중 15–48개가 달라진다(PSNR 80 dB 이상). Closest point의 내적을 BLAS(FMA) 대신 명시적 산술로 계산하면서 마지막 자리 반올림이 달라졌고, 거리가 같은 이웃 삼각형 사이의 선택이 바뀐 texel이다.
- `test_wild_fast_paths.py`: 배치 비용과 단일 비용, 비용 재사용과 기준 루프, scalar/array closest point, Numba/NumPy history 역추적, rasterize kernel과 삼각형별 rasterize를 각각 대조한다. Numba 유무 양쪽에서 통과한다.

Numba는 `requirements.txt`에 추가했다. 없어도 동작하지만 texture 단계가 느려진다.
