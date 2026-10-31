# Collapse normal guard 실험

이전 affine RGB field QEM은 활성 코드에서 제거했다. 원본 소스와 결과는
`outputs/DTU/scan24/appearance_qem/rolled_back_source/`에 기록으로 보관한다.

새 방법은 Wild 비용과 위치 계산을 유지하고, heap에서 선택된 후보를 적용하기
직전에 남는 one-ring 삼각형의 전후 normal을 검사한다. normal 내적이 0 이하이면
(90도 이상 회전), 또는 새 면적이 기존 면적의 1e-12 이하이면 해당 후보를 거부한다.
1e-12는 면적 품질을 조절하는 하이퍼파라미터가 아니라 수치적 퇴화 검사다.
Collapse 자체로 제거되는 face와 이미 면적이 0인 face는 검사에서 제외한다.
인접 Collapse로 해당 edge가 갱신되면 다시 검사할 수 있다.

이는 [CGAL의 bounded normal change filter](https://doc.cgal.org/latest/Surface_mesh_simplification/classCGAL_1_1Surface__mesh__simplification_1_1Bounded__normal__change__filter.html)
와 같은 국소 orientation 검사 원리를 이 코드의 physical/virtual edge에 적용한
실험이다. 원본 Wild 논문 재현의 필수 수정이라고 주장하지 않는다.
자기 교차·위상 보존·여러 단계에 걸친 누적 normal 변화 제한을 보장하지 않는다.

```powershell
.ablation_env/Scripts/python.exe simplification/run_normal_guard_comparison.py
.ablation_env/Scripts/python.exe simplification/report_normal_guard_comparison.py
.ablation_env/Scripts/python.exe -m unittest discover -s simplification -p test_normal_guard.py -v
```

개별 실행은 기존 명령에 `--method guarded`를 사용한다. 기본값은 Wild다.
`.statistics.json`에 검사/거부/수락 횟수와 수락된 face flip·퇴화 사건 수를 기록한다.
동일 face가 여러 번 포함될 수 있으므로 사건 수를 최종 뒤집힌 face 수로 해석하면
안 된다. Wild 실행도 진단을 기록하지만 후보 선택은 바꾸지 않는다.

scan24 20k/10k/5k, atlas 1024, padding 2, 기존 texture transfer를 고정한다.
Wild 20k는 새로 생성해 저장된 control의 OBJ·texture와 byte 동일성을 검증한다.
모든 control과 입력·실험 소스의 SHA256을 manifest에 기록한다.
같은 98개 novel pose와 표면 샘플 seed 0/1/2로 평가한다.
목표 face 수에 도달하지 못하면 비교를 중단하고, face 수를 맞춘 것처럼 보고하지 않는다.
