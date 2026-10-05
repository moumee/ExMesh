"""Write publication tables from measured ablation JSON, without manual numbers."""
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'outputs/DTU/scan24/ablation'
OUT=ROOT/'deliverables'


def main():
    rows=json.loads((SOURCE/'ablation_results.json').read_text())
    if len(rows)!=13: raise ValueError('All 13 configurations must finish before reporting')
    out=OUT/'Sobel_ablation_results.csv'
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    get=lambda label:next(r for r in rows if r['label']==label)
    sys.path.insert(0,str(ROOT/'simplification'))
    from evaluate_mesh_preservation import paired_summary
    with (SOURCE/'evaluation/all/per_view_metrics.csv').open(encoding='utf-8') as f:
        per_view=[{k:(v if k in ('view','source_view') else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
    default_pairs=paired_summary(per_view,'lambda4',[r['label'] for r in rows],rgb_only=True)
    (OUT/'Sobel_ablation_vs_default.json').write_text(json.dumps(default_pairs,indent=2),encoding='utf-8')
    sections=[('λ 변화',[f'lambda{x}' for x in (0,1,2,4,8)],'σ=1, Gaussian 5×5, Sobel 3×3 고정'),
        ('Gaussian σ 변화',['sigma0.5','lambda4','sigma2'],'λ=4, Gaussian 5×5, Sobel 3×3 고정'),
        ('Gaussian 커널 크기 변화',['gaussian3','lambda4','gaussian9','gaussian13'],'λ=4, σ=1, Sobel 3×3 고정'),
        ('Sobel 커널 크기 변화',['lambda4','sobel5','sobel7'],'λ=4, σ=1, Gaussian 5×5 고정'),
        ('σ와 Gaussian 커널 조합',['lambda4','sigma2','sigma2_gaussian13'],'λ=4, Sobel 3×3 고정')]
    parts=['# Sobel 중요도 ablation 결과\n',
        '실제 DTU scan24 출력물 13개를 같은 modified 코드로 생성하고 평가했다. 원본 face 수 207,856개에서 약 20,785개(10%)로 줄였다. 모든 실행에서 입력 메시, 텍스처, xatlas 1024×1024, padding 2를 고정했다. 기존 파일은 덮어쓰지 않았다.\n',
        '평가 지표는 PSNR·SSIM·LPIPS이다. 기존 49개 카메라를 각각 ±5° 이동한 고정 98개 pose에서 원본 메시 렌더링과 비교한다. 해상도 777×581, 공통 reference crop, bilinear texture, CPU raycast renderer, LPIPS [-1,1] 입력이다. 이 표는 같은 평가기 안에서 비교한 실제 측정값이며 과거 발표의 LPIPS 절대값과 직접 비교하지 않는다.\n',
        '## 중요도 재계산\n',
        '입력 사진의 face ID는 Open3D로 재투영했다. 기본 중요도와 과거 nvdiffrast 중요도 파일의 상관계수는 0.9988578, 평균 절대 차이는 0.00200496이다. 모든 조건은 재계산한 중요도를 공유하는 방식으로 통제했다. 학습 체크포인트와 OBJ의 삼각형 좌표/face 순서 일치를 확인했다. λ=0은 가중치가 모두 1이므로 중요도 파일에 무관하고, 다시 생성한 baseline의 기하와 텍스처는 기존 baseline과 동일했다.\n',
        'P95는 각 필터 조건의 전체 유효 픽셀에서 동일한 규칙으로 구했다. Sobel 커널에 따른 단순 배율 증가를 상쇄한다. face마다 보이는 시점만 같은 비중으로 평균한다. 보이지 않은 face 1,905개에는 중요도 0을 둔다.\n']
    for title,labels,fixed in sections:
        parts+=['## '+title+'\n\n',fixed+'\n\n',
            '| λ | σ | Gaussian | Sobel | Faces | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ |\n',
            '|---:|---:|:---:|:---:|---:|---:|---:|---:|\n']
        for label in labels:
            r=get(label);g=r['gaussian_kernel'];s=r['sobel_kernel']
            parts.append(f"| {r['importance_lambda']} | {r['sigma']} | {g}×{g} | {s}×{s} | {r['faces']:,} | {r['psnr']:.4f} | {r['ssim']:.6f} | {r['lpips']:.6f} |\n")
        parts.append('\n')
    parts+=['## 지표별 최고 조건\n',
        '| 지표 | λ | σ | Gaussian | Sobel | 측정값 |\n','|---|---:|---:|:---:|:---:|---:|\n']
    for m in ('psnr','ssim','lpips'):
        best=min(rows,key=lambda r:r[m]) if m=='lpips' else max(rows,key=lambda r:r[m])
        parts.append(f"| {m.upper()} | {best['importance_lambda']} | {best['sigma']} | {best['gaussian_kernel']}×{best['gaussian_kernel']} | {best['sobel_kernel']}×{best['sobel_kernel']} | {best[m]:.6f} |\n")
    base=get('lambda0');default=get('lambda4');best=min(rows,key=lambda r:r['lpips'])
    parts+=['\n## 기본 설정 및 최저 LPIPS 조건의 개선량\n',
        '| 조건 | ΔPSNR (dB) | ΔSSIM | LPIPS 감소율 |\n','|---|---:|---:|---:|\n']
    for r in (default,best):
        parts.append(f"| {r['label']} | {r['psnr']-base['psnr']:+.4f} | {r['ssim']-base['ssim']:+.6f} | {(base['lpips']-r['lpips'])/base['lpips']*100:+.2f}% |\n")
    if best['label']!='lambda4':
        parts+=['\n## 최저 LPIPS 조건과 기본 λ4 설정의 직접 비교\n',
            '| 지표 | 개선량 평균 | 조건부 95% 구간 | 카메라 그룹 승률 |\n',
            '|---|---:|---|---:|\n']
        for metric in ('psnr','ssim','lpips'):
            entry=default_pairs[best['label']][metric];low,high=entry['cluster_bootstrap_95ci']
            parts.append(f"| {metric.upper()} | {entry['improvement_mean']:+.6f} | [{low:+.6f}, {high:+.6f}] | {entry['source_view_win_rate']*100:.2f}% |\n")
    parts+=['\n## 해석 범위\n',
        '한 변수씩 바꾸는 ablation이며 모든 λ×σ×커널 조합의 전체 탐색은 아니다. σ=2와 Gaussian 13×13 조합을 추가해, 작은 커널에서 Gaussian 분포가 잘리는 영향을 확인했다. σ=1의 Gaussian 9×9와 13×13은 중요도 최대 차이가 약 6.26×10⁻⁶으로 거의 같다.\n\n',
        '그런데 9×9와 13×13 출력물의 xatlas UV vertex 수는 15,320개와 15,503개로 다르며 최종 지표도 차이가 있다. 작은 중요도 변화가 collapse/매핑 결과를 바꾸는 수치적 민감성이 영향을 줄 가능성이 있다. 이 관측을 필터의 영상 처리 효과만으로 설명하거나 13×13의 보편적 우월성으로 해석하지 않는다. 중요도 미세 교란 및 출력 반복 검증이 후속 과제다.\n\n',
        '동일 장면에서 가장 좋은 설정을 고르는 탐색 실험이다. 이 98개 pose에서 선택한 최고 설정을 독립 test 성능으로 부르지 않는다. 비교 결과를 보고 설정을 고정한 다음 다른 장면에서 재검증해야 한다. 단순화/atlas 전체 실행의 seed 반복은 하지 않았으며 단일 출력물을 비교한다. face 수의 최대 차이는 raw CSV에 기록했다.\n',
        '각 조건의 원본 카메라별 두 pose를 묶어 49개 그룹을 bootstrap(2,000회)한 조건부 95% 구간과 승률은 CSV와 evaluation/all/summary.json에 있다. LPIPS improvement는 baseline−candidate, PSNR/SSIM은 candidate−baseline이다. 최고 설정의 사후 선택이나 다중 비교를 보정한 확증 구간은 아니다.\n',
        '이번 표에서 기하 지표를 제외했으므로 외관 보존에 대한 결론만 낸다.\n',
        '## 재실행\n\n```powershell\n./.ablation_env/Scripts/python.exe simplification/run_sobel_ablation.py\npython simplification/summarize_sobel_ablation.py\n```\n',
        '환경이 없다면 Python 가상환경에 numpy, Pillow, scipy, Open3D, torch, torchvision, numba, xatlas, opencv-python을 준비한다. pipeline은 완료한 중요도·단순화·평가 결과를 재사용한다. 입력 해시/실험 조건이 바뀌면 기존 폴더 재사용을 거부한다. manifest.json, 중요도 JSON, 실행 로그, output SHA256, 카메라 JSON 및 per-view CSV를 함께 보존한다.\n']
    (OUT/'Sobel_ablation_report.md').write_text(''.join(parts),encoding='utf-8')
    print(out)


if __name__=='__main__':main()
