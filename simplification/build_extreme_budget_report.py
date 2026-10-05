"""Export fixed-coordinate evidence crops, report and local comparison viewer."""
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/DTU/scan24/extreme_budgets'


def main():
    rows = json.loads((OUT / 'results.json').read_text())
    visual = OUT / 'visuals'
    # Coordinates are frozen from the reference render, identical for all methods.
    crops = {'0024_orbit_+5': (820, 240, 1340, 850),
             '0000_orbit_+5': (580, 140, 1080, 740),
             '0048_orbit_+5': (560, 340, 1060, 940)}
    for view, box in crops.items():
        for src in visual.glob(view+'_*.png'):
            if src.stem.endswith('_crop'): continue
            Image.open(src).crop(box).save(src.with_name(src.stem+'_crop.png'))
    (OUT / 'crop_coordinates.json').write_text(json.dumps(crops, indent=2))
    text = ['# 극단적인 face budget 비교', '',
            'DTU scan24 원본 207,856 faces. baseline λ=0과 Sobel λ=4, Gaussian σ=1 / 13×13, Sobel 3×3을 비교했다. 설정은 약 2만 face ablation 결과에서 선택한 그대로 고정했다.', '',
            '평가: 기존 49개 카메라에서 ±5° 회전시킨 같은 98개 pose. 각 pose에서 원본 메시 렌더링을 GT로 사용한다. PSNR·SSIM은 높을수록, LPIPS는 낮을수록 좋다. 지표는 777×581 렌더의 공통 reference bbox에서 계산하고, 시각 비교 이미지는 1554×1162로 별도 렌더했다. 모든 조건의 atlas는 1024×1024, padding 2다.', '',
            '| 목표 faces | 방법 | 실제 faces | PSNR (dB) | SSIM | LPIPS |',
            '|---:|:---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f'| {r["target_faces"]:,} | {r["method"]} | {r["actual_faces"]:,} | {r["psnr"]:.4f} | {r["ssim"]:.6f} | {r["lpips"]:.6f} |')
    text += ['', '## Budget별 Sobel 변화', '', '| 목표 faces | ΔPSNR | ΔSSIM | LPIPS 감소율 |', '|---:|---:|---:|---:|']
    for b in (20785, 1000, 500, 100):
        base = next(r for r in rows if r['target_faces']==b and r['method']=='baseline')
        sobel = next(r for r in rows if r['target_faces']==b and r['method']=='sobel')
        text.append(f'| {b:,} | {sobel["psnr"]-base["psnr"]:+.4f} | {sobel["ssim"]-base["ssim"]:+.6f} | {(base["lpips"]-sobel["lpips"])/base["lpips"]*100:+.2f}% |')
    text += ['', '## 시점별 차이의 불확실성', '',
             '±5° 두 pose를 원본 카메라별로 묶은 49개 그룹에서 bootstrap 2,000회. 아래 구간은 시점 변동만 반영하며 메시 생성 반복이나 설정 선택의 불확실성을 포함하지 않는다. 양수는 Sobel 개선이다.', '',
             '| 목표 faces | 지표 | 평균 개선 | 95% 구간 | 그룹 승률 |', '|---:|:---|---:|:---|---:|']
    for b in (1000, 500, 100):
        paired = json.loads((OUT/'evaluation'/str(b)/'summary.json').read_text())['paired_comparison']['sobel']
        for metric, result in paired.items():
            lo, hi = result['cluster_bootstrap_95ci']
            text.append(f'| {b:,} | {metric} | {result["improvement_mean"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {result["source_view_win_rate"]*100:.2f}% |')
    text += ['', '## 시각 비교 조건', '',
             '0000, 0024, 0048의 +5° pose를 지표 확인 전에 고정했다. 동일 pose, 조명 없는 texture color 렌더, 동일 검정 배경으로 비교한다. 확대 이미지는 crop_coordinates.json에 기록한 공통 좌표를 사용한다. 각 후보의 bbox로 재정렬하지 않는다.', '',
             '고해상도 텍스처가 낮은 face 수에서도 창문과 벽돌을 표현할 수 있다. 따라서 RGB 이미지가 좋아 보여도 세부 3D 형상이 보존되었다는 의미는 아니다. 결과에는 geometry와 texture transfer, UV atlas의 영향이 함께 포함된다.', '',
             '단일 장면에서 한 번씩 생성한 탐색 실험이다. 98개 pose는 가까운 새 시점이며 독립 장면이나 실제 촬영 사진의 test set은 아니다. Budget 간 절대 지표와 같은 budget의 방법 차이를 구분해서 해석해야 한다.', '',
             '재현: `.ablation_env/Scripts/python.exe simplification/run_extreme_budget_comparison.py` 이후 `simplification/build_extreme_budget_report.py`. raw summary와 per-view CSV는 outputs/DTU/scan24/extreme_budgets/evaluation/<budget>/에 저장한다.']
    (ROOT/'deliverables/Sobel_extreme_budget_report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Sobel 극단적 단순화 비교</title>
<style>body{font-family:Malgun Gothic,Arial;background:#f7f9fb;color:#18344b;margin:25px}h1{font-size:28px}select{font-size:18px;margin:8px;padding:6px}.row{display:flex;gap:14px}.col{width:33.33%}img{width:100%;background:black}h2{font-size:20px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #ccd4db;text-align:right}th:first-child,td:first-child{text-align:left}p{line-height:1.7}.hint{color:#596b79}</style>
<h1>Sobel 중요도: face 수를 극단적으로 줄였을 때</h1><p>고정 설정 λ=4, Gaussian σ=1 / 13×13, Sobel 3×3. 원본 메시 렌더링과 비교.</p>
<label>목표 faces <select id="budget"><option value="1000">1,000</option><option value="500">500</option><option value="100">100</option><option value="20785">약 20,000</option></select></label>
<label>새 시점 <select id="view"><option value="0024_orbit_+5">0024 +5°</option><option value="0000_orbit_+5">0000 +5°</option><option value="0048_orbit_+5">0048 +5°</option></select></label>
<label><input type="checkbox" id="crop"> 공통 좌표 확대</label>
<div class="row"><div class="col"><h2>원본 207,856 faces</h2><a id="ar" target="_blank"><img id="ir"></a></div><div class="col"><h2 id="lb"></h2><a id="ab" target="_blank"><img id="ib"></a></div><div class="col"><h2 id="ls"></h2><a id="as" target="_blank"><img id="is"></a></div></div>
<p class="hint">이미지를 누르면 원본 해상도로 열립니다. 동일 카메라 및 동일 좌표 crop. 1024×1024 텍스처 고정.</p><table id="metrics"></table>
<p>지표: 같은 98개 새 pose에서 원본 메시 렌더와 비교한 평균. PSNR·SSIM은 높을수록, LPIPS는 낮을수록 좋음. 지표 해상도 777×581, 위 이미지 1554×1162.</p>
<p>이 설정은 scan24의 약 2만 face 실험에서 선택했습니다. 낮은 face 수에서는 재튜닝하지 않았습니다. 텍스처가 표현하는 세부 무늬와 실제 3D 형상 보존을 구분해야 합니다.</p>
<script>const rows=ROWS;const budget=document.querySelector('#budget'),view=document.querySelector('#view'),crop=document.querySelector('#crop');
function update(){const b=+budget.value,v=view.value,suffix=crop.checked?'_crop':'';for(const [short,method] of [['r','reference'],['b','baseline_'+b],['s','sobel_'+b]]){const src='../outputs/DTU/scan24/extreme_budgets/visuals/'+v+'_'+method+suffix+'.png';document.querySelector('#i'+short).src=src;document.querySelector('#a'+short).href=src}const selected=rows.filter(r=>r.target_faces===b);for(const [id,m] of [['lb','baseline'],['ls','sobel']])document.querySelector('#'+id).textContent=(m==='baseline'?'Baseline λ=0':'Sobel λ=4')+' / '+selected.find(r=>r.method===m).actual_faces.toLocaleString()+' faces';document.querySelector('#metrics').innerHTML='<tr><th>방법</th><th>PSNR (dB)</th><th>SSIM</th><th>LPIPS</th></tr>'+selected.map(r=>'<tr><td>'+r.method+'</td><td>'+r.psnr.toFixed(4)+'</td><td>'+r.ssim.toFixed(6)+'</td><td>'+r.lpips.toFixed(6)+'</td></tr>').join('')};for(const e of [budget,view,crop])e.onchange=update;update();</script></html>'''.replace('ROWS', json.dumps(rows))
    (ROOT/'deliverables/Sobel_extreme_budget_viewer.html').write_text(html, encoding='utf-8')


if __name__ == '__main__': main()
