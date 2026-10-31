"""Produce numerical report and fixed-view comparisons for normal guard."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from evaluate_mesh_preservation import load_mesh, texture_data, perturb_cameras, render_cpu
from ablation_visuals import clay
from run_normal_guard_comparison import ROOT, OUT, REF, TEX, CAM, BUDGETS, METHODS, paths

DEST = ROOT/'deliverables'
LABELS = dict(wild='Wild', sobel='Sobel λ4', guarded='Normal guard')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.loads((OUT/'results.json').read_text())
    lookup = {(r['target_faces'], r['method']): r for r in rows}
    changes = []
    for budget in BUDGETS:
        new = lookup[budget, 'guarded']
        for method in ('wild', 'sobel'):
            base = lookup[budget, method]
            changes.append(dict(target_faces=budget, baseline=method,
                psnr_delta_db=new['psnr']-base['psnr'], ssim_delta=new['ssim']-base['ssim'],
                lpips_change_pct=100*(new['lpips']/base['lpips']-1),
                mean_distance_change_pct=100*(new['mean_surface_distance']/base['mean_surface_distance']-1),
                hd95_change_pct=100*(new['hd95']/base['hd95']-1),
                sampled_hd_change_pct=100*(new['sampled_hausdorff']/base['sampled_hausdorff']-1),
                iou_delta=new['silhouette_iou']-base['silhouette_iou']))
    with (DEST/'NormalGuard_변화량.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(changes[0]))
        writer.writeheader(); writer.writerows(changes)
    plots = [('psnr', 'PSNR (dB), higher better'), ('ssim', 'SSIM, higher better'),
             ('lpips', 'LPIPS, lower better'),
             ('mean_surface_distance_pct_bbox', 'Mean surface distance (% bbox), lower better'),
             ('hd95_pct_bbox', 'HD95 (% bbox), lower better'),
             ('sampled_hausdorff_pct_bbox', 'Sampled Hausdorff (% bbox), lower better')]
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    for ax, (key, title) in zip(axes.flat, plots):
        for method, color in zip(METHODS, ('#6b7785', '#d99b27', '#168196')):
            ax.plot(range(3), [lookup[b, method][key] for b in reversed(BUDGETS)],
                    'o-', color=color, linewidth=2, label=LABELS[method])
        ax.set_xticks(range(3), ['5k', '10k', '20k']); ax.set_xlabel('Target faces')
        ax.set_title(title, fontsize=12); ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('scan24: Rejecting locally inverted collapse candidates', fontsize=20)
    chart = DEST/'NormalGuard_지표비교.png'
    fig.savefig(chart, dpi=160); plt.close(fig)

    ref = load_mesh(REF); texture_data(ref, TEX)
    cameras = perturb_cameras(json.loads(CAM.read_text()), (ref['v'].min(0)+ref['v'].max(0))/2, 5, 1)
    camera = next(c for c in cameras if c['img_name'] == '0024_orbit_+5')
    _, mask = render_cpu(ref, camera)
    yy, xx = np.where(mask)
    y0, y1 = max(0, yy.min()-16), min(mask.shape[0], yy.max()+17)
    x0, x1 = max(0, xx.min()-16), min(mask.shape[1], xx.max()+17)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 25)
    for budget in BUDGETS:
        meshes = {'reference': ref}
        for method in METHODS:
            meshpath, texpath = paths(method, budget)
            data = load_mesh(meshpath); texture_data(data, texpath); meshes[method] = data
        for mode in ('rgb', 'clay'):
            tiles = []
            for method, mesh in meshes.items():
                rgb = render_cpu(mesh, camera)[0] if mode == 'rgb' else clay(mesh, camera)
                im = Image.fromarray((np.clip(rgb[y0:y1, x0:x1], 0, 1)*255).astype(np.uint8))
                im = im.resize((450, round(im.height*450/im.width)), Image.Resampling.LANCZOS)
                tiles.append((method, im))
            panel = Image.new('RGB', (1800, 90+tiles[0][1].height), 'white')
            draw = ImageDraw.Draw(panel)
            draw.text((15, 8), f'{budget:,} target faces / 0024_orbit_+5 / {mode}', fill='black', font=font)
            for index, (method, im) in enumerate(tiles):
                draw.text((450*index+15, 46), 'Original' if method == 'reference' else LABELS[method], fill='black', font=font)
                panel.paste(im, (450*index, 90))
            panel.save(OUT/'visuals'/f'{budget}_{mode}.png')

    lines = ['# Collapse 방향 반전 방지 실험', '', '실행일: 2026-10-21', '',
             '## 변경과 가설', '',
             '이전 색 모델 QEM은 활성 코드에서 롤백하고 소스를 '
             '`outputs/DTU/scan24/appearance_qem/rolled_back_source/`에 보관했다.', '',
             '이번 방법은 Wild의 비용·최적 위치를 유지하며 선택된 Collapse 직전에 남는 삼각형을 검사한다. '
             'normal 내적이 0 이하(90도 이상 변화) 또는 새 면적이 기존의 1e-12 이하인 후보는 거부한다. '
             '제거되는 face와 원래 면적이 0인 face는 제외한다. 이웃 Collapse로 갱신되면 다시 후보가 될 수 있다.', '',
             '가설: 평면 거리 QEM만으로 허용되던 국소 뒤집힘을 막으면 표면과 텍스처 전달 품질이 좋아질 수 있다. '
             '이는 [CGAL bounded normal change filter](https://doc.cgal.org/latest/Surface_mesh_simplification/classCGAL_1_1Surface__mesh__simplification_1_1Bounded__normal__change__filter.html)'
             '와 같은 국소 검사 원리다. 원 논문 구현의 오류 수정이라고 주장하지 않는다.', '',
             '## 비교 조건', '',
             '- scan24 원본 207,856 faces, 원본 OBJ 위치 topology와 텍스처.',
             '- 목표 20k/10k/5k. 동일 xatlas 1024·padding 2·collapse-history texture baking.',
             '- 각 조건의 실제 face 수 차이 최대 2 허용. 목표 미도달 시 실행 중단.',
             '- 기존 Wild/Sobel control hash 검증. 새 Wild 20k OBJ·texture가 기존과 byte 동일한지 검증.',
             '- 기하: 방향별 100,000 표면 샘플, seed 0/1/2. 원본 bbox 대각선 대비 %.',
             '- 외관: 동일 98개 시점(49 카메라 × ±5도), 절반 해상도, 공통 reference crop.',
             '- normal 기준은 결과를 보기 전 고정했고 조건별 튜닝하지 않았다.', '',
             '## 결과', '',
             '| 목표 faces | 방법 | 실제 faces | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | 평균 거리 % ↓ | HD95 % ↓ | sampled HD % ↓ |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['target_faces']:,} | {LABELS[r['method']]} | {r['faces']:,} | {r['psnr']:.4f} | {r['ssim']:.6f} | {r['lpips']:.6f} | {r['silhouette_iou']:.6f} | {r['mean_surface_distance_pct_bbox']:.6f} | {r['hd95_pct_bbox']:.6f} | {r['sampled_hausdorff_pct_bbox']:.6f} |")
    lines += ['', f'![지표 비교]({chart.as_posix()})', '', '## 기준별 변화', '',
              '| 목표 faces | 기준 | PSNR 변화 dB | LPIPS 변화 % | 평균 거리 변화 % | HD95 변화 % | sampled HD 변화 % |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for r in changes:
        lines.append(f"| {r['target_faces']:,} | {r['baseline']} | {r['psnr_delta_db']:+.4f} | {r['lpips_change_pct']:+.2f} | {r['mean_distance_change_pct']:+.2f} | {r['hd95_change_pct']:+.2f} | {r['sampled_hd_change_pct']:+.2f} |")
    lines += ['', 'PSNR은 양수, 오차 지표는 음수가 개선이다.', '', '## Collapse 진단 및 실행 시간', '',
              '| 실행 | 검사 후보 | 거부 후보 | 수락 Collapse | 수락 flip 사건 | 수락 퇴화 사건 | 단순화 초 | 전체 초 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, budget in [('wild', 20000)] + [('guarded', b) for b in BUDGETS]:
        t = json.loads((OUT/'meshes'/f'{method}_{budget}.statistics.json').read_text())
        lines.append(f"| {LABELS[method]} {budget:,} | {t['checked']:,} | {t['rejected']:,} | {t['accepted']:,} | {t['accepted_flip_events']:,} | {t['accepted_degenerate_events']:,} | {t['simplification_seconds']:.1f} | {t['total_seconds']:.1f} |")
    lines += ['', '사건 수는 각 Collapse에서 해당하는 face 수를 누적한 것이다. 동일 face가 여러 번 포함될 수 '
              '있으며 최종 뒤집힌 face 수가 아니다. 거부 후보도 이웃 갱신 이후 재검사될 수 있다. '
              '병렬 실행으로 부하가 달라 시간 비율은 독립 성능 벤치마크가 아니다. 단순화 시간은 초기화·진단·Collapse를 '
              '포함하고 baking은 제외한다.', '', '## 카메라 그룹별 변화', '',
              '±5도 시점을 원본 카메라별로 묶은 bootstrap 2,000회. Wild 대비 개선량이며 양수가 좋다.', '',
              '| 목표 faces | 지표 | 개선량 | 95% 구간 | 그룹 승률 |', '|---:|---|---:|---:|---:|']
    for budget in BUDGETS:
        report = json.loads((OUT/'evaluation'/str(budget)/'summary.json').read_text())
        for metric, r in report['paired_comparison']['guarded'].items():
            lo, hi = r['cluster_bootstrap_95ci']
            lines.append(f"| {budget:,} | {metric} | {r['improvement_mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {r['source_view_win_rate']*100:.1f}% |")
    lines += ['', '## 해석 범위', '']
    for base in ('wild', 'sobel'):
        selected = [r for r in changes if r['baseline'] == base]
        wins = sum(r['psnr_delta_db'] > 0 and r['lpips_change_pct'] < 0 for r in selected)
        lines.append(f'- {base} 대비 PSNR·LPIPS 동시 개선: {wins}/3 조건.')
    lines += ['', '같은 scan24에서 진행한 탐색 실험이며 다른 장면으로 일반화하지 않는다. '
              '이 검사는 국소 normal 반전만 막으며 자기 교차·위상 보존·누적 normal 변화 제한을 보장하지 않는다. '
              '최종 RGB 변화에는 geometry와 이에 따라 바뀐 UV packing·baking이 함께 포함된다. '
              'sampled Hausdorff는 표면 샘플과 원본 vertex로 계산한 근사값이다. 기본 실행 방식은 Wild로 유지한다.', '',
              '## 재현', '', '```powershell',
              '.ablation_env/Scripts/python.exe simplification/run_normal_guard_comparison.py',
              '.ablation_env/Scripts/python.exe simplification/report_normal_guard_comparison.py', '```', '',
              '개별 실행 옵션: `--method guarded`. 입력·소스·control hash는 manifest, 상세 지표는 '
              '`outputs/DTU/scan24/normal_guard/evaluation/`에 저장한다.', '']
    (DEST/'NormalGuard_비교보고서.md').write_text('\n'.join(lines), encoding='utf-8')
    viewer = '''<!doctype html><meta charset="utf-8"><title>Normal guard 비교</title>
<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px}img{width:100%}select{font:inherit;margin:0 20px 20px 0}</style>
<h1>Collapse 방향 반전 방지 비교</h1><p>scan24, 동일 원본과 시점. 최종 geometry와 texture 비교.</p>
<select id="budget"><option>20000</option><option>10000</option><option>5000</option></select>
<select id="mode"><option value="rgb">RGB</option><option value="clay">Clay</option></select><img id="panel">
<script>function update(){document.getElementById('panel').src='../outputs/DTU/scan24/normal_guard/visuals/'+document.getElementById('budget').value+'_'+document.getElementById('mode').value+'.png'}for(const id of ['budget','mode'])document.getElementById(id).onchange=update;update()</script>
<h2>정량 비교</h2><img src="NormalGuard_지표비교.png">'''
    (DEST/'NormalGuard_비교뷰어.html').write_text(viewer, encoding='utf-8')
    print('Normal guard report generated', flush=True)


if __name__ == '__main__':
    main()
