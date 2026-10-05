"""Tables and scientific comparison plates from verified measured outputs."""
import csv,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run_realityscan_compare import ROOT,OUT,sha256
from evaluate_mesh_preservation import paired_summary

D=ROOT/'deliverables';BUDGETS=(20000,10000,5000)
NAMES={'wild':'Wild','sobel':'Sobel λ=4','realityscan':'RealityScan Simplification'}
FONT='C:/Windows/Fonts/malgun.ttf'

def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plate(path,items,mode,footer,crop=True):
    w,h=(520,610) if crop else (622,465)
    canvas=Image.new('RGB',(w*len(items),h+150),'white');draw=ImageDraw.Draw(canvas)
    for i,(label,name) in enumerate(items):
        draw.text((i*w+12,15),label,font=ImageFont.truetype(FONT,32),fill='#153954')
        source=OUT/'visuals'/f'0024_orbit_+5_{name}_{mode}{"_crop" if crop else ""}.png'
        image=Image.open(source).convert('RGB')
        if not crop:image=image.resize((w,h),Image.Resampling.LANCZOS)
        canvas.paste(image,(i*w,78))
    draw.text((14,h+100),footer,font=ImageFont.truetype(FONT,21),fill='#273C50')
    canvas.save(path)

def main():
    assets=json.loads((OUT/'assets.json').read_text())
    verification=json.loads((OUT/'same_input_verification.json').read_text())
    assert verification['triangle_geometry_identical']
    cameras=json.loads((OUT/'evaluation_cameras.json').read_text())
    assert len(cameras)==98
    direct=[];paired=[]
    old=ROOT/'outputs/DTU/scan24/sobel_current'
    for budget in BUDGETS:
        folder=OUT/'evaluation/same_input'/str(budget)
        summary=json.loads((folder/'summary.json').read_text())
        assert json.loads((folder/'evaluation_cameras.json').read_text())==cameras
        original=json.loads((old/'evaluation'/str(budget)/'summary.json').read_text())
        for method in ('wild','sobel','realityscan'):
            if method!='realityscan':
                assert all(abs(summary['render_mean'][method][m]-original['render_mean'][method][m])<1e-7 for m in ('psnr','ssim','lpips'))
            direct.append(dict(target_faces=budget,method=method,
                actual_faces=summary['candidates'][method]['faces'],reference='ExMesh original 207856 faces',
                texture_side=1024,views=98,**summary['render_mean'][method]))
        with (folder/'per_view_metrics.csv').open() as f:
            per_view=[{k:float(v) if k not in ('view','source_view') else v for k,v in r.items()} for r in csv.DictReader(f)]
        comparisons=paired_summary(per_view,'realityscan',('sobel',),rgb_only=True)['sobel']
        for metric,s in comparisons.items():
            lo,hi=s['cluster_bootstrap_95ci']
            paired.append(dict(target_faces=budget,comparison='Sobel vs RealityScan, same ExMesh input',
                metric=metric,improvement_mean=s['improvement_mean'],ci95_low=lo,ci95_high=hi,source_view_win_rate=s['source_view_win_rate']))
    native_summary=json.loads((OUT/'evaluation/native_curve/summary.json').read_text())
    assert json.loads((OUT/'evaluation/native_curve/evaluation_cameras.json').read_text())==cameras
    native=[dict(target_faces=b,method='RealityScan reconstruction + native simplification',actual_faces=assets[f'RS_{b}']['faces'],
        reference='RealityScan original 476282 faces, 4096 texture',texture_side=1024,views=98,
        **native_summary['render_mean'][f'RS_{b}']) for b in BUDGETS]
    csv_write(D/'RealityScan_same_input_results.csv',direct)
    csv_write(D/'RealityScan_native_pipeline_results.csv',native)
    csv_write(D/'RealityScan_same_input_paired.csv',paired)
    (OUT/'report_data.json').write_text(json.dumps(dict(direct=direct,native=native,paired=paired),indent=2))
    for budget in BUDGETS:
        for mode in ('rgb','clay','wire'):
            for scope in ('native','same_input'):
                rs=f'RS_{budget}' if scope=='native' else f'RS_same_{budget}'
                labels=[('ExMesh + Wild',f'wild_{budget}'),('ExMesh + Sobel λ=4',f'sobel_{budget}'),
                    ('RealityScan + RS Simplify' if scope=='native' else 'ExMesh + RS Simplify',rs)]
                plate(D/f'RealityScan_{scope}_{budget}_{mode}.png',labels,mode,
                    f'{budget:,} faces / 0024 +5° / same camera and crop / '+('different reconstruction originals' if scope=='native' else 'same ExMesh original'))
    for mode in ('rgb','clay','wire'):
        plate(D/f'RealityScan_original_{mode}.png',[('ExMesh original','exmesh_original'),('RealityScan original','RS_original_fixed')],mode,
            'ExMesh 207,856 faces / RealityScan 476,282 faces / original textures 2560 and 4096',crop=mode!='rgb')
    text=['# RealityScan 복원·자체 Simplification 비교','',
        '동일 사진49장을 RealityScan2.2.0.119430에서 정렬하고 Normal quality로476,282 faces 원본을 생성했다. '
        '원본에서 RealityScan 자체 Simplification을 각각 독립 실행하여20,000·10,000·5,000 faces를 생성했다. '
        'Wild 코드를 RealityScan 메시의 단순화에 사용하지 않았다.','',
        '## 비교를 구분한 이유','',
        '1. 전체 파이프라인: ExMesh+Wild, ExMesh+Sobelλ4, RealityScan 복원+RealityScan Simplification을 동일 카메라에서 RGB·단색·wireframe으로 비교했다.',
        '2. 동일 입력 대조군: 같은 ExMesh 원본을 RealityScan에도 가져와 자체 Simplification을 적용했다. 이때만 공통 ExMesh GT로 PSNR/SSIM/LPIPS를 직접 비교한다.',
        '3. RealityScan 자체 보존성: RealityScan의 고해상도 원본을 GT로 두고 세 예산의 출력 손실을 계산했다. 다른 GT를 사용한 ExMesh 점수와의 직접 순위 비교는 하지 않는다.','',
        '## 동일 ExMesh 입력 결과','',
        '| 목표 faces | 방법 | 실제 faces | PSNR↑ | SSIM↑ | LPIPS↓ |','|---:|:---|---:|---:|---:|---:|']
    for r in direct:text.append(f'|{r["target_faces"]:,}|{NAMES[r["method"]]}|{r["actual_faces"]:,}|{r["psnr"]:.4f}|{r["ssim"]:.6f}|{r["lpips"]:.6f}|')
    text+=['','같은 원본 비교에서 Sobelλ4는 RealityScan 대조군보다 세 예산 모두 PSNR·SSIM·LPIPS가 좋았다. '
           '이 결과는 단순화 기하뿐 아니라 UV·텍스처 전달 방식의 차이도 포함한다. QEM 비용 항만의 우위라고 해석할 수 없다.','',
           '## RealityScan 자체 원본 대비 보존성','',
           '| 목표/실제 faces | PSNR↑ | SSIM↑ | LPIPS↓ |','|---:|---:|---:|---:|']
    for r in native:text.append(f'|{r["actual_faces"]:,}|{r["psnr"]:.4f}|{r["ssim"]:.6f}|{r["lpips"]:.6f}|')
    text+=['','이 표의 GT는 RealityScan 원본이다. 위 표의 GT는 ExMesh 원본이므로 두 표의 절대 점수로 전체 파이프라인의 승자를 정할 수 없다.','',
        '## 조건과 검증','',
        '- RealityScan 자체 SfM 정렬. 동일 카메라 촬영 조건을 반영하여 보정 그룹1개. Normal quality, depth downscale2. ExMesh COLMAP 카메라를 RealityScan의 복원 카메라로 대체하지 않았다.',
        '- 카메라49개의 위치로 하나의 similarity transform(회전·균일 scale·이동)을 계산했다. 메시 ICP나 비선형 변형을 사용하지 않았다. 카메라 중심 분포 RMS 반경 대비 정합오차0.2758%.',
        '- 복원 region은 ExMesh 원본 bbox를10% 확장한 범위를 RealityScan 좌표로 변환하여 감싸는 RS 축 정렬 박스다. 배경 조각도 복원될 수 있으며 정답 geometry 자체를 RealityScan에 주입한 것은 아니다.',
        '- 동일 입력 대조군의 RealityScan 재출력 원본은207,856 faces이며 삼각형별 형상이 원본과 일치한다. 최대 정점 위치 차이5.67e-8.',
        '- ExMesh2560 원본 텍스처는 직접 가져온 상태의 RealityScan 재투영 오류 때문에 RealityScan 자체 export를 통해4096 power-of-two 텍스처로 변환하여 재가져왔다. 형상 검증을 통과했다. 이 전처리도 대조군 UV/텍스처 파이프라인의 일부다.',
        '- RealityScan 내부 텍스처는 원본 해상도를 유지했다. 최종 파일은 RealityScan 자체 export의 oneTextureMaxSide=1024, exportToOneTexture=1로 제한했다. 모든 최종 diffuse PNG가1024×1024임을 확인했다. Wild/Sobel은xatlas1024/padding2다. UV 배열, 유효 gutter, 텍스처 필터링까지 동일한 실험은 아니다.',
        '- 98개의 가까운 새 시점: 원래49카메라에서 각각±5°. 이전 Wild/Sobel 평가 카메라를 그대로 재사용했다. 777×581, 검은 배경, 같은 ExMesh bbox crop+16, CPU raycast/bilinear texture, LPIPS VGG.',
        '- 같은 입력 대조군의 RealityScan 실제 수는19,991·9,990·4,988 faces다. 최대0.24% 목표 차이를 명시했다. 자체 복원 원본의 단순화는 정확히20,000·10,000·5,000 faces다.',
        '- 기존 Wild/Sobel 평균 지표를 이번 evaluator로 다시 계산했으며 기존 결과와1e-7 이내에서 일치했다.',
        '- paired CI는원래49카메라를 cluster로 묶어2,000회 bootstrap했다. 같은 장면에서 Sobelλ를 이미 선택했으므로 독립 장면 성능 증거가 아니다.','',
        '## 시각적 해석','',
        '원본 비교에서 RealityScan과 ExMesh의 벽 표면 및 창문 주변 형상부터 다르다. 선택한 창문 영역에서 RealityScan 원본은 더 매끈한 벽 부분을 보인다. '
        '5천 faces에서는 두 파이프라인 모두 창문 주변 형상을 단순화한다. Sobel이 모든 창문 형태를 항상 더 잘 보존한다고 결론 내릴 수 없다. '
        '발표에서는 동일 입력 RGB 보존 결과와 전체 복원 결과의 시각 비교를 구분해 설명하는 것이 정확하다.','',
        '## 자료','',
        '- RealityScan_native_pipeline_results.csv: 자체 원본 대비 세 예산 결과',
        '- RealityScan_same_input_results.csv / RealityScan_same_input_paired.csv: 공통 ExMesh GT의 직접 비교와 CI',
        '- RealityScan_native_<budget>_<rgb/clay/wire>.png: 전체 파이프라인 확대 비교',
        '- RealityScan_same_input_<budget>_<rgb/clay/wire>.png: 같은 ExMesh 원본 확대 비교',
        '- outputs/DTU/scan24/realityscan_comparison: 원본/출력OBJ, MTL, 텍스처, 프로젝트, 모든 CLI 명령·로그·카메라·좌표 변환·평가','',
        '[RealityScan CLI 공식 문서](https://rshelp.capturingreality.com/en-US/appbasics/allcommands.htm) '
        '[Simplification 공식 문서](https://rshelp.capturingreality.com/en-US/tools/simplify.htm)']
    (D/'RealityScan_비교_보고서.md').write_text('\n'.join(text),encoding='utf-8')
    records={k:{'obj_sha256':v['aligned_obj_sha256'],'texture_sha256':v['texture_sha256'],'faces':v['faces']} for k,v in assets.items()}
    (D/'RealityScan_mesh_manifest.json').write_text(json.dumps(records,indent=2))
    with zipfile.ZipFile(D/'RealityScan_비교용_메시.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in ('RS_20000','RS_10000','RS_5000','RS_same_20000','RS_same_10000','RS_same_5000'):
            data=assets[name];p=Path(data['mesh']);folder='native_pipeline' if not name.startswith('RS_same_') else 'same_ExMesh_input'
            for f in (p,p.with_suffix('.mtl'),Path(data['texture'])):z.write(f,f'{folder}/{f.name}')
        z.write(D/'RealityScan_mesh_manifest.json','manifest.json')
        z.write(OUT/'coordinate_alignment.json','coordinate_alignment.json')
    print(json.dumps(dict(direct=direct,native=native,paired=paired),indent=2),flush=True)

if __name__=='__main__':main()
