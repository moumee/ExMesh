"""Evaluate fixed 98 poses, distinguishing preservation and same-input tests."""
import argparse,json,subprocess,sys,time
from run_realityscan_compare import ROOT,OUT
from run_sobel_ablation import REF,TEX,CAM

CONTROL=ROOT/'outputs/DTU/scan24/sobel_current'

def evaluate(folder,reference,texture,candidates,baseline,tolerance):
    folder.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(ROOT/'simplification/evaluate_mesh_preservation.py'),
        '--protocol','novel','--skip_geometry','--rgb_only',
        '--reference',str(reference),'--texture',f'reference={texture}',
        '--cameras_json',str(CAM),'--evaluation_cameras_json',str(OUT/'evaluation_cameras.json'),
        '--crop_reference',str(REF),'--crop_texture',str(TEX),'--output_dir',str(folder),
        '--baseline_label',baseline,'--face_budget_tolerance',str(tolerance),'--resolution','2']
    for label,(mesh,tex) in candidates.items():
        command+=['--candidate',f'{label}={mesh}','--texture',f'{label}={tex}']
    start=time.time();(folder/'command.json').write_text(json.dumps(command,indent=2))
    print('[evaluate]',folder.name,flush=True)
    with (folder/'evaluation.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((folder/'summary.json').read_text())
    assert len(json.loads((folder/'evaluation_cameras.json').read_text()))==98
    report['comparison_scope']='Preservation versus own original' if folder.name=='native_curve' else 'Same ExMesh input and GT, different simplification/UV/texture pipelines'
    report['inference_caveat']='Single scan24, nearby novel poses. Different reconstruction originals cannot establish end-to-end ranking. Sobel lambda previously selected on this scene.'
    report['evaluation_seconds']=time.time()-start
    (folder/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['render_mean'],indent=2),flush=True)

def native():
    assets=json.loads((OUT/'assets.json').read_text());ref=assets['RS_original_fixed']
    candidates={f'RS_{b}':(assets[f'RS_{b}']['mesh'],assets[f'RS_{b}']['texture']) for b in (20000,10000,5000)}
    # Intentional budget curve on one pipeline, not equal-budget method ranking.
    evaluate(OUT/'evaluation/native_curve',ref['mesh'],ref['texture'],candidates,'RS_20000',15000)

def same_input():
    assets=json.loads((OUT/'assets.json').read_text())
    assert json.loads((OUT/'same_input_verification.json').read_text())['triangle_geometry_identical']
    for b in (20000,10000,5000):
        candidates={}
        for method in ('wild','sobel'):
            p=CONTROL/'meshes'/f'{method}_{b}.obj'
            candidates[method]=(p,p.with_name(p.stem+'_texture.png'))
        rs=assets[f'RS_same_{b}'];candidates['realityscan']=(rs['mesh'],rs['texture'])
        evaluate(OUT/'evaluation/same_input'/str(b),REF,TEX,candidates,'wild',20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['native','same_input'])
    args=p.parse_args();{'native':native,'same_input':same_input}[args.stage]()
