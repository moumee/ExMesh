"""RealityScan native reconstruction/simplification, with reproducible CLI logs."""
import argparse, hashlib, json, shutil, subprocess, time, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/DTU/scan24/realityscan_comparison'
EXE = Path('D:/EpicGames/RealityScan_2.2/RealityScan.exe')
PROJECT = OUT/'scan24_grouped.rsproj'
SETTINGS = {
    'appQuitOnError': 'true', 'suppressErrors': 'true',
    'appAutoSaveMode': 'false', 'appIncSubdirs': 'false',
    'sfmFeatureDetectionQuality': 'High', 'sfmImageDownscaleFactor': '1',
    'sfmMaxFeaturesPerMpx': '10000', 'sfmMaxFeaturesPerImage': '40000',
    'sfmDistortionModel': 'Brown3', 'sfmImagesOverlap': 'Medium',
    'mvsNormalDownscaleFactor': '2', 'MvsGeometryGpuAccel': 'true',
    'mvsDecimationFactor': '1',
    'unwrapStyle': 'MaxTexturesCount', 'unwrapMaximalTexCount': '1',
    'unwrapMinTexResolution': '1024', 'unwrapMaxTexResolution': '1024',
    'unwrapGutter': '2', 'txtImageDownscaleTexture': '1',
}

def sha256(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def invoke(stage, commands):
    """Run a dedicated instance; never delegate to an existing user project."""
    (OUT/'logs').mkdir(parents=True, exist_ok=True)
    (OUT/'crash').mkdir(exist_ok=True)
    command = [str(EXE), '-headless', '-silent', str(OUT/'crash'), '-stdConsole',
               '-writeProgress', str(OUT/'logs'/f'{stage}_progress.csv'), '10']
    if stage == 'align':
        command += ['-exportGlobalSettings', str(OUT/'original_settings.rcconfig')]
    for key, value in SETTINGS.items():
        command += ['-set', f'{key}={value}']
    command += commands
    command += ['-exportGlobalSettings', str(OUT/f'{stage}_settings.rcconfig'),
                '-importGlobalSettings', str(OUT/'original_settings.rcconfig'), '-quit']
    job = {'stage': stage, 'command': command, 'started': time.time()}
    (OUT/'logs'/f'{stage}_job.json').write_text(json.dumps(job, indent=2))
    with (OUT/'logs'/f'{stage}.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        job['pid'] = process.pid
        (OUT/'logs'/f'{stage}_job.json').write_text(json.dumps(job, indent=2))
        print(json.dumps({'stage': stage, 'pid': process.pid}), flush=True)
        code = process.wait()
    job.update(exit_code=code, finished=time.time())
    contents=(OUT/'logs'/f'{stage}.log').read_text(encoding='utf-8',errors='replace')
    job['reported_failure']=('failed after' in contents.lower() or 'was suppressed' in contents.lower() and 'The error' in contents)
    (OUT/'logs'/f'{stage}_job.json').write_text(json.dumps(job, indent=2))
    if code or job['reported_failure']:
        raise RuntimeError(f'RealityScan {stage} exit {code}; inspect logs/{stage}.log')

def align():
    images = sorted((ROOT/'workdir/DTU/scan24/images').glob('*.png'))
    assert len(images) == 49
    (OUT/'images').mkdir(parents=True, exist_ok=True)
    records = []
    for p in images:
        dest = OUT/'images'/p.name
        if not dest.exists():
            shutil.copy2(p, dest)
        assert sha256(p) == sha256(dest)
        records.append({'name': p.name, 'sha256': sha256(p)})
    (OUT/'inputs.json').write_text(json.dumps({'images': records,
        'pipeline': 'RealityScan align -> native normal-quality MVS mesh -> native simplify',
        'budgets': [20000, 10000, 5000], 'settings': SETTINGS}, indent=2))
    assert not PROJECT.exists(), 'Existing project: resume with a subsequent stage.'
    invoke('align', ['-newScene', '-addFolder', str(OUT/'images'), '-align',
        '-selectMaximalComponent', '-exportXMPForSelectedComponent',
        '-setReconstructionRegionAuto', '-exportReconstructionRegion', str(OUT/'auto_region.rsbox'),
        '-save', str(PROJECT)])
    assert PROJECT.exists()
    assert len(list((OUT/'images').glob('*.xmp'))) >= 3

def align_grouped():
    folder=OUT/'images_grouped'
    folder.mkdir(exist_ok=True)
    for p in (ROOT/'workdir/DTU/scan24/images').glob('*.png'):
        dest=folder/p.name
        if not dest.exists():shutil.copy2(p,dest)
        assert sha256(p)==sha256(dest)
    assert not PROJECT.exists()
    invoke('align_grouped',['-newScene','-addFolder',str(folder),
        '-selectAllImages','-setConstantCalibrationGroups','-align',
        '-selectMaximalComponent','-exportXMPForSelectedComponent',
        '-setReconstructionRegionAuto','-exportReconstructionRegion',str(OUT/'auto_region.rsbox'),
        '-save',str(PROJECT)])

def mesh():
    assert PROJECT.exists()
    assert (OUT/'evaluation_region.rsbox').exists()
    (OUT/'raw').mkdir(exist_ok=True)
    invoke('mesh',['-load',str(PROJECT),'-selectMaximalComponent',
        '-setReconstructionRegion',str(OUT/'evaluation_region.rsbox'),
        '-calculateNormalModel','-renameSelectedModel','RS_original_fixed',
        '-set','unwrapMinTexResolution=4096','-set','unwrapMaxTexResolution=4096',
        '-unwrap','-calculateTexture','-save',str(PROJECT),
        '-exportSelectedModel',str(OUT/'raw/RS_original_fixed.obj')])

def simplify():
    info=OUT/'raw/RS_original_fixed.obj.rsInfo'
    tree=ET.fromstring('<root>'+info.read_text()+'</root>')
    export=tree.find('ModelExport')
    assert export is not None
    model=tree.find('Model')
    assert model.attrib['transformToModel']=='1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1', 'Export frame changed'
    ET.ElementTree(export).write(OUT/'export_settings.xml',encoding='utf-8',xml_declaration=True)
    commands=['-load',str(PROJECT)]
    # Loading a project restores its unwrap settings; apply final texture
    # budget after loading, before creating any simplified model.
    for key,value in SETTINGS.items():
        if key.startswith('unwrap'):commands+=['-set',f'{key}={value}']
    for budget in (20000,10000,5000):
        name=f'RS_{budget}_1024'
        commands += ['-selectModel','RS_original_fixed','-simplify',str(budget),
            '-renameSelectedModel',name,'-unwrap',
            '-reprojectTexture','RS_original_fixed',name,
            '-exportSelectedModel',str(OUT/'raw'/f'RS_{budget}.obj'),str(OUT/'export_settings.xml'),
            '-save',str(PROJECT)]
    invoke('simplify',commands)
    for budget in (20000,10000,5000):
        p=OUT/'raw'/f'RS_{budget}.obj'
        faces=sum(line.startswith('f ') for line in p.open())
        assert abs(faces-budget)<=2,(faces,budget)
    print('Native RealityScan simplification completed: 20000, 10000, 5000',flush=True)

def export_final():
    tree=ET.fromstring('<root>'+(OUT/'raw/RS_original_fixed.obj.rsInfo').read_text()+'</root>')
    export=tree.find('ModelExport')
    export.set('exportToOneTexture','1');export.set('oneTextureMaxSide','1024')
    export.set('shrinkTextures','1')
    # Export the color texture, excluding the unwrap-checker image layer.
    diffuse=next(child for child in export if child.tag.startswith('Layer') and child.attrib.get('type')=='1')
    for child in list(export):
        if child.tag.startswith('Layer'):export.remove(child)
    diffuse.tag='Layer0';export.append(diffuse);export.set('exportedLayerCount','1')
    params=OUT/'export_1024_settings.xml'
    ET.ElementTree(export).write(params,encoding='utf-8',xml_declaration=True)
    commands=['-load',str(PROJECT)]
    for budget in (20000,10000,5000):
        commands+=['-selectModel',f'RS_{budget}_1024','-exportSelectedModel',
            str(OUT/'raw'/f'RS_{budget}.obj'),str(params)]
    invoke('export_final',commands)

def same_input():
    """Additional native simplification control on the exact ExMesh source."""
    import numpy as np
    source=ROOT/'outputs/DTU/scan24/export/simp_mod/mesh_iter_10000.obj'
    folder=OUT/'same_input';folder.mkdir(exist_ok=True)
    dest=folder/source.name
    align=json.loads((OUT/'coordinate_alignment.json').read_text())
    rotation=np.array(align['rotation']);scale=align['scale'];translation=np.array(align['translation'])
    with source.open() as reader,dest.open('w') as writer:
        for line in reader:
            if line.startswith('v '):
                v=(np.array(list(map(float,line.split()[1:4])))-translation)@rotation/scale
                line='v '+' '.join(f'{x:.12g}' for x in v)+'\n'
            elif line.startswith('vn '):
                v=np.array(list(map(float,line.split()[1:4])))@rotation
                line='vn '+' '.join(f'{x:.12g}' for x in v)+'\n'
            writer.write(line)
    shutil.copy2(source.with_suffix('.mtl'),dest.with_suffix('.mtl'))
    shutil.copy2(source.with_suffix('.png'),dest.with_suffix('.png'))
    params=OUT/'export_1024_settings.xml'
    commands=['-load',str(PROJECT),'-selectMaximalComponent','-importModel',str(dest),
              '-renameSelectedModel','ExMesh_imported']
    for budget in (20000,10000,5000):
        name=f'RS_same_{budget}'
        commands+=['-selectModel','ExMesh_imported','-simplify',str(budget),
            '-renameSelectedModel',name,'-unwrap','-reprojectTexture','ExMesh_imported',name,
            '-exportSelectedModel',str(OUT/'raw'/f'{name}.obj'),str(params)]
    commands+=['-save',str(PROJECT)]
    invoke('same_input',commands)

def verify_import():
    invoke('verify_import',['-load',str(PROJECT),'-selectModel','ExMesh_imported',
        '-exportSelectedModel',str(OUT/'raw/ExMesh_imported_reference.obj'),str(OUT/'export_1024_settings.xml')])

def normalize_import():
    export=ET.parse(OUT/'export_1024_settings.xml').getroot()
    export.set('oneTextureMaxSide','4096');export.set('shrinkTextures','0')
    params=OUT/'export_4096_settings.xml'
    ET.ElementTree(export).write(params,encoding='utf-8',xml_declaration=True)
    invoke('normalize_import',['-load',str(PROJECT),'-selectModel','ExMesh_imported',
        '-exportSelectedModel',str(OUT/'raw/ExMesh_power2.obj'),str(params)])

def same_input_retry():
    params=OUT/'export_1024_settings.xml'
    commands=['-load',str(PROJECT),'-selectMaximalComponent','-importModel',str(OUT/'raw/ExMesh_power2.obj'),
        '-renameSelectedModel','ExMesh_power2']
    for budget in (20000,10000,5000):
        name=f'RS_same_power2_{budget}'
        commands+=['-selectModel','ExMesh_power2','-simplify',str(budget),
            '-renameSelectedModel',name,'-unwrap','-reprojectTexture','ExMesh_power2',name,
            '-exportSelectedModel',str(OUT/'raw'/f'RS_same_{budget}.obj'),str(params)]
    commands+=['-save',str(PROJECT)]
    invoke('same_input_retry',commands)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    stages={'align':align,'align_grouped':align_grouped,'mesh':mesh,'simplify':simplify,'export_final':export_final,
            'same_input':same_input,'verify_import':verify_import,'normalize_import':normalize_import,'same_input_retry':same_input_retry}
    parser.add_argument('stage', choices=list(stages))
    args = parser.parse_args()
    stages[args.stage]()
