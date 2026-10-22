"""Inspect legacy sparse and new dense runs without PyCOLMAP.

Outputs in experiments/<scene>/<run>/inspections/review_<timestamp>/
"""
import argparse
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLMAP = ROOT/'tools'/'COLMAP'/'COLMAP.bat'
RUN_RE = re.compile(r'([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)')
ALIASES = {'gerrad-hall': 'gerrard-hall', 'garred-hall': 'gerrard-hall'}


def ply_counts(p):
    if not p.is_file():
        return {'exists': False}
    stats = {'exists': True, 'bytes': p.stat().st_size, 'path': str(p)}
    with p.open('rb') as f:
        if f.readline().strip() != b'ply':
            stats['parse_error'] = 'Invalid PLY header'
            return stats
        for _ in range(500):
            line = f.readline()
            if not line or line.strip() == b'end_header':
                break
            a = line.split()
            if len(a) == 3 and a[0] == b'element' and a[1] in (b'vertex', b'face'):
                stats[a[1].decode()+'_count'] = int(a[2])
    return stats


def glb_counts(p):
    if not p.is_file():
        return {'exists': False}
    import struct
    stats = {'exists': True, 'bytes': p.stat().st_size, 'path': str(p)}
    with p.open('rb') as f:
        h = f.read(12)
        if h[:4] != b'glTF':
            stats['parse_error'] = 'Invalid GLB'
            return stats
        count, t = struct.unpack('<I4s', f.read(8))
        if t != b'JSON' or count > 50_000_000:
            stats['parse_error'] = 'Invalid JSON chunk'
            return stats
        doc = json.loads(f.read(count).decode('utf-8').strip(' \x00'))
    stats['meshes'] = len(doc.get('meshes', []))
    stats['materials'] = len(doc.get('materials', []))
    stats['textures'] = len(doc.get('textures', []))
    stats['cameras'] = len(doc.get('cameras', []))
    accessors = doc.get('accessors', [])
    vert = faces = 0
    for mesh in doc.get('meshes', []):
        for pri in mesh.get('primitives', []):
            pi = pri.get('attributes', {}).get('POSITION')
            if pi is not None:
                vert += accessors[pi]['count']
            ii = pri.get('indices')
            count = accessors[ii]['count'] if ii is not None else (accessors[pi]['count'] if pi is not None else 0)
            if pri.get('mode', 4) == 4:
                faces += count // 3
    stats['mesh_vertex_samples'] = vert
    stats['triangle_faces'] = faces
    return stats


def model_converter(model, out, typ):
    import subprocess
    cmd = ['cmd.exe','/d','/c',str(COLMAP), 'model_converter','--input_path',str(model),
           '--output_path',str(out),'--output_type',typ]
    try:
        p = subprocess.run(cmd, cwd=ROOT, text=True, encoding='utf-8', errors='replace',
                           capture_output=True, timeout=180)
        return p.returncode, subprocess.list2cmdline(cmd) + '\n' + p.stdout + p.stderr
    except Exception as e:
        return -1, f'Could not run COLMAP converter: {e}'


def parse_registered(path):
    if not path.is_file():
        return []
    registered = []
    lines = path.read_text(encoding='utf-8',errors='replace').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and not line.startswith('#'):
            f = line.split()
            if len(f) >= 10 and re.fullmatch(r'\d+', f[0]):
                registered.append(f[9]); i += 2; continue
        i += 1
    return registered


def inspect_one(group, run_id, dataset_label, stamp, summary_only=False):
    exp = ROOT/'experiments'/group/run_id
    if not (exp/'summary.json').is_file():
        raise ValueError(f'No experiment summary: {exp}')
    co = ROOT/'colmap_output'/group/run_id
    ma = ROOT/'mapanything_output'/group/run_id
    folder = exp/'inspections'/f'review_{stamp}'
    folder.mkdir(parents=True)
    summary = json.loads((exp/'summary.json').read_text(encoding='utf-8'))
    report = {'dataset': dataset_label, 'group': group, 'run': run_id,
              'run_ref': f'{group}/{run_id}',
              'input_images': summary.get('input_images'),
              'input_fingerprint_sha256': summary.get('input_fingerprint_sha256'),
              'mapanything': glb_counts(ma/'scene.glb'),
              'colmap_dense_cloud': ply_counts(co/'dense'/'fused.ply'),
              'colmap_dense_mesh': ply_counts(co/'dense'/'meshed-poisson.ply'),
              'summary': summary}
    sparse = co/'sparse'
    models = [d for d in sparse.iterdir() if d.is_dir() and (d/'cameras.bin').is_file()] if sparse.is_dir() else []
    reported = str(summary.get('colmap',{}).get('sparse_model') or summary.get('colmap',{}).get('output') or '')
    model = next((m for m in models if m.name == reported.replace('\\','/').rstrip('/').split('/')[-1]), None)
    if model is None and models:
        model = sorted(models)[0]
    report['sparse_model_path'] = str(model) if model else None
    if model and COLMAP.is_file() and not summary_only:
        txt = folder/'colmap_text';txt.mkdir()
        rc,log = model_converter(model,txt,'TXT')
        (folder/'model_to_text.log').write_text(log,encoding='utf-8')
        registered = parse_registered(txt/'images.txt') if rc == 0 else []
        # Do not claim unregistered images unless COLMAP text conversion succeeded.
        if rc == 0:
            report['registered_filenames'] = registered
            manifest = exp/'input_manifest.csv'
            if manifest.is_file():
                with manifest.open(encoding='utf-8-sig', newline='') as f:
                    imgs = [r['filename'] for r in csv.DictReader(f)]
                report['unregistered_filenames'] = sorted(set(imgs)-set(registered))
        sparse_ply = folder/'colmap_sparse.ply'
        rc,log = model_converter(model,sparse_ply,'PLY')
        (folder/'model_to_ply.log').write_text(log,encoding='utf-8')
        report['colmap_sparse_ply'] = ply_counts(sparse_ply)
    (folder/'inspection.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    summary_md = [f'# Inspection - {group}/{run_id}', '',
        f"MapAnything GLB: {report['mapanything']}", '',
        f"COLMAP dense fused PLY: {report['colmap_dense_cloud']}", '',
        f"COLMAP Poisson mesh: {report['colmap_dense_mesh']}", '',
        f"COLMAP registered names: {report.get('registered_filenames','unavailable')}", '',
        f"COLMAP unregistered names: {report.get('unregistered_filenames','unavailable')}", '',
        'Point counts, mesh faces, output size and reprojection errors are NOT metric accuracy scores.']
    (folder/'inspection.md').write_text('\n'.join(summary_md)+'\n',encoding='utf-8')
    print('\n'.join(summary_md));print('Saved:',folder)
    colmap = summary.get('colmap', {})
    return {
        'run_ref': f'{group}/{run_id}', 'dataset': summary.get('dataset', dataset_label),
        'input_images': summary.get('input_images'),
        'input_fingerprint_sha256': summary.get('input_fingerprint_sha256'),
        'mapanything_status': summary.get('mapanything', {}).get('status'),
        'mapanything_vertices': report['mapanything'].get('mesh_vertex_samples'),
        'mapanything_triangles': report['mapanything'].get('triangle_faces'),
        'colmap_status': colmap.get('status'),
        'registered_images': colmap.get('registered_images'),
        'registration_rate_percent': colmap.get('registration_rate_percent'),
        'sparse_points': colmap.get('sparse_points'),
        'mean_reprojection_error_px': colmap.get('mean_reprojection_error_px'),
        'dense_points': report['colmap_dense_cloud'].get('vertex_count'),
        'mesh_faces': report['colmap_dense_mesh'].get('face_count'),
        'inspection_directory': str(folder),
    }


def main():
    p = argparse.ArgumentParser(description='Inspect one or more MapAnything/COLMAP runs')
    p.add_argument('--dataset', help='legacy single-run scene group, e.g. office_26')
    p.add_argument('--run', help='legacy single-run ID')
    p.add_argument('--runs', nargs='+', help='group/run references for batch inspection')
    p.add_argument('--summary-only', action='store_true',
                   help='Skip COLMAP text/PLY conversion and inspect existing outputs only')
    args = p.parse_args()
    if args.runs and (args.dataset or args.run):
        p.error('Use either --runs or the --dataset/--run pair, not both')
    refs = []
    if args.runs:
        for value in args.runs:
            match = RUN_RE.fullmatch(value.replace('\\', '/'))
            if not match:
                p.error(f'Invalid run reference: {value}; use group/run')
            group, run_id = match.groups()
            refs.append((ALIASES.get(group, group), run_id, group))
    else:
        if not args.dataset or not args.run:
            p.error('provide --runs or both --dataset and --run')
        group = ALIASES.get(args.dataset.removesuffix('_undistorted'),
                            args.dataset.removesuffix('_undistorted'))
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in (group, args.run)):
            p.error('Unsafe dataset or run name')
        refs.append((group, args.run, args.dataset))
    refs = list(dict.fromkeys(refs))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    rows = []
    try:
        for group, run_id, label in refs:
            rows.append(inspect_one(group, run_id, label, stamp, args.summary_only))
    except ValueError as exc:
        p.error(str(exc))
    if len(rows) > 1:
        batch = ROOT/'experiments'/'inspections'/f'batch_{stamp}'
        batch.mkdir(parents=True)
        fields = list(rows[0].keys())
        with (batch/'inspection_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        (batch/'inspection_summary.json').write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print('Batch inspection summary:', batch)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
