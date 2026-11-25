"""Inspect saved generic reconstruction runs on either workstation, without PyCOLMAP.

Outputs in experiments/<scene>/<run>/inspections/review_<timestamp>/
"""
import argparse
import csv
import json
import re
import struct
import subprocess
from datetime import datetime
from pathlib import Path
from reconstruction_pipeline import colmap_command, command_display, validate_component

ROOT = Path(__file__).resolve().parent.parent
COLMAP = ROOT/'tools'/'COLMAP'/'COLMAP.bat'


def parse_run_reference(value):
    parts = value.replace('\\', '/').split('/')
    if len(parts) != 2:
        raise ValueError('Run reference must contain exactly two folder names: group/run')
    group, run_id = parts
    validate_component(group, 'scene group')
    validate_component(run_id, 'run ID')
    return group, run_id


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('Truncated artifact record')
    return data


def ply_counts(p):
    if not p.is_file():
        return {'exists': False, 'geometry_state': 'missing'}
    stats = {'exists': True, 'bytes': p.stat().st_size, 'path': str(p)}
    types = {'char': 'b', 'int8': 'b', 'uchar': 'B', 'uint8': 'B',
             'short': 'h', 'int16': 'h', 'ushort': 'H', 'uint16': 'H',
             'int': 'i', 'int32': 'i', 'uint': 'I', 'uint32': 'I',
             'float': 'f', 'float32': 'f', 'double': 'd', 'float64': 'd'}
    elements, format_name, header_bytes = [], None, 0
    with p.open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('Invalid PLY header')
        while True:
            line = f.readline(100_001)
            header_bytes += len(line)
            if not line or len(line) > 100_000 or header_bytes > 1_000_000:
                raise ValueError('Missing or invalid PLY end_header')
            fields = line.decode('ascii').split()
            if fields == ['end_header']:
                break
            if not fields or fields[0] in ('comment', 'obj_info'):
                continue
            if fields[0] == 'format':
                if len(fields) != 3 or fields[2] != '1.0':
                    raise ValueError('Unsupported PLY format/version')
                format_name = fields[1]
            elif fields[0] == 'element':
                if len(fields) != 3 or int(fields[2]) < 0:
                    raise ValueError('Invalid PLY element count')
                if any(element[0] == fields[1] for element in elements):
                    raise ValueError('Duplicate PLY element')
                elements.append((fields[1], int(fields[2]), []))
            elif fields[0] == 'property':
                if not elements:
                    raise ValueError('PLY property without an element')
                prop = fields[1:]
                if len(prop) == 2 and prop[0] in types:
                    elements[-1][2].append((None, types[prop[0]]))
                elif len(prop) == 4 and prop[0] == 'list' and prop[1] in types and prop[2] in types:
                    if types[prop[1]] in ('f', 'd'):
                        raise ValueError('PLY list count must be an integer')
                    elements[-1][2].append((types[prop[1]], types[prop[2]]))
                else:
                    raise ValueError('Unsupported PLY property')
        if format_name not in ('ascii', 'binary_little_endian', 'binary_big_endian'):
            raise ValueError('Missing or unsupported PLY format')
        if not any(element[0] == 'vertex' for element in elements):
            raise ValueError('PLY has no vertex element')
        endian = '<' if format_name == 'binary_little_endian' else '>'
        for name, count, properties in elements:
            if count and not properties:
                raise ValueError('PLY nonempty element has no properties')
            if name in ('vertex', 'face'):
                stats[name + '_count'] = count
            if format_name != 'ascii' and all(kind is None for kind, _ in properties):
                byte_count = count * sum(struct.calcsize(endian + kind) for _, kind in properties)
                if f.tell() + byte_count > stats['bytes']:
                    raise ValueError('Truncated PLY scalar records')
                f.seek(byte_count, 1)
                continue
            for _ in range(count):
                if format_name == 'ascii':
                    values, offset = f.readline().decode('ascii').split(), 0
                    for count_type, value_type in properties:
                        length = 1
                        if count_type is not None:
                            if offset >= len(values):
                                raise ValueError('Truncated PLY list count')
                            length = int(values[offset]); offset += 1
                        if length < 0 or offset + length > len(values):
                            raise ValueError('Truncated PLY ASCII record')
                        parse = float if value_type in ('f', 'd') else int
                        for value in values[offset:offset + length]:
                            parse(value)
                        offset += length
                    if offset != len(values):
                        raise ValueError('Unexpected PLY ASCII values')
                else:
                    for count_type, value_type in properties:
                        length = (struct.unpack(endian + count_type, read_exact(
                            f, struct.calcsize(endian + count_type)))[0] if count_type is not None else 1)
                        byte_count = length * struct.calcsize(endian + value_type)
                        if length < 0 or f.tell() + byte_count > stats['bytes']:
                            raise ValueError('Truncated PLY list record')
                        f.seek(byte_count, 1)
    vertices, faces = stats.get('vertex_count', 0), stats.get('face_count')
    stats['geometry_state'] = 'nonempty' if vertices > 0 and (faces is None or faces > 0) else 'empty'
    stats['validation_scope'] = 'PLY record sizes/ASCII values; topology, finiteness and geometric accuracy are not validated'
    return stats


def glb_counts(p):
    if not p.is_file():
        return {'exists': False, 'geometry_state': 'missing'}
    stats = {'exists': True, 'bytes': p.stat().st_size, 'path': str(p)}
    with p.open('rb') as f:
        magic, version, total = struct.unpack('<4sII', read_exact(f, 12))
        if magic != b'glTF' or version != 2 or total != stats['bytes']:
            raise ValueError('Invalid GLB version or declared file length')
        count, kind = struct.unpack('<I4s', read_exact(f, 8))
        if kind != b'JSON' or count % 4 or count > 50_000_000 or f.tell() + count > total:
            raise ValueError('Invalid or truncated GLB JSON chunk')
        doc = json.loads(read_exact(f, count).decode('utf-8').strip(' \x00'))
        while f.tell() < total:
            count, kind = struct.unpack('<I4s', read_exact(f, 8))
            if kind == b'JSON' or count % 4 or f.tell() + count > total:
                raise ValueError('Invalid or truncated GLB additional chunk')
            f.seek(count, 1)
    if not isinstance(doc, dict):
        raise ValueError('GLB JSON root must be an object')
    for field in ('meshes', 'materials', 'textures', 'cameras', 'accessors'):
        if not isinstance(doc.get(field, []), list):
            raise ValueError(f'Invalid GLB {field} array')
    stats['meshes'] = len(doc.get('meshes', []))
    stats['materials'] = len(doc.get('materials', []))
    stats['textures'] = len(doc.get('textures', []))
    stats['cameras'] = len(doc.get('cameras', []))
    accessors = doc.get('accessors', [])

    def accessor_count(index):
        if type(index) is not int or not 0 <= index < len(accessors):
            raise ValueError('Invalid GLB accessor index')
        accessor = accessors[index]
        count = accessor.get('count') if isinstance(accessor, dict) else None
        if type(count) is not int or count < 0:
            raise ValueError('Invalid GLB accessor count')
        return count

    vert = faces = 0
    for mesh in doc.get('meshes', []):
        if not isinstance(mesh, dict) or not isinstance(mesh.get('primitives', []), list):
            raise ValueError('Invalid GLB mesh')
        for pri in mesh.get('primitives', []):
            if not isinstance(pri, dict) or not isinstance(pri.get('attributes', {}), dict):
                raise ValueError('Invalid GLB primitive')
            pi = pri.get('attributes', {}).get('POSITION')
            if pi is not None:
                vert += accessor_count(pi)
            ii = pri.get('indices')
            count = accessor_count(ii) if ii is not None else (accessor_count(pi) if pi is not None else 0)
            if pri.get('mode', 4) == 4:
                faces += count // 3
            elif pri.get('mode') in (5, 6):
                faces += max(0, count - 2)
    stats['mesh_vertex_samples'] = vert
    stats['triangle_faces'] = faces
    stats['geometry_state'] = 'nonempty' if vert > 0 else 'empty'
    stats['validation_scope'] = 'GLB v2 container/accessor counts; buffer values, scene instancing and geometric accuracy are not validated'
    return stats


def checked_counts(reader, path):
    try:
        return reader(path)
    except (OSError, ValueError, struct.error, KeyError, IndexError, TypeError) as exc:
        return {'exists': path.is_file(), 'path': str(path), 'geometry_state': 'invalid', 'parse_error': str(exc)}


def mesh_counts(path):
    result = ply_counts(path)
    if result.get('geometry_state') == 'nonempty' and not result.get('face_count', 0):
        result['geometry_state'] = 'empty'
        result['note'] = 'Points exist, but there are no faces; this is not a nonempty surface mesh'
    return result


def model_converter(model, out, typ):
    try:
        cmd = colmap_command('model_converter','--input_path',str(model),
                             '--output_path',str(out),'--output_type',typ)
        p = subprocess.run(cmd, cwd=ROOT, text=True, encoding='utf-8', errors='replace',
                           capture_output=True, timeout=180)
        return p.returncode, command_display(cmd) + '\n' + p.stdout + p.stderr
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return -1, f'Could not run COLMAP converter: {e}'


def parse_registered(path):
    if not path.is_file():
        raise FileNotFoundError(f'No image records: {path}')
    registered = []
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and not line.startswith('#'):
            f = line.split(maxsplit=9)
            if len(f) != 10 or not re.fullmatch(r'\d+', f[0]):
                raise ValueError('Invalid COLMAP text image record')
            [float(value) for value in f[1:8]]
            int(f[8])
            if i + 1 >= len(lines) or len(lines[i + 1].split()) % 3:
                raise ValueError('Truncated or invalid COLMAP text observations')
            registered.append(f[9]); i += 2; continue
        i += 1
    return registered


def registered_names(model):
    binary = model / 'images.bin'
    if not binary.is_file():
        return parse_registered(model / 'images.txt')
    names = []
    with binary.open('rb') as stream:
        size = binary.stat().st_size
        count, = struct.unpack('<Q', read_exact(stream, 8))
        for _ in range(count):
            read_exact(stream, 64)
            name = bytearray()
            while True:
                char = read_exact(stream, 1)
                if char == b'\0':
                    break
                name.extend(char)
                if len(name) > 1_000_000:
                    raise ValueError('Invalid COLMAP image filename length')
            names.append(name.decode('utf-8'))
            observations, = struct.unpack('<Q', read_exact(stream, 8))
            if stream.tell() + observations * 24 > size:
                raise ValueError('Truncated binary image observations')
            stream.seek(observations * 24, 1)
        if stream.read(1):
            raise ValueError('Unexpected bytes after COLMAP image records')
    return names


def model_image_count(model):
    try:
        return len(registered_names(model))
    except (OSError, ValueError, struct.error):
        return -1


def inspect_one(group, run_id, dataset_label, stamp, summary_only=False):
    validate_component(group, 'scene group')
    validate_component(run_id, 'run ID')
    exp = ROOT/'experiments'/group/run_id
    if not (exp/'summary.json').is_file():
        raise ValueError(f'No experiment summary: {exp}')
    co = ROOT/'colmap_output'/group/run_id
    ma = ROOT/'mapanything_output'/group/run_id
    summary = json.loads((exp/'summary.json').read_text(encoding='utf-8-sig'))
    if not isinstance(summary, dict):
        raise ValueError('Experiment summary must be a JSON object')
    report = {'dataset': summary.get('dataset', dataset_label), 'group': group, 'run': run_id,
              'run_ref': f'{group}/{run_id}',
              'input_images': summary.get('input_images'),
              'input_fingerprint_sha256': summary.get('input_fingerprint_sha256'),
              'mapanything': checked_counts(glb_counts, ma/'scene.glb'),
              'colmap_dense_cloud': checked_counts(ply_counts, co/'dense'/'fused.ply'),
              'colmap_dense_mesh': checked_counts(mesh_counts, co/'dense'/'meshed-poisson.ply'),
              'summary': summary, 'conversion_status': 'skipped_summary_only' if summary_only else 'unavailable'}
    sparse = co/'sparse'
    paths = [sparse, *sorted(sparse.iterdir())] if sparse.is_dir() else []
    models = [d for d in paths if d.is_dir() and
              ((d/'images.bin').is_file() or (d/'images.txt').is_file())]
    reported = str((summary.get('colmap') or {}).get('sparse_model') or '')
    model = next((m for m in models if m.name == reported.replace('\\','/').rstrip('/').split('/')[-1]), None)
    selection = 'recorded_sparse_model' if model else 'unavailable'
    if model is None and models:
        model = max(models, key=model_image_count)
        selection = 'most_registered_images'
    if model is None and any((co/'dense'/'sparse'/name).is_file() for name in ('images.bin','images.txt')):
        model, selection = co/'dense'/'sparse', 'dense_workspace_sparse_fallback'
    report['sparse_model_path'] = str(model) if model else None
    report['sparse_model_selection'] = selection
    if model:
        try:
            registered = registered_names(model)
            report['registered_filenames'] = registered
            report['observed_registered_images'] = len(registered)
            manifest = exp/'input_manifest.csv'
            if manifest.is_file():
                with manifest.open(encoding='utf-8-sig', newline='') as stream:
                    inputs = [row['filename'] for row in csv.DictReader(stream)]
                report['unregistered_filenames'] = sorted(set(inputs)-set(registered))
                report['unexpected_registered_filenames'] = sorted(set(registered)-set(inputs))
            else:
                report['manifest_status'] = 'missing; unregistered filenames unavailable'
        except (OSError, ValueError, struct.error, KeyError, csv.Error) as exc:
            report['sparse_parse_error'] = str(exc)
    folder = exp/'inspections'/f'review_{stamp}'
    folder.mkdir(parents=True, exist_ok=False)
    if model and COLMAP.is_file() and not summary_only:
        txt = folder/'colmap_text';txt.mkdir()
        text_rc,log = model_converter(model,txt,'TXT')
        (folder/'model_to_text.log').write_text(log,encoding='utf-8')
        sparse_ply = folder/'colmap_sparse.ply'
        rc,log = model_converter(model,sparse_ply,'PLY')
        (folder/'model_to_ply.log').write_text(log,encoding='utf-8')
        report['colmap_sparse_ply'] = checked_counts(ply_counts, sparse_ply)
        report['conversion_return_codes'] = {'TXT': text_rc, 'PLY': rc}
        text_exists = all((txt/name).is_file() for name in ('cameras.txt','images.txt','points3D.txt'))
        report['conversion_status'] = ('success' if text_rc == rc == 0 and text_exists and
            report['colmap_sparse_ply'].get('geometry_state') not in ('missing','invalid') else 'failed')
    elif model and not summary_only:
        report['conversion_status'] = 'unavailable_colmap_launcher'
    report['interpretation'] = ('Artifact counts and conversion success do not establish geometric accuracy, '
                               'scene completeness or a usable mesh. Read geometry_state and validation_scope.')
    (folder/'inspection.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    summary_md = [f'# Inspection - {group}/{run_id}', '',
        f"MapAnything GLB: {report['mapanything']}", '',
        f"COLMAP dense fused PLY: {report['colmap_dense_cloud']}", '',
        f"COLMAP Poisson mesh: {report['colmap_dense_mesh']}", '',
        f"COLMAP registered names: {report.get('registered_filenames','unavailable')}", '',
        f"COLMAP unregistered names: {report.get('unregistered_filenames','unavailable')}", '',
        f"Sparse model selection: {selection}", '',
        f"Sparse parsing error: {report.get('sparse_parse_error','none')}", '',
        f"Conversion status: {report['conversion_status']}", '',
        report['interpretation']]
    (folder/'inspection.md').write_text('\n'.join(summary_md)+'\n',encoding='utf-8')
    print('\n'.join(summary_md));print('Saved:',folder)
    colmap = summary.get('colmap') or {}
    registered_count = report.get('observed_registered_images')
    input_count = summary.get('input_images')
    return {
        'run_ref': f'{group}/{run_id}', 'dataset': summary.get('dataset', dataset_label),
        'input_images': summary.get('input_images'),
        'input_fingerprint_sha256': summary.get('input_fingerprint_sha256'),
        'mapanything_status': (summary.get('mapanything') or {}).get('status'),
        'mapanything_geometry_state': report['mapanything']['geometry_state'],
        'mapanything_vertices': report['mapanything'].get('mesh_vertex_samples'),
        'mapanything_triangles': report['mapanything'].get('triangle_faces'),
        'colmap_status': colmap.get('status'),
        'registered_images': registered_count,
        'recorded_registered_images': colmap.get('registered_images'),
        'registration_rate_percent': (100 * registered_count / input_count if registered_count is not None
            and type(input_count) is int and input_count > 0 else None),
        'sparse_points': colmap.get('sparse_points'),
        'mean_reprojection_error_px': colmap.get('mean_reprojection_error_px'),
        'dense_points': report['colmap_dense_cloud'].get('vertex_count'),
        'dense_geometry_state': report['colmap_dense_cloud']['geometry_state'],
        'mesh_faces': report['colmap_dense_mesh'].get('face_count'),
        'mesh_geometry_state': report['colmap_dense_mesh']['geometry_state'],
        'inspection_directory': str(folder),
        'conversion_status': report['conversion_status'],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description='Inspect one or more MapAnything/COLMAP runs')
    p.add_argument('--dataset', help='Exact output scene group (normally the dataset folder name)')
    p.add_argument('--run', help='Run ID with --dataset, or a group/run reference')
    p.add_argument('--runs', nargs='+', help='group/run references for batch inspection')
    p.add_argument('--summary-only', action='store_true',
                   help='Write an inspection without COLMAP conversion; native sparse names are still read')
    args = p.parse_args(argv)
    if args.runs and (args.dataset or args.run):
        p.error('Use either --runs or the --dataset/--run pair, not both')
    refs = []
    if args.runs:
        for value in args.runs:
            try:
                group, run_id = parse_run_reference(value)
            except ValueError as exc:
                p.error(str(exc))
            refs.append((group, run_id, group))
    else:
        if not args.run:
            p.error('provide --run group/run, --dataset group --run ID, or --runs group/run ...')
        value = f'{args.dataset}/{args.run}' if args.dataset else args.run
        try:
            if args.dataset:
                validate_component(args.dataset, 'scene group')
                validate_component(args.run, 'run ID')
            group, run_id = parse_run_reference(value)
        except ValueError as exc:
            p.error(str(exc))
        refs.append((group, run_id, group))
    refs = list(dict.fromkeys(refs))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    rows = []
    try:
        for group, run_id, label in refs:
            rows.append(inspect_one(group, run_id, label, stamp, args.summary_only))
    except (OSError, ValueError) as exc:
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
