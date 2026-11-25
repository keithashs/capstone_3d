"""MapAnything vs standalone COLMAP SfM + dense MVS benchmark runner.

The single-run interface remains compatible with the earlier 6/8/12-image and
Gerrard Hall experiments.  The suite interface adds repeatable full-set tests:

  python scripts\run_compare.py --suite full-resolution --suite-tag fullres_v1

That suite runs Office (26 images) and Kicker (31 images) twice each.  The
baseline explicitly uses SIFT max dimension 1600 and Lowe ratio 0.8; the
higher-resolution configuration uses 2400 and 0.7.  Existing output folders
are never overwritten.  No PyCOLMAP, Nerfstudio, Splatfacto, or ETH3D
reference poses are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP_REPO = ROOT / 'tools' / 'map-anything'
MAP_SCRIPT = MAP_REPO / 'scripts' / 'demo_images_only_inference.py'
COLMAP = ROOT / 'tools' / 'COLMAP' / 'COLMAP.bat'
SUPPORTED = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
COLUMNS = ['dataset', 'scene', 'run_id', 'method', 'input_images',
           'input_fingerprint_sha256', 'settings_fingerprint_sha256',
           'sift_max_size', 'sift_ratio', 'mvs_max_size', 'status',
           'elapsed_seconds', 'peak_gpu_memory_mib', 'registered_images',
           'registration_rate_percent', 'sparse_points', 'mean_reprojection_error_px',
           'dense_points', 'mesh_faces', 'mesh_status', 'gpu_oom', 'output']

FULL_RESOLUTION_CASES = [
    {'dataset': 'office_26_undistorted', 'limit': 0, 'expected_images': 26,
     'run': 'office_full26_sift1600_ratio080', 'profile': 'baseline',
     'sift_max_size': 1600, 'sift_ratio': 0.8},
    {'dataset': 'office_26_undistorted', 'limit': 0, 'expected_images': 26,
     'run': 'office_full26_sift2400_ratio070', 'profile': 'higher_resolution',
     'sift_max_size': 2400, 'sift_ratio': 0.7},
    {'dataset': 'kicker_31_undistorted', 'limit': 0, 'expected_images': 31,
     'run': 'kicker_full31_sift1600_ratio080', 'profile': 'baseline',
     'sift_max_size': 1600, 'sift_ratio': 0.8},
    {'dataset': 'kicker_31_undistorted', 'limit': 0, 'expected_images': 31,
     'run': 'kicker_full31_sift2400_ratio070', 'profile': 'higher_resolution',
     'sift_max_size': 2400, 'sift_ratio': 0.7},
]

LEGACY_CASES = [
    *[{'dataset': 'office_26_undistorted', 'limit': count,
       'expected_images': count, 'run': f'office_undistorted_{count}img',
       'profile': 'legacy_baseline', 'sift_max_size': 1600, 'sift_ratio': 0.8}
      for count in (6, 8, 12)],
    *[{'dataset': 'kicker_31_undistorted', 'limit': count,
       'expected_images': count, 'run': f'kicker_undistorted_{count}img',
       'profile': 'legacy_baseline', 'sift_max_size': 1600, 'sift_ratio': 0.8}
      for count in (6, 8, 12)],
    {'dataset': 'gerrard-hall', 'limit': 0, 'expected_images': 100,
     'run': 'gerrard-hall-full100', 'profile': 'legacy_baseline',
     'sift_max_size': 1600, 'sift_ratio': 0.8},
]


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def natural_key(x):
    return [int(i) if i.isdigit() else i.lower() for i in re.split(r'(\d+)', x.name)]


def save_json(path, record):
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def csv_write(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def command_output(args, cwd=None):
    try:
        p = subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', cwd=cwd, timeout=25)
        return (p.stdout + p.stderr).strip()
    except Exception as e:
        return f'Unavailable: {e}'


def image_folder(dataset):
    """Locate DSLR images regardless of archive-created parent folders.

    Never quietly mix distorted and undistorted files from several directories.
    """
    base = ROOT / 'data' / dataset
    if not base.is_dir():
        raise ValueError(f'Missing dataset: {base}')
    preferred = [base / 'images' / 'dslr_images', base / 'dslr_images',
                 base / 'images', base]
    def list_images(folder):
        return sorted((p for p in folder.iterdir() if p.is_file()
                       and p.suffix.lower() in SUPPORTED), key=natural_key) if folder.is_dir() else []
    for candidate in preferred:
        files = list_images(candidate)
        if files:
            return candidate, files
    choices = []
    for folder in base.rglob('*'):
        if folder.is_dir() and folder.name.lower() == 'dslr_images':
            photos = list_images(folder)
            if photos:
                choices.append((folder, photos))
    if len(choices) != 1:
        raise ValueError(f'Expected one DSLR image directory under {base}; found {len(choices)}. '
                         'Move the images to data/<dataset>/dslr_images.')
    return choices[0]


def ply_vertex_count(path):
    """Read PLY header; not the potentially huge binary payload."""
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            return None
        for _ in range(1000):
            line = stream.readline()
            if not line:
                break
            if line.startswith(b'element vertex '):
                try:
                    return int(line.strip().split()[-1])
                except (ValueError, IndexError):
                    return None
            if line.strip() == b'end_header':
                break
    return None


def ply_face_count(path):
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        for _ in range(1000):
            line = stream.readline()
            if not line:
                break
            if line.startswith(b'element face '):
                return int(line.strip().split()[-1])
            if line.strip() == b'end_header':
                break
    return None


def glb_counts(path):
    """Counts mesh vertices/triangles without installing trimesh."""
    import struct
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        hdr = stream.read(12)
        if len(hdr) != 12 or hdr[:4] != b'glTF':
            raise ValueError('GLB header is invalid')
        jsn_len, kind = struct.unpack('<I4s', stream.read(8))
        if kind != b'JSON' or jsn_len > 50_000_000:
            raise ValueError('GLB JSON chunk invalid')
        doc = json.loads(stream.read(jsn_len).decode('utf-8').strip(' \x00'))
    accessors = doc.get('accessors', [])
    verts = triangles = 0
    for mesh in doc.get('meshes', []):
        for prim in mesh.get('primitives', []):
            pos = prim.get('attributes', {}).get('POSITION')
            if pos is not None:
                verts += int(accessors[pos].get('count', 0))
            idx = prim.get('indices')
            count = int(accessors[idx]['count']) if idx is not None else (
                int(accessors[pos]['count']) if pos is not None else 0)
            if prim.get('mode', 4) == 4:
                triangles += count // 3
    return {'mesh_count': len(doc.get('meshes', [])),
            'vertices': verts, 'triangles': triangles, 'bytes': path.stat().st_size}


def database_stats(path):
    if not path.is_file():
        return {}
    result = {}
    with sqlite3.connect(path) as connection:
        for key, qry in {
            'database_images': 'SELECT COUNT(*) FROM images',
            'keypoints': 'SELECT COALESCE(SUM(rows),0) FROM keypoints',
            'verified_pairs': 'SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0',
        }.items():
            try:
                result[key] = connection.execute(qry).fetchone()[0]
            except sqlite3.Error:
                result[key] = None
    return result


def parse_analyzer(output):
    result = {}
    for key, rx, conv in [
        ('registered_images', r'Registered images:\s*(\d+)', int),
        ('sparse_points', r'Points:\s*(\d+)', int),
        ('observations', r'Observations:\s*(\d+)', int),
        ('mean_reprojection_error_px', r'Mean reprojection error:\s*([\d.]+)', float),
    ]:
        match = re.search(rx, output, re.IGNORECASE)
        if match:
            result[key] = conv(match.group(1))
    return result


def records_fingerprint(records):
    """Canonical ordered filename/content fingerprint used by every report tool."""
    payload = ''.join(f"{row['filename']}\0{row['sha256']}\n" for row in records)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def dict_fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def suite_cases(name):
    if name == 'full-resolution':
        return [dict(row) for row in FULL_RESOLUTION_CASES]
    if name == 'legacy':
        return [dict(row) for row in LEGACY_CASES]
    if name == 'all':
        return [dict(row) for row in LEGACY_CASES + FULL_RESOLUTION_CASES]
    raise ValueError(f'Unknown suite: {name}')


def suite_command(case, args, run_id, allow_mapanything_failure):
    command = [sys.executable, str(Path(__file__).resolve()),
               '--dataset', case['dataset'], '--limit', str(case['limit']),
               '--expected-image-count', str(case['expected_images']),
               '--run', run_id, '--profile', case['profile'],
               '--sift-max-size', str(case['sift_max_size']),
               '--sift-ratio', str(case['sift_ratio']),
               '--mvs-max-size', str(args.mvs_max_size), '--method', args.method]
    if args.single_camera:
        command.append('--single-camera')
    if args.geom_consistency:
        command.append('--geom-consistency')
    if args.skip_mesh:
        command.append('--skip-mesh')
    if allow_mapanything_failure:
        command.append('--allow-mapanything-failure')
    return command


def run_suite(args, parser):
    """Run a fixed experiment matrix sequentially without overwriting any run."""
    if args.dataset or args.run:
        parser.error('--suite cannot be combined with --dataset or --run')
    if args.start:
        parser.error('--suite always starts at image index 0; omit --start')
    tag = args.suite_tag or datetime.now().strftime('%Y%m%d_%H%M%S')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        parser.error('--suite-tag may contain only letters, numbers, underscores and hyphens')
    allow_ma_failure = (not args.require_mapanything_success and
                        args.method in ('both', 'mapanything'))
    prepared = []
    fingerprint_cache = {}
    for source_case in suite_cases(args.suite):
        case = dict(source_case)
        try:
            folder, photos = image_folder(case['dataset'])
        except Exception as exc:
            parser.error(str(exc))
        selected = photos[:case['limit']] if case['limit'] else photos
        if len(selected) != case['expected_images']:
            parser.error(f"{case['dataset']} selected {len(selected)} images; "
                         f"suite requires exactly {case['expected_images']}")
        cache_key = tuple(str(path.resolve()) for path in selected)
        if cache_key not in fingerprint_cache:
            rows = [{'filename': path.name, 'sha256': sha256(path)} for path in selected]
            fingerprint_cache[cache_key] = records_fingerprint(rows)
        case['input_fingerprint_sha256'] = fingerprint_cache[cache_key]
        case['source_folder'] = str(folder)
        case['selected_images'] = [path.name for path in selected]
        case['run_id'] = f"{case['run']}_{tag}"
        case['command'] = suite_command(case, args, case['run_id'], allow_ma_failure)
        prepared.append(case)

    print(f"\n===== {args.suite.upper()} SUITE PLAN =====")
    print('Suite tag:', tag)
    print('MapAnything failures allowed:', allow_ma_failure)
    for index, case in enumerate(prepared, 1):
        print(f"{index:02d}. {case['dataset']} | {case['expected_images']} images | "
              f"profile={case['profile']} | SIFT={case['sift_max_size']} | "
              f"ratio={case['sift_ratio']} | run={case['run_id']}")
        print('    fingerprint:', case['input_fingerprint_sha256'])
        print('    command:', subprocess.list2cmdline(case['command']))
    if args.dry_run:
        print('\nDry run only: no experiment or suite output was created.')
        return 0

    suite_id = f"{args.suite.replace('-', '_')}_{tag}"
    suite_dir = ROOT / 'experiments' / 'suites' / suite_id
    if suite_dir.exists():
        parser.error(f'Suite output already exists: {suite_dir}. Choose another --suite-tag.')
    suite_dir.mkdir(parents=True)
    plan = {'suite': args.suite, 'suite_id': suite_id, 'suite_tag': tag,
            'created': timestamp(), 'method': args.method,
            'allow_mapanything_failure': allow_ma_failure,
            'cases': [{**case, 'command': subprocess.list2cmdline(case['command'])}
                      for case in prepared]}
    save_json(suite_dir / 'suite_plan.json', plan)

    results = []
    for index, case in enumerate(prepared, 1):
        print(f"\n===== SUITE RUN {index}/{len(prepared)}: {case['run_id']} =====", flush=True)
        started = time.perf_counter()
        completed = subprocess.run(case['command'], cwd=ROOT)
        elapsed = round(time.perf_counter() - started, 3)
        group = re.sub(r'_undistorted$', '', case['dataset'])
        summary_path = ROOT / 'experiments' / group / case['run_id'] / 'summary.json'
        summary = {}
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as exc:
                summary = {'suite_summary_read_error': str(exc)}
        ma_status = summary.get('mapanything', {}).get('status', 'missing')
        colmap_status = summary.get('colmap', {}).get('status', 'missing')
        required_ok = bool(summary) and 'runner_error' not in summary
        if args.method in ('both', 'colmap'):
            required_ok = required_ok and colmap_status == 'success'
        if args.method in ('both', 'mapanything'):
            required_ok = required_ok and (
                ma_status in ('success', 'failed') if allow_ma_failure else ma_status == 'success')
        result = {'index': index, 'dataset': case['dataset'], 'scene_group': group,
                  'run_id': case['run_id'], 'profile': case['profile'],
                  'input_images': case['expected_images'],
                  'input_fingerprint_sha256': case['input_fingerprint_sha256'],
                  'sift_max_size': case['sift_max_size'], 'sift_ratio': case['sift_ratio'],
                  'mapanything_status': ma_status, 'colmap_status': colmap_status,
                  'child_exit_code': completed.returncode, 'suite_required_success': required_ok,
                  'elapsed_seconds': elapsed, 'summary_path': str(summary_path)}
        results.append(result)
        save_json(suite_dir / 'suite_results.json', {'suite_id': suite_id, 'results': results})
        csv_write(suite_dir / 'suite_results.csv', results,
                  ['index', 'dataset', 'scene_group', 'run_id', 'profile', 'input_images',
                   'input_fingerprint_sha256', 'sift_max_size', 'sift_ratio',
                   'mapanything_status', 'colmap_status', 'child_exit_code',
                   'suite_required_success', 'elapsed_seconds', 'summary_path'])
        if not required_ok and args.stop_on_failure:
            print('Stopping suite because --stop-on-failure was requested.', file=sys.stderr)
            break

    finished = {'suite': args.suite, 'suite_id': suite_id, 'suite_tag': tag,
                'created': plan['created'], 'completed': timestamp(),
                'allow_mapanything_failure': allow_ma_failure, 'results': results,
                'required_runs_succeeded': all(row['suite_required_success'] for row in results),
                'planned_runs': len(prepared), 'completed_runs': len(results)}
    save_json(suite_dir / 'suite_results.json', finished)
    print('\n===== SUITE COMPLETE =====')
    print('Results:', suite_dir)
    print('Required COLMAP/method outcomes succeeded:', finished['required_runs_succeeded'])
    return 0 if (finished['required_runs_succeeded'] and len(results) == len(prepared)) else 1


class GPUSampler:
    """Whole-system NVIDIA WDDM GPU polling, not per-process allocations."""
    def __init__(self):
        self.stop = threading.Event()
        self.stage = 'initializing'
        self.samples = []
        self.thread = threading.Thread(target=self._work, daemon=True)

    def _work(self):
        while not self.stop.is_set():
            raw = command_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                                  '--format=csv,noheader,nounits'])
            try:
                parts = [float(x.strip()) for x in raw.splitlines()[0].split(',')[:3]]
                if len(parts) != 3:
                    raise ValueError('Expected 3 GPU measurements')
                self.samples.append({'timestamp': timestamp(), 'stage': self.stage,
                                     'gpu_memory_mib': parts[0], 'gpu_util_percent': parts[1],
                                     'gpu_temperature_c': parts[2]})
            except Exception:
                pass
            self.stop.wait(1)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=30)

    def peak(self, prefix):
        vals = [x['gpu_memory_mib'] for x in self.samples if x['stage'].startswith(prefix)]
        return max(vals) if vals else None


def run_step(name, params, working_dir, logs, monitor, stages):
    """Stream to console and unique stage log. Failure never masquerades as success."""
    monitor.stage = name
    start = time.perf_counter()
    logfile = logs / f'{name}.log'
    args = [str(x) for x in params]
    cmd_display = subprocess.list2cmdline(args)
    print(f'\n[{timestamp()}] START {name}\n$ {cmd_display}', flush=True)
    last_lines = deque(maxlen=250)
    out_of_memory = False
    exit_code = -1
    try:
        with logfile.open('w', encoding='utf-8') as log:
            log.write(f'Stage: {name}\nStart: {timestamp()}\nCWD: {working_dir}\n'
                      f'Command: {cmd_display}\n\n')
            log.flush()
            env = dict(os.environ)
            env['PYTHONUNBUFFERED'] = '1'
            p = subprocess.Popen(args, cwd=working_dir, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                 errors='replace', bufsize=1)
            assert p.stdout is not None
            for line in p.stdout:
                print(line, end='', flush=True)
                log.write(line)
                last_lines.append(line)
                if 'out of memory' in line.lower() or 'std::bad_alloc' in line.lower():
                    out_of_memory = True
            exit_code = p.wait()
    except Exception as exc:
        last_lines.append(f'EXCEPTION: {exc}')
        logfile.open('a', encoding='utf-8').write(f'\nEXCEPTION: {exc}\n')
    elapsed = round(time.perf_counter() - start, 3)
    stages.append({'stage': name, 'seconds': elapsed, 'exit_code': exit_code,
                   'out_of_memory': out_of_memory, 'command': cmd_display})
    logfile.open('a', encoding='utf-8').write(
        f'\nEND {timestamp()} exit={exit_code} duration_seconds={elapsed}\n')
    print(f'[{timestamp()}] END {name}: exit={exit_code} time={elapsed}s', flush=True)
    return exit_code, elapsed, ''.join(last_lines), out_of_memory


def colmap_command(*args):
    return ['cmd.exe', '/d', '/c', str(COLMAP), *map(str, args)]


def main():
    parser = argparse.ArgumentParser(description='Independent MapAnything + COLMAP dense MVS benchmark')
    parser.add_argument('--dataset', help='data/<dataset>, e.g. office_26_undistorted')
    parser.add_argument('--suite', choices=['full-resolution', 'legacy', 'all'],
                        help='Run a reproducible experiment matrix instead of one dataset/run')
    parser.add_argument('--suite-tag',
                        help='Suffix for every suite run ID; defaults to a timestamp')
    parser.add_argument('--stop-on-failure', action='store_true',
                        help='Stop a suite after the first required-method failure')
    parser.add_argument('--require-mapanything-success', action='store_true',
                        help='Make a MapAnything failure fail the suite (suite default allows it)')
    parser.add_argument('--limit', type=int, default=6, help='Image count (0 means all images)')
    parser.add_argument('--start', type=int, default=0, help='Start index (natural filename sort)')
    parser.add_argument('--run', help='Run identifier; no existing run is overwritten')
    parser.add_argument('--profile', default='custom',
                        help='Configuration label recorded in the run metadata')
    parser.add_argument('--expected-image-count', type=int,
                        help='Fail before creating output unless exactly this many images are selected')
    parser.add_argument('--single-camera', action='store_true', help='Share intrinsics; NOT default for ETH3D')
    parser.add_argument('--mvs-max-size', type=int, default=1000, help='Dense MVS image max dimension')
    parser.add_argument('--sift-max-size', type=int, default=1600, help='SIFT max image dimension')
    parser.add_argument('--sift-ratio', '--sift-max-ratio', dest='sift_ratio', type=float,
                        default=0.8,
                        help='COLMAP SiftMatching.max_ratio (0.8 baseline; 0.7 stricter)')
    parser.add_argument('--geom-consistency', action='store_true', help='More memory-intensive geometric MVS')
    parser.add_argument('--skip-mesh', action='store_true', help='Stop after dense fused point cloud')
    parser.add_argument('--method', choices=['both', 'mapanything', 'colmap'], default='both')
    parser.add_argument('--allow-mapanything-failure', action='store_true',
                        help='Keep a failed MapAnything status but return success if required COLMAP succeeds')
    parser.add_argument('--dry-run', action='store_true', help='Print selected images/settings without creating output')
    args = parser.parse_args()
    if args.suite:
        return run_suite(args, parser)
    if not args.dataset:
        parser.error('one of --dataset or --suite is required')
    aliases = {'gerrad-hall': 'gerrard-hall', 'garred-hall': 'gerrard-hall'}
    if args.dataset in aliases:
        print(f"NOTE: using canonical dataset name {aliases[args.dataset]!r} for {args.dataset!r}.")
        args.dataset = aliases[args.dataset]
    if any(not re.fullmatch(r'[a-zA-Z0-9_-]+', name)
           for name in (args.dataset, args.run, args.profile) if name is not None):
        parser.error('Dataset, run and profile names may contain only letters, numbers, underscores and hyphens')
    if (args.start < 0 or args.limit < 0 or args.mvs_max_size <= 0 or
            args.sift_max_size <= 0 or not 0 < args.sift_ratio <= 1):
        parser.error('Invalid start, limit, image max-size or SIFT ratio value')
    if args.expected_image_count is not None and args.expected_image_count < 2:
        parser.error('--expected-image-count must be at least 2')

    group = re.sub(r'_undistorted$', '', args.dataset)
    if group not in {'office_26', 'kicker_31'} and args.dataset.endswith('_undistorted'):
        parser.error('Unexpected ETH3D dataset name')
    run = args.run or datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    try:
        photo_folder, photos = image_folder(args.dataset)
    except Exception as exc:
        parser.error(str(exc))
    selected = photos[args.start:]
    if args.limit:
        selected = selected[:args.limit]
    if len(selected) < 2:
        parser.error(f'Selected {len(selected)} images. Need at least two.')
    if args.expected_image_count is not None and len(selected) != args.expected_image_count:
        parser.error(f'Selected {len(selected)} images, but --expected-image-count requires '
                     f'{args.expected_image_count}. No output was created.')
    if args.dataset.endswith('_undistorted') and len(photos) < 12:
        print(f'WARNING: only {len(photos)} images found; ensure the ETH3D archive was fully extracted.')

    print(f'Project root: {ROOT}\nDataset directory: {photo_folder}\n'
          f'Scene group: {group}\nRun ID: {run}\nInput count: {len(selected)}\n'
          f'Methods: {args.method}\nMVS max dimension: {args.mvs_max_size}\n'
          f'SIFT max dimension: {args.sift_max_size}\nSIFT ratio threshold: {args.sift_ratio}\n'
          f'Profile: {args.profile}\nGeometric consistency: {args.geom_consistency}\n'
          f'Allow MapAnything failure: {args.allow_mapanything_failure}')
    for p in selected:
        print(' -', p.name)
    if args.dry_run:
        return 0
    if args.method in ('both', 'mapanything') and not MAP_SCRIPT.is_file():
        parser.error(f'MapAnything CLI not found: {MAP_SCRIPT}')
    if args.method in ('both', 'colmap') and not COLMAP.is_file():
        parser.error(f'Standalone COLMAP launcher not found: {COLMAP}')

    exp = ROOT / 'experiments' / group / run
    ma = ROOT / 'mapanything_output' / group / run
    co = ROOT / 'colmap_output' / group / run
    # Preserve prior logs and 3D outputs, even on repeated invocation with same run name.
    for folder in (exp, ma, co):
        if folder.exists():
            parser.error(f'Output already exists: {folder}. Choose a different --run ID.')
    for folder in (exp, ma, co):
        folder.mkdir(parents=True)
    logs = exp / 'logs'
    snapshot = exp / 'input_images'
    logs.mkdir()
    snapshot.mkdir()

    records = []
    for src in selected:
        dst = snapshot / src.name
        if dst.exists():
            raise RuntimeError(f'Duplicate filename: {src.name}')
        shutil.copy2(src, dst)
        records.append({'filename': src.name, 'original_path': str(src),
                        'bytes': dst.stat().st_size, 'sha256': sha256(dst)})
    csv_write(exp / 'input_manifest.csv', records, ['filename','original_path','bytes','sha256'])
    fingerprint = records_fingerprint(records)
    camera_model = 'PINHOLE' if args.dataset.endswith('_undistorted') else 'SIMPLE_RADIAL'
    settings = {'profile': args.profile, 'single_camera': args.single_camera,
                'camera_model': camera_model, 'mvs_max_size': args.mvs_max_size,
                'sift_max_size': args.sift_max_size, 'sift_ratio': args.sift_ratio,
                'geom_consistency': args.geom_consistency,
                'poisson_meshing': not args.skip_mesh, 'method': args.method,
                'allow_mapanything_failure': args.allow_mapanything_failure,
                'input_start_index': args.start, 'input_limit': args.limit,
                'expected_image_count': args.expected_image_count}
    settings_fingerprint = dict_fingerprint(settings)
    experiment_fingerprint = hashlib.sha256(
        f'{fingerprint}\0{settings_fingerprint}'.encode('ascii')).hexdigest()
    environment = {'run_started': timestamp(), 'project_root': str(ROOT),
                   'source_folder': str(photo_folder), 'dataset': args.dataset, 'scene_group': group,
                   'images': len(selected), 'input_fingerprint_sha256': fingerprint,
                   'settings_fingerprint_sha256': settings_fingerprint,
                   'experiment_fingerprint_sha256': experiment_fingerprint,
                   'python': sys.version, 'python_executable': sys.executable,
                   'conda': os.getenv('CONDA_DEFAULT_ENV'), 'platform': platform.platform(),
                   'nvidia_smi': command_output(['nvidia-smi']),
                   'mapanything_git_commit': command_output(['git','rev-parse','HEAD'], cwd=MAP_REPO),
                   'colmap_help': command_output(colmap_command('-h')) if COLMAP.is_file() else 'not found',
                   'eth3d_reference_calibration_used': False,
                   'eth3d_reference_poses_used': False,
                   'settings': settings}
    for mod in ('torch', 'torchvision', 'mapanything'):
        try: environment[mod] = importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError: environment[mod] = None
    save_json(exp/'environment.json', environment)

    metrics = {'dataset': args.dataset, 'scene_group': group, 'run_id': run,
               'input_images': len(selected), 'input_fingerprint_sha256': fingerprint,
               'settings_fingerprint_sha256': settings_fingerprint,
               'experiment_fingerprint_sha256': experiment_fingerprint,
               'settings': settings,
               'mapanything': {'status': 'not_run'}, 'colmap': {'status': 'not_run'},
               'colmap_dense': {'status': 'not_run'}, 'stages': []}
    monitor = GPUSampler()
    monitor.start()
    overall_start = time.perf_counter()
    try:
        if args.method in ('both', 'mapanything'):
            glb = ma / 'scene.glb'
            rc, sec, _, oom = run_step('mapanything',
                [sys.executable, '-u', MAP_SCRIPT, '--image_folder', snapshot,
                 '--save_glb', '--output_path', glb], MAP_REPO, logs, monitor, metrics['stages'])
            valid = rc == 0 and glb.is_file() and glb.stat().st_size > 0
            obj = {'status': 'success' if valid else 'failed', 'elapsed_seconds': sec,
                   'exit_code': rc, 'gpu_oom': oom, 'output': str(glb),
                   'failure_allowed': bool(args.allow_mapanything_failure and not valid),
                   'output_bytes': glb.stat().st_size if glb.exists() else None}
            if valid:
                try: obj.update(glb_counts(glb) or {})
                except Exception as exc: obj['glb_parse_error'] = str(exc)
            metrics['mapanything'] = obj
        if args.method in ('both', 'colmap'):
            db = co / 'database.db'
            sparse = co / 'sparse'; sparse.mkdir()
            dense = co / 'dense'; dense.mkdir()
            colmap = {'status': 'failed', 'elapsed_seconds': 0, 'dense_status': 'not_started',
                      'mesh_status': 'not_started', 'output': None}
            metrics['colmap'] = colmap
            first_steps = [
                ('colmap_features', ['feature_extractor', '--database_path', db, '--image_path', snapshot,
                                     '--ImageReader.camera_model', camera_model,
                                     '--FeatureExtraction.max_image_size', args.sift_max_size] +
                     (['--ImageReader.single_camera', '1'] if args.single_camera else [])),
                ('colmap_matching', ['exhaustive_matcher', '--database_path', db,
                                     '--SiftMatching.max_ratio', args.sift_ratio]),
                ('colmap_mapping', ['mapper', '--database_path', db, '--image_path', snapshot,
                                    '--output_path', sparse]),
            ]
            sparse_ok = True
            for stage, pars in first_steps:
                rc, sec, _, oom = run_step(stage, colmap_command(*pars), ROOT, logs, monitor, metrics['stages'])
                colmap['elapsed_seconds'] += sec
                if rc != 0:
                    colmap.update({'failed_stage': stage, 'exit_code': rc, 'gpu_oom': oom})
                    sparse_ok = False
                    break
            colmap.update(database_stats(db))
            candidates = []
            if sparse_ok:
                for p in sparse.iterdir():
                    if p.is_dir() and (p/'cameras.bin').is_file():
                        rc, _, txt, _ = run_step(f'colmap_analyzer_{p.name}',
                             colmap_command('model_analyzer', '--path', p), ROOT, logs, monitor, metrics['stages'])
                        if rc == 0:
                            candidates.append((parse_analyzer(txt), p))
            candidates.sort(key=lambda pair: (pair[0].get('registered_images', 0),
                                               pair[0].get('sparse_points', 0)), reverse=True)
            if not candidates:
                colmap['failed_stage'] = 'no_valid_sparse_model'
            else:
                stats, best = candidates[0]
                colmap.update(stats)
                colmap['sparse_model'] = str(best)
                colmap['sparse_model_candidates'] = [str(p) for _, p in candidates]
                colmap['registration_rate_percent'] = round(
                    100 * stats.get('registered_images', 0) / len(selected), 2)
                colmap['sparse_status'] = 'success'
                # Dense MVS is a separate stage: COLMAP's image_undistorter creates the
                # required workspace even when ETH3D images are ALREADY undistorted.
                geom = args.geom_consistency
                stereo_type = 'geometric' if geom else 'photometric'
                mvs_steps = [
                    ('colmap_image_undistorter', ['image_undistorter', '--image_path', snapshot,
                      '--input_path', best, '--output_path', dense, '--output_type', 'COLMAP',
                      '--max_image_size', args.mvs_max_size]),
                    ('colmap_patch_match', ['patch_match_stereo', '--workspace_path', dense,
                      '--workspace_format', 'COLMAP', '--PatchMatchStereo.geom_consistency',
                      str(geom).lower(), '--PatchMatchStereo.max_image_size', args.mvs_max_size,
                      '--PatchMatchStereo.cache_size', '1', '--PatchMatchStereo.gpu_index', '0']),
                    ('colmap_stereo_fusion', ['stereo_fusion', '--workspace_path', dense,
                      '--workspace_format', 'COLMAP', '--input_type', stereo_type,
                      '--output_path', dense/'fused.ply', '--StereoFusion.cache_size', '1']),
                ]
                stage_ok = True
                for stage, pars in mvs_steps:
                    rc, sec, _, oom = run_step(stage, colmap_command(*pars), ROOT, logs, monitor, metrics['stages'])
                    colmap['elapsed_seconds'] += sec
                    if rc != 0:
                        colmap.update({'failed_stage': stage, 'exit_code': rc, 'gpu_oom': oom})
                        stage_ok = False
                        break
                cloud = dense / 'fused.ply'
                count = ply_vertex_count(cloud)
                colmap.update({'dense_points': count, 'dense_output': str(cloud),
                               'dense_output_bytes': cloud.stat().st_size if cloud.is_file() else None,
                               'dense_status': 'success' if stage_ok and count and count > 0 else 'failed'})
                colmap['status'] = 'success' if colmap['dense_status'] == 'success' else 'failed'
                colmap['output'] = str(cloud) if colmap['status'] == 'success' else None
                if colmap['status'] == 'success' and not args.skip_mesh:
                    mesh = dense/'meshed-poisson.ply'
                    rc, sec, _, oom = run_step('colmap_poisson_mesher',
                        colmap_command('poisson_mesher', '--input_path', cloud, '--output_path', mesh),
                        ROOT, logs, monitor, metrics['stages'])
                    colmap['elapsed_seconds'] += sec
                    faces = ply_face_count(mesh)
                    colmap.update({'mesh_status': 'success' if rc == 0 and faces and faces > 0 else 'failed',
                                   'mesh_faces': faces, 'mesh_output': str(mesh), 'mesh_gpu_oom': oom})
                elif args.skip_mesh:
                    colmap['mesh_status'] = 'skipped'
                else:
                    colmap['mesh_status'] = 'not_run_due_to_dense_failure'
            metrics['colmap_dense'] = {
                'status': colmap.get('dense_status', 'failed'),
                'dense_points': colmap.get('dense_points'),
                'mesh_status': colmap.get('mesh_status'),
                'mesh_faces': colmap.get('mesh_faces'),
                'output': colmap.get('dense_output'),
            }
            colmap['elapsed_seconds'] = round(colmap['elapsed_seconds'], 3)
    except Exception as exc:
        metrics['runner_error'] = f'{type(exc).__name__}: {exc}'
        (logs/'runner_error.log').write_text(metrics['runner_error'], encoding='utf-8')
        print('ERROR:', metrics['runner_error'], file=sys.stderr)
    finally:
        monitor.close()
        metrics['run_completed'] = timestamp()
        metrics['total_benchmark_seconds'] = round(time.perf_counter() - overall_start, 3)
        for kind in ('mapanything','colmap'):
            if kind in metrics:
                metrics[kind]['peak_gpu_memory_mib'] = monitor.peak(kind)
        csv_write(exp/'gpu_samples.csv', monitor.samples,
                  ['timestamp','stage','gpu_memory_mib','gpu_util_percent','gpu_temperature_c'])
        save_json(exp/'summary.json', metrics)
        rows = []
        for method in ('mapanything', 'colmap'):
            m = metrics[method]
            row = {'dataset': args.dataset, 'scene': group, 'run_id': run, 'method': method,
                   'input_images': len(selected), 'input_fingerprint_sha256': fingerprint,
                   'settings_fingerprint_sha256': settings_fingerprint,
                   'sift_max_size': args.sift_max_size, 'sift_ratio': args.sift_ratio,
                   'mvs_max_size': args.mvs_max_size, 'status': m.get('status'),
                   'elapsed_seconds': m.get('elapsed_seconds'),
                   'peak_gpu_memory_mib': m.get('peak_gpu_memory_mib'),
                   'registered_images': m.get('registered_images'),
                   'registration_rate_percent': m.get('registration_rate_percent'),
                   'sparse_points': m.get('sparse_points'),
                   'mean_reprojection_error_px': m.get('mean_reprojection_error_px'),
                   'dense_points': m.get('dense_points'), 'mesh_faces': m.get('mesh_faces'),
                   'mesh_status': m.get('mesh_status'), 'gpu_oom': m.get('gpu_oom'),
                   'output': m.get('output')}
            rows.append(row)
        csv_write(exp/'comparison.csv', rows, COLUMNS)
        # Keep the original aggregate intact: it contains duplicate/stale rows from
        # earlier experiments and has a different schema. Reports read per-run JSON.
        master = ROOT/'experiments'/'all_runs_dense_v2.csv'
        master.parent.mkdir(exist_ok=True)
        if master.is_file():
            with master.open('r', encoding='utf-8-sig', newline='') as f:
                header = next(csv.reader(f), None)
            if header != COLUMNS:
                master = ROOT/'experiments'/f'all_runs_dense_v2_{datetime.now():%Y%m%d_%H%M%S}.csv'
        # Write BOM only for a NEW Excel-compatible CSV, never in the middle
        # of an existing file when appending repeat experiments.
        new_master = not master.exists() or master.stat().st_size == 0
        with master.open('a', encoding='utf-8-sig' if new_master else 'utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_master:
                w.writeheader()
            w.writerows(rows)
        print('\n===== COMPLETED EXPERIMENT =====')
        print('Run:', group, '/', run, '| selected images:', len(selected))
        print('MapAnything:', metrics['mapanything'].get('status'),
              '| COLMAP dense:', metrics['colmap'].get('dense_status'))
        print('COLMAP Poisson mesh:', metrics['colmap'].get('mesh_status'))
        print('Detailed results:', exp)
        print('Dense comparison summary:', master)
    done = []
    for method in ('mapanything', 'colmap'):
        if args.method not in ('both', method):
            continue
        if (method == 'mapanything' and args.allow_mapanything_failure and
                metrics[method]['status'] == 'failed'):
            continue
        done.append(metrics[method]['status'] == 'success')
    return 0 if all(done) and 'runner_error' not in metrics else 1


if __name__ == '__main__':
    raise SystemExit(main())
