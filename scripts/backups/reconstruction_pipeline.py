"""Independent MapAnything and standalone COLMAP reconstruction for one dataset.

Both Windows entry points use this module. Defaults apply to arbitrary image
folders; no historical scene suites or name-based camera assumptions are built in.
Existing run folders are never overwritten.
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
from contextlib import closing
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
    """Find one image directory without mixing image sets."""
    validate_component(dataset, 'dataset')
    base = ROOT / 'data' / dataset
    if not base.is_dir():
        raise ValueError(f'Missing dataset: {base}')

    def images_in(folder):
        return sorted((p for p in folder.iterdir() if p.is_file()
                       and p.suffix.lower() in SUPPORTED), key=natural_key)

    for folder in (base / 'images', base):
        if folder.is_dir():
            images = images_in(folder)
            if images:
                return folder, images
    choices = []
    for folder in base.rglob('*'):
        if folder.is_dir():
            images = images_in(folder)
            if images:
                choices.append((folder, images))
    if len(choices) != 1:
        raise ValueError(f'Expected one image directory under {base}; found {len(choices)}. '
                         'Place the intended images in data/<dataset>/images/.')
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
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
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


def stop_child_process(process):
    """Stop only this stage's child, including its cmd/bat descendants on Windows.

    Give a non-forced tree termination a short grace period before escalation.
    Cleanup errors propagate so the caller cannot launch another stage while a
    child might still be running.
    """
    if process is None or process.poll() is not None:
        return
    if os.name == 'nt':
        try:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T'],
                           capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass  # Retry forcefully below if the process has not stopped.
    else:
        process.terminate()
    try:
        process.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       capture_output=True, timeout=10)
    else:
        process.kill()
    # Never continue the pipeline until its stage process has exited.
    process.wait(timeout=10)


def append_stage_log(path, message):
    """Best-effort diagnostics; a broken log must not prevent child cleanup."""
    try:
        with path.open('a', encoding='utf-8') as log:
            log.write(message)
    except OSError:
        pass


def run_step(name, params, working_dir, logs, monitor, stages):
    """Stream to console and unique stage log. Failure never masquerades as success."""
    monitor.stage = name
    start = time.perf_counter()
    logfile = logs / f'{name}.log'
    args = params if isinstance(params, str) else [str(x) for x in params]
    cmd_display = command_display(args)
    print(f'\n[{timestamp()}] START {name}\n$ {cmd_display}', flush=True)
    last_lines = deque(maxlen=250)
    out_of_memory = False
    exit_code = -1
    process = None
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
            process = p
            assert p.stdout is not None
            for line in p.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                last_lines.append(line)
                if 'out of memory' in line.lower() or 'std::bad_alloc' in line.lower():
                    out_of_memory = True
            exit_code = p.wait()
    except KeyboardInterrupt:
        stop_child_process(process)
        stages.append({'stage': name, 'seconds': round(time.perf_counter() - start, 3),
                       'exit_code': 130, 'out_of_memory': out_of_memory,
                       'command': cmd_display, 'interrupted': True})
        append_stage_log(logfile, '\nINTERRUPTED: child process stopped; partial outputs retained.\n')
        raise
    except Exception as exc:
        stop_child_process(process)
        exit_code = -1
        last_lines.append(f'EXCEPTION: {exc}')
        append_stage_log(logfile, f'\nEXCEPTION: {exc}\n')
    finally:
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
    elapsed = round(time.perf_counter() - start, 3)
    stages.append({'stage': name, 'seconds': elapsed, 'exit_code': exit_code,
                   'out_of_memory': out_of_memory, 'command': cmd_display})
    append_stage_log(logfile, f'\nEND {timestamp()} exit={exit_code} duration_seconds={elapsed}\n')
    print(f'[{timestamp()}] END {name}: exit={exit_code} time={elapsed}s', flush=True)
    return exit_code, elapsed, ''.join(last_lines), out_of_memory


def command_display(command):
    """Display the exact command without escaping a cmd.exe string twice."""
    return command if isinstance(command, str) else subprocess.list2cmdline([str(x) for x in command])


def colmap_command(*args):
    """Launch .bat with quoted paths on both Windows workstations."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = tuple(args[0])
    if os.name != 'nt':
        return [str(COLMAP), *map(str, args)]
    values = [str(COLMAP), *map(str, args)]
    if any('"' in value or '\n' in value or '\r' in value or '%' in value for value in values):
        raise ValueError('COLMAP arguments cannot contain quotes, newlines or percent signs')
    command = ' '.join('"' + value + '"' for value in values)
    return f'cmd.exe /d /v:off /s /c "{command}"'


PIPELINE_STAGES = [
    {'stage': 1, 'name': 'SIFT Feature Extraction', 'execution_stage': 'colmap_features',
     'artifacts': ['database.db:keypoints', 'database.db:descriptors']},
    {'stage': 2, 'name': 'Feature Matching', 'execution_stage': 'colmap_matching',
     'artifacts': ['database.db:matches'],
     'note': 'Matching and geometric verification run in one exhaustive_matcher command.'},
    {'stage': 3, 'name': 'Geometric Verification Using RANSAC', 'execution_stage': 'colmap_matching',
     'artifacts': ['database.db:two_view_geometries'],
     'note': 'Verified matches show the result, not a replay of individual RANSAC iterations.'},
    {'stage': 4, 'name': 'Camera Registration and Sparse Structure-from-Motion',
     'execution_stage': 'colmap_mapping', 'artifacts': ['sparse/<model>/images.bin']},
    {'stage': 5, 'name': 'Triangulation and Bundle Adjustment',
     'execution_stage': 'colmap_mapping', 'artifacts': ['sparse/<model>/points3D.bin'],
     'note': 'Registration, triangulation and bundle adjustment are interleaved in mapper; '
             'only final sparse models are saved, not independent before/after stages.'},
    {'stage': 6, 'name': 'Dense Reconstruction Using PatchMatch Stereo',
     'execution_stage': 'colmap_patch_match',
     'artifacts': ['dense/stereo/depth_maps', 'dense/stereo/normal_maps'],
     'note': 'image_undistorter first prepares the dense workspace.'},
    {'stage': 7, 'name': 'Stereo Fusion', 'execution_stage': 'colmap_stereo_fusion',
     'artifacts': ['dense/fused.ply']},
    {'stage': 8, 'name': 'Poisson Surface Reconstruction',
     'execution_stage': 'colmap_poisson_mesher', 'artifacts': ['dense/meshed-poisson.ply']},
]


def validate_component(value, field):
    """Allow normal Windows folder names, never paths, traversal or devices."""
    if (not value or value in {'.', '..'} or value != value.strip()
            or value.endswith('.') or re.search(r'[<>:"/\\|?*\x00-\x1f]', value)):
        raise ValueError(f'{field} must be one ordinary folder name, not a path: {value!r}')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                *(f'LPT{i}' for i in range(1, 10))}
    if value.split('.')[0].upper() in reserved:
        raise ValueError(f'{field} cannot be a Windows reserved name: {value!r}')
    if '%' in value:
        raise ValueError(f'{field} cannot contain % because cmd.exe expands environment variables')


def preserve_source_metadata(dataset_root, experiment):
    """Copy provenance as evidence; do not execute its contents."""
    records = []
    for filename in ('frame_manifest.csv', 'input_video_details.json'):
        source = dataset_root / filename
        if not source.is_file():
            continue
        destination = experiment / 'source_metadata' / filename
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        records.append({'filename': filename, 'original_path': str(source),
                        'snapshot_relative_path': destination.relative_to(experiment).as_posix(),
                        'bytes': destination.stat().st_size, 'sha256': sha256(destination)})
    return records


def required_outcomes_succeeded(metrics, args):
    if metrics.get('runner_error') or metrics.get('interrupted'):
        return False
    if args.method in ('both', 'mapanything'):
        if metrics['mapanything'].get('status') != 'success' and not args.allow_mapanything_failure:
            return False
    if args.method in ('both', 'colmap'):
        colmap = metrics['colmap']
        if colmap.get('dense_status') != 'success':
            return False
        if not args.skip_mesh and colmap.get('mesh_status') != 'success':
            return False
    return True


def preflight(method='both'):
    """Read-only checks; no inference, downloads, reconstruction or files."""
    print('PREFLIGHT (no reconstruction, downloads or output files)')
    print('Python:', sys.version.replace('\n', ' '))
    print('Python executable:', sys.executable)
    print('Project root:', ROOT)
    ok = True

    def check_command(label, command, timeout=45):
        nonlocal ok
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace', timeout=timeout)
            print(f'\n{label}: exit={result.returncode}\n{(result.stdout + result.stderr).strip()}')
            if result.returncode != 0:
                ok = False
        except Exception as exc:
            print(f'\n{label}: FAILED: {exc}')
            ok = False

    check_command('NVIDIA driver/device', ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                         '--format=csv,noheader'])
    if method in ('both', 'colmap'):
        if not COLMAP.is_file():
            print('COLMAP launcher missing:', COLMAP)
            ok = False
        else:
            check_command('COLMAP version/help', colmap_command('-h'))
    if method in ('both', 'mapanything'):
        if not MAP_SCRIPT.is_file():
            print('MapAnything inference script missing:', MAP_SCRIPT)
            ok = False
        check_command('PyTorch CUDA availability', [sys.executable, '-c',
            "import json,sys,torch; ready=torch.cuda.is_available(); "
            "print(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
            "'cuda_available':ready,"
            "'device':torch.cuda.get_device_name(0) if ready else None,"
            "'capability':torch.cuda.get_device_capability(0) if ready else None,"
            "'compiled_architectures':torch.cuda.get_arch_list() if ready else []},indent=2)); "
            "sys.exit(0 if ready else 1)"])
        for package in ('torch', 'torchvision', 'mapanything'):
            try:
                print(f'{package}: {importlib.metadata.version(package)}')
            except importlib.metadata.PackageNotFoundError:
                print(f'{package}: package metadata unavailable')
                if package in ('torch', 'torchvision'):
                    ok = False
    print('\nBasic environment checks:', 'PASS' if ok else 'FAIL')
    print('Passing preflight does not prove CUDA kernel compatibility, free VRAM for all '
          'views, or reconstruction quality. Inspect actual run logs and output geometry.')
    return 0 if ok else 1


def parse_confidence_percentile(value):
    percentile = float(value)
    if not 0 <= percentile <= 100:
        raise argparse.ArgumentTypeError('confidence percentile must be finite and between 0 and 100')
    return percentile


def main(argv=None, workstation='local'):
    parser = argparse.ArgumentParser(description='Independent MapAnything + COLMAP dense MVS benchmark')
    parser.add_argument('--dataset', help='Folder name under data/; no aliases are applied')
    parser.add_argument('--scene-group', help='Output group; defaults to dataset folder name')
    parser.add_argument('--limit', type=int, default=0, help='Image count; default 0 uses all images')
    parser.add_argument('--start', type=int, default=0, help='Start index (natural filename sort)')
    parser.add_argument('--run', help='Run identifier; no existing run is overwritten')
    parser.add_argument('--profile', default='baseline',
                        help='Configuration label recorded in the run metadata')
    parser.add_argument('--expected-image-count', type=int,
                        help='Fail before creating output unless exactly this many images are selected')
    parser.add_argument('--single-camera', action='store_true',
                        help='Share intrinsics only for the same camera/lens/zoom')
    parser.add_argument('--camera-model', default='SIMPLE_RADIAL',
                        choices=['SIMPLE_PINHOLE', 'PINHOLE', 'SIMPLE_RADIAL', 'RADIAL',
                                 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV',
                                 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE'],
                        help='Explicit COLMAP camera model; default SIMPLE_RADIAL')
    parser.add_argument('--mvs-max-size', type=int, default=1000, help='Dense MVS image max dimension')
    parser.add_argument('--sift-max-size', type=int, default=1600, help='SIFT max image dimension')
    parser.add_argument('--sift-ratio', '--sift-max-ratio', dest='sift_ratio', type=float,
                        default=0.8,
                        help='COLMAP SiftMatching.max_ratio (0.8 baseline; 0.7 stricter)')
    parser.add_argument('--geom-consistency', action='store_true', help='More memory-intensive geometric MVS')
    parser.add_argument('--skip-mesh', action='store_true', help='Stop after dense fused point cloud')
    parser.add_argument('--method', choices=['both', 'mapanything', 'colmap'], default='both')
    parser.add_argument('--apply_confidence_mask', action='store_true',
                        help='MapAnything only: enable learned-confidence filtering (default off)')
    parser.add_argument('--confidence_percentile', type=parse_confidence_percentile,
                        help='MapAnything confidence cutoff percentile, 0-100; default 10 when enabled')
    parser.add_argument('--allow-mapanything-failure', action='store_true',
                        help='With --method both, allow MapAnything failure if required COLMAP stages succeed')
    parser.add_argument('--dry-run', action='store_true', help='Print selected images/settings without creating output')
    parser.add_argument('--preflight', action='store_true',
                        help='Read-only environment checks; no reconstruction or files written')
    args = parser.parse_args(argv)
    if args.method == 'colmap' and (args.apply_confidence_mask or args.confidence_percentile is not None):
        parser.error('Confidence options require --method mapanything or --method both')
    if args.confidence_percentile is not None and not args.apply_confidence_mask:
        parser.error('--confidence_percentile requires --apply_confidence_mask')
    if args.confidence_percentile is None:
        args.confidence_percentile = 10.0
    if args.dry_run and args.preflight:
        parser.error('Choose --dry-run or --preflight, not both')
    if args.allow_mapanything_failure and args.method != 'both':
        parser.error('--allow-mapanything-failure requires --method both')
    if not args.dataset:
        if args.preflight:
            return preflight(args.method)
        parser.error('--dataset is required (except with --preflight)')
    try:
        for field in ('dataset', 'scene_group', 'run', 'profile'):
            value = getattr(args, field)
            if value is not None:
                validate_component(value, field)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.start < 0 or args.limit < 0 or args.mvs_max_size <= 0 or
            args.sift_max_size <= 0 or not 0 < args.sift_ratio <= 1):
        parser.error('Invalid start, limit, image max-size or SIFT ratio value')
    if args.expected_image_count is not None and args.expected_image_count < 2:
        parser.error('--expected-image-count must be at least 2')

    group = args.scene_group or args.dataset
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
    print(f'Project root: {ROOT}\nDataset directory: {photo_folder}\n'
          f'Scene group: {group}\nRun ID: {run}\nInput count: {len(selected)}\n'
          f'Methods: {args.method}\nMVS max dimension: {args.mvs_max_size}\n'
          f'SIFT max dimension: {args.sift_max_size}\nSIFT ratio threshold: {args.sift_ratio}\n'
          f'Profile: {args.profile}\nGeometric consistency: {args.geom_consistency}\n'
          f'Allow MapAnything failure: {args.allow_mapanything_failure}\n'
          f'Camera model: {args.camera_model}; shared intrinsics: {args.single_camera}\n'
          f'Exhaustive candidate pairs: {len(selected) * (len(selected) - 1) // 2}\n'
          f'Entry point label: {workstation}')
    if args.method in ('both', 'mapanything'):
        print(f'MapAnything confidence mask: {args.apply_confidence_mask}\n'
              f'MapAnything confidence percentile: {args.confidence_percentile:g}'
              + ('' if args.apply_confidence_mask else ' (inactive; confidence mask off)'))
    for p in selected:
        print(' -', p.name)
    if args.preflight:
        return preflight(args.method)
    if not args.dry_run and args.method in ('both', 'mapanything') and not MAP_SCRIPT.is_file():
        parser.error(f'MapAnything CLI not found: {MAP_SCRIPT}')
    if not args.dry_run and args.method in ('both', 'colmap') and not COLMAP.is_file():
        parser.error(f'Standalone COLMAP launcher not found: {COLMAP}')

    exp = ROOT / 'experiments' / group / run
    ma = ROOT / 'mapanything_output' / group / run
    co = ROOT / 'colmap_output' / group / run
    # Preserve prior logs and 3D outputs, even on repeated invocation with same run name.
    for folder in (exp, ma, co):
        if folder.exists():
            parser.error(f'Output already exists: {folder}. Choose a different --run ID.')
    print('Experiment reports:', exp)
    print('COLMAP outputs:', co)
    print('MapAnything outputs:', ma)
    if args.dry_run:
        print('DRY RUN: selection and names validated; no tools launched or files written.')
        return 0
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
    provenance = preserve_source_metadata(ROOT / 'data' / args.dataset, exp)
    camera_model = args.camera_model
    settings = {'profile': args.profile, 'single_camera': args.single_camera,
                'camera_model': camera_model, 'mvs_max_size': args.mvs_max_size,
                'sift_max_size': args.sift_max_size, 'sift_ratio': args.sift_ratio,
                'geom_consistency': args.geom_consistency,
                'poisson_meshing': not args.skip_mesh, 'method': args.method,
                'allow_mapanything_failure': args.allow_mapanything_failure,
                'input_start_index': args.start, 'input_limit': args.limit,
                'expected_image_count': args.expected_image_count}
    if args.method in ('both', 'mapanything'):
        settings.update({'apply_confidence_mask': args.apply_confidence_mask,
                         'confidence_percentile': args.confidence_percentile})
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
                   'reference_calibration_used': False,
                   'reference_poses_used': False,
                   'settings': settings,
                   'entry_point': workstation, 'pipeline_stages': PIPELINE_STAGES,
                   'source_metadata': provenance,
                   'runner_sha256': sha256(Path(__file__)),
                   'entry_point_script': str(Path(sys.argv[0]).resolve()),
                   'entry_point_sha256': sha256(Path(sys.argv[0])) if Path(sys.argv[0]).is_file() else None}
    if args.method in ('both', 'mapanything'):
        environment['mapanything_inference_script_sha256'] = sha256(MAP_SCRIPT)
    for mod in ('torch', 'torchvision', 'mapanything', 'numpy', 'Pillow', 'trimesh', 'huggingface-hub'):
        try: environment[mod] = importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError: environment[mod] = None
    save_json(exp/'environment.json', environment)

    metrics = {'dataset': args.dataset, 'scene_group': group, 'run_id': run,
               'input_images': len(selected), 'input_fingerprint_sha256': fingerprint,
               'settings_fingerprint_sha256': settings_fingerprint,
               'experiment_fingerprint_sha256': experiment_fingerprint,
               'settings': settings, 'pipeline_stages': PIPELINE_STAGES,
               'source_metadata': provenance, 'entry_point': workstation,
               'mapanything': {'status': 'not_run'}, 'colmap': {'status': 'not_run'},
               'colmap_dense': {'status': 'not_run'}, 'stages': []}
    monitor = GPUSampler()
    monitor.start()
    overall_start = time.perf_counter()
    try:
        if args.method in ('both', 'mapanything'):
            glb = ma / 'scene.glb'
            confidence_options = (['--apply_confidence_mask', '--confidence_percentile',
                                   str(args.confidence_percentile)]
                                  if args.apply_confidence_mask else [])
            rc, sec, _, oom = run_step('mapanything',
                [sys.executable, '-u', MAP_SCRIPT, '--image_folder', snapshot,
                 '--save_glb', '--output_path', glb] + confidence_options,
                MAP_REPO, logs, monitor, metrics['stages'])
            valid = rc == 0 and glb.is_file() and glb.stat().st_size > 0
            obj = {'status': 'success' if valid else 'failed', 'elapsed_seconds': sec,
                   'exit_code': rc, 'gpu_oom': oom, 'output': str(glb),
                   'failure_allowed': bool(args.allow_mapanything_failure and not valid),
                   'output_bytes': glb.stat().st_size if glb.exists() else None}
            if valid:
                try: obj.update(glb_counts(glb) or {})
                except Exception as exc:
                    obj['glb_parse_error'] = str(exc)
                    obj['status'] = 'failed'
                if obj.get('vertices', 0) <= 0:
                    obj['status'] = 'failed'
                obj['failure_allowed'] = bool(args.allow_mapanything_failure and obj['status'] == 'failed')
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
                colmap.setdefault('failed_stage', 'no_valid_sparse_model')
            else:
                stats, best = candidates[0]
                colmap.update(stats)
                colmap['sparse_model'] = str(best)
                colmap['sparse_model_candidates'] = [str(p) for _, p in candidates]
                colmap['registration_rate_percent'] = round(
                    100 * stats.get('registered_images', 0) / len(selected), 2)
                colmap['sparse_status'] = 'success'
                # Dense MVS is a separate stage: COLMAP's image_undistorter creates the
                # required workspace using the estimated cameras and selected sparse model.
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
    except KeyboardInterrupt:
        metrics['interrupted'] = True
        metrics['runner_error'] = 'Interrupted by user; partial outputs retained.'
        print(metrics['runner_error'], file=sys.stderr)
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
        metrics['required_outcomes_succeeded'] = required_outcomes_succeeded(metrics, args)
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
        print('\n===== EXPERIMENT FINISHED =====')
        print('Run:', group, '/', run, '| selected images:', len(selected))
        print('MapAnything:', metrics['mapanything'].get('status'),
              '| COLMAP dense:', metrics['colmap'].get('dense_status'))
        print('COLMAP Poisson mesh:', metrics['colmap'].get('mesh_status'))
        print('Detailed results:', exp)
        print('Dense comparison summary:', master)
        print('Required outcomes succeeded:', metrics['required_outcomes_succeeded'])
    return 130 if metrics.get('interrupted') else (0 if metrics['required_outcomes_succeeded'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
