"""Verify, compare, report, and package reconstruction experiment runs.

The authoritative inputs are each run's summary.json, environment.json, and
input_manifest.csv. The historical all_runs_dense.csv is deliberately not
used because it may contain repeated rows from retried experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_RE = re.compile(r'([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
ALIASES = {'gerrad-hall': 'gerrard-hall', 'garred-hall': 'gerrard-hall'}

RUN_FIELDS = [
    'run_ref', 'dataset', 'scene_group', 'run_id', 'profile', 'input_images',
    'input_fingerprint_sha256', 'settings_fingerprint_sha256',
    'snapshot_fingerprint_sha256', 'input_snapshot_valid', 'source_files_match',
    'sift_max_size', 'sift_ratio', 'sift_ratio_source', 'mvs_max_size',
    'single_camera', 'camera_model', 'geom_consistency',
    'mapanything_status', 'mapanything_elapsed_seconds',
    'mapanything_peak_gpu_memory_mib', 'mapanything_gpu_oom',
    'mapanything_vertices', 'mapanything_triangles',
    'colmap_status', 'colmap_elapsed_seconds', 'colmap_peak_gpu_memory_mib',
    'colmap_database_images', 'colmap_keypoints', 'colmap_verified_pairs',
    'colmap_registered_images', 'colmap_registration_rate_percent',
    'colmap_sparse_points', 'colmap_mean_reprojection_error_px',
    'colmap_dense_points', 'colmap_mesh_faces', 'colmap_mesh_status',
    'total_benchmark_seconds', 'experiment_directory'
]

METRICS = {
    'mapanything.elapsed_seconds': ('mapanything_elapsed_seconds', 'lower'),
    'mapanything.peak_gpu_memory_mib': ('mapanything_peak_gpu_memory_mib', 'lower'),
    'mapanything.vertices': ('mapanything_vertices', 'neutral'),
    'mapanything.triangles': ('mapanything_triangles', 'neutral'),
    'colmap.elapsed_seconds': ('colmap_elapsed_seconds', 'lower'),
    'colmap.peak_gpu_memory_mib': ('colmap_peak_gpu_memory_mib', 'lower'),
    'colmap.keypoints': ('colmap_keypoints', 'neutral'),
    'colmap.verified_pairs': ('colmap_verified_pairs', 'higher'),
    'colmap.registered_images': ('colmap_registered_images', 'higher'),
    'colmap.registration_rate_percent': ('colmap_registration_rate_percent', 'higher'),
    'colmap.sparse_points': ('colmap_sparse_points', 'higher'),
    'colmap.mean_reprojection_error_px': ('colmap_mean_reprojection_error_px', 'lower'),
    'colmap.dense_points': ('colmap_dense_points', 'higher'),
    'colmap.mesh_faces': ('colmap_mesh_faces', 'higher'),
    'total_benchmark_seconds': ('total_benchmark_seconds', 'lower'),
}

SETTING_FIELDS = [
    'sift_max_size', 'sift_ratio', 'mvs_max_size', 'single_camera',
    'camera_model', 'geom_consistency'
]


def canonical_run_ref(value):
    match = RUN_RE.fullmatch(value.replace('\\', '/'))
    if not match:
        raise ValueError('Use group/run IDs, e.g. office_26/office_undistorted_6img')
    group, run_id = match.groups()
    return f"{ALIASES.get(group, group)}/{run_id}"


def sha256(path, cache=None):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if cache is not None and key in cache:
        return cache[key]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    value = digest.hexdigest()
    if cache is not None:
        cache[key] = value
    return value


def records_fingerprint(records):
    payload = ''.join(f"{row['filename']}\0{row['sha256']}\n" for row in records)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot read valid JSON from {path}: {exc}') from exc


def read_manifest(path):
    if not path.is_file():
        raise ValueError(f'Missing input manifest: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'filename', 'original_path', 'bytes', 'sha256'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Invalid or empty input manifest: {path}')
    return rows


def load_run(run_ref):
    run_ref = canonical_run_ref(run_ref)
    group, run_id = run_ref.split('/', 1)
    experiment = ROOT / 'experiments' / group / run_id
    summary_path = experiment / 'summary.json'
    environment_path = experiment / 'environment.json'
    if not summary_path.is_file():
        raise ValueError(f'Experiment not found: {experiment}')
    summary = read_json(summary_path)
    environment = read_json(environment_path) if environment_path.is_file() else {}
    manifest = read_manifest(experiment / 'input_manifest.csv')
    return {'run_ref': run_ref, 'group': group, 'run_id': run_id,
            'experiment': experiment, 'summary': summary,
            'environment': environment, 'manifest': manifest}


def command_setting(summary, flag, converter):
    pattern = re.compile(re.escape(flag) + r'(?:=|\s+)([^\s"]+)')
    for stage in summary.get('stages', []):
        match = pattern.search(str(stage.get('command', '')))
        if match:
            try:
                return converter(match.group(1)), 'recorded_command'
            except (TypeError, ValueError):
                pass
    return None, None


def get_settings(run):
    summary = run['summary']
    environment = run['environment']
    settings = dict(environment.get('settings') or {})
    settings.update(summary.get('settings') or {})
    sift_size = settings.get('sift_max_size')
    if sift_size is None:
        sift_size, _ = command_setting(summary, '--FeatureExtraction.max_image_size', int)
    ratio = settings.get('sift_ratio')
    ratio_source = 'recorded_settings' if ratio is not None else None
    if ratio is None:
        ratio, ratio_source = command_setting(summary, '--SiftMatching.max_ratio', float)
    if ratio is None:
        # Earlier runs omitted the flag. COLMAP 4.2 reports 0.8 as its default.
        ratio, ratio_source = 0.8, 'inferred_colmap_4_2_default'
    return {
        'profile': settings.get('profile', 'legacy_or_custom'),
        'sift_max_size': sift_size,
        'sift_ratio': ratio,
        'sift_ratio_source': ratio_source,
        'mvs_max_size': settings.get('mvs_max_size'),
        'single_camera': settings.get('single_camera'),
        'camera_model': settings.get(
            'camera_model', 'PINHOLE' if str(summary.get('dataset', '')).endswith('_undistorted')
            else 'SIMPLE_RADIAL'),
        'geom_consistency': settings.get('geom_consistency'),
    }


def verify_inputs(run, hash_cache):
    manifest = run['manifest']
    experiment = run['experiment']
    snapshot = experiment / 'input_images'
    declared_fingerprint = records_fingerprint(manifest)
    summary_fingerprint = run['summary'].get('input_fingerprint_sha256')
    environment_fingerprint = run['environment'].get('input_fingerprint_sha256')
    file_rows = []
    actual_records = []
    missing_snapshot = changed_snapshot = missing_source = changed_source = 0
    for row in manifest:
        filename = row['filename']
        expected_hash = row['sha256'].lower()
        try:
            expected_bytes = int(row['bytes'])
        except (TypeError, ValueError):
            expected_bytes = None
        snapshot_path = snapshot / filename
        source_path = Path(row['original_path'])
        snapshot_hash = source_hash = ''
        snapshot_status = source_status = 'missing'
        if snapshot_path.is_file():
            snapshot_hash = sha256(snapshot_path, hash_cache)
            size_matches = expected_bytes is None or snapshot_path.stat().st_size == expected_bytes
            snapshot_status = 'match' if snapshot_hash == expected_hash and size_matches else 'changed'
            actual_records.append({'filename': filename, 'sha256': snapshot_hash})
        else:
            missing_snapshot += 1
        if snapshot_status == 'changed':
            changed_snapshot += 1
        if source_path.is_file():
            source_hash = sha256(source_path, hash_cache)
            source_status = 'match' if source_hash == expected_hash else 'changed'
        else:
            missing_source += 1
        if source_status == 'changed':
            changed_source += 1
        file_rows.append({
            'run_ref': run['run_ref'], 'filename': filename,
            'manifest_sha256': expected_hash, 'manifest_bytes': expected_bytes,
            'snapshot_status': snapshot_status, 'snapshot_sha256': snapshot_hash,
            'source_status': source_status, 'source_sha256': source_hash,
            'source_path': str(source_path)
        })
    expected_names = {row['filename'] for row in manifest}
    actual_names = ({path.name for path in snapshot.iterdir()
                     if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
                    if snapshot.is_dir() else set())
    extra_snapshot = sorted(actual_names - expected_names)
    actual_fingerprint = (records_fingerprint(actual_records)
                          if len(actual_records) == len(manifest) else '')
    snapshot_valid = bool(
        summary_fingerprint == declared_fingerprint and
        (not environment_fingerprint or environment_fingerprint == declared_fingerprint) and
        actual_fingerprint == declared_fingerprint and
        not extra_snapshot and not missing_snapshot and not changed_snapshot
    )
    source_matches = not missing_source and not changed_source
    report = {
        'run_ref': run['run_ref'], 'manifest_files': len(manifest),
        'summary_fingerprint_sha256': summary_fingerprint,
        'environment_fingerprint_sha256': environment_fingerprint,
        'manifest_fingerprint_sha256': declared_fingerprint,
        'snapshot_fingerprint_sha256': actual_fingerprint,
        'summary_matches_manifest': summary_fingerprint == declared_fingerprint,
        'environment_matches_manifest': (not environment_fingerprint or
                                         environment_fingerprint == declared_fingerprint),
        'snapshot_matches_manifest': actual_fingerprint == declared_fingerprint,
        'missing_snapshot_files': missing_snapshot,
        'changed_snapshot_files': changed_snapshot,
        'extra_snapshot_files': len(extra_snapshot),
        'extra_snapshot_names': ';'.join(extra_snapshot),
        'input_snapshot_valid': snapshot_valid,
        'missing_source_files': missing_source,
        'changed_source_files': changed_source,
        'source_files_match': source_matches,
    }
    return report, file_rows


def run_metrics(run, verification):
    summary = run['summary']
    environment = run['environment']
    settings = get_settings(run)
    ma = summary.get('mapanything') or {}
    co = summary.get('colmap') or {}
    result = {
        'run_ref': run['run_ref'], 'dataset': summary.get('dataset'),
        'scene_group': summary.get('scene_group', run['group']),
        'run_id': summary.get('run_id', run['run_id']),
        'input_images': summary.get('input_images'),
        'input_fingerprint_sha256': summary.get('input_fingerprint_sha256'),
        'settings_fingerprint_sha256': (summary.get('settings_fingerprint_sha256') or
                                        environment.get('settings_fingerprint_sha256')),
        'snapshot_fingerprint_sha256': verification['snapshot_fingerprint_sha256'],
        'input_snapshot_valid': verification['input_snapshot_valid'],
        'source_files_match': verification['source_files_match'],
        'mapanything_status': ma.get('status'),
        'mapanything_elapsed_seconds': ma.get('elapsed_seconds'),
        'mapanything_peak_gpu_memory_mib': ma.get('peak_gpu_memory_mib'),
        'mapanything_gpu_oom': ma.get('gpu_oom'),
        'mapanything_vertices': ma.get('vertices'),
        'mapanything_triangles': ma.get('triangles'),
        'colmap_status': co.get('status'),
        'colmap_elapsed_seconds': co.get('elapsed_seconds'),
        'colmap_peak_gpu_memory_mib': co.get('peak_gpu_memory_mib'),
        'colmap_database_images': co.get('database_images'),
        'colmap_keypoints': co.get('keypoints'),
        'colmap_verified_pairs': co.get('verified_pairs'),
        'colmap_registered_images': co.get('registered_images'),
        'colmap_registration_rate_percent': co.get('registration_rate_percent'),
        'colmap_sparse_points': co.get('sparse_points'),
        'colmap_mean_reprojection_error_px': co.get('mean_reprojection_error_px'),
        'colmap_dense_points': co.get('dense_points'),
        'colmap_mesh_faces': co.get('mesh_faces'),
        'colmap_mesh_status': co.get('mesh_status'),
        'total_benchmark_seconds': summary.get('total_benchmark_seconds'),
        'experiment_directory': str(run['experiment']),
    }
    result.update(settings)
    return result


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def compare_runs(pair_index, baseline, candidate):
    base_fp = baseline.get('snapshot_fingerprint_sha256') or baseline.get('input_fingerprint_sha256')
    cand_fp = candidate.get('snapshot_fingerprint_sha256') or candidate.get('input_fingerprint_sha256')
    same_input = bool(base_fp and cand_fp and base_fp == cand_fp)
    changed_settings = [field for field in SETTING_FIELDS if baseline.get(field) != candidate.get(field)]
    pair_id = f'pair_{pair_index:02d}'
    pair_row = {
        'pair_id': pair_id, 'baseline_run': baseline['run_ref'],
        'candidate_run': candidate['run_ref'], 'same_input_fingerprint': same_input,
        'baseline_input_fingerprint_sha256': base_fp,
        'candidate_input_fingerprint_sha256': cand_fp,
        'both_input_snapshots_valid': bool(baseline['input_snapshot_valid'] and
                                           candidate['input_snapshot_valid']),
        'changed_setting_count': len(changed_settings),
        'changed_settings': ';'.join(changed_settings),
        'comparison_type': ('input_mismatch' if not same_input else
                            'controlled_single_factor' if len(changed_settings) == 1 else
                            'controlled_multi_factor' if changed_settings else
                            'repeat_same_configuration'),
        'baseline_mapanything_status': baseline.get('mapanything_status'),
        'candidate_mapanything_status': candidate.get('mapanything_status'),
        'baseline_colmap_status': baseline.get('colmap_status'),
        'candidate_colmap_status': candidate.get('colmap_status'),
    }
    differences = []
    for metric_name, (field, preferred) in METRICS.items():
        baseline_value = number(baseline.get(field))
        candidate_value = number(candidate.get(field))
        difference = (candidate_value - baseline_value
                      if baseline_value is not None and candidate_value is not None else None)
        percent = (difference / abs(baseline_value) * 100
                   if difference is not None and baseline_value != 0 else None)
        if difference is None:
            outcome = 'not_available'
        elif difference == 0:
            outcome = 'no_change'
        elif preferred == 'higher':
            outcome = 'improved' if difference > 0 else 'regressed'
        elif preferred == 'lower':
            outcome = 'improved' if difference < 0 else 'regressed'
        else:
            outcome = 'changed'
        differences.append({
            'pair_id': pair_id, 'baseline_run': baseline['run_ref'],
            'candidate_run': candidate['run_ref'], 'metric': metric_name,
            'preferred_direction': preferred, 'baseline_value': baseline_value,
            'candidate_value': candidate_value,
            'difference_candidate_minus_baseline': difference,
            'percent_difference': percent, 'outcome': outcome,
            'same_input_fingerprint': same_input,
        })
    return pair_row, differences


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def collect_model_files(runs, include_depth_maps):
    files = []
    for run in runs:
        exp = run['experiment']
        group, run_id = run['group'], run['run_id']
        files.extend(path for path in exp.rglob('*') if path.is_file() and
                     'input_images' not in path.relative_to(exp).parts and
                     path.suffix.lower() != '.zip')
        ma = ROOT / 'mapanything_output' / group / run_id / 'scene.glb'
        if ma.is_file():
            files.append(ma)
        colmap = ROOT / 'colmap_output' / group / run_id
        sparse = colmap / 'sparse'
        if sparse.is_dir():
            files.extend(path for path in sparse.rglob('*') if path.is_file())
        dense = colmap / 'dense'
        if include_depth_maps and dense.is_dir():
            files.extend(path for path in dense.rglob('*') if path.is_file())
        else:
            for name in ('fused.ply', 'meshed-poisson.ply'):
                path = dense / name
                if path.is_file():
                    files.append(path)
    return list(dict.fromkeys(files))


def markdown_report(run_rows, verification_rows, pair_rows):
    lines = ['# Reconstruction experiment review', '',
             f'Generated: {datetime.now().astimezone().isoformat(timespec="seconds")}', '',
             '## Input verification', '',
             '| Run | Snapshot valid | Current source files match | Input fingerprint |',
             '| --- | --- | --- | --- |']
    for row in verification_rows:
        lines.append(f"| {row['run_ref']} | {row['input_snapshot_valid']} | "
                     f"{row['source_files_match']} | `{row['manifest_fingerprint_sha256']}` |")
    lines += ['', '## Run metrics', '',
              '| Run | Profile | Images | SIFT max | Ratio | MapAnything | COLMAP | Registered | Dense points |',
              '| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |']
    for row in run_rows:
        lines.append(f"| {row['run_ref']} | {row['profile']} | {row['input_images']} | "
                     f"{row['sift_max_size']} | {row['sift_ratio']} | "
                     f"{row['mapanything_status']} | {row['colmap_status']} | "
                     f"{row['colmap_registered_images']} | {row['colmap_dense_points']} |")
    if pair_rows:
        lines += ['', '## Pair checks', '',
                  '| Pair | Baseline | Candidate | Same inputs | Changed settings | Type |',
                  '| --- | --- | --- | --- | --- | --- |']
        for row in pair_rows:
            lines.append(f"| {row['pair_id']} | {row['baseline_run']} | {row['candidate_run']} | "
                         f"{row['same_input_fingerprint']} | {row['changed_settings']} | "
                         f"{row['comparison_type']} |")
        lines += ['', 'Detailed numeric differences are in `metric_differences.csv`. '
                  'Positive difference means candidate minus baseline. Percent difference is '
                  'blank when the baseline is zero or either value is unavailable.']
    lines += ['', '## Interpretation limits', '',
              '- Point counts, triangle counts, and file sizes do not measure geometric accuracy.',
              '- A comparison with more than one changed setting cannot isolate one setting as the cause.',
              '- A failed MapAnything runtime is time to failure, not a completed reconstruction time.',
              '- Whole-GPU memory samples include other Windows WDDM processes.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Verify inputs, compare metrics, export CSV reports, and package models')
    parser.add_argument('--runs', nargs='*', default=[],
                        help='group/run references to inspect and package')
    parser.add_argument('--compare', nargs=2, action='append', default=[],
                        metavar=('BASELINE', 'CANDIDATE'),
                        help='Pair to compare; repeat for multiple scenes')
    parser.add_argument('--include-depth-maps', action='store_true',
                        help='Include potentially very large dense/stereo intermediate files')
    parser.add_argument('--report-only', action='store_true',
                        help='Write verification/comparison reports without creating a ZIP')
    parser.add_argument('--allow-input-mismatch', action='store_true',
                        help='Allow pair differences even when the selected images differ')
    parser.add_argument('--allow-invalid-inputs', action='store_true',
                        help='Continue when a run snapshot no longer matches its manifest')
    parser.add_argument('--label', default='review',
                        help='Short safe label used in report and ZIP names')
    parser.add_argument('--dry-run', action='store_true',
                        help='Verify and print the plan without writing reports or a ZIP')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.label):
        parser.error('--label may contain only letters, numbers, underscores and hyphens')
    refs = []
    try:
        for value in args.runs:
            refs.append(canonical_run_ref(value))
        comparisons = []
        for baseline, candidate in args.compare:
            baseline_ref, candidate_ref = canonical_run_ref(baseline), canonical_run_ref(candidate)
            comparisons.append((baseline_ref, candidate_ref))
            refs.extend([baseline_ref, candidate_ref])
    except ValueError as exc:
        parser.error(str(exc))
    refs = list(dict.fromkeys(refs))
    if not refs:
        parser.error('provide at least one --runs item or --compare pair')
    try:
        runs = [load_run(ref) for ref in refs]
    except ValueError as exc:
        parser.error(str(exc))
    hash_cache = {}
    verification_rows, input_file_rows, run_rows = [], [], []
    for run in runs:
        verification, file_rows = verify_inputs(run, hash_cache)
        verification_rows.append(verification)
        input_file_rows.extend(file_rows)
        run_rows.append(run_metrics(run, verification))
    invalid = [row['run_ref'] for row in verification_rows if not row['input_snapshot_valid']]
    if invalid and not args.allow_invalid_inputs:
        parser.error('input snapshot verification failed for: ' + ', '.join(invalid) +
                     '. Use --allow-invalid-inputs only after reviewing the run files.')

    metrics_by_ref = {row['run_ref']: row for row in run_rows}
    pair_rows, difference_rows = [], []
    mismatches = []
    for index, (baseline_ref, candidate_ref) in enumerate(comparisons, 1):
        pair_row, differences = compare_runs(
            index, metrics_by_ref[baseline_ref], metrics_by_ref[candidate_ref])
        pair_rows.append(pair_row)
        difference_rows.extend(differences)
        if not pair_row['same_input_fingerprint']:
            mismatches.append(f'{baseline_ref} vs {candidate_ref}')
    if mismatches and not args.allow_input_mismatch:
        parser.error('comparison input fingerprints differ: ' + ', '.join(mismatches) +
                     '. Use --allow-input-mismatch only for non-controlled scaling comparisons.')

    print('Verified runs:', len(run_rows))
    for row in run_rows:
        print(f" - {row['run_ref']}: inputs_valid={row['input_snapshot_valid']} "
              f"images={row['input_images']} COLMAP={row['colmap_status']} "
              f"MapAnything={row['mapanything_status']}")
    for row in pair_rows:
        print(f" - {row['pair_id']}: same_inputs={row['same_input_fingerprint']} "
              f"changed={row['changed_settings']} type={row['comparison_type']}")
    if args.dry_run:
        print('Dry run only: no report or ZIP was created.')
        return 0

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    report_dir = ROOT / 'experiments' / 'reports' / f'{args.label}_{stamp}'
    report_dir.mkdir(parents=True)
    write_csv(report_dir / 'run_metrics.csv', run_rows, RUN_FIELDS)
    write_csv(report_dir / 'input_fingerprint_verification.csv', verification_rows,
              list(verification_rows[0].keys()))
    write_csv(report_dir / 'input_file_verification.csv', input_file_rows,
              list(input_file_rows[0].keys()))
    if pair_rows:
        write_csv(report_dir / 'comparison_pairs.csv', pair_rows, list(pair_rows[0].keys()))
        write_csv(report_dir / 'metric_differences.csv', difference_rows,
                  list(difference_rows[0].keys()))
    report_json = {'generated': datetime.now().astimezone().isoformat(timespec='seconds'),
                   'runs': run_rows, 'input_verification': verification_rows,
                   'comparison_pairs': pair_rows, 'metric_differences': difference_rows}
    (report_dir / 'review_report.json').write_text(
        json.dumps(report_json, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
    (report_dir / 'review_report.md').write_text(
        markdown_report(run_rows, verification_rows, pair_rows), encoding='utf-8')

    print('CSV and review reports:', report_dir)
    if args.report_only:
        return 0

    model_files = collect_model_files(runs, args.include_depth_maps)
    report_files = [path for path in report_dir.iterdir() if path.is_file()]
    package_rows = [{'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size}
                    for path in model_files + report_files]
    write_csv(report_dir / 'package_manifest.csv', package_rows, ['path', 'bytes'])
    report_files.append(report_dir / 'package_manifest.csv')
    target = ROOT / 'experiments' / f'models_for_review_{args.label}_{stamp}.zip'
    unique = list(dict.fromkeys(model_files + report_files))
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=2, allowZip64=True) as archive:
        for path in unique:
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
    print(f'Review ZIP created: {target}')
    print(f'File count: {len(unique)}')
    print(f'Size: {target.stat().st_size / (1024 * 1024):.1f} MiB')
    if not args.include_depth_maps:
        print('Intermediate dense/stereo depth and normal maps were not archived.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
