"""Optional GT distance metrics for an *already aligned* reconstruction.

REQUIRES explicit model-to-reference 4x4 transform, NumPy, SciPy, trimesh.
The transform must be estimated/justified separately; this tool does not align.
ETH3D visibility/occlusion masks and evaluation protocol are NOT reproduced here.

Example:
  python scripts/evaluate_aligned_geometry.py --reference data/ETH3D_GT/office_eval.ply \
    --prediction experiments/office_26/office_undistorted_6img/samples/.../mapanything_surface_samples.ply \
    --transform docs/office_mapanything_to_ETH3D.txt --threshold 0.05
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def get_points(file: Path, limit: int, rng):
    mesh = trimesh.load(file,process=False)
    if isinstance(mesh,trimesh.Scene):
        vertices = np.concatenate([g.vertices for g in mesh.geometry.values() if len(g.vertices)])
    else:
        vertices = np.asarray(mesh.vertices)
    if not len(vertices):
        raise ValueError(f'Empty point cloud: {file}')
    if not np.isfinite(vertices).all():
        vertices = vertices[np.isfinite(vertices).all(axis=1)]
    ind = rng.choice(len(vertices),min(len(vertices),limit),replace=False)
    return vertices[ind].astype(np.float64)


def main():
    ap=argparse.ArgumentParser(description='Quantitative geometry metrics after external alignment; NOT ETH3D official evaluator')
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--prediction',type=Path,required=True)
    ap.add_argument('--transform',type=Path,required=True,help='Explicit 4x4 similarity/rigid transform mapping predicted to reference coordinates')
    ap.add_argument('--threshold',type=float,default=0.05,help='Reference-coordinate units (e.g. 0.05 meters)')
    ap.add_argument('--max-pred',type=int,default=50000)
    ap.add_argument('--max-ref',type=int,default=100000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.threshold<=0 or args.max_pred<100 or args.max_ref<100:
        ap.error('Threshold/sample counts must be positive and >=100')
    transform=np.loadtxt(args.transform)
    if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3,:],[0,0,0,1]):
        ap.error('Transform file must contain 4 lines of 4 finite numbers, last line 0 0 0 1')
    a=transform[:3,:3]
    determinant=float(np.linalg.det(a))
    if determinant<=0:
        ap.error('Invalid/reflection/non-invertible transform (requires positive determinant)')
    sv=np.linalg.svd(a)[1]
    uniform_scale=float(sv.mean())
    if not np.allclose(sv,uniform_scale,rtol=1e-3):
        ap.error('Transform must be rigid/similarity (uniform scale); anisotropic transforms not accepted')
    rng=np.random.default_rng(args.seed)
    pred=get_points(args.prediction,args.max_pred,rng)
    ref=get_points(args.reference,args.max_ref,rng)
    pred_world=pred@a.T+transform[:3,3]
    pred_dist=cKDTree(ref).query(pred_world,k=1,workers=-1)[0]
    ref_dist=cKDTree(pred_world).query(ref,k=1,workers=-1)[0]
    precision=float((pred_dist<=args.threshold).mean())
    recall=float((ref_dist<=args.threshold).mean())
    fscore=2*precision*recall/(precision+recall) if precision+recall else 0
    obj={
      'reference_path':str(args.reference),'prediction_path':str(args.prediction),
      'transform_path':str(args.transform),'transform_SHA256':hashlib.sha256(args.transform.read_bytes()).hexdigest(),
      'transform_uniform_scale':uniform_scale,'threshold_reference_units':args.threshold,
      'reference_units_assumed':'user asserts metres if threshold interpreted as metres',
      'alignment': 'externally supplied transform; validity must be independently assessed',
      'mean_pred_to_reference':float(pred_dist.mean()),
      'mean_reference_to_pred':float(ref_dist.mean()),
      'symmetric_chamfer_L1':float(pred_dist.mean()+ref_dist.mean())/2,
      'median_pred_to_ref':float(np.median(pred_dist)),
      'p95_pred_to_ref':float(np.percentile(pred_dist,95)),
      'precision_within_threshold':precision,
      'completeness_within_threshold':recall,
      'fscore':fscore,
      'pred_sample_count':len(pred),'ref_sample_count':len(ref),'seed':args.seed,
      'limitations':['Requires correct alignment and comparable units',
        'Samples from PLY points, NOT the official ETH3D visibility/occlusion protocol',
        'Fit-to-ground-truth similarity scaling removes the ability to test native metric scale',
        'Unobserved surfaces and scanner visibility bias raw nearest-neighbour scores']}
    out=args.prediction.parent/f'gt_eval_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    out.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print('Saved:',out)
    for k in ['symmetric_chamfer_L1','precision_within_threshold','completeness_within_threshold','fscore','transform_uniform_scale']:
        print(k,obj[k])

if __name__=='__main__':
    main()
