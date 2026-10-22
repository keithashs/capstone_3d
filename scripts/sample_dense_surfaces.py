"""Create fixed-size comparable *representations*, NOT a geometry accuracy score.

Requires trimesh and numpy (optional):
  python -m pip install trimesh
  python scripts/sample_dense_surfaces.py --dataset office_26 --run office_undistorted_6img --samples 50000

Neither model is aligned to ground truth by this script. Per-view MapAnything mesh
may have overlapping/duplicate surfaces; uniform surface samples inherit that bias.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parent.parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True,help='office_26 or kicker_31 output group')
    ap.add_argument('--run',required=True)
    ap.add_argument('--samples',type=int,default=50000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.samples<100 or any(re.fullmatch(r'[a-zA-Z0-9_-]+',s) is None for s in [args.dataset,args.run]):
        ap.error('Invalid parameters')
    scene=args.dataset.removesuffix('_undistorted')
    glb=ROOT/'mapanything_output'/scene/args.run/'scene.glb'
    ply=ROOT/'colmap_output'/scene/args.run/'dense'/'fused.ply'
    if not glb.is_file() or not ply.is_file():
        ap.error('Missing MapAnything scene.glb or COLMAP dense/fused.ply')
    output=ROOT/'experiments'/scene/args.run/'samples'/datetime.now().strftime('sample_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    ma=trimesh.load(glb,force='scene',process=False)
    meshes=[]
    for node in ma.graph.nodes_geometry:
        transform,name=ma.graph[node]
        mesh=ma.geometry[name]
        if isinstance(mesh,trimesh.Trimesh) and len(mesh.faces) and mesh.area>0:
            copy=mesh.copy()
            copy.apply_transform(transform)
            meshes.append(copy)
    if not meshes:
        ap.error('MapAnything GLB has no usable triangle surfaces')
    areas=np.array([mesh.area for mesh in meshes],dtype=float)
    rng=np.random.default_rng(args.seed)
    allocation=rng.multinomial(args.samples,areas/areas.sum())
    # Deterministic area-weighted barycentric sampling (same NumPy seed each run).
    parts = []
    for mesh, n in zip(meshes, allocation):
        if n <= 0:
            continue
        face_area = np.asarray(mesh.area_faces, dtype=float)
        face_ids = rng.choice(len(mesh.faces), size=int(n), p=face_area/face_area.sum())
        tri = np.asarray(mesh.vertices)[mesh.faces[face_ids]]
        u = np.sqrt(rng.random(int(n)))[:, None]
        v = rng.random(int(n))[:, None]
        pts = (1-u)*tri[:, 0] + u*(1-v)*tri[:, 1] + u*v*tri[:, 2]
        parts.append(pts)
    cloud = np.concatenate(parts)
    ma_file=output/'mapanything_surface_samples.ply'
    trimesh.points.PointCloud(cloud).export(ma_file)
    dense=trimesh.load(ply,process=False)
    vertices=np.array(dense.vertices)
    if len(vertices)==0:
        ap.error('COLMAP dense fused PLY is empty')
    indices=rng.choice(len(vertices),args.samples,replace=len(vertices)<args.samples)
    co_file=output/'colmap_dense_samples.ply'
    trimesh.points.PointCloud(vertices[indices]).export(co_file)
    obj={'dataset':scene,'run_id':args.run,'sample_count':args.samples,'seed':args.seed,
         'mapanything_surface':str(ma_file),'colmap_dense':str(co_file),
         'colmap_original_dense_points':len(vertices),
         'aligned':False,
         'accuracy_measurement':'NOT AVAILABLE - register both samples to ETH3D laser ground truth first',
         'warning':'MapAnything GLB includes overlapping per-view surfaces; surface sampling is area-weighted and may duplicate surfaces'}
    (output/'sampling.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print('Saved common point-cloud representations to',output)
    print('Unaligned point clouds must NOT be compared for metric accuracy.')

if __name__=='__main__':
    main()
