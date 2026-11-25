"""Render saved COLMAP stages for the Office 26 Blackwell baseline.

Run in the mapanything environment:
    python scripts/plot_office_blackwell_stages.py

Requires numpy, matplotlib, Pillow and plyfile (no CUDA or COLMAP executable).
Reads the completed reconstruction only. Creates a NEW timestamped review folder
with nine PNGs, README.md and manifest.json. No reconstruction, SIFT detection,
matching, RANSAC, bundle adjustment, fusion or meshing is executed.

Format references:
https://colmap.github.io/database.html
https://colmap.github.io/format.html
https://github.com/colmap/colmap/blob/main/src/colmap/mvs/fusion.cc
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from PIL import Image
from plyfile import PlyData


RUN = "office_full26_sift1600_ratio080_blackwell_fullres_v1"
PAIR = ("DSC_0220.JPG", "DSC_0229.JPG")
MAP_IMAGES = ("DSC_0220.JPG", "DSC_0229.JPG", "DSC_0242.JPG", "DSC_0255.JPG")
PAIR_BASE = 2147483647
FILES = ["01_features.png", "02_raw_matches.png", "03_verified_matches.png",
         "04_camera_registration.png", "05_sparse_points.png", "06_depth_maps.png",
         "07_dense_fusion.png", "08_poisson_mesh.png", "normal_maps.png"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints(paths, root):
    return {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size,
                                          "sha256": sha256(p)}
            for p in sorted(set(paths))}


def read_exact(stream, count):
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("Truncated COLMAP binary input")
    return value


def unpack(stream, fmt):
    return struct.unpack(fmt, read_exact(stream, struct.calcsize(fmt)))


def read_sparse(folder):
    """Read the saved final world-to-camera poses and sparse landmarks."""
    cameras = []
    with (folder / "images.bin").open("rb") as stream:
        for _ in range(unpack(stream, "<Q")[0]):
            image_id, w, x, y, z, tx, ty, tz, _ = unpack(stream, "<I7dI")
            name = bytearray()
            while True:
                char = read_exact(stream, 1)
                if char == b"\0":
                    break
                name.extend(char)
            read_exact(stream, unpack(stream, "<Q")[0] * 24)
            q = np.array([w, x, y, z])
            if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
                raise ValueError("Invalid camera quaternion")
            w, x, y, z = q / np.linalg.norm(q)
            rotation = np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            cameras.append({"id": image_id, "name": name.decode("utf-8"),
                            "R": rotation, "center": -rotation.T @ [tx, ty, tz]})
        if stream.read(1):
            raise ValueError("Unexpected trailing data in images.bin")
    points, colours, errors, tracks = [], [], [], []
    with (folder / "points3D.bin").open("rb") as stream:
        for _ in range(unpack(stream, "<Q")[0]):
            record = unpack(stream, "<Q3d3Bd")
            track = unpack(stream, "<Q")[0]
            read_exact(stream, track * 8)
            points.append(record[1:4]); colours.append(record[4:7])
            errors.append(record[7]); tracks.append(track)
        if stream.read(1):
            raise ValueError("Unexpected trailing data in points3D.bin")
    return (cameras, np.asarray(points), np.asarray(colours) / 255.,
            np.asarray(errors), np.asarray(tracks))


def read_map(path, channels):
    """COLMAP mixed text/binary layout; channel-major float32 payload."""
    with path.open("rb") as stream:
        header = bytearray()
        while header.count(b"&") < 3:
            header.extend(read_exact(stream, 1))
            if len(header) > 100:
                raise ValueError(f"Invalid dense-map header: {path}")
        width, height, actual_channels = map(int, header[:-1].split(b"&"))
        if width < 1 or height < 1 or actual_channels != channels:
            raise ValueError(f"Invalid dense-map dimensions: {path}")
        data = np.frombuffer(stream.read(), dtype="<f4")
    if data.size != width * height * channels:
        raise ValueError(f"Incorrect dense-map payload size: {path}")
    return data.reshape((width, height, channels), order="F").transpose(1, 0, 2)


def read_ply(path, mesh=False):
    # plyfile honours the actual list-count type (this mesh uses int, not uchar).
    ply = PlyData.read(str(path), mmap=False)
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex[k] for k in ("x", "y", "z")])
    rgb = np.column_stack([vertex[k] for k in ("red", "green", "blue")]) / 255.
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError(f"Empty/nonfinite geometry: {path}")
    faces = None
    if mesh:
        rows = ply["face"].data["vertex_indices"]
        if not len(rows) or any(len(row) != 3 for row in rows):
            raise ValueError("This renderer requires a nonempty triangle mesh")
        faces = np.stack(rows).astype(np.int64)
        if faces.min() < 0 or faces.max() >= len(xyz):
            raise ValueError("Mesh contains invalid vertex indices")
    return xyz, rgb, faces


def read_database(path):
    wal = path.with_name(path.name + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise ValueError("Nonempty WAL present: use a completed, closed database snapshot")
    # Immutable prevents SQLite from creating journal/shared-memory sidecars.
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        images = {name: image_id for image_id, name in
                  connection.execute("SELECT image_id,name FROM images")}
        order = sorted(PAIR, key=images.__getitem__)
        ids = [images[name] for name in order]
        keypoints = {}
        for name in order:
            rows, cols, blob = connection.execute(
                "SELECT rows,cols,data FROM keypoints WHERE image_id=?", (images[name],)).fetchone()
            # Current records have six affine columns. Do not invent SIFT angles.
            if cols not in (2, 4, 6):
                raise ValueError(f"Unsupported keypoint layout: {cols}")
            keypoints[name] = np.frombuffer(blob, dtype="<f4").reshape(rows, cols)
        pair_id = ids[0] * PAIR_BASE + ids[1]
        matches = []
        stats = {}
        for table in ("matches", "two_view_geometries"):
            rows, cols, blob = connection.execute(
                f"SELECT rows,cols,data FROM {table} WHERE pair_id=?", (pair_id,)).fetchone()
            if cols != 2:
                raise ValueError("Unexpected match column count")
            array = np.frombuffer(blob, dtype="<u4").reshape(rows, cols)
            if any(array[:, k].max(initial=0) >= len(keypoints[name]) for k, name in enumerate(order)):
                raise ValueError("Match index outside keypoint array")
            matches.append(array)
            total, positive = connection.execute(
                f"SELECT COALESCE(SUM(rows),0),COUNT(*) FROM {table} WHERE rows>0").fetchone()
            stats[table] = {"correspondences": total, "nonempty_pairs": positive}
        stats["keypoints"] = connection.execute("SELECT SUM(rows) FROM keypoints").fetchone()[0]
    finally:
        connection.close()
    raw, verified = matches
    if not set(map(tuple, verified)).issubset(set(map(tuple, raw))):
        raise ValueError("Verified matches are not a subset; cannot use paired display sampling")
    return images, order, keypoints, raw, verified, stats


def preview(path, max_width=1200):
    with Image.open(path) as image:
        original_size = image.size
        width = min(image.width, max_width)
        size = (width, round(image.height * width / image.width))
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image), np.asarray(size) / original_size


def canvas_figure(title, size=(15, 9)):
    fig = plt.figure(figsize=size, facecolor="white")
    fig.suptitle(title, x=.055, y=.97, ha="left", fontsize=21, weight="bold", color="#142f43")
    return fig


def save_figure(fig, path, caption, dpi):
    fig.text(.055, .028, caption, fontsize=10, color="#465664", va="bottom")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {path.name}", flush=True)


def features_plot(output, snapshot, order, keypoints, dpi):
    fig = canvas_figure("01  SIFT features | saved 2D observations", (15, 11))
    axes = fig.subplots(2, 2)
    for row, name in enumerate(order):
        image, scale = preview(snapshot / name)
        points = keypoints[name][:, :2] * scale
        for col in range(2):
            ax = axes[row, col]
            ax.imshow(image, extent=(0, image.shape[1], image.shape[0], 0))
            ax.set_axis_off()
            ax.set_title(f"{name} | " + ("original photograph" if col == 0 else
                         f"all {len(points):,} keypoint centres"), loc="left", fontsize=12)
        axes[row, 1].scatter(points[:, 0], points[:, 1], s=7, facecolors="none",
                            edgecolors="#ffcc00", linewidths=.5)
    fig.subplots_adjust(left=.055, right=.96, bottom=.105, top=.90, wspace=.05, hspace=.15)
    save_figure(fig, output / FILES[0],
        "Original-image coordinates scaled to the preview. Yellow rings mark centres only.\n"
        "37,531 features across all 26 inputs; these are 2D measurements, not 3D points. No features recomputed.", dpi)


def matches_plots(output, snapshot, order, keypoints, raw, verified, limit, dpi):
    previews = [preview(snapshot / name) for name in order]
    gap = 36
    width = sum(image.shape[1] for image, _ in previews) + gap
    height = max(image.shape[0] for image, _ in previews)
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    locations = []
    offset = 0
    for name, (image, scale) in zip(order, previews):
        canvas[:image.shape[0], offset:offset+image.shape[1]] = image
        locations.append(keypoints[name][:, :2] * scale + [offset, 0])
        offset += image.shape[1] + gap
    chosen = np.sort(np.random.default_rng(0).choice(len(raw), min(limit, len(raw)), replace=False))
    raw_display = raw[chosen]
    verified_set = set(map(tuple, verified))
    verified_display = np.asarray([p for p in raw_display if tuple(p) in verified_set], dtype=np.uint32).reshape(-1, 2)
    for index, name, rows, total, colour in [
            (1, "Raw feature matches", raw_display, len(raw), "#45baff"),
            (2, "Geometrically verified matches", verified_display, len(verified), "#54f0a8")]:
        fig = canvas_figure(f"0{index+1}  {name}", (16, 8))
        ax = fig.add_axes([.045, .19, .91, .64])
        ax.imshow(canvas, extent=(0, width, height, 0))
        segments = np.stack([locations[0][rows[:, 0]], locations[1][rows[:, 1]]], axis=1)
        ax.add_collection(LineCollection(segments, colors=colour, linewidths=.6, alpha=.68))
        ax.scatter(segments[:, :, 0], segments[:, :, 1], s=4, c=colour, linewidths=0)
        ax.set_axis_off()
        fig.text(.055, .87, f"{order[0]}  <->  {order[1]}     |     {total:,} saved correspondences; {len(rows)} displayed",
                 fontsize=13, color="#142f43")
        detail = ("Candidates from database.db / matches; visual similarity alone is not verification."
                  if index == 1 else
                  f"database.db / two_view_geometries: {len(raw)} raw -> {len(verified)} verified; {len(raw)-len(verified)} excluded.")
        save_figure(fig, output / FILES[index], detail +
            "\nFixed random seed; stage 03 shows only verified members of the SAME sampled candidates as stage 02.\n"
            "Lines are display overlays. No matching or RANSAC was rerun.", dpi)
    return {"raw_displayed": len(raw_display), "verified_displayed": len(verified_display), "seed": 0}


def setup_3d(ax, bounds, args):
    low, high = bounds
    mid = (low + high) / 2
    span = np.maximum(high - low, .1)
    for setter, m, s in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid, span):
        setter(m - .54*s, m + .54*s)
    ax.set_box_aspect(span)
    ax.view_init(elev=args.elev, azim=args.azim, roll=args.roll)
    ax.set_axis_off()


def frustums(ax, cameras, length, labels=False):
    local = length * np.array([[-.75,-.5,1], [.75,-.5,1], [.75,.5,1], [-.75,.5,1]])
    for camera in cameras:
        center = camera["center"]
        corners = local @ camera["R"] + center
        lines = [[center, corner] for corner in corners]
        lines += [[corners[k], corners[(k+1) % 4]] for k in range(4)]
        ax.add_collection3d(Line3DCollection(lines, colors="#d52c53", linewidths=.8))
        if labels:
            ax.text(*center, str(camera["id"]), fontsize=7)


def cameras_plot(output, images, cameras, args):
    fig = canvas_figure("04  Camera registration | final saved poses", (15, 10))
    ax = fig.add_axes([.035, .15, .53, .71], projection="3d")
    centers = np.array([c["center"] for c in cameras])
    margin = max(np.ptp(centers, axis=0).max() * .12, .1)
    bounds = (centers.min(axis=0) - margin, centers.max(axis=0) + margin)
    setup_3d(ax, bounds, args)
    ax.scatter(*centers.T, s=26, c="#d52c53")
    frustums(ax, cameras, margin * .45, labels=True)
    ax.set_title("20 estimated camera positions + viewing directions\nLabels are database image IDs", fontsize=12)
    registered = {c["name"] for c in cameras}
    fig.text(.61, .85, "26 input photographs", fontsize=15, weight="bold")
    for row, (name, image_id) in enumerate(sorted(images.items(), key=lambda p:p[1])):
        present = name in registered
        fig.text(.61, .81 - row*.0235, f"{image_id:02d}  {name}   {'registered' if present else 'not in retained model'}",
                 fontsize=10, family="monospace", color="#176f5b" if present else "#b33a27")
    save_figure(fig, output / FILES[3],
        "Camera positions are read from sparse/0/images.bin; these are the FINAL saved poses, not an initial stage snapshot.\n"
        "Six photographs are absent from the retained model. Camera coordinates are arbitrary SfM units, not verified metres.", args.dpi)


def geometry_plot(output, index, xyz, rgb, bounds, focus, args, faces=None):
    names = {4: "05  Sparse points | final triangulation and bundle adjustment",
             6: "07  Stereo fusion | dense point cloud",
             7: "08  Poisson reconstruction | triangle mesh"}
    fig = canvas_figure(names[index], (16, 9))
    counts = []
    for panel, region in enumerate((bounds, focus)):
        ax = fig.add_axes([.03 + panel*.49, .16, .46, .68], projection="3d")
        setup_3d(ax, region, args)
        inside = np.ones(len(xyz), dtype=bool) if panel == 0 else ((xyz >= region[0]) & (xyz <= region[1])).all(axis=1)
        if faces is None:
            ax.scatter(*xyz[inside].T, c=rgb[inside], s=3 if index == 4 else .6,
                       depthshade=False, linewidths=0, rasterized=True)
            count = int(inside.sum()); unit = "points"
        else:
            keep = np.ones(len(faces), dtype=bool) if panel == 0 else inside[faces].all(axis=1)
            triangles = xyz[faces[keep]]
            colours = rgb[faces[keep]].mean(axis=1)
            normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            norms = np.linalg.norm(normals, axis=1)
            normals = normals / np.maximum(norms[:, None], 1e-12)
            light = np.array([.3, -.4, .85]); light /= np.linalg.norm(light)
            colours *= (.55 + .45 * np.abs(normals @ light))[:, None]
            ax.add_collection3d(Poly3DCollection(triangles, facecolors=colours, edgecolors="none",
                                                linewidths=0, zsort="average", rasterized=True))
            count = int(keep.sum()); unit = "triangles"
        counts.append(count)
        ax.set_title(("Full extent: " if panel == 0 else "Display crop: ") + f"{count:,} {unit}\n" +
                     ("All geometry retained in this panel" if panel == 0 else
                      "Inside the dense cloud's 1st-99th percentile bounds"), fontsize=12)
    note = ("Final optimized sparse model only; no saved pre-adjustment geometry exists."
            if index == 4 else "Samples are separate coloured 3D points; no triangle surfaces."
            if index == 6 else "Original triangle faces rendered with vertex colours and display lighting; no remeshing or decimation.")
    save_figure(fig, output / FILES[index], note +
        "\nStages 05 / 07 / 08 share bounds, view angles and display crop. Right-hand crop is visualization only.\n"
        "Source files are unchanged. Scale is arbitrary SfM units, not verified metres.", args.dpi)
    return {"full_display_count": counts[0], "crop_display_count": counts[1]}


def maps_plot(output, dense, map_data, normal, args):
    fig = canvas_figure("Normal maps | saved surface orientations" if normal else
                        "06  PatchMatch Stereo | saved depth maps", (13, 15))
    axes = fig.subplots(len(MAP_IMAGES), 2)
    all_depths = np.concatenate([depth[valid] for depth, _, valid in map_data.values()])
    low, high = np.percentile(all_depths, [2, 98])
    if high <= low:
        high = low + 1e-6
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#333b44")
    details = {}
    for row, name in enumerate(MAP_IMAGES):
        depth, normals, valid = map_data[name]
        image, _ = preview(dense / "images" / name, max_width=1000)
        if image.shape[:2] != depth.shape:
            raise ValueError(f"Dense image/map dimensions differ: {name}")
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{name} | dense workspace image", fontsize=11, loc="left")
        if normal:
            norm = np.linalg.norm(normals, axis=-1)
            shown = valid & np.isfinite(normals).all(axis=-1) & (norm > 1e-8)
            encoded = np.full(normals.shape, .20)
            # Stored valid normals are unit length; normalise only for RGB display.
            encoded[shown] = np.clip((normals[shown] / norm[shown, None] + 1) / 2, 0, 1)
            axes[row, 1].imshow(encoded)
            kind = "camera-frame normals"
        else:
            shown = valid
            axes[row, 1].imshow(np.ma.array(depth, mask=~valid), cmap=cmap,
                                norm=Normalize(low, high), interpolation="nearest")
            kind = "positive finite depth"
        axes[row, 1].set_title(f"{kind} | {shown.sum():,}/{shown.size:,} pixels", fontsize=11, loc="left")
        for ax in axes[row]:
            ax.set_axis_off()
        details[name] = {"width": int(depth.shape[1]), "height": int(depth.shape[0]),
                         "valid_pixels_displayed": int(shown.sum()), "total_pixels": int(shown.size),
                         "positive_depth_min": float(depth[valid].min()), "positive_depth_max": float(depth[valid].max())}
    fig.subplots_adjust(left=.055, right=.945, top=.915, bottom=.115, wspace=.045, hspace=.19)
    if not normal:
        cax = fig.add_axes([.58, .08, .34, .012])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(low, high), cmap=cmap), cax=cax,
                           orientation="horizontal", extend="both")
        bar.set_label("Depth (arbitrary SfM units)", fontsize=9)
        bar.ax.tick_params(labelsize=8)
    caption = ("RGB = (unit normal + 1) / 2: R encodes camera X, G camera Y, B camera Z.\n"
               "Normals are camera-relative; equal colours in different views need not mean equal world orientation."
               if normal else
               "Shared 2nd-98th percentile colour limits across these four maps; extremes saturate, not removed.\n"
               "Depth is not calibrated in metres. Positive depth does not guarantee successful stereo fusion.")
    save_figure(fig, output / ("normal_maps.png" if normal else FILES[5]), caption +
        "\nDark grey marks invalid pixels. Four illustrative views from 20 saved photometric map pairs; no maps recomputed.", args.dpi)
    return {"views": details, "depth_colour_limits": [float(low), float(high)] if not normal else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="New review folder inside this experiment; must not already exist")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-match-lines", type=int, default=120, help="Deterministic display sample; saved matches unchanged")
    parser.add_argument("--elev", type=float, default=20)
    parser.add_argument("--azim", type=float, default=-62)
    parser.add_argument("--roll", type=float, default=0)
    args = parser.parse_args()
    if args.dpi < 72 or args.max_match_lines < 1:
        parser.error("--dpi must be >=72 and --max-match-lines must be positive")
    root = args.project_root.resolve()
    reconstruction = root / "colmap_output" / "office_26" / RUN
    experiment = root / "experiments" / "office_26" / RUN
    snapshot = experiment / "input_images"
    summary_path = experiment / "summary.json"
    dense = reconstruction / "dense"
    output = (args.output_dir or experiment / f"stage_plots_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    # A new direct child avoids touching logs, input snapshots or reconstruction data.
    if output.parent != experiment.resolve() or output.exists():
        parser.error("Output must be a NEW direct child directory of this experiment")
    protected_paths = sorted(p for p in reconstruction.rglob("*") if p.is_file())
    protected_paths += sorted(p for p in snapshot.rglob("*") if p.is_file()) + [summary_path]
    print("Hashing original reconstruction, input snapshot and summary...", flush=True)
    before = fingerprints(protected_paths, root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["run_id"] != RUN or summary["input_images"] != 26:
        raise ValueError("Summary does not describe the expected Office 26 run")
    images, order, keypoints, raw, verified, database_stats = read_database(reconstruction / "database.db")
    cameras, sparse_xyz, sparse_rgb, errors, tracks = read_sparse(reconstruction / "sparse" / "0")
    dense_xyz, dense_rgb, _ = read_ply(dense / "fused.ply")
    mesh_xyz, mesh_rgb, faces = read_ply(dense / "meshed-poisson.ply", mesh=True)
    measured = {"input_images": len(images), "keypoints": database_stats["keypoints"],
                "registered_images": len(cameras), "sparse_points": len(sparse_xyz),
                "dense_points": len(dense_xyz), "mesh_vertices": len(mesh_xyz), "mesh_faces": len(faces),
                "observations": int(tracks.sum()), "mean_point_reprojection_error_px": float(errors.mean())}
    for key in ("keypoints", "registered_images", "sparse_points", "dense_points", "mesh_faces", "observations"):
        if measured[key] != summary["colmap"][key]:
            raise ValueError(f"Saved geometry disagrees with summary: {key}")
    if not np.isfinite(sparse_xyz).all():
        raise ValueError("Nonfinite sparse geometry")
    map_data = {}
    for name in MAP_IMAGES:
        depth = read_map(dense / "stereo" / "depth_maps" / f"{name}.photometric.bin", 1)[..., 0]
        normals = read_map(dense / "stereo" / "normal_maps" / f"{name}.photometric.bin", 3)
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any() or normals.shape[:2] != depth.shape:
            raise ValueError(f"Invalid map pair: {name}")
        map_data[name] = depth, normals, valid
    all_xyz = np.vstack([sparse_xyz, dense_xyz, mesh_xyz, [c["center"] for c in cameras]])
    bounds = all_xyz.min(axis=0), all_xyz.max(axis=0)
    focus = np.percentile(dense_xyz, 1, axis=0), np.percentile(dense_xyz, 99, axis=0)
    output.mkdir()  # Fail rather than overwrite an earlier report.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlepad": 9})
    features_plot(output, snapshot, order, keypoints, args.dpi)
    sampling = matches_plots(output, snapshot, order, keypoints, raw, verified, args.max_match_lines, args.dpi)
    cameras_plot(output, images, cameras, args)
    displayed = {"sparse": geometry_plot(output, 4, sparse_xyz, sparse_rgb, bounds, focus, args)}
    depth_info = maps_plot(output, dense, map_data, False, args)
    maps_plot(output, dense, map_data, True, args)
    displayed["dense"] = geometry_plot(output, 6, dense_xyz, dense_rgb, bounds, focus, args)
    displayed["mesh"] = geometry_plot(output, 7, mesh_xyz, mesh_rgb, bounds, focus, args, faces)
    # Re-enumerate too: detect additions/removals as well as changed file bytes.
    protected_after = sorted(p for p in reconstruction.rglob("*") if p.is_file())
    protected_after += sorted(p for p in snapshot.rglob("*") if p.is_file()) + [summary_path]
    after = fingerprints(protected_after, root)
    if before != after:
        raise RuntimeError("Source files changed during rendering; preservation verification failed")
    manifest = {"run_id": RUN, "created_local": datetime.now().astimezone().isoformat(),
                "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
                "original_files_unchanged": True, "source_file_count": len(before), "sources": before,
                "metrics": measured, "database": database_stats,
                "illustrative_pair": {"images": order, "raw": len(raw), "verified": len(verified),
                                      "excluded": len(raw)-len(verified), **sampling},
                "depth_maps": depth_info, "display_counts": displayed,
                "view": {"elev": args.elev, "azim": args.azim, "roll": args.roll,
                         "common_full_bounds": [v.tolist() for v in bounds],
                         "common_display_crop_bounds": [v.tolist() for v in focus]},
                "outputs": {name: {"sha256": sha256(output/name)} for name in FILES}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = f"""# Office 26 Blackwell: saved COLMAP stage plots

Run: `{RUN}`. Generated by `scripts/plot_office_blackwell_stages.py`.

No reconstruction was rerun. SHA-256 before/after checks matched for all {len(before)}
original files: the complete COLMAP run directory, all snapshot inputs and summary.json.
The database was opened read-only and immutable. Source hashes and actual counts are in manifest.json.

| Plot | How to read it |
| --- | --- |
| [01_features.png](01_features.png) | Original photographs and every saved feature centre for the chosen pair. |
| [02_raw_matches.png](02_raw_matches.png) | {len(raw)} candidates; {sampling['raw_displayed']} sampled lines displayed. |
| [03_verified_matches.png](03_verified_matches.png) | {len(verified)} verified; only the {sampling['verified_displayed']} surviving members of the SAME display sample. |
| [04_camera_registration.png](04_camera_registration.png) | Final 20 camera poses plus status of all 26 inputs. |
| [05_sparse_points.png](05_sparse_points.png) | Final 1,237 triangulated/bundle-adjusted points. |
| [06_depth_maps.png](06_depth_maps.png) | Four illustrative saved photometric depth maps paired with their dense images. |
| [normal_maps.png](normal_maps.png) | Same four views; RGB encodes camera-frame normal direction. |
| [07_dense_fusion.png](07_dense_fusion.png) | All 79,094 saved fused points in the full-extent panel. |
| [08_poisson_mesh.png](08_poisson_mesh.png) | All 141,837 original triangle faces in the full-extent panel. |

## Interpretation

- Stages 04/05 both read the final saved sparse model. No pre-BA or early-registration snapshot exists.
- Match figures use {order[0]} and {order[1]}; this illustrative pair does not summarise every pair.
- Feature overlays respect original-image coordinates and six-column keypoint layout; centres only are drawn.
- Depths must be finite and positive. Grey means invalid. Shared 2nd-98th percentile colour limits improve
  contrast; extreme valid depths saturate at the colourbar endpoints, not deleted from the saved map.
- Normals are camera-relative. R/G/B encode X/Y/Z via (normal+1)/2. Identical colours across cameras
  do not establish identical world orientation. Positive depth and finite nonzero normal are required.
- 05/07/08 share view angles and full scene bounds. Their right panel is an explicitly labelled display crop
  based on the dense cloud's 1st/99th percentiles. Full panels preserve distant fragments/outliers.
  Mesh crop includes only faces with all three vertices inside the crop. There is no source filtering.
- Poisson faces are rendered as triangles, not merely vertex dots. Lighting affects display only.
- All COLMAP coordinates/depths are in arbitrary SfM units. More points/triangles do not establish accuracy.

## Run again

```powershell
conda activate mapanything
python scripts\\plot_office_blackwell_stages.py
```

Each invocation makes a new timestamped folder; it never replaces earlier plots. Root is inferred from the
script location, including on the Blackwell workstation with spaces in its username. Optional settings:
`--dpi 200 --max-match-lines 200 --elev 20 --azim -62 --roll 0`.
`--output-dir` must name a new direct child of this experiment folder. Dependencies: numpy, matplotlib,
Pillow, plyfile. No PyTorch, GPU, COLMAP executable, internet or model download is required.

Format references: [database](https://colmap.github.io/database.html),
[binary models and maps](https://colmap.github.io/format.html),
[normal-frame convention](https://github.com/colmap/colmap/blob/main/src/colmap/mvs/fusion.cc).
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"Verified unchanged: {len(before)} source files.\nReview folder: {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, sqlite3.Error, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
