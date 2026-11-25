"""Review eight conceptual COLMAP stages using saved artifacts only.

Examples (from the project folder):
  python scripts/visualize_colmap_stages.py --dataset my_images --run test_v1
  python scripts/visualize_colmap_stages.py --run my_images/test_v1
  python scripts/visualize_colmap_stages.py my_images/test_v1 --all-images

No reconstruction is run. Stages 2/3 share one matcher execution. Stages 4/5
show the FINAL sparse model, not invented registration/BA snapshots.
Dependencies: numpy, matplotlib, Pillow, plyfile; no GPU is needed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import struct
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

from plot_colmap_dense import read_fused_ply
from plot_colmap_sfm import read_sparse_model, resolve_sparse_model
from reconstruction_pipeline import validate_component

ROOT = Path(__file__).resolve().parents[1]
PAIR_BASE = 2147483647
STAGES = [
    "SIFT feature extraction", "Feature matching", "Geometric verification",
    "Camera registration and sparse SfM", "Triangulation and bundle adjustment",
    "PatchMatch stereo", "Stereo fusion", "Poisson surface reconstruction",
]


def portable(path, root=ROOT):
    """Project-relative source references remain useful after copying projects."""
    path = Path(path).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated COLMAP binary artifact")
    return data


def read_dense_map(path, channels):
    """COLMAP width&height&channels& header, float32 Fortran-order payload."""
    with Path(path).open("rb") as stream:
        header = bytearray()
        while header.count(b"&") < 3:
            header.extend(read_exact(stream, 1))
            if len(header) > 100:
                raise ValueError(f"Invalid dense-map header: {path}")
        width, height, found = map(int, header[:-1].split(b"&"))
        if width < 1 or height < 1 or found != channels:
            raise ValueError(f"Invalid dense-map dimensions: {path}")
        data = np.frombuffer(stream.read(), dtype="<f4")
    if data.size != width * height * channels:
        raise ValueError(f"Incorrect dense-map payload size: {path}")
    return data.reshape((width, height, channels), order="F").transpose(1, 0, 2)


def read_point_diagnostics(model, kind):
    rows = []
    if kind == "binary":
        with (model / "points3D.bin").open("rb") as stream:
            count, = struct.unpack("<Q", read_exact(stream, 8))
            for _ in range(count):
                rec = struct.unpack("<Q3d3Bd", read_exact(stream, 43))
                length, = struct.unpack("<Q", read_exact(stream, 8))
                read_exact(stream, length * 8)
                rows.append((rec[0], rec[-1], length))
            if stream.read(1):
                raise ValueError("Trailing bytes in points3D.bin")
    else:
        with (model / "points3D.txt").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 8 or (len(fields) - 8) % 2:
                    raise ValueError("Invalid points3D.txt record")
                rows.append((int(fields[0]), float(fields[7]), (len(fields) - 8) // 2))
    return rows


def read_triangle_mesh(path):
    """Load real triangle faces, including PLY variable-length index lists."""
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("Stage 8 requires plyfile: python -m pip install plyfile") from exc
    ply = PlyData.read(str(path), mmap=False)
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex[k] for k in ("x", "y", "z")])
    names = vertex.dtype.names or ()
    colors = (np.column_stack([vertex[k] for k in ("red", "green", "blue")]) / 255.
              if all(k in names for k in ("red", "green", "blue")) else None)
    if "face" not in ply:
        return xyz, colors, np.empty((0, 3), dtype=np.int64)
    face_data = ply["face"].data
    key = next((k for k in ("vertex_indices", "vertex_index")
                if k in (face_data.dtype.names or ())), None)
    if key is None:
        raise ValueError("PLY faces lack a vertex_indices property")
    records = face_data[key]
    if any(len(face) != 3 for face in records):
        raise ValueError("Expected triangle faces; this mesh has non-triangle polygons")
    faces = np.stack(records).astype(np.int64) if len(records) else np.empty((0, 3), dtype=np.int64)
    if len(faces) and (faces.min() < 0 or faces.max() >= len(xyz)):
        raise ValueError("PLY face indices are outside the vertex array")
    return xyz, colors, faces


def safe_child(parent, name):
    path = (parent / name).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f"Image name escapes its source directory: {name}")
    return path


def sample_indices(count, limit, seed=42):
    if not limit or count <= limit:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, limit, replace=False))


def choose(items, limit):
    if not limit or len(items) <= limit:
        return list(items)
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, limit)]


def csv_write(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def thumbnail(path, size=1400):
    with Image.open(path) as source:
        original_size = source.size
        result = source.convert("RGB")
        result.thumbnail((size, size))
    return np.asarray(result), original_size


class Review:
    def __init__(self, args, source, experiment, output):
        self.args, self.source, self.experiment, self.output = args, source, experiment, output
        self.image_dir = args.images or experiment / "input_images"
        self.stages, self.images, self.db, self.sparse = [], {}, None, None

    def save(self, fig, path):
        fig.savefig(path, dpi=self.args.dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)

    def open_db(self):
        if self.db is not None:
            return self.db
        path = self.source / "database.db"
        if not path.is_file():
            raise FileNotFoundError(f"Missing database: {portable(path)}")
        for suffix in ("-wal", "-journal"):
            sidecar = path.with_name(path.name + suffix)
            if sidecar.exists() and sidecar.stat().st_size:
                raise ValueError("Database has a nonempty WAL/journal. Finish and close COLMAP before review.")
        # immutable=1 also prevents creation of SHM/WAL sidecars; reject existing
        # uncheckpointed WAL above so immutable mode does not silently miss data.
        self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        self.db.execute("PRAGMA query_only=ON")
        self.images = dict(self.db.execute("SELECT image_id,name FROM images ORDER BY image_id"))
        return self.db

    def keypoints(self, image_id):
        row = self.open_db().execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
        if row is None:
            return np.empty((0, 2))
        count, cols, blob = row
        if cols not in (2, 4, 6):
            raise ValueError(f"Unsupported keypoint column count: {cols}")
        return np.frombuffer(blob or b"", dtype="<f4").reshape(count, cols)[:, :2]

    def stage1(self, folder):
        rows = list(self.open_db().execute("SELECT i.image_id,i.name,COALESCE(k.rows,0) FROM images i LEFT JOIN keypoints k USING(image_id) ORDER BY i.name"))
        csv_write(folder / "all_images_keypoints.csv", ["image_id", "image", "keypoints"], rows)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(np.arange(len(rows)), [r[2] for r in rows], color="#2563a6")
        ax.set(xlabel="Image index (see CSV)", ylabel="Stored keypoints", title=f"All {len(rows)} database images")
        self.save(fig, folder / "keypoint_counts.png")
        previews = []
        for image_id, name, count in choose(rows, 0 if self.args.all_images else self.args.max_images):
            path = safe_child(self.image_dir, name)
            if not path.is_file():
                previews.append({"image": name, "status": "missing original image"})
                continue
            pixels, (width, height) = thumbnail(path)
            points = self.keypoints(image_id)
            ids = sample_indices(len(points), self.args.max_keypoints, self.args.seed)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(pixels, extent=(0, width, height, 0))
            ax.scatter(points[ids, 0], points[ids, 1], s=5, facecolors="none", edgecolors="#ffdd00", linewidths=.45)
            ax.set_title(f"{name}\n{len(ids):,}/{count:,} stored keypoint positions shown")
            ax.axis("off")
            file = f"features_{image_id}.png"
            self.save(fig, folder / file)
            previews.append({"image": name, "file": file, "shown": len(ids), "total": count})
        return {"images": len(rows), "keypoints": sum(r[2] for r in rows), "previews": previews,
                "notes": ["All image counts are exported; overlays follow image/keypoint preview limits.",
                          "Keypoints use original image coordinates. Undistorted dense images cannot replace original input images.",
                          "Circles mark positions only; no scale or orientation is invented."]}

    def match_inventory(self):
        db = self.open_db()
        raw = dict(db.execute("SELECT pair_id,rows FROM matches"))
        verified = dict(db.execute("SELECT pair_id,rows FROM two_view_geometries"))
        rows = []
        for pair in sorted(raw.keys() | verified.keys()):
            first, second = pair // PAIR_BASE, pair % PAIR_BASE
            n, v = raw.get(pair, 0), verified.get(pair, 0)
            rows.append((pair, first, self.images.get(first, "UNKNOWN"), second,
                         self.images.get(second, "UNKNOWN"), n, v, v / n if n else ""))
        return rows

    def match_array(self, pair, verified):
        table = "two_view_geometries" if verified else "matches"
        row = self.open_db().execute(f"SELECT rows,cols,data FROM {table} WHERE pair_id=?", (pair,)).fetchone()
        if row is None or row[0] == 0:
            return np.empty((0, 2), dtype=np.uint32)
        if row[1] != 2:
            raise ValueError("Invalid stored match shape")
        return np.frombuffer(row[2], dtype="<u4").reshape(row[0], 2)

    def pair_plot(self, folder, row, verified):
        pair, image1, name1, image2, name2, raw_count, verified_count, _ = row
        paths = [safe_child(self.image_dir, name) for name in (name1, name2)]
        if not all(path.is_file() for path in paths):
            return {"pair_id": pair, "status": "missing original image"}
        keys = [self.keypoints(image1), self.keypoints(image2)]
        matches = self.match_array(pair, verified)
        if len(matches) and any(matches[:, i].max() >= len(keys[i]) for i in (0, 1)):
            raise ValueError(f"Match index outside keypoint array: {pair}")
        panels, coordinates, offset = [], [], 0
        for path, key in zip(paths, keys):
            pixels, (width, height) = thumbnail(path, 1000)
            h, w = pixels.shape[:2]
            coords = key * np.array([w / width, h / height])
            coords[:, 0] += offset
            panels.append(pixels)
            coordinates.append(coords)
            offset += w
        canvas = np.full((max(p.shape[0] for p in panels), offset, 3), 235, dtype=np.uint8)
        offset = 0
        for pixels in panels:
            h, w = pixels.shape[:2]
            canvas[:h, offset:offset + w] = pixels
            offset += w
        ids = sample_indices(len(matches), self.args.max_match_lines, self.args.seed)
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(canvas)
        if len(ids):
            lines = np.stack([coordinates[0][matches[ids, 0]], coordinates[1][matches[ids, 1]]], axis=1)
            ax.add_collection(LineCollection(lines, colors="#42ff68" if verified else "#ffcc22", linewidths=.5, alpha=.7))
        ax.set_title(f"{name1} | {name2}\n{'Verified' if verified else 'Raw'} correspondences: {len(ids):,}/{len(matches):,} shown; raw={raw_count:,}, verified={verified_count:,}")
        ax.axis("off")
        file = f"pair_{pair}.png"
        self.save(fig, folder / file)
        result = {"pair_id": pair, "file": file, "shown": len(ids), "total": len(matches)}
        if verified:
            raw = self.match_array(pair, False)
            inliers, raw_set = set(map(tuple, matches.tolist())), set(map(tuple, raw.tolist()))
            csv_write(folder / f"pair_{pair}_raw_verified_mask.csv",
                      ["keypoint_index_1", "keypoint_index_2", "present_in_verified_table"],
                      ((int(a), int(b), int((int(a), int(b)) in inliers)) for a, b in raw))
            result["verified_not_in_raw"] = sum((int(a), int(b)) not in raw_set for a, b in matches)
        return result

    def match_stage(self, folder, verified=False):
        rows = self.match_inventory()
        csv_write(folder / "all_pairs.csv", ["pair_id", "image_id_1", "image_1", "image_id_2", "image_2",
                                             "raw_matches", "verified_matches", "verified_to_raw_count_ratio"], rows)
        if not rows:
            raise FileNotFoundError("No stored match pairs in database")
        values = [r[6 if verified else 5] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(values, bins=min(40, max(1, len(rows))), color="#207b52" if verified else "#2563a6")
        axes[0].set(xlabel="Verified correspondences" if verified else "Raw correspondences", ylabel="Stored pairs")
        axes[1].scatter([r[5] for r in rows], [r[6] for r in rows], s=10)
        axes[1].set(xlabel="Raw match count", ylabel="Verified match count")
        fig.suptitle(f"{'Geometric verification' if verified else 'Matching'}: all {len(rows):,} stored pairs")
        self.save(fig, folder / "pair_statistics.png")
        ordered = sorted(rows, key=lambda r: (-r[5], r[0]))
        selected = ordered if self.args.all_pairs else ordered[:self.args.max_pairs]
        previews = [self.pair_plot(folder, row, verified) for row in selected]
        return {"stored_pairs": len(rows), "nonempty_pairs": sum(v > 0 for v in values),
                "correspondences": sum(values), "previews": previews,
                "notes": ["Stages 2/3 are saved database tables from the same matcher execution.",
                          "Both preview stages select the same highest raw-match-count pairs. Full pair counts are in CSV.",
                          "Verified-table membership is the saved geometric-verification result. RANSAC iterations were not recorded.",
                          "Guided matching can add verified correspondences absent from raw matches; the count ratio is descriptive."]}

    def sparse_data(self):
        if self.sparse is None:
            model = kind = None
            for candidate in (self.source / "dense" / "sparse", self.source / "sparse", self.source):
                if not candidate.is_dir():
                    continue
                try:
                    model, kind = resolve_sparse_model(candidate)
                    break
                except FileNotFoundError:
                    continue
            if model is None:
                raise FileNotFoundError(f"No saved sparse model below {portable(self.source)}")
            if (model / "images.bin").is_file() and (model / "points3D.bin").is_file():
                kind = "binary"
            cameras, points, colors = read_sparse_model(model, kind)
            self.sparse = model, kind, cameras, points, colors
        return self.sparse

    def multiview(self, xyz, colors, path, title, cameras=None, faces=None):
        good = np.isfinite(xyz).all(axis=1)
        centers = np.array([c["center"] for c in cameras]) if cameras else np.empty((0, 3))
        extent = np.concatenate([xyz[good], centers])
        if not len(extent):
            raise ValueError("No finite geometry or cameras to plot")
        low, high = extent.min(axis=0), extent.max(axis=0)
        middle, radius = (low + high) / 2, max(float((high - low).max()) / 2, 1e-6) * 1.05
        finite_ids = np.flatnonzero(good)
        chosen = finite_ids[sample_indices(len(finite_ids), self.args.max_points, self.args.seed)]
        if faces is not None:
            valid_faces = np.isfinite(xyz[faces]).all(axis=(1, 2))
            face_ids = np.flatnonzero(valid_faces)
            face_ids = face_ids[sample_indices(len(face_ids), self.args.max_faces, self.args.seed)]
            triangles = xyz[faces[face_ids]]
            face_colors = colors[faces[face_ids]].mean(axis=1) if colors is not None else "#a5b6bf"
        fig = plt.figure(figsize=(16, 5))
        fig.suptitle(title)
        for number, dimensions in enumerate((None, (0, 2), (0, 1)), 1):
            if dimensions is None:
                ax = fig.add_subplot(1, 3, number, projection="3d")
                if faces is None:
                    ax.scatter(*xyz[chosen].T, c=colors[chosen] if colors is not None else "#417fa9", s=1, linewidths=0, rasterized=True)
                else:
                    ax.add_collection3d(Poly3DCollection(triangles, facecolors=face_colors, edgecolors="none", rasterized=True))
                if len(centers):
                    ax.scatter(*centers.T, c="#db3225", s=18, marker="^")
                    for camera in cameras:
                        start = camera["center"]
                        forward = camera["rotation"].T @ np.array([0., 0., 1.]) * radius * .08
                        ax.plot(*np.stack([start, start + forward]).T, color="#db3225", linewidth=.7)
                ax.set(xlim=(middle[0] - radius, middle[0] + radius), ylim=(middle[1] - radius, middle[1] + radius),
                       zlim=(middle[2] - radius, middle[2] + radius), xlabel="X", ylabel="Y", zlabel="Z", title="Oblique view")
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=25, azim=-60)
            else:
                ax = fig.add_subplot(1, 3, number)
                a, b = dimensions
                if faces is None:
                    ax.scatter(xyz[chosen, a], xyz[chosen, b], c=colors[chosen] if colors is not None else "#417fa9", s=1, linewidths=0, rasterized=True)
                else:
                    hidden = next(d for d in range(3) if d not in dimensions)
                    order = np.argsort(triangles[:, :, hidden].mean(axis=1))
                    fc = face_colors[order] if isinstance(face_colors, np.ndarray) else face_colors
                    ax.add_collection(PolyCollection(triangles[order][:, :, [a, b]], facecolors=fc, edgecolors="none", rasterized=True))
                if len(centers):
                    ax.scatter(centers[:, a], centers[:, b], c="#db3225", s=18, marker="^")
                    for camera in cameras:
                        start = camera["center"]
                        forward = camera["rotation"].T @ np.array([0., 0., 1.]) * radius * .08
                        segment = np.stack([start, start + forward])
                        ax.plot(segment[:, a], segment[:, b], color="#db3225", linewidth=.7)
                ax.set(xlim=(middle[a] - radius, middle[a] + radius), ylim=(middle[b] - radius, middle[b] + radius),
                       xlabel="XYZ"[a], ylabel="XYZ"[b], title=f"{'XYZ'[a]}{'XYZ'[b]} projection")
                ax.set_aspect("equal")
                ax.grid(alpha=.2)
        self.save(fig, path)
        return {"total_vertices": len(xyz), "nonfinite_vertices": int((~good).sum()),
                "plotted_points": len(chosen) if faces is None else None,
                "total_faces": len(faces) if faces is not None else None,
                "plotted_faces": len(face_ids) if faces is not None else None,
                "nonfinite_faces": int((~valid_faces).sum()) if faces is not None else None,
                "coordinate_min": low.tolist(), "coordinate_max": high.tolist()}

    def stage4(self, folder):
        model, kind, cameras, points, colors = self.sparse_data()
        csv_write(folder / "registered_cameras.csv", ["image_id", "image", "center_x", "center_y", "center_z"],
                  ((c["id"], c["name"], *c["center"]) for c in cameras))
        result = self.multiview(points, colors, folder / "cameras_and_sparse.png",
                                f"Final sparse model: {len(cameras)} registered cameras (red), {len(points):,} points", cameras=cameras)
        result.update(model=portable(model), model_format=kind, registered_images=len(cameras),
                      notes=["Final mapper output, after triangulation and bundle adjustment; per-registration snapshots are unavailable.",
                             "dense/sparse identifies the actual MVS model. Otherwise the largest registered sparse component is selected.",
                             "All cameras are shown. Sparse point rendering follows --max-points. No spatial trimming is applied.",
                             "Axes are reconstruction coordinates; they are not guaranteed metres or upright scene coordinates."])
        return result

    def stage5(self, folder):
        model, kind, _, _, _ = self.sparse_data()
        rows = read_point_diagnostics(model, kind)
        csv_write(folder / "all_sparse_point_diagnostics.csv", ["point3D_id", "final_reprojection_error_px", "track_length"], rows)
        if not rows:
            return {"status": "empty", "point_count": 0, "notes": ["Sparse model contains no triangulated points."]}
        errors = np.array([r[1] for r in rows])
        tracks = np.array([r[2] for r in rows])
        valid = np.isfinite(errors) & (errors >= 0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(errors[valid], bins=50, color="#2563a6")
        axes[0].set(xlabel="Final mean reprojection error per point (pixels)", ylabel="3D points")
        axes[1].hist(tracks, bins=min(50, max(1, int(tracks.max() - tracks.min() + 1))), color="#207b52")
        axes[1].set(xlabel="Track length (observations per 3D point)", ylabel="3D points")
        fig.suptitle(f"Final triangulation / bundle-adjustment diagnostics: {len(rows):,} points")
        self.save(fig, folder / "final_sparse_diagnostics.png")
        return {"point_count": len(rows), "mean_point_error_px": float(errors[valid].mean()) if valid.any() else None,
                "median_point_error_px": float(np.median(errors[valid])) if valid.any() else None,
                "invalid_error_count": int((~valid).sum()), "mean_track_length": float(tracks.mean()),
                "notes": ["These are final stored errors/tracks, not a pre/post-BA comparison or convergence history.",
                          "Mean per-point error can differ from observation-weighted model-analyzer error.",
                          "Reprojection error measures image agreement, not physical scene accuracy. Every point record is exported."]}

    def stage6(self, folder):
        stereo = self.source / "dense" / "stereo"
        depths = sorted((stereo / "depth_maps").glob("*.bin"))
        normals = sorted((stereo / "normal_maps").glob("*.bin"))
        if not depths and not normals:
            raise FileNotFoundError("No saved PatchMatch depth or normal maps")
        rows, previews = [], []
        for kind, files in (("depth", depths), ("normal", normals)):
            selected = set(choose(files, 0 if self.args.all_images else self.args.max_images))
            for index, path in enumerate(files):
                data = read_dense_map(path, 1 if kind == "depth" else 3)
                valid = np.isfinite(data).all(axis=2)
                valid &= data[:, :, 0] > 0 if kind == "depth" else np.linalg.norm(data, axis=2) > 0
                count = int(valid.sum())
                rows.append([kind, path.name, data.shape[1], data.shape[0], count, valid.size, count / valid.size])
                if path not in selected:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(11, 5))
                if kind == "depth":
                    view = axes[0].imshow(np.ma.array(data[:, :, 0], mask=~valid), cmap="viridis")
                    fig.colorbar(view, ax=axes[0], label="Depth in reconstruction units")
                else:
                    color = np.clip((data + 1.) / 2., 0., 1.)
                    color[~valid] = 0
                    axes[0].imshow(color)
                axes[0].set_title("Depth (full valid range)" if kind == "depth" else "Normal RGB = (normal + 1) / 2")
                axes[1].imshow(valid, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title(f"Valid: {count:,}/{valid.size:,} ({100 * count / valid.size:.1f}%)")
                for axis in axes:
                    axis.axis("off")
                fig.suptitle(path.name)
                file = f"{kind}_{index:04d}.png"
                self.save(fig, folder / file)
                previews.append({"source": portable(path), "file": file})
        csv_write(folder / "all_dense_maps.csv", ["kind", "map", "width", "height", "valid_pixels", "total_pixels", "valid_fraction"], rows)
        return {"depth_maps": len(depths), "normal_maps": len(normals), "previews": previews,
                "notes": ["Every map is scanned for counts; previews follow --max-images per map type, or --all-images.",
                          "Valid depth is finite and positive. Valid normal is finite and nonzero; these tests are independent.",
                          "White in the validity panel means valid. Depth is not percentile-clipped. Normal colours encode direction.",
                          "No PatchMatch iterations or dense reconstruction are rerun."]}

    def stage7(self, folder):
        path = self.source / "dense" / "fused.ply"
        if not path.is_file():
            raise FileNotFoundError("Missing dense/fused.ply")
        points, colors, total = read_fused_ply(path)
        if not len(points):
            csv_write(folder / "fusion_counts.csv", ["source_vertices", "finite_vertices", "plotted_vertices"], [(total, 0, 0)])
            return {"status": "empty", "source_vertex_count": total, "finite_vertices": 0}
        result = self.multiview(points, colors, folder / "fused_cloud.png",
                                f"Fused cloud: {len(points):,} finite points; plot cap {self.args.max_points or 'all'}")
        result.update(source=portable(path), source_vertex_count=total, source_nonfinite_vertices=total - len(points),
                      notes=["Complete coordinate extent is retained. Point sampling affects PNGs only; counts cover the entire PLY.",
                             "No automatic alignment to other models or ground truth is performed."])
        csv_write(folder / "fusion_counts.csv", ["source_vertices", "finite_vertices", "plotted_vertices"],
                  [(total, len(points), result["plotted_points"])])
        return result

    def stage8(self, folder):
        path = self.source / "dense" / "meshed-poisson.ply"
        if not path.is_file():
            raise FileNotFoundError("Missing dense/meshed-poisson.ply")
        points, colors, faces = read_triangle_mesh(path)
        csv_write(folder / "mesh_counts.csv", ["vertices", "faces"], [(len(points), len(faces))])
        if not len(points) or not len(faces):
            return {"status": "empty", "vertices": len(points), "faces": len(faces), "source": portable(path),
                    "notes": ["The mesh file exists but has no nonempty triangle surface."]}
        result = self.multiview(points, colors, folder / "triangle_mesh.png",
                                f"Poisson triangle mesh: {len(faces):,} faces; plot cap {self.args.max_faces or 'all'}", faces=faces)
        result.update(source=portable(path),
                      notes=["Actual triangle faces are rendered. Point scatter is never presented as a mesh.",
                             "Subsampling faces can create apparent gaps. Use --max-faces 0 or the original mesh in MeshLab for a full surface view.",
                             "No cleaning, hole filling, or spatial trimming is applied. Nonfinite triangles cannot be drawn and are counted.",
                             "Existing vertex colours are shown; this does not generate texture maps."])
        return result

    def execute(self):
        methods = [self.stage1, lambda p: self.match_stage(p), lambda p: self.match_stage(p, True),
                   self.stage4, self.stage5, self.stage6, self.stage7, self.stage8]
        try:
            for number, method in enumerate(methods, 1):
                print(f"Stage {number}/8: {STAGES[number - 1]}", flush=True)
                folder = self.output / f"{number:02d}_{STAGES[number - 1].lower().replace(' ', '_')}"
                folder.mkdir()
                entry = {"stage": number, "name": STAGES[number - 1], "directory": folder.name, "status": "available"}
                try:
                    entry.update(method(folder))
                except FileNotFoundError as exc:
                    entry.update(status="unavailable", reason=str(exc))
                except Exception as exc:
                    entry.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                finally:
                    plt.close("all")
                entry["files"] = sorted(p.relative_to(self.output).as_posix() for p in folder.iterdir() if p.is_file())
                self.stages.append(entry)
                print(f"  {entry['status']}: {entry.get('reason', folder.name)}", flush=True)
        finally:
            if self.db:
                self.db.close()
        self.write_index()
        return 1 if any(s["status"] == "error" for s in self.stages) else 0

    def write_index(self):
        settings = {k: portable(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()}
        manifest = {"schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
                    "run_ref": self.args.run_ref, "source": portable(self.source),
                    "experiment": portable(self.experiment), "original_images": portable(self.image_dir),
                    "path_base": "Project root for relative source paths; review folder for generated-file links",
                    "settings": settings, "stages": self.stages}
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        lines = [f"# COLMAP saved-stage review: {self.args.run_ref}", "", f"Source: `{portable(self.source)}`", "",
                 "This report reads saved artifacts. It does not rerun reconstruction or modify source files.", "",
                 "Stages 2/3 share one matching execution. Stages 4/5 show final sparse results; registration snapshots and pre/post-BA histories were not saved.", "",
                 "Coordinates are not automatically upright, metric-scaled, aligned across runs, or evaluated against ground truth.", "",
                 "## Rendering coverage", "",
                 f"- Image/map previews: {'all' if self.args.all_images else self.args.max_images} per stage/type.",
                 f"- Pair previews: {'all stored pairs' if self.args.all_pairs else self.args.max_pairs}; highest raw-match counts first.",
                 f"- Keypoints per overlay: {self.args.max_keypoints or 'all'}. Lines per pair: {self.args.max_match_lines or 'all'}.",
                 f"- Point cap: {self.args.max_points or 'all'}. Triangle cap: {self.args.max_faces or 'all'}.",
                 "- Inventory CSVs include all stored image/pair/map/point records. Geometry counts cover complete files. PNGs may be sampled.",
                 "- Full spatial extent is retained. Sampled triangles can appear to have holes; use `--max-faces 0` or MeshLab to inspect the full surface.", ""]
        for entry in self.stages:
            lines.extend([f"## Stage {entry['stage']}: {entry['name']}", "", f"Status: **{entry['status']}**", ""])
            if entry.get("reason"):
                lines.extend([entry["reason"], ""])
            lines.extend(f"- {note}" for note in entry.get("notes", []))
            lines.append("")
            for file in entry["files"]:
                lines.extend([f"![{Path(file).stem}]({file})" if file.endswith(".png") else f"[{Path(file).name}]({file})", ""])
        (self.output / "index.md").write_text("\n".join(lines), encoding="utf-8")
        (self.output / "README.md").write_text("# Saved COLMAP stage review\n\nOpen [index.md](index.md) for plots and explanations.\n\nMachine-readable counts, sampling limits and stage statuses are in [manifest.json](manifest.json).\n\nA missing/empty artifact is reported explicitly; successful script execution is not evidence of geometric accuracy.\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_reference", nargs="?", help="Optional positional dataset/run_id")
    parser.add_argument("--dataset", help="Dataset/scene group (folder under experiments and colmap_output)")
    parser.add_argument("--run", help="Run ID with --dataset, or dataset/run_id")
    parser.add_argument("--run-ref", help="Alternative explicit dataset/run_id")
    parser.add_argument("--input", type=Path, help="Override COLMAP run directory")
    parser.add_argument("--images", type=Path, help="Override ORIGINAL input images directory; do not use undistorted dense/images")
    parser.add_argument("--output", type=Path, help="NEW review directory; existing paths are always refused")
    parser.add_argument("--max-images", type=int, default=6, help="Representative previews per image/map type (default 6)")
    parser.add_argument("--all-images", action="store_true", help="Preview every available feature image and dense map")
    parser.add_argument("--max-pairs", type=int, default=3, help="Highest raw-match-count pairs to preview (default 3)")
    parser.add_argument("--all-pairs", action="store_true", help="Preview every stored pair; potentially thousands of PNGs")
    parser.add_argument("--max-keypoints", type=int, default=1500, help="Points per keypoint overlay; 0 means all")
    parser.add_argument("--max-match-lines", type=int, default=150, help="Lines per pair preview; 0 means all")
    parser.add_argument("--max-points", type=int, default=100000, help="Rendered sparse/fused points; 0 means all")
    parser.add_argument("--max-faces", type=int, default=40000, help="Rendered triangles; 0 means all (can be slow)")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Show paths/settings without loading artifacts or writing files")
    args = parser.parse_args(argv)
    references = [v for v in (args.run_reference, args.run_ref) if v]
    if args.run:
        references.append(f"{args.dataset}/{args.run}" if args.dataset else args.run)
    elif args.dataset:
        parser.error("--dataset requires --run")
    if len(references) != 1:
        parser.error("Use one reference: --dataset GROUP --run ID, --run GROUP/ID, --run-ref GROUP/ID, or positional GROUP/ID")
    parts = references[0].replace("\\", "/").split("/")
    if len(parts) != 2:
        parser.error("A run reference must contain exactly dataset/run_id")
    try:
        validate_component(parts[0], "dataset")
        validate_component(parts[1], "run ID")
    except ValueError as exc:
        parser.error(str(exc))
    args.run_ref = "/".join(parts)
    args.dataset, args.run = parts
    for name in ("max_images", "max_pairs", "dpi"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_keypoints", "max_match_lines", "max_points", "max_faces", "seed"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    experiment = ROOT / "experiments" / args.dataset / args.run
    source = (args.input or ROOT / "colmap_output" / args.dataset / args.run).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or experiment / "visualizations" / f"stages_{stamp}").resolve()
    if output.exists():
        raise FileExistsError(f"Output exists; choose a NEW directory: {output}")
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the COLMAP source directory")
    image_dir = (args.images or experiment / "input_images").resolve()
    if output == image_dir or output.is_relative_to(image_dir):
        raise ValueError("Output must be outside the original image directory")
    print(json.dumps({"run_ref": args.run_ref, "source": str(source), "original_images": str(image_dir),
                      "output": str(output), "source_exists": source.is_dir(), "dry_run": args.dry_run}, indent=2))
    if args.dry_run:
        return 0
    if not source.is_dir():
        raise FileNotFoundError(f"COLMAP run does not exist yet: {source}")
    if args.all_pairs:
        print("WARNING: --all-pairs can create thousands of images. Complete pair counts are exported without this flag.")
    print("Preview limits affect PNGs only. No hidden outlier trimming or reconstruction occurs.")
    output.mkdir(parents=True, exist_ok=False)
    status = Review(args, source, experiment, output).execute()
    print(f"Review: {output / 'index.md'}")
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
