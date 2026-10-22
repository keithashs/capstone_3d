r"""Plot real COLMAP SfM sparse points and registered camera poses.

Accepts COLMAP text or binary models directly.  ``--model`` may be a model
directory, ``sparse`` directory, or a complete
``colmap_output/<scene>/<run>`` directory.  Run this after an existing
reconstruction: no images are reprocessed and COLMAP is not required.

Example (from capstone_3d project root):
    python scripts\plot_colmap_sfm.py \
        --model "colmap_output\office_26\office_full26_sift1600_ratio080" \
        --output "experiments\office_26\office_full26_sift1600_ratio080\sfm_points_cameras.png" \
        --label-cameras
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import struct

import matplotlib
matplotlib.use("Agg")  # Render files on servers and from non-interactive shells.
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

IMAGE_OUTPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".tif", ".tiff", ".webp"}


def _model_format(path: Path) -> str | None:
    if (path / "images.txt").is_file() and (path / "points3D.txt").is_file():
        return "text"
    if (path / "images.bin").is_file() and (path / "points3D.bin").is_file():
        return "binary"
    return None


def _binary_record_count(path: Path) -> int:
    with path.open("rb") as stream:
        data = stream.read(8)
    if len(data) != 8:
        raise ValueError(f"Truncated COLMAP binary file: {path}")
    return int(struct.unpack("<Q", data)[0])


def _model_score(path: Path) -> tuple[int, int]:
    """Prefer the model registering the most images, then most points."""
    kind = _model_format(path)
    try:
        if kind == "binary":
            cameras = _binary_record_count(path / "images.bin")
            points = _binary_record_count(path / "points3D.bin")
        else:
            cameras = len(read_images_txt(path / "images.txt"))
            points = sum(
                1 for row in (path / "points3D.txt").read_text(encoding="utf-8").splitlines()
                if row.strip() and not row.lstrip().startswith("#")
            )
    except (OSError, UnicodeError, ValueError, struct.error):
        cameras, points = -1, -1
    # max() is stable, and the caller sorts paths first, so equal models select
    # the alphabetically first component deterministically.
    return cameras, points


def resolve_sparse_model(input_path: Path) -> tuple[Path, str]:
    """Resolve exactly one text/binary sparse model from common run layouts."""
    path = input_path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"--model must be a directory: {path}")

    direct_kind = _model_format(path)
    if direct_kind:
        return path, direct_kind

    # These are ordered by intent. sparse_txt is an explicit review export;
    # dense/sparse is the selected model used by image_undistorter.
    for candidate in (path / "sparse_txt", path / "dense" / "sparse", path / "colmap_text"):
        kind = _model_format(candidate)
        if kind:
            return candidate, kind

    candidates = {
        item.parent
        for pattern in ("images.txt", "images.bin")
        for item in path.rglob(pattern)
        if _model_format(item.parent)
    }
    ordered = sorted(candidates, key=lambda item: str(item).lower())
    if not ordered:
        raise FileNotFoundError(
            f"No COLMAP sparse model found below {path}. Expected images/points3D .txt or .bin files."
        )
    if len(ordered) == 1:
        selected = ordered[0]
        return selected, _model_format(selected) or "unknown"

    # A sparse directory commonly contains mapper components 0, 1, ... .
    # Choosing the component with the most registered images matches the
    # benchmark runner. Across unrelated nested runs, an explicit path is safer.
    numeric_siblings = all(
        candidate.parent.resolve() == path.resolve() and candidate.name.isdigit()
        for candidate in ordered
    )
    if numeric_siblings:
        selected = max(ordered, key=_model_score)
        return selected, _model_format(selected) or "unknown"

    choices = "\n  - ".join(str(candidate) for candidate in ordered)
    raise ValueError(f"Multiple sparse models found; pass one run/model explicitly:\n  - {choices}")


def _experiment_run_dir(artifact: Path) -> Path | None:
    resolved = artifact.resolve()
    for parent in resolved.parents:
        if parent.name.lower() != "colmap_output":
            continue
        relative = resolved.relative_to(parent)
        if len(relative.parts) >= 2:
            return parent.parent / "experiments" / relative.parts[0] / relative.parts[1]
    return None


def read_run_metadata(artifact: Path) -> dict:
    """Find summary/environment/manifest data associated with a model."""
    locations: list[Path] = []
    experiment_dir = _experiment_run_dir(artifact)
    if experiment_dir is not None:
        locations.append(experiment_dir)
    locations.extend((artifact, *artifact.parents))

    metadata: dict = {}
    seen: set[Path] = set()
    for folder in locations:
        key = folder.resolve()
        if key in seen:
            continue
        seen.add(key)
        for filename in ("summary.json", "environment.json"):
            candidate = folder / filename
            if not candidate.is_file():
                continue
            try:
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(loaded, dict):
                for name, value in loaded.items():
                    metadata.setdefault(name, value)
                metadata.setdefault("metadata_file", str(candidate))
        if "input_images" not in metadata:
            manifest = folder / "input_manifest.csv"
            if manifest.is_file():
                try:
                    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
                        metadata["input_images"] = sum(1 for _ in csv.DictReader(stream))
                    metadata.setdefault("metadata_file", str(manifest))
                except (OSError, UnicodeError, csv.Error):
                    pass
        if metadata.get("scene_group") and metadata.get("input_images") is not None:
            break
    return metadata


def infer_image_count(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.search(r"(?:^|[_-])(\d+)(?:img|images?)(?:$|[_-])", part, re.IGNORECASE)
        if not match:
            match = re.search(r"(?:^|[_-])full[_-]?(\d+)(?:$|[_-])", part, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def infer_scene(path: Path) -> str | None:
    resolved = path.resolve()
    for parent in resolved.parents:
        if parent.name.lower() == "colmap_output":
            relative = resolved.relative_to(parent)
            return relative.parts[0] if relative.parts else None
    return None


def display_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip().title()


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP qvec describes world-to-camera rotation, Hamilton convention."""
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Invalid zero-length camera quaternion")
    w, x, y, z = q / n
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def read_images_txt(path: Path):
    """Read two-line-per-image COLMAP images.txt; keep blank 2D-observation lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    cameras = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        items = line.split(maxsplit=9)
        if len(items) < 10:
            raise ValueError(f"Unrecognized COLMAP images.txt camera line {i+1}: {line}")
        image_id = int(items[0])
        qw, qx, qy, qz = map(float, items[1:5])
        t = np.asarray(list(map(float, items[5:8])), dtype=np.float64)
        R = quaternion_to_rotation(qw, qx, qy, qz)
        center = -R.T @ t  # COLMAP pose is world-to-camera: x_camera = R*x_world + t.
        cameras.append({"id": image_id, "center": center, "rotation": R, "name": items[9]})
        i += 2  # Next line lists x/y/point3D_id triples; may be blank.
    if not cameras:
        raise ValueError(f"No registered camera poses found in {path}")
    return sorted(cameras, key=lambda item: item["id"])


def read_points3d_txt(path: Path):
    xyz, rgb = [], []
    with path.open("r", encoding="utf-8") as stream:
        for row in stream:
            row = row.strip()
            if not row or row.startswith("#"):
                continue
            cols = row.split()
            if len(cols) < 7:
                continue
            xyz.append([float(cols[1]), float(cols[2]), float(cols[3])])
            rgb.append([int(cols[4]) / 255, int(cols[5]) / 255, int(cols[6]) / 255])
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3), np.asarray(rgb, dtype=np.float64).reshape(-1, 3)


def _read_exact(stream, byte_count: int, description: str) -> bytes:
    data = stream.read(byte_count)
    if len(data) != byte_count:
        raise ValueError(f"Truncated COLMAP binary while reading {description}")
    return data


def _unpack(stream, fmt: str, description: str):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(stream, size, description))


def _read_c_string(stream, description: str) -> str:
    value = bytearray()
    while True:
        char = _read_exact(stream, 1, description)
        if char == b"\0":
            return value.decode("utf-8", errors="replace")
        value.extend(char)
        if len(value) > 1_048_576:
            raise ValueError(f"Unreasonably long {description} in COLMAP binary")


def read_images_binary(path: Path):
    """Read camera poses from COLMAP images.bin (little-endian format)."""
    cameras = []
    with path.open("rb") as stream:
        (image_count,) = _unpack(stream, "<Q", "registered image count")
        for index in range(image_count):
            record = _unpack(stream, "<I7dI", f"image record {index + 1}")
            image_id = int(record[0])
            qw, qx, qy, qz = map(float, record[1:5])
            t = np.asarray(record[5:8], dtype=np.float64)
            name = _read_c_string(stream, f"image name {index + 1}")
            (observation_count,) = _unpack(stream, "<Q", f"observation count {index + 1}")
            observation_bytes = int(observation_count) * 24  # x/y doubles + int64 point ID
            if stream.seek(observation_bytes, 1) < 0:
                raise ValueError(f"Invalid observation block in {path}")
            rotation = quaternion_to_rotation(qw, qx, qy, qz)
            cameras.append({
                "id": image_id,
                "center": -rotation.T @ t,
                "rotation": rotation,
                "name": name,
            })
    if not cameras:
        raise ValueError(f"No registered camera poses found in {path}")
    return sorted(cameras, key=lambda item: item["id"])


def read_points3d_binary(path: Path):
    """Read coordinates and RGB values from COLMAP points3D.bin."""
    with path.open("rb") as stream:
        (point_count,) = _unpack(stream, "<Q", "3D point count")
        if point_count > max(path.stat().st_size, 1):
            raise ValueError(f"Implausible point count in {path}: {point_count}")
        xyz = np.empty((point_count, 3), dtype=np.float64)
        rgb = np.empty((point_count, 3), dtype=np.float64)
        for index in range(point_count):
            record = _unpack(stream, "<Q3d3Bd", f"3D point record {index + 1}")
            xyz[index] = record[1:4]
            rgb[index] = np.asarray(record[4:7], dtype=np.float64) / 255.0
            (track_length,) = _unpack(stream, "<Q", f"track length {index + 1}")
            track_bytes = int(track_length) * 8  # uint32 image ID + uint32 point2D index
            if stream.seek(track_bytes, 1) < 0:
                raise ValueError(f"Invalid point track block in {path}")
    return xyz, rgb


def read_sparse_model(model: Path, model_format: str):
    if model_format == "text":
        cameras = read_images_txt(model / "images.txt")
        points, colors = read_points3d_txt(model / "points3D.txt")
    elif model_format == "binary":
        cameras = read_images_binary(model / "images.bin")
        points, colors = read_points3d_binary(model / "points3D.bin")
    else:
        raise ValueError(f"Unsupported sparse model format: {model_format}")

    if len(points):
        valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        points, colors = points[valid], np.clip(colors[valid], 0.0, 1.0)
    return cameras, points, colors


def draw_frustum(ax, center: np.ndarray, rotation: np.ndarray, length: float):
    """Draw camera center + four image-plane corners (+Z is camera forward)."""
    plane = length * np.array([
        [-0.75, -0.5, 1.0], [0.75, -0.5, 1.0],
        [0.75, 0.5, 1.0], [-0.75, 0.5, 1.0],
    ])
    corners = (rotation.T @ plane.T).T + center
    segments = [[center, corner] for corner in corners]
    segments.extend([[corners[k], corners[(k + 1) % 4]] for k in range(4)])
    ax.add_collection3d(Line3DCollection(segments, colors="crimson", linewidths=1.2))


def plot_sparse(args, model_format: str):
    cameras, points, colors = read_sparse_model(args.model, model_format)
    extent_points = points
    point_total = len(points)
    if args.max_points > 0 and len(points) > args.max_points:
        sel = np.sort(
            np.random.default_rng(args.seed).choice(len(points), args.max_points, replace=False)
        )
        points, colors = points[sel], colors[sel]

    centers = np.asarray([x["center"] for x in cameras])
    if args.trim_percentile and point_total:
        # Trim point outliers for display limits, but never crop camera centers.
        finite = extent_points[np.isfinite(extent_points).all(axis=1)]
        if len(finite):
            low = np.minimum(
                np.percentile(finite, args.trim_percentile, axis=0), centers.min(axis=0)
            )
            high = np.maximum(
                np.percentile(finite, 100 - args.trim_percentile, axis=0), centers.max(axis=0)
            )
        else:
            low, high = centers.min(axis=0), centers.max(axis=0)
    else:
        extents = np.vstack([centers, extent_points]) if len(extent_points) else centers
        low, high = extents.min(axis=0), extents.max(axis=0)
    span = max(float(np.max(high - low)), 0.001)
    camera_length = span * 0.045
    midpoint = 0.5 * (low + high)
    radius = span * 0.55

    fig = plt.figure(figsize=(11, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    if len(points):
        point_label = f"Sparse 3D points ({point_total:,})"
        if len(points) != point_total:
            point_label += f"; {len(points):,} displayed"
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors,
                   s=args.point_size, alpha=0.85, linewidths=0, rasterized=True,
                   depthshade=False, label=point_label)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="crimson", s=args.camera_size,
               marker="^", depthshade=False, label=f"Registered cameras ({len(cameras)})")
    for cam in cameras:
        draw_frustum(ax, cam["center"], cam["rotation"], camera_length)
        if args.label_cameras:
            x, y, z = cam["center"]
            ax.text(x, y, z, cam["name"], fontsize=7)

    metadata = read_run_metadata(args.model)
    scene = args.scene or metadata.get("scene_group") or metadata.get("dataset") or infer_scene(args.model)
    input_images = args.images if args.images is not None else metadata.get("input_images")
    if input_images is None:
        input_images = infer_image_count(args.model)
    if args.title:
        title = args.title
    elif scene or input_images is not None:
        title = "COLMAP SfM"
        if scene:
            title += f": {display_name(str(scene))}"
        if input_images is not None:
            title += f"\n{input_images} input images; {len(cameras)} registered"
    else:
        title = "COLMAP SfM: Sparse 3D Points and Registered Cameras"

    ax.set(xlim=(midpoint[0] - radius, midpoint[0] + radius),
           ylim=(midpoint[1] - radius, midpoint[1] + radius),
           zlim=(midpoint[2] - radius, midpoint[2] + radius),
           xlabel="X (arbitrary SfM units)", ylabel="Y (arbitrary SfM units)",
           zlabel="Z (arbitrary SfM units)", title=title)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=args.elev, azim=args.azim, roll=args.roll)
    if args.no_grid:
        ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.025, 0.93))
    if args.trim_percentile:
        fig.text(
            0.5, 0.015,
            (f"Display axes use {args.trim_percentile:g}th to "
             f"{100 - args.trim_percentile:g}th point percentiles; cameras retained"),
            ha="center", fontsize=9, color="dimgray",
        )
    fig.subplots_adjust(left=0.03, right=0.95, bottom=0.07, top=0.88)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {args.output.resolve()}")
    print(f"Sparse model: {args.model.resolve()} ({model_format})")
    print(f"Registered cameras: {len(cameras)}; sparse points: {point_total}"
          + (f"; displayed: {len(points)}" if len(points) != point_total else ""))
    if metadata.get("metadata_file"):
        print(f"Run metadata: {Path(metadata['metadata_file']).resolve()}")
    if args.trim_percentile:
        print(
            f"NOTE: Display axes trimmed to [{args.trim_percentile}, "
            f"{100 - args.trim_percentile}] point percentiles; cameras retained"
        )
    print("Important: SfM axes are not metric metres unless you have calibrated/aligned scale.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot COLMAP sparse SfM points and cameras without rerunning COLMAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=Path, required=True, metavar="MODEL_OR_RUN_DIR",
        help=("Text/binary model directory, sparse directory, or run directory; "
              "the selected model is discovered automatically"),
    )
    parser.add_argument("--output", type=Path, required=True, metavar="IMAGE", help="Plot image to save")
    parser.add_argument("--title", default=None, help="Exact plot title; otherwise inferred from run metadata")
    parser.add_argument("--scene", default=None, help="Scene label override")
    parser.add_argument("--images", type=int, default=None, help="Input image count override")
    parser.add_argument("--label-cameras", action="store_true", help="Show image names next to camera centers")
    parser.add_argument("--max-points", type=int, default=100000, help="Subsample for display only; 0 means all")
    parser.add_argument("--point-size", type=float, default=2.0, help="Sparse-point marker size")
    parser.add_argument("--camera-size", type=float, default=32.0, help="Camera-center marker size")
    parser.add_argument("--elev", type=float, default=22.0, help="Viewing elevation in degrees")
    parser.add_argument("--azim", type=float, default=-64.0, help="Viewing azimuth in degrees")
    parser.add_argument("--roll", type=float, default=0.0, help="Camera roll in degrees")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic point-sampling seed")
    parser.add_argument(
        "--trim-percentile", type=float, default=0.0,
        help="Optional plot-only point outlier trim from each tail, from 0 to 10",
    )
    parser.add_argument("--no-grid", action="store_true", help="Hide axes and grid for a clean slide")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output image")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve model/metadata and print what would be plotted without loading all points",
    )
    args = parser.parse_args()

    if args.max_points < 0:
        parser.error("--max-points must be 0 (all) or a positive integer")
    if args.point_size <= 0 or args.camera_size <= 0 or args.dpi <= 0:
        parser.error("--point-size, --camera-size and --dpi must be positive")
    if args.images is not None and args.images < 1:
        parser.error("--images must be positive")
    if not 0 <= args.trim_percentile <= 10:
        parser.error("--trim-percentile must be between 0 and 10")
    try:
        args.model, model_format = resolve_sparse_model(args.model)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output.suffix.lower() not in IMAGE_OUTPUT_SUFFIXES:
        parser.error(
            "--output must use an image extension: "
            + ", ".join(sorted(IMAGE_OUTPUT_SUFFIXES))
        )
    model_inputs = {
        (args.model / filename).resolve()
        for filename in ("images.txt", "points3D.txt", "images.bin", "points3D.bin")
        if (args.model / filename).is_file()
    }
    if args.output.resolve() in model_inputs:
        parser.error("--output cannot overwrite a COLMAP model file")

    metadata = read_run_metadata(args.model)
    if args.dry_run:
        scene = args.scene or metadata.get("scene_group") or metadata.get("dataset") or infer_scene(args.model)
        input_images = args.images if args.images is not None else metadata.get("input_images")
        if input_images is None:
            input_images = infer_image_count(args.model)
        print(f"Resolved sparse model: {args.model.resolve()}")
        print(f"Model format: {model_format}")
        print(f"Output image: {args.output.resolve()}")
        print(f"Scene label: {display_name(str(scene)) if scene else 'COLMAP'}")
        print(f"Input images: {input_images if input_images is not None else 'unknown'}")
        if metadata.get("metadata_file"):
            print(f"Run metadata: {Path(metadata['metadata_file']).resolve()}")
        return
    if args.output.exists() and not args.overwrite:
        parser.error(
            f"Output file already exists: {args.output}. Use --overwrite or choose another filename."
        )
    try:
        plot_sparse(args, model_format)
    except (OSError, UnicodeError, ValueError, struct.error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
