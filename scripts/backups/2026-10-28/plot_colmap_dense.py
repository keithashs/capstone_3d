r"""Plot a genuine COLMAP dense fused point cloud (fused.ply) as a 3D image.

This visualizes COLMAP's Stereo Fusion output, NOT the sparse SfM model.
No reconstruction is performed or modified.

``--input`` may be the PLY itself, its ``dense`` directory, or a complete
``colmap_output/<scene>/<run>`` directory.  A run directory is especially
convenient for the 6/8/12-image baselines and the full office/kicker runs.

Install: python -m pip install numpy matplotlib

Example (run from capstone_3d root on Windows):
 python scripts\plot_colmap_dense.py --input "colmap_output\office_26\office_full26_sift1600_ratio080" --output "experiments\office_26\office_full26_sift1600_ratio080\dense_fused_plot.png"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Save a PNG without needing a GUI window.
import matplotlib.pyplot as plt

IMAGE_OUTPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".tif", ".tiff", ".webp"}


def resolve_fused_ply(input_path: Path) -> Path:
    """Resolve one fused cloud from a PLY, dense directory, or run directory.

    Recursive discovery is intentionally strict: silently choosing one of
    several clouds could put the wrong reconstruction in an experiment report.
    """
    path = input_path.expanduser()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Input must be a PLY file or directory: {path}")

    direct = [path / "fused.ply", path / "dense" / "fused.ply"]
    direct = [candidate for candidate in direct if candidate.is_file()]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        # This is only possible for a directory containing both a cloud and a
        # nested dense workspace; require an explicit choice rather than guess.
        choices = "\n  - ".join(str(candidate) for candidate in direct)
        raise ValueError(f"Multiple fused clouds found; pass one explicitly:\n  - {choices}")

    matches = sorted(path.rglob("fused.ply"), key=lambda item: str(item).lower())
    if not matches:
        raise FileNotFoundError(
            f"No fused.ply found below {path}. Run COLMAP stereo_fusion first."
        )
    if len(matches) > 1:
        choices = "\n  - ".join(str(candidate) for candidate in matches)
        raise ValueError(f"Multiple fused clouds found; pass one explicitly:\n  - {choices}")
    return matches[0]


def _experiment_run_dir(artifact: Path) -> Path | None:
    """Map colmap_output/<group>/<run>/... to experiments/<group>/<run>."""
    resolved = artifact.resolve()
    for parent in resolved.parents:
        if parent.name.lower() != "colmap_output":
            continue
        relative = resolved.relative_to(parent)
        if len(relative.parts) >= 2:
            return parent.parent / "experiments" / relative.parts[0] / relative.parts[1]
    return None


def read_run_metadata(artifact: Path) -> dict:
    """Read nearby run metadata without requiring a particular run name."""
    locations: list[Path] = []
    for parent in (artifact.parent, *artifact.parents):
        locations.append(parent)
    experiment_dir = _experiment_run_dir(artifact)
    if experiment_dir is not None:
        locations.insert(0, experiment_dir)

    seen: set[Path] = set()
    metadata: dict = {}
    for folder in locations:
        try:
            key = folder.resolve()
        except OSError:
            key = folder
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
                # summary.json is checked first and therefore wins on overlap.
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
    """Infer counts from conventional run IDs only when metadata is absent."""
    for part in reversed(path.parts):
        match = re.search(r"(?:^|[_-])(\d+)(?:img|images?)(?:$|[_-])", part, re.IGNORECASE)
        if not match:
            match = re.search(r"(?:^|[_-])full[_-]?(\d+)(?:$|[_-])", part, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def infer_scene(path: Path) -> str | None:
    """Infer the scene group from a standard colmap_output path."""
    resolved = path.resolve()
    for parent in resolved.parents:
        if parent.name.lower() == "colmap_output":
            relative = resolved.relative_to(parent)
            return relative.parts[0] if relative.parts else None
    return None


def display_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip().title()


def read_ply_vertices(path: Path):
    """Read vertex properties from ASCII/binary little- or big-endian PLY.

    Supports COLMAP fused.ply with ordinary scalar vertex properties; face data
    and other elements after vertices are not read or modified.
    """
    field_types = {
        "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
        "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
        "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    }
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Not a PLY file")
        encoding = None
        vertex_count = None
        element = None
        fields = []
        for _ in range(10000):
            raw = stream.readline()
            if not raw:
                raise ValueError("PLY header ended unexpectedly")
            parts = raw.decode("ascii", errors="replace").strip().split()
            if not parts or parts[0] in ("comment", "obj_info"):
                continue
            if parts[0] == "format":
                encoding = parts[1]
            elif parts[0] == "element":
                element = parts[1]
                if element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and element == "vertex":
                if parts[1] == "list":
                    raise ValueError("PLY vertex list properties are not supported")
                if parts[1] not in field_types:
                    raise ValueError(f"Unsupported PLY property type: {parts[1]}")
                fields.append((parts[2], field_types[parts[1]]))
            elif parts[0] == "end_header":
                break
        else:
            raise ValueError("PLY header too long")

        if vertex_count is None or not fields:
            raise ValueError("PLY does not contain scalar vertex properties")
        if encoding == "ascii":
            data = np.loadtxt(stream, dtype=np.dtype(fields), max_rows=vertex_count, ndmin=1)
        elif encoding in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if encoding == "binary_little_endian" else ">"
            dtype = np.dtype([(name, np.dtype(kind).newbyteorder(endian)) for name, kind in fields])
            data = np.fromfile(stream, dtype=dtype, count=vertex_count)
        else:
            raise ValueError(f"Unsupported PLY format: {encoding}")
        if len(data) != vertex_count:
            raise ValueError(f"PLY was truncated: expected {vertex_count} vertices, read {len(data)}")
    return data


def read_fused_ply(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"COLMAP dense point cloud not found: {path}")

    vertex = read_ply_vertices(path)
    names = set(vertex.dtype.names or [])
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertices are missing x/y/z coordinates")

    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    color_cols = next((cols for cols in (("red", "green", "blue"),
                                           ("diffuse_red", "diffuse_green", "diffuse_blue"))
                       if set(cols).issubset(names)), None)
    if color_cols is not None:
        colors = np.column_stack([vertex[name] for name in color_cols]).astype(np.float64)
        if colors.size and np.nanmax(colors) > 1.0:
            colors /= 255.0
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = None

    valid = np.isfinite(xyz).all(axis=1)
    if colors is not None:
        valid &= np.isfinite(colors).all(axis=1)
    return xyz[valid], (colors[valid] if colors is not None else None), len(vertex)


def plot_dense(args):
    xyz, rgb, original_total = read_fused_ply(args.input)
    if not len(xyz):
        raise ValueError("No finite 3D points in the supplied PLY")

    # Only for rendering speed: the source PLY is NEVER changed.
    if len(xyz) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(xyz), size=args.max_points, replace=False))
        show_xyz = xyz[keep]
        show_rgb = rgb[keep] if rgb is not None else None
    else:
        show_xyz, show_rgb = xyz, rgb

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.scatter(show_xyz[:, 0], show_xyz[:, 1], show_xyz[:, 2],
               c=show_rgb if show_rgb is not None else "#555555",
               s=args.point_size, marker=".", depthshade=False,
               linewidths=0, alpha=1.0, rasterized=True)

    # Calculate axes from the FULL cloud, not the downsampled display points.
    # Opt-in percentile limits can be useful when isolated outliers stretch the axes.
    if args.trim_percentile == 0:
        mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    else:
        mins = np.percentile(xyz, args.trim_percentile, axis=0)
        maxs = np.percentile(xyz, 100 - args.trim_percentile, axis=0)

    center = (mins + maxs) / 2.0
    spans = np.maximum(maxs - mins, 1e-6)
    ax.set_xlim(center[0] - spans[0] * 0.53, center[0] + spans[0] * 0.53)
    ax.set_ylim(center[1] - spans[1] * 0.53, center[1] + spans[1] * 0.53)
    ax.set_zlim(center[2] - spans[2] * 0.53, center[2] + spans[2] * 0.53)
    ax.set_box_aspect(spans)
    ax.view_init(elev=args.elev, azim=args.azim, roll=args.roll)
    ax.set_xlabel("X (arbitrary SfM units)", labelpad=7)
    ax.set_ylabel("Y (arbitrary SfM units)", labelpad=7)
    ax.set_zlabel("Z (arbitrary SfM units)", labelpad=7)
    if args.no_grid:
        ax.set_axis_off()

    metadata = read_run_metadata(args.input)
    scene = args.scene or metadata.get("scene_group") or metadata.get("dataset") or infer_scene(args.input)
    label = display_name(str(scene)) if scene else "COLMAP"
    image_count = args.images if args.images is not None else metadata.get("input_images")
    if image_count is None:
        image_count = infer_image_count(args.input)
    image_label = f": {image_count} images" if image_count is not None else ""
    point_label = f"{len(xyz):,} fused 3D points"
    if len(show_xyz) < len(xyz):
        point_label += f" ({len(show_xyz):,} displayed)"
    ax.set_title(f"{label}{image_label}\n{point_label}", pad=14)
    footer = "Dense Stereo Fusion point cloud - not metric-scale aligned"
    if args.trim_percentile:
        footer += (f" - display axes use {args.trim_percentile:g}th to "
                   f"{100 - args.trim_percentile:g}th percentiles")
    fig.text(0.5, 0.018, footer, ha="center", fontsize=9, color="dimgray")
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.08, top=0.88)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"PLY: {args.input.resolve()}")
    print(f"Valid 3D points: {len(xyz):,} / {original_total:,}")
    print(f"Displayed points: {len(show_xyz):,}")
    print(f"Vertex RGB colours: {'yes' if rgb is not None else 'no'}")
    if metadata.get("metadata_file"):
        print(f"Run metadata: {Path(metadata['metadata_file']).resolve()}")
    if args.trim_percentile:
        print(f"NOTE: Display axes trimmed to [{args.trim_percentile}, {100-args.trim_percentile}] percentiles; full point cloud unchanged")
    print(f"Saved PNG: {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a COLMAP dense fused point cloud without rerunning COLMAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, type=Path, metavar="PLY_OR_RUN_DIR",
        help="fused.ply, its dense directory, or a run directory containing one fused.ply",
    )
    parser.add_argument("--output", required=True, type=Path, metavar="IMAGE", help="Plot image to save")
    parser.add_argument(
        "--scene", default=None,
        help="Scene label; inferred from summary.json or the colmap_output path when omitted",
    )
    parser.add_argument(
        "--images", type=int, default=None,
        help="Input image count; inferred from summary/manifest/run ID when omitted",
    )
    parser.add_argument("--max-points", type=int, default=200000, help="Maximum plotted points, never modifies PLY")
    parser.add_argument("--point-size", type=float, default=1.8, help="Matplotlib dot size")
    parser.add_argument("--elev", type=float, default=22, help="Viewing elevation (degrees)")
    parser.add_argument("--azim", type=float, default=-62, help="Viewing azimuth (degrees)")
    parser.add_argument("--dpi", type=int, default=200, help="Output PNG DPI")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic downsampling seed")
    parser.add_argument("--trim-percentile", type=float, default=0.0, help="Optional plot-only outlier trim: 0 to 10")
    parser.add_argument("--no-grid", action="store_true", help="Hide axes and grid for a clean slide")
    parser.add_argument("--roll", type=float, default=0, help="Camera roll in degrees")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output image")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve the input and metadata, print what would be plotted, and do not render",
    )
    args = parser.parse_args()
    if args.max_points < 1 or args.point_size <= 0 or args.dpi <= 0:
        parser.error("--max-points, --point-size and --dpi must be positive")
    if not 0 <= args.trim_percentile <= 10:
        parser.error("--trim-percentile must be between 0 and 10")
    if args.images is not None and args.images < 1:
        parser.error("--images must be positive")
    try:
        args.input = resolve_fused_ply(args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output.suffix.lower() not in IMAGE_OUTPUT_SUFFIXES:
        parser.error(
            "--output must use an image extension: "
            + ", ".join(sorted(IMAGE_OUTPUT_SUFFIXES))
        )
    if args.output.resolve() == args.input.resolve():
        parser.error("--output cannot be the input fused.ply")
    if args.dry_run:
        metadata = read_run_metadata(args.input)
        image_count = args.images if args.images is not None else metadata.get("input_images")
        if image_count is None:
            image_count = infer_image_count(args.input)
        scene = args.scene or metadata.get("scene_group") or metadata.get("dataset") or infer_scene(args.input)
        print(f"Resolved PLY: {args.input.resolve()}")
        print(f"Output image: {args.output.resolve()}")
        print(f"Scene label: {display_name(str(scene)) if scene else 'COLMAP'}")
        print(f"Input images: {image_count if image_count is not None else 'unknown'}")
        if metadata.get("metadata_file"):
            print(f"Run metadata: {Path(metadata['metadata_file']).resolve()}")
        return
    if args.output.exists() and not args.overwrite:
        parser.error(
            f"Output file already exists: {args.output}. Use --overwrite or choose another filename."
        )
    try:
        plot_dense(args)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
