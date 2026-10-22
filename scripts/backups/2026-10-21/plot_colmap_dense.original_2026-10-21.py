r"""Plot a genuine COLMAP dense fused point cloud (fused.ply) as a 3D PNG.

This visualizes COLMAP's Stereo Fusion output, NOT the sparse SfM model.
No reconstruction is performed or modified.

Install: python -m pip install numpy matplotlib

Example (run from capstone_3d root on Windows):
 python scripts\plot_colmap_dense.py --input "colmap_output\office_26\office_undistorted_8img\dense\fused.ply" --output "experiments\office_26\office_undistorted_8img\dense_fused_plot.png" --scene Office --images 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Save a PNG without needing a GUI window.
import matplotlib.pyplot as plt


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

    label = args.scene if args.scene else "COLMAP"
    image_label = f": {args.images} images" if args.images else ""
    point_label = f"{len(xyz):,} fused 3D points"
    if len(show_xyz) < len(xyz):
        point_label += f" ({len(show_xyz):,} displayed)"
    ax.set_title(f"{label}{image_label}\n{point_label}", pad=14)
    fig.text(0.5, 0.018, "Dense Stereo Fusion point cloud · not metric-scale aligned",
             ha="center", fontsize=9, color="dimgray")
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.08, top=0.88)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"PLY: {args.input.resolve()}")
    print(f"Valid 3D points: {len(xyz):,} / {original_total:,}")
    print(f"Displayed points: {len(show_xyz):,}")
    print(f"Vertex RGB colours: {'yes' if rgb is not None else 'no'}")
    if args.trim_percentile:
        print(f"NOTE: Display axes trimmed to [{args.trim_percentile}, {100-args.trim_percentile}] percentiles; full point cloud unchanged")
    print(f"Saved PNG: {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Plot COLMAP dense/fused.ply as a 3D PNG")
    parser.add_argument("--input", required=True, type=Path, help="Existing COLMAP dense/fused.ply")
    parser.add_argument("--output", required=True, type=Path, help="PNG file to save")
    parser.add_argument("--scene", default="COLMAP", help="Scene label, e.g. Office")
    parser.add_argument("--images", type=int, default=None, help="Input image count for figure title")
    parser.add_argument("--max-points", type=int, default=200000, help="Maximum plotted points, never modifies PLY")
    parser.add_argument("--point-size", type=float, default=1.8, help="Matplotlib dot size")
    parser.add_argument("--elev", type=float, default=22, help="Viewing elevation (degrees)")
    parser.add_argument("--azim", type=float, default=-62, help="Viewing azimuth (degrees)")
    parser.add_argument("--dpi", type=int, default=200, help="Output PNG DPI")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic downsampling seed")
    parser.add_argument("--trim-percentile", type=float, default=0.0, help="Optional plot-only outlier trim: 0–10")
    parser.add_argument("--no-grid", action="store_true", help="Hide axes and grid for a clean slide")
    parser.add_argument("--roll", type=float, default=0, help="Camera roll in degrees")
    args = parser.parse_args()
    if args.max_points < 1 or args.point_size <= 0 or args.dpi <= 0:
        parser.error("--max-points, --point-size and --dpi must be positive")
    if not 0 <= args.trim_percentile <= 10:
        parser.error("--trim-percentile must be between 0 and 10")
    plot_dense(args)


if __name__ == "__main__":
    main()
