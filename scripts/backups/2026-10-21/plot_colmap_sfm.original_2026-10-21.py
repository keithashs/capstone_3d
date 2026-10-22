r"""Plot real COLMAP SfM sparse points and registered camera poses.

Designed for the text model produced by COLMAP's `model_converter --output_type TXT`.
Run this after an existing reconstruction: no images are reprocessed.

Example (from capstone_3d project root):
    python scripts\plot_colmap_sfm.py \
        --model "colmap_output\office_26\office_undistorted_8img\sparse_txt" \
        --output "experiments\office_26\office_undistorted_8img\sfm_points_cameras.png" \
        --label-cameras
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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


def main():
    parser = argparse.ArgumentParser(description="Plot COLMAP sparse SfM points and registered cameras")
    parser.add_argument("--model", type=Path, required=True, help="Directory containing COLMAP images.txt and points3D.txt")
    parser.add_argument("--output", type=Path, required=True, help="PNG output path")
    parser.add_argument("--title", default="COLMAP SfM: Sparse 3D Points and Registered Cameras")
    parser.add_argument("--label-cameras", action="store_true", help="Show image names next to camera centers")
    parser.add_argument("--max-points", type=int, default=100000, help="Subsample for display only; 0 means all")
    args = parser.parse_args()

    img_path, pts_path = args.model / "images.txt", args.model / "points3D.txt"
    for p in (img_path, pts_path):
        if not p.is_file():
            parser.error(f"Missing {p}. First run COLMAP model_converter --output_type TXT.")
    cameras = read_images_txt(img_path)
    points, colors = read_points3d_txt(pts_path)
    point_total = len(points)
    if args.max_points > 0 and len(points) > args.max_points:
        sel = np.random.default_rng(0).choice(len(points), args.max_points, replace=False)
        points, colors = points[sel], colors[sel]

    centers = np.asarray([x["center"] for x in cameras])
    extents = np.vstack([centers, points]) if len(points) else centers
    low, high = extents.min(axis=0), extents.max(axis=0)
    span = max(float(np.max(high - low)), 0.001)
    camera_length = span * 0.045
    midpoint = 0.5 * (low + high)
    radius = span * 0.55

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2, alpha=0.85,
                   depthshade=False, label=f"Sparse 3D points ({point_total:,})")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="crimson", s=32,
               marker="^", depthshade=False, label=f"Registered cameras ({len(cameras)})")
    for cam in cameras:
        draw_frustum(ax, cam["center"], cam["rotation"], camera_length)
        if args.label_cameras:
            x, y, z = cam["center"]
            ax.text(x, y, z, cam["name"], fontsize=7)

    ax.set(xlim=(midpoint[0] - radius, midpoint[0] + radius),
           ylim=(midpoint[1] - radius, midpoint[1] + radius),
           zlim=(midpoint[2] - radius, midpoint[2] + radius),
           xlabel="X (arbitrary SfM units)", ylabel="Y (arbitrary SfM units)",
           zlabel="Z (arbitrary SfM units)", title=args.title)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=-64)
    ax.legend(loc="upper left")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"Output file already exists: {args.output}. Choose another --output filename.")
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {args.output.resolve()}")
    print(f"Registered cameras: {len(cameras)}; sparse points: {point_total}"
          + (f"; displayed: {len(points)}" if len(points) != point_total else ""))
    print("Important: SfM axes are not metric metres unless you have calibrated/aligned scale.")


if __name__ == "__main__":
    main()
