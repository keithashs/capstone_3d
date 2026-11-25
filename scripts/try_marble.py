"""Preview or submit a Marble 1.1 world from classroom images or a local video."""

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT = Path(__file__).resolve().parents[1]
IMAGES = PROJECT / "data" / "e2_04_03_77_20261027_015732_676342" / "images"
API = "https://api.worldlabs.ai/marble/v1"
STARTER_FRAMES = [57, 59, 61, 63]
VIDEO_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}
MAX_VIDEO_BYTES = 100_000_000


def api(path, key, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(API + path, data=data, headers={"Content-Type": "application/json"})
    request.add_unredirected_header("WLT-Api-Key", key)
    try:
        with urlopen(request, timeout=90) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"{request.get_method()} {path} returned HTTP {error.code}: {detail}") from error


def upload(path, key, kind="image"):
    is_video = kind == "video"
    extension = path.suffix.lstrip(".").lower() if is_video else "jpg"
    content_type = VIDEO_TYPES[path.suffix.lower()] if is_video else "image/jpeg"
    prepared = api("/media-assets:prepare_upload", key, {
        "file_name": path.name, "kind": kind, "extension": extension,
    })
    asset = prepared["media_asset"]
    asset_id = asset.get("media_asset_id") or asset["id"]
    info = prepared["upload_info"]
    headers = dict(info.get("required_headers") or {})
    if not any(name.lower() == "content-type" for name in headers):
        headers["Content-Type"] = content_type
    # The signed URL authenticates the upload. Only the API receives the key.
    request = Request(info["upload_url"], method=info.get("upload_method", "PUT"),
                      data=path.read_bytes(), headers=headers)
    try:
        with urlopen(request, timeout=600 if is_video else 90):
            pass
    except HTTPError as error:
        raise RuntimeError(f"Upload of {path.name} returned HTTP {error.code}.") from error
    return asset_id


def selected_frames(numbers, images_dir=IMAGES):
    if not 2 <= len(numbers) <= 8 or len(set(numbers)) != len(numbers):
        raise ValueError("Choose 2 to 8 different frame numbers for reconstruction.")
    if not images_dir.is_dir():
        raise ValueError(f"Image folder not found or not a directory: {images_dir}")
    paths = [images_dir / f"{n:04d}.jpg" for n in numbers]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty frame: {path}")
    return paths


def selected_video(path):
    path = path.expanduser()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty video: {path}")
    if path.suffix.lower() not in VIDEO_TYPES:
        raise ValueError("Choose an MP4, WebM, MOV, or AVI video.")
    if path.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError("Video exceeds the 100 MB limit. Compress or shorten it first.")
    return path


def build_request(asset_ids, display_name=None, text_prompt=None, *, video=False):
    if video:
        name = display_name or "Video world"
        prompt = {
            "type": "video",
            "video_prompt": {"source": "media_asset", "media_asset_id": asset_ids[0]},
        }
    elif asset_ids:
        name = display_name or "E2-04-03 classroom"
        prompt = {
            "type": "multi-image",
            "reconstruct_images": True,
            "multi_image_prompt": [
                {"content": {"source": "media_asset", "media_asset_id": asset_id}}
                for asset_id in asset_ids
            ],
        }
        if text_prompt:
            prompt["text_prompt"] = text_prompt
    else:
        name = "Mystical Forest"
        prompt = {"type": "text", "text_prompt": "A mystical forest with glowing mushrooms"}
    return {"display_name": name, "model": "marble-1.1", "world_prompt": prompt}


def watch(operation_id, key):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", operation_id):
        raise ValueError("Expected an operation ID such as the UUID printed after submission.")
    print(f"Resume polling: python -B scripts/try_marble.py status {operation_id}", flush=True)
    previous = None
    while True:
        op = api(f"/operations/{operation_id}", key)
        if op.get("error") or op["done"]:
            break
        progress = (op.get("metadata") or {}).get("progress") or {}
        label = progress.get("description") or progress.get("status") or "Generating world..."
        if label != previous:
            print(label, flush=True)
            previous = label
        time.sleep(15)
    if op.get("error"):
        raise RuntimeError(f"World generation failed for {operation_id}: {op['error']}")
    result = op.get("response") or {}
    world_id = result.get("world_id") or result.get("id") or (op.get("metadata") or {}).get("world_id")
    if not world_id:
        raise RuntimeError(f"Completed operation {operation_id} returned no world ID.")
    print("Open in Marble:", result.get("world_marble_url") or f"https://marble.worldlabs.ai/world/{world_id}")
    latest = api(f"/worlds/{world_id}", key)
    world = latest.get("world", latest)
    assets = world.get("assets") or result.get("assets") or {}
    splats = assets.get("splats") or {}
    for resolution, url in (splats.get("spz_urls") or {}).items():
        if url:
            print(f"Gaussian splat ({resolution} SPZ): {url}")
    mesh = assets.get("mesh") or {}
    imagery = assets.get("imagery") or {}
    for label, url in (
        ("Collider mesh (GLB)", mesh.get("collider_mesh_url")),
        ("Textured mesh (GLB)", mesh.get("hq_mesh_url")),
        ("Full-resolution mesh (GLB)", mesh.get("full_res_mesh_url")),
        ("Panorama", imagery.get("pano_url")),
    ):
        if url:
            print(f"{label}: {url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(display_name=None, text_prompt=None)
    commands = parser.add_subparsers(dest="command", required=True)
    images = commands.add_parser("images", help="Preview or submit a classroom world")
    images.add_argument("--images-dir", type=Path, default=IMAGES, metavar="FOLDER",
                        help="Folder containing numbered JPEGs such as 0057.jpg (default: classroom dataset)")
    images.add_argument("--frames", nargs="+", type=int, default=STARTER_FRAMES,
                        help="2 to 8 frame numbers, e.g. 60 for 0060.jpg")
    images.add_argument("--display-name", type=str.strip, help="World title in Marble (up to 64 characters)")
    images.add_argument("--prompt", dest="text_prompt", type=str.strip,
                        help="Optional text guidance to accompany the selected images")
    video = commands.add_parser("video", help="Preview or submit a world from a local video")
    video.add_argument("--video-file", type=Path, required=True, metavar="FILE",
                       help="Path to an MP4, WebM, MOV, or AVI file, up to 100 MB")
    video.add_argument("--display-name", type=str.strip,
                       help="World title in Marble (up to 64 characters)")
    forest = commands.add_parser("text", help="Preview or submit the Mystical Forest example")
    for command in (images, video, forest):
        command.add_argument("--submit", action="store_true", help="Send a paid generation request")
    commands.add_parser("credits", help="Check authentication and remaining API credits")
    commands.add_parser("status", help="Resume polling an existing job").add_argument("operation_id")
    args = parser.parse_args()
    if args.display_name is not None and not 1 <= len(args.display_name) <= 64:
        (video if args.command == "video" else images).error(
            "--display-name must contain 1 to 64 characters")

    paths = []
    if args.command == "images":
        images_dir = args.images_dir.expanduser()
        paths = selected_frames(args.frames, images_dir)
        image_count = sum(file.is_file() for file in images_dir.glob("*.jpg"))
        print(f"Selected {len(paths)} of {image_count} JPEGs in {images_dir}")
        print(", ".join(path.name for path in paths))
    elif args.command == "video":
        paths = [selected_video(args.video_file)]
        print(f"Selected video: {paths[0]} ({paths[0].stat().st_size / 1_000_000:.2f} MB)")
    if args.command in ("images", "video", "text") and not args.submit:
        placeholders = [f"<uploaded ID for {path.name}>" for path in paths]
        print("Offline preview. Add --submit to generate this world.")
        print(json.dumps(build_request(placeholders, args.display_name, args.text_prompt,
                                       video=args.command == "video"), indent=2))
        return

    key = os.getenv("WLT_API_KEY") or getpass.getpass("World Labs API key (hidden): ")
    if not key.strip():
        raise ValueError("An API key is required.")
    key = key.strip()
    if args.command == "credits":
        print("Remaining API credits:", api("/credits", key)["remaining_credits"])
    elif args.command == "status":
        watch(args.operation_id, key)
    else:
        asset_ids = []
        for path in paths:
            print(f"Uploading {path.name}...", flush=True)
            asset_ids.append(upload(path, key, kind="video" if args.command == "video" else "image"))
        payload = build_request(asset_ids, args.display_name, args.text_prompt,
                                video=args.command == "video")
        print("Submitting one paid generation request...", flush=True)
        submitted = api("/worlds:generate", key, payload)
        operation_id = submitted["operation_id"]
        print(f"Operation ID: {operation_id}", flush=True)
        watch(operation_id, key)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped locally. Use status with the operation ID to continue polling.")
    except (OSError, ValueError, RuntimeError, KeyError, EOFError) as exc:
        sys.exit(f"Marble: {exc}")
