#!/usr/bin/env python3
r"""Run official MapAnything image-only inference with live console + saved log.

Place this file in: capstone_3d/scripts/run_mapanything_timed.py
Run from Anaconda Prompt with the 'mapanything' Conda environment active:
    cd /d "%USERPROFILE%\Desktop\capstone_3d"
    python scripts\run_mapanything_timed.py

No third-party packages are required by this wrapper. MapAnything must already
be installed and its official demo script must be present.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_FOLDER = PROJECT_ROOT / "data" / "mapanything_smoke_test"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "mapanything_output" / "smoke_test" / "scene.glb"
)
OFFICIAL_DEMO = (
    PROJECT_ROOT / "tools" / "map-anything" / "scripts"
    / "demo_images_only_inference.py"
)


def display_elapsed(seconds: float) -> str:
    """Render elapsed wall time as HH:MM:SS, including hours >= 24."""
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER,
        help="Folder containing input photos (default: project data/mapanything_smoke_test).",
    )
    parser.add_argument(
        "--output-path", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Output GLB path (default: project mapanything_output/smoke_test/scene.glb).",
    )
    return parser.parse_args()


def main() -> int:
    args = cli()
    images = args.image_folder.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not OFFICIAL_DEMO.is_file():
        print(f"ERROR: Official MapAnything script not found:\n{OFFICIAL_DEMO}", file=sys.stderr)
        return 2
    if not images.is_dir():
        print(f"ERROR: Image folder not found:\n{images}", file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    log_path = output_path.parent / f"mapanything_{stamp}.log"
    command = [
        sys.executable, "-u", str(OFFICIAL_DEMO),
        "--image_folder", str(images),
        "--save_glb", "--output_path", str(output_path),
    ]

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        def tee(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")

        tee("=== MapAnything: timed image-only inference ===")
        tee(f"Start time: {now.isoformat(timespec='seconds')}")
        tee(f"Working directory: {OFFICIAL_DEMO.parent.parent}")
        tee(f"Input images: {images}")
        tee(f"GLB output: {output_path}")
        tee(f"Command: {subprocess.list2cmdline(command)}")
        tee(f"Log file: {log_path}")
        tee("=== Begin MapAnything terminal output ===")

        total_start = time.perf_counter()
        infer_start: float | None = None
        infer_end: float | None = None
        returncode = 1
        try:
            process = subprocess.Popen(
                command,
                cwd=str(OFFICIAL_DEMO.parent.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    observed_at = time.perf_counter()
                    print(line, end="", flush=True)
                    log.write(line)
                    if "Running inference..." in line and infer_start is None:
                        infer_start = observed_at
                    if "Inference complete!" in line and infer_end is None:
                        infer_end = observed_at
                returncode = process.wait()
            except KeyboardInterrupt:
                tee("\nInterrupted by user (Ctrl+C).")
                if process.poll() is None:
                    process.terminate()
                process.wait()
                returncode = 130
            finally:
                process.stdout.close()
        except OSError as error:
            tee(f"ERROR: Could not launch MapAnything: {error}")
            returncode = 1

        total_elapsed = time.perf_counter() - total_start
        end = datetime.now().astimezone()
        tee("=== End MapAnything terminal output ===")
        tee(f"End time: {end.isoformat(timespec='seconds')}")
        if infer_start is not None and infer_end is not None:
            inference_elapsed = max(0.0, infer_end - infer_start)
            tee(f"Inference elapsed (between status messages): {display_elapsed(inference_elapsed)} ({inference_elapsed:.2f} s)")
        else:
            tee("Inference elapsed: unavailable (expected status messages not both found)")
        tee(f"Total elapsed (model loading + inference + GLB export): {display_elapsed(total_elapsed)} ({total_elapsed:.2f} s)")
        tee(f"Process exit code: {returncode}")
        if output_path.is_file() and output_path.stat().st_size > 0:
            tee(f"GLB exists: yes ({output_path.stat().st_size:,} bytes)")
        else:
            tee("GLB exists: no (or file is empty)")
        tee(f"Saved terminal log: {log_path}")

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
