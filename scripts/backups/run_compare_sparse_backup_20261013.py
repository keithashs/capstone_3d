
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Uses the existing project folder containing this scripts folder.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAP_REPO = ROOT / "tools" / "map-anything"
MAP_SCRIPT = MAP_REPO / "scripts" / "demo_images_only_inference.py"
COLMAP = ROOT / "tools" / "COLMAP" / "COLMAP.bat"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def natural_key(path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.name)
    ]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8"
    )

def command_output(cmd, cwd=None):
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            errors="replace", timeout=30
        )
        return (result.stdout + result.stderr).strip()
    except Exception as exc:
        return f"Unavailable: {exc}"

def gpu_worker(stop, stage, samples):
    while not stop.is_set():
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=10
            )
            if result.returncode == 0:
                parts = result.stdout.splitlines()[0].split(",")
                def number(s):
                    try:
                        return float(s.strip())
                    except ValueError:
                        return None
                samples.append({
                    "timestamp": stamp(),
                    "stage": stage[0],
                    "gpu_memory_mib": number(parts[0]),
                    "gpu_util_percent": number(parts[1]),
                    "gpu_temperature_c": number(parts[2])
                })
        except Exception:
            pass
        stop.wait(1.0)

def run_step(name, cmd, cwd, logs, stage):
    stage[0] = name
    path = logs / f"{name}.log"
    start = time.perf_counter()

    print(f"\n[{stamp()}] START: {name}", flush=True)
    print("Command:", subprocess.list2cmdline(list(map(str, cmd))))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with path.open("w", encoding="utf-8") as log:
        log.write(f"Stage: {name}\nStart: {stamp()}\n")
        log.write(f"Working directory: {cwd}\n")
        log.write(f"Command: {subprocess.list2cmdline(list(map(str, cmd)))}\n\n")
        log.flush()

        process = subprocess.Popen(
            list(map(str, cmd)), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1
        )

        output = []
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            output.append(line)

        rc = process.wait()
        elapsed = round(time.perf_counter() - start, 3)

        log.write(
            f"\nEnd: {stamp()}\nExit code: {rc}\n"
            f"Elapsed seconds: {elapsed}\n"
        )

    print(f"[{stamp()}] END: {name}; exit={rc}; {elapsed}s")
    return rc, elapsed, "".join(output)

def database_stats(db_path):
    result = {}
    if not db_path.exists():
        return result
    with sqlite3.connect(db_path) as db:
        for label, sql in {
            "database_images": "SELECT COUNT(*) FROM images",
            "total_keypoints": "SELECT COALESCE(SUM(rows),0) FROM keypoints",
            "matched_pairs": "SELECT COUNT(*) FROM matches WHERE rows > 0",
            "verified_pairs": (
                "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"
            )
        }.items():
            try:
                result[label] = db.execute(sql).fetchone()[0]
            except sqlite3.Error:
                result[label] = None
    return result

def analyzer_metrics(text):
    fields = {
        "registered_images": r"Registered images:\s*(\d+)",
        "sparse_points": r"Points:\s*(\d+)",
        "observations": r"Observations:\s*(\d+)",
        "mean_reprojection_error_px":
            r"Mean reprojection error:\s*([\d.]+)"
    }
    result = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            result[key] = (
                float(value) if "error" in key else int(value)
            )
    return result

def write_csv(path, records, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=6,
                        help="Number of consecutive images; 0 = all")
    parser.add_argument("--start", type=int, default=0,
                        help="Zero-based starting image index")
    parser.add_argument("--run", default=None,
                        help="Optional unique run identifier")
    parser.add_argument("--single-camera", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.dataset):
        parser.error("Dataset name must contain only letters, numbers, _ or -")
    if args.run and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("Run ID must contain only letters, numbers, _ or -")
    if args.limit < 0 or args.start < 0:
        parser.error("--limit and --start cannot be negative")

    folder = DATA / args.dataset
    if (folder / "images").is_dir():
        folder = folder / "images"

    if not folder.is_dir():
        parser.error(f"Image folder not found: {folder}")
    if not MAP_SCRIPT.is_file():
        parser.error(f"MapAnything script not found: {MAP_SCRIPT}")
    if not COLMAP.is_file():
        parser.error(f"COLMAP launcher not found: {COLMAP}")

    images = sorted(
        [p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in EXTS],
        key=natural_key
    )
    if not images:
        parser.error(f"No supported images found in {folder}")

    selected = images[args.start:]
    if args.limit:
        selected = selected[:args.limit]
    if len(selected) < 2:
        parser.error("Select at least two input images")

    run_id = args.run or datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )

    exp = ROOT / "experiments" / args.dataset / run_id
    ma_out = ROOT / "mapanything_output" / args.dataset / run_id
    co_out = ROOT / "colmap_output" / args.dataset / run_id

    for path in (exp, ma_out, co_out):
        if path.exists():
            parser.error(f"Run already exists; choose another ID: {path}")

    exp.mkdir(parents=True)
    ma_out.mkdir(parents=True)
    co_out.mkdir(parents=True)
    logs = exp / "logs"
    logs.mkdir()
    snapshot = exp / "input_images"
    snapshot.mkdir()

    # One immutable image snapshot used by both methods.
    manifest = []
    for source in selected:
        target = snapshot / source.name
        shutil.copy2(source, target)
        manifest.append({
            "filename": source.name,
            "source": str(source),
            "bytes": target.stat().st_size,
            "sha256": sha256(target)
        })

    write_csv(exp / "input_manifest.csv", manifest,
              ["filename", "source", "bytes", "sha256"])

    environment = {
        "timestamp": stamp(),
        "root": str(ROOT),
        "dataset": args.dataset,
        "run_id": run_id,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "conda_environment": os.getenv("CONDA_DEFAULT_ENV"),
        "gpu": command_output(["nvidia-smi", "-L"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "mapanything_commit": command_output(
            ["git", "rev-parse", "HEAD"], MAP_REPO
        ),
        "colmap_launcher": str(COLMAP),
        "single_camera": args.single_camera,
        "image_count": len(selected),
        "image_start_index": args.start,
        "image_limit": args.limit,
    }

    for package in ("torch", "torchvision", "gradio"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None

    save_json(exp / "environment.json", environment)

    stage = ["idle"]
    samples = []
    stop = threading.Event()
    monitor = threading.Thread(
        target=gpu_worker, args=(stop, stage, samples),
        daemon=True
    )
    monitor.start()

    metrics = {
        "dataset": args.dataset,
        "run_id": run_id,
        "images": len(selected),
        "mapanything": {
            "status": "not_started",
            "elapsed_seconds": None
        },
        "colmap": {
            "status": "not_started",
            "elapsed_seconds": None
        },
        "stages": {}
    }

    try:
        # A: MapAnything, independent image-only reconstruction.
        glb = ma_out / "scene.glb"
        rc, sec, output = run_step(
            "mapanything",
            [
                sys.executable, str(MAP_SCRIPT),
                "--image_folder", str(snapshot),
                "--save_glb",
                "--output_path", str(glb)
            ],
            MAP_REPO, logs, stage
        )
        metrics["stages"]["mapanything"] = sec
        metrics["mapanything"] = {
            "status": "success" if rc == 0 and glb.is_file()
                      else "failed",
            "exit_code": rc,
            "elapsed_seconds": sec,
            "output": str(glb),
            "output_bytes": glb.stat().st_size if glb.exists() else None,
            "cuda_oom": "out of memory" in output.lower()
        }

        # B: Standalone COLMAP sparse SfM.
        db = co_out / "database.db"
        sparse = co_out / "sparse"
        sparse.mkdir()

        extra = (
            ["--ImageReader.single_camera", "1"]
            if args.single_camera else []
        )
        stages = [
            (
                "colmap_features",
                ["feature_extractor", "--database_path", str(db),
                 "--image_path", str(snapshot), *extra]
            ),
            (
                "colmap_matching",
                ["exhaustive_matcher", "--database_path", str(db)]
            ),
            (
                "colmap_mapping",
                ["mapper", "--database_path", str(db),
                 "--image_path", str(snapshot),
                 "--output_path", str(sparse)]
            )
        ]

        total = 0.0
        last_rc = 0
        for name, parameters in stages:
            rc, sec, _ = run_step(
                name,
                ["cmd.exe", "/d", "/c", str(COLMAP), *parameters],
                ROOT, logs, stage
            )
            total += sec
            metrics["stages"][name] = sec
            last_rc = rc
            if rc != 0:
                break

        db_metrics = database_stats(db)

        # Analyze all sparse models; choose most registered images.
        candidates = []
        if last_rc == 0:
            for model in sorted(sparse.iterdir()):
                if not model.is_dir():
                    continue
                if not ((model / "cameras.bin").exists()
                        or (model / "cameras.txt").exists()):
                    continue

                rc, sec, text = run_step(
                    f"colmap_analyzer_{model.name}",
                    ["cmd.exe", "/d", "/c", str(COLMAP),
                     "model_analyzer", "--path", str(model)],
                    ROOT, logs, stage
                )
                if rc == 0:
                    candidates.append(
                        (analyzer_metrics(text), model)
                    )

        candidates.sort(
            key=lambda item: (
                item[0].get("registered_images", 0),
                item[0].get("sparse_points", 0)
            ), reverse=True
        )

        best = candidates[0] if candidates else None
        metrics["colmap"] = {
            "status": "success" if best else "failed",
            "exit_code": last_rc,
            "elapsed_seconds": round(total, 3),
            "output": str(best[1]) if best else None,
            **db_metrics,
            **(best[0] if best else {})
        }

    except Exception as exc:
        (logs / "runner_error.log").write_text(
            repr(exc), encoding="utf-8"
        )
        metrics["runner_error"] = repr(exc)

    finally:
        stage[0] = "finished"
        stop.set()
        monitor.join(timeout=12)

        write_csv(
            exp / "gpu_samples.csv", samples,
            ["timestamp", "stage", "gpu_memory_mib",
             "gpu_util_percent", "gpu_temperature_c"]
        )

        for method in ("mapanything", "colmap"):
            readings = [
                x["gpu_memory_mib"] for x in samples
                if x["stage"].startswith(method)
                and x["gpu_memory_mib"] is not None
            ]
            metrics[method]["peak_sampled_gpu_mib"] = (
                max(readings) if readings else None
            )

        save_json(exp / "summary.json", metrics)

        fields = [
            "dataset", "run_id", "method", "images", "status",
            "elapsed_seconds", "peak_sampled_gpu_mib",
            "registered_images", "sparse_points",
            "mean_reprojection_error_px", "verified_pairs", "output"
        ]
        rows = []
        for method in ("mapanything", "colmap"):
            info = metrics[method]
            row = {
                "dataset": args.dataset,
                "run_id": run_id,
                "method": method,
                "images": len(selected),
                "status": info["status"]
            }
            for key in fields:
                if key in info:
                    row[key] = info[key]
            rows.append(row)

        write_csv(exp / "comparison.csv", rows, fields)

        # Append every experiment to one master comparison file.
        master = ROOT / "experiments" / "all_runs.csv"
        with master.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("Dataset:", args.dataset)
        print("Run ID:", run_id)
        print("Images:", len(selected))
        print("MapAnything:", metrics["mapanything"]["status"])
        print("COLMAP:", metrics["colmap"]["status"])
        print("Logs:", exp)
        print("Comparison:", exp / "comparison.csv")
        print("Master comparison:", master)
        print("=" * 60)

if __name__ == "__main__":
    main()
