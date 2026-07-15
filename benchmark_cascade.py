#!/usr/bin/env python3
# Standalone timing benchmark for the preprocessing cascade (see
# CASCADE_PIPELINE in aruco_pose.py) — measures, per image:
#   - time for EACH cascade variant tried, in order, until the first
#     successful detection (or all variants exhausted)
#   - total time to find the first successful detection ("time to find
#     fastest") for that image
#   - overall end-to-end detection time for that image (YOLO load-once is
#     excluded from per-image timing, matching production: the model loads
#     once at server startup, not per detection)
#
# Purpose: presentation evidence comparing GPU (this machine) vs CPU
# (rosject) cascade performance. Results APPEND to the same findings/
# cascade_benchmark.csv regardless of which machine/device ran them, tagged
# by a `device` column, so results from both runs live in one file for
# direct comparison — this script does NOT overwrite prior runs.
#
# Run inside YOLO-pipeline/venv only.
#
# Usage:
#   python3 benchmark_cascade.py                  # auto-detects GPU/CPU
#   python3 benchmark_cascade.py --device cpu      # force CPU even if a GPU is present
#   python3 benchmark_cascade.py --label "rosject run 1"   # custom note in the CSV
#
# After running on both machines, generate the comparison graphs with:
#   python3 plot_cascade_benchmark.py

import argparse
import csv
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from aruco_pose import (
    D415_CAMERA_MATRIX_424x240,
    D415_DIST_COEFFS_424x240,
    D415_NATIVE_HEIGHT,
    D415_NATIVE_WIDTH,
    estimate_marker_pose,
    get_marker_crop,
    scale_camera_matrix,
)
from preprocess_variants import PIPELINES

MODEL_PATH = "runs/detect/runs/aruco_cupholder-8/weights/best.pt"
IMAGE_DIRS = ["dataset/images/train", "dataset/images/val"]
FINDINGS_DIR = Path("findings")
CSV_PATH = FINDINGS_DIR / "cascade_benchmark.csv"

CSV_FIELDS = [
    "device", "label", "image", "variant", "variant_order",
    "variant_time_ms", "detected", "is_first_success",
    "time_to_first_success_ms", "image_total_time_ms",
]


def run_benchmark(device, label):
    FINDINGS_DIR.mkdir(exist_ok=True)

    print(f"Loading model on device={device} ...")
    model = YOLO(MODEL_PATH)
    model.to(device)

    image_paths = sorted(
        p for d in IMAGE_DIRS for p in Path(d).glob("*.png")
    )
    print(f"Found {len(image_paths)} images to benchmark.")

    rows = []
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        camera_matrix = scale_camera_matrix(
            D415_CAMERA_MATRIX_424x240, (D415_NATIVE_WIDTH, D415_NATIVE_HEIGHT), (w, h)
        )

        image_start = time.perf_counter()
        found = get_marker_crop(model, image)

        if found is None:
            image_total_ms = (time.perf_counter() - image_start) * 1000
            rows.append({
                "device": device, "label": label, "image": img_path.name,
                "variant": "", "variant_order": "", "variant_time_ms": "",
                "detected": False, "is_first_success": False,
                "time_to_first_success_ms": "", "image_total_time_ms": f"{image_total_ms:.3f}",
            })
            print(f"{img_path.name}: no YOLO candidate ({image_total_ms:.2f} ms)")
            continue

        crop, _offset = found

        first_success_found = False
        cumulative_ms = 0.0
        for order, (name, pipeline_fn) in enumerate(PIPELINES, start=1):
            variant_start = time.perf_counter()
            variant_img = pipeline_fn(crop)
            pose = estimate_marker_pose(variant_img, camera_matrix, D415_DIST_COEFFS_424x240)
            variant_ms = (time.perf_counter() - variant_start) * 1000
            cumulative_ms += variant_ms

            detected = pose is not None
            is_first = detected and not first_success_found
            if is_first:
                first_success_found = True

            rows.append({
                "device": device, "label": label, "image": img_path.name,
                "variant": name, "variant_order": order,
                "variant_time_ms": f"{variant_ms:.3f}",
                "detected": detected, "is_first_success": is_first,
                "time_to_first_success_ms": f"{cumulative_ms:.3f}" if is_first else "",
                "image_total_time_ms": "",  # filled on the image's last row below
            })

            print(f"{img_path.name} | {name}: {'DETECTED' if detected else 'failed'} "
                  f"({variant_ms:.2f} ms, cumulative {cumulative_ms:.2f} ms)")

        image_total_ms = (time.perf_counter() - image_start) * 1000
        # attach the image's total wall-clock time (all variants, whole image) to its last row
        for r in reversed(rows):
            if r["image"] == img_path.name:
                r["image_total_time_ms"] = f"{image_total_ms:.3f}"
                break

    # append (not overwrite) to the shared CSV
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nAppended {len(rows)} rows to {CSV_PATH}")
    print(f"Run again with a different --device/--label (e.g. on the rosject, CPU) "
          f"to add a comparison dataset to the same file.")
    print(f"Then run: python3 plot_cascade_benchmark.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                         help="Force a device; default auto-detects (cuda if available, else cpu).")
    parser.add_argument("--label", default="",
                         help="Free-text note stored per-row (e.g. 'rosject run 1') to distinguish runs on the same device.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_benchmark(device, args.label)