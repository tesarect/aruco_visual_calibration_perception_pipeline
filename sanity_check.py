#!/usr/bin/env python3
# Visual sanity check: run best.pt on val images, save annotated copies with
# boxes+labels drawn so we can SEE what the model actually detects.
# Run inside YOLO-pipeline/venv only.
#
# Usage:
#   source venv/bin/activate
#   python3 sanity_check.py
#   deactivate

from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/aruco_cupholder-8/weights/best.pt"
VAL_DIR = "dataset/images/val"
OUT_DIR = "sanity_check_output"

model = YOLO(MODEL_PATH)

Path(OUT_DIR).mkdir(exist_ok=True)

# predict() runs inference; save=True writes annotated images (boxes + class + confidence)
results = model.predict(
    source=VAL_DIR,
    save=True,
    project=".",          # keep output inside YOLO-pipeline/, not a nested "runs/detect/predict"
    name=OUT_DIR,
    exist_ok=True,         # overwrite previous sanity-check run instead of creating predict2, predict3...
    conf=0.25,              # only show detections the model is at least 25% confident about
)

# print a quick per-image summary so we don't have to open every file to get the gist
for r in results:
    names = r.names
    counts = {}
    for cls_id in r.boxes.cls.tolist():
        cls_name = names[int(cls_id)]
        counts[cls_name] = counts.get(cls_name, 0) + 1
    print(f"{Path(r.path).name}: {counts if counts else 'NOTHING DETECTED'}")

print(f"\nAnnotated images saved to: {OUT_DIR}/")