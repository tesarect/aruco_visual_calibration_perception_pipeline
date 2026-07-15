#!/usr/bin/env python3
# Fine-tunes yolo11n on our 3-class dataset (aruco_marker, cup_holder, hole).
# Run inside YOLO-pipeline/venv only — never import ultralytics in the same
# process as rclpy/cv_bridge (ABI-isolation rule, see project notes).
#
# Usage:
#   source venv/bin/activate
#   python3 train.py
#   deactivate

from ultralytics import YOLO

# Start from the pretrained nano checkpoint instead of random weights —
# this is "transfer learning": the model already knows general shapes/edges
# from COCO, so it needs far fewer of our 48 images to learn our 3 classes.
model = YOLO("yolo11n.pt")

model.train(
    data="dataset/data.yaml",  # tells YOLO where images/labels/classes live
    epochs=100,                # how many full passes over the training set
    imgsz=640,                 # images are resized to 640x640 before training
    batch=4,                   # images processed per training step (fits 8GB VRAM)
    device=0,                  # 0 = first GPU (our RTX 2060); "cpu" as fallback
    patience=20,                # stop early if val accuracy hasn't improved in 20 epochs
    workers=0,                  # 0 = load data in the main process (no worker fork/deadlock)
    amp=False,                  # skip AMP self-test (it silently hangs downloading a 2nd checkpoint on this network)
    project="runs",            # top-level folder for training outputs
    name="aruco_cupholder",    # this run's subfolder: runs/aruco_cupholder/
)
