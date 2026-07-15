#!/bin/bash
# Lighter cleanup for pipeline ARTIFACTS only — downloaded checkpoints,
# __pycache__ dirs, and generated dataset/label cache — WITHOUT touching the
# venv itself. Use this when you want to "start the dataset over" (e.g. redo
# labeling from scratch, or clear a bad train run's outputs) without paying
# the cost of reinstalling ultralytics/torch/label-studio again.
#
# Contrast with uninstall_local.sh, which removes the venv + all installed
# tooling but leaves dataset/ alone. clean.sh is the mirror image: it clears
# dataset/generated artifacts but leaves the venv/tooling alone.
#
# By default this script only removes GENERATED/derivative content:
#   - dataset/images/{train,val}/* and dataset/labels/{train,val}/*
#     (copies made from captures/ + their YOLO-format label .txt files —
#     the originals in captures/ are never touched by anything in this
#     pipeline)
#   - any runs/ or */runs/ dir ultralytics creates from train/val/predict
#   - __pycache__ dirs anywhere under YOLO-pipeline/
#   - stray *.pt checkpoint files inside YOLO-pipeline/ (re-downloaded on
#     next install_local.sh run or first inference call)
#
# It will NOT remove:
#   - venv/            (use uninstall_local.sh for that)
#   - dataset/*.yaml    (your data.yaml config — small, hand-edited, not
#                        worth regenerating by accident)
#   - captures/ (outside YOLO-pipeline/, source images — never touched)
#
# Usage:
#   bash clean.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "This will delete generated dataset images/labels and training run"
echo "artifacts under: $SCRIPT_DIR"
echo "It will NOT touch venv/, data.yaml, or the original captures/ directory."
read -r -p "Continue? [y/N] " REPLY
case "$REPLY" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Aborted, nothing removed."; exit 0 ;;
esac

for d in "$SCRIPT_DIR/dataset/images/train" "$SCRIPT_DIR/dataset/images/val" \
         "$SCRIPT_DIR/dataset/labels/train" "$SCRIPT_DIR/dataset/labels/val"; do
    if [ -d "$d" ]; then
        echo "Clearing $d"
        find "$d" -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true
    fi
done

echo "Removing any ultralytics runs/ output dirs..."
find "$SCRIPT_DIR" -type d -name "runs" -exec rm -rf {} + 2>/dev/null || true

echo "Removing __pycache__ dirs..."
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "Removing stray *.pt checkpoints inside YOLO-pipeline/..."
find "$SCRIPT_DIR" -maxdepth 1 -iname "*.pt" -print -delete 2>/dev/null || true

echo ""
echo "Done. Dataset image/label dirs cleared, venv/ and data.yaml left intact."
echo "Re-populate dataset/images/{train,val} (e.g. by copying from"
echo "../captures/) and re-label before training again."