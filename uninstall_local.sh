#!/bin/bash
# Full rollback for install_local.sh: removes the local YOLO-pipeline venv
# and every cache/checkpoint it created, restoring this machine to its
# pre-install state. Does NOT touch system Python, ROS, cv_bridge, or
# anything outside YOLO-pipeline/ + the known ultralytics/torch cache dirs
# listed below.
#
# This does NOT remove the dataset/ directory (your labeled images/labels) —
# that's real work product, not installed tooling. If you also want to wipe
# dataset artifacts, run clean.sh separately (it's opt-in and documents what
# it deletes). If you genuinely want to delete the dataset too, do it
# manually — this script deliberately won't do that for you.
#
# Usage:
#   bash uninstall_local.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo "Removing venv: $VENV_DIR"
    rm -rf "$VENV_DIR"
else
    echo "No venv found at $VENV_DIR (already removed or never installed)."
fi

# ultralytics caches downloaded weights/config under ~/.config/Ultralytics
# and torch caches under ~/.cache/torch — both outside the venv, so clean
# them separately for a true full rollback.
CONFIG_DIR="$HOME/.config/Ultralytics"
TORCH_CACHE_DIR="$HOME/.cache/torch"

if [ -d "$CONFIG_DIR" ]; then
    echo "Removing $CONFIG_DIR"
    rm -rf "$CONFIG_DIR"
fi

if [ -d "$TORCH_CACHE_DIR" ]; then
    echo "Removing $TORCH_CACHE_DIR"
    rm -rf "$TORCH_CACHE_DIR"
fi

# Label Studio, by default (no LABEL_STUDIO_BASE_DATA_DIR override), stores
# its SQLite DB, media uploads, and exported annotations under
# ~/.local/share/label-studio — OUTSIDE the venv. This is real label/project
# data, not just a cache, so it's removed here for a true rollback, but if
# you want to keep your labeled projects, back this directory up first.
LABEL_STUDIO_DATA_DIR="$HOME/.local/share/label-studio"

if [ -d "$LABEL_STUDIO_DATA_DIR" ]; then
    echo "Removing $LABEL_STUDIO_DATA_DIR (Label Studio project DB + media —"
    echo "back this up first if you want to keep your labeled projects)"
    rm -rf "$LABEL_STUDIO_DATA_DIR"
fi

# Any checkpoint file this pipeline downloaded, inside YOLO-pipeline/ itself.
if [ -f "$SCRIPT_DIR/yolo11n.pt" ]; then
    echo "Removing $SCRIPT_DIR/yolo11n.pt"
    rm -f "$SCRIPT_DIR/yolo11n.pt"
fi

echo ""
echo "Local YOLO-pipeline venv and caches removed. Machine restored to"
echo "pre-install state (dataset/ under YOLO-pipeline/ was left untouched —"
echo "use clean.sh if you want to reset dataset artifacts too)."