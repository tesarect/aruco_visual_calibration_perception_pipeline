#!/bin/bash
# Installs an isolated Python venv for LOCAL YOLO labeling/training work at
# YOLO-pipeline/venv, fully separate from this machine's system Python /
# dist-packages (this machine also has ROS + system OpenCV 4.5.4 installed,
# so the same cv_bridge/ultralytics ABI-isolation rule from the rosject
# applies here too — see error-mitigation.md #15). No --system-site-packages,
# so ultralytics' own bundled OpenCV can never shadow or conflict with the
# ROS-side one.
#
# This is the LOCAL counterpart to
# ros2_ws/src/visual_calibration/resources/scripts/shell/install_yolo.sh
# (which targets ~/yolo_venv on the rosject/cloud). Kept separate on purpose:
# different machine, different GPU (local has an RTX 2060), different venv
# location (self-contained inside YOLO-pipeline/ so it's trivial to nuke).
#
# GPU-enabled build: this machine has an NVIDIA RTX 2060 (8GB VRAM, driver
# 595.71.05, CUDA 13.2) confirmed via diagnostics, so this script installs
# the CUDA build of torch instead of the rosject's CPU-only build. Labeling
# doesn't need the GPU, but local training later will.
#
# Idempotent: safe to re-run any time — recreates the venv from scratch.
# Companion scripts:
#   uninstall_local.sh — full rollback (removes venv + everything this
#                         script added), restores machine to pre-install
#                         state.
#   clean.sh            — lighter cleanup of pipeline artifacts (checkpoints,
#                          __pycache__, dataset cache) WITHOUT removing the
#                          venv. Use that instead if you just want to restart
#                          the dataset, not reinstall tooling.
#
# Usage:
#   bash install_local.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo " ✚ Checking that python3-venv works on this machine before doing anything..."
TMP_VENV_CHECK="$(mktemp -d)"
if ! python3 -m venv "$TMP_VENV_CHECK/venvcheck" >/dev/null 2>&1; then
    echo " ❌ python3 -m venv failed. This likely means the python3-venv apt"
    echo "    package is missing on this machine. This script will NOT run"
    echo "    apt automatically (apt state is real system-wide state, not"
    echo "    project-local) — install it yourself first, e.g.:"
    echo "      sudo apt install python3.10-venv"
    echo "    then re-run this script."
    rm -rf "$TMP_VENV_CHECK"
    exit 1
fi
rm -rf "$TMP_VENV_CHECK"
echo "    python3-venv OK, no apt changes needed."

if [ -d "$VENV_DIR" ]; then
    echo " ❌ Removing existing venv at $VENV_DIR before reinstall..."
    rm -rf "$VENV_DIR"
fi

echo " ✚ Creating venv at $VENV_DIR (isolated, no system-site-packages)..."
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo " ⬇️ Installing CUDA-enabled torch (RTX 2060 present on this machine)..."
echo "    If this fails or you'd rather stay CPU-only, comment this line out"
echo "    and uncomment the CPU-only line below, then re-run."
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
# pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

echo " ⬇️ Installing ultralytics (YOLO) + opencv-python-headless..."
pip install --quiet ultralytics opencv-python-headless

echo " ⬇️ Installing flask (debug_server.py + inference_server.py both need it)..."
pip install --quiet flask

echo " ⬇️ Installing Label Studio (labeling tool)..."
echo "    Note: label-studio is significantly heavier than labelImg was — it"
echo "    pulls in Django, Django REST Framework, and a local SQLite-backed"
echo "    web server, so expect a noticeably longer install and a larger"
echo "    venv/ footprint (tens of MB of extra deps, not just a few)."
pip install --quiet label-studio

echo "Fetching the nano checkpoint (smallest, ~6MB) so first run doesn't stall on a cold download..."
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# ultralytics' YOLO(...) call above drops the checkpoint in the CWD by
# default; make sure it lands inside YOLO-pipeline/ (self-contained), not
# wherever the user happened to invoke this script from.
if [ -f "yolo11n.pt" ] && [ "$(pwd)" != "$SCRIPT_DIR" ]; then
    mv -f "yolo11n.pt" "$SCRIPT_DIR/yolo11n.pt"
fi

deactivate

echo ""
echo "Done. Venv installed at: $VENV_DIR"
du -sh "$VENV_DIR"
echo ""
echo "To use it:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 -c \"from ultralytics import YOLO; print('ok')\""
echo "  label-studio start   # launch the labeling web server (browser UI)"
echo "  deactivate"
echo ""
echo "Do NOT source this venv in the same shell as a sourced ROS setup.bash"
echo "when running anything that imports cv_bridge — keep YOLO/labeling work"
echo "and ROS/cv_bridge work in separate processes (see architecture notes)."