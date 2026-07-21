#!/bin/bash
# Copies (never moves/deletes) images from a source captures directory into
# <dataset_dir>/images/train and <dataset_dir>/images/val, in an 80/20
# split, so the originals stay intact for reruns after clean.sh.
#
# This only stages images for labeling — it does NOT create label .txt
# files. Label <dataset_dir>/images/train and <dataset_dir>/images/val with
# Label Studio after this (see README.md in this directory).
#
# Safe to re-run: skips files that already exist at the destination, so it
# won't clobber labels you've already made progress on if you add more
# captures later (delete the destination file yourself first if you want a
# fresh copy of a specific image).
#
# Usage:
#   bash prepare_dataset.sh [val_fraction] [captures_dir] [dataset_dir]
#   (val_fraction defaults to 0.2; captures_dir defaults to ../captures;
#    dataset_dir defaults to ./dataset — pass sim_captures/ and dataset_sim/
#    to stage a separate sim dataset without touching the real one)
#
# Examples:
#   bash prepare_dataset.sh                                    # real (default)
#   bash prepare_dataset.sh 0.2 sim_captures dataset_sim        # sim

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAL_FRACTION="${1:-0.2}"
CAPTURES_ARG="${2:-}"
DATASET_ARG="${3:-dataset}"

if [ -z "$CAPTURES_ARG" ]; then
    CAPTURES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/captures"
elif [ -d "$CAPTURES_ARG" ]; then
    CAPTURES_DIR="$(cd "$CAPTURES_ARG" && pwd)"
else
    CAPTURES_DIR="$(cd "$SCRIPT_DIR/$CAPTURES_ARG" && pwd)"
fi

TRAIN_DIR="$SCRIPT_DIR/$DATASET_ARG/images/train"
VAL_DIR="$SCRIPT_DIR/$DATASET_ARG/images/val"

if [ ! -d "$CAPTURES_DIR" ]; then
    echo " ❌ No captures directory found at $CAPTURES_DIR"
    exit 1
fi

mkdir -p "$TRAIN_DIR" "$VAL_DIR"

mapfile -t IMAGES < <(find "$CAPTURES_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort)
TOTAL="${#IMAGES[@]}"

if [ "$TOTAL" -eq 0 ]; then
    echo " ❌ No images found in $CAPTURES_DIR"
    exit 1
fi

# Every Nth image goes to val, spread evenly rather than all-at-the-end,
# so val isn't just "whatever was captured last" (e.g. one lighting run).
N_VAL=$(python3 -c "t=$TOTAL; f=$VAL_FRACTION; print(max(1, round(t*f)))")
STEP=$(python3 -c "print(max(1, $TOTAL // $N_VAL))")

COPIED_TRAIN=0
COPIED_VAL=0
for i in "${!IMAGES[@]}"; do
    SRC="${IMAGES[$i]}"
    BASENAME="$(basename "$SRC")"
    if [ $(( (i + 1) % STEP )) -eq 0 ] && [ "$COPIED_VAL" -lt "$N_VAL" ]; then
        DEST="$VAL_DIR/$BASENAME"
        COPIED_VAL=$((COPIED_VAL + 1))
    else
        DEST="$TRAIN_DIR/$BASENAME"
        COPIED_TRAIN=$((COPIED_TRAIN + 1))
    fi
    if [ ! -f "$DEST" ]; then
        cp "$SRC" "$DEST"
    fi
done

echo ""
echo "Done. Source images in $CAPTURES_DIR were only copied, never modified."
echo "  train: $COPIED_TRAIN images -> $TRAIN_DIR"
echo "  val:   $COPIED_VAL images -> $VAL_DIR"
echo ""
echo "Next: label them with Label Studio (see README.md), exporting YOLO-format"
echo "labels into dataset/labels/train/ and dataset/labels/val/ respectively."