#!/usr/bin/env python3
# Candidate preprocessing pipelines for helping cv::aruco detect small/blurry
# markers in low-res (424x240) crops. Each function takes a BGR crop and
# returns a BGR (or grayscale) image ready for cv::aruco detection.
#
# These are the standard, well-known fixes for "detector can't find edges":
#   upscale        - more pixels for the detector to work with (no new detail,
#                     just makes existing edges span more pixels)
#   clahe          - contrast enhancement, helps washed-out/shadowed markers
#   sharpen        - exaggerates edges, helps borderline-blurry images
#   gamma          - brightens/darkens midtones, helps too-dark/bright images
#   upscale+clahe  - combinations, since real captures often have more than
#   upscale+sharpen  one problem at once
#
# Run inside YOLO-pipeline/venv only.

import cv2
import numpy as np


def upscale(crop, factor=4):
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


def clahe(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe_op.apply(gray)


def sharpen(crop):
    # unsharp mask: blur the image, then subtract the blur from the original
    # to emphasize edges (amount=1.5 is a moderate, not extreme, sharpen)
    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=3)
    return cv2.addWeighted(crop, 1.5, blurred, -0.5, 0)


def gamma_correct(crop, gamma=1.5):
    # gamma > 1 brightens midtones, gamma < 1 darkens them
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(crop, table)


# Each entry: (name, function). Order matters only for display order.
# This is the FULL candidate list used by the debug sweep (aruco_pose.py's
# run_debug_sweep) to compare every variant against every image. The
# production cascade (which variants to actually try, in what order, and
# when to stop) is a separate, smaller, explicitly-ordered list — see
# CASCADE_PIPELINE in aruco_pose.py — built FROM this list by name, not a
# hardcoded duplicate, so both stay in sync with the same underlying functions.
PIPELINES = [
    ("original", lambda c: c),
    ("upscale_4x", lambda c: upscale(c, 4)),
    ("clahe", clahe),
    ("sharpen", sharpen),
    ("gamma_1.5", lambda c: gamma_correct(c, 1.5)),
    ("gamma_0.7", lambda c: gamma_correct(c, 0.7)),
    ("gamma_0.5", lambda c: gamma_correct(c, 0.5)),
    ("upscale_4x+clahe", lambda c: clahe(upscale(c, 4))),
    ("upscale_4x+sharpen", lambda c: sharpen(upscale(c, 4))),
]
