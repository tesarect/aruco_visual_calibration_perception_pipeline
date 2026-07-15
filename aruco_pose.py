#!/usr/bin/env python3
# Hybrid pose extraction for the ArUco marker: YOLO gives a rough bbox,
# then cv::aruco does the real 6-DOF pose math on the cropped region.
#
# WHY crop at all if the YOLO bbox is already tight: cv::aruco's own
# detector still needs to find the marker's 4 exact corners itself (that's
# what solvePnP actually uses, not the bbox) — cropping just makes that
# search faster/more robust by ruling out the rest of the frame first.
#
# WHY we need intrinsics: a 2D box/corners alone can't tell us how far the
# marker is or how it's tilted in 3D. solvePnP combines 2D corner pixels +
# known 3D marker geometry (its real 45mm size) + camera intrinsics to
# solve for the actual 3D position/orientation — that's what "6-DOF pose"
# means (x,y,z position + roll,pitch,yaw orientation).
#
# Run inside YOLO-pipeline/venv only.

import cv2
import numpy as np
from ultralytics import YOLO

MARKER_LENGTH_M = 0.045  # 45mm ArUco marker, per project spec
ARUCO_MARKER_CLASS_ID = 0  # matches dataset/data.yaml: 0=aruco_marker

# Real D415 intrinsics, captured live via:
#   ros2 topic echo /D415/color/camera_info --once --full-length
# on 2026-07-15. Valid ONLY at this exact resolution (424x240) — fx/fy/cx/cy
# are in pixels and scale with resolution, so these numbers become wrong if
# color/image_raw is ever published at a different size.
#
# IMPORTANT: this is a REFERENCE snapshot, not a substitute for reading the
# topic live. The camera_info topic must still be subscribed to at runtime
# on the ROS side (ros2_ws node), not re-derived from this file, in case the
# camera/resolution/calibration ever changes. Never hardcode intrinsics
# further downstream than this one documented constant.
D415_CAMERA_MATRIX_424x240 = np.array([
    [306.80584716796875,   0.0,               214.4418487548828],
    [  0.0,               306.6424560546875,  124.9103012084961],
    [  0.0,                 0.0,                1.0],
], dtype=np.float64)
D415_DIST_COEFFS_424x240 = np.zeros(5, dtype=np.float64)  # plumb_bob, all zeros — no lens distortion
D415_NATIVE_WIDTH = 424
D415_NATIVE_HEIGHT = 240


def crop_with_margin(image, box_xyxy, margin_frac=0.2):
    """Crop image to a YOLO box, padded by margin_frac so cv::aruco's own
    corner detector has room to work even if the YOLO box is slightly tight."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin_frac, bh * margin_frac
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))
    return image[y1:y2, x1:x2], (x1, y1)  # also return crop offset, in case corners need un-cropping later


def scale_camera_matrix(camera_matrix, native_size, actual_size):
    """Intrinsics are in pixels, tied to the resolution they were captured
    at. If the image we're actually running on is a different size (e.g.
    YOLO's own inference resize, or a differently-configured camera stream),
    fx/fy/cx/cy must be scaled proportionally or the pose will be wrong."""
    nw, nh = native_size
    aw, ah = actual_size
    if (nw, nh) == (aw, ah):
        return camera_matrix
    sx, sy = aw / nw, ah / nh
    scaled = camera_matrix.copy()
    scaled[0, 0] *= sx  # fx
    scaled[1, 1] *= sy  # fy
    scaled[0, 2] *= sx  # cx
    scaled[1, 2] *= sy  # cy
    return scaled


def estimate_marker_pose(crop, camera_matrix, dist_coeffs, marker_length_m=MARKER_LENGTH_M):
    """Run cv::aruco detection + solvePnP on a cropped region.
    Returns (rvec, tvec, corners) if a marker was found, else None.
    `crop` may already be grayscale (preprocessing variants can return
    either) — cvtColor is skipped in that case."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # 4x4 dictionary, per project spec — matches physical markers in use
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None  # YOLO thought a marker was here, but cv::aruco couldn't confirm it

    # estimatePoseSingleMarkers is deprecated in newer opencv-contrib; solvePnP
    # directly is the modern equivalent and works the same way under the hood.
    half = marker_length_m / 2.0
    object_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float64)

    image_points = corners[0][0]  # first detected marker's 4 corners (pixel coords)

    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not ok:
        return None
    return rvec, tvec, corners[0][0]


# ---------------------------------------------------------------------------
# PRODUCTION CASCADE — an ordered, explicitly-named subset of
# preprocess_variants.PIPELINES, picked from empirical debug-sweep results
# (see debug_output/ + debug_server.py): fastest/most-reliable variants
# first, more expensive ones only tried if earlier ones fail. Tune this list
# and CASCADE_MIN_CONFIRMATIONS freely as more data comes in — nothing else
# in this file needs to change when you do.
#
# CASCADE_MIN_CONFIRMATIONS: how many variants in a row must agree a marker
# was detected before the cascade accepts the pose and stops. 1 = accept the
# very first successful variant (fastest, but no cross-check). Higher values
# add a cheap "does a second/third variant also agree" sanity check without
# the cost of a real corner-quality analysis — trades a bit of latency for
# confidence that it wasn't a one-off false positive.
CASCADE_PIPELINE = [
    "gamma_0.7",
    "gamma_1.5",
    "clahe",
    "upscale_4x",
    "upscale_4x+clahe",
    "upscale_4x+sharpen",
]
CASCADE_MIN_CONFIRMATIONS = 1


def _cascade_functions():
    """Resolve CASCADE_PIPELINE names to their functions from
    preprocess_variants.PIPELINES, so the cascade always uses the exact same
    preprocessing code as the debug sweep — no duplicated logic to drift."""
    from preprocess_variants import PIPELINES
    by_name = dict(PIPELINES)
    missing = [name for name in CASCADE_PIPELINE if name not in by_name]
    if missing:
        raise ValueError(f"CASCADE_PIPELINE names not found in preprocess_variants.PIPELINES: {missing}")
    return [(name, by_name[name]) for name in CASCADE_PIPELINE]


def detect_and_estimate(yolo_model_path, image_path, camera_matrix=None, dist_coeffs=None, conf=0.25,
                         min_confirmations=None):
    """Full pipeline: YOLO coarse localization -> crop -> cascade of
    preprocessing variants (see CASCADE_PIPELINE) -> cv::aruco precise pose.

    Tries each variant in CASCADE_PIPELINE order; stops as soon as
    `min_confirmations` consecutive-from-the-start variants have all
    successfully detected a marker (default: CASCADE_MIN_CONFIRMATIONS).
    Returns the pose from the variant that satisfied the last confirmation.

    camera_matrix/dist_coeffs default to the D415's real captured intrinsics
    (see D415_CAMERA_MATRIX_424x240 above), auto-scaled to match the actual
    image size being processed. On the real ROS-facing server, pass in the
    matrix read live from camera_info instead of relying on this default.
    """
    min_confirmations = min_confirmations if min_confirmations is not None else CASCADE_MIN_CONFIRMATIONS

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    h, w = image.shape[:2]
    if camera_matrix is None:
        camera_matrix = scale_camera_matrix(
            D415_CAMERA_MATRIX_424x240,
            (D415_NATIVE_WIDTH, D415_NATIVE_HEIGHT),
            (w, h),
        )
    dist_coeffs = dist_coeffs if dist_coeffs is not None else D415_DIST_COEFFS_424x240

    model = YOLO(yolo_model_path)
    found = get_marker_crop(model, image, conf=conf)
    if found is None:
        print(f"{image_path}: YOLO found no aruco_marker candidate")
        return None

    crop, _offset = found

    consecutive_successes = 0
    last_pose = None
    for name, pipeline_fn in _cascade_functions():
        variant_img = pipeline_fn(crop)
        pose = estimate_marker_pose(variant_img, camera_matrix, dist_coeffs)

        if pose is None:
            consecutive_successes = 0  # a failure resets the confirmation streak
            continue

        consecutive_successes += 1
        last_pose = pose
        if consecutive_successes >= min_confirmations:
            rvec, tvec, _corners = pose
            print(f"{image_path}: pose found via '{name}' "
                  f"({consecutive_successes} consecutive confirmation(s)) — "
                  f"tvec (x,y,z in meters) = {tvec.ravel()}")
            return rvec, tvec

    if last_pose is not None:
        # cascade ran out of variants before reaching min_confirmations in a
        # row, but at least one variant did detect something — surface it
        # rather than silently discarding a plausible pose.
        rvec, tvec, _corners = last_pose
        print(f"{image_path}: cascade exhausted without {min_confirmations} consecutive "
              f"confirmations; returning last successful pose anyway — "
              f"tvec (x,y,z in meters) = {tvec.ravel()}")
        return rvec, tvec

    print(f"{image_path}: YOLO found a candidate box, but cv::aruco could not "
          f"confirm/detect a marker in any cascade variant (false positive, or crop too tight)")
    return None


def get_marker_crop(yolo_model, image, conf=0.25, margin_frac=0.2):
    """YOLO-detect + crop only (no cv::aruco yet) — the shared first half of
    the pipeline, reused by both normal detection and the preprocessing
    debug sweep below."""
    results = yolo_model.predict(source=image, conf=conf, verbose=False)[0]
    marker_boxes = [
        box.xyxy[0].tolist()
        for box in results.boxes
        if int(box.cls[0]) == ARUCO_MARKER_CLASS_ID
    ]
    if not marker_boxes:
        return None
    crop, offset = crop_with_margin(image, marker_boxes[0], margin_frac)
    return crop, offset


# ---------------------------------------------------------------------------
# DEBUG MODE — sweeps every preprocessing variant against every image, so we
# can empirically see which preprocessing (if any) helps cv::aruco detect
# markers that fail on the raw crop, and visually inspect why on a local
# webpage. NEVER runs in production (the future rosject HTTP server) unless
# DEBUG_MODE is explicitly set True — keep it False there.
# ---------------------------------------------------------------------------
DEBUG_MODE = True  # set False before this file is used as the production server's pose module


def run_debug_sweep(yolo_model_path, image_dirs, out_dir="debug_output", conf=0.25):
    """For every image: YOLO-detect+crop once, then try every preprocessing
    variant from preprocess_variants.PIPELINES against that same crop,
    recording which ones let cv::aruco find a pose. Saves each variant's
    image + a results.json for debug_server.py to display. Also times each
    variant (preprocessing + detection) and the image as a whole, so slow
    pipelines are visible, not just successful ones.

    image_dirs: a single directory, or a list of directories (e.g. both
    dataset/images/train and dataset/images/val) — all *.png files across
    them are pooled into one sweep/results.json.
    """
    import json
    import time
    from pathlib import Path
    from preprocess_variants import PIPELINES

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    model = YOLO(yolo_model_path)
    all_results = []
    sweep_start = time.perf_counter()

    if isinstance(image_dirs, (str, Path)):
        image_dirs = [image_dirs]
    image_paths = sorted(p for d in image_dirs for p in Path(d).glob("*.png"))
    for img_path in image_paths:
        image_start = time.perf_counter()
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        camera_matrix = scale_camera_matrix(
            D415_CAMERA_MATRIX_424x240, (D415_NATIVE_WIDTH, D415_NATIVE_HEIGHT), (w, h)
        )

        entry = {"image": img_path.name, "yolo_found_box": False, "variants": [], "image_time_s": None}
        found = get_marker_crop(model, image, conf=conf)

        if found is None:
            entry["image_time_s"] = time.perf_counter() - image_start
            all_results.append(entry)
            print(f"{img_path.name}: YOLO found no aruco_marker candidate")
            continue

        entry["yolo_found_box"] = True
        crop, _offset = found

        image_out_dir = out_path / img_path.stem
        image_out_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(image_out_dir / "00_yolo_crop.png"), crop)

        for name, pipeline_fn in PIPELINES:
            variant_start = time.perf_counter()

            variant_img = pipeline_fn(crop)
            variant_filename = f"{name}.png"
            cv2.imwrite(str(image_out_dir / variant_filename), variant_img)

            pose = estimate_marker_pose(variant_img, camera_matrix, D415_DIST_COEFFS_424x240)
            variant_time_s = time.perf_counter() - variant_start

            variant_result = {
                "name": name,
                "file": variant_filename,
                "detected": pose is not None,
                "time_s": variant_time_s,
            }
            if pose is not None:
                rvec, tvec, corners = pose
                variant_result["tvec_m"] = tvec.ravel().tolist()

                # draw detected corners on a copy so it's visually obvious
                # cv::aruco succeeded on this variant, not just a number
                overlay = variant_img.copy()
                if overlay.ndim == 2:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                pts = corners.astype(int)
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
                overlay_filename = f"{name}_detected.png"
                cv2.imwrite(str(image_out_dir / overlay_filename), overlay)
                variant_result["overlay_file"] = overlay_filename

            entry["variants"].append(variant_result)
            status = "DETECTED" if pose is not None else "failed"
            print(f"{img_path.name} | {name}: {status} ({variant_time_s * 1000:.1f} ms)")

        entry["image_time_s"] = time.perf_counter() - image_start
        all_results.append(entry)

    sweep_total_s = time.perf_counter() - sweep_start

    results_file = out_path / "results.json"
    with open(results_file, "w") as f:
        json.dump({"total_sweep_time_s": sweep_total_s, "images": all_results}, f, indent=2)

    # quick summary: which pipeline variant succeeded most often, across
    # images where YOLO at least found a candidate box
    variant_success_counts = {}
    attempted = [e for e in all_results if e["yolo_found_box"]]
    for entry in attempted:
        for v in entry["variants"]:
            variant_success_counts.setdefault(v["name"], [0, 0])
            variant_success_counts[v["name"]][1] += 1
            if v["detected"]:
                variant_success_counts[v["name"]][0] += 1

    print(f"\n=== Total sweep time: {sweep_total_s:.2f}s across {len(image_paths)} images ===")
    print("=== Preprocessing variant success rate (out of images YOLO found a box for) ===")
    for name, (success, total) in variant_success_counts.items():
        print(f"  {name}: {success}/{total}")

    print(f"\nFull results saved to {results_file}")
    print(f"Run 'python3 debug_server.py' to visually inspect on localhost.")


if __name__ == "__main__":
    from pathlib import Path

    model_path = "runs/detect/runs/aruco_cupholder-8/weights/best.pt"
    all_dirs = [Path("dataset/images/train"), Path("dataset/images/val")]

    if DEBUG_MODE:
        run_debug_sweep(model_path, all_dirs)
    else:
        for d in all_dirs:
            for img_path in sorted(d.glob("*.png")):
                detect_and_estimate(model_path, str(img_path))
