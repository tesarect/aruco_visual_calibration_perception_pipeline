#!/usr/bin/env python3
# Production HTTP inference server — the actual integration boundary between
# this isolated YOLO venv and the ROS side. Loads the model ONCE at startup,
# then serves detection requests over localhost. No rclpy/cv_bridge import
# anywhere in this process, ever — see the ABI-isolation notes in README.md.
#
# This is the server documented in README.md's "How this package talks to
# visual_calibration" section — request/response shapes here must stay in
# sync with that doc if either changes.
#
# Model/host/port come from config/server_<env>.yaml (sim vs real — same
# split convention as aruco_perception's aruco_detector_sim.yaml/_real.yaml),
# NOT hardcoded here, so switching environments never needs a code change.
# --env also sets INFERENCE_ENV so aruco_pose.py's cascade config
# (config/cascade_<env>.yaml) matches automatically — see aruco_pose.py.
#
# Run inside YOLO-pipeline/venv only:
#   source venv/bin/activate
#   pip install flask   # one-time, if not already installed
#   python3 inference_server.py --env sim    # or --env real (default)
#
# On the rosject this is the process ~/yolo_venv's startup script should
# launch in the background (matching install_yolo.sh's convention) — see
# start_inference_server.sh.

import argparse
import base64
import os

# --env must be parsed and INFERENCE_ENV set BEFORE importing aruco_pose,
# since aruco_pose reads INFERENCE_ENV at import time to pick its cascade
# config. Only relevant when this file is run directly (python3
# inference_server.py) — if ever imported as a module instead, set
# INFERENCE_ENV in the environment before importing this file.
if __name__ == "__main__":
    _early_parser = argparse.ArgumentParser(add_help=False)
    _early_parser.add_argument("--env", choices=["sim", "real"], default=None)
    _early_args, _ = _early_parser.parse_known_args()
    if _early_args.env:
        os.environ["INFERENCE_ENV"] = _early_args.env

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, request
from ultralytics import YOLO

from aruco_pose import (
    CONFIG_DIR,
    CUP_HOLDER_CLASS_ID,
    HOLE_CLASS_ID,
    INFERENCE_ENV,
    detect_centroids,
    estimate_marker_pose_cascade,
)


def load_server_config(env):
    config_path = CONFIG_DIR / f"server_{env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No server config found at {config_path} (env='{env}'). "
            f"Expected config/server_sim.yaml or config/server_real.yaml."
        )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for required in ("model_path", "host", "port"):
        if required not in config:
            raise ValueError(f"{config_path} is missing required field '{required}'")
    return config


_server_config = load_server_config(INFERENCE_ENV)
MODEL_PATH = _server_config["model_path"]
HOST = _server_config["host"]
PORT = _server_config["port"]

app = Flask(__name__)

# Loaded once at import/startup time, reused across every request — the
# whole reason this is a persistent server and not a subprocess-per-call
# (see the locked process-boundary decision in .claude/agents/yolopp.md).
_model = None


def get_model():
    global _model
    if _model is None:
        print(f"Loading model from {MODEL_PATH} ...")
        _model = YOLO(MODEL_PATH)
        print("Model loaded.")
    return _model


def _decode_image(image_jpeg_base64):
    """Base64 JPEG string -> decoded BGR numpy array, or None if the bytes
    don't decode to a valid image (bad/corrupt payload, not a crash)."""
    try:
        jpeg_bytes = base64.b64decode(image_jpeg_base64)
    except (ValueError, TypeError):
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _camera_matrix_from_request(data):
    """camera_matrix/dist_coeffs are REQUIRED fields — this server must
    never assume or cache intrinsics itself (see README's "never hardcode
    intrinsics downstream" rule). Returns (camera_matrix, dist_coeffs, error)
    — error is a string if the request is missing/malformed, else None."""
    camera_matrix = data.get("camera_matrix")
    dist_coeffs = data.get("dist_coeffs")
    if camera_matrix is None or dist_coeffs is None:
        return None, None, "camera_matrix and dist_coeffs are required fields"
    try:
        camera_matrix = np.array(camera_matrix, dtype=np.float64)
        dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
    except (ValueError, TypeError):
        return None, None, "camera_matrix/dist_coeffs could not be parsed as numeric arrays"
    if camera_matrix.shape != (3, 3):
        return None, None, f"camera_matrix must be 3x3, got shape {camera_matrix.shape}"
    return camera_matrix, dist_coeffs, None


@app.route("/health", methods=["GET"])
def health():
    # cheap liveness/readiness check — does NOT load the model if it isn't
    # loaded yet, so this stays fast even before the first /detect call.
    # Includes env/model_path so a poller (e.g. start_inference_server.sh,
    # or a tmux pane's wait_for_inference_server.sh) can confirm not just
    # "a server is up" but "the RIGHT env's server is up".
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "env": INFERENCE_ENV,
        "model_path": MODEL_PATH,
    })


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True)
    if data is None or "image_jpeg_base64" not in data:
        return jsonify({"error": "request body must be JSON with an 'image_jpeg_base64' field"}), 400

    image = _decode_image(data["image_jpeg_base64"])
    if image is None:
        return jsonify({"error": "image_jpeg_base64 could not be decoded as a valid JPEG"}), 400

    camera_matrix, dist_coeffs, error = _camera_matrix_from_request(data)
    if error is not None:
        return jsonify({"error": error}), 400

    conf = data.get("conf", 0.25)

    try:
        model = get_model()
    except Exception as e:  # model file missing/corrupt at startup time
        return jsonify({"error": f"model failed to load: {e}"}), 500

    response = {}

    # ArUco marker: coarse YOLO bbox -> crop -> cascade -> cv::aruco/solvePnP.
    # Absent from the response entirely if not found this frame — that's an
    # expected, common case, not a server error.
    pose = estimate_marker_pose_cascade(
        model, image, camera_matrix, dist_coeffs, conf=conf, log_prefix="[/detect] "
    )
    if pose is not None:
        rvec, tvec, corners = pose
        response["aruco_marker"] = {
            "rvec": rvec.ravel().tolist(),
            "tvec": tvec.ravel().tolist(),
            # 4 corner points (full-frame pixel coords, NOT crop-relative)
            # for drawing the yellow border + axes overlay on the ROS side
            # — same visual convention as aruco_detector_node's classical
            # overlay_image. [[x,y], [x,y], [x,y], [x,y]].
            "corners": corners.tolist(),
        }

    # cup_holder / hole: bbox-centroid + bbox, 0+ instances each. bbox is
    # included (not just cx/cy) so depth-perception can sample a small
    # patch within it for a more robust depth read than a single pixel —
    # see visual_calibration_msgs/Detection2D.msg's rationale.
    for key, class_id in (("cup_holder", CUP_HOLDER_CLASS_ID), ("hole", HOLE_CLASS_ID)):
        centroids = detect_centroids(model, image, class_id, conf=conf)
        if centroids:
            response[key] = [
                {"cx": c["cx"], "cy": c["cy"], "confidence": c["confidence"], "bbox": c["bbox"]}
                for c in centroids
            ]

    return jsonify(response)


if __name__ == "__main__":
    print(f"Starting inference_server.py (env='{INFERENCE_ENV}', model='{MODEL_PATH}', "
          f"host='{HOST}', port={PORT})")
    get_model()  # load at startup, not on first request — a cold first
                 # request would otherwise pay the load cost and look like
                 # a hang/timeout to the caller
    app.run(host=HOST, port=PORT, debug=False)