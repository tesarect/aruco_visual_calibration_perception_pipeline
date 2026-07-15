#!/usr/bin/env python3
# Production HTTP inference server — the actual integration boundary between
# this isolated YOLO venv and the ROS side. Loads the model ONCE at startup,
# then serves detection requests over localhost. No rclpy/cv_bridge import
# anywhere in this process, ever — see the ABI-isolation notes in README.md.
#
# This is the server documented in README.md's "Planned: HTTP server
# contract" section — request/response shapes here must stay in sync with
# that doc if either changes.
#
# Run inside YOLO-pipeline/venv only:
#   source venv/bin/activate
#   pip install flask   # one-time, if not already installed
#   python3 inference_server.py
#
# On the rosject this is the process ~/yolo_venv's startup script should
# launch in the background (matching install_yolo.sh's convention) — not
# built yet, see todo.txt Thread D2.

import base64

import cv2
import numpy as np
from flask import Flask, jsonify, request
from ultralytics import YOLO

from aruco_pose import (
    CUP_HOLDER_CLASS_ID,
    HOLE_CLASS_ID,
    detect_centroids,
    estimate_marker_pose_cascade,
)

MODEL_PATH = "runs/detect/runs/aruco_cupholder-8/weights/best.pt"
HOST = "127.0.0.1"
PORT = 8600

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
    # loaded yet, so this stays fast even before the first /detect call
    return jsonify({"status": "ok", "model_loaded": _model is not None})


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
        rvec, tvec = pose
        response["aruco_marker"] = {
            "rvec": rvec.ravel().tolist(),
            "tvec": tvec.ravel().tolist(),
        }

    # cup_holder / hole: bbox-centroid only, 0+ instances each.
    for key, class_id in (("cup_holder", CUP_HOLDER_CLASS_ID), ("hole", HOLE_CLASS_ID)):
        centroids = detect_centroids(model, image, class_id, conf=conf)
        if centroids:
            response[key] = [
                {"cx": c["cx"], "cy": c["cy"], "confidence": c["confidence"]}
                for c in centroids
            ]

    return jsonify(response)


if __name__ == "__main__":
    get_model()  # load at startup, not on first request — a cold first
                 # request would otherwise pay the load cost and look like
                 # a hang/timeout to the caller
    app.run(host=HOST, port=PORT, debug=False)