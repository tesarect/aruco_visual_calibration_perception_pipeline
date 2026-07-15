# YOLO-pipeline (local labeling + training + hybrid pose extraction)

Local-only, self-contained under this directory. Separate from the
rosject/cloud venv at `~/yolo_venv` set up by
`ros2_ws/src/visual_calibration/resources/scripts/shell/install_yolo.sh`.
This directory targets THIS machine (RTX 2060 SUPER, CUDA build of torch) —
labeling and training happen here for faster iteration; only the final
trained `best.pt` is meant to travel to the rosject for production inference.

This machine also has ROS + system OpenCV 4.5.4 installed, same as the
rosject — so the same `cv_bridge`/`ultralytics` ABI-isolation rule applies:
everything in this directory runs inside `./venv`, never system Python, and
this venv is never sourced in the same shell/process as a ROS `setup.bash`.

## What's in here (3 classes)

- `aruco_marker` — the ArUco marker on the end-effector (Task 2's
  calibration target)
- `cup_holder` — the cup holder disc (Task 3, Barista top-plate)
- `hole` — the 4 mounting holes on the cup holder

All 3 are coarse-localization-only classes. YOLO never regresses a precise
pose directly — `aruco_pose.py` crops each detected region and hands off to
classical CV (`cv::aruco`/`solvePnP` for the marker) for the real geometry,
per this project's locked hybrid-detection design.

## Folder structure this pipeline expects/creates

```
YOLO-pipeline/
├── venv/                        # local venv (git-ignored, disposable)
├── dataset/
│   ├── images/{train,val}/      # *.png, copied from ../captures/
│   ├── labels/{train,val}/      # *.txt, YOLO format, from Label Studio exports
│   └── data.yaml                # class list + train/val paths
├── label_studio_exports/        # archived Label Studio .zip exports
├── runs/detect/runs/<name>/     # training run outputs (ultralytics default)
│   └── weights/best.pt          # the trained model you actually want
├── debug_output/                # aruco_pose.py's debug-sweep images + results.json
│   └── results.json
├── yolo11n.pt                   # pretrained nano checkpoint (base for transfer learning)
├── install_local.sh
├── uninstall_local.sh
├── clean.sh
├── prepare_dataset.sh
├── train.py
├── sanity_check.py
├── aruco_pose.py
├── preprocess_variants.py
├── debug_server.py
└── README.md
```

`../captures/` (one level up, `Finalproject-VisualCalibration/captures/`) is
the source of truth for real-world images (`rgb_*.png`, from
`resources/scripts/python/capture_camera.py`). Nothing in this pipeline
moves or deletes files there — only copies. Drop new captures into that
folder before running `prepare_dataset.sh`.

## Zero to ready-to-label

```bash
cd YOLO-pipeline
bash install_local.sh      # creates ./venv, installs torch (CUDA)/ultralytics/label-studio
bash prepare_dataset.sh    # copies captures/*.png into dataset/images/{train,val} (80/20 split)

source venv/bin/activate
pip install flask          # one-time, only needed for debug_server.py later
label-studio start         # launches the local web server, opens browser UI
                            # (default: http://localhost:8080)
```

Leave that terminal running the Label Studio server; use a separate terminal
for everything else.

`install_local.sh` is idempotent but NOT incremental — it `rm -rf`s and
rebuilds `venv/` from scratch every time (re-downloading torch/ultralytics).
For a one-off tool swap in an already-working venv, just
`pip install`/`pip uninstall` directly instead of re-running the whole
script.

## Labeling (Label Studio)

1. Create an account on first run (local-only, stored in the SQLite DB under
   `~/.local/share/label-studio` — see `uninstall_local.sh` for cleanup).
2. Create a new project, then import images via **Upload Files** (direct
   browser upload) — point it at `dataset/images/train`. Avoid the "Local
   Storage" / filesystem-path import option unless you've explicitly enabled
   `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED` first; otherwise images show a
   `$undefined$` load error when you click into them.
3. Under project Settings → Labeling Interface, choose **Object Detection
   with Bounding Boxes**. Add all 3 label names, one per line, exactly:
   ```
   aruco_marker
   cup_holder
   hole
   ```
4. Label every image — draw ONE bounding box per visible object, tagged
   with the right class. **Always draw axis-aligned boxes, even for a
   tilted marker** — never rotate the box to hug a tilted object. YOLO's
   label format has no rotation field; a rotated box either gets silently
   flattened on export or teaches the model an inconsistent box shape for
   that class (this cost real accuracy once — see `error-mitigation.md` #27).
   Label Studio caps a project at ~100 images — split across multiple
   projects if needed and export/merge each separately.
5. Export: project page → **Export** → format **YOLO**. Downloads a `.zip`
   containing `labels/*.txt` (UUID-prefixed filenames), `classes.txt`, and
   `notes.json`.
6. Move the `.zip` into `dataset/labels/`, unzip, strip the UUID prefix off
   each label filename (everything before the first `-`) so it matches the
   corresponding image's basename, and copy into `dataset/labels/train/`
   (or `val/`). Archive the `.zip` into `label_studio_exports/` afterward —
   don't leave it sitting in `dataset/labels/`.
7. Repeat steps 2–6 for the `val` split (`dataset/images/val`).
8. Sanity-check before training: `classes.txt`'s order must match
   `dataset/data.yaml`'s `names:` order (`0: aruco_marker, 1: cup_holder,
   2: hole`), and every label filename must have a matching image filename
   in the same split.

## Training

```bash
cd YOLO-pipeline
source venv/bin/activate
python3 -u train.py
```

`train.py` fine-tunes `yolo11n.pt` on `dataset/data.yaml` — a direct call
into the `ultralytics` library (no training loop written by hand). Key
params (edit directly in `train.py`): `epochs=100` (early-stops via
`patience=20` once val accuracy plateaus), `imgsz=640`, `batch=4` (fits an
8GB card), `device=0` (GPU), `workers=0` (avoids a dataloader fork
deadlock), `amp=False` (avoids a network hang — see below).

### Where the model file lands, and what to check

Output lands at `runs/detect/runs/<name>/weights/best.pt` — `<name>`
auto-increments (`aruco_cupholder`, `aruco_cupholder-2`, ...) on every run,
so old runs are never overwritten and nothing needs to be renamed/moved by
hand between runs. `weights/last.pt` (same folder) is the final epoch's
weights, not necessarily the best one — always use `best.pt`, not `last.pt`.

At the end of the run, ultralytics prints a per-class table:

```
Class          Precision   Recall   mAP50   mAP50-95
aruco_marker   0.988       1.0      0.995   0.901
cup_holder     0.984       1.0      0.995   0.995
hole           0.999       1.0      0.995   0.909
```

What to look for:
- **Precision** — of predicted boxes for this class, what fraction were
  actually correct (low = false positives).
- **Recall** — of real objects of this class, what fraction were found
  (low = missed detections).
- **mAP50 / mAP50-95** — the standard combined precision+recall detection
  score, at a loose (50% box overlap) vs. strict (50-95% overlap averaged)
  threshold. A big gap between mAP50 and mAP50-95 for one class (but not
  the others) means it's finding the object but the box is loose/imprecise,
  not actually missing it.
- **One class scoring far below the others despite a similar instance
  count** is a labeling-quality red flag (e.g. rotated vs. axis-aligned
  boxes, see `error-mitigation.md` #27) — check the labels before assuming
  you just need more images.

`best.pt` is *not* moved anywhere automatically — every downstream script
in this pipeline (`sanity_check.py`, `aruco_pose.py`) reads it from wherever
training left it, via a `MODEL_PATH`/`yolo_model_path` variable you update
by hand to point at the specific run you want (see below). There is
currently no "latest"/symlink convention — always double-check which
`runs/detect/runs/<name>/` you're pointing at after a fresh training run,
since an old hardcoded path will silently keep using a stale model.

Eventually (once wired into the rosject, not done yet — see `progress.md`'s
Feature Additions) only this one file needs to leave this machine: copy
`best.pt` to the rosject and point the production HTTP inference server's
model-load path at it. Nothing else in `runs/` needs to travel with it.

**First-run network gotchas (already fixed in this repo, but worth knowing
if training hangs again on a fresh machine):** `ultralytics` makes two
network calls with no visible progress and no working timeout on a flaky
connection — `check_font()` (downloads `Arial.ttf`) and, unless
`amp=False`, `check_amp()` (downloads a second, different checkpoint just to
self-test). Both can hang indefinitely instead of failing cleanly. Fixed
here by pre-caching a local font as `~/.config/Ultralytics/Arial.ttf` and
setting `amp=False`. See `error-mitigation.md` #28 for the full diagnosis.

## Visual sanity check

```bash
python3 -u sanity_check.py
```

Runs `best.pt` on every val image, saves annotated copies (boxes + class +
confidence drawn on) to `sanity_check_output/`, and prints a per-image
detection-count summary. Update `MODEL_PATH` at the top of the file to
point at whichever `runs/detect/runs/<name>/weights/best.pt` you want to
inspect — it is NOT auto-detected. Open a few images from
`sanity_check_output/` directly to visually confirm box placement/size
looks right, not just trust the printed mAP numbers.

## Hybrid pose extraction (`aruco_pose.py`)

The production-shaped pipeline: YOLO detects a coarse `aruco_marker` bbox →
crop the region (+20% margin) → `cv::aruco`'s own detector finds the
marker's precise 4 corners inside the crop → `cv::solvePnP` (real 45mm
marker size + camera intrinsics) computes the actual 6-DOF pose (position +
orientation), not just a 2D box.

Camera intrinsics default to a real D415 snapshot captured via
`ros2 topic echo /D415/color/camera_info --once --full-length` (424×240
native resolution, see `D415_CAMERA_MATRIX_424x240` in the file) —
auto-rescaled if run against a differently-sized image. This is a reference
default, not a substitute for reading `camera_info` live on the ROS side;
never hardcode intrinsics further downstream than this one documented
constant.

Two ways to run it:

```bash
python3 -u aruco_pose.py
```

- **`DEBUG_MODE = True`** (current default, top of file) — runs
  `run_debug_sweep()`: sweeps every candidate preprocessing variant from
  `preprocess_variants.py` against every image in
  `dataset/images/{train,val}`, records which ones let `cv::aruco` detect a
  marker, timing per variant/image, and writes `debug_output/results.json`
  + per-variant images for `debug_server.py` to display. Never run this
  mode on the production rosject server.
- **`DEBUG_MODE = False`** — runs the real production path: for each image,
  `detect_and_estimate()` tries `CASCADE_PIPELINE`'s preprocessing variants
  in order (fastest/most-reliable first), stopping after
  `CASCADE_MIN_CONFIRMATIONS` consecutive successful detections (default 1
  = accept the first hit). Both `CASCADE_PIPELINE` (currently
  `gamma_0.7`, `gamma_1.5`, `clahe`, `upscale_4x`, `upscale_4x+clahe`,
  `upscale_4x+sharpen`) and `CASCADE_MIN_CONFIRMATIONS` are plain
  module-level config — tune freely, no other code needs to change.
  This is the mode to set before this file is reused as the rosject
  production server's pose module.

## Debug mode / visual inspection webpage

```bash
pip install flask         # one-time
python3 -u aruco_pose.py  # with DEBUG_MODE = True — generates debug_output/
python3 debug_server.py   # serves the viewer
```

Open `http://localhost:5050`. Pick an image from the dropdown to see: the
raw YOLO crop, every preprocessing variant as an equal-size tile (green
border = `cv::aruco` detected a marker in it, red = it didn't; detected
tiles show the found corners overlaid), a results table (per-variant
detected/time/tvec, plus a "quickest detection" column ranking successful
variants 1st/2nd/3rd... by actual measured time), and an overall chart
(variant name vs. average detection time, inverted Y-axis so faster
variants plot lower). Debugging/inspection only — never runs on the
production rosject server.

## Cleaning up

- `bash clean.sh` — clears `dataset/images`, `dataset/labels`, training run
  artifacts, and debug output. Keeps the venv. Use this to restart
  labeling/dataset from scratch.
- `bash uninstall_local.sh` — removes `venv/` and all installed
  tooling/caches (torch, ultralytics, label-studio's data dir). Restores
  the machine to its pre-install state. Leaves `dataset/` and
  `label_studio_exports/` alone.

## Notes

- `install_local.sh` recreates `venv/` from scratch every run (not
  incremental) — see the "Zero to ready-to-label" section above for the
  lighter one-off-swap alternative.
- Do NOT source a ROS `setup.bash` in the same shell/process as this venv,
  and do NOT import `ultralytics` in the same Python process as
  `rclpy`/`cv_bridge` — this is the whole reason the venv is isolated at
  all. The eventual production integration point is a local HTTP server
  (image-in/pose-out) that a normal `rclpy` node calls over `localhost`,
  matching the pattern already used for the rosject's `~/yolo_venv`.

## What must be running on the ROS side to connect this to `visual_calibration`

This pipeline is a **local, offline, standalone** workflow — none of the
above (labeling, training, `aruco_pose.py`, `debug_server.py`) needs any
ROS process running at all. ROS only comes into play in two situations:

**1. Capturing new images / reading camera_info** (feeding this pipeline)

- The camera driver must be publishing — real robot: `/D415/color/image_raw`
  + `/D415/color/camera_info` over the Zenoh bridge (see the root
  `CLAUDE.md`'s Real Robot Camera Setup — `ROS_DOMAIN_ID=1`,
  `unset CYCLONEDDS_URI`, `zenoh-pointcloud/init/rosject.sh` running); sim:
  `/wrist_rgbd_depth_sensor/image_raw` + `camera_info`, published as soon as
  Gazebo is up (`starbots_ur3e.launch.xml`).
- That's it — `capture_camera.py` and a one-off
  `ros2 topic echo .../camera_info --once` are the only things this
  pipeline needs from the live system, and neither needs the UR3e arm
  itself connected (confirmed: intrinsics were captured live with only the
  camera reachable, no arm connection).

**2. Actually running the trained model in the real pipeline** (not built
yet — see `progress.md`'s Feature Additions) — once `best.pt` + the HTTP
inference server are wired in, the full picture on the ROS side will be the
same set of nodes `aruco_perception`'s classical detector already needs,
since YOLO is meant to be a drop-in swap, not a parallel system:

- The camera driver (as above)
- `robot_state_publisher` / the arm's controllers — for the known
  `base_link → rg2_gripper_aruco_link` (sim) or
  `base_link → marker` (real) chain that `calibration_broadcaster_node`
  combines with the detected pose
- `calibration_broadcaster_node` (`aruco_perception`) — subscribes to
  whichever node publishes `geometry_msgs/PoseStamped` on
  `/aruco_perception/marker_pose`; today that's classical
  `aruco_detector_node`, and a YOLO-backed replacement node would publish
  the same message on the same topic so nothing downstream changes
- The `YOLO-pipeline` HTTP inference server itself (once built) — a
  separate, non-ROS process on the rosject inside `~/yolo_venv`, called
  over `localhost` by a small `rclpy` node that does the `cv_bridge`
  conversion and re-publishes the result as the `PoseStamped` above

## How this package talks to `visual_calibration` — API structure

This directory has **zero ROS imports anywhere, by design** (the whole
point of the venv isolation — see the ABI-conflict note at the top). It
never touches `rclpy`/`cv_bridge` directly. Everything below describes the
boundary a *different*, ROS-side node is responsible for crossing.

### Today: direct Python function calls (what actually exists right now)

No server, no network call — a caller running inside this same venv
(e.g. a quick script, or `debug_server.py`) just imports and calls
`aruco_pose.py`'s functions directly.

**`detect_and_estimate(yolo_model_path, image_path, camera_matrix=None, dist_coeffs=None, conf=0.25, min_confirmations=None)`**
— the ArUco marker pipeline (YOLO → crop → cascade → `cv::aruco`/`solvePnP`).

```python
from aruco_pose import detect_and_estimate

result = detect_and_estimate(
    "runs/detect/runs/aruco_cupholder-8/weights/best.pt",
    "dataset/images/val/rgb_1783949386628.png",
)
if result is not None:
    rvec, tvec = result
```

| field | type | meaning |
|---|---|---|
| `yolo_model_path` | `str` | path to a trained `best.pt` |
| `image_path` | `str` | path to an image file on disk (not yet a raw `numpy` array — see Planned HTTP contract below) |
| `camera_matrix` | `np.ndarray` (3x3) or `None` | intrinsics; `None` falls back to the hardcoded D415 reference snapshot (see the Training section's caveat — a real integration must pass this in live from `camera_info`, never rely on the default) |
| `dist_coeffs` | `np.ndarray` (5,) or `None` | lens distortion; `None` falls back to the D415 default (all zeros — no distortion) |
| `min_confirmations` | `int` or `None` | overrides `CASCADE_MIN_CONFIRMATIONS` for this call only |
| **returns** | `(rvec, tvec)` tuple, or `None` | `rvec`: `np.ndarray` (3,), Rodrigues-format rotation vector. `tvec`: `np.ndarray` (3,), position in **meters**, in the camera's optical frame. `None` means no marker was found by any cascade variant — a real ROS-side caller should treat this as "no detection this frame," not an error. |

**`detect_centroids(yolo_model_path_or_model, image, class_id, conf=0.25)`**
— the `cup_holder`/`hole` pipeline (YOLO bbox-centroid only, no crop/classical-CV step — see the function's docstring for why that's sufficient for a straight-on circular object).

```python
from aruco_pose import detect_centroids, CUP_HOLDER_CLASS_ID, HOLE_CLASS_ID
import cv2

image = cv2.imread("dataset/images/val/rgb_1783949117515.png")
holes = detect_centroids("runs/detect/runs/aruco_cupholder-8/weights/best.pt", image, HOLE_CLASS_ID)
```

| field | type | meaning |
|---|---|---|
| `yolo_model_path_or_model` | `str` path, or an already-loaded `ultralytics.YOLO` | accepts either, so a caller processing many images/classes can load the model once and reuse it |
| `image` | `np.ndarray` (already-decoded BGR image, NOT a file path) | note this differs from `detect_and_estimate`'s `image_path` — an inconsistency worth normalizing before this becomes a real server endpoint |
| `class_id` | `int` | `CUP_HOLDER_CLASS_ID` (1) or `HOLE_CLASS_ID` (2) |
| **returns** | `list[dict]`, possibly empty | one dict per detected instance (0+, e.g. up to 4 for `hole`): `{"cx": float, "cy": float, "bbox": [x1,y1,x2,y2], "confidence": float}` — `cx`/`cy` are **2D pixel coordinates only**, not a 3D pose (no known real-world size to run `solvePnP` against, unlike the marker) |

### Planned: HTTP server contract (not built yet)

Per the locked architecture (`.claude/agents/yolopp.md`, `todo.txt` Thread
D2): a persistent local HTTP server inside `~/yolo_venv` on the rosject,
loading the model once at startup, exposing one inference endpoint. Image
bytes in, detection JSON out — no ROS types cross this boundary in either
direction. Shape not finalized, but the intended contract:

**Request** — `POST /detect` (endpoint name illustrative, not locked)
```json
{
  "image_jpeg_base64": "<base64-encoded JPEG bytes>",
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```
`camera_matrix`/`dist_coeffs` are **required fields on every call**, sourced
by the ROS-side caller from a live `camera_info` subscription — this server
must never assume/cache intrinsics itself, for the same reason
`detect_and_estimate`'s `camera_matrix=None` default is explicitly flagged
as reference-only, not for real use.

**Response** — one of two shapes depending on what was found:
```json
{
  "aruco_marker": {"rvec": [rx, ry, rz], "tvec": [tx, ty, tz]},
  "cup_holder": [{"cx": 123.4, "cy": 88.2, "confidence": 0.98}],
  "hole": [{"cx": 110.1, "cy": 75.0, "confidence": 0.99}, ...]
}
```
or, if nothing was detected for a given class, that key is either omitted
or an empty list/`null` (exact convention TBD when this is built) — never a
crash/500 for "no marker in this frame," since that's an expected, common
case, not a server error.

**The ROS-side integration point** (a small `rclpy` node, not yet written):
receives a `sensor_msgs/Image`, converts via `cv_bridge`, base64-encodes the
JPEG, reads the current `camera_info` message, POSTs both to this server
over `localhost`, and republishes the response as
`geometry_msgs/PoseStamped` on `/aruco_perception/marker_pose` (matching
`aruco_detector_node`'s existing topic/message contract exactly — see
`aruco_perception/config/aruco_detector_real.yaml`) for the `aruco_marker`
case. What topic/message type the `cup_holder`/`hole` centroids should
publish on (Task 3 territory) is not yet decided — flagged in `todo.txt`'s
Thread C as not started.

Nothing in this directory currently starts, calls, or depends on any of
those — this is a forward-looking checklist for the integration step, not
a description of what exists today.
