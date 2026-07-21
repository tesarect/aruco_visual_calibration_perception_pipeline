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
├── findings/                    # benchmark_cascade.py's CSV + plot_cascade_benchmark.py's PNGs
│   ├── cascade_benchmark.csv
│   ├── time_to_first_success.png
│   ├── total_time_per_image.png
│   └── per_variant_breakdown.png
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
├── inference_server.py
├── benchmark_cascade.py
├── plot_cascade_benchmark.py
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

## GPU vs CPU cascade benchmark (`findings/`)

Why this exists: real-time video-rate detection was considered (e.g. for a
live tracking loop during arm motion) and deliberately NOT built for the
calibration use case — the cascade's worst case (a hard frame falling
through to the slow upscale variants) is an uneven latency spike, and a
continuously-moving arm risks the detection lagging behind the arm's actual
current pose by the time a hard frame resolves. This is the same class of
motion-blur/stale-sample bug already fixed once in
`calibration_broadcaster_node` (`error-mitigation.md` #20) — the existing
settle-then-sample design (wait for the arm to stop, then wait for a fresh
detection) already avoids it, so a continuous video pipeline isn't needed
for calibration itself. See `todo.txt`'s Thread D2 "FINDING for Thread A"
note for the full reasoning. What IS useful from this investigation:
concrete GPU-vs-CPU timing numbers for the cascade, e.g. for presenting
what the preprocessing cost actually looks like on this project's two real
target machines. This benchmark is presentation/analysis tooling only —
production (`inference_server.py`) never runs it.

```bash
cd YOLO-pipeline
source venv/bin/activate
python3 -u benchmark_cascade.py --label "local RTX 2060 SUPER"
```

Sweeps all 60 real images (train + val) through every variant in
`preprocess_variants.PIPELINES` (the full candidate list, broader than just
the locked-in 6-variant `CASCADE_PIPELINE` — this is a "which variant is
fastest, period" investigation, not a re-test of the production cascade
order). For every image records: time for each variant tried, in order,
cumulative time to the first successful detection, and total per-image
time. Auto-detects GPU vs CPU (`--device cuda`/`--device cpu` to force
either); results **append** (never overwrite) to
`findings/cascade_benchmark.csv`, tagged by a `device` column plus your
`--label`, so a GPU run done here and a CPU run done later on the rosject
(same script, same command, run there) both live in the same file for
direct comparison.

```bash
python3 -u plot_cascade_benchmark.py
```

Reads the CSV (however many device/label groups it currently has — 1 after
just the GPU run, 2 once the rosject's CPU run is appended) and
(re-)generates three bar-chart PNGs under `findings/`:
`time_to_first_success.png` (average time to the first successful cascade
variant per group — the presentation headline number), `total_time_per_image.png`
(average total per-image time, all variants attempted, whether or not one
succeeded), and `per_variant_breakdown.png` (average time per individual
variant, grouped by device — shows where time actually goes inside the
cascade, not just the aggregate). Safe to re-run any time after new
benchmark data is appended; always regenerates all three from the
CSV's full current contents, so running it again after the rosject's CPU
data lands will produce the actual side-by-side comparison graphs.

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

### The HTTP inference server (`inference_server.py`) — built and tested

Per the locked architecture (`.claude/agents/yolopp.md`, `todo.txt` Thread
D2): a persistent local HTTP server, loading the model **once** at startup
(not per-request — the whole reason this is a server, not
subprocess-per-call), exposing one inference endpoint. Image bytes in,
detection JSON out — no ROS types cross this boundary in either direction,
no `rclpy`/`cv_bridge` import anywhere in `inference_server.py`.

```bash
cd YOLO-pipeline
source venv/bin/activate
python3 inference_server.py   # binds 127.0.0.1:8600, loads MODEL_PATH at startup
```

**`GET /health`** — cheap liveness check, does not force a model load.
```json
{"status": "ok", "model_loaded": true}
```

**`POST /detect`**

Request:
```json
{
  "image_jpeg_base64": "<base64-encoded JPEG bytes>",
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
  "conf": 0.25
}
```
| field | type | required | meaning |
|---|---|---|---|
| `image_jpeg_base64` | `str` | yes | a JPEG-encoded frame, base64-encoded |
| `camera_matrix` | `[[3],[3],[3]]` | yes | intrinsics — **no server-side default or cache**, must come from a live `camera_info` subscription on the caller's side, every request |
| `dist_coeffs` | `[5]` | yes | lens distortion coefficients |
| `conf` | `float` | no (default `0.25`) | YOLO confidence threshold |

A missing/malformed `image_jpeg_base64`, `camera_matrix`, or `dist_coeffs`
returns HTTP 400 with `{"error": "<reason>"}` — confirmed via manual
testing (bad base64, empty body). A model load failure at request time
(e.g. a missing/corrupt weights file) returns HTTP 500. Neither case is a
process crash.

Response — confirmed via a real request against a val image:
```json
{
  "aruco_marker": {
    "rvec": [-3.1334, 0.2223, 0.0011],
    "tvec": [-0.4081, -0.2183, 0.6506],
    "corners": [[180.0, 110.0], [201.0, 107.0], [204.0, 128.0], [183.0, 131.0]]
  },
  "hole": [
    {"cx": 206.46, "cy": 148.44, "confidence": 0.96, "bbox": [190.1, 130.2, 222.8, 166.7]},
    {"cx": 206.88, "cy": 190.64, "confidence": 0.89, "bbox": [188.4, 172.9, 225.3, 208.1]}
  ]
}
```
**A key is entirely omitted (not `null`/empty-list) if that class wasn't
found in the frame** — confirmed by testing an image with no marker: the
response contained only `cup_holder`/`hole`, no `aruco_marker` key at all,
and no error. This is expected/common, not a server error — a caller
should check for key presence, not assume all three always appear.
`aruco_marker.rvec`/`tvec` are the same Rodrigues-vector/meters convention
as the direct Python API above. `aruco_marker.corners` is the marker's 4
detected corners, already converted back to FULL-FRAME pixel coordinates
(`corners_to_full_frame()` undoes both the YOLO crop offset and any
preprocessing-variant resize, e.g. the cascade's `upscale_4x` variant —
verified correct via a direct unit test simulating the upscale case, not
just the no-op case) — used by `yolo_marker_bridge_node` to draw the same
yellow-border+axes overlay classical's `aruco_detector_node` does, on
`/aruco_perception/overlay_image`, matching its `bgr8` encoding and visual
convention exactly. `cup_holder`/`hole` entries are the exact same shape
`detect_centroids()` returns (`cx`, `cy`, `confidence`, `bbox` as
`[x1, y1, x2, y2]` pixels) — `bbox` is included (not stripped) so a
consumer like depth-perception can sample a small patch within it for a
depth lookup, more robust than a single noisy pixel read. On the ROS side,
`yolo_marker_bridge_node` republishes this array as
`visual_calibration_msgs/Detection2DArray` on
`/aruco_perception/detections_2d` — see that node's own docstring.

**Still not built — the actual ROS-side integration point** (a small
`rclpy` node): receives a `sensor_msgs/Image`, converts via `cv_bridge`,
base64-encodes the JPEG, reads the current `camera_info` message, POSTs
both to this server over `localhost`, and republishes the response as
`geometry_msgs/PoseStamped` on `/aruco_perception/marker_pose` (matching
`aruco_detector_node`'s existing topic/message contract exactly — see
`aruco_perception/config/aruco_detector_real.yaml`) for the `aruco_marker`
case. What topic/message type the `cup_holder`/`hole` centroids should
publish on (Task 3 territory) is not yet decided — flagged in `todo.txt`'s
Thread C as not started. Also not yet done: deploying `inference_server.py`
+ `best.pt` to the rosject's `~/yolo_venv` and wiring a startup/stop script
for it (matching `install_yolo.sh`'s convention) — this has only been run
and tested on the local machine so far.
