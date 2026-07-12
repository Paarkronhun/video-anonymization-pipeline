# 🎥 Video Anonymization Pipeline

## 📋 Project Goal
A framework for processing local video files to detect people and obscure them
(face or full body) for privacy purposes — e.g. preparing footage for GDPR-style
compliance review before wider distribution.

## ✨ Key Features (as implemented)

*   **Detection & Tracking:** Uses a **YOLO pose model** (via `ultralytics`) for
    per-frame person detection, combined with a custom **Kalman Filter** for
    box smoothing and **ByteTrack** (`model.track(..., tracker="bytetrack.yaml")`)
    for persistent track IDs across frames. *(No DeepSort is used anywhere in
    the codebase.)*
*   **Two anonymization modes:**
    *   `face` — masks a region around the detected head, using facial
      keypoints when available, with a ratio-based fallback when they aren't.
      Per the code's own docstring, this mode is a work in progress and
      doesn't perform as reliably as `body` yet.
    *   `body` — fills a convex-hull silhouette built from pose keypoints
      (falling back to the full bounding box if no keypoints are confident
      enough).
*   **Multi-scale inference:** each frame is run through the model at two
    resolutions (1280px and 640px) and the results are merged with NMS, to
    catch both close-up and distant/small people.
*   **Low-light preprocessing:** CLAHE contrast enhancement is applied to each
    frame before inference to help detect people in shadow or backlit scenes.
*   **Experimental age-based mask color:** bounding boxes are classified as
    "adult" (black mask) or "child" (white mask) based on a width/height
    aspect-ratio heuristic. This is explicitly flagged in the source as not
    working correctly yet — treat it as a placeholder, not a reliable feature.
*   **Input:** local video files only (anything OpenCV's `VideoCapture` can
    open). Live/streaming input is not implemented — there is no stream
    reader, reconnect logic, or continuous-write handling in the code.

## ⚙️ Features That Exist in the Code but Are NOT Wired to the CLI

The `VideoAnonymizer` class in `src/anonymizer.py` supports additional
capabilities, but **`main.py` does not use that class** — it calls
`anonymize_frame()` directly in its own frame loop, with these options left
at their defaults (mostly off). To use them today you'd need to call
`VideoAnonymizer` yourself instead of running `main.py`:

*   **Temporal coherence pass:** a second pass that interpolates masks across
    short gaps where a tracked person was briefly lost (e.g. behind an
    obstruction).
*   **Global blur:** an optional light Gaussian blur applied to the whole
    frame as a final safety net, after all masking.
*   **Border crop:** an optional crop of the frame edges to remove
    partially-detected faces that clip in/out at frame boundaries.
*   **Silhouette refinement via a segmentation model** is referenced in a
    docstring as a processing step, but no such model or code path currently
    exists — only `face` and `body` modes are implemented.

There is also no data-retention or source-deletion logic anywhere in the
pipeline — any compliance workflow around deleting/archiving the original
footage would need to be built and handled outside this codebase.

## 🏗️ System Architecture and Workflow

### 1. Input Stage
*   Loads a local video file via `cv2.VideoCapture`.

### 2. Core Processing Stage (per frame, in `anonymize_frame`)
1.  **Detection (YOLO pose model):** identifies people and keypoints,
    running at two scales and merging with NMS.
2.  **Tracking (Kalman Filter + ByteTrack):** maintains a persistent ID per
    person, predicting position on frames where a full detection pass is
    skipped (`frame_skip`), and smoothing box size/position across frames.
3.  **Masking:** fills either a face region or full-body convex hull with a
    solid color, based on `--mode`.
4.  *(Optional, not exposed via `main.py` today)* border crop, then global
    blur — always applied last so they never affect what the detector sees.

### 3. Output Stage
*   The masked frames are written to the output path with `cv2.VideoWriter`
    (`mp4v` codec).

## 🚀 Installation and Usage

### Prerequisites
*   Python 3.8+
*   `git`
*   A CUDA-enabled GPU is recommended but not required — the code checks
    `torch.cuda.is_available()` and falls back to CPU automatically.
*   A **pose-capable YOLO model** (e.g. `yolo11x-pose.pt`) — the detector
    reads keypoints, so a plain object-detection weight file
    (`yolo11x.pt`) will not produce keypoints and `face`/`body` masking
    quality will degrade to bounding-box fallbacks only. Download an
    appropriate `-pose` weight file from Ultralytics and place it at
    `models/yolo11x.pt` (or update `model_path` in `YoloDetector`).

### Installation Steps
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Paarkronhun/video-anonymization-pipeline
    cd video-anonymization-pipeline
    ```
2.  **Set Up Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    # venv\Scripts\activate    # On Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    See the comments in `requirements.txt` if you're installing on a
    CPU-only machine (no CUDA GPU).
4.  **Add a model file:**
    ```bash
    mkdir -p models
    # place a pose-capable YOLO weight file here, e.g. yolo11x-pose.pt
    ```

### Usage Guide

The entry point is `main.py`. It requires an input file, an output path, and
a mode (`face` or `body`):

```bash
python main.py \
    --input data/local_video_source.mp4 \
    --output data/local_anon.mp4 \
    --mode body
```

`--mode face` masks only the head region; `--mode body` masks the full
tracked silhouette. There is currently no CLI flag for streaming sources,
global blur, border crop, or temporal-coherence gap filling — see the
"Features Not Wired to the CLI" section above if you need those.

***

### 🎯 Technical Notes & Design Decisions

*   **Tracking:** Kalman Filter (per-track state prediction/smoothing) +
    ByteTrack (via `ultralytics`'s `model.track()`) for ID persistence —
    **not** DeepSort.
*   **`frame_skip`:** full detection inference doesn't run on every frame;
    on skipped frames, box positions are predicted from the Kalman Filter
    instead, trading some accuracy for speed.
*   **Resource management:** `main.py` uses `try...finally` to ensure the
    video capture and writer objects are always released.
*   **Known limitations:** `face` mode is noted in the source as less
    reliable than `body` mode; the child/adult classification heuristic is
    marked as not functioning correctly yet; there's no live/stream input
    support despite `VideoCapture` technically accepting network URLs.
