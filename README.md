# Real-Time Coffee Pickup Monitor

An anonymized, real-time computer-vision pipeline for monitoring pickup
interactions and producing operational sales evidence from a private camera
stream. The public repository contains the reusable engineering layer; all
company-specific data remains local and protected.

## Project scope

The original system was designed for a retail environment where a single
event may produce many video frames and where a person can briefly disappear
from the tracker. It therefore combines detection, tracking, spatial rules,
temporal state, and evidence generation instead of treating each frame as an
independent classification.

### End-to-end processing flow

```text
Private RTSP/H.265 stream
          |
          v
GStreamer + OpenCV frame ingestion
          |
          v
Configurable ROI extraction
          |
          +--> YOLO Pose person / wrist detection
          |          |
          |          v
          |     ByteTrack identity persistence
          |
          +--> HSV color segmentation for uniform classification
          |
          +--> Cup, receipt-reader, and hand/wrist spatial rules
                     |
                     v
             Temporal event state machine
                     |
          +----------+-----------+
          v                      v
Timestamped sales logs     Evidence screenshots/video
          |
          v
Local MJPEG preview for headless monitoring
```

## Core capabilities

- **Real-time video ingestion:** Reads a private RTSP stream through a
  low-latency GStreamer H.265 pipeline and automatically retries after a
  camera connection loss.
- **Pose-aware detection:** Uses a YOLO Pose model to reason about people and
  wrist/hand positions rather than relying only on person bounding boxes.
- **Persistent identities:** Uses ByteTrack tracking IDs and short-term
  position-based re-identification to reduce duplicate counts when a tracked
  person temporarily disappears.
- **Configurable spatial reasoning:** Separates the cup pickup, receipt
  reader, person tracking, and hand/wrist monitoring regions into editable ROIs.
- **Employee/customer separation:** Applies HSV color ratios, saturation
  thresholds, and repeated observations to identify uniform-like appearance
  patterns and reduce staff-triggered false positives.
- **Transaction grouping:** Combines related detections into one logical
  transaction using inactivity and hard time limits instead of counting every
  frame as a sale.
- **Evidence generation:** Stores timestamped sales records, event logs,
  screenshots, and short incident recordings locally for later review.
- **Remote local preview:** Exposes the current annotated frame through a
  lightweight MJPEG HTTP endpoint suitable for a headless deployment.
- **GPU-ready inference:** Supports CUDA through the `OPET_DEVICE` setting and
  can be switched to CPU for development environments.

## Event and state logic

The monitor evaluates multiple signals in parallel:

1. A tracked person enters the configured monitoring region.
2. The system checks whether the person appears to be staff based on repeated
   uniform-color observations.
3. Pose keypoints and spatial ROIs determine whether a hand/wrist interacts
   with the pickup region.
4. Receipt-reader activity can be used as an additional event signal.
5. Candidate events are assigned priorities and grouped into one transaction.
6. After the configured inactivity or maximum transaction window, the system
   writes the final sales record and saves the strongest evidence frame.

This design deliberately favors auditable event evidence over a single opaque
prediction. Thresholds are environment-specific and must be calibrated for
the camera angle, lighting, uniforms, and store layout.

## Privacy-preserving evidence strategy

Real store footage, screenshots, logs, camera URLs, site identifiers, and
model weights are intentionally excluded from this repository. For a public
portfolio, use one of these evidence levels:

1. **Best option: synthetic scene** — generate an abstract top-down scene with
   geometric people, a pickup zone, and an annotated event timeline. Label it
   visibly as `SYNTHETIC / ILLUSTRATIVE`; it demonstrates the workflow without
   exposing people or company layout.
2. **Anonymized internal frame** — only with written approval, remove faces,
   badges, screens, reflections, audio, timestamps, and identifiable
   background details. Keep the original file outside Git and document the
   anonymization method.
3. **Non-visual evidence** — publish a redacted sample event log, a state
   transition diagram, and architecture diagram. This is safer than uploading
   a cartoon image that could be mistaken for model output.

Synthetic images are useful as product documentation, but they are **not**
valid accuracy benchmarks for YOLO or the production pipeline. Any public
demo must distinguish clearly between illustrative evidence and measurements
obtained from authorized real data.

This repository includes a clearly labelled example event log at
[`examples/synthetic_event_log.json`](examples/synthetic_event_log.json). It
shows the type of audit record produced by the system without containing
people, camera metadata, or company information.

Generated visual examples are available in
[`examples/synthetic/`](examples/synthetic/):

- [`synthetic_frame_001.png`](examples/synthetic/synthetic_frame_001.png) —
  customer pickup
- [`synthetic_frame_002.png`](examples/synthetic/synthetic_frame_002.png) —
  staff interaction filtered
- [`synthetic_frame_003.png`](examples/synthetic/synthetic_frame_003.png) —
  pickup with receipt-reader signal

The images can be regenerated with:

```powershell
python examples/generate_synthetic_demo.py
```

## Repository privacy boundary

The repository intentionally contains no company footage, screenshots, logs,
camera addresses, credentials, site identifiers, model weights, or production
configuration. The following local paths are ignored:

- `models/` — private/licensed model weights
- `config/settings.json` — camera-specific ROI coordinates
- `data/` — logs, recordings, sales output, and screenshots
- image and video extensions — accidental media uploads

The model file is excluded. Use a model you are licensed to use and place it
locally at `models/pose-model.pt`, or set `OPET_MODEL_PATH`.

## Local setup

1. Install GStreamer with the H.265 plugins required by your camera stream.
2. Create a virtual environment and install Python dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Copy `config/settings.example.json` to `config/settings.json` and set the
   four ROI rectangles for your own authorized camera view.
4. Set private environment variables in the local shell:

   ```powershell
   $env:OPET_RTSP_URL = "rtsp://user:password@private-camera/stream"
   $env:OPET_MODEL_PATH = "models/pose-model.pt"
   $env:OPET_CONFIG_PATH = "config/settings.json"
   $env:OPET_DEVICE = "cuda"
   $env:OPET_OUTPUT_DIR = "data"
   ```

5. Start the monitor:

   ```powershell
   python src/monitor.py
   ```

The local MJPEG preview is available at `http://localhost:5000/`. The private
camera URL is never required in source control.

## Configuration reference

| Variable | Purpose | Default |
|---|---|---|
| `OPET_RTSP_URL` | Private RTSP camera stream | Required |
| `OPET_MODEL_PATH` | Local YOLO Pose model | `models/pose-model.pt` |
| `OPET_CONFIG_PATH` | Local ROI JSON file | `config/settings.json` |
| `OPET_OUTPUT_DIR` | Local runtime output root | `data/` |
| `OPET_DEVICE` | Ultralytics device, e.g. `cuda` or `cpu` | `cuda` |
| `OPET_FPS_LIMIT` | Maximum processing rate | `5` |

## Limitations and responsible use

- Uniform-color heuristics are sensitive to lighting, camera white balance,
  occlusion, and clothing variation.
- Tracking IDs are not biometric identities and should not be used to
  identify a person outside the active scene.
- The system should not be used as the sole basis for disciplinary, legal, or
  employment decisions.
- Production thresholds require validation on authorized, representative
  footage and should be monitored for false positives and false negatives.
- Follow applicable privacy, workplace-monitoring, retention, and access
  control requirements before deployment.

## Technology stack

Python, OpenCV, Ultralytics YOLO Pose, ByteTrack, NumPy, GStreamer, RTSP,
CUDA, JSON configuration, HTTP/MJPEG streaming.
