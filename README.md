# CV_StreetDetector_MLOps
## Edge Computer Vision Deployment — Street CV AI Analog

I built a generic edge computer vision system — YOLOv3-based person/vehicle/street detector — specifically to get hands-on with the full deployment lifecycle of edge CV: model selection trade-offs, containerized inference as a FastAPI service, telemetry and latency benchmarking, and the architectural constraints of running multiple camera streams concurrently. 

I framed the use case around something like a transit or street-camera deployment — multiple simultaneous video feeds, resource-constrained hardware — because that's exactly the kind of scenario where the choices matter: what happens to inference latency on cold start, what confidence threshold is defensible, and where you hit a wall with software-only concurrency versus needing hardware acceleration.DeepStream/Jetson NPU integration is analyzed as a production path, not implemented in this repo


## Stack
---
| Layer | Technology |
|---|---|
| Detection model | YOLOv3 / YOLOv3-tiny (cvlib + OpenCV DNN backend) |
| Inference API | FastAPI with versioned model endpoints |
| Containerization | Docker |
| Telemetry | Structured JSONL logging (AWS IoT device shadow analog) |
| Runtime | Python 3.11 / TensorFlow 2.15 |

---

## Project Structure

```
CV_StreetDetector_MLOps/
├── server.py              # Detection functions, telemetry, benchmarks
├── main.py                # FastAPI server — /predict, /health, /metrics
├── images/                # Source images (street scenes, vehicles, persons)
├── images_with_boxes/     # Inference output with bounding boxes
├── images_uploaded/       # Images uploaded via API /predict endpoint
├── telemetry.jsonl        # Structured inference log (auto-generated)
└── README.md
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check message |
| `/predict` | POST | Upload image → returns image with bounding boxes |
| `/health` | GET | Server status, model version, uptime |
| `/metrics` | GET | p50/p95 latency, total inferences from telemetry log |
| `/docs` | GET | FastAPI interactive UI |

### Response Headers on /predict

Every prediction returns custom headers for fleet monitoring:

```
X-Model-Version: v1.0.0
X-Inference-Latency-Ms: 174.15
X-Objects-Detected: 4
```

---

## STREET Detection Classes

The system filters all 80 COCO (Common Object in Context) classes down to street targets:

```python
STREET_CLASSES = ['person', 'car', 'bus', 'truck', 'bicycle']
```

This mirrors the enforcement-relevant object classes for bus lane,
bus stop, and street safety applications.

---

## Benchmark Results

### Model Comparison (steady-state, cached weights)

| Model | Avg Latency | p50 | p95 |
|---|---|---|---|
| yolov3-tiny | 178ms | 172ms | 494ms |
| yolov3-full | 139ms | 147ms | 494ms |

**Finding 1 — Model size paradox:**
yolov3-full is NOT slower than yolov3-tiny on cached weights.
OpenCV's DNN backend is better optimized for the full architecture on x86.
TPM implication: model selection requires benchmarking on TARGET hardware —
spec-sheet assumptions do not hold.

---

### Full Benchmark Table (steady-state, cached weights)

| Model | Image | Objects | Latency |
|---|---|---|---|
| yolov3-tiny | street.jpg | 4 cars | 174ms |
| yolov3-tiny | bus1.jpg | car + bus | 188ms |
| yolov3-tiny | person.jpg | 1 person | 172ms |
| yolov3-tiny | car1.jpg | 1 car | 210ms |
| yolov3-tiny | car2.jpg | 3 cars | 145ms |
| yolov3 | street.jpg | 4 cars | 154ms |
| yolov3 | bus1.jpg | car + bus | 155ms |
| yolov3 | person.jpg | 1 person | 147ms |
| yolov3 | car1.jpg | 1 car | 127ms |
| yolov3 | car2.jpg | 3 cars | 113ms |

**Sequential 3-stream:** 540ms wall time, 147ms p50
**p50 overall:** 171ms | **p95 overall:** 494ms

---

### Cold-Start vs Cached Latency

| Condition | Latency | Ratio |
|---|---|---|
| First run (weights not cached) | ~1,073ms | 1.0x |
| Cached weights | ~77ms | 13.9x |

**Finding 2 — Cold-start penalty:**
13.9x latency penalty on first inference when weights are not pre-staged.
On a vehicle with no prior deployment, first inference is unreliable
for enforcement decisions.
TPM implication: model weights must be pre-staged during vehicle provisioning
via OTA — not pulled on first use. This is a required exit criterion
for fleet activation.

---

### Confidence Threshold Operating Point

| Threshold | Detections | Behavior |
|---|---|---|
| 0.5 | 0 | Misses real objects entirely |
| 0.3 | 1–4 | Correct operating point |
| 0.2 | 55–148 | False positive flood |

**Finding 3 — Threshold is a program decision, not a parameter:**
Default threshold (0.5) missed bus and person detections entirely.
0.2 flooded results with false positives. 0.3 is the validated operating
point for this model/hardware profile.

TPM implication: confidence threshold requires re-validation on every model
version update and every hardware change. Perception sets the technical floor;
Legal/Operations own the false-positive tolerance. Both must sign off before
fleet deployment.

---

### Multi-Stream Architecture Constraint

**Finding 4 — cvlib/YOLOv3 is not thread-safe:**
Concurrent threading produces shared model state corruption across streams.
Multiprocessing isolates correctly but incurs Python/TF spawn overhead
(~13s on Windows dev machine — not representative of Linux edge hardware).

**Production path for multi-camera vehicles:**

| Approach | Viable | Trade-off |
|---|---|---|
| Linux + multiprocessing | Yes | RAM-bound — one model copy per process |
| NVIDIA DeepStream / Jetson NPU | Yes | Hardware cost — true concurrent streams |
| Python threading (cvlib) | No | Shared state corruption |

TPM implication: multi-camera vehicle configurations require an explicit
architectural decision before hardware is locked. This is a platform decision,
not a software configuration.

---

## Sequential 3-Stream Benchmark

| Metric | Value |
|---|---|
| Wall time | 540ms |
| Avg per image | 147ms |
| p50 | 147ms |
| Total objects detected | 7 |

At 1 fps per stream, single-stream 147ms p50 has 6x headroom.
At 10 fps that headroom disappears entirely.
Frame rate requirement is a program dependency that must be defined
before hardware selection.

---

## Telemetry Pipeline

Every inference writes a structured JSON record to `telemetry.jsonl`:

```json
{
  "timestamp": "2026-06-07T20:00:42Z",
  "image": "street.jpg",
  "model": "yolov3-tiny",
  "model_version": "v1.0.0",
  "confidence_thresh": 0.3,
  "latency_ms": 174.15,
  "objects_detected": 4,
  "detections": [
    {"class": "car", "confidence": 0.773},
    {"class": "car", "confidence": 0.370},
    {"class": "car", "confidence": 0.332},
    {"class": "car", "confidence": 0.301}
  ]
}
```

This mirrors AWS IoT Core device shadow reporting — the "reported state"
pattern used in fleet OTA management. Each record represents the edge device
reporting: what it saw, when it saw it, and how long inference took.

The `/metrics` endpoint aggregates this log in real time:

```json
{
  "total_inferences": 42,
  "avg_latency_ms": 171.3,
  "p50_latency_ms": 147.0,
  "p95_latency_ms": 494.0,
  "model_version": "v1.0.0",
  "uptime_seconds": 3612.4
}
```

---

## TPM Takeaways

This project was built to answer four program questions a Principal TPM
would ask before authorizing a fleet deployment:

**1. Which model fits the edge latency budget?**
Both yolov3-tiny and yolov3-full are viable at ~150ms p50 when weights
are cached. Tiny is not automatically the right choice — benchmark on
target hardware before deciding.

**2. What does cold-start cost and how do we mitigate it?**
13.9x latency penalty on first inference. Mitigation is pre-staging weights
during vehicle provisioning via OTA — not on-demand download at first use.

**3. What confidence threshold is safe for enforcement?**
0.3 is the validated operating point. Re-validation is required on every
model version update and every hardware change. This is a cross-functional
sign-off, not a unilateral engineering decision.

**4. Can one device handle multiple camera streams?**
Requires explicit architectural decision between multiprocessing (RAM-bound)
and hardware NPU (cost-bound). Must be resolved before vehicle hardware is
locked — it cannot be solved in software after the fact.

---

## Running the Project

```bash
# Create and activate virtual environment
py -3.11 -m venv .venv_cv
.venv_cv\Scripts\Activate.ps1

# Install dependencies
pip install numpy==1.26.4 opencv-python==4.8.1.78 tensorflow==2.15.0
pip install matplotlib==3.8.4 cvlib==0.2.7 fastapi==0.104.1
pip install uvicorn==0.24.0 python-multipart==0.0.6 nest-asyncio pillow

# Run benchmarks
python server.py

# Start API server
python main.py
# → http://localhost:8000/docs
```
---

## Images Detected
<img width="640" height="425" alt="bus1" src="https://github.com/user-attachments/assets/080496f0-00c0-4e05-8869-ece86b65a5c9" />


```


## Dependencies
---
| Package | Version | Purpose |
| Python | 3.11 | Runtime |
| TensorFlow | 2.15.0 | Model backend |
| OpenCV | 4.8.1.78 | Image processing |
| cvlib | 0.2.7 | YOLOv3 wrapper |
| numpy | 1.26.4 | Array processing |
| FastAPI | 0.104.1 | Inference API |
| uvicorn | 0.24.0 | ASGI server |
| matplotlib | 3.8.4 | Visualization |

> Note: numpy must be pinned to 1.26.4 for TF 2.15 compatibility.
> opencv-python 4.8.x required — 4.13+ requires numpy>=2 which conflicts.
