# Core ML / ANE Inference Benchmark — YOLOv8n on Apple Silicon

## Motivation

Prompted by a question about video/camera firmware experience during an interview screen,
this benchmark investigates on-device ML inference optimization for performance, power, and
memory — the same tradeoff space involved in features like Cinematic Mode, Smart HDR, and
real-time video stabilization on Apple devices.

## What was built

- Exported a YOLOv8n object detection model to Core ML format (`.mlpackage`) using
  `ultralytics` + `coremltools`.
- Benchmarked inference latency (p50/p95) across three Core ML compute unit configurations:
  `CPU_ONLY`, `CPU_AND_GPU`, and `ALL` (CPU + GPU + Apple Neural Engine).
- Ran the identical benchmark script in two environments to isolate a real engineering
  question: **does compute-unit selection behave the same on virtualized vs. bare-metal
  Apple Silicon?**

## Environment 1 — Virtualized (GitHub Actions `macos-14` runner)

Confirmed `arm64`, Apple M1 (Virtual) via `uname -m` / `sysctl`.

| Compute Unit | p50 (ms) | p95 (ms) |
|---|---|---|
| CPU_ONLY | 57.11 | 78.51 |
| CPU_AND_GPU | 111.79 | 131.99 |
| ALL | 93.77 | 127.94 |

**Result: CPU_ONLY was fastest.** `CPU_AND_GPU` and `ALL` were both *slower* than CPU alone —
the opposite of what real hardware acceleration should produce.

## Environment 2 — Bare-metal (rented physical Mac mini, Apple M4)

Confirmed `arm64`, Apple M4 via `uname -m` / `sysctl`.

| Compute Unit | p50 (ms) | p95 (ms) | Speedup vs. CPU_ONLY |
|---|---|---|---|
| CPU_ONLY | 10.50 | 11.08 | 1.0x |
| CPU_AND_GPU | 4.82 | 5.03 | 2.2x |
| ALL | 2.01 | 2.21 | **5.2x** |

**Result: clean, monotonic speedup** — CPU → CPU+GPU → ALL each get faster, consistent with
genuine hardware acceleration being engaged at each step.

## Why the two environments disagree

Research into Apple's virtualization stack indicates the mechanism: Apple does not expose a
public ANE instruction set or driver interface — execution routes through Core ML or private
frameworks. Hypervisors (including the one backing GitHub's macOS runners) block the direct
device-driver interactions required to engage the ANE, and GPU passthrough in a virtualized
macOS guest is similarly limited compared to bare-metal. The CI result is best explained by
compute-unit requests silently falling back to CPU-equivalent execution, with virtualization
overhead actually making `CPU_AND_GPU`/`ALL` paths slower than plain CPU.

## Verification caveat

`coremltools.models.MLModel.get_compute_plan()` was not available in this coremltools version
(9.0) in either environment, so per-operation device placement could not be programmatically
confirmed in either run. The bare-metal result's clean, monotonic latency pattern is strong
circumstantial evidence of real heterogeneous compute engagement, but it is **inferred from
performance behavior, not directly confirmed via Xcode's Performance tab or a compute-plan
API** — that direct confirmation remains a follow-up step if revisited.

## Power draw

`powermetrics` requires elevated (sudo/root) privileges not available on this rental tier;
power measurement was not obtainable in this session. Latency was the primary metric
captured.

## Real-time sustainability (30 FPS budget: 33.3ms/frame)

All three compute-unit configurations on bare-metal M4 comfortably sustain a 30 FPS budget
(p95 well under 33.3ms in every case) — the model used (YOLOv8n) is small enough that this
particular check wasn't the limiting factor on current-generation hardware.

## Summary for discussion

Built a Core ML benchmark, ran it in a virtualized CI environment first, got a result that was
suspicious on its face (CPU-only fastest), researched why (hypervisor-level ANE/GPU
restrictions), then obtained real Apple Silicon hardware and reproduced the benchmark —
confirming a 5.2x speedup consistent with genuine ANE engagement. Hands-on experience with
Core ML's compute-unit model, MIL graph compilation, and the practical gap between virtualized
and bare-metal ML inference on Apple hardware.

<img width="1914" height="1029" alt="mac_M4" src="https://github.com/user-attachments/assets/53aa2ecd-5cb1-4f62-a23b-da4a660f8e9b" />



## Artifacts

- `benchmark_coreml.py` — benchmark script (repo: `CV_Person_Street_Detector`,
  branch `feature/coreml-ane-benchmarking`)
- `.github/workflows/ane-benchmark.yml` — CI workflow definition
- `metrics/ane_results.csv`, `metrics/device_placement_log.json` — raw output, both
  environments
