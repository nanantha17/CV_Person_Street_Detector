# server.py — CLEAN VERSION
# Everything below the functions must be removed or wrapped

import cv2
import os
import time
import json
from datetime import datetime
from IPython.display import Image, display

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cvlib as cv
from cvlib.object_detection import draw_bbox

# ── CONFIG ───────────────────────────────────────────────────
MODEL_VERSION  = "v1.0.0"
HAYDEN_CLASSES = ['person', 'car', 'bus', 'truck', 'bicycle']
TELEMETRY_LOG  = "telemetry.jsonl"

for d in ["images_with_boxes", "images_uploaded"]:
    os.makedirs(d, exist_ok=True)


def detect_and_draw_box(filename, model="yolov3-tiny",
                        confidence=0.5, filter_hayden=True,
                        images_dir="images"):
    img_filepath = f'{images_dir}/{filename}'
    img = cv2.imread(img_filepath)

    if img is None:
        print(f"[ERROR] Could not read image: {img_filepath}")
        return None

    start = time.perf_counter()
    bbox, label, conf = cv.detect_common_objects(
        img, confidence=confidence, model=model
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    if filter_hayden:
        filtered = [(b, l, c) for b, l, c in zip(bbox, label, conf)
                    if l in HAYDEN_CLASSES]
        bbox  = [x[0] for x in filtered]
        label = [x[1] for x in filtered]
        conf  = [x[2] for x in filtered]

    print(f"\n{'='*50}")
    print(f"Image    : {filename}")
    print(f"Model    : {model} ({MODEL_VERSION})")
    print(f"Latency  : {latency_ms} ms")
    print(f"Detected : {len(label)} object(s)")
    for l, c in zip(label, conf):
        marker = "✓" if l in HAYDEN_CLASSES else "–"
        print(f"  {marker} {l:<12} confidence: {c:.3f}")

    output_image = draw_bbox(img, bbox, label, conf)
    out_path = f"images_with_boxes/{filename}"
    cv2.imwrite(out_path, output_image)
    display(Image(out_path))

    telemetry = {
        "timestamp"        : datetime.utcnow().isoformat(),
        "image"            : filename,
        "model"            : model,
        "model_version"    : MODEL_VERSION,
        "confidence_thresh": confidence,
        "latency_ms"       : latency_ms,
        "objects_detected" : len(label),
        "detections"       : [
            {"class": l, "confidence": round(float(c), 3)}
            for l, c in zip(label, conf)
        ]
    }
    with open(TELEMETRY_LOG, "a") as f:
        f.write(json.dumps(telemetry) + "\n")

    return telemetry


def print_session_metrics():
    if not os.path.exists(TELEMETRY_LOG):
        print("No telemetry data yet.")
        return

    records   = [json.loads(l) for l in
                 open(TELEMETRY_LOG).readlines()]
    latencies = sorted([r["latency_ms"] for r in records])
    n         = len(latencies)

    if n == 0:
        print("No records found.")
        return

    print(f"\n{'='*50}")
    print(f"SESSION METRICS  ({n} inferences)")
    print(f"{'='*50}")
    print(f"  Avg latency : {round(sum(latencies)/n, 2)} ms")
    print(f"  p50 latency : {latencies[n//2]} ms")
    print(f"  p95 latency : {latencies[int(n*0.95)]} ms")
    print(f"  Total objects detected : "
          f"{sum(r['objects_detected'] for r in records)}")
    print(f"  Model version          : {MODEL_VERSION}")
    print(f"  Telemetry log          : {TELEMETRY_LOG}")


def process_stream(image_path, model, confidence, result_list):
    """Isolated process — own model instance, no shared state."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    import cv2
    import cvlib as cv
    import time

    start = time.perf_counter()
    img   = cv2.imread(image_path)

    if img is None:
        return

    bbox, label, conf = cv.detect_common_objects(
        img, confidence=confidence, model=model
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    HAYDEN_CLASSES = ['person', 'car', 'bus', 'truck', 'bicycle']
    filtered = [(l, c) for l, c in zip(label, conf)
                if l in HAYDEN_CLASSES]

    result = {
        "image"            : os.path.basename(image_path),
        "model"            : model,
        "latency_ms"       : latency_ms,
        "objects_detected" : len(filtered),
        "detections"       : [{"class": l,
                                "confidence": round(float(c), 3)}
                               for l, c in filtered]
    }
    result_list.append(result)


def run_multiprocess_benchmark(images, model='yolov3-tiny',
                               confidence=0.3):
    import multiprocessing
    print(f"\nMULTIPROCESS BENCHMARK ({len(images)} parallel streams)")
    print("="*55)

    manager     = multiprocessing.Manager()
    result_list = manager.list()
    processes   = []
    wall_start  = time.perf_counter()

    for img in images:
        p = multiprocessing.Process(
            target=process_stream,
            args=(f"images/{img}", model, confidence, result_list)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    wall_time = round((time.perf_counter() - wall_start) * 1000, 2)
    results   = list(result_list)

    if not results:
        print("No results collected.")
        return

    print(f"\n{'IMAGE':<15} {'OBJECTS':<10} {'LATENCY_MS'}")
    print("-"*40)
    for r in sorted(results, key=lambda x: x['image']):
        print(f"{r['image']:<15} {r['objects_detected']:<10} "
              f"{r['latency_ms']}")

    latencies = [r['latency_ms'] for r in results]
    seq_equiv = sum(latencies)
    print(f"\nParallel wall time : {wall_time}ms")
    print(f"Sequential equiv   : {seq_equiv:.1f}ms")
    print(f"Parallelism gain   : {seq_equiv/wall_time:.1f}x")
    print(f"Total objects      : "
          f"{sum(r['objects_detected'] for r in results)}")
    return results


# ── ONLY RUNS WHEN EXECUTING server.py DIRECTLY ──────────────
if __name__ == "__main__":

    # Clear telemetry once here — safe, won't run on import
    if os.path.exists("telemetry.jsonl"):
        os.remove("telemetry.jsonl")
        print("Telemetry log cleared — clean benchmark starting")

    # Benchmark runs
    for image_file in ['street.jpg', 'bus1.jpg', 'person.jpg']:
        detect_and_draw_box(image_file, confidence=0.3)

    results = []
    for model in ['yolov3-tiny', 'yolov3']:
        for image in ['street.jpg', 'bus1.jpg', 'person.jpg',
                      'car1.jpg', 'car2.jpg']:
            t = detect_and_draw_box(image, model=model,
                                    confidence=0.3)
            if t:
                results.append(t)

    print(f"\n{'='*65}")
    print(f"{'MODEL':<15} {'IMAGE':<15} {'OBJECTS':<10} "
          f"{'LATENCY_MS':<12}")
    print(f"{'='*65}")
    for r in results:
        print(f"{r['model']:<15} {r['image']:<15} "
              f"{r['objects_detected']:<10} {r['latency_ms']:<12}")

    print_session_metrics()

    # Sequential benchmark
    print("\nSEQUENTIAL BENCHMARK")
    print("="*50)
    seq_start   = time.perf_counter()
    seq_results = []
    for img in ['street.jpg', 'bus1.jpg', 'person.jpg']:
        t = detect_and_draw_box(img, model='yolov3-tiny',
                                confidence=0.3)
        if t:
            seq_results.append(t)
    seq_total    = (time.perf_counter() - seq_start) * 1000
    seq_lat      = [r['latency_ms'] for r in seq_results]
    print(f"\nSequential (realistic edge model):")
    print(f"  Total wall time  : {seq_total:.1f}ms")
    print(f"  Avg per image    : "
          f"{sum(seq_lat)/len(seq_lat):.1f}ms")
    print(f"  p50              : "
          f"{sorted(seq_lat)[len(seq_lat)//2]:.1f}ms")
    print(f"  Total objects    : "
          f"{sum(r['objects_detected'] for r in seq_results)}")

    #run_multiprocess_benchmark(
        #['street.jpg', 'bus1.jpg', 'person.jpg'])