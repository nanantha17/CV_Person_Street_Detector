import io
import os
import uvicorn
import cv2
import numpy as np
import nest_asyncio
import subprocess
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cvlib as cv
from cvlib.object_detection import draw_bbox
from server import detect_and_draw_box, HAYDEN_CLASSES
import time
import json
# ── CONSTANTS ────────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
MODEL_VERSION = "v1.0.0"
START_TIME    = time.time()          # START_TIME needed for /health
TELEMETRY_LOG = "telemetry.jsonl"
os.makedirs("images_uploaded", exist_ok=True)

# Assign an instance of the FastAPI class to the variable "app".

app = FastAPI(title='Deploying an ML Model with FastAPI')


# List available models using Enum for convenience. This is useful when the options are pre-defined.
class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://serve/docs"


# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict")
def prediction(model: Model, confidence: float=0.3,file: UploadFile = File(...)):
    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    # 2. TRANSFORM RAW IMAGE INTO CV2 image

    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())

    # Start the stream from the beginning (position zero)
    image_stream.seek(0)

    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400,
                            detail="Could not decode image.")

    # 3. RUN OBJECT DETECTION MODEL
    start = time.perf_counter()
    # Run object detection
    bbox, label, conf = cv.detect_common_objects(image, model=model)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    # Filter to Hayden-relevant classes
    filtered = [(b, l, c) for b, l, c in zip(bbox, label, conf)
                if l in HAYDEN_CLASSES]
    bbox = [x[0] for x in filtered]
    label = [x[1] for x in filtered]
    conf = [x[2] for x in filtered]

    # Create image that includes bounding boxes and labels
    output_image = draw_bbox(image, bbox, label, conf)

    # Save it in a folder within the server
    cv2.imwrite(f'images_uploaded/{filename}', output_image)
    # 5. LOG TELEMETRY
    telemetry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                   time.gmtime()),
        "image": filename,
        "model": model,
        "model_version": MODEL_VERSION,
        "latency_ms": latency_ms,
        "objects_detected": len(label),
        "detections": [{"class": l,
                        "confidence": round(float(c), 3)}
                       for l, c in zip(label, conf)]
    }
    with open(TELEMETRY_LOG, "a") as f:
        f.write(json.dumps(telemetry) + "\n")

    # 6. STREAM THE RESPONSE BACK TO THE CLIENT

    # Open the saved image for reading in binary mode
    file_image = open(f'images_uploaded/{filename}', mode="rb")

    # Return the image as a stream specifying media type
    return StreamingResponse(file_image,
        media_type="image/jpeg",
        headers={
            "X-Model-Version": MODEL_VERSION,
            "X-Inference-Latency-Ms": str(latency_ms),
            "X-Objects-Detected": str(len(label))
        }
        )

@app.get("/health")
def health():
    return {
           "status": "healthy",
            "model_version": MODEL_VERSION,
            "uptime_seconds": round(time.time() - START_TIME,1)
    }


@app.get("/metrics")
def metrics():
    # handle missing or empty telemetry file gracefully
    if not os.path.exists(TELEMETRY_LOG):
        return {"error": "No telemetry data yet. Run /predict first."}

    # Read telemetry.jsonl and aggregate
    records = [json.loads(l) for l in
               open("telemetry.jsonl").readlines()
               if line.strip()]

    if not records:  # add this guard to prevent empty records
        return {"error": "Telemetry file is empty."}

    latencies = [r["latency_ms"] for r in records]
    n= len(latencies)
    return {
        "total_inferences": n,
        "avg_latency_ms": round(sum(latencies) / n, 2),
        "p50_latency_ms": latencies[n // 2],
        "p95_latency_ms": latencies[int(n * 0.95)],
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - START_TIME, 1)
    }

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)