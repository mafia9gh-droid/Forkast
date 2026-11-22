
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = ROOT / "data/videos/restaurant.mp4"
OUTPUT_VIDEO = ROOT / "outputs/detected_restaurant.mp4"

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Restaurant-related classes
TARGET_CLASSES = {
    "person",
    "dining table",
    "chair",
    "cup",
    "bowl"
}

class_ids = [
    idx for idx, name in model.names.items()
    if name in TARGET_CLASSES
]

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")
print(f"Target classes: {class_ids}")

if not INPUT_VIDEO.exists():
    raise FileNotFoundError(
        f"Video not found: {INPUT_VIDEO}"
    )

cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

OUTPUT_VIDEO.parent.mkdir(
    parents=True, exist_ok=True
)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Cannot create output video")

frame_count = 0

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(
            frame,
            classes=class_ids,
            conf=0.35,
            device=device,
            verbose=False
        )

        annotated = results[0].plot()
        writer.write(annotated)

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

finally:
    cap.release()
    writer.release()

print(f"Completed: {frame_count} frames")
print(f"Saved video: {OUTPUT_VIDEO}")
