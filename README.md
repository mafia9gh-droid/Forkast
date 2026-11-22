# Forkast – AI Restaurant Monitoring System

Forkast is an AI-powered restaurant monitoring system using YOLO and Computer Vision.

## Project Objectives

- Detect customers and restaurant objects.
- Monitor table states (Empty, Occupied, Awaiting Service, Awaiting Cleaning).
- Count and track customers.
- Measure service and cleaning times.
- Build a monitoring dashboard.

## Current Progress

- Python development environment configured.
- YOLO object detection pipeline implemented.
- NVIDIA GPU acceleration configured.
- Initial restaurant video processed successfully (515 frames).
- Initial detection limitations identified.

## Project Structure

- `src/` – Python source code
- `data/videos/` – Input videos
- `data/images/` – Training images
- `data/labels/` – YOLO annotations
- `models/` – Custom models
- `outputs/` – Detection results

## Installation

Create and activate a Python virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Detection

Place a valid MP4 video at `data/videos/restaurant.mp4`, then run:

```powershell
python src/detect.py
```

The processed video will be saved to `outputs/detected_restaurant.mp4`.

## Next Steps

1. Analyze missing classes.
2. Collect restaurant datasets.
3. Generate synthetic training images.
4. Fine-tune YOLO.
5. Implement table state detection.
6. Evaluate the model and build the dashboard.

## Team Collaboration

Each team member should create a separate branch for their assigned task and submit a Pull Request for review.