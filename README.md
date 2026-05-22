# AI Driver Drowsiness Detection System

An AI-based driver monitoring project that uses a webcam and MediaPipe Face Mesh to track facial landmarks in real time. The current implementation covers the Day 4 milestone: webcam capture, face detection, landmark topology visualization, Eye Aspect Ratio (EAR) drowsiness detection, Mouth Aspect Ratio (MAR) yawning detection, and no-person interlock status.

## Current Status

Completed:

- Real-time webcam capture using OpenCV
- Async camera capture thread with a small frame queue
- 480p frame resizing for stable processing
- MediaPipe 468-point Face Mesh detection
- Eye landmark mapping for future EAR calculation
- Live Eye Aspect Ratio (EAR) calculation
- Clock-based drowsiness alert when EAR remains below threshold
- Mouth landmark mapping for future MAR/yawn detection
- Live Mouth Aspect Ratio (MAR) calculation
- Clock-based yawning alert when MAR remains above threshold
- No-face/operator-absent interlock state
- Nose tip tracking for future head pose estimation
- HUD overlay with FPS, face status, and landmark boxes

Planned next:

- Day 5: Head pose estimation using OpenCV solvePnP
- Later: audio alerts, no-person interlock, database logging, and dashboard analytics

## Project Goal

The goal of this system is to detect unsafe driver states such as drowsiness, yawning, distraction, and operator absence using lightweight computer vision techniques. The project is designed as an edge-friendly pipeline that can later be extended with telemetry logging, analytics, and IoT safety actions.

## Tech Stack

- Python
- OpenCV
- MediaPipe Face Mesh
- NumPy
- Threading and Queue for async frame capture

## Repository Files

```text
face_topology.py    # Main Day 4 file: EAR + MAR + no-person interlock
DAY4.py             # Separate Day 4 backup/alternate file
README.md           # Project overview and setup guide
```

## Setup

This project uses the legacy MediaPipe `mp.solutions.face_mesh` API. Use Python 3.12 for the most reliable setup.

Create and activate a virtual environment:

```cmd
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```cmd
python -m pip install --upgrade pip
pip install mediapipe==0.10.21 opencv-python numpy
```

Verify MediaPipe:

```cmd
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

Expected output:

```text
0.10.21
True
```

## Run

```cmd
python face_topology.py
```

Controls:

- `Q`: Quit
- `M`: Toggle full face mesh overlay

## Current Output

The program opens a webcam window and displays:

- Face detection status
- FPS counter
- Eye landmark boxes
- Brighter light-green eye landmark dots for better visibility
- Live EAR value
- Eye state: open, closed, or drowsy alert
- Mouth landmark box
- Live MAR value
- Mouth state: normal, mouth open, or yawning detected
- No-face interlock state
- Nose tip point
- Day 4 status HUD

If the face is detected, landmarks are visible, and the EAR/MAR values update live, the Day 4 milestone is working correctly.

## Detection Roadmap

| Stage | Module | Purpose |
| --- | --- | --- |
| Day 1-2 | Face topology | Detect face and visualize landmarks |
| Day 3 | EAR | Detect closed eyes and drowsiness |
| Day 4 | MAR | Detect yawning and operator absence |
| Day 5 | Head pose | Detect nodding and distraction |
| Later | State machine | Combine signals into driver state IDs |
| Later | Database | Store telemetry logs for analytics |

## Notes

- Keep the virtual environment active while running the project.
- If `mp.solutions` gives an error, check that you are using the correct virtual environment and Python 3.12.
- Low lighting can reduce landmark accuracy. A front-facing light improves detection.
