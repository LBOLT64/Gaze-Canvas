# Gaze Canvas

**Draw with your eyes.** Gaze Canvas is a Python desktop application that
uses your webcam and iris tracking to let you paint on a digital canvas —
hands-free.

---

## Quick Start (Windows)

```powershell
# 1. Open a terminal and cd into the project
cd gaze_canvas

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate

# 4. Install dependencies (pinned, stable versions)
pip install -r requirements.txt

# 5. Run the app
python main.py
```

> **Requires a webcam.** Make sure no other application (Zoom, Teams, etc.)
> is using it before you start.

---

## How It Works

| Stage | What happens |
|-------|-------------|
| **1. Camera** | A background thread grabs webcam frames with minimal latency. |
| **2. Iris tracking** | MediaPipe Face Mesh (with iris refinement) extracts normalised gaze coordinates from landmark 468 (left iris centre). |
| **3. Smoothing** | A rolling-window mean filter stabilises the noisy raw signal. |
| **4. Calibration** | You look at 9 on-screen targets and press **SPACE**; a Ridge regression model learns the mapping *iris position → screen coordinates*. |
| **5. Drawing** | Your calibrated gaze becomes the brush. Dwell in one spot for 1.5 s to auto-cycle colours. |

---

## Controls

| Key | Action |
|-----|--------|
| **SPACE** | Record calibration sample |
| **Q** | Quit |
| **C** | Recalibrate |
| **E** | Toggle eraser mode |
| **DEL** | Clear the canvas |
| **S** | Save canvas to `saves/` folder |
| **[** / **]** | Decrease / increase brush size |

---

## Project Structure

```
gaze_canvas/
├── main.py              # Application entry point (GazeCanvasApp)
├── requirements.txt     # Pinned Python dependencies
├── .gitignore           # venv, __pycache__, saves, etc.
├── README.md            # This file
└── src/
    ├── __init__.py
    ├── config.py        # All constants (screen, brush, palette, logging)
    ├── camera.py        # Threaded webcam capture — thread-safe
    ├── gaze_tracker.py  # MediaPipe iris tracking (solutions + tasks API)
    ├── smoother.py      # Moving-average filter with NaN rejection
    ├── calibration.py   # 9-point Ridge regression calibration
    ├── canvas.py        # RGBA drawing surface + dwell logic
    └── ui.py            # HUD overlays (cursor, toolbar, PiP, status)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.10.0.84 | Webcam capture & image ops |
| mediapipe | 0.10.14 | Face mesh + iris landmarks |
| numpy | 1.26.4 | Array maths |
| scikit-learn | 1.5.1 | Ridge regression calibration |
| pygame | 2.6.1 | Window, drawing, input events |

---

## MediaPipe Compatibility

The gaze tracker automatically selects the right backend:

* **mediapipe ≤ 0.10.20** → uses `mp.solutions.face_mesh` (legacy, no
  model download needed).
* **mediapipe ≥ 0.10.21** → uses `mp.tasks.python.vision.FaceLandmarker`
  and auto-downloads the model file on first run.

The pinned version in `requirements.txt` (0.10.14) uses the legacy path.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | You have a newer mediapipe. Run `pip install mediapipe==0.10.14` inside your venv. |
| Red "No face" dot | Ensure your face is well-lit and centred in the webcam. |
| Cursor jumps wildly | Recalibrate (**C**). Keep your head still during calibration. |
| Webcam not found | Close other apps using the camera, then restart. |

---

## License

MIT
