# Gaze Canvas

**Draw with your eyes.** Gaze Canvas is a Python desktop application that uses your webcam and iris tracking to let you paint on a digital canvas — hands-free.

## How It Works

1. **Camera Capture** — A background thread grabs webcam frames with minimal latency.
2. **Iris Tracking** — MediaPipe Face Mesh (with iris refinement) extracts normalised gaze coordinates from landmark 468 (left iris centre).
3. **Smoothing** — A rolling-window mean filter stabilises the noisy raw signal.
4. **Calibration** — You look at 9 on-screen targets and press SPACE; a Ridge regression model learns the mapping from iris position → screen coordinates.
5. **Drawing** — Your calibrated gaze becomes the brush. Dwell in one spot for 1.5 s to cycle colours automatically.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
python main.py
```

> **Requires a webcam.**  Make sure no other app is using it.

## Controls

| Key             | Action                          |
|-----------------|---------------------------------|
| **SPACE**       | Record calibration sample       |
| **Q**           | Quit                            |
| **C**           | Recalibrate                     |
| **E**           | Toggle eraser mode              |
| **DEL**         | Clear the canvas                |
| **S**           | Save canvas to `saves/` folder  |
| **[** / **]**   | Decrease / increase brush size  |

## Project Structure

```
gaze_canvas/
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── src/
    ├── __init__.py
    ├── config.py        # All constants (screen, brush, palette, etc.)
    ├── camera.py        # Threaded webcam capture
    ├── gaze_tracker.py  # MediaPipe iris tracking
    ├── smoother.py      # Moving-average filter
    ├── calibration.py   # 9-point Ridge regression calibration
    ├── canvas.py        # RGBA drawing surface + dwell logic
    └── ui.py            # HUD overlays (cursor, toolbar, PiP, etc.)
```

## Dependencies

| Package         | Purpose                        |
|-----------------|--------------------------------|
| opencv-python   | Webcam capture & image ops     |
| mediapipe       | Face mesh + iris landmarks     |
| numpy           | Array maths                    |
| scikit-learn    | Ridge regression calibration   |
| pygame          | Window, drawing, input events  |

## License

MIT
