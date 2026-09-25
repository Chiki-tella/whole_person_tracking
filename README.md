# Face Recognition with ArcFace ONNX and 5-Point Alignment

A local, CPU-friendly face-recognition pipeline for enrolling people from a webcam and recognizing them in real time. The pipeline is:

```text
webcam -> Haar face detection -> MediaPipe FaceMesh 5-point landmarks
		  -> 112 x 112 alignment -> ArcFace ONNX embedding -> cosine-distance match
```

The project keeps the face database on disk as NumPy and JSON files. It does not send camera frames or embeddings to a remote service.

## Requirements

- Windows, macOS, or Linux
- Python 3.10-3.13
- A working webcam
- A compatible ArcFace-style ONNX model at:

```text
models/embedder_arcface.onnx
```

The model is not included in this repository. The model must accept an aligned `112 x 112` image and return a face embedding. Do not commit model files or captured face data unless you intend to share them.

## Installation

Run these commands from the repository root. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python numpy onnxruntime mediapipe==0.10.21
```

If PowerShell blocks activation, allow scripts for your user account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The `.venv` directory is local to your machine and is excluded from git. Keep the terminal in the repository root when running the modules so relative paths such as `models/` and `data/` resolve correctly.

## Quick Start

1. Put the ONNX model at `models/embedder_arcface.onnx`.
2. Check that the webcam opens:

	```powershell
	python -m src.camera
	```

3. Enroll a person:

	```powershell
	python -m src.enroll
	```

	Enter a name, then capture at least 15 varied samples. Use `SPACE` for a capture, `a` to toggle automatic capture, `s` to save, `r` to reset only new captures, and `q` to quit.

4. Start recognition:

	```powershell
	python -m src.recognize
	```

	Recognition labels a face only when its cosine distance is at or below the current threshold. Use `q` to quit, `r` to reload the database, `+` or `-` to adjust the threshold, and `d` to toggle debug overlays.

## Development Commands

Run each module from the repository root:

| Command | Purpose | Keys |
| --- | --- | --- |
| `python -m src.camera` | Webcam smoke test | `q` quit |
| `python -m src.detect` | Haar face-detection preview | `q` quit |
| `python -m src.align` | Preview 5-point alignment | `q` quit, `s` save an aligned crop |
| `python -m src.embed` | Preview ArcFace embeddings | `q` quit, `p` print embedding stats |
| `python -m src.evaluate` | Measure enrolled genuine/impostor distances | terminal output |

The camera demos currently use camera index `1`. If the webcam is exposed at another index, change the `cv2.VideoCapture(...)` value in the relevant script.

## Generated Data

Enrollment creates or updates:

```text
data/
├── db/
│   ├── face_db.npz       # normalized embedding per enrolled name
│   └── face_db.json      # enrollment metadata
└── enroll/
	 └── <person-name>/    # aligned 112 x 112 JPG crops
```

The alignment demo saves previews under `data/debug_aligned/`. These folders contain biometric data and should be treated as private. Delete them when they are no longer needed.

## Evaluation and Thresholds

After enrolling at least two people with at least five valid aligned crops each, run:

```powershell
python -m src.evaluate
```

The evaluator reports genuine distances for same-person pairs and impostor distances for different-person pairs. It then suggests a distance threshold that keeps the measured false-accept rate at or below the configured target (`1%` by default). More varied enrollment samples produce a more useful threshold; the result is only a local estimate, not a security certification.

## Project Layout

```text
face-recognition-5pt/
├── data/                 Captured crops and generated face database
├── models/               Local ONNX model files
├── src/
│   ├── camera.py         Webcam smoke test
│   ├── detect.py         Haar detection demo
│   ├── align.py          Five-point alignment demo
│   ├── embed.py          ArcFace embedding demo
│   ├── enroll.py         Build or update the local face database
│   ├── evaluate.py       Evaluate enrolled samples and thresholds
│   ├── haar_5pt.py       Haar and landmark alignment pipeline
│   └── recognize.py      Real-time multi-face recognition
└── README.md
```

## Troubleshooting

### Camera does not open

Close other applications using the webcam. Try camera indices `0`, `1`, or `2` by changing `cv2.VideoCapture(...)` in the module you are running.

### MediaPipe cannot be imported

Confirm that `.venv` is active and reinstall the pinned version:

```powershell
python -m pip install --force-reinstall mediapipe==0.10.21
```

### The ONNX model cannot be found

Run from the repository root and confirm this file exists:

```text
models/embedder_arcface.onnx
```

### Recognition is unreliable

Enroll more varied samples with different lighting, angles, and expressions. Then run `python -m src.evaluate` and use its suggested threshold as a starting point. Avoid treating a camera-only demo as an access-control system without additional safeguards.
