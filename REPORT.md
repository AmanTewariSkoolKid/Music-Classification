# Music Genre Classification — Project Report

- Students: Aman Tewari
- Emails: Aman.125832@stu.upes.ac.in
- Course:  Btech CSE AML B11
- Instructor:Rakesh Ranjan
- Semester/Term: 5th
- Date: 18-11-25

## Abstract

This project implements a music genre classification system using deep learning on log-mel spectrograms. The pipeline covers robust audio preprocessing, a CNN-based classifier trained with cross-entropy loss, and a Material Design-inspired desktop GUI for end users. A FastAPI service exposes prediction and metadata endpoints, and a Windows batch script orchestrates clean startup/shutdown of the API and GUI. The system supports NVIDIA GPU acceleration (preferring discrete RTX devices) and handles corrupted or missing audio gracefully. Model checkpoints include validation metrics and are surfaced directly in the GUI.

## System Overview

- Data: FMA-Medium (Free Music Archive)
- Preprocessing: Resample to target rate, mono conversion, log-mel spectrograms, optional SpecAugment.
- Model: CNN-based classifier over log-mel inputs.
- Training: Adam optimizer, ReduceLROnPlateau scheduler, best-checkpoint saving with metrics.
- Inference: Torch model wrapped by a predictor with consistent preprocessing.
- UX: CustomTkinter GUI (Material 3 vibe), displays model metrics and predictions.
- API: FastAPI + Uvicorn service exposing health, metrics, genres, and prediction endpoints.
- Orchestration: `run_app.bat` launches API then GUI, and stops both on exit. Uses global Python environment.

## Dataset

- Source: FMA (Free Music Archive) — Medium subset recommended for balance between size and diversity.
- Classes: Multiple music genres; class names in `src/config.py` (`GENRE_LABELS`).
- Splits: Stratified train/validation split (seeded for reproducibility).
- Storage: Audio files organized under a dataset directory; spectrograms computed on the fly during training/inference.
- Notes: Ensure licensing compliance when distributing data; FMA is research-friendly.

## Data Preprocessing

- Loading: `librosa.load` with target sample rate; exceptions caught and handled.
- Corrupted/Unsupported Audio: Replaced with a short silent waveform to keep batches valid.
- Features: Log-mel spectrograms (configurable n_fft, hop_length, n_mels).
- Normalization: Per-sample normalization of spectrograms.
- Augmentation: Optional SpecAugment (time/frequency masking) on training only.
- Dependencies: `librosa`, `soundfile`, and system `ffmpeg` installed for MP3/varied codecs.

## Model Architecture

- Backbone: Convolutional neural network tailored to spectrogram inputs (2D conv, pooling, dropout).
- Head: Fully connected layers projecting to `num_classes = len(GENRE_LABELS)` with softmax in evaluation.
- Loss: CrossEntropyLoss.
- Optimizer: Adam with standard betas; weight decay optional.
- Scheduler: ReduceLROnPlateau on validation loss.

## Training Configuration

- Device: Auto-detect CUDA; prefer discrete RTX device if present; fallback to CPU.
- Epochs: 10
- Batch Size: 32
- Learning Rate: 0.01
- Early Stopping-Like Behavior: Track best validation loss; save `models/best_model.pt` with metrics.
- Checkpoint Contents: `{ epoch, model_state_dict, optimizer_state_dict, val_acc, val_loss }`.
- Curves: Training/validation loss and accuracy plotted and saved during/after training.

## Results

- Best Validation Accuracy: 72.71 %
- Best Validation Loss: 0.9128
- Best Epoch: 8
- Confusion Matrix: [Insert figure]
- Training Curves: [Insert loss/accuracy plots]
- Observations: Summarize which genres are frequently confused; note class imbalance effects if any.

## Application (GUI)

- Framework: CustomTkinter (Material Design 3 styling approach).
- Features:
  - Select local audio files and run predictions.
  - Display predicted genre with confidence.
  - Read and display model metrics (val acc/loss/epoch) from `models/best_model.pt`.
  - Non-blocking UI during loading/inference where feasible.
- Entry Point: `run_gui.py` (imports `app.gui:main`).

## API Service

- Framework: FastAPI + Uvicorn.
- Startup: Loads predictor once and reuses for incoming requests.
- Endpoints:
  - `GET /api/health` → Basic readiness status.
  - `GET /api/metrics` → `{ model_exists, model_path, val_acc, val_loss, epoch }` plus `ready` flag.
  - `GET /api/genres` → Returns list of genre labels.
  - `POST /api/predict` → Accept audio (file/path/spec) and return predicted label + probabilities.
  - `GET /` → Project metadata + endpoint index.
- Module Path: `app.api:app`.

## Orchestration & Deployment (Windows)

- Batch Script: `run_app.bat` (global environment; no venv).
  - Starts API: `uvicorn app.api:app --host 127.0.0.1 --port 8000` in background.
  - Launches GUI: `python .\run_gui.py`.
  - On GUI close, shuts down API and cleans related processes.
- Environment: Python 3.11 with PyTorch (CUDA build for GPU), librosa, fastapi, uvicorn, customtkinter, requests, soundfile; `ffmpeg` in PATH.

## Reproducibility & How To Run

1. Prerequisites
   - Windows, Python 3.11 on PATH.
   - PyTorch with CUDA (if using GPU), `ffmpeg` installed system-wide.
   - Install Python deps (if needed):
     ```powershell
     cd <your filepath>
     pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
     pip install librosa soundfile fastapi uvicorn customtkinter requests tqdm matplotlib scikit-learn
     ```
2. One-click (recommended)
   - Double-click `run_app.bat`.
   - GUI opens; API runs at `http://127.0.0.1:8000`.
3. Manual run (optional)
   ```powershell
   cd <your filepath>
   python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
   # In another terminal
   python .\run_gui.py
   ```

## Challenges & Mitigations

- Invalid CUDA device ordinal → Implemented safe auto-selection of best CUDA device; fallback to CPU.
- Audio decode errors (NoBackendError/unsupported codecs) → Installed `soundfile`, `ffmpeg`, and added robust exception handling with silent fallback.
- Import paths after restructuring → Made `app/` a package; added `src/` to `sys.path` where needed.
- Clean shutdown of API/GUI → Batch script coordinates startup and ensures teardown on exit.

## Limitations & Future Work

- Improve class balance and augmentations; consider mixup or class-weighted loss.
- Explore stronger architectures (e.g., CRNNs, transformers) and larger input contexts.
- Add test-time augmentation and calibration for probabilities.
- Provide in-API training metrics history and confusion matrix endpoints.
- Add file drag-and-drop and batch prediction to GUI.

## Ethical Considerations

- Dataset biases may affect genre predictions; misclassification is possible.
- Respect licensing terms of training data; avoid sharing copyrighted tracks.

## Evaluation Protocol

- Stratified train/validation split with fixed seed.
- Report accuracy, loss; optionally macro/micro F1 and per-class metrics.
- Use held-out validation set only for model selection; avoid test set leakage.

## References

- Defferrard et al., FMA: A Dataset for Music Analysis.
- McFee et al., librosa: Audio and Music Signal Analysis in Python.
- Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library.
- FastAPI documentation.
- Uvicorn ASGI server.
- Park et al., SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.

## Appendix A — Project Structure

```
application root/
├─ app/
│  ├─ api.py
│  └─ gui.py
├─ src/
│  ├─ config.py
│  ├─ inference.py
│  ├─ model.py
│  ├─ preprocessing.py
│  └─ metrics.py
├─ models/
│  └─ best_model.pt           # created after training
├─ notebooks/
│  └─ train_model.ipynb       # training workflow (if present)
├─ run_app.bat
├─ run_gui.py                 # GUI launcher
└─ REPORT.md                  # this report
```

## Appendix B — API Quick Test

```powershell
# Health
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/health
# Metrics
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/metrics
```
