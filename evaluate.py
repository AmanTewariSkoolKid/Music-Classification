"""
Evaluate the trained model on a validation split and export:
- Confusion matrix image to src/logs/confusion_matrix.png
- Metrics JSON (accuracy, macro precision/recall/F1) to src/logs/metrics.json

Usage (PowerShell):
  python src/evaluate.py
Options:
  --limit N    Evaluate on first N validation samples (default: 300)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
from config import GENRE_LABELS, LOGS_DIR, DATA_DIR
from inference import load_predictor

# Paths: use config.DATA_DIR to be robust regardless of cwd
DATA_ROOT = Path(DATA_DIR)
AUDIO_DIR = DATA_ROOT / "fma_medium"
TRACKS_CSV = DATA_ROOT / "fma_metadata" / "tracks.csv"


def _ensure_paths():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")
    if not TRACKS_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {TRACKS_CSV}")


def _make_val_split(test_size: float = 0.2, seed: int = 42) -> Tuple[List[str], List[int]]:
    """Build a validation split directly from FMA metadata.

    Returns: (val_paths, val_label_indices)
    """
    # FMA tracks.csv uses MultiIndex columns
    df = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])

    samples = []
    for track_id in df.index:
        try:
            top_genre = df.loc[track_id, ("track", "genre_top")]
        except Exception:
            top_genre = None
        if not isinstance(top_genre, str):
            continue
        if top_genre not in GENRE_LABELS:
            continue
        folder = str(int(track_id)).zfill(6)[:3]
        file_path = AUDIO_DIR / folder / f"{str(int(track_id)).zfill(6)}.mp3"
        if file_path.exists():
            samples.append((str(file_path), GENRE_LABELS.index(top_genre)))

    if not samples:
        raise RuntimeError("No valid samples found. Ensure FMA-medium files and metadata are present.")

    # Stratified split
    from sklearn.model_selection import train_test_split
    X = [p for p, _ in samples]
    y = [lbl for _, lbl in samples]
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_val, y_val


def _plot_confusion(cm: np.ndarray, labels: List[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="Evaluate on first N validation samples")
    args = parser.parse_args()

    _ensure_paths()

    # Load predictor (uses src/models/best_model.pt by default via inference.load_predictor)
    predictor = load_predictor()

    # Build/Load validation split
    val_paths, val_labels = _make_val_split()

    if args.limit and len(val_paths) > args.limit:
        val_paths = val_paths[: args.limit]
        val_labels = val_labels[: args.limit]

    # Predict top-1 labels
    pred_labels: List[int] = []
    label_map = {g: i for i, g in enumerate(predictor.genre_labels)}

    for p in val_paths:
        preds = predictor.predict(p, top_k=1)
        top_genre = preds[0]["genre"]
        pred_labels.append(label_map.get(top_genre, -1))

    # Filter any failures (-1)
    filtered_true = []
    filtered_pred = []
    for t, pr in zip(val_labels, pred_labels):
        if pr != -1:
            filtered_true.append(t)
            filtered_pred.append(pr)

    if not filtered_true:
        raise RuntimeError("No valid predictions to evaluate.")

    # Metrics
    cm = confusion_matrix(filtered_true, filtered_pred, labels=list(range(len(GENRE_LABELS))))
    report = classification_report(
        filtered_true,
        filtered_pred,
        labels=list(range(len(GENRE_LABELS))),
        target_names=GENRE_LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Save metrics JSON
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(report.get("accuracy", 0.0)),
        "precision_macro": float(report.get("macro avg", {}).get("precision", 0.0)),
        "recall_macro": float(report.get("macro avg", {}).get("recall", 0.0)),
        "f1_macro": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "samples": len(filtered_true),
    }
    with (LOGS_DIR / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix image
    _plot_confusion(cm, GENRE_LABELS, LOGS_DIR / "confusion_matrix.png")

    print(f"Saved metrics to: {LOGS_DIR / 'metrics.json'}")
    print(f"Saved confusion matrix to: {LOGS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
