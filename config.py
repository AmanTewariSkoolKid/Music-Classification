"""
Configuration settings for Music Genre Classification
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Audio preprocessing config
PREPROCESSING_CONFIG = {
    "sample_rate": 22050,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "top_db": 80,
    "duration": 30,  # seconds
}

# Model config
MODEL_CONFIG = {
    "n_classes": 16,  # FMA-medium has 16 genres
    "input_shape": (128, 1292),  # (n_mels, time_frames for 30s)
}

# Training config
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.01,
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

# Genre labels (FMA-medium)
GENRE_LABELS = [
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock",
    "Classical", "Country", "Easy Listening", "Jazz",
    "Soul-RnB", "Spoken", "Blues", "Punk"
]

# API config
API_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "max_file_size": 30 * 1024 * 1024,  # 30MB
    "default_top_k": 5,
}
