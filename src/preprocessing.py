"""
Audio preprocessing utilities
"""
import librosa
import numpy as np
from config import PREPROCESSING_CONFIG


def load_and_preprocess_audio(audio_path, config=None):
    """
    Load audio file and compute log-mel spectrogram
    
    Args:
        audio_path: Path to audio file
        config: Preprocessing configuration dict
        
    Returns:
        np.ndarray: Log-mel spectrogram of shape (n_mels, time_frames)
    """
    if config is None:
        config = PREPROCESSING_CONFIG
    
    try:
        # Load audio
        y, sr = librosa.load(
            audio_path,
            sr=config["sample_rate"],
            duration=config["duration"],
            mono=True
        )
    except Exception as e:
        # If file fails to load, return silent spectrogram
        print(f"Warning: Failed to load {audio_path}: {e}")
        # Create silent audio
        y = np.zeros(config["sample_rate"] * config["duration"])
        sr = config["sample_rate"]
    
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=config["top_db"])
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"]
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
    
    # Pad or truncate to fixed length
    target_frames = int(config["duration"] * sr / config["hop_length"])
    if log_mel_spec.shape[1] < target_frames:
        # Pad
        pad_width = target_frames - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate
        log_mel_spec = log_mel_spec[:, :target_frames]
    
    return log_mel_spec


def spec_augment(spec, time_masks=2, freq_masks=2, time_mask_width=20, freq_mask_width=10):
    """
    Apply SpecAugment to spectrogram for training augmentation
    
    Args:
        spec: Input spectrogram (n_mels, time_frames)
        time_masks: Number of time masks
        freq_masks: Number of frequency masks
        time_mask_width: Maximum width of time masks
        freq_mask_width: Maximum width of frequency masks
        
    Returns:
        np.ndarray: Augmented spectrogram
    """
    spec = spec.copy()
    n_mels, n_frames = spec.shape
    
    # Apply frequency masks
    for _ in range(freq_masks):
        f = np.random.randint(0, freq_mask_width)
        f0 = np.random.randint(0, n_mels - f)
        spec[f0:f0+f, :] = 0
    
    # Apply time masks
    for _ in range(time_masks):
        t = np.random.randint(0, time_mask_width)
        t0 = np.random.randint(0, n_frames - t)
        spec[:, t0:t0+t] = 0
    
    return spec
