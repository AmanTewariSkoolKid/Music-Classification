"""
Inference utilities for genre prediction
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from model import create_model
from preprocessing import load_and_preprocess_audio
from config import MODEL_CONFIG, GENRE_LABELS, MODELS_DIR


class GenrePredictor:
    """Class for loading model and making predictions"""
    
    def __init__(self, model_path):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = create_model(
            n_mels=MODEL_CONFIG["input_shape"][0],
            n_classes=MODEL_CONFIG["n_classes"]
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load genre labels
        self.genre_labels = checkpoint.get('genre_labels', GENRE_LABELS)
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Using device: {self.device}")
    
    def predict(self, audio_path, top_k=5):
        """
        Predict genre for an audio file
        
        Args:
            audio_path: Path to audio file
            top_k: Number of top predictions to return
            
        Returns:
            list: List of dicts with 'genre' and 'probability' keys
        """
        # Preprocess audio
        spec = load_and_preprocess_audio(audio_path)
        
        # Convert to tensor and add batch dimension
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time_frames)
        spec_tensor = spec_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(spec_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.genre_labels)))
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            predictions.append({
                'genre': self.genre_labels[idx],
                'probability': float(prob)
            })
        
        return predictions
    
    def predict_batch(self, audio_paths, top_k=5):
        """
        Predict genres for multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            top_k: Number of top predictions to return
            
        Returns:
            list: List of prediction results for each file
        """
        results = []
        for path in audio_paths:
            try:
                predictions = self.predict(path, top_k)
                results.append({
                    'file': str(path),
                    'predictions': predictions,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'file': str(path),
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results


def load_predictor(model_path=None):
    """
    Convenience function to load predictor
    
    Args:
        model_path: Path to model checkpoint (if None, looks for 'best_model.pt' in MODELS_DIR)
        
    Returns:
        GenrePredictor: Initialized predictor
    """
    if model_path is None:
        model_path = MODELS_DIR / "best_model.pt"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return GenrePredictor(model_path)


if __name__ == "__main__":
    print("Inference module ready.")
    print("\nUsage example:")
    print("  from inference import load_predictor")
    print("  predictor = load_predictor('models/best_model.pt')")
    print("  results = predictor.predict('path/to/audio.mp3', top_k=5)")
