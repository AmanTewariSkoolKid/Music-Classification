"""
FastAPI REST API for music genre classification
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
from datetime import datetime

import sys
from pathlib import Path
# Ensure src/ is importable for core modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from inference import load_predictor
from config import API_CONFIG, CACHE_DIR, MODELS_DIR
from metrics import read_checkpoint_metrics

# Initialize FastAPI app
app = FastAPI(
    title="Music Genre Classifier",
    description="API for classifying music genres using deep learning",
    version="1.0.0"
)

# Global predictor (loaded on startup)
predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global predictor
    model_path = MODELS_DIR / "best_model.pt"
    
    if model_path.exists():
        try:
            predictor = load_predictor(model_path)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("  API will run but predictions will fail until model is trained")
    else:
        print(f"✗ Model not found at {model_path}")
        print("  Please train a model first using train.py")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "ready": predictor is not None,
        "message": "Model ready" if predictor else "Model not loaded"
    }


@app.get("/api/metrics")
async def get_metrics():
    """Return training/validation metrics from the best checkpoint if present."""
    meta = read_checkpoint_metrics(MODELS_DIR)
    return {
        "ready": predictor is not None,
        **meta
    }


@app.post("/api/predict")
async def predict_single(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=16, description="Number of top predictions")
):
    """
    Predict genre for a single audio file
    
    Args:
        file: Audio file (mp3, wav, flac, etc.)
        top_k: Number of top predictions to return (1-16)
        
    Returns:
        JSON with prediction results
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    # Validate file size
    if file.size and file.size > API_CONFIG["max_file_size"]:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {API_CONFIG['max_file_size'] / (1024*1024):.0f}MB"
        )
    
    # Save uploaded file temporarily
    prediction_id = str(uuid.uuid4())
    temp_dir = CACHE_DIR / prediction_id
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / file.filename
    
    try:
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Make prediction
        predictions = predictor.predict(str(temp_file), top_k=top_k)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return {
            "prediction_id": prediction_id,
            "filename": file.filename,
            "top_k": top_k,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        # Clean up on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/genres")
async def list_genres():
    """List all supported genres"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "genres": predictor.genre_labels,
        "count": len(predictor.genre_labels)
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Music Genre Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "metrics": "/api/metrics",
            "predict": "/api/predict",
            "genres": "/api/genres",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        log_level="info"
    )
