"""
Simple API demo and documentation
"""

# API Summary
API_INFO = """
╔══════════════════════════════════════════════════════════════╗
║          Music Genre Classifier API - Quick Guide           ║
╚══════════════════════════════════════════════════════════════╝

🌐 Server URL: http://127.0.0.1:8000
📚 Interactive Docs: http://127.0.0.1:8000/docs

═══════════════════════════════════════════════════════════════
📍 ENDPOINTS
═══════════════════════════════════════════════════════════════

1. Root Information
   GET /
   
   Returns API info and available endpoints

2. Health Check
   GET /api/health
   
   Check if API and model are ready
   Response:
   {
     "status": "ok",
     "ready": true/false,
     "message": "Model ready" or "Model not loaded"
   }

3. List Genres
   GET /api/genres
   
   Get all supported genres (16 FMA-medium genres)
   Response:
   {
     "genres": ["Electronic", "Rock", ...],
     "count": 16
   }

4. Predict Genre
   POST /api/predict?top_k=5
   
   Upload audio file and get predictions
   
   Request:
   - multipart/form-data
   - field: "file" (audio file)
   - query param: "top_k" (1-16, default=5)
   
   Response:
   {
     "prediction_id": "uuid",
     "filename": "song.mp3",
     "top_k": 5,
     "predictions": [
       {"genre": "Rock", "probability": 0.85},
       {"genre": "Pop", "probability": 0.10},
       ...
     ],
     "timestamp": "2025-11-13T..."
   }

═══════════════════════════════════════════════════════════════
🔧 USING THE API
═══════════════════════════════════════════════════════════════

Python Example:
--------------
import requests

# Health check
response = requests.get("http://127.0.0.1:8000/api/health")
print(response.json())

# Predict genre
with open("song.mp3", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://127.0.0.1:8000/api/predict?top_k=5",
        files=files
    )
    print(response.json())


PowerShell Example:
------------------
# Health check
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health"

# List genres
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/genres"


Browser:
-------
Open http://127.0.0.1:8000/docs for interactive testing!


═══════════════════════════════════════════════════════════════
⚠️  CURRENT STATUS
═══════════════════════════════════════════════════════════════

✅ API server is running
✅ All endpoints are accessible
⚠️  Model not trained yet - predictions will fail

To enable predictions:
1. Prepare your FMA dataset
2. Run: python train.py
3. Model will be saved to models/best_model.pt
4. Restart the API server

═══════════════════════════════════════════════════════════════
"""

print(API_INFO)

# Quick test if requests is available
try:
    import requests
    print("\n✓ 'requests' library is available for API testing")
    print("\nRun 'python test_api.py' to test all endpoints!")
except ImportError:
    print("\n⚠️  Install 'requests' to test the API:")
    print("   pip install requests")
