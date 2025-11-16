"""
Quick API test script
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    """Test all API endpoints"""
    
    print("=" * 60)
    print("Testing Music Genre Classifier API")
    print("=" * 60)
    
    # Test root endpoint
    print("\n1. Testing root endpoint (GET /)...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test health check
    print("\n2. Testing health check (GET /api/health)...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test genres list
    print("\n3. Testing genres list (GET /api/genres)...")
    try:
        response = requests.get(f"{BASE_URL}/api/genres")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {data['count']} genres:")
            for i, genre in enumerate(data['genres'], 1):
                print(f"      {i}. {genre}")
        else:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test prediction (will fail without model)
    print("\n4. Testing prediction endpoint (POST /api/predict)...")
    print("   Note: This will fail until you train a model")
    try:
        # This would require an audio file
        print("   Skipping - requires audio file upload")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 60)

if __name__ == "__main__":
    print("\nMake sure the API server is running first!")
    print("Run in another terminal: python api.py\n")
    input("Press Enter when server is ready...")
    test_api()
