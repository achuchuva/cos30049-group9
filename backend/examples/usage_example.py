"""
Example usage scripts for the Spam Detection API.
"""
import requests
import json


# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health check endpoint."""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_single_prediction():
    """Test single text prediction."""
    print("=" * 60)
    print("Testing Single Prediction Endpoint")
    print("=" * 60)
    
    # Test spam text
    spam_text = "Congratulations! You've won $1000. Click here NOW to claim your prize!"
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json={"text": spam_text}
    )
    print(f"Text: {spam_text}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    # Test ham text
    ham_text = "Hello! I hope you're having a great day. Let's meet for coffee tomorrow at 3pm."
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json={"text": ham_text}
    )
    print(f"Text: {ham_text}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_batch_prediction():
    """Test batch prediction."""
    print("=" * 60)
    print("Testing Batch Prediction Endpoint")
    print("=" * 60)
    
    texts = [
        "Hello, how are you doing today?",
        "WIN FREE CASH NOW!!! URGENT!!!",
        "Meeting scheduled for tomorrow at 10am in conference room A",
        "Click here for amazing deals on pharmacy products!",
        "Can you send me the report by end of day?"
    ]
    
    response = requests.post(
        f"{BASE_URL}/api/v1/predict/batch",
        json={"texts": texts}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_model_info():
    """Test model info endpoint."""
    print("=" * 60)
    print("Testing Model Info Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/api/v1/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_error_handling():
    """Test error handling."""
    print("=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    # Test empty text
    print("Test 1: Empty text")
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json={"text": ""}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    # Test missing text field
    print("Test 2: Missing text field")
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json={}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Spam Detection API - Example Usage")
    print("Make sure the server is running on http://localhost:8000")
    print("=" * 60 + "\n")
    
    try:
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_error_handling()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API server.")
        print("Please make sure the server is running:")
        print("  cd backend && python -m uvicorn app.main:app --reload")
    except Exception as e:
        print(f"ERROR: {e}")
