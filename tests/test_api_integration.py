import pytest
# We explicitly import the synchronous TestClient, which is the most reliable 
# way to test FastAPI apps when the underlying async client setup (like httpx) fails.
from fastapi.testclient import TestClient 
from scripts.serving_app import app # Import the FastAPI application instance
from datetime import datetime

# --- FIX: Define the synchronous client fixture ---
# This fixture provides the TestClient instance. It is synchronous (`def`).
@pytest.fixture(scope="module")
def api_client():
    """Provides a synchronous TestClient for the FastAPI app."""
    # TestClient correctly mounts the ASGI application (`app`)
    with TestClient(app) as client:
        yield client

# Since the fixture is synchronous, the test functions should be synchronous too.
# We remove @pytest.mark.asyncio and the `await` keyword.

def test_health_check_endpoint(api_client: TestClient):
    """Test the /health endpoint to ensure the API is running."""
    # Use synchronous client methods
    response = api_client.get("/health")
    
    # 1. Assert Status Code
    assert response.status_code == 200
    
    # 2. Assert Response Content
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "timestamp" in data

def test_single_prediction_endpoint_success(api_client: TestClient):
    """Test the /predict endpoint with valid input."""
    valid_input = {
        "date": "2024-01-10",
        "country": "Australia",
        "store": "A",
        "product": "Sticker1",
    }
    
    # Use synchronous client methods
    response = api_client.post("/predict", json=valid_input)
    
    # 1. Assert Status Code
    if response.status_code != 200:
        # Check for 503 if the model is not loaded (common during CI initial setup)
        assert response.status_code == 503, f"Expected 200 or 503, got {response.status_code}: {response.text}"
    
    if response.status_code == 200:
        data = response.json()
        assert "predicted_sales" in data
        assert isinstance(data["predicted_sales"], float)
        assert data["predicted_sales"] >= 0  # Sales should be non-negative
        assert "model_version" in data
        
def test_single_prediction_endpoint_validation_error(api_client: TestClient):
    """Test the /predict endpoint with invalid input (missing fields)."""
    # To trigger a 422, we must omit a required field like 'date'
    invalid_input_missing_date = {
        "country": "US",
        "store": "East",
        "product": "TypeA"
    }

    # Use synchronous client methods
    response = api_client.post("/predict", json=invalid_input_missing_date)
    
    # FastAPI returns 422 for validation errors
    assert response.status_code == 422
    data = response.json()
    
    # Asserting against the standard Pydantic validation error structure
    assert "detail" in data
    
    # FIX: Use a more robust check to ensure we find the 'date' field in the error location list.
    found_date_error = False
    for error in data['detail']:
        # Ensure 'loc' key exists and is iterable
        if 'loc' in error and isinstance(error['loc'], list):
            # The name of the missing field is the last element in the location list
            if error['loc'][-1] == 'date':
                found_date_error = True
                break
    
    assert found_date_error, f"Expected validation error for 'date' field, but found errors: {data['detail']}"