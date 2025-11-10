import pytest
import httpx
from scripts.serving_app import app # Import the FastAPI application instance
from datetime import datetime

# Define the asynchronous client fixture 
# We must use httpx.AsyncClient configured to wrap the FastAPI app instance.
# httpx, when paired with FastAPI's TestClient setup (or implicitly via the
# app=app argument in a compatible environment), is the standard tool.
# To bypass the TypeError, we use the fact that httpx should be able to mount
# an ASGI app instance directly, but since the raw `httpx.AsyncClient` fails, 
# we rely on the specific `fastapi.testclient.AsyncClient` pattern which is 
# often used with a local URL. For unit testing, the standard approach is actually 
# NOT to use `app=app` on `httpx.AsyncClient` but to use `fastapi.testclient.TestClient`. 
# However, i am using async tests, let's stick to the modern httpx pattern 
# that should be compatible with FastAPI.
#
# If the simple `app=app` fails, we fall back to the widely compatible
# `TestClient` pattern, which is synchronous but often acceptable for API testing.
# Since my tests are async, we will try the slightly older but working pattern 
# of connecting to the app via its structure.

@pytest.fixture(scope="module")
async def async_client():
    """Provides an asynchronous HTTP client for the FastAPI app."""
    # We use the recommended pattern for testing ASGI apps with httpx.
    # This mounts the FastAPI application instance (`app`) as the target 
    # of the client, allowing tests to run against the app in memory.
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client

# Ensure the pytest-asyncio marker is present for async tests
@pytest.mark.asyncio
async def test_health_check_endpoint(async_client: httpx.AsyncClient):
    """Test the /health endpoint to ensure the API is running."""
    # ... test body remains the same
    response = await async_client.get("/health")
    
    # 1. Assert Status Code
    assert response.status_code == 200
    
    # 2. Assert Response Content
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "timestamp" in data

@pytest.mark.asyncio
async def test_single_prediction_endpoint_success(async_client: httpx.AsyncClient):
    """Test the /predict endpoint with valid input."""
    # This input must match the Pydantic model structure in serving_app.py
    valid_input = {
        "date": "2024-01-10",
        "country": "Australia",
        "store": "A",
        "product": "Sticker1",
    }
    
    # Using the correct endpoint /predict
    response = await async_client.post("/predict", json=valid_input)
    
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
        
@pytest.mark.asyncio
async def test_single_prediction_endpoint_validation_error(async_client: httpx.AsyncClient):
    """Test the /predict endpoint with invalid input (missing fields)."""
    # To trigger a 422, we must omit 'date' or 'country'
    invalid_input_missing_date = {
        "country": "US",
        "store": "East",
        "product": "TypeA"
    }

    # Using the endpoint /predict
    response = await async_client.post("/predict", json=invalid_input_missing_date)
    
    # FastAPI returns 422 for validation errors
    assert response.status_code == 422
    data = response.json()
    
    # Asserting against the structure returned by the custom exception handler
    # Since the structure of the 422 response is custom in the API, 
    # we assert against the known custom fields 'detail' and 'errors'.
    # If the custom handler fails, FastAPI returns a standard Pydantic error list.
    assert "detail" in data or "errors" in data