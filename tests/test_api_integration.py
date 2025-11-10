"""
Integration tests for the FastAPI application.
These tests verify that the API endpoints are functioning correctly
and can handle single and batch prediction requests.
"""
import pytest
from httpx import AsyncClient
from pathlib import Path
from serving_app import app # Import the application object

# --- Conditional Skipping Setup ---
# Define the expected path for the local fallback model (used by the API)
# The model file path used in serving_app.py and this test should match.
MODEL_FALLBACK_PATH = Path("artifacts/test_pipeline.joblib")
MODEL_IS_PRESENT = MODEL_FALLBACK_PATH.exists()

# Define the skip decorator to use for tests requiring the model
requires_model = pytest.mark.skipif(
    not MODEL_IS_PRESENT,
    reason="Model artifact (artifacts/test_pipeline.joblib) not found. Run training step first to enable prediction tests."
)

# --- Define an async client fixture for proper pytest/httpx integration ---
@pytest.fixture
@pytest.mark.asyncio # MANDATORY: Ensures the async fixture is executed correctly by pytest-asyncio
async def async_client():
    """Provides an httpx.AsyncClient instance wrapped around the FastAPI app."""
    # Using 'async with' ensures correct setup and teardown of the client connection.
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client
# ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    """Verify the health check endpoint returns 200 OK and reports model status."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    
    # We check the content of the response to ensure the API is honest about the model status
    health_data = response.json()
    assert health_data["status"] == "healthy"
    
    # The API should correctly report if the model is loaded
    assert health_data["model_loaded"] == MODEL_IS_PRESENT


@pytest.mark.asyncio
# Only run this test if the model is present on disk
@requires_model 
async def test_predict_single(async_client: AsyncClient):
    """Verify single prediction endpoint works when the model is loaded."""
    payload = {
        "country": "US",
        "store": "Store_123",
        "product": "Sticker_ABC",
        "date": "2025-11-07",
        "gdp_per_capita": 65000.0
    }
    response = await async_client.post("/predict", json=payload)
    
    # If the model is loaded (as required by the decorator), we expect a 200 OK
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_sales" in data
    assert data["predicted_sales"] >= 0.0


@pytest.mark.asyncio
# Only run this test if the model is present on disk
@requires_model
async def test_predict_batch(async_client: AsyncClient, tmp_path):
    """Verify batch prediction endpoint works when the model is loaded."""
    
    # Create a temporary CSV file
    csv_content = "country,store,product,date,gdp_per_capita\nUS,Store_123,Sticker_ABC,2025-11-07,65000.0\nUK,Store_456,Sticker_DEF,2025-11-08,42000.0\n"
    csv_file = tmp_path / "batch.csv"
    csv_file.write_text(csv_content)
    
    with open(csv_file, "rb") as f:
        # The request uses the standard file format for FastAPI
        response = await async_client.post(
            "/predict/batch", 
            files={"file": ("batch.csv", f, "text/csv")}
        )

    # If the model is loaded (as required by the decorator), we expect a 200 OK
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert "mean_prediction" in data