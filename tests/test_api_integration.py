"""
Integration tests for the FastAPI application.
These tests verify that the API endpoints are functioning correctly
and can handle single and batch prediction requests.
"""
import pytest
from httpx import AsyncClient
from pathlib import Path

# NOTE: Importing the app directly for testing
from serving_app import app

# Create a test client that will be used across all tests
# Using AsyncClient is the correct practice for testing FastAPI async endpoints
client = AsyncClient(app=app, base_url="http://testserver")

# --- Conditional Skipping Setup ---
# Define the expected path for the local fallback model (used by the API)
MODEL_FALLBACK_PATH = Path("artifacts/test_pipeline.joblib")
MODEL_IS_PRESENT = MODEL_FALLBACK_PATH.exists()

# Define the skip decorator to use for tests requiring the model
requires_model = pytest.mark.skipif(
    not MODEL_IS_PRESENT,
    reason="Model artifact (artifacts/test_pipeline.joblib) not found. Run training step first to enable prediction tests."
)
# -----------------------------------


@pytest.mark.asyncio
async def test_health_check():
    """Verify the health check endpoint returns 200 OK and reports model status."""
    response = await client.get("/health")
    assert response.status_code == 200
    
    # We check the content of the response to ensure the API is honest about the model status
    health_data = response.json()
    assert health_data["status"] == "healthy"
    
    # The API should correctly report if the model is loaded
    # It will be False if the artifact is missing, but the health check should still pass (200)
    assert health_data["model_loaded"] == MODEL_IS_PRESENT


@pytest.mark.asyncio
# Only run this test if the model is present on disk
@requires_model 
async def test_predict_single():
    """Verify single prediction endpoint works when the model is loaded."""
    payload = {
        "country": "US",
        "store": "Store_123",
        "product": "Sticker_ABC",
        "date": "2025-11-07",
        "gdp_per_capita": 65000.0
    }
    response = await client.post("/predict", json=payload)
    
    # If the model is loaded (as required by the decorator), we expect a 200 OK
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_sales" in data
    assert data["predicted_sales"] >= 0.0


@pytest.mark.asyncio
# Only run this test if the model is present on disk
@requires_model
async def test_predict_batch(tmp_path):
    """Verify batch prediction endpoint works when the model is loaded."""
    
    # Create a temporary CSV file
    csv_content = "country,store,product,date,gdp_per_capita\nUS,Store_123,Sticker_ABC,2025-11-07,65000.0\nUK,Store_456,Sticker_DEF,2025-11-08,42000.0\n"
    csv_file = tmp_path / "batch.csv"
    csv_file.write_text(csv_content)
    
    with open(csv_file, "rb") as f:
        # The request uses the standard file format for FastAPI
        response = await client.post(
            "/predict/batch", 
            files={"file": ("batch.csv", f, "text/csv")}
        )

    # If the model is loaded (as required by the decorator), we expect a 200 OK
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert "mean_prediction" in data