import pytest
from fastapi.testclient import TestClient
from api import app
import os

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_single():
    payload = {
        "country": "US",
        "store": "Store_123",
        "product": "Sticker_ABC",
        "date": "2025-11-07",
        "gdp_per_capita": 65000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_sales" in data
    assert "model_version" in data

def test_predict_batch(tmp_path):
    # Create a temporary CSV file
    csv_content = "country,store,product,date,gdp_per_capita\nUS,Store_123,Sticker_ABC,2025-11-07,65000.0\nUK,Store_456,Sticker_DEF,2025-11-08,42000.0\n"
    csv_file = tmp_path / "batch.csv"
    csv_file.write_text(csv_content)
    with open(csv_file, "rb") as f:
        response = client.post("/predict/batch", files={"file": ("batch.csv", f, "text/csv")})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert data["count"] == 2
    assert isinstance(data["mean_prediction"], float)
