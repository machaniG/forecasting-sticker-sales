"""
FastAPI application for serving the sticker sales prediction model.
"""
import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path
import logging
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError
from fastapi import status

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
log_file = os.path.join("logs", "api.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Logging initialized successfully.")

# Initialize FastAPI app
app = FastAPI(
    title="Sticker Sales Predictor API",
    description="API for predicting sticker sales using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
MODEL_PATH = Path("artifacts/xgb_pipeline.joblib")
PROCESSED_PATH = Path("data/processed/cleaned.csv")
PIPELINE_PATH = Path("artifacts/inference_pipeline.joblib")
pipeline = None

# Data models
class SalesPredictionInput(BaseModel):
    """Input data model for sales prediction."""
    country: str = Field(..., description="Country code (e.g., US, UK)")
    store: str = Field(..., description="Store identifier")
    product: str = Field(..., description="Product identifier")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    gdp_per_capita: Optional[float] = Field(None, description="GDP per capita (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "country": "US",
                "store": "Store_123",
                "product": "Sticker_ABC",
                "date": "2025-11-07",
                "gdp_per_capita": 65000.0
            }
        }

class SalesPredictionResponse(BaseModel):
    """Response model for sales prediction."""
    predicted_sales: float = Field(..., description="Predicted number of sales")
    prediction_date: str = Field(..., description="Date of prediction")
    model_version: str = Field(..., description="Version of the model used")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[float] = Field(..., description="List of predictions")
    mean_prediction: float = Field(..., description="Mean of all predictions")
    count: int = Field(..., description="Number of predictions made")
    model_version: str = Field(..., description="Version of the model used")

# Model loading
def load_model(model_path: Path = MODEL_PATH):
    """Load the trained model pipeline."""
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global pipeline
    pipeline = joblib.load(PIPELINE_PATH)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )

@app.post("/predict", response_model=SalesPredictionResponse)
async def predict_sales(input_data: SalesPredictionInput):
    """
    Predict sales for a single input.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline not loaded.")
    try:
        df = pd.DataFrame([input_data.dict()])
        predictions, enriched = pipeline.predict(df)
        return SalesPredictionResponse(
            predicted_sales=float(predictions[0]),
            prediction_date=datetime.now().isoformat(),
            model_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict sales for multiple inputs from a CSV file.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline not loaded.")
    try:
        df = pd.read_csv(file.file)
        predictions, enriched = pipeline.predict(df)
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            mean_prediction=float(np.mean(predictions)),
            count=len(predictions),
            model_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

@app.get("/health")
async def health_check():
    """
    Check the health of the API and model.
    """
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")