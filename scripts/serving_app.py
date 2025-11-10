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
# Import the custom transformer (required to load the pipeline)
from scripts.inference_transformer import FeatureEnrichmentTransformer 


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
MODEL_PATH = Path("artifacts/xgb_pipeline.pkl") 
pipeline = None

def load_pipeline():
    """Load the trained scikit-learn pipeline."""
    global pipeline
    try:
        if not MODEL_PATH.exists():
             logger.warning(f"Pipeline file not found at {MODEL_PATH}. Attempting fallback load from MLflow...")
             # NOTE: In a real environment, you'd integrate MLflow here to load the model
             # from the registry if it's not present locally.
             # For now, we'll keep the local load logic, but if this fails, pipeline remains None.
             pipeline = None
             return

        pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Inference pipeline loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        pipeline = None

# Load the pipeline on startup
load_pipeline()

# === Pydantic Schemas for API Input/Output ===
class SalesPredictionRequest(BaseModel):
    date: str = Field(..., description="Date of the prediction (YYYY-MM-DD)")
    country: str = Field(..., description="Country code (e.g., US, CA, UK)")
    store: str = Field("NA", description="Store name/ID")
    product: str = Field("NA", description="Product category/ID")

class SalesPredictionResponse(BaseModel):
    predicted_sales: float = Field(..., description="Predicted number of stickers sold")
    prediction_date: str = Field(..., description="Timestamp of the prediction")
    model_version: str = Field(..., description="Version of the model used")

class BatchPredictionResponse(BaseModel):
    predictions: List[float] = Field(..., description="List of predicted sales for all rows")
    mean_prediction: float = Field(..., description="Mean of all predicted sales")
    count: int = Field(..., description="Number of predictions made")
    model_version: str = Field(..., description="Version of the model used")

# === Exception Handlers ===
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Custom handler for Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": exc.errors()},
    )

# === API Endpoints ===

@app.post("/predict", response_model=SalesPredictionResponse)
async def predict(request: SalesPredictionRequest):
    """
    Predict sales for a single input record.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline not loaded.")
    
    try:
        # 1. Convert Pydantic model to DataFrame row
        input_data = request.model_dump()
        df = pd.DataFrame([input_data])
        
        # 2. Predict
        # The pipeline.predict method is assumed to return predictions and enriched features (which we ignore here)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)