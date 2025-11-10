"""
Model inference script for sticker sales predictions.
Loads the latest Production model from MLflow, with a local fallback for testing.
"""

import pandas as pd
import numpy as np
import logging
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import sys
from pathlib import Path
import joblib 

# Configure logging
logger = logging.getLogger(__name__)

MODEL_NAME = "StickerSalesBestModel"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")

# Define fallback path for local testing/CI (MUST match train1.py)
LOCAL_FALLBACK_PATH = Path("artifacts/test_pipeline.joblib")


def load_production_model(model_name=MODEL_NAME):
    """
    Loads the model currently marked as 'Production' in MLflow, 
    with a fallback to a locally saved pipeline for testing environments.
    """
    model = None
    
    # 1. Attempt to load from MLflow Registry
    try:
        logger.info(f"Attempting to load Production model '{model_name}' from MLflow...")
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if production_versions:
            version = production_versions[0]
            model_uri = f"models:/{model_name}/{version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Successfully loaded Production model version {version.version} from MLflow.")
            return model
        else:
            logger.warning(f"No model found in 'Production' stage for '{model_name}' in MLflow.")
            
    except Exception as e:
        # This is expected to fail during the test stage if MLflow server is not running
        logger.warning(f"MLflow model loading failed: {e}. Falling back to local model for testing.")
    
    # 2. Fallback to local joblib file for testing
    if model is None and LOCAL_FALLBACK_PATH.exists():
        try:
            model = joblib.load(LOCAL_FALLBACK_PATH)
            logger.info(f"✅ Successfully loaded model from local fallback path: {LOCAL_FALLBACK_PATH}")
            return model
        except Exception as e:
            logger.error(f"FATAL: Failed to load model from local fallback path {LOCAL_FALLBACK_PATH}: {e}")
            return None
    
    # 3. Final failure state
    logger.error("❌ Model loading failed: No Production model in MLflow and no local fallback found.")
    return None 


# --- CLI EXECUTION LOGIC (omitted for brevity) ---
def make_prediction(raw_data_path):
    """
    Loads raw data, runs the full pipeline, and makes predictions.
    """
    model = load_production_model()
    if model is None:
        return 

    try:
        # 1. Load RAW data 
        df_raw = pd.read_csv(raw_data_path)
        logger.info(f"Loaded {len(df_raw)} raw rows for prediction.")

        # Ensure 'date' column is datetime
        df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
        
        # 2. Predict: The model runs ALL ETL/preprocessing internally.
        predictions = model.predict(df_raw)
        
        # Post-processing
        predictions = np.maximum(0, predictions)

        df_raw['predicted_sales'] = predictions
        output_file = 'predictions.csv'
        df_raw.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        make_prediction(input_file)
    else:
        logger.error("Please provide an input file path")
        sys.exit(1)