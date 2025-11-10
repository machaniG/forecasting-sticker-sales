"""
Model inference script for sticker sales predictions.
Loads the latest Production model from MLflow and makes predictions on raw data.
The deployed model is a full Scikit-learn pipeline containing both ETL and 
preprocessing steps, ensuring consistency.
"""

import pandas as pd
import numpy as np
import logging
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = "StickerSalesBestModel"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")

def load_production_model(model_name=MODEL_NAME):
    """Loads the model currently marked as 'Production' in MLflow."""
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not production_versions:
            logger.error(f"No model found in 'Production' stage for '{model_name}'.")
            return None
            
        version = production_versions[0]
        model_uri = f"models:/{model_name}/{version.version}"
        logger.info(f"Loading Production model version {version.version} from URI: {model_uri}")
        
        # Load the complete Scikit-learn pipeline
        model = mlflow.sklearn.load_model(model_uri)
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        return None

def make_prediction(raw_data_path):
    """
    Loads raw data, runs the full pipeline, and makes predictions.
    """
    model = load_production_model()
    if model is None:
        return

    try:
        # 1. Load RAW data (e.g., just 'date', 'country', 'store', 'product')
        df_raw = pd.read_csv(raw_data_path)
        logger.info(f"Loaded {len(df_raw)} raw rows for prediction.")

        # Ensure 'date' column is datetime
        df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
        
        # 2. Predict: The model (which is the full pipeline) runs ALL ETL/preprocessing internally.
        predictions = model.predict(df_raw)
        
        # Post-processing (e.g., ensuring non-negative predictions)
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
        logger.error("Please provide an input file path (e.g., python predict.py input.csv)")
        sys.exit(1)