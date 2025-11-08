"""
Model inference script for sticker sales predictions.
Loads the trained model and makes predictions on new data.
"""

import pandas as pd
import joblib
from pathlib import Path
import logging

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

# Paths
MODEL_PATH = Path("artifacts/xgb_pipeline.joblib")  # Using XGBoost as default
PROCESSED_PATH = Path("data/processed/cleaned.csv")  # For feature names reference

def load_model(model_path=MODEL_PATH):
    """Load the trained pipeline"""
    try:
        pipeline = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return pipeline
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def prepare_features(df, reference_data_path=PROCESSED_PATH):
    """Prepare features in the same format as training data"""
    # Load reference data to get feature names
    ref_data = pd.read_csv(reference_data_path)
    feature_cols = [col for col in ref_data.columns 
                   if col not in ['num_sold', 'id', 'date']]
    
    # Ensure all required columns are present
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        logger.warning(f"Missing columns in input data: {missing_cols}")
        for col in missing_cols:
            df[col] = 0  # or another appropriate default value
    
    return df[feature_cols]

def predict(data, model_path=MODEL_PATH):
    """Make predictions on new data"""
    # Load model
    pipeline = load_model(model_path)
    
    # Prepare features
    X = prepare_features(data)
    
    # Make predictions
    try:
        predictions = pipeline.predict(X)
        logger.info(f"Generated predictions for {len(predictions)} samples")
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        try:
            # Load and prepare data
            data = pd.read_csv(input_file)
            
            # Make predictions
            predictions = predict(data)
            
            # Add predictions to data
            data['predicted_sales'] = predictions
            
            # Save results
            output_file = 'predictions.csv'
            data.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}")
            sys.exit(1)
    else:
        logger.error("Please provide an input file path")
        sys.exit(1)