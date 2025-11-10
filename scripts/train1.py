from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error
from .inference_transformer import FeatureEnrichmentTransformer # NEW IMPORT!

import logging
import mlflow
import mlflow.sklearn
from datetime import datetime
import subprocess
import shap
import matplotlib.pyplot as plt


# === Directory Setup (FIX for FileNotFoundError) ===
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True) # FIX: Create logs directory
# ===================================================


# NOTE: PROCESSED_PATH name changed to reflect the simplified ETL output
PROCESSED_PATH = Path("processed/base_cleaned.csv")


problem_type = "regression"
target_column = "num_sold"
id_col = "id" # Assuming you have an ID column
date_col = "date"

# time_based train_test split logic: use earlier dates for training and later dates for validation
TRAIN_MASK = lambda df: df["date"].dt.year <= 2015
VAL_MASK = lambda df: (df["date"].dt.year >= 2016) & (df["date"].dt.year <= 2017)


# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log"), # Changed log name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# model configurations
MODEL_CONFIGS = {
    "RandomForest": {
        "model": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10),
        "params": {"model__max_depth": 10, "model__n_estimators": 100},
    },
    "XGBoost": {
        "model": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        ),
        "params": {"model__learning_rate": 0.1, "model__n_estimators": 100},
    },
}

# Preprocessor definition
def get_preprocessor(df):
    """
    Defines the ColumnTransformer for feature scaling and encoding.
    This runs AFTER the FeatureEnrichmentTransformer.
    """
    # Columns created by FeatureEnrichmentTransformer
    numerical_cols = [
        "lag_1", "rolling_7", "month", "day", "dayofweek", "weekofyear", "gdp_per_capita"
    ]
    
    # Columns from raw data
    categorical_cols = ["country", "store", "product", "is_holiday"]

    # Ensure all columns exist before creating the preprocessor
    missing_num = [c for c in numerical_cols if c not in df.columns]
    if missing_num:
        logger.warning(f"Missing numerical columns in dataframe: {missing_num}. Adding as 0.")
        for c in missing_num:
             df[c] = 0.0

    missing_cat = [c for c in categorical_cols if c not in df.columns]
    if missing_cat:
        logger.warning(f"Missing categorical columns in dataframe: {missing_cat}. Adding as 'NA'.")
        for c in missing_cat:
             df[c] = 'NA'

    # 1. Standard Scaling for numerical features
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # 2. One-Hot Encoding for categorical features
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder='drop' # Drop all other columns (like 'date', 'num_sold')
    )
    return preprocessor

def run_training(commit):
    """Loads data, trains models, logs results to MLflow, and registers the best model."""
    
    # Ensure MLflow is configured
    try:
        if not mlflow.get_tracking_uri():
             mlflow.set_tracking_uri("file:./mlruns") # Fallback to local
    except Exception:
        # Handle cases where tracking URI might not be fully set up
        mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("Sticker Sales Time Series Forecasting")
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log MLflow parameters
        mlflow.log_param("data_path", str(PROCESSED_PATH))
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("split_strategy", "time_based (pre-2016 train, 2016-2017 val)")
        
        # 1. Load Data
        try:
            df = pd.read_csv(PROCESSED_PATH)
            df[date_col] = pd.to_datetime(df[date_col])
        except FileNotFoundError:
            logger.error(f"Processed data file not found at {PROCESSED_PATH}. Run ETL first.")
            return

        # 2. Split Data (Time-based split)
        X_train = df[TRAIN_MASK(df)].drop(columns=[target_column])
        y_train = df[TRAIN_MASK(df)][target_column]
        X_val = df[VAL_MASK(df)].drop(columns=[target_column])
        y_val = df[VAL_MASK(df)][target_column]

        logger.info(f"Train/Validation split: {len(X_train)}/{len(X_val)} samples.")

        # 3. Define the Full End-to-End Pipeline
        # Step 1: Feature Enrichment (handles lag, rolling, holiday, GDP)
        enrichment_transformer = FeatureEnrichmentTransformer(target_column=target_column)
        
        # Step 2: Preprocessor (handles scaling and OHE on features created by Step 1)
        # Note: We fit the transformer on the training data BEFORE defining the preprocessor
        # to ensure it learns lag/rolling medians from the training set only.
        X_train_enriched = enrichment_transformer.fit_transform(X_train, y_train)
        preprocessor = get_preprocessor(X_train_enriched)
        
        best_mape = float("inf")
        best_model_name = ""
        best_pipeline = None
        results = {}

        # 4. Train Models
        for name, config in MODEL_CONFIGS.items():
            logger.info(f"Training {name}...")

            # Combine all steps into a single, comprehensive pipeline
            pipe = Pipeline(
                steps=[
                    ("enricher", enrichment_transformer), # The custom feature engineering step
                    ("preprocessor", preprocessor),      # Scaling and Encoding
                    ("model", config["model"]),          # The ML model
                ]
            )

            # Fit the pipeline
            pipe.fit(X_train, y_train)

            # Predict on validation set
            y_pred = pipe.predict(X_val)
            
            # Post-processing: predictions cannot be negative
            y_pred = np.maximum(0, y_pred)

            # Calculate metrics
            mape = mean_absolute_percentage_error(y_val, y_pred)
            results[name] = mape
            logger.info(f"{name} MAPE: {mape:.4f}")

            # MLflow logging for this model
            mlflow.log_params(config["params"])
            mlflow.log_metric(f"{name}_mape", mape)
            
            try:
                # Log the full pipeline for this model
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    artifact_path=f"model_{name}",
                    signature=mlflow.models.infer_signature(X_val, y_pred),
                    input_example=X_val.head(1).to_dict('records')
                )
            except Exception as e:
                logger.warning(f"Could not log MLflow model artifact for {name}: {e}")

            # Track best model
            if mape < best_mape:
                best_mape = mape
                best_model_name = name
                best_pipeline = pipe

        # Save metrics with versioning
        metrics_path = ARTIFACTS_DIR / "metrics.txt"
        metrics_text = "\n".join([f"{m.upper()}: {v:.4f}" for m, v in results.items()])
        with open(metrics_path, "w") as f:
            f.write(metrics_text)
        mlflow.log_artifact(str(metrics_path))

        # Register the best model (the complete end-to-end pipeline)
        if best_pipeline is not None:
            logger.info(f"Registering best model: {best_model_name} (MAPE={best_mape:.4f})")
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path="best_model", # Log under a generic path for registration
                registered_model_name="StickerSalesBestModel"
            )
            # Log version metadata
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_mape", best_mape)
            mlflow.set_tag("commit", commit)
            mlflow.set_tag("train_date", datetime.now().isoformat())

        logger.info("Training complete. Complete pipeline, metrics, and MLflow logs saved.")


if __name__ == "__main__":
    import sys
    # Fetch the commit SHA passed as an argument from the CI workflow
    commit_sha = sys.argv[1] if len(sys.argv) > 1 else 'local_run'
    # Use the first 7 characters for display/tagging
    commit = commit_sha[:7]
    
    run_training(commit)