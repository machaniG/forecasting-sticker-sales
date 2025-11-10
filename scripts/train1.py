from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error
from scripts.inference_transformer import FeatureEnrichmentTransformer # NEW IMPORT!

import logging
import mlflow
import mlflow.sklearn
from datetime import datetime
import subprocess
import shap
import matplotlib.pyplot as plt


# NOTE: PROCESSED_PATH name changed to reflect the simplified ETL output
PROCESSED_PATH = Path("processed/base_cleaned.csv")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
MODELS = {
    "xgb": {
        "class": XGBRegressor,
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror"
        }
    },
    "rf": {
        "class": RandomForestRegressor,
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        }
    }
}


def prepare_features(df):
    """
    Defines the ColumnTransformer for the standard Scikit-learn steps 
    AFTER the custom ETL transformer has run.
    """
    
    # Run the ETL transformer first to create all the required columns
    etl_transformer = FeatureEnrichmentTransformer(target_column=target_column)
    
    # We fit the ETL transformer manually on the training data BEFORE splitting the pipeline.
    # We pass the target column in the fit/transform step for lag/rolling feature calculation.
    train_mask = TRAIN_MASK(df)
    val_mask = VAL_MASK(df)

    X_train_raw = df[train_mask].drop(columns=[target_column], errors="ignore")
    y_train = df[train_mask][target_column]
    X_val_raw = df[val_mask].drop(columns=[target_column], errors="ignore")
    y_val = df[val_mask][target_column]

    # Fit the ETL transformer on raw training data (X_train_raw) and its target (y_train)
    etl_transformer.fit(X_train_raw, y_train) 
    
    # Transform all data using the fitted ETL transformer
    X_train = etl_transformer.transform(X_train_raw)
    X_val = etl_transformer.transform(X_val_raw)

    # --- Feature Selection for Scikit-learn Preprocessing ---
    # These are the columns created by the ETL step and the base data
    
    # Identify features *after* ETL transformation
    numerical_features = X_train.drop(columns=[id_col], errors="ignore") \
                                 .select_dtypes(include="number").columns.tolist()
    
    # Remove features that are already numbers but should be treated as categories/IDs
    exclude_from_scaling = ['year', 'month', 'day', 'weekday', 'weekofyear', 'is_holiday'] 
    numerical_features = [f for f in numerical_features if f not in exclude_from_scaling]
    
    # Identify categorical/temporal features for OHE
    categorical_features = [
        c for c in X_train.columns
        if c not in numerical_features and c not in [id_col]
    ]

    # --- Scikit-learn Preprocessor ---
    # This only handles scaling and encoding of the now-generated features
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ], remainder="passthrough")

    return X_train_raw, y_train, X_val_raw, y_val, etl_transformer, preprocessor

# model pipeline

def build_models(etl_transformer, preprocessor):
    pipelines = {}
    for name, cfg in MODELS.items():
        # The final pipeline now consists of 3 steps: 
        # 1. FeatureEnrichmentTransformer (runs ETL)
        # 2. ColumnTransformer (runs OHE/Scaling)
        # 3. Final Model
        pipelines[name] = Pipeline([
            ("etl_features", etl_transformer), # Step 1: Runs ALL ETL features
            ("preprocessor", preprocessor),   # Step 2: Runs Scaling/Encoding
            ("model", cfg["class"](**cfg["params"]))  # Step 3: Runs the final model
        ])
    return pipelines

# model evaluation 
def evaluate_model(model, X_val_raw, y_val):
    """Note: Evaluation now takes the RAW input data for transformation."""
    preds = model.predict(X_val_raw)
    # Ensure no negative predictions for MAPE calculation if applicable
    preds[preds < 0] = 0
    return mean_absolute_percentage_error(y_val, preds)

# training models
def run_training():

    df = pd.read_csv(PROCESSED_PATH)
    # Ensure 'date' column is datetime for TRAIN_MASK/VAL_MASK to work
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    logger.info(f"Columns in processed dataset: {list(df.columns)}")
    X_train_raw, y_train, X_val_raw, y_val, etl_transformer, preprocessor = prepare_features(df)
    
    pipelines = build_models(etl_transformer, preprocessor)
    results = {}
    best_model_name = None
    best_mape = float('inf')
    best_pipeline = None

    # Get git commit hash
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception:
        commit = "unknown"

    mlflow.set_tracking_uri("file:./mlruns")  # Local tracking by default
    experiment_name = "StickerSalesPrediction"
    mlflow.set_experiment(experiment_name)


    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    with mlflow.start_run(run_name=f"train_{run_id}") as run:
        for name, pipe in pipelines.items():
            logger.info(f"\nTraining {name.upper()}...")
            # Fit the entire end-to-end pipeline on raw data
            # The pipeline runs ETL -> Preprocessing -> Model Training
            pipe.fit(X_train_raw, y_train)
            
            # Evaluate using the raw validation data
            mape = evaluate_model(pipe, X_val_raw, y_val)
            results[name] = mape
            logger.info(f"{name.upper()} MAPE: {mape:.4f}")

            # Log params, metrics, and model to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_params(MODELS[name]["params"])
            mlflow.log_metric("mape", mape)

            # Save model artifact (The complete pipeline)
            model_path = ARTIFACTS_DIR / f"{name}_pipeline.joblib"
            joblib.dump(pipe, model_path)
            mlflow.log_artifact(str(model_path))

            # SHAP plot logic is complex with ColumnTransformer and custom steps.
            # Skipping complex SHAP plot generation for brevity and pipeline focus.
            try:
                # Log model using mlflow.sklearn which handles pipelines well
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    artifact_path=f"model_{name}"
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
        logger.info(metrics_text)


if __name__ == "__main__":
    run_training()