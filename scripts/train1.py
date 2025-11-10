from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error
from scripts.inference_transformer import FeatureEnrichmentTransformer 

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
# Define fallback path for local testing (MUST match predict.py)
LOCAL_FALLBACK_PATH = ARTIFACTS_DIR / "test_pipeline.joblib" 

problem_type = "regression"
target_column = "num_sold"
id_col = "id" # Assuming you have an ID column
date_col = "date"

# time_based train_test split logic: use earlier dates for training and later dates for validation
TRAIN_MASK = lambda df: df["date"].dt.year <= 2015
VAL_MASK = lambda df: (df["date"].dt.year >= 2016) & (df["date"].dt.year <= 2017)


# Basic logging configuration
logging.basicConfig(\
    level=logging.INFO,\
    format="%(asctime)s - %(levelname)s - %(message)s",\
    handlers=[\
        logging.FileHandler("logs/train.log"), # Changed log name\
        logging.StreamHandler()\
    ]\
)
logger = logging.getLogger(__name__)\


# model configurations
model_configs = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

# --- Utility to get current commit hash for MLflow tagging ---
def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        commit = "no-git-repo"
    return commit

# --- Main Feature & Pipeline Definition ---
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a simplified feature preparation logic for unit testing purposes.
    In the main training pipeline, this is done by FeatureEnrichmentTransformer.
    This function mimics the *minimum* required preparation before the transformer runs.
    """
    df = df.copy()
    
    # 1. Date conversion (MANDATORY before temporal feature creation)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    # 2. Add year column (REQUIRED for TRAIN_MASK and unit testing assertion)
    df["year"] = df["date"].dt.year
    
    # 3. Apply the full transformer pipeline
    # NOTE: This is necessary to generate the rest of the features 
    # (month, day, weekday, lag, rolling, etc.) for the assertion test to pass on column names.
    enricher = FeatureEnrichmentTransformer(target_column=target_column)
    # Fit_transform is used here because the unit test runs without a previously fitted model
    df_transformed = enricher.fit_transform(df) 
    
    # Drop the date column which is no longer needed in the final feature set
    if date_col in df_transformed.columns:
        df_transformed = df_transformed.drop(columns=[date_col])

    return df_transformed


def create_pipeline(regressor, target_column):
    """Creates the full end-to-end Scikit-learn pipeline."""
    
    # 1. Feature Engineering (Custom Transformer)
    enrichment_step = ('enrichment', FeatureEnrichmentTransformer(target_column=target_column))

    # 2. Preprocessing (Standard Scaling for numerical, One-Hot for categorical)
    # Define columns after enrichment (e.g., year, month, day, is_holiday, lag_1, rolling_7, gdp_per_capita)
    
    # Example columns that will exist AFTER the FeatureEnrichmentTransformer runs:
    numerical_features = ['year', 'month', 'day', 'dayofweek', 'weekofyear', 'lag_1', 'rolling_7', 'gdp_per_capita']
    categorical_features = ['country', 'store', 'product'] 
    
    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        # The remainder columns (like ID, original date, etc.) will be dropped by default.
        remainder='drop' 
    )

    # 3. Final Pipeline: Enrichment -> Preprocessing -> Regressor
    pipeline = Pipeline(steps=[
        enrichment_step, 
        ('preprocessor', preprocessor), 
        ('regressor', regressor)
    ])
    
    return pipeline, numerical_features # Return numerical features for SHAP

# --- Main Training Function ---

def run_training():
    commit = get_git_commit()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("sticker-sales-forecasting")
    
    with mlflow.start_run(run_name=f"Training Run {run_id}") as run:
        
        # Load and split data
        try:
            df = pd.read_csv(PROCESSED_PATH)
            df[date_col] = pd.to_datetime(df[date_col])
        except FileNotFoundError:
            logger.error(f"Processed file not found at {PROCESSED_PATH}. Run ETL first.")
            return

        df_train = df[TRAIN_MASK(df)].copy()
        df_val = df[VAL_MASK(df)].copy()

        X_train, y_train = df_train.drop(columns=[target_column]), df_train[target_column]
        X_val, y_val = df_val.drop(columns=[target_column]), df_val[target_column]
        
        logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
        
        # MLflow metadata
        mlflow.log_param("commit", commit)
        mlflow.log_param("train_start_date", df_train[date_col].min().strftime('%Y-%m-%d'))
        mlflow.log_param("val_end_date", df_val[date_col].max().strftime('%Y-%m-%d'))
        
        best_mape = float('inf')
        best_model_name = ""
        best_pipeline = None
        results = {}

        for name, regressor in model_configs.items():
            logger.info(f"Starting training for {name}...")
            pipe, numerical_features = create_pipeline(regressor, target_column)
            
            # Train model
            pipe.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = pipe.predict(X_val)
            y_pred = np.maximum(0, y_pred) # Ensure no negative sales
            mape = mean_absolute_percentage_error(y_val, y_pred)
            results[name] = mape
            
            logger.info(f"{name} MAPE: {mape:.4f}")
            mlflow.log_metric(f"val_mape_{name}", mape)
            
            # Logging model artifact (optional, but useful for inspection)
            try:
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    artifact_path=f"model_{name}",
                    registered_model_name=None # Do not register individual models
                )
            except Exception as e:
                logger.warning(f"Could not log MLflow model artifact for {name}: {e}")

            # Optional: SHAP Plot generation (requires model to be saved)
            # (SHAP code omitted for brevity/stability)

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
            
            # --- CRITICAL STEP: Save local fallback for testing ---
            logger.info(f"Saving best model locally to {LOCAL_FALLBACK_PATH} for testing/CI fallback.")
            joblib.dump(best_pipeline, LOCAL_FALLBACK_PATH)
            # ----------------------------------------------------

            # Log version metadata
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_mape", best_mape)
            mlflow.set_tag("commit", commit)
            mlflow.set_tag("train_date", datetime.now().isoformat())

        logger.info("Training complete. Complete pipeline, metrics, and MLflow logs saved.")

if __name__ == "__main__":
    run_training()