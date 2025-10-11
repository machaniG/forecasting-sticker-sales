from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error
import logging


PROCESSED_PATH = Path("processed/cleaned.csv")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Adjust target and problem_type to your dataset (regression/classification)
problem_type = "regression"
target_column = "num_sold"
id_col = "id"
date_col = "date"

# time_based train_test split logic: use earlier dates for training and later dates for validation
# adjustable for other datasets

TRAIN_MASK = lambda df: df["year"] <= 2015
VAL_MASK = lambda df: (df["year"] >= 2016) & (df["year"] <= 2017)


# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,  # can be DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/etl.log"),  # save to file
        logging.StreamHandler()               # also print to console
    ]
)
logger = logging.getLogger(__name__)


# model configurations
# List all models you want to train — no code changes below needed

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
    # Identify numeric and categorical columns

    numerical_features = df.drop([target_column, date_col, id_col], axis=1, errors="ignore") \
                           .select_dtypes(include="number").columns.tolist()
    categorical_features = [
        c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if c not in [target_column, date_col, id_col] ]

    
    #preprocessor

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ], remainder="passthrough")

    # Train/val split
    train_mask = TRAIN_MASK(df)
    val_mask = VAL_MASK(df)

    X_train = df[train_mask].drop(columns=[target_column, date_col, id_col], errors="ignore")
    y_train = df[train_mask][target_column]
    X_val = df[val_mask].drop(columns=[target_column, date_col, id_col], errors="ignore")
    y_val = df[val_mask][target_column]

    return X_train, y_train, X_val, y_val, preprocessor

# model pipeline

def build_models(preprocessor):
    pipelines = {}
    for name, cfg in MODELS.items():
        model = cfg["class"](**cfg["params"])
        pipelines[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    return pipelines

# model evaluation 

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    return mean_absolute_percentage_error(y_val, preds)

# training models
def run_training():
    df = pd.read_csv(PROCESSED_PATH)
    logger.info(f"Columns in processed dataset: {list(df.columns)}")
    X_train, y_train, X_val, y_val, preprocessor = prepare_features(df)

    pipelines = build_models(preprocessor)

    results = {}
    for name, pipe in pipelines.items():
        logger.info(f"\nTraining {name.upper()}...")
        pipe.fit(X_train, y_train)
        mape = evaluate_model(pipe, X_val, y_val)
        results[name] = mape
        logger.info(f"{name.upper()} MAPE: {mape:.4f}")

        # save models
        joblib.dump(pipe, ARTIFACTS_DIR / f"{name}_pipeline.joblib")

    # save metrics
    metrics_text = "\n".join([f"{m.upper()}: {v:.4f}" for m, v in results.items()])
    with open(ARTIFACTS_DIR / "metrics.txt", "w") as f:
        f.write(metrics_text)

    logger.info("Training complete. Pipelines and metrics saved to artifacts/")
    logger.info(metrics_text)


if __name__ == "__main__":
    run_training()
