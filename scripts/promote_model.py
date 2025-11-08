"""
Promote the latest registered MLflow model to Production and archive previous Production versions.
"""
import mlflow
from mlflow.tracking import MlflowClient
import os

MODEL_NAME = "StickerSalesBestModel"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Get all versions of the model
versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production", "Archived"])

# Find the latest version (highest version number)
latest_version = max(versions, key=lambda v: int(v.version))

print(f"Promoting model version {latest_version.version} to Production...")

# Transition latest to Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version.version,
    stage="Production",
    archive_existing_versions=True  # This will archive any previous Production versions
)

print(f"Model version {latest_version.version} is now in Production. Previous Production versions archived.")
