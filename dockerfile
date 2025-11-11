# Use the same official Python image as the GitHub Actions runner
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- MLOps Configuration ---
# 1. Define a build-time argument for the MLflow Tracking URI.
#    This value is passed from the GitHub Actions YAML.
ARG MLFLOW_TRACKING_URI
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MODEL_NAME=StickerSalesModel

# 2. Copy the application and prediction script.
#    We assume serving_app.py is in the root directory and uses the script.
COPY serving_app.py .
COPY scripts/predict.py ./scripts/

# Create necessary directories
RUN mkdir -p logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Command to run the API: Uvicorn now points to the correct module and app variable.
# Format: [module name]:[FastAPI app variable name]
CMD ["uvicorn", "serving_app:app", "--host", "0.0.0.0", "--port", "8000"]