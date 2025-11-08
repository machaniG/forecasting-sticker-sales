# You can build and run the Docker container with the following commands:

# Build the container:

docker build -t sticker-sales-model 

# Run the container in different modes:
#1.Full pipeline (ETL + Training + Predictions):

docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts sticker-sales-model

# 2. Training only:

docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts sticker-sales-model train

# Predictions only (provide input file):

docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts sticker-sales-model predict /app/data/your_input_file.csv

# The containerized model includes:

"""
1. A Dockerfile that sets up the Python environment and dependencies
2. An inference script (predict.py) for making predictions with the trained model
3. An entry point script (run.sh) that handles different operations (train/predict/pipeline)
4. Volume mounts for data and artifacts to persist them outside the container

The container can be run in three modes:

pipeline: Runs the full ETL + training + prediction pipeline (default)
train: Runs only the ETL and training steps
predict: Runs predictions on new data (requires input file)
"""


# after API endpoints

"""I've created a clean, well-structured FastAPI application for serving the ML model. Here are the key features:

API Endpoints:

POST /predict: Single prediction endpoint
POST /predict/batch: Batch prediction from CSV file
GET /health: Health check endpoint
Features:

Proper data validation using Pydantic models
Automatic API documentation (Swagger UI at /docs)
CORS middleware for cross-origin requests
Proper error handling and logging
Automatic feature preprocessing
Support for batch predictions
Model Management:

Automatic model loading on startup
Global model instance for efficiency
Model version tracking

To use the API:

Single Prediction:
"""
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"country": "US", "store": "Store_123", "product": "Sticker_ABC", "date": "2025-11-07", "gdp_per_capita": 65000.0}'


# batch prediction:
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_data.csv"

# heath check:
curl "http://localhost:8000/health"


"""
The API automatically handles:

Feature preprocessing
Input validation
Error handling
Model loading/unloading
Batch processing
Health monitoring

"""

# how to clone this repository and run the project
git clone https://github.com/yourusername/sticker-sales-mlops.git
cd sticker-sales-mlops
docker-compose up --build





@app.post("/predict", response_model=SalesPredictionResponse)
async def predict_single(input_data: SalesInput):
    """