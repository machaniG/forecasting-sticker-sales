# 📈 Sticker Sales Forecast Service: Production-Ready MLOps

## 🧩 Project Summary & Business Value

This project implements an **end-to-end MLOps pipeline** for *real-time sticker sales forecasting*, demonstrating how to operationalize machine learning models as reliable, production-grade microservices.


**Objective:**

Provide actionable, real-time sales predictions via an API to support **inventory optimization, marketing decisions,** and **revenue forecasting**.

## 📦 Docker Image & Adoption

The Docker image for this project is publicly available on Docker Hub.

<div align="center">
  <img width="1416" height="653" alt="image_in_dockerhub" src="https://github.com/user-attachments/assets/a2fe706b-940c-4d5e-b352-d1e80d49bae5" />

</div>

You can pull the image via:

```bash
docker pull frida33876/sticker-sales-mlops:fe1964ce
```

## ⚙️ MLOps Architecture & Tech Stack

The pipeline is fully containerized and automated from model training to CI/CD integration and Docker deployment.

### 🔁 Core MLOps Principles Demonstrated

- **Automation:** Automated build, test, and push workflows on every code or data change.

- **Model Governance:** Versioned model registration and tracking via MLflow.

- **Scalability:** Fast, low-latency inference through FastAPI and Uvicorn.

- **Reproducibility:** End-to-end data, model, and environment version control.


### 🧠 Integrated Feature Engineering & Prediction Pipeline

The deployed model is a **self-contained inference pipeline** that handles both **feature transformation and prediction.**

You can send *raw input data* (e.g., id, date, country, store, product), and the model automatically applies the full preprocessing logic used during training before generating predictions.

This design ensures:

- Consistent preprocessing between training and serving
- Minimal API input requirements
- Reproducible and scalable predictions


### 🧩 Architecture Overview
```
    A[Raw Data] --> B[Training (train1.py)]
    B --> C[MLflow Tracking & Model Registry]
    C --> D[CI/CD (GitHub Actions)]
    D --> E[Docker Build & Push]
    E --> F[FastAPI Serving (serving_app.py)]
    F --> G[Prediction Endpoint (/predict)]
```

## 🏗️ Models & Tools

  | Category                | Tools / Libraries      | Highlights                                                                            |
| ----------------------- | ---------------------- | ------------------------------------------------------------------------------------- |
| **Data Processing**     | Pandas                 | Robust, widely-used data manipulation library                                         |
| **Modeling**            | Random Forest, XGBoost | High-performance, ensemble-based forecasting models                                   |
| **Tracking & Registry** | MLflow                 | Full experiment tracking and model versioning                                         |
| **Serving Layer**       | FastAPI + Uvicorn      | High-speed, modern REST API for prediction                                            |
| **Deployment**          | Docker                 | Portable, reproducible service containerization                                       |
| **CI/CD**               | GitHub Actions         | Automated build, test, and push pipeline                                              |
| **Orchestration**       | Master Workflow        | Runs weekly and conditionally skips retraining when no code/data changes are detected |



## 🚀 Quick Start (Local Deployment)

Run the entire prediction service locally in minutes using Docker.

**1. Prerequisites**

Git

Docker

**2. Build and Run the Image (from Docker Hub)**
```bash
# Clone the project
git clone https://github.com/machaniG/sticker-sales-mlops-pipeline.git
cd sticker-sales-mlops-pipeline

# Build the Docker image
docker build -t frida33876/sticker-forecast:v1.0 .

# Run the API service on port 8080
docker run -d -p 8080:8000 --name sales_predictor frida33876/sticker-forecast:v1.0
```

**3. Test the Prediction Endpoint**

The API will be available at http://localhost:8080/predict
 (POST method).

Example Request Body:
```json
[
  { "id": 1, "date": "2025-10-05", "country": "Germany", "store": "Berlin_Alex", "product": "Sticker_A" },
  { "id": 2, "date": "2025-10-06", "country": "Germany", "store": "Munich_Center", "product": "Sticker_B" }
]
```

## 🔄 Automated Workflows

**Master Orchestrator Workflow** (mlops_orchestrator.yml):

- Runs automatically weekly to retrain and redeploy the model.
- Detects code and data changes but **skips training** if no updates are found.
- Triggers the **CI/CD pipeline** for build, registration, and Docker push.

**CI/CD Pipeline** (ci_pipeline.yml):

- Runs unit tests and model evaluation
- Registers the best model to MLflow
- Builds and pushes the Docker image to Docker Hub


## 🧰 Deployment Options

This project is designed for flexible deployment, both locally and in the cloud.

**Local Orchestration**

A docker-compose.yml file is included for easy local service orchestration. For example, to run the ML API, tracking server (MLflow), and supporting components together in one command.
```bash
docker-compose up --build
```
This approach enables seamless local testing before production deployment.

**Cloud Deployment (AWS-Ready)**

A reusable GitHub Actions workflow (deploy.yml) is included for future AWS deployment.
Once AWS credentials and infrastructure are configured, the workflow will automatically pull the latest image from Docker Hub and deploy it to an AWS service such as **ECS or ECR**.


## 👤 Author

Fridah Machani
📎 [LinkedIn Profile or Portfolio Link]
🐳 Docker Hub: frida33876
