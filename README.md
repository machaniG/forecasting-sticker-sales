
# Sticker Sales MLOps Pipeline  

An automated end-to-end machine learning pipeline for predicting sticker sales.
This project demonstrates data cleaning, feature engineering, model training, evaluation, and daily automation using GitHub Actions.

## Project Overview
- **ETL:** Cleans and enriches data (adds holiday + GDP features)
- **Modeling:** Trains XGBoost and Random Forest models
- **Metrics:** Evaluates performance using MAPE
- **Automation:** Runs daily with GitHub Actions and uploads model artifacts

---

## Folder Structure

sticker-sales-mlops-pipeline/

│
├── data/ # Contains raw and processed datasets

│ ├── raw/ # Unmodified input data

│ └── processed/ # Cleaned and feature-engineered data
│

├── scripts/ # All Python scripts

│ ├── etl.py # Data cleaning, transformation, feature creation

│ └── train.py # Model training, evaluation, and saving artifacts
│

├── artifacts/ # Trained models and evaluation metrics

│ └── .gitkeep
│

├── logs/ # Log files for pipeline runs

│ └── .gitkeep
│

├── .github/workflows/ # GitHub Actions automation workflow

│ └── ml_pipeline.yml
│

├── requirements.txt # Python dependencies

├── .gitignore # Ignored files/folders (data, logs, artifacts)

└── README.md # Project documentation

---
## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/machaniG/sticker-sales-mlops-pipeline.git
cd sticker-sales-mlops-pipeline

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the pipeline manually (optional)

python scripts/etl.py
python scripts/train.py
```

### 4. Automation

The full pipeline runs daily via **GitHub Actions** (.github/workflows/ml_pipeline.yml)

## Tech Stack

Python

Pandas, Scikit-learn, XGBoost

WBGAPI, holidays

GitHub Actions for CI/CD automation

## 📊 Author

**Machani G**  
[GitHub Profile](https://github.com/machaniG)
