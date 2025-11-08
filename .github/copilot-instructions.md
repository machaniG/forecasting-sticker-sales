# AI Assistant Instructions for Sticker Sales MLOps Pipeline

## Project Overview
This is an ML pipeline for predicting sticker sales using historical data, enriched with GDP and holiday features. The pipeline includes ETL, model training (XGBoost/RandomForest), and automated daily runs via GitHub Actions.

## Key Architecture Components
- **ETL Pipeline** (`scripts/etl1.py`):
  - Handles data cleaning, feature engineering
  - Enriches data with World Bank GDP data and holiday information
  - Creates temporal features (lag, rolling windows)
  - Outputs processed data to `data/processed/cleaned.csv`

- **Training Pipeline** (`scripts/train1.py`):
  - Implements time-based train/validation split (train ≤2015, val 2016-2017)
  - Uses sklearn Pipeline with ColumnTransformer for preprocessing
  - Trains both XGBoost and RandomForest models
  - Saves models and metrics to `artifacts/` directory

## Project Conventions
1. **Data Processing**:
   - Raw data expected in `data/raw/sticker_sales.csv`
   - All intermediate files go to `data/processed/`
   - Features are auto-generated from base data (no manual feature files)

2. **Model Training**:
   - Models defined in `MODELS` dictionary in `train1.py`
   - All models must implement sklearn's estimator interface
   - Train/val split uses time-based masks (defined at top of `train1.py`)

3. **Logging**:
   - All scripts use Python's `logging` module
   - Logs saved to `logs/` directory
   - Both file and console handlers are configured

## Common Workflows
1. **Adding a New Feature**:
   - Add feature generation code to `etl1.py`'s ETL pipeline
   - Features should be computed within the main `run_etl()` function
   - Update logging messages appropriately

2. **Adding a New Model**:
   - Add model config to `MODELS` dictionary in `train1.py`
   - Must include 'class' and 'params' keys
   - Model class must be compatible with sklearn Pipeline

3. **Debugging Pipeline**:
   - Check logs in `logs/etl.log` and `logs/train.log`
   - Intermediate data saved in `data/processed/`
   - Model metrics stored in `artifacts/metrics.txt`

## Integration Points
- **World Bank API**: Uses `wbgapi` for GDP data (see `fetch_gdp_data()` in `etl1.py`)
- **Holidays API**: Uses `holidays` package for country-specific holidays
- **GitHub Actions**: Daily automated runs (see `.github/workflows/ml_pipeline.yml`)

## Dependencies
Key packages and versions are specified in `requirements.txt`. Core dependencies:
- pandas (≥2.0)
- scikit-learn (≥1.3)
- xgboost (≥2.0)
- wbgapi (≥1.0)
- holidays (≥0.50)