"""
ETL Script for Sticker Sales Dataset
- Cleans and performs basic imputation on raw data.
- Saves cleaned dataset for use by the training pipeline (which now handles all feature engineering).
"""

import pandas as pd
from pathlib import Path
import logging
import os

# Ensure the logs directory exists before configuring the file handler
Path("logs").mkdir(parents=True, exist_ok=True)
# Ensure the raw and processed folders exist
RAW_PATH = Path("data/raw/sticker_sales.csv")       # Input dataset
PROCESSED_PATH = Path("processed/base_cleaned.csv") # Output dataset
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
# ===================================================

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/etl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Main ETL 
def run_etl():
    """Performs full ETL: cleaning, feature engineering, and enrichment."""
    logger.info("Starting base ETL process...")

    # 1️⃣ Extract
    try:
        df = pd.read_csv(RAW_PATH)
        logger.info(f"Loaded {len(df)} rows from {RAW_PATH}")
    except FileNotFoundError:
        logger.error(f"Raw data file not found at {RAW_PATH}. Please ensure the file exists.")
        return
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        return

    # 2️⃣ Basic Cleaning (Keep only basic cleaning that is NOT feature creation)
    # The transformer will handle date creation, lag, rolling, etc.
    df.dropna(subset=["num_sold", "date", "country"], inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    # 3️⃣ Ensure required columns exist for the next step (even if empty in raw data)
    for col in ["country", "store", "product"]:
        if col not in df.columns:
            df[col] = "NA"
    
    # 4️⃣ Save Base Processed Data
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"✅ Base ETL complete. Cleaned data saved to {PROCESSED_PATH}")


# Script entrypoint
if __name__ == "__main__":
    run_etl()
