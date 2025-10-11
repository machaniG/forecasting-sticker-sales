"""
ETL Script for Sticker Sales Dataset
- Cleans and prepares data
- Adds temporal, rolling, and holiday features
- Enriches with World Bank GDP per capita data dynamically
- Saves cleaned dataset for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import wbgapi as wb
import holidays
import logging
import os

# === Paths ===
RAW_PATH = Path("data/raw/sticker_sales.csv")       # Input dataset
PROCESSED_PATH = Path("processed/cleaned.csv")      # Output dataset
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

# Ensure necessary directories exist
RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# Then configure logging
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

# === GDP Fetcher ===
def fetch_gdp_data(df, indicator="NY.GDP.PCAP.CD"):
    """
    Fetch GDP per capita data from World Bank dynamically
    based on country codes and years present in the dataset.
    Returns a tidy DataFrame with columns ['country', 'year', 'gdp_per_capita'].
    """
    country_codes = df["country"].unique().tolist()
    years = range(df["year"].min(), df["year"].max() + 1)

    logger.info(f"🌍 Fetching GDP data for {len(country_codes)} countries ({years.start}–{years.stop - 1})...")

    try:
        # Download GDP data
        df_gdp_wide = wb.data.DataFrame(indicator, country_codes, years)
        df_gdp_wide = df_gdp_wide.reset_index()

        # Normalize columns (World Bank can change naming conventions)
        if "economy" in df_gdp_wide.columns:
            df_gdp_wide.rename(columns={"economy": "country"}, inplace=True)
        if "Time" in df_gdp_wide.columns:
            df_gdp_wide.rename(columns={"Time": "year"}, inplace=True)

        # Extract relevant columns
        if indicator in df_gdp_wide.columns:
            df_gdp = df_gdp_wide[["country", "year", indicator]].rename(columns={indicator: "gdp_per_capita"})
        else:
            # Sometimes the indicator name is part of a MultiIndex; handle that
            df_gdp = df_gdp_wide.melt(id_vars=["country", "year"], value_name="gdp_per_capita")[
                ["country", "year", "gdp_per_capita"]
            ]

        # Ensure correct types
        df_gdp["year"] = df_gdp["year"].astype(int)
        df_gdp["gdp_per_capita"] = pd.to_numeric(df_gdp["gdp_per_capita"], errors="coerce")

        logger.info(f"✅ GDP data fetched: {df_gdp.shape[0]} rows")
        return df_gdp

    except Exception as e:
        logger.info(f"⚠️ Warning: Failed to fetch GDP data. Error: {e}")
        return pd.DataFrame(columns=["country", "year", "gdp_per_capita"])


# === Main ETL ===
def run_etl():
    logger.info("Starting ETL process...")

    # 1️⃣ Extract
    df = pd.read_csv(RAW_PATH)
    logger.info(f"Loaded {len(df)} rows from {RAW_PATH}")

    # 2️⃣ Basic Cleaning
    df.dropna(subset=["num_sold"], inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    # 3️⃣ Lag and Rolling Features
    df = df.sort_values(["country", "store", "product", "date"])
    df["lag_1"] = df.groupby(["country", "store", "product"])["num_sold"].shift(1)
    df["rolling_7"] = df.groupby(["country", "store", "product"])["num_sold"].rolling(7, min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True)

    # 4️⃣ Holiday Features
    def is_holiday(row):
        try:
            country_code = row["country"]
            date = row["date"]
            country_holidays = holidays.country_holidays(country_code)
            return int(date in country_holidays)
        except Exception:
            return 0

    df["is_holiday"] = df.apply(is_holiday, axis=1)

    # 5️⃣ Enrich with GDP
    df_gdp = fetch_gdp_data(df)
    if not df_gdp.empty:
        df = df.merge(df_gdp, on=["country", "year"], how="left")
        df["gdp_per_capita"] = df["gdp_per_capita"].fillna(df["gdp_per_capita"].median())
    else:
        logger.info("⚠️ No GDP data added (API failed).")

    # 6️⃣ Handle missing values and sanity checks
    df["lag_1"] = df["lag_1"].fillna(df["lag_1"].median())
    df["rolling_7"] = df["rolling_7"].fillna(df["rolling_7"].median())

    # 7️⃣ Save Processed Data
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"✅ ETL complete. Cleaned data saved to {PROCESSED_PATH}")

    return df


# === Script entrypoint ===
if __name__ == "__main__":
    run_etl()
