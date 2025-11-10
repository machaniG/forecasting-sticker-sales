"""
Custom Scikit-learn Transformer that encapsulates ALL feature engineering logic
(including time, lag, holiday, and external data enrichment) required by the model.
This guarantees training-serving feature consistency.
"""
import pandas as pd
import numpy as np
import holidays
import wbgapi as wb
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

logger = logging.getLogger(__name__)

# === Utility Functions (Copied/Adapted from etl1.py) ===

def _fetch_gdp_data(df, indicator="NY.GDP.PCAP.CD"):
    """Fetch and prepare GDP data (static function for reuse)."""
    # NOTE: In a real-world deployed environment, dynamic API calls should be avoided
    # or rely on a feature store/cached database. For this exercise, we keep the call.
    country_codes = df["country"].unique().tolist()
    years = range(df["year"].min(), df["year"].max() + 1)
    
    logger.info(f"🌍 Fetching GDP data for {len(country_codes)} countries...")

    try:
        df_gdp_wide = wb.data.DataFrame(indicator, country_codes, years)
        # Simplified renaming for merging
        df_gdp = df_gdp_wide.reset_index().rename(columns={"economy": "country"})
        
        # Melt/Extract GDP
        gdp_col = [col for col in df_gdp.columns if col.startswith('YR')]
        if gdp_col:
            df_gdp = df_gdp.melt(
                id_vars=["country"], 
                value_vars=gdp_col, 
                var_name="year", 
                value_name="gdp_per_capita"
            )
            df_gdp["year"] = df_gdp["year"].str.replace('YR', '').astype(int)
        else:
            raise ValueError("Could not parse GDP columns from World Bank API.")

        df_gdp["gdp_per_capita"] = pd.to_numeric(df_gdp["gdp_per_capita"], errors="coerce")
        df_gdp.dropna(subset=['gdp_per_capita'], inplace=True)
        return df_gdp

    except Exception as e:
        logger.warning(f"⚠️ Warning: Failed to fetch GDP data. Error: {e}")
        return pd.DataFrame(columns=["country", "year", "gdp_per_capita"])

class FeatureEnrichmentTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to perform all non-Scikit-learn feature engineering
    (ETL steps) on the raw input DataFrame.
    """
    
    def __init__(self, target_column=None):
        self.target_column = target_column
        self.median_gdp = None
        self.median_lag_1 = None
        self.median_rolling_7 = None
        self.gdp_data = None
    
    def fit(self, X, y=None):
        """
        Fit method calculates statistics (like medians) from the training data
        and fetches external data (GDP).
        """
        X_copy = X.copy()
        
        # --- 1. Basic Date Features & Indexing ---
        X_copy['date'] = pd.to_datetime(X_copy['date'], errors='coerce')
        X_copy.dropna(subset=['date'], inplace=True)
        X_copy["year"] = X_copy["date"].dt.year

        # --- 2. Fetch and Fit GDP Data ---
        self.gdp_data = _fetch_gdp_data(X_copy)
        if not self.gdp_data.empty:
            self.median_gdp = self.gdp_data["gdp_per_capita"].median()
        
        # --- 3. Fit Lag/Rolling Medians (Requires target if available) ---
        if self.target_column and y is not None:
            X_copy[self.target_column] = y
            X_copy = X_copy.sort_values(["country", "store", "product", "date"])
            
            # Calculate Lag 1 on training data
            X_copy["lag_1"] = X_copy.groupby(["country", "store", "product"])[self.target_column].shift(1)
            self.median_lag_1 = X_copy["lag_1"].median()

            # Calculate Rolling 7 on training data
            X_copy["rolling_7"] = X_copy.groupby(["country", "store", "product"])[self.target_column].rolling(7, min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True)
            self.median_rolling_7 = X_copy["rolling_7"].median()

        logger.info("FeatureEnrichmentTransformer fitted successfully.")
        return self

    def transform(self, X):
        """
        Transform method applies all feature engineering steps using fitted statistics.
        """
        X_transformed = X.copy()
        
        # --- 1. Date Features ---
        X_transformed['date'] = pd.to_datetime(X_transformed['date'], errors='coerce')
        X_transformed.dropna(subset=['date'], inplace=True)
        
        X_transformed["year"] = X_transformed["date"].dt.year
        X_transformed["month"] = X_transformed["date"].dt.month
        X_transformed["day"] = X_transformed["date"].dt.day
        X_transformed["weekday"] = X_transformed["date"].dt.weekday
        X_transformed["weekofyear"] = X_transformed["date"].dt.isocalendar().week.astype(int)

        # --- 2. Holiday Features ---
        def is_holiday(row):
            try:
                country_code = row["country"]
                date = row["date"]
                country_holidays = holidays.country_holidays(country_code)
                return int(date in country_holidays)
            except Exception:
                return 0

        X_transformed["is_holiday"] = X_transformed.apply(is_holiday, axis=1)

        # --- 3. Enrich with GDP ---
        if self.gdp_data is not None and not self.gdp_data.empty:
            X_transformed = X_transformed.merge(self.gdp_data, on=["country", "year"], how="left")
            X_transformed["gdp_per_capita"] = X_transformed["gdp_per_capita"].fillna(self.median_gdp)
        else:
             # If GDP fetch failed during fit, create a zero column
             X_transformed["gdp_per_capita"] = 0.0

        # --- 4. Lag and Rolling Features (Only possible if target/historical data is present in X) ---
        # NOTE: For live inference, these must be passed in the input data or derived from a feature store.
        # Here we rely on the raw input (X) potentially containing 'num_sold' or we use the median from fit.
        
        # Recalculate or use median fallback for lag/rolling (crucial for prediction data)
        X_transformed = X_transformed.sort_values(["country", "store", "product", "date"])

        # Create dummy lag column if the target column is missing (i.e., prediction time)
        lag_source = self.target_column if self.target_column in X_transformed.columns else 'num_sold_proxy'
        if lag_source not in X_transformed.columns:
             # Use a temporary proxy column for grouping/shifting if target is missing
             X_transformed[lag_source] = self.median_lag_1 if self.median_lag_1 is not None else 0
        
        X_transformed["lag_1"] = X_transformed.groupby(["country", "store", "product"])[lag_source].shift(1).fillna(self.median_lag_1 if self.median_lag_1 is not None else 0)
        X_transformed["rolling_7"] = X_transformed.groupby(["country", "store", "product"])[lag_source].rolling(7, min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True).fillna(self.median_rolling_7 if self.median_rolling_7 is not None else 0)
        
        if 'num_sold_proxy' in X_transformed.columns:
             X_transformed = X_transformed.drop(columns=['num_sold_proxy'])

        logger.info("FeatureEnrichmentTransformer transformed data successfully.")
        return X_transformed.drop(columns=['date', 'year'], errors='ignore') # Drop helper columns