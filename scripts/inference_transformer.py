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
    
    # Ensure 'year' is present for the API call filter
    if 'year' not in df.columns:
        logger.warning("GDP fetch failed: 'year' column is missing from input DataFrame.")
        return pd.DataFrame()
        
    country_codes = df["country"].unique().tolist()
    
    # Use max/min to define the time range for the WB API
    min_year = df["year"].min()
    max_year = df["year"].max()
    years = range(min_year, max_year + 1)
    
    # Format years for WB API call
    time_filter = ";".join([f"YR{y}" for y in years])

    # Attempt to use the country names directly
    country_filter = country_codes if country_codes else 'all'
    
    logger.info(f"🌍 Fetching GDP data for {len(country_codes)} countries and years {min_year}-{max_year}...")

    try:
        # Fetch data. WBGAPI will attempt to map country names to codes.
        df_gdp_wide = wb.data.DataFrame(
            indicator, 
            country_filter, 
            time=time_filter, 
            numeric_times=True,
            columns='series', 
            time_filter=time_filter
        )
        
        # WBGAPI returns a complex index/column structure. Simplify it.
        df_gdp = df_gdp_wide.reset_index().rename(columns={"economy": "country", "time": "year"})
        
        # GDP indicator column name
        gdp_col = indicator
        
        # Check if the GDP column exists and rename it
        if gdp_col in df_gdp.columns:
            df_gdp = df_gdp.rename(columns={gdp_col: "gdp_per_capita"})
        else:
            logger.warning(f"GDP indicator column '{gdp_col}' not found in World Bank response.")
            return pd.DataFrame()

        # Keep only the required columns and ensure types match
        df_gdp = df_gdp[["country", "year", "gdp_per_capita"]]
        df_gdp["year"] = df_gdp["year"].astype(int)
        
        logger.info(f"✅ Successfully fetched GDP data for {len(df_gdp['country'].unique())} countries.")
        return df_gdp

    except Exception as e:
        logger.warning(f"⚠️ Warning: Failed to fetch GDP data. Error: {e}")
        return pd.DataFrame()


# === Transformer Class ===

class FeatureEnrichmentTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn Transformer for creating all necessary features during training and inference.
    """
    def __init__(self, target_column="num_sold"):
        self.target_column = target_column
        # Attributes to store summary statistics from training data (for inference fallback)
        self.median_lag_1 = None
        self.median_rolling_7 = None
        self.median_gdp = None

    def _is_holiday(self, row):
        """Helper to determine if a date is a public holiday in a given country."""
        try:
            # NOTE: We use the country name directly as holidays.country_holidays is smart enough
            country_holidays = holidays.country_holidays(row["country"])
            return int(row["date"] in country_holidays)
        except Exception:
            # Fallback for unknown countries
            return 0

    def fit(self, X, y=None):
        """
        Fit the transformer on the training data.
        Calculates time-based features, lag/rolling features, and captures median fallbacks.
        """
        X_fit = X.copy()
        
        # Ensure date is datetime type
        if X_fit[self.target_column].dtype != 'datetime64[ns]':
             X_fit["date"] = pd.to_datetime(X_fit["date"], errors="coerce")
        
        # 1️⃣ Time-based Features (Needed for subsequent steps like GDP)
        X_fit["year"] = X_fit["date"].dt.year # Ensure 'year' is present
        
        # --- Lag and Rolling Features for Median Capture ---
        X_fit = X_fit.sort_values(["country", "store", "product", "date"])
        
        # Lag
        X_fit["lag_1"] = X_fit.groupby(["country", "store", "product"])[self.target_column].shift(1)
        
        # Rolling
        X_fit["rolling_7"] = X_fit.groupby(["country", "store", "product"])[self.target_column].rolling(
            7, min_periods=1
        ).mean().reset_index(level=[0, 1, 2], drop=True)

        # Capture median fallbacks from training data
        self.median_lag_1 = X_fit["lag_1"].median()
        self.median_rolling_7 = X_fit["rolling_7"].median()

        # --- GDP Median Capture (for inference fallback if API fails) ---
        X_gdp = _fetch_gdp_data(X_fit)
        if not X_gdp.empty:
            X_fit = X_fit.merge(X_gdp, on=["country", "year"], how="left")
            self.median_gdp = X_fit["gdp_per_capita"].median()
        else:
            self.median_gdp = 0.0 # Default to 0 if API fails on fit

        logger.info(f"FeatureEnrichmentTransformer fitted. Medians: Lag={self.median_lag_1:.2f}, Rolling={self.median_rolling_7:.2f}, GDP={self.median_gdp:.2f}")

        return self

    def transform(self, X):
        """
        Apply feature engineering to the input DataFrame X.
        """
        X_transformed = X.copy()
        
        # 0️⃣ Clean date column
        X_transformed["date"] = pd.to_datetime(X_transformed["date"], errors="coerce")
        
        # 1️⃣ Time-based Features
        # FIX: Ensure 'year' is created and retained.
        X_transformed["year"] = X_transformed["date"].dt.year
        X_transformed["month"] = X_transformed["date"].dt.month
        X_transformed["day"] = X_transformed["date"].dt.day
        X_transformed["weekday"] = X_transformed["date"].dt.weekday
        
        # The .dt.isocalendar().week returns a non-integer Series in recent pandas, so ensure conversion
        X_transformed["weekofyear"] = X_transformed["date"].dt.isocalendar().week.astype(int)
        
        # 2️⃣ Holiday Features
        X_transformed["is_holiday"] = X_transformed.apply(self._is_holiday, axis=1)

        # 3️⃣ GDP Enrichment
        df_gdp = _fetch_gdp_data(X_transformed)
        if not df_gdp.empty:
            X_transformed = X_transformed.merge(df_gdp, on=["country", "year"], how="left")
            X_transformed["gdp_per_capita"] = X_transformed["gdp_per_capita"].fillna(self.median_gdp if self.median_gdp is not None else 0.0)
        else:
            logger.info("⚠️ No GDP data added during transform (API failed or data empty). Using median fallback.")
            # If API fails, ensure the column still exists and is filled with the training median
            X_transformed["gdp_per_capita"] = self.median_gdp if self.median_gdp is not None else 0.0

        # 4️⃣ Lag and Rolling Features
        X_transformed = X_transformed.sort_values(["country", "store", "product", "date"])

        # Determine the source for lag/rolling calculation: actual target column if present, else use fallback
        lag_source = self.target_column if self.target_column in X_transformed.columns else 'num_sold_proxy'

        # If target column is missing (inference time), create a proxy column initialized with median
        if lag_source not in X_transformed.columns:
             X_transformed[lag_source] = self.median_lag_1 if self.median_lag_1 is not None else 0
        
        # Calculate lag
        X_transformed["lag_1"] = X_transformed.groupby(["country", "store", "product"])[lag_source].shift(1).fillna(self.median_lag_1 if self.median_lag_1 is not None else 0)
        
        # Calculate rolling mean
        X_transformed["rolling_7"] = X_transformed.groupby(["country", "store", "product"])[lag_source].rolling(
            7, min_periods=1
        ).mean().reset_index(level=[0, 1, 2], drop=True).fillna(self.median_rolling_7 if self.median_rolling_7 is not None else 0)
        
        # Drop the proxy column if it was created
        if 'num_sold_proxy' in X_transformed.columns:
             X_transformed = X_transformed.drop(columns=['num_sold_proxy'])

        logger.info(f"FeatureEnrichmentTransformer applied. Shape: {X_transformed.shape}")
        
        return X_transformed