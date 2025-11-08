import pytest
import pandas as pd
from api import prepare_features, load_model
from pathlib import Path

def test_prepare_features():
    df = pd.DataFrame({
        "country": ["US"],
        "store": ["Store_1"],
        "product": ["Sticker_X"],
        "date": ["2025-11-07"],
        "gdp_per_capita": [65000.0]
    })
    result = prepare_features(df.copy())
    assert "year" in result.columns
    assert "month" in result.columns
    assert "weekday" in result.columns
    assert "weekofyear" in result.columns
    assert result.loc[0, "year"] == 2025
    assert result.loc[0, "month"] == 11
    assert result.loc[0, "weekday"] == 4

def test_load_model():
    # Should not raise if model exists, else skip
    model_path = Path("artifacts/xgb_pipeline.joblib")
    if model_path.exists():
        model = load_model(model_path)
        assert model is not None
    else:
        pytest.skip("Trained model not found; skipping load_model test.")
