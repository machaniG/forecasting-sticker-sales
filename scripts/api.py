"""
API Wrapper Module (scripts/api.py)

This file aggregates core functions (like feature preparation and model loading)
and is primarily used by unit tests to maintain a clean import structure.
"""
from scripts.train1 import prepare_features
from scripts.predict import load_production_model as load_model
# NOTE: We now import the load_production_model function from predict.py 
# as the 'load_model' reference for testing purposes.

# The functions 'prepare_features' and 'load_model' are now directly 
# available for import by 'tests/test_core_unit.py' from scripts.api.

# The main FastAPI application is located at scripts.serving_app.