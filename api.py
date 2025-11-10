"""
API Wrapper Module (api.py)

This file acts as a simple module to export core functions used 
by unit tests (tests/test_core_unit.py) from their original locations 
(e.g., train1.py) into the root namespace.

The main serving application is now in serving_app.py.
"""
from scripts.train1 import prepare_features
from scripts.predict import pipeline as load_model 
# NOTE: We use the pipeline object instantiated in predict.py 
# as the 'load_model' reference for testing purposes.
# If predict.py only defined a function, we would import that function.

# The functions 'prepare_features' and 'load_model' are now directly 
# available for import by 'tests/test_core_unit.py'.