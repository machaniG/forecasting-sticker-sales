#!/bin/bash

# Entry point script for the Docker container
# Handles different operations: train, predict, or full pipeline

case "$1" in
  "train")
    echo "Running training pipeline..."
    python scripts/etl1.py && python scripts/train1.py
    ;;
  "predict")
    echo "Running predictions..."
    if [ -z "$2" ]; then
      echo "Error: Input file path required for predictions"
      exit 1
    fi
    python scripts/predict.py "$2"
    ;;
  "pipeline")
    echo "Running full pipeline (ETL + Training + Predictions)..."
    python scripts/etl1.py && python scripts/train1.py && \
    python scripts/predict.py "data/processed/cleaned.csv"
    ;;
  *)
    echo "Usage: $0 {train|predict|pipeline} [input_file_for_predictions]"
    exit 1
    ;;
esac