#!/usr/bin/env bash

set -e

if [ ! -f model/model.joblib ]; then
    echo "No trained model found."
    if [ -f data/paysim.csv ]; then
        echo "Training on data/paysim.csv ..."
        python -m src.train --data data/paysim.csv --out model
    else
        echo "No data/paysim.csv either -- training on synthetic data instead"
        echo "(download the real PaySim CSV and rerun for real results: https://www.kaggle.com/datasets/ealaxi/paysim1)"
        EXIT
    fi
fi

echo "Starting API on http://localhost:8000 ..."
uvicorn api.main:app --port 8000 &
API_PID=$!
trap "kill $API_PID 2>/dev/null" EXIT

sleep 2
echo "Starting dashboard on http://localhost:8501 ..."
streamlit run app/dashboard.py
