#!/usr/bin/env bash
echo "========================================================"
echo "Starting SignaFlow AI on macOS / Linux..."
echo "========================================================"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    streamlit run app.py
else
    streamlit run app.py
fi
