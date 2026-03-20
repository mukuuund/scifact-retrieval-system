#!/bin/bash
# Azure App Service sets the PORT environment variable. Streamlit by default uses 8501.
# We map it to $PORT if specified, otherwise default to 8000 for Azure Linux Web App.
if [ -z "$PORT" ]; then
  PORT=8000
fi

python -m streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0
