#!/bin/bash

# Streamlit Cluster Recommendation Wizard Launch Script

echo "🚀 Starting Streamlit Cluster Recommendation Wizard..."
echo "======================================================"

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit and dependencies..."
    pip install streamlit pandas numpy scikit-learn boto3
    echo "✅ Installation complete"
fi

# Check if the main file exists
if [ ! -f "streamlit_cluster_wizard.py" ]; then
    echo "❌ streamlit_cluster_wizard.py not found in current directory"
    echo "Please ensure you're in the correct directory"
    exit 1
fi

echo "🧪 Testing components..."
python3 -c "
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
print('✅ All imports successful')
"

if [ $? -ne 0 ]; then
    echo "❌ Component test failed"
    exit 1
fi

echo "✅ Component test passed"
echo ""
echo "🌐 Launching Streamlit application..."
echo "📡 The app will be available at:"
echo "   - Local: http://localhost:8501"
echo "   - Network: http://0.0.0.0:8501 (accessible from other devices)"
echo ""
echo "🔗 For public sharing:"
echo "   1. Deploy to Streamlit Cloud (free): https://share.streamlit.io"
echo "   2. Or use ngrok for temporary public URLs"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Launch Streamlit with public access configuration
streamlit run streamlit_cluster_wizard.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
