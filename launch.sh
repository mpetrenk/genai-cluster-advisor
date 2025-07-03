#!/bin/bash

# Cluster Recommendation Wizard Launch Script

echo "🧙‍♂️ Starting Cluster Recommendation Wizard..."
echo "================================================"

# Check if dataset exists, generate if not
if [ ! -f "cluster_dataset.csv" ]; then
    echo "📊 Generating synthetic dataset..."
    python generate_dataset.py
    echo "✅ Dataset generated successfully!"
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import gradio, pandas, sklearn, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required packages..."
    pip install -r requirements.txt
fi

echo "🚀 Launching application..."
echo "📡 Generating publicly shareable URL..."
echo ""
echo "The application will be available at:"
echo "- Local: http://127.0.0.1:7860"
echo "- Public: [URL will be displayed below]"
echo ""

# Launch the application
python app.py
