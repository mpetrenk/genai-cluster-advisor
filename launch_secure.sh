#!/bin/bash

# Cluster Recommendation Wizard - CrowdStrike Compatible Launch Script

echo "🧙‍♂️ Starting Cluster Recommendation Wizard (CrowdStrike Compatible)..."
echo "=================================================================="

# Function to check if CrowdStrike is running
check_crowdstrike() {
    if pgrep -f "falcon" > /dev/null 2>&1; then
        echo "🛡️  CrowdStrike Falcon detected - using compatible mode"
        return 0
    else
        echo "ℹ️  CrowdStrike not detected"
        return 1
    fi
}

# Function to setup frpc file safely
setup_frpc() {
    echo "🔧 Setting up Gradio sharing components..."
    
    FRPC_DIR="/Users/maxpetre/.cache/huggingface/gradio/frpc"
    FRPC_FILE="$FRPC_DIR/frpc_darwin_arm64_v0.3"
    
    # Create directory if it doesn't exist
    mkdir -p "$FRPC_DIR"
    
    # Check if file exists and is executable
    if [ -f "$FRPC_FILE" ] && [ -x "$FRPC_FILE" ]; then
        echo "✅ frpc file already configured"
        return 0
    fi
    
    # Download frpc file if missing
    if [ ! -f "$FRPC_FILE" ]; then
        echo "📥 Downloading frpc component..."
        
        # Try to download with curl
        if command -v curl > /dev/null 2>&1; then
            curl -L "https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_darwin_arm64" \
                 -o "$FRPC_FILE" \
                 --connect-timeout 30 \
                 --max-time 300 \
                 --retry 3 \
                 --retry-delay 5
        else
            echo "❌ curl not found - cannot download frpc"
            return 1
        fi
    fi
    
    # Make executable
    if [ -f "$FRPC_FILE" ]; then
        chmod +x "$FRPC_FILE"
        echo "✅ frpc configured successfully"
        return 0
    else
        echo "❌ Failed to setup frpc"
        return 1
    fi
}

# Function to check Python dependencies
check_dependencies() {
    echo "🔍 Checking Python dependencies..."
    
    python3 -c "
import sys
missing = []
try:
    import gradio
    print('✅ gradio')
except ImportError:
    missing.append('gradio')
    
try:
    import pandas
    print('✅ pandas')
except ImportError:
    missing.append('pandas')
    
try:
    import sklearn
    print('✅ scikit-learn')
except ImportError:
    missing.append('scikit-learn')
    
try:
    import numpy
    print('✅ numpy')
except ImportError:
    missing.append('numpy')

if missing:
    print(f'❌ Missing packages: {missing}')
    sys.exit(1)
else:
    print('✅ All dependencies satisfied')
" || {
        echo "📦 Installing missing dependencies..."
        pip3 install -r requirements.txt
    }
}

# Function to generate dataset if needed
setup_dataset() {
    if [ ! -f "cluster_dataset.csv" ]; then
        echo "📊 Generating synthetic dataset..."
        python3 generate_dataset.py
        if [ $? -eq 0 ]; then
            echo "✅ Dataset generated successfully"
        else
            echo "❌ Failed to generate dataset"
            exit 1
        fi
    else
        echo "✅ Dataset already exists"
    fi
}

# Function to test application before launch
test_application() {
    echo "🧪 Testing application components..."
    
    python3 -c "
from app_improved import ClusterRecommendationWizard
try:
    wizard = ClusterRecommendationWizard()
    print('✅ Application components working')
except Exception as e:
    print(f'❌ Application test failed: {e}')
    exit(1)
" || {
        echo "❌ Application test failed"
        exit 1
    }
}

# Main execution
main() {
    # Check for CrowdStrike
    CROWDSTRIKE_DETECTED=false
    if check_crowdstrike; then
        CROWDSTRIKE_DETECTED=true
    fi
    
    # Setup components
    check_dependencies
    setup_dataset
    
    # Setup frpc (may fail with CrowdStrike)
    FRPC_AVAILABLE=false
    if setup_frpc; then
        FRPC_AVAILABLE=true
    fi
    
    # Test application
    test_application
    
    # Launch application
    echo ""
    echo "🚀 Launching Cluster Recommendation Wizard..."
    echo "🛡️  Security compatibility: ENABLED"
    
    if [ "$FRPC_AVAILABLE" = true ]; then
        echo "📡 Public sharing: AVAILABLE"
        echo "🌐 Generating shareable URL..."
    else
        echo "⚠️  Public sharing: UNAVAILABLE (security restrictions)"
        echo "🌐 Local access only: http://127.0.0.1:7860"
    fi
    
    echo ""
    echo "Starting application..."
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Launch with appropriate settings
    if [ "$CROWDSTRIKE_DETECTED" = true ]; then
        echo "🛡️  Using CrowdStrike-compatible settings"
        # Use environment variables to configure secure mode
        export GRADIO_ANALYTICS_ENABLED=false
        export GRADIO_DEBUG=false
    fi
    
    # Launch the improved application
    python3 app_improved.py
}

# Error handling
set -e
trap 'echo "❌ Script interrupted"; exit 1' INT TERM

# Change to script directory
cd "$(dirname "$0")"

# Run main function
main

echo "👋 Application stopped"
