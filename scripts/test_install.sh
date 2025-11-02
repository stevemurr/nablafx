#!/usr/bin/env bash
# Script to test NablaFX installation in a clean environment

set -e  # Exit on error

echo "========================================="
echo "Testing NablaFX installation"
echo "========================================="
echo ""

# Create clean test environment
TEST_ENV="test_nablafx_env"

echo "🧪 Creating clean test environment: $TEST_ENV"
rm -rf "$TEST_ENV"
python -m venv "$TEST_ENV"

echo "✅ Test environment created"
echo ""

# Activate environment
echo "🔌 Activating test environment..."
source "$TEST_ENV/bin/activate"

echo "✅ Environment activated"
echo ""

# Install the package
echo "📦 Installing NablaFX with proper installation order..."
if [ ! -d "dist" ] || [ -z "$(ls -A dist/*.whl 2>/dev/null)" ]; then
    echo "❌ Error: No wheel file found in dist/"
    echo "Please run ./scripts/build_package.sh first"
    deactivate
    rm -rf "$TEST_ENV"
    exit 1
fi

# Step 1: Install NablaFX without rational-activations
echo "Step 1/3: Installing NablaFX with all dependencies..."
pip install --upgrade pip
pip install dist/nablafx-*.whl

echo "✅ NablaFX installed"
echo ""

# Step 2: Install rational-activations separately (ignoring its torch dependency)
echo "Step 2/3: Installing rational-activations (ignoring torch==1.7.1 requirement)..."
pip install rational-activations==0.2.0 --no-deps

echo "✅ Rational activations installed (works with torch>=2.2.2)"
echo ""

# Step 3: Copy rational config
echo "Step 3/3: Setting up rational config..."
if [ -f "weights/rationals_config.json" ]; then
    RATIONAL_DIR=$(python -c "import rational; import os; print(os.path.dirname(rational.__file__))")
    cp weights/rationals_config.json "$RATIONAL_DIR/rationals_config.json"
    echo "✅ Rational config copied"
else
    echo "⚠️  Warning: weights/rationals_config.json not found"
    echo "You may need to copy this file manually"
fi

echo ""

# Test import
echo "🧪 Testing imports..."
python -c "
import nablafx
print(f'✅ NablaFX version: {nablafx.__version__}')

# Test core imports
from nablafx.core import GreyBoxModel, BlackBoxModel
print('✅ Core models imported')

# Test processors
from nablafx.processors import Gain, ParametricEQ
print('✅ Processors imported')

# Test controllers
from nablafx.controllers import DynamicController
print('✅ Controllers imported')

# Test data
from nablafx.data import PluginDataset
print('✅ Data modules imported')

# Test evaluation
from nablafx.evaluation import FlexibleLoss
print('✅ Evaluation modules imported')

# Test callbacks
from nablafx.callbacks import AudioLoggingCallback
print('✅ Callbacks imported')

print('')
print('✅ All imports successful!')
"

echo ""
echo "📊 Installed package info:"
pip show nablafx

echo ""
echo "========================================="
echo "✅ Installation test passed!"
echo "========================================="
echo ""

# Cleanup
echo "🧹 Cleaning up test environment..."
deactivate
rm -rf "$TEST_ENV"

echo "✅ Cleanup complete"
echo ""
echo "The package is ready for distribution! 🎉"
