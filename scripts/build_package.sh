#!/usr/bin/env bash
# Script to build NablaFX distributions for PyPI

set -e  # Exit on error

echo "======================================"
echo "Building NablaFX distribution packages"
echo "======================================"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous build artifacts..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf nablafx.egg-info/

echo "✅ Cleaned"
echo ""

# Install/upgrade build tools
echo "📦 Ensuring build tools are up to date..."
python -m pip install --upgrade pip
python -m pip install --upgrade build twine

echo "✅ Build tools ready"
echo ""

# Build the package
echo "🔨 Building source distribution and wheel..."
python -m build

echo "✅ Build complete"
echo ""

# Check the distribution
echo "🔍 Checking distribution..."
python -m twine check dist/*

echo "✅ Distribution check passed"
echo ""

# List the built files
echo "📋 Built packages:"
ls -lh dist/

echo ""
echo "======================================"
echo "✅ Build successful!"
echo "======================================"
echo ""
echo "⚠️  IMPORTANT: NablaFX requires a two-step installation!"
echo ""
echo "Users must install in this order:"
echo "  1. pip install torch==1.7.1 rational-activations==0.2.0"
echo "  2. pip install nablafx"
echo "  3. python -m nablafx.scripts.post_install  # Copy config file"
echo ""
echo "See INSTALL.md for detailed installation instructions."
echo ""
echo "Next steps for package maintainer:"
echo "  1. Test installation in a clean environment:"
echo "     ./scripts/test_install.sh"
echo ""
echo "  2. Upload to TestPyPI (for testing):"
echo "     python -m twine upload --repository testpypi dist/*"
echo ""
echo "  3. Upload to PyPI (production):"
echo "     python -m twine upload dist/*"
