#!/bin/bash
# Script to install d4rl after the conda environment is created
# This avoids setuptools compatibility issues during environment creation

set -e  # Exit on error

echo "Installing d4rl..."

# Make sure we're in the diffuser environment
if [[ "$CONDA_DEFAULT_ENV" != "diffuser" ]]; then
    echo "Error: Please activate the diffuser environment first:"
    echo "  conda activate diffuser"
    exit 1
fi

# Install d4rl without dependencies (since we already have them)
echo "Cloning d4rl repository..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl

echo "Installing d4rl with --no-deps to avoid dependency resolution issues..."
pip install -e . --no-deps

echo "Verifying installation..."
python -c "import d4rl; print('d4rl version:', d4rl.__version__ if hasattr(d4rl, '__version__') else 'installed successfully')"

echo "Cleaning up..."
cd -
rm -rf "$TEMP_DIR"

echo "✓ d4rl installed successfully!"
