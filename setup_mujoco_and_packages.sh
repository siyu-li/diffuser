#!/bin/bash
# Complete setup script for MuJoCo, mujoco-py, d4rl, and doodad
# Run this AFTER creating the conda environment

set -e  # Exit on error

echo "=========================================="
echo "Setting up MuJoCo and related packages"
echo "=========================================="

# Make sure we're in the diffuser environment
if [[ "$CONDA_DEFAULT_ENV" != "diffuser" ]]; then
    echo "Error: Please activate the diffuser environment first:"
    echo "  conda activate diffuser"
    exit 1
fi

# Step 1: Install MuJoCo
echo ""
echo "Step 1/5: Installing MuJoCo 2.1.0 (free version)..."
mkdir -p ~/.mujoco
cd ~/.mujoco

# Download MuJoCo 2.1.0 (free version, no license required)
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
    echo "Downloading MuJoCo 2.1.0..."
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz
    tar -xzf mujoco210.tar.gz
    rm mujoco210.tar.gz
    echo "✓ MuJoCo 2.1.0 installed"
else
    echo "✓ MuJoCo already installed"
fi

# Step 2: Set environment variables
echo ""
echo "Step 2/5: Setting up environment variables..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
echo "✓ Environment variables set"

# Step 3: Install mujoco-py
echo ""
echo "Step 3/5: Installing mujoco-py..."
# mujoco-py 2.1.x works with MuJoCo 2.1.0 and doesn't require a license
pip install mujoco-py==2.1.2.14
echo "✓ mujoco-py installed"

# Step 4: Install doodad
echo ""
echo "Step 4/5: Installing doodad..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
git clone https://github.com/JannerM/doodad.git
cd doodad
git checkout -b janner --track origin/janner
pip install -e . --no-deps
echo "✓ doodad installed"

# Step 5: Install d4rl
echo ""
echo "Step 5/5: Installing d4rl..."
cd "$TEMP_DIR"
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl

# Use the specific commit (uncomment the line below to use master instead)
git checkout f2a05c0d66722499bf8031b094d9af3aea7c372b
# git checkout master

pip install -e . --no-deps
echo "✓ d4rl installed"

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import mujoco_py; print('✓ mujoco-py imported successfully')"
python -c "import doodad; print('✓ doodad imported successfully')"
python -c "import d4rl; print('✓ d4rl imported successfully')"

# Cleanup
echo ""
echo "Cleaning up temporary files..."
cd ~
rm -rf "$TEMP_DIR"

# Add environment variables to bashrc/zshrc
echo ""
echo "Adding MuJoCo to your shell configuration..."
SHELL_RC="$HOME/.bashrc"
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

if ! grep -q "MUJOCO" "$SHELL_RC"; then
    echo "" >> "$SHELL_RC"
    echo "# MuJoCo environment variables" >> "$SHELL_RC"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/.mujoco/mujoco210/bin" >> "$SHELL_RC"
    echo "export MUJOCO_PY_MUJOCO_PATH=\$HOME/.mujoco/mujoco210" >> "$SHELL_RC"
    echo "✓ Added MuJoCo paths to $SHELL_RC"
else
    echo "✓ MuJoCo paths already in $SHELL_RC"
fi

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Source your shell config: source $SHELL_RC"
echo "2. Install the diffuser package: pip install -e ."
echo "3. Try importing d4rl: python -c 'import d4rl'"
