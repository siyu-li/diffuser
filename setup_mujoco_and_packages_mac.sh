#!/bin/bash
# Complete setup script for MuJoCo, mujoco-py, d4rl, and doodad for macOS
# Run this AFTER creating the conda environment

set -e  # Exit on error

echo "=========================================="
echo "Setting up MuJoCo and related packages for macOS"
echo "=========================================="

# Step 1: Install MuJoCo
echo ""
echo "Step 1/6: Installing MuJoCo 2.1.0 (free version)..."
mkdir -p ~/.mujoco
cd ~/.mujoco

# Download MuJoCo 2.1.0 for macOS (free version, no license required)
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
    echo "Downloading MuJoCo 2.1.0 for macOS..."
    
    # Use curl instead of wget (curl is pre-installed on macOS)
    if [[ $(uname -m) == 'arm64' ]]; then
        # For Apple Silicon (M1/M2/M3)
        echo "Detected Apple Silicon (ARM64)"
        curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-arm64.tar.gz -o mujoco210.tar.gz
    else
        # For Intel Macs
        echo "Detected Intel Mac (x86_64)"
        curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz -o mujoco210.tar.gz
    fi
    
    tar -xzf mujoco210.tar.gz
    rm mujoco210.tar.gz
    echo "✓ MuJoCo 2.1.0 installed"
else
    echo "✓ MuJoCo already installed"
fi

# Step 2: Set environment variables for macOS
echo ""
echo "Step 2/6: Setting up environment variables..."
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
echo "✓ Environment variables set"

# Step 3: Install mujoco-py
echo ""
echo "Step 3/6: Installing mujoco-py..."
# mujoco-py 2.1.x works with MuJoCo 2.1.0 and doesn't require a license
pip install mujoco-py==2.1.2.14
echo "✓ mujoco-py installed"

# Step 4: Install doodad
echo ""
echo "Step 4/6: Installing doodad..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
git clone https://github.com/JannerM/doodad.git
cd doodad
git checkout -b janner --track origin/janner
pip install -e . --no-deps
echo "✓ doodad installed"

# Step 5: Install d4rl
echo ""
echo "Step 5/6: Installing d4rl..."
cd "$TEMP_DIR"
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl

# Use the specific commit (uncomment the line below to use master instead)
git checkout f2a05c0d66722499bf8031b094d9af3aea7c372b
# git checkout master

pip install -e . --no-deps
echo "✓ d4rl installed"

# Step 6: Install additional d4rl dependencies
echo ""
echo "Step 6/6: Installing additional dependencies..."
pip install h5py dm_control termcolor pybullet
pip install git+https://github.com/aravindr93/mjrl@master#egg=mjrl
echo "✓ Additional dependencies installed"

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

# Add environment variables to zshrc (macOS default shell)
echo ""
echo "Adding MuJoCo to your shell configuration..."
SHELL_RC="$HOME/.zshrc"
if [ -f "$HOME/.bash_profile" ]; then
    # Also add to bash_profile if it exists
    if ! grep -q "MUJOCO" "$HOME/.bash_profile"; then
        echo "" >> "$HOME/.bash_profile"
        echo "# MuJoCo environment variables" >> "$HOME/.bash_profile"
        echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:\$HOME/.mujoco/mujoco210/bin" >> "$HOME/.bash_profile"
        echo "export MUJOCO_PY_MUJOCO_PATH=\$HOME/.mujoco/mujoco210" >> "$HOME/.bash_profile"
    fi
fi

if ! grep -q "MUJOCO" "$SHELL_RC"; then
    echo "" >> "$SHELL_RC"
    echo "# MuJoCo environment variables" >> "$SHELL_RC"
    echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:\$HOME/.mujoco/mujoco210/bin" >> "$SHELL_RC"
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
echo ""
echo "Note: If you get any OpenGL errors when running, you may need to install additional dependencies."
echo "See: https://github.com/openai/mujoco-py#macos"
