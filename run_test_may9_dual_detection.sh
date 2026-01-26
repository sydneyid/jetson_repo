#!/bin/bash
# Shell script to set up and run test_may9_dual_detection.py on Jetson Nano
# This script installs dependencies, builds pyprogressivex, and runs the test

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Progressive-X Setup and Test Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${YELLOW}Warning: This script is designed for Linux (Jetson Nano).${NC}"
    echo -e "${YELLOW}Continuing anyway...${NC}"
    echo ""
fi

# Check Python version
echo -e "${GREEN}[1/7] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python: $PYTHON_VERSION"

# Check if Python 3.8+ is available
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8 or higher is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

# Install system dependencies (for Ubuntu/Debian-based systems like Jetson Nano)
echo ""
echo -e "${GREEN}[2/7] Installing system dependencies...${NC}"
if command -v apt-get &> /dev/null; then
    echo "Installing system packages via apt-get..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        libopencv-dev \
        libeigen3-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        python3-dev \
        python3-pip \
        git \
        pkg-config
    echo -e "${GREEN}✓ System dependencies installed${NC}"
else
    echo -e "${YELLOW}Warning: apt-get not found. Please install dependencies manually:${NC}"
    echo "  - build-essential, cmake"
    echo "  - libopencv-dev, libeigen3-dev"
    echo "  - libgflags-dev, libgoogle-glog-dev"
    echo "  - python3-dev, python3-pip"
    echo ""
    read -p "Press Enter to continue if dependencies are already installed..."
fi

# Install Python dependencies
echo ""
echo -e "${GREEN}[3/7] Installing Python dependencies...${NC}"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Initialize git submodules (graph-cut-ransac)
echo ""
echo -e "${GREEN}[4/7] Initializing git submodules...${NC}"
if [ -d ".git" ]; then
    git submodule update --init --recursive
    echo -e "${GREEN}✓ Git submodules initialized${NC}"
else
    echo -e "${YELLOW}Warning: Not a git repository. If graph-cut-ransac is missing, clone it manually:${NC}"
    echo "  git clone --recursive https://github.com/sydneyid/star_streak_detection.git"
fi

# Build and install pyprogressivex
echo ""
echo -e "${GREEN}[5/7] Building pyprogressivex module...${NC}"
echo "This may take several minutes on Jetson Nano..."

# Set CMAKE_PREFIX_PATH if using conda (optional)
if [ -n "$CONDA_PREFIX" ]; then
    export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
    echo "Using conda environment: $CONDA_PREFIX"
fi

# Build with pip install
python3 -m pip install -e . --verbose
echo -e "${GREEN}✓ pyprogressivex built and installed${NC}"

# Verify installation
echo ""
echo -e "${GREEN}[6/7] Verifying installation...${NC}"
python3 -c "import pyprogressivex; print('✓ pyprogressivex imported successfully')" || {
    echo -e "${RED}✗ Failed to import pyprogressivex${NC}"
    echo "Try rebuilding: pip install -e . --force-reinstall"
    exit 1
}

# Check if test file exists
TEST_FILE="examples/test_may9_dual_detection.py"
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file not found: $TEST_FILE${NC}"
    exit 1
fi

# Check if HDF5 file exists
HDF5_FILE="examples/img/may_9.hdf5"
if [ ! -f "$HDF5_FILE" ]; then
    echo -e "${YELLOW}Warning: HDF5 file not found: $HDF5_FILE${NC}"
    echo "The test will fail if the file is not present."
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to abort..."
fi

# Run the test
echo ""
echo -e "${GREEN}[7/7] Running test_may9_dual_detection.py...${NC}"
echo "=========================================="
cd examples
python3 test_may9_dual_detection.py
EXIT_CODE=$?
cd ..

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Test completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Test failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE
