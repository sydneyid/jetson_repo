# Setup Instructions for Jetson Nano

This guide explains how to set up and run `test_may9_dual_detection.py` on a Jetson Nano running Linux.

## Quick Start

The easiest way to set up and run the test is using the provided shell script:

```bash
./run_test_may9_dual_detection.sh
```

This script will:
1. Check Python version
2. Install system dependencies (CMake, OpenCV, Eigen, GFlags, GLog, etc.)
3. Install Python dependencies from `requirements.txt`
4. Initialize git submodules (graph-cut-ransac)
5. Build and install the `pyprogressivex` module
6. Run the test script

## Manual Setup

If you prefer to set up manually or the script fails, follow these steps:

### 1. Install System Dependencies

On Ubuntu/Debian (Jetson Nano typically runs Ubuntu):

```bash
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
```

### 2. Install Python Dependencies

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

### 3. Initialize Git Submodules

If you cloned the repository, initialize submodules:

```bash
git submodule update --init --recursive
```

If the repository doesn't have the `graph-cut-ransac` submodule, you may need to clone it:

```bash
git clone https://github.com/danini/graph-cut-ransac.git graph-cut-ransac
```

### 4. Build and Install pyprogressivex

```bash
# If using conda, set CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"  # Optional, only if using conda

# Build and install
pip install -e .
```

### 5. Verify Installation

```bash
python3 -c "import pyprogressivex; print('Success!')"
```

### 6. Run the Test

```bash
cd examples
python3 test_may9_dual_detection.py
```

## Requirements

### System Requirements
- **OS**: Linux (Ubuntu/Debian recommended for Jetson Nano)
- **Python**: 3.8 or higher
- **CMake**: 3.5 or higher (3.20+ recommended)
- **C++ Compiler**: GCC with C++17 support
- **OpenCV**: 3.0 or higher
- **Eigen**: 3.0 or higher
- **GFlags**: Latest version
- **GLog**: Latest version

### Python Dependencies
See `requirements.txt` for the complete list. Main dependencies:
- numpy >= 1.20.0
- scipy >= 1.7.0
- h5py >= 3.0.0
- matplotlib >= 3.3.0
- pybind11 >= 2.9.0
- setuptools >= 60.0.0
- wheel >= 0.37.0

## Troubleshooting

### Build Fails

1. **Missing dependencies**: Ensure all system packages are installed
   ```bash
   sudo apt-get install -y build-essential cmake libopencv-dev libeigen3-dev libgflags-dev libgoogle-glog-dev
   ```

2. **CMake can't find packages**: Set CMAKE_PREFIX_PATH
   ```bash
   export CMAKE_PREFIX_PATH="$CONDA_PREFIX"  # If using conda
   # Or set to where packages are installed
   ```

3. **Clean rebuild**: Sometimes a clean rebuild helps
   ```bash
   rm -rf build/ src/pyprogressivex/*.so
   pip install -e . --force-reinstall
   ```

### Import Error

If `import pyprogressivex` fails:

1. **Check if module was built**:
   ```bash
   ls -la src/pyprogressivex/*.so
   ```

2. **Rebuild**:
   ```bash
   pip install -e . --force-reinstall --no-deps
   ```

3. **Check Python version**: Ensure you're using the same Python version that was used to build

### Missing HDF5 File

The test requires `examples/img/may_9.hdf5`. If missing:
- Check if the file exists in the repository
- Download it from the repository if available
- The test will fail with a clear error message if the file is missing

### Jetson Nano Specific Notes

- **Build time**: Building pyprogressivex on Jetson Nano can take 10-30 minutes depending on the model
- **Memory**: Ensure you have enough swap space if building fails due to memory
- **CUDA**: Not required for this test, but OpenCV should be built with CUDA support if available

## Repository

GitHub: https://github.com/sydneyid/star_streak_detection

## License

See LICENSE file in the repository.
