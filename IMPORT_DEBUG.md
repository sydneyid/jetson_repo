# Debugging pyprogressivex Import Issue

## Problem
The `pyprogressivex` module crashes on import with exit code 137 (process killed), indicating a segfault or memory issue.

## Status
- ✓ Module is installed (`pip list` shows `pyprogressivex`)
- ✓ `.so` files exist in both `build/` and `src/pyprogressivex/`
- ✗ Import crashes immediately on `import pyprogressivex`

## Possible Causes

1. **C++ Library Dependencies**: The module depends on:
   - `libGraphCutRANSAC.dylib` (found via `@rpath`)
   - OpenCV libraries (found in `/opt/homebrew/opt/opencv/lib/`)
   - Possibly missing: GFlags, GLog, Eigen

2. **Python Version Mismatch**: 
   - Python 3.12 is being used
   - There's also a `.so` file for Python 3.10
   - The module might have been built for a different Python version

3. **Initialization Code**: The C++ initialization might be crashing during module load

## Solutions to Try

### 1. Check Dependencies
```bash
# Check if all required libraries are available
otool -L src/pyprogressivex/pyprogressivex.cpython-312-darwin.so
ldd src/pyprogressivex/pyprogressivex.cpython-312-darwin.so  # Linux
```

### 2. Rebuild with Correct Python
```bash
# Activate conda environment
conda activate progressivex  # or base

# Clean and rebuild
cd build
rm -rf *
cmake ..
make

# Reinstall
cd ..
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
pip install -e . --force-reinstall --no-deps
```

### 3. Check Environment
```bash
# Ensure you're using the conda Python
which python3
python3 --version

# Try with conda Python directly
/opt/anaconda3/bin/python3 -c "import pyprogressivex"
```

### 4. Use Python 3.10
If Python 3.10 is available:
```bash
conda install python=3.10
# Then rebuild
```

### 5. Check System Logs
```bash
# macOS crash logs
log show --predicate 'process == "Python"' --last 5m

# Or check Console.app for crash reports
```

### 6. Test with Minimal Code
Try importing in a minimal script to isolate the issue:
```python
import sys
print("Python:", sys.version)
import pyprogressivex
```

## Created Files

I've created Python scripts from the Jupyter notebooks:

1. **`examples/example_multi_lines.py`** - Line detection example
2. **`examples/example_multi_homography.py`** - Homography detection example
3. **`test_import.py`** - Simple import test script

These scripts will work once the import issue is resolved.

## Next Steps

1. Try rebuilding the module with the correct Python version
2. Check if all dependencies (GFlags, GLog, Eigen) are installed
3. Verify the conda environment is properly activated
4. Check system crash logs for more details

Once the import works, you can run:
```bash
cd examples
python3 example_multi_lines.py
python3 example_multi_homography.py
```
