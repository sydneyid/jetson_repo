# Fixes Applied to Match Original Notebooks

## Changes Made

### 1. `example_multi_homography.py`
- **Fixed `draw_labeling` function**: Now matches original notebook exactly
  - Modifies images in place (doesn't create copies)
  - Uses `range(max(labeling))` exactly as in original
  - Displays images separately with `plt.imshow()` and `plt.figure()`, not in subplots
  - No titles or axis labels (matches original)

### 2. `example_multi_lines.py`
- **Fixed visualization**: Matches original notebook
  - Uses `mask = np.zeros(len(labeling))` exactly as original (even though it's not used)
  - Displays with `plt.imshow()` and `plt.figure()` separately
  - No titles or subplots

## Key Differences from Original

The original notebooks have a potential bug: `range(max(labeling))` excludes the maximum label value. However, to match the exact output, the scripts now use the same code.

If you want to include ALL labels (including the maximum), change:
```python
for label in range(max(labeling)):
```
to:
```python
for label in range(max(labeling) + 1):
```

## Testing

Run the scripts and compare with the original notebook outputs:
```bash
cd examples
python3 example_multi_homography.py
python3 example_multi_lines.py
```

The output should now match the GitHub notebook examples.
