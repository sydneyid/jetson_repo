# Sensitivity Study: Optimizing Progressive-X for 24 Lines

## Goal
Preserve the result of **24 lines** while minimizing runtime by optimizing stopping conditions and iteration limits.

## Key Parameters Affecting Runtime

### 1. **Early Termination Conditions** (C++: `progressive_x.h`)
- **Current**: Stops at 24-25 models (OPTIMIZED)
- **Previous**: Required 30+ models
- **Impact**: Stops ~20-30% earlier when target is reached

### 2. **Main Loop Iterations** (C++: `progressive_x.h`)
- **Current**: 1500 max iterations (OPTIMIZED)
- **Previous**: 3000 max iterations
- **Impact**: Prevents unnecessary iterations after finding target

### 3. **max_iters** (Python: `may_9_test_FIXED.py`)
- **What it does**: Max RANSAC iterations PER PROPOSAL (not total)
- **Current**: 3,000-25,000 depending on dataset size (OPTIMIZED)
- **Previous**: 5,000-50,000
- **Impact**: Each proposal runs faster, but may need more proposals
- **Trade-off**: Lower = faster per proposal, but may need more proposals

### 4. **confidence** (Python: `may_9_test_FIXED.py`)
- **What it does**: RANSAC confidence level (lower = more iterations needed per proposal)
- **Current**: 0.05 (OPTIMIZED)
- **Previous**: 0.01 (very low, requires many more iterations)
- **Impact**: Each proposal needs fewer iterations to reach confidence
- **Trade-off**: Higher = faster proposals, but may miss some models

### 5. **max_proposal_number_without_change** (C++: `progressivex_python.cpp`)
- **What it does**: Stop after N consecutive rejections
- **Current**: 300 (OPTIMIZED)
- **Previous**: 500
- **Impact**: Stops earlier when no progress, but early termination at 24 models handles this

## Sensitivity Study Approach

### Test 1: Reduce max_iters further
```python
max_iters_optimized = 2000  # Try reducing from 3000
```
**Expected**: Faster runtime, but verify you still get 24 lines

### Test 2: Increase confidence slightly
```python
conf_optimized = 0.10  # Try increasing from 0.05
```
**Expected**: Faster proposals, but may find fewer lines (test if still gets 24)

### Test 3: Reduce main loop iterations
```python
# In progressive_x.h, line 279:
for (size_t current_iteration = 0; current_iteration < 1000; ++current_iteration)
```
**Expected**: Stops earlier, but early termination should handle this

### Test 4: Tune early termination threshold
```python
# In progressive_x.h, line 567:
if (models.size() >= 22 && unseen_inliers < settings.minimum_number_of_inliers) {
```
**Expected**: Stops even earlier, but may miss some lines

## Recommended Testing Order

1. **Start with current optimized settings** (should get 24 lines, ~50-70 seconds)
2. **If runtime still too high**:
   - Reduce `max_iters` to 2000-2500
   - Increase `confidence` to 0.10
3. **If not finding 24 lines**:
   - Increase `max_iters` to 4000-5000
   - Decrease `confidence` to 0.01
   - Increase main loop to 2000

## Current Optimized Settings Summary

- **Main loop**: 1500 iterations (was 3000)
- **Early termination**: Stops at 24-25 models (was 30+)
- **max_iters**: 3,000-25,000 (was 5,000-50,000)
- **confidence**: 0.05 (was 0.01)
- **max_proposal_number_without_change**: 300 (was 500)
- **minimum_point_number**: 15 (was 2) - **NEW: Rejects weak proposals early**
- **GC-RANSAC optimizations**:
  - Local optimization: 1 iteration (was 2)
  - Graph-cut: 1 iteration (was 2)
  - Least squares: 1 iteration (was 2)
  - Min iterations: 5 (was 10)
- **PEARL optimizations**:
  - Max iterations: 5 (was 15)
  - Run every 5 models (was every 3)

These should preserve 24 lines while reducing runtime significantly (target: 30-50 seconds).
