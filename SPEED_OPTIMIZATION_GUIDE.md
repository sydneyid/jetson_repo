# Progressive-X Speed Optimization Guide

## Why Progressive-X is Slow

Progressive-X has several computational bottlenecks:

### 1. **GC-RANSAC Iterations** (Main Bottleneck)
- **Low confidence (0.05)** requires many RANSAC iterations to be confident
- Each iteration can run up to `max_iters` (10,000 in optimized version)
- **Solution**: Increase confidence moderately (0.20-0.30) to reduce iterations while still finding models

### 2. **Local Optimization** (Secondary Bottleneck)
- Each RANSAC iteration can do up to 50 local optimizations (default)
- Local optimization is expensive (refits models, checks inliers)
- **Solution**: Reduce `max_local_optimization_number` from 50 to 10-20

### 3. **PEARL Optimization** (After Each Model)
- Runs after each model is added to optimize labeling
- Can run up to 100 iterations per model
- Uses graph cuts (alpha-expansion) which is expensive
- **Solution**: This is hardcoded in PEARL, but happens less frequently with fewer models

### 4. **Main Progressive-X Loop**
- Can run up to 100 iterations
- Each iteration: GC-RANSAC → validation → PEARL → update
- **Solution**: Reduce `max_proposal_number_without_change` from 100 to 50

### 5. **Neighborhood Graph Construction**
- FLANN-based neighborhood graph built once
- Usually fast, but can be slow for very large datasets
- **Solution**: Already optimized with downsampling for >500k points

## Current Optimizations Applied

1. ✅ **Reduced max_iters**: 2M → 10k-100k (scaled by data size)
2. ✅ **Reduced max_local_optimization_number**: 50 → 10 (in C++)
3. ✅ **Moderate confidence**: 0.05 → 0.20 (balances speed vs detection)
4. ✅ **NAPSAC sampler**: Faster than uniform for spatial clustering
5. ✅ **Downsampling**: For datasets >500k points
6. ✅ **Reduced max_proposal_number_without_change**: 100 → 50 (in C++)

## Trade-offs

- **Higher confidence (0.50+)**: Faster but finds fewer models
- **Lower confidence (0.05)**: Slower but finds more models
- **Moderate confidence (0.20-0.30)**: Good balance

## Further Optimizations (If Needed)

1. **Reduce PEARL max iterations**: Modify `maximum_iteration_number` in PEARL.h (default 100)
2. **Reduce main loop iterations**: Modify max iterations in progressive_x.h (default 100)
3. **Use GridNeighborhoodGraph**: Faster than FLANN for regular grids
4. **Parallelize**: GC-RANSAC iterations could be parallelized (requires code changes)

## Expected Speedup

With current optimizations:
- **Before**: ~45-60 seconds for 7k points
- **After**: Should be 30-40% faster
- **Trade-off**: May find slightly fewer models (but still many)
