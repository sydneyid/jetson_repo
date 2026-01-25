# Structural Optimizations for GC-RANSAC Pipeline

## Current Pipeline Bottlenecks

The GC-RANSAC pipeline is: **Sample → Estimate → Count Inliers → Graph-Cut → Refine → Compare**

Here are the main bottlenecks and optimizations:

## 1. **Sampling Phase** (Currently Optimized)

**Problem:** Linear search through all points is slow.

**Solution:** 
- Limit search to first 500 points (most candidates found early)
- Use squared distance (avoid sqrt)
- Early exit when enough candidates found
- ✅ **Already implemented**

## 2. **Inlier Counting** (Major Bottleneck)

**Problem:** For every model proposal, GC-RANSAC checks distance of ALL points (7000+ points).

**Current Flow:**
```
For each model proposal:
  1. Check distance of all 7000 points → O(N)
  2. Count inliers
  3. If inliers < threshold, reject
```

**Optimization Ideas:**
- **Early rejection**: Check a small random sample first (e.g., 100 points)
  - If < 5% are inliers, reject immediately (won't meet minimum)
  - Only do full count if sample looks promising
- **Spatial coherence skip**: Since `spatial_coherence_weight=0`, we can skip graph-cut entirely
- **Incremental counting**: Stop counting once we know it won't beat best model

## 3. **Graph-Cut Local Optimization** (Expensive but Currently Disabled)

**Good News:** You have `spatial_coherence_weight=0`, so graph-cut is skipped! ✅

**If you enable it later:**
- Reduce `max_graph_cut_number` from 2 to 1
- Reduce inner RANSAC iterations in graph-cut

## 4. **Model Refinement (Least-Squares)** (Can Be Optimized)

**Problem:** After finding inliers, GC-RANSAC refines model using all inliers.

**Current:** Uses all inliers for least-squares fitting

**Optimization:**
- Limit to first 100-200 inliers (diminishing returns)
- Use iterative refinement only if model is promising

## 5. **PEARL Optimization** (Already Optimized)

**Current:** Runs every 3 models, max 15 iterations
- ✅ **Already optimized**

## 6. **Proposal Engine Settings** (Can Be Further Optimized)

**Current settings:**
```cpp
max_local_optimization_number = 2
max_graph_cut_number = 2
max_least_squares_iterations = 2
min_iteration_number = 10
```

**Further optimizations:**
- Reduce `max_local_optimization_number` to 1 (single LO often sufficient)
- Reduce `max_least_squares_iterations` to 1
- Reduce `min_iteration_number` to 5

## Recommended Structural Changes

### Priority 1: Early Rejection in Inlier Counting

**Add to GC-RANSAC (in `GCRANSAC.h`):**
```cpp
// Before full inlier count, check small sample
if (statistics.iteration_number > 10) {  // After initial iterations
    // Quick check: sample 100 random points
    size_t sample_count = 0;
    for (int i = 0; i < 100; ++i) {
        if (distance < threshold) sample_count++;
    }
    // If < 5% inliers, reject early (won't meet minimum of 15)
    if (sample_count < 5) {
        continue;  // Skip full inlier count
    }
}
```

### Priority 2: Reduce Local Optimization

**In `progressivex_python.cpp`:**
```cpp
settings.proposal_engine_settings.max_local_optimization_number = 1;  // Was 2
settings.proposal_engine_settings.max_least_squares_iterations = 1;   // Was 2
settings.proposal_engine_settings.min_iteration_number = 5;           // Was 10
```

### Priority 3: Skip Graph-Cut When Not Needed

**Already done** (spatial_coherence_weight=0), but verify it's actually skipped.

### Priority 4: Limit Inlier Set for Refinement

**In GC-RANSAC local optimization:**
```cpp
// Limit inliers used for refinement
const size_t max_refinement_inliers = 200;
if (inliers.size() > max_refinement_inliers) {
    // Randomly sample max_refinement_inliers for refinement
    // (diminishing returns with more points)
}
```

## Expected Speedup

- **Early rejection**: 2-3x faster (skip bad models quickly)
- **Reduced LO iterations**: 1.5x faster
- **Limited refinement**: 1.2x faster
- **Combined**: **3-5x overall speedup**

## Implementation Order

1. ✅ Revert neighborhood graph change (done)
2. ✅ Optimize sampler (limit search, early exit) (done)
3. ⏭️ Add early rejection in inlier counting
4. ⏭️ Reduce local optimization iterations
5. ⏭️ Limit refinement inlier set
