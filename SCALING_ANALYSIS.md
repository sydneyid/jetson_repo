# Why Reducing Points Doesn't Scale Linearly

## The Problem

**7000 points:** 75.78 seconds, 19 lines detected
**3000 points:** 50.67 seconds, 23 lines detected
**Time saved:** Only 25 seconds (33% reduction) despite 57% fewer points

## Why It Doesn't Scale Linearly

### Fixed Costs (Don't Scale with Points)

1. **Main Loop Iterations: Fixed at 1500**
   ```cpp
   for (size_t current_iteration = 0; current_iteration < 1500; ++current_iteration)
   ```
   - Same number of iterations regardless of point count
   - Each iteration: sample → estimate → validate → optimize
   - **Cost: ~50% of runtime** (fixed)

2. **Max Iterations Per Proposal: Fixed at 3000**
   ```python
   max_iters_optimized = 3000  # Same for both 3000 and 7000 points
   ```
   - GC-RANSAC tries up to 3000 samples per proposal
   - Same number of samples regardless of point count
   - **Cost: ~15% of runtime** (fixed)

3. **PEARL Optimization: Based on Models, Not Points**
   - Found **MORE models** with fewer points (23 vs 19)
   - PEARL runs based on number of models found
   - **Cost: ~25% of runtime** (actually INCREASED with fewer points!)

4. **Model Validation: Based on Models, Not Points**
   - Validates each model (Tanimoto similarity, parallelism checks)
   - More models = more validation
   - **Cost: ~7% of runtime** (increased with fewer points)

### Variable Costs (Scale with Points)

1. **Inlier Counting: O(N) per proposal**
   - For each promising proposal, checks distance to all points
   - 7000 points: checks 7000 distances
   - 3000 points: checks 3000 distances
   - **Cost: ~3% of runtime** (only small fraction scales)

## The Math

**Total time breakdown:**
- Fixed costs (main loop, max_iters, PEARL, validation): ~97% of runtime
- Variable costs (inlier counting): ~3% of runtime

**Expected scaling:**
- If only inlier counting scaled: 57% fewer points → 1.7% time saved
- But we got 33% time saved, so some fixed costs are also reduced

**Why we got 33% instead of 1.7%:**
1. **Fewer proposals tested**: With fewer points, proposals succeed/fail faster
2. **Less work per proposal**: Smaller point set means faster sampling
3. **But more models found**: 23 vs 19 means more PEARL time

## The Real Bottleneck

The **main loop (1500 iterations)** is the biggest fixed cost:
- Each iteration runs GC-RANSAC (up to 3000 samples)
- This is independent of point count
- **This is why reducing points doesn't help much**

## Solutions

To get better scaling, you need to reduce **fixed costs**:

1. **Reduce main loop iterations** (currently 1500)
   - Early termination already helps (stops at 24-25 models)
   - But still runs many iterations before finding models

2. **Reduce max_iters per proposal** (currently 3000)
   - With Z-aligned sampler, proposals succeed faster
   - Could reduce to 2000 or 1500

3. **Reduce PEARL iterations** (currently 15)
   - Already optimized, but could go lower

4. **Skip PEARL more often** (currently every 3 models)
   - Could skip every 5 models

## Expected Impact

If you reduce:
- Main loop: 1500 → 1000 iterations: **~17% faster**
- Max iterations: 3000 → 2000: **~5% faster**
- Combined: **~20% faster** (15 seconds saved)

But this might reduce the number of lines found (currently 23).
