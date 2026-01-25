# What Each Step Does in GC-RANSAC and Their Value

## The Pipeline Flow

```
Sample (2 points) 
  → Estimate line from 2 points
    → Count inliers (all 7000 points)
      → Local Optimization (if promising)
        → Graph-Cut (refine inliers)
          → Inner RANSAC (on refined inliers)
            → Least Squares (fit to all inliers)
              → Compare score
```

## 1. **Local Optimization** (The Wrapper)

**What it does:**
- It's the **outer loop** that wraps graph-cut and inner RANSAC
- Runs multiple times (we set `max_local_optimization_number = 1`)
- Each iteration: runs graph-cut → inner RANSAC → least squares

**Value:**
- ✅ **Necessary** - It's the framework that runs the refinement steps
- But with `max_local_optimization_number = 1`, it only runs once

**Cost:** Low (just a loop wrapper)

## 2. **Graph-Cut** (Spatial Coherence Refinement)

**What it does:**
- Uses **spatial coherence** to refine inlier/outlier labels
- Idea: If a point is an inlier, its **neighbors** are likely inliers too
- Uses graph-cut algorithm to optimize: "Should this point be inlier or outlier?"
- Considers both:
  - Distance to model (data term)
  - Neighbor labels (smoothness term, weighted by `spatial_coherence_weight`)

**Value:**
- ❌ **NOT NEEDED in your case!** You have `spatial_coherence_weight=0.0`
- When `spatial_coherence_weight=0`, graph-cut just does simple thresholding
- It's essentially: "points within threshold = inliers" (same as initial count)
- **Graph-cut is expensive** (builds graph, runs min-cut algorithm)

**Cost:** High (graph construction + min-cut algorithm)

**Recommendation:** Since `spatial_coherence_weight=0`, graph-cut is doing redundant work!

## 3. **Inner RANSAC** (Inside Graph-Cut Loop)

**What it does:**
- After graph-cut refines inliers, runs a **mini RANSAC** on those inliers
- Samples from the refined inlier set
- Estimates new models from those samples
- Finds the best model among candidates

**Value:**
- ⚠️ **Partially useful** - Can find better models from refined inliers
- But if graph-cut didn't help (because `spatial_coherence_weight=0`), this is working with the same inliers as before

**Cost:** Medium (sampling + model estimation)

## 4. **Least Squares** (Final Refinement)

**What it does:**
- Takes all inliers and fits the model using **least-squares**
- More accurate than fitting to just 2 points
- Iterative refinement (we set `max_least_squares_iterations = 1`)

**Value:**
- ✅ **Very valuable!** This significantly improves model accuracy
- Fitting to 20-110 inliers is much better than fitting to 2 points
- This is where the model actually gets refined

**Cost:** Low-Medium (matrix operations, but only on inliers)

## The Problem: Redundant Steps

Since you have `spatial_coherence_weight=0.0`:

1. **Graph-Cut** is doing redundant work:
   - It's just thresholding (same as initial inlier count)
   - But it's expensive (graph construction + min-cut)
   - **Could skip entirely!**

2. **Inner RANSAC** is working with the same inliers:
   - Graph-cut didn't refine them (no spatial coherence)
   - So it's re-sampling from the same set
   - **Less valuable than it could be**

3. **Least Squares** is still valuable:
   - Fits model to all inliers (much better than 2 points)
   - This is the step that actually improves accuracy

## Recommended Optimization

Since `spatial_coherence_weight=0.0`, you can:

### Option 1: Skip Graph-Cut Entirely (Best for Speed)

**Skip the entire local optimization if spatial_coherence_weight=0:**
```cpp
if (settings.spatial_coherence_weight > 0.0) {
    // Run graph-cut local optimization
} else {
    // Skip graph-cut, just do least-squares on initial inliers
    // This is much faster!
}
```

**Speedup:** 3-5x faster (skip expensive graph-cut)

### Option 2: Keep Current (1 iteration each)

**Current settings:**
- Local optimization: 1 iteration
- Graph-cut: 1 iteration (redundant but fast)
- Least squares: 1 iteration

**This is already optimized** - we reduced everything to 1 iteration.

## Summary Table

| Step | Purpose | Value (spatial_coherence_weight=0) | Cost | Keep? |
|------|---------|-----------------------------------|------|-------|
| **Local Optimization** | Wrapper loop | Necessary (framework) | Low | ✅ Yes |
| **Graph-Cut** | Spatial coherence refinement | **Redundant** (just thresholding) | **High** | ❌ **Skip if possible** |
| **Inner RANSAC** | Re-sample from refined inliers | Less valuable (same inliers) | Medium | ⚠️ Maybe |
| **Least Squares** | Fit model to all inliers | **Very valuable** (improves accuracy) | Low-Medium | ✅ **Yes** |

## The Real Value

**Least Squares is the most valuable step:**
- Takes model from 2 points → fits to 20-110 inliers
- This is where accuracy improvement happens

**Graph-Cut is least valuable (in your case):**
- With `spatial_coherence_weight=0`, it's redundant
- But it's the most expensive step

**Recommendation:**
- Keep least squares (it's valuable and not too expensive)
- Consider skipping graph-cut entirely when `spatial_coherence_weight=0`
- This would give you the biggest speedup!
