# Why Proposal Engine (GC-RANSAC) Takes So Long

## The Bottleneck: Inlier Counting (`getScore()`)

**What happens:**
1. GC-RANSAC samples 2 points (~686 iterations with 1600 points)
2. Estimates a line from those 2 points (fast)
3. **Calls `getScore()` to count inliers** ← **THIS IS THE BOTTLENECK**
4. `getScore()` checks distance to **ALL 1600 points** for each promising model
5. This happens **many times** (for each sample that produces a valid model)

## The Math

**For 1600 points:**
- Sampling: ~686 iterations (now optimized)
- Model estimation: Fast (just fit line to 2 points)
- **Inlier counting: Checks all 1600 points for each promising model**
- If 10% of samples produce promising models: 68 calls to `getScore()`
- Each call: 1600 distance calculations
- **Total: ~109,000 distance calculations per proposal**

**This is why it's slow!**

## What `getScore()` Does

```cpp
for (int point_idx = 0; point_idx < point_number; point_idx += verify_every_kth_point)
{
    // Calculate distance from point to model
    squared_residual = estimator_.squaredResidual(points_.row(point_idx), model_.descriptor);
    
    // If within threshold, count as inlier
    if (squared_residual < squared_truncated_threshold) {
        inliers_.emplace_back(point_idx);
        ++score.inlier_number;
    }
}
```

**It checks EVERY point!**

## Why RANSAC Seems Fast But GC-RANSAC Is Slow

**Standard RANSAC:**
- Samples, estimates, counts inliers
- But stops early if model is bad
- Only counts inliers for promising models

**GC-RANSAC (in Progressive-X):**
- Counts inliers for **every model** that passes initial checks
- No early rejection before inlier counting
- More thorough, but slower

## Potential Optimizations

### 1. **Early Rejection Before Inlier Counting** (Best)

Check a small sample first (e.g., 50-100 points):
- If < 5% are inliers, reject immediately
- Only do full count if sample looks promising
- **Speedup: 5-10x for bad models**

### 2. **Use `verify_every_kth_point`** (Easy)

Skip points during inlier counting:
- Check every 2nd or 3rd point
- Estimate inlier count from sample
- **Speedup: 2-3x**

### 3. **Limit Inlier Counting** (Moderate)

Only count inliers for top N models:
- Track best models so far
- Only count inliers if model might beat best
- **Speedup: 2-5x**

### 4. **Spatial Hashing** (Complex)

Pre-compute spatial structure:
- Only check points near the model
- Skip points that are clearly outliers
- **Speedup: 3-5x**

## Current Status

The code already has some early termination (line 182-183 in scoring_function.h):
```cpp
// Interrupt if there is no chance of being better than the best model
if (point_number - point_idx - score.value < -best_score_.value)
    return Score();
```

But this only helps if we already have a good model. For the first few proposals, it doesn't help.

## Recommendation

**Best immediate fix:** Use `verify_every_kth_point` to skip points:
- Set to 2: Check every 2nd point (2x faster)
- Set to 3: Check every 3rd point (3x faster)
- Still accurate enough for model selection

This is the easiest optimization that will give immediate speedup!
