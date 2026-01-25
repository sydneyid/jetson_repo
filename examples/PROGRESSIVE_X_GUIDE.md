# Progressive-X Algorithm Guide (Based on Paper)

## Understanding Progressive-X

According to the paper (Barath & Matas, 2019), Progressive-X is an **any-time, multi-model fitting algorithm** that interleaves three steps:

1. **Hypothesis Proposal**: Uses Graph-Cut RANSAC to propose new model instances
2. **Fast Rejection**: Rejects redundant hypotheses using Tanimoto similarity
3. **Integration**: Uses PEARL (energy minimization) to integrate new hypotheses into the current model set

## Key Insight: Why "Modes" Appear

The paper explains that Progressive-X finds **modes in the model parameter space**. However, these modes might not always correspond to **consistent line segments** in the data space. This is why you're seeing classifications that look like "two intersecting lines that aren't close" - Progressive-X is finding a line that passes through scattered points, even if those points don't form a contiguous segment.

## Energy Function Components

The PEARL optimization uses an energy function with:

1. **Data Term**: `E_data = (1-λ) * Σ residual(point, model)` - how well points fit the model
2. **Spatial Coherence Term**: `E_spatial = λ * Σ [neighbors have different labels]` - encourages spatial clustering
3. **Label Cost**: Encourages fewer models (handled automatically)

**Important**: The total energy is: `Energy = (1-λ) * E_data + λ * E_spatial`

The **spatial coherence weight (λ)** is crucial - it controls how much neighboring points prefer the same label.

### Truncated Threshold

From the PEARL implementation, the algorithm uses a **truncated squared threshold**:
- `truncated_squared_threshold = 9/4 * threshold²`
- This means points up to **1.5× the threshold** distance can still contribute to the energy (with reduced weight)
- This is why scattered points can be assigned to a line - the algorithm is more lenient than the base threshold suggests
- **This is why post-processing validation is essential** - we need to check with the actual threshold

## Recommended Parameters for Line Detection

Based on the paper and your use case:

### 1. **Spatial Coherence Weight (λ)**
- **Purpose**: Controls spatial clustering of inliers
- **For line detection**: Should be **higher** (0.1-0.3) to ensure points on the same line segment are grouped together
- **Current**: 0.2 (good)
- **Note**: Higher values help prevent scattered points from being assigned to the same line

### 2. **Inlier-Outlier Threshold**
- **Purpose**: Distance threshold for determining inliers
- **For line detection**: Should be **strict** (2-4× noise level)
- **Current**: `adaptive_threshold = 4 × noise_sigma` (good)
- **Important**: This is used in the energy function, so it directly affects which points are considered inliers

### 3. **Neighborhood Ball Radius**
- **Purpose**: Defines spatial neighborhood for spatial coherence
- **For line detection**: Should capture local point density
- **Current**: 5th percentile of distances (adaptive, good)
- **Note**: This affects which points are considered "neighbors" for spatial coherence

### 4. **Maximum Tanimoto Similarity**
- **Purpose**: Threshold for rejecting redundant hypotheses
- **For line detection**: Lower values (0.7-0.9) help merge similar lines
- **Current**: 0.85 (good)
- **Note**: This prevents duplicate detections of the same line

### 5. **Minimum Point Number**
- **Purpose**: Minimum inliers required to keep a model
- **For line detection**: Should be reasonable (20-30% of expected points per line)
- **Current**: 6 (might be too low - try 8-10)

## The Problem: Scattered Points Forming "Modes"

When Progressive-X finds a mode that isn't a consistent line, it's because:

1. **Spatial coherence is too weak**: Points far apart can still be assigned to the same line if the data term is strong enough
2. **Threshold is too lenient**: Points far from the line are still considered inliers
3. **No validation of line consistency**: Progressive-X finds modes in parameter space, not necessarily contiguous segments

## Solution: Post-Processing Validation

The validation we added checks:

1. **Point-to-line distance**: At least 75% of points must be within the **actual threshold** (not the truncated one)
2. **Contiguity**: Points must form a contiguous segment (not scattered) - checks for large gaps in projections
3. **Distance variance**: Low variance indicates consistent line

This post-processing is **necessary** because:
- Progressive-X uses a truncated threshold internally (1.5× the base threshold)
- Progressive-X optimizes for modes in parameter space, not necessarily for consistent line segments in data space
- The spatial coherence term can assign scattered points to the same line if the data term is strong enough

## Recommended Configuration

Based on the paper and your observations:

```python
threshold = 4 × noise_sigma  # Strict threshold
spatial_coherence_weight = 0.2-0.3  # Strong spatial coherence
neighborhood_ball_radius = 5th percentile of distances  # Adaptive
maximum_tanimoto_similarity = 0.8-0.9  # Allow some merging
max_iters = 3000-5000  # Enough iterations
minimum_point_number = 8-10  # Reasonable minimum
maximum_model_number = 8-10  # Limit to prevent over-segmentation
sampler_id = 2  # NAPSAC (spatial-aware sampling)
```

## Why Post-Processing is Essential

The paper focuses on finding **modes in the model parameter space**, but for line detection, we need **consistent line segments in the data space**. The validation step bridges this gap by:

1. Checking that points assigned to a line actually form a contiguous segment
2. Ensuring most points are close to the line
3. Refitting lines using only validated inliers

This is consistent with the paper's approach - Progressive-X finds candidate modes, and we validate them for our specific application.

## References

Barath, D., & Matas, J. (2019). Progressive-X: Efficient, Anytime, Multi-Model Fitting Algorithm. *arXiv preprint arXiv:1906.02290*. https://arxiv.org/pdf/1906.02290
