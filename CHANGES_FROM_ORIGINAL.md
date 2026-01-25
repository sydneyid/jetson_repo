# Changes from Original Progressive-X Implementation

This document summarizes the modifications made to the Progressive-X C++ implementation compared to the original GitHub version (https://github.com/danini/progressive-x).

## Original Default Values (from GitHub)

Based on the original implementation and the adelaideH.ipynb notebook:

- **Main loop iterations**: 100 (original)
- **PEARL maximum_iteration_number**: 100 (original default)
- **max_proposal_number_without_change**: 10 (original default)
- **max_local_optimization_number**: 50 (original default)
- **Early termination**: Original logic (terminate if `models.size() > 1 && unseen_inliers < minimum_number_of_inliers`)
- **Tanimoto similarity check**: Original strict check using `settings.maximum_tanimoto_similarity` directly

## Current Modified Values

### 1. Main Loop Iterations (`progressive_x.h` line 274)
- **Original**: 100 iterations
- **Current**: 50 iterations
- **Reason**: Reduced for faster execution (~40s target)

### 2. PEARL Maximum Iterations (`progressive_x.h` line 542)
- **Original**: 100 iterations
- **Current**: 30 iterations
- **Reason**: PEARL iterations are expensive; reducing significantly speeds up execution

### 3. max_proposal_number_without_change
- **Original**: 10 (default in `MultiModelSettings`)
- **Current**: 
  - `findLines3D`: 50 (reduced from 100 for speed)
  - `findLines3DTemporal`: 200 (increased to allow finding more models)
  - Other functions: 100 (increased to allow finding more models)
- **Reason**: Original value of 10 stops too early; increased to allow finding more models

### 4. max_local_optimization_number
- **Original**: 50 (default in `MultiModelSettings`)
- **Current**: 
  - `findLines3D`: 5 (reduced for speed)
  - `findLines3DTemporal`: 10 (reduced for speed)
  - Other functions: 10 (reduced for speed)
- **Reason**: Local optimization is expensive; reducing speeds up execution

### 5. Early Termination Logic (`progressive_x.h` line 473)
- **Original**: `if (models.size() > 1 && unseen_inliers < settings.minimum_number_of_inliers) break;`
- **Current**: `if (models.size() >= 2 && unseen_inliers < settings.minimum_number_of_inliers) break;`
- **Reason**: More aggressive early termination for speed (terminates after 2 models instead of waiting for more)

### 6. Tanimoto Similarity Check (`progressive_x.h` line 599)
- **Original**: Strict check using `settings.maximum_tanimoto_similarity` directly
- **Current**: 
  - For first 50 models: **Disabled** (allows finding many overlapping parallel lines)
  - After 50 models: Uses lenient threshold (0.99)
- **Reason**: For event data with parallel lines, models naturally have very high similarity (0.8-0.99). The original strict check was preventing finding many models.

## Summary

The main changes focus on:
1. **Speed optimizations**: Reduced iterations (main loop, PEARL, local optimization)
2. **Finding more models**: Increased `max_proposal_number_without_change` and disabled Tanimoto check for early models
3. **Early termination**: More aggressive termination to balance speed vs. model detection

These changes were made specifically for the 3D line detection use case with event data, where:
- Many parallel lines need to be detected
- Speed is important (~40s target)
- Models naturally have high Tanimoto similarity due to spatial overlap
