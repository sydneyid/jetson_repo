#!/usr/bin/env python3
"""
Test Progressive-X Regular 3D Line Detection on May 9 HDF5 Event Data
Reads events from may_9.hdf5 and treats (x, y, t) as a 3D point cloud to detect lines
Uses regular 3D line detection (findLines3D) instead of temporal version

IMPORTANT: Due to a gflags conflict in conda environments, you MUST run this script with:
    python test_may9_events_3d_regular.py 2>/dev/null
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import h5py
from time import time
import warnings

# Add parent directory to path to import pyprogressivex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Redirect stderr at file descriptor level (works for C++ code too)
_stderr_fd = sys.stderr.fileno()
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
_stderr_dup = os.dup(_stderr_fd)  # Save original stderr

try:
    # Redirect stderr to devnull
    os.dup2(_devnull_fd, _stderr_fd)
    import pyprogressivex
except SystemExit:
    raise
except Exception as e:
    os.dup2(_stderr_dup, _stderr_fd)  # Restore stderr to show error
    raise
finally:
    try:
        os.dup2(_stderr_dup, _stderr_fd)
        os.close(_devnull_fd)
        os.close(_stderr_dup)
    except:
        pass

def read_hdf5_events(hdf5_path, time_window=0.1, start_time=None):
    """
    Read events from HDF5 file and create 3D point cloud from (x, y, t)
    
    Args:
        hdf5_path: Path to HDF5 event file
        time_window: Time window in seconds to extract (default: 0.1)
        start_time: Start time in microseconds (if None, starts from beginning)
    
    Returns:
        points: N×3 array of points [x, y, t] where t is in microseconds
        metadata: Dictionary with file metadata
    """
    print(f"Reading HDF5 file: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check available groups
        print(f"  Available groups: {list(f.keys())}")
        
        # Try to find CD events (case-insensitive)
        cd_group = None
        for key in f.keys():
            if key.upper() == 'CD':
                cd_group = f[key]
                break
        
        if cd_group is None:
            raise ValueError(f"Could not find 'CD' group in HDF5 file. Available groups: {list(f.keys())}")
        
        print(f"  Found CD group: {list(cd_group.keys())}")
        
        # Read events
        events = cd_group['events']
        print(f"  Total events in file: {len(events)}")
        
        # Extract event fields: x, y, p, t
        x = events['x'][:]
        y = events['y'][:]
        t = events['t'][:]  # Timestamp in microseconds
        p = events['p'][:]  # Polarity (not used, but we treat all events the same)
        
        # Filter by time window
        if start_time is None:
            start_time = t.min()
        
        end_time = start_time + time_window * 1e6  # Convert to microseconds
        time_mask = (t >= start_time) & (t < end_time)
        
        x_filtered = x[time_mask]
        y_filtered = y[time_mask]
        t_filtered = t[time_mask]
        
        print(f"  Time range: {t.min()/1e6:.3f}s to {t.max()/1e6:.3f}s (duration: {(t.max()-t.min())/1e6:.3f}s)")
        print(f"  Spatial range: X=[{x.min()}, {x.max()}], Y=[{y.min()}, {y.max()}]")
        print(f"  Events in time window [{start_time/1e6:.3f}s, {end_time/1e6:.3f}s]: {len(x_filtered)}")
        
        # Get geometry if available
        geometry = None
        if 'geometry' in cd_group:
            geometry = cd_group['geometry'][:]
            print(f"  Geometry: {geometry[0]}x{geometry[1]}")
        elif 'width' in cd_group.attrs and 'height' in cd_group.attrs:
            geometry = (cd_group.attrs['width'], cd_group.attrs['height'])
            print(f"  Geometry: {geometry[0]}x{geometry[1]}")
        
        # Create 3D point cloud: [x, y, t]
        points = np.column_stack([x_filtered, y_filtered, t_filtered])
        
        metadata = {
            'geometry': geometry,
            'time_range': (t.min(), t.max()),
            'spatial_range': ((x.min(), x.max()), (y.min(), y.max()))
        }
        
        return points, metadata

def normalize_time_coordinate(points, time_scale_factor=None):
    """
    Normalize time coordinate to match spatial scale
    
    Args:
        points: N×3 array [x, y, t]
        time_scale_factor: Scale factor for time (if None, computed automatically)
    
    Returns:
        points_normalized: N×3 array with normalized time
        time_offset: Offset applied to time
        time_scale: Scale factor used
    """
    if time_scale_factor is None:
        spatial_range = max(points[:, 0].max() - points[:, 0].min(),
                           points[:, 1].max() - points[:, 1].min())
        time_range = points[:, 2].max() - points[:, 2].min()
        if time_range > 0:
            time_scale_factor = spatial_range / time_range
        else:
            time_scale_factor = 1e-6
    
    time_offset = points[:, 2].min()
    points_normalized = points.copy()
    points_normalized[:, 2] = (points[:, 2] - time_offset) * time_scale_factor
    
    return points_normalized, time_offset, time_scale_factor

def point_to_line_distance_3d(point, line_point, line_dir):
    """Calculate perpendicular distance from a 3D point to a 3D line"""
    vec = point - line_point
    cross = np.cross(vec, line_dir)
    return np.linalg.norm(cross)

def main():
    print("="*70)
    print("Progressive-X Regular 3D Line Detection on May 9 HDF5 Event Data")
    print("="*70)
    
    # Read HDF5 file
    hdf5_path = 'img/may_9.hdf5'
    
    if not os.path.exists(hdf5_path):
        print(f"✗ Error: File not found: {hdf5_path}")
        return 1
    
    print("\n1. Reading HDF5 event file...")
    try:
        points_raw, metadata = read_hdf5_events(hdf5_path, time_window=0.1, start_time=None)
        print(f"   ✓ Loaded {len(points_raw)} events")
    except Exception as e:
        print(f"   ✗ Error reading HDF5 file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    

    # Convert to integer coordinates for efficient filtering
    x_int = points_raw[:, 0].astype(int)
    y_int = points_raw[:, 1].astype(int)
    t_int = points_raw[:, 2].astype(np.int64)  # Keep as int64 for time
    
    print(f"   ✓ Using all {len(points_raw):,} events (activity filter disabled for maximum line detection)")
    
    # OPTIMIZATION: Optional downsampling for very large datasets
    # If dataset is extremely large (>500k points), consider downsampling
    # This can significantly speed up Progressive-X without losing too much information
    MAX_POINTS_FOR_SPEED = 500000  # Downsample if more than this
    if len(points_raw) > MAX_POINTS_FOR_SPEED:
        downsample_factor = len(points_raw) / MAX_POINTS_FOR_SPEED
        original_size = len(points_raw)
        # Random downsampling
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(points_raw), size=MAX_POINTS_FOR_SPEED, replace=False)
        indices = np.sort(indices)  # Keep temporal order
        points_raw = points_raw[indices]
        print(f"   ⚡ OPTIMIZATION: Downsampled from {original_size:,} to {len(points_raw):,} points "
              f"(factor: {downsample_factor:.1f}x) for speed")
    
    # Normalize time coordinate for better detection
    print("\n3. Normalizing coordinates...")
    
    spatial_range = max(points_raw[:, 0].max() - points_raw[:, 0].min(),
                       points_raw[:, 1].max() - points_raw[:, 1].min())
    time_range = points_raw[:, 2].max() - points_raw[:, 2].min()
    
    if time_range > 0:
        time_scale_factor = spatial_range / time_range
    else:
        time_scale_factor = 1e-6
    
    print(f"   Spatial range: {spatial_range:.1f} pixels")
    print(f"   Time range: {time_range/1e6:.3f} seconds ({time_range:.1f} microseconds)")
    print(f"   Time scale factor: {time_scale_factor:.6f}")
    
    points_normalized, time_offset, time_scale = normalize_time_coordinate(
        points_raw, time_scale_factor=time_scale_factor
    )
    
    print(f"   ✓ Spatial range: X=[{points_raw[:, 0].min():.1f}, {points_raw[:, 0].max():.1f}], "
          f"Y=[{points_raw[:, 1].min():.1f}, {points_raw[:, 1].max():.1f}]")
    print(f"   ✓ Time range: [{points_raw[:, 2].min()/1e6:.3f}s, {points_raw[:, 2].max()/1e6:.3f}s]")
    print(f"   ✓ Time scale factor: {time_scale_factor:.6f} (1 unit = {1/time_scale_factor:.1f} microseconds)")
    print(f"   ✓ Normalized coordinate ranges:")
    print(f"     X: [{points_normalized[:, 0].min():.1f}, {points_normalized[:, 0].max():.1f}]")
    print(f"     Y: [{points_normalized[:, 1].min():.1f}, {points_normalized[:, 1].max():.1f}]")
    print(f"     T: [{points_normalized[:, 2].min():.1f}, {points_normalized[:, 2].max():.1f}]")
    
    # Compute optimal parameters - tuned for parallel lines
    print("\n4. Computing optimal parameters (tuned for parallel lines)...")
    
    from scipy.spatial.distance import cdist
    n_sample = min(100, len(points_normalized))
    sample_indices = np.random.choice(len(points_normalized), n_sample, replace=False)
    sample_points = points_normalized[sample_indices]
    dists = cdist(sample_points, sample_points)
    np.fill_diagonal(dists, np.inf)
    nearest_dists = np.min(dists, axis=1)
    estimated_noise = np.median(nearest_dists)
    
    data_range = np.max(points_normalized.max(axis=0) - points_normalized.min(axis=0))
    
    # Set threshold - EXTREMELY PERMISSIVE to detect MANY more lines
    # Lower threshold to detect many more lines (including very sparse ones)
    threshold_from_range = data_range * 0.0005  # 0.05% of range (extremely permissive)
    threshold_from_noise = estimated_noise * 0.8  # 0.8× noise (extremely permissive)
    final_threshold = max(threshold_from_range, threshold_from_noise)
    final_threshold = max(final_threshold, data_range * 0.0003)  # Minimum 0.03% of range
    final_threshold = min(final_threshold, data_range * 0.008)  # Maximum 0.8% (extremely permissive)
    
    # Neighborhood radius: scaled to data range (smaller to allow more distinct lines)
    neighbor_radius = data_range * 0.002  # 0.2% of range (smaller to allow more distinct lines)
    neighbor_radius = max(neighbor_radius, estimated_noise * 0.15)
    neighbor_radius = min(neighbor_radius, data_range * 0.010)
    
    # Minimum point number - EXTREMELY LOW to allow very small/sparse lines to be detected
    minimum_point_number = 2  # Minimum 2 points (allows detection of very sparse lines)
    
    print(f"   ✓ Estimated noise: {estimated_noise:.6f}")
    print(f"   ✓ Threshold: {final_threshold:.6f} ({100*final_threshold/data_range:.2f}% of range)")
    print(f"   ✓ Neighborhood radius: {neighbor_radius:.6f}")
    print(f"   ✓ Minimum points per model: {minimum_point_number}")
    
    # Run Progressive-X Regular 3D line detection (NO temporal constraint)
    print("\n5. Running Progressive-X Regular 3D line detection (no temporal constraint)...")
    
    # Restore to working variant settings that produced ~18 lines:
    # - Low confidence (0.05) to find many models
    # - Higher max_iters to test more samples (decent number relative to total)
    n_points = len(points_normalized)

    # Increase max_iters to find more lines (was too low)
    if n_points < 10000:
        max_iters_optimized = 10000  # Increased from 10k to find more models
    elif n_points < 100000:
        max_iters_optimized = 200000  # Increased from 50k to find more models
    else:
        max_iters_optimized = 500000  # Increased from 100k to find more models
    
    print(f"   Using max_iters={max_iters_optimized:,} (scaled from {n_points:,} points)")
    
    # OPTIMIZATIONS FOR FINDING MORE LINES:
    # 1. Lower confidence (0.05) to find more models - requires more iterations but finds more lines
    # 2. Use NAPSAC sampler (sampler_id=2) - faster than uniform for spatial data
    # 3. max_iters increased above to find more models
    # Note: Lower confidence finds more models but takes longer
    conf_optimized = 0.5  # Low confidence to find many models
    sampler_id_optimized = 2  # NAPSAC sampler - faster for spatial clustering
    
    print(f"   Settings: conf={conf_optimized} (low for many models), sampler=NAPSAC, max_iters={max_iters_optimized:,}")
    
    try:
        t = time()
        lines, labeling = pyprogressivex.findLines3D(
            np.ascontiguousarray(points_normalized, dtype=np.float64),
            np.ascontiguousarray([], dtype=np.float64),  # No weights
            threshold=final_threshold,
            conf=conf_optimized,  # OPTIMIZED: Increased from 0.05 to reduce RANSAC iterations
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=neighbor_radius,
            maximum_tanimoto_similarity=0.20,  # Increased from 0.20 to allow finding more overlapping parallel lines
            max_iters=max_iters_optimized,  # OPTIMIZED: Reduced from 2M to reasonable value
            minimum_point_number=minimum_point_number,
            maximum_model_number=20000,  # Allow MANY more lines
            sampler_id=sampler_id_optimized,  # OPTIMIZED: NAPSAC sampler (faster for spatial data)
            scoring_exponent=0.0,  # NO penalty for shared support (allows many overlapping lines)
            do_logging=True  # Enable logging to see timing breakdown
        )
        elapsed_time = time() - t
        
        num_models = lines.shape[0] if lines.size > 0 else 0
        
        print(f"   ✓ Completed in {elapsed_time:.2f} seconds")
        print(f"   ✓ Detected {num_models} line models (before filtering)")
        
        # Print detailed timing breakdown if available
        # Note: Statistics are not directly exposed in Python bindings, so we use elapsed_time
        # The breakdown would be:
        # - Proposal engine (GC-RANSAC): ~60-70% of time
        # - PEARL optimization: ~20-30% of time  
        # - Model validation: ~5-10% of time
        # - Compound model update: ~1-5% of time
        print(f"   → Note: Timing breakdown not directly available in Python bindings")
        print(f"   → Estimated breakdown (typical):")
        print(f"      - Proposal engine (GC-RANSAC): ~{elapsed_time*0.65:.2f}s (65%)")
        print(f"      - PEARL optimization: ~{elapsed_time*0.25:.2f}s (25%)")
        print(f"      - Model validation: ~{elapsed_time*0.07:.2f}s (7%)")
        print(f"      - Compound model update: ~{elapsed_time*0.03:.2f}s (3%)")
        
        # Check if labels are 0-indexed (before filtering)
        use_zero_indexed = False
        if num_models > 0:
            label0_mask = (labeling == 0)
            if np.sum(label0_mask) > minimum_point_number:
                points_label0 = points_normalized[label0_mask]
                if len(points_label0) >= 2:
                    centroid = points_label0.mean(axis=0)
                    centered = points_label0 - centroid
                    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
                    line_dir = Vt[0]
                    line_dir = line_dir / np.linalg.norm(line_dir)
                    distances_label0 = [point_to_line_distance_3d(p, centroid, line_dir) for p in points_label0]
                    inlier_ratio_label0 = np.sum(np.array(distances_label0) <= final_threshold) / len(distances_label0)
                    if inlier_ratio_label0 > 0.6:
                        use_zero_indexed = True
                        print(f"   ✓ Using 0-indexed interpretation: label 0 = first model")
        
        # Post-process: Filter out poor quality lines
        print(f"\n   Post-processing: Filtering lines by quality...")
        
        # First pass: collect all line sizes to determine typical size
        line_sizes = []
        for idx in range(num_models):
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            mask = (labeling == instance_label)
            line_sizes.append(np.sum(mask))
        
        if len(line_sizes) > 0:
            median_line_size = np.median(line_sizes)
            mean_line_size = np.mean(line_sizes)
            max_reasonable_size = max(median_line_size * 3.5, mean_line_size * 3.0)  # More lenient
            print(f"   Line size stats: median={median_line_size:.0f}, mean={mean_line_size:.0f}, max_reasonable={max_reasonable_size:.0f}")
        else:
            max_reasonable_size = len(points_normalized) * 0.25  # 25% max (more lenient)
        
        valid_line_indices = []
        max_points_per_line = len(points_normalized) * 0.20  # 20% max
        
        for idx in range(num_models):
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            
            mask = (labeling == instance_label)
            points_assigned = points_normalized[mask]
            
            if len(points_assigned) == 0:
                continue
            
            # Calculate quality metrics
            line_point = np.array([lines[idx, 0], lines[idx, 1], lines[idx, 2]])
            line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
            line_dir = line_dir / np.linalg.norm(line_dir)
            
            distances = [point_to_line_distance_3d(p, line_point, line_dir) for p in points_assigned]
            distances = np.array(distances)
            inlier_count = np.sum(distances <= final_threshold)
            inlier_ratio = inlier_count / len(distances)
            mean_distance = distances.mean()
            max_distance = distances.max()
            
            # Filter criteria: VERY PERMISSIVE since there are few outliers
            is_valid = True
            reasons = []
            
            if inlier_ratio < 0.25:  # Extremely low threshold to keep many more lines
                is_valid = False
                reasons.append(f"low inlier ratio ({inlier_ratio*100:.1f}%)")
            
            if mean_distance > final_threshold * 2.5:  # Extremely lenient to keep more lines
                is_valid = False
                reasons.append(f"high mean distance ({mean_distance:.2f} vs threshold {final_threshold:.2f})")
            
            if len(points_assigned) > max_points_per_line:
                is_valid = False
                reasons.append(f"too many points ({len(points_assigned)} = {100*len(points_assigned)/len(points_normalized):.1f}% > {100*max_points_per_line/len(points_normalized):.1f}%)")
            
            if len(line_sizes) > 1 and len(points_assigned) > max_reasonable_size:
                is_valid = False
                reasons.append(f"unusually large ({len(points_assigned)} > {max_reasonable_size:.0f}, likely merged)")
            
            if is_valid:
                valid_line_indices.append(idx)
                # Calculate line extent
                projections = np.array([np.dot(p - line_point, line_dir) for p in points_assigned])
                extent = projections.max() - projections.min()
                
                print(f"   Line {len(valid_line_indices)}: {len(points_assigned)} points, "
                      f"{inlier_ratio*100:.1f}% within threshold, "
                      f"mean distance: {mean_distance:.4f}, max distance: {max_distance:.4f}, "
                      f"std distance: {np.std(distances):.4f}, extent: {extent:.2f}")
                print(f"      Direction: ({line_dir[0]:.4f}, {line_dir[1]:.4f}, {line_dir[2]:.4f})")
            else:
                print(f"   Rejected line {idx+1}: {', '.join(reasons)}")
        
        print(f"   ✓ Kept {len(valid_line_indices)} high-quality lines out of {num_models} detected")
        
        # FILTER: Remove lines that aren't parallel to time axis (for event data)
        # Lines should be parallel to Z (time) axis, so direction should be close to (0, 0, 1)
        print(f"\n   Filtering: Removing non-parallel lines (must be parallel to time axis)...")
        expected_direction = np.array([0.0, 0.0, 1.0])  # Time axis direction
        parallel_valid_indices = []
        # Stricter threshold: 1° = cos(1°) ≈ 0.99985
        parallel_threshold = np.cos(np.deg2rad(1.0))  # Lines must be within 1° of time axis
        
        for idx in valid_line_indices:
            line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
            line_dir = line_dir / np.linalg.norm(line_dir)
            
            # Check if line is parallel to time axis
            dot_with_time = np.abs(np.dot(line_dir, expected_direction))
            angle_with_time = np.arccos(np.clip(dot_with_time, -1, 1)) * 180 / np.pi
            
            if dot_with_time >= parallel_threshold:
                parallel_valid_indices.append(idx)
            else:
                print(f"   ✗ Removed line {idx+1}: not parallel to time axis (angle={angle_with_time:.1f}°, threshold=1.0°)")
        
        # FILTER: Remove lines that aren't parallel to each other
        # Ensure ALL pairs of lines are within 1° of each other
        # OPTIMIZATION: Use vectorized operations for O(n²) pairwise comparisons
        if len(parallel_valid_indices) > 1:
            print(f"\n   Filtering: Ensuring all lines are parallel to each other (all pairs within 1°)...")
            final_valid_indices = list(parallel_valid_indices)
            
            # OPTIMIZATION: Vectorized approach - compute all pairwise dot products at once
            # Build matrix of normalized directions (only once)
            n_lines = len(final_valid_indices)
            dirs_matrix = np.zeros((n_lines, 3))
            for pos, idx in enumerate(final_valid_indices):
                dir_vec = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
                dirs_matrix[pos] = dir_vec / np.linalg.norm(dir_vec)
            
            # Compute pairwise dot products: dirs_matrix @ dirs_matrix.T gives n×n matrix
            # Each element [i,j] is dot(dirs[i], dirs[j])
            pairwise_dots = np.abs(dirs_matrix @ dirs_matrix.T)  # n×n matrix
            
            # Remove lines that aren't parallel to all others
            removed_any = True
            iteration = 0
            max_iterations = 10  # Prevent infinite loops
            
            while removed_any and len(final_valid_indices) > 1 and iteration < max_iterations:
                removed_any = False
                iteration += 1
                lines_to_remove = []
                
                # OPTIMIZATION: Use vectorized check - for each line, check if ALL other dots >= threshold
                # Create a mask to exclude diagonal (self-comparisons)
                mask = np.ones_like(pairwise_dots, dtype=bool)
                np.fill_diagonal(mask, False)  # Exclude self-comparisons
                
                # For each line, check if minimum dot product with others is >= threshold
                # If min < threshold, this line is not parallel to at least one other
                min_dots_per_line = np.where(mask, pairwise_dots, 1.0).min(axis=1)
                invalid_positions = np.where(min_dots_per_line < parallel_threshold)[0]
                
                for pos in invalid_positions:
                    idx = final_valid_indices[pos]
                    lines_to_remove.append((pos, idx))
                    removed_any = True
                
                # Remove invalid lines (in reverse order to maintain indices)
                for pos, idx in sorted(lines_to_remove, reverse=True):
                    # Find which line it wasn't parallel to for reporting
                    non_parallel_pos = np.where(pairwise_dots[pos] < parallel_threshold)[0]
                    if len(non_parallel_pos) > 0:
                        other_pos = non_parallel_pos[0]
                        if other_pos != pos:  # Don't compare with self
                            other_idx = final_valid_indices[other_pos]
                            angle_deg = np.arccos(np.clip(pairwise_dots[pos, other_pos], -1, 1)) * 180 / np.pi
                            print(f"   ✗ Removed line {idx+1}: not parallel to line {other_idx+1} (angle={angle_deg:.1f}°, threshold=1.0°)")
                    
                    # Remove from final_valid_indices
                    final_valid_indices.pop(pos)
                    # Remove row and column from matrices (more efficient than rebuilding)
                    dirs_matrix = np.delete(dirs_matrix, pos, axis=0)
                    pairwise_dots = np.delete(np.delete(pairwise_dots, pos, axis=0), pos, axis=1)
            
            valid_line_indices = final_valid_indices
        else:
            valid_line_indices = parallel_valid_indices
        
        print(f"   ✓ Kept {len(valid_line_indices)} parallel lines out of {len(parallel_valid_indices)} time-axis-parallel lines")
        
        # Report parallelism statistics
        if len(valid_line_indices) > 1:
            print(f"\n   Parallelism check of final lines...")
            parallelism_scores = []
            for i, idx1 in enumerate(valid_line_indices):
                dir1 = np.array([lines[idx1, 3], lines[idx1, 4], lines[idx1, 5]])
                dir1 = dir1 / np.linalg.norm(dir1)
                for idx2 in valid_line_indices[i+1:]:
                    dir2 = np.array([lines[idx2, 3], lines[idx2, 4], lines[idx2, 5]])
                    dir2 = dir2 / np.linalg.norm(dir2)
                    dot_product = np.abs(np.dot(dir1, dir2))
                    angle_deg = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
                    parallelism_scores.append(dot_product)
                    print(f"      Lines {i+1} & {len([v for v in valid_line_indices if v <= idx2]):.0f}: "
                          f"dot={dot_product:.4f}, angle={angle_deg:.2f}° ✓ parallel")
            
            if parallelism_scores:
                avg_parallelism = np.mean(parallelism_scores)
                print(f"   Average parallelism score: {avg_parallelism:.4f} (1.0 = perfectly parallel)")
        
        # Visualization
        print("\n6. Visualizing results...")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 6))
        
        # Plot 1: 3D point cloud of all event data
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(points_normalized[:, 0], points_normalized[:, 1], points_normalized[:, 2],
                   c=points_normalized[:, 2], cmap='viridis', s=1, alpha=0.3)
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_zlabel('Time (normalized)')
        ax1.set_title('All Event Data (3D Point Cloud)')
        
        # Plot 2: 3D view showing only points assigned to detected lines
        ax2 = fig.add_subplot(132, projection='3d')
        colors_det = plt.cm.tab10(np.linspace(0, 1, len(valid_line_indices)))
        
        for i, idx in enumerate(valid_line_indices):
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            mask = (labeling == instance_label)
            points_line = points_normalized[mask]
            if len(points_line) > 0:
                ax2.scatter(points_line[:, 0], points_line[:, 1], points_line[:, 2],
                           c=[colors_det[i]], s=5, alpha=0.6, label=f'Line {i+1}')
        
        # Draw detected lines
        for i, idx in enumerate(valid_line_indices):
            line_point = np.array([lines[idx, 0], lines[idx, 1], lines[idx, 2]])
            line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
            line_dir = line_dir / np.linalg.norm(line_dir)
            
            # Find extent of points on this line
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            mask = (labeling == instance_label)
            points_line = points_normalized[mask]
            if len(points_line) > 0:
                projections = np.array([np.dot(p - line_point, line_dir) for p in points_line])
                t_min, t_max = projections.min(), projections.max()
                p_start = line_point + t_min * line_dir
                p_end = line_point + t_max * line_dir
                ax2.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], [p_start[2], p_end[2]],
                        'k-', linewidth=2, alpha=0.5)
        
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_zlabel('Time (normalized)')
        ax2.set_title(f'Detected Lines ({len(valid_line_indices)} lines)')
        ax2.legend(loc='upper right', fontsize=8)
        
        # Plot 3: Detected lines projected to (x, y) space
        ax3 = fig.add_subplot(133)
        
        # Plot all points as background
        ax3.scatter(points_normalized[:, 0], points_normalized[:, 1], c='lightgray', s=1, alpha=0.3)
        
        # Plot detected lines
        for i, idx in enumerate(valid_line_indices):
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            mask = (labeling == instance_label)
            points_line = points_normalized[mask]
            if len(points_line) > 0:
                ax3.scatter(points_line[:, 0], points_line[:, 1], c=[colors_det[i]], s=5, alpha=0.6, label=f'Line {i+1}')
            
            # Draw line in 2D projection
            line_point = np.array([lines[idx, 0], lines[idx, 1], lines[idx, 2]])
            line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
            line_dir = line_dir / np.linalg.norm(line_dir)
            
            # Project line to 2D (x, y) plane
            # Find extent
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            mask = (labeling == instance_label)
            points_line = points_normalized[mask]
            if len(points_line) > 0:
                projections = np.array([np.dot(p - line_point, line_dir) for p in points_line])
                t_min, t_max = projections.min(), projections.max()
                p_start = line_point + t_min * line_dir
                p_end = line_point + t_max * line_dir
                ax3.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'k-', linewidth=2, alpha=0.7)
        
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        ax3.set_title('Detected Lines (X-Y Projection)')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_aspect('equal')
        
        plt.tight_layout()
        output_path = 'may9_events_3d_regular_detection.png'
        plt.show()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved visualization to {output_path}")
        plt.close()
        
        print("\n" + "="*70)
        print("✓ Test completed successfully!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"   ✗ Error during line detection: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
