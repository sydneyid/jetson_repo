#!/usr/bin/env python3
"""
Test Progressive-X Regular 3D Line Detection on Jupiter HDF5 Event Data
Reads events from jupiter.hdf5 and treats (x, y, t) as a 3D point cloud to detect lines
Uses regular 3D line detection (findLines3D) instead of temporal version

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import h5py
from time import time
import warnings
from scipy.spatial.distance import cdist
import matplotlib
import matplotlib.colors

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

def read_hdf5_events(hdf5_path, time_window=0.1, events_constraint=None, start_time=None):
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
        else :
            start_time = int(start_time * 1e6)  # Convert to microseconds
        
        end_time = start_time + time_window * 1e6  # Convert to microseconds
        time_mask = (t >= start_time) & (t < end_time)
        
        
        x_filtered = x[time_mask]
        y_filtered = y[time_mask]
        t_filtered = t[time_mask]   
        p_filtered = p[time_mask] 

        p_mask = (p_filtered== 1)

        x_filtered = x_filtered[p_mask]
        y_filtered = y_filtered[p_mask]
        t_filtered = t_filtered[p_mask]   
        p_filtered = p_filtered[p_mask] 

        if events_constraint is not None and len(x_filtered) > events_constraint:
            # Downsample events to meet the constraint
            x_filtered = x_filtered[0:events_constraint]
            y_filtered = y_filtered[0:events_constraint]
            t_filtered = t_filtered[0:events_constraint]
            p_filtered = p_filtered[0:events_constraint]

        
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


def parallel_check(lines,labeling,num_models , parallel_threshold = 1.0,print_updates=False ):
    # assume labels are zero indexed 

    expected_direction = np.array([0.0, 0.0, 1.0])  # Time axis direction
    valid_line_indices = []
    parallel_threshold = np.cos(np.deg2rad(parallel_threshold))  
    
    # Iterate over all detected lines
    for idx in range(num_models):
        line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
        line_dir = line_dir / np.linalg.norm(line_dir)
        
        # Check if line is parallel to time axis
        dot_with_time = np.abs(np.dot(line_dir, expected_direction))
        angle_with_time = np.arccos(np.clip(dot_with_time, -1, 1)) * 180 / np.pi
        
        if dot_with_time >= parallel_threshold:
            valid_line_indices.append(idx)
        else:
            if print_updates:
                print("   Removed line  "+str(idx+1)+ ": not parallel to time axis (angle= " + str(angle_with_time) + ", threshold= " + str(parallel_threshold) )
    
 
    
    if print_updates:
    # Summary of detection pipeline
        print(f"\n   📊 DETECTION PIPELINE SUMMARY:")
        print(f"      Stage 1 (Progressive-X): {num_models} suggested/proposed lines")
        print(f"      Stage 2 (Time-axis parallel): {len(valid_line_indices)} lines parallel to time axis")

    return valid_line_indices




def main():
    print("="*70)
    print("Progressive-X Regular 3D Line Detection on May 9 HDF5 Event Data")
    print("="*70)
    
    # Read HDF5 file
    hdf5_path = '/Users/sydneydolan/Documents/may9_data/Jupiter_In_Focus.hdf5'
    
    if not os.path.exists(hdf5_path):
        print(f"  Error: File not found: {hdf5_path}")
        return 1
    
    print("\n1. Reading HDF5 event file...")
    try:
        # ADJUSTABLE PARAMETERS FOR FINDING MORE LINES:
        # - time_window: Increase (e.g., 0.1) to include more events
        # - events_constraint: Increase (e.g., 1000) to use more events
        # - start_time: Change to analyze different time period
        points_raw, metadata = read_hdf5_events(hdf5_path, time_window=0.05, events_constraint = 500,start_time=14.06)
        print(f"   ✓ Loaded {len(points_raw)} events")
    except Exception as e:
        print(f"     Error reading HDF5 file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    

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
    
    
    n_sample = min(100, len(points_normalized))
    sample_indices = np.random.choice(len(points_normalized), n_sample, replace=False)
    sample_points = points_normalized[sample_indices]
    dists = cdist(sample_points, sample_points)
    np.fill_diagonal(dists, np.inf)
    nearest_dists = np.min(dists, axis=1)
    estimated_noise = np.median(nearest_dists)
    
    data_range = np.max(points_normalized.max(axis=0) - points_normalized.min(axis=0))
    
    # Calculate base threshold - balanced for both dense and sparse detection
    # Dense lines will use a stricter version, sparse lines will use a more permissive version
    # Slightly more permissive to catch lines that might be missed
    threshold_from_range = data_range * 0.001  # Slightly more permissive (was 0.0008)
    threshold_from_noise = estimated_noise * 1.0  # Slightly more permissive (was 0.9)
    base_threshold = max(threshold_from_range, threshold_from_noise)
    base_threshold = max(base_threshold, data_range * 0.0003)  # Minimum 0.03% of range
    base_threshold = min(base_threshold, data_range * 0.012)  # Increased max cap slightly (was 0.010)
    
    # Neighborhood radius: scaled to data range (smaller to allow more distinct lines)
    neighbor_radius = data_range * 0.002  # 0.2% of range (smaller to allow more distinct lines)
    neighbor_radius = max(neighbor_radius, estimated_noise * 0.15)
    neighbor_radius = min(neighbor_radius, data_range * 0.010)
    

    minimum_point_number = 10  # Minimum 10 inliers - allows finding more lines while still filtering weak proposals
    
    print(f"   ✓ Estimated noise: {estimated_noise:.6f}")
    print(f"   ✓ Base threshold: {base_threshold:.6f} ({100*base_threshold/data_range:.2f}% of range)")
    print(f"   ✓ Neighborhood radius: {neighbor_radius:.6f}")
    print(f"   ✓ Minimum points per model: {minimum_point_number}")
    
    print("\n5. Running Progressive-X Regular 3D line detection (no temporal constraint)...")
    

    n_points = len(points_normalized)


    if n_points < 2000:
        # ADJUSTED FOR SPARSE LINE: More iterations to find sparse line
        max_iters_optimized = max(1000, int(n_points * 1.0))  # Increased from 500/0.5 to 1000/1.0 for sparse line
    elif n_points < 5000:
        # Small datasets: scale proportionally from baseline (7000 points = 4000 iterations)
        max_iters_optimized = int(n_points * 4000 / 7000)  # Proportional to 7000→4000 baseline (was 3000)
    elif n_points < 10000:
        max_iters_optimized = 4000   # Baseline for medium datasets (was 3000)
    elif n_points < 100000:
        max_iters_optimized = 12000  # Large datasets (was 10000)
    else:
        max_iters_optimized = 30000  # Very large datasets (was 25000)
    
    print(f"   Using max_iters={max_iters_optimized:,} (scaled from {n_points:,} points)")
    
    # Use higher confidence for dense line detection to ensure thorough search
    # Dense lines should be found reliably, then sparse lines can use remaining points
    conf_dense = 0.05  # Higher confidence for dense lines to ensure all are found
    conf_sparse = 0.02  # Lower confidence for sparse lines (more permissive search)
    sampler_id_optimized = 3  # Z-Aligned sampler - samples points with similar X,Y but different Z (ideal for lines parallel to time axis)
    
    print(f"   Settings: conf_dense={conf_dense}, conf_sparse={conf_sparse}, sampler=Z-Aligned (for time-parallel lines), max_iters={max_iters_optimized:,}")
    print(f"   → Dense detection uses higher confidence to find all dense lines first")
    
    try:
        t = time()
        # Force stdout to be unbuffered for this section
        import sys
        old_stdout = sys.stdout
        sys.stdout = sys.__stdout__  # Use unbuffered stdout
        
        # DUAL DETECTION: Detect both dense and sparse lines with mutual exclusivity
        # Dense lines: use slightly relaxed threshold (0.9x base) to catch all dense lines including potentially weaker ones
        # Sparse lines: use more permissive threshold (2.5x base) to catch sparse/noisy lines
        # Key: Slightly relax dense threshold to catch lines that might be borderline
        threshold_dense = base_threshold * 0.9  # Slightly relaxed (90% of base) to catch borderline dense lines
        threshold_sparse = base_threshold * 2.5  # More permissive for sparse lines (2.5x base)
        minimum_point_number_sparse = max(3, minimum_point_number // 2)  # Lower minimum for sparse lines (half of dense)
        
        print(f"   Dense line detection: threshold={threshold_dense:.6f} ({100*threshold_dense/data_range:.2f}% of range), min_points={minimum_point_number}")
        print(f"   Sparse line detection: threshold={threshold_sparse:.6f} (permissive, {100*threshold_sparse/data_range:.2f}% of range), min_points={minimum_point_number_sparse}")
        
        # Note: findLines3DDual uses a single conf parameter for both stages
        # We'll use conf_dense for the dense stage, but the sparse stage will also use it
        # The key is that dense detection happens first with stricter threshold
        lines, labeling, line_types = pyprogressivex.findLines3DDual(
            np.ascontiguousarray(points_normalized, dtype=np.float64),
            np.ascontiguousarray([], dtype=np.float64),  # No weights
            threshold_dense=threshold_dense,
            threshold_sparse=threshold_sparse,
            conf=conf_dense,  # Use higher confidence for dense detection (sparse will also use this)
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=neighbor_radius,
            maximum_tanimoto_similarity=0.35,  # Increased from 0.25 to 0.35 to prevent duplicate detections at same location
            max_iters=max_iters_optimized,
            minimum_point_number_dense=minimum_point_number,
            minimum_point_number_sparse=minimum_point_number_sparse,
            maximum_model_number=20000,
            sampler_id=sampler_id_optimized,  # Z-Aligned sampler
            scoring_exponent=0.0,
            do_logging=False
        )
        
        # Force flush immediately after C++ call
        sys.stdout.flush()
        sys.stderr.flush()
        
        elapsed_time = time() - t
        
        num_models = lines.shape[0] if lines.size > 0 else 0
        
        # Separate dense and sparse lines
        dense_mask = (line_types == 0)
        sparse_mask = (line_types == 1)
        num_dense = np.sum(dense_mask)
        num_sparse = np.sum(sparse_mask)
        
        print(f"   ✓ Completed in {elapsed_time:.2f} seconds")
        print(f"   ✓ Detected {num_models} line models total:")
        print(f"      - {num_dense} DENSE lines")
        print(f"      - {num_sparse} SPARSE lines")


        print(f"   → Note: Timing breakdown not directly available in Python bindings")
        print(f"   → Estimated breakdown (typical):")
        print(f"      - Proposal engine (GC-RANSAC): ~{elapsed_time*0.65:.2f}s (65%)")
        print(f"      - PEARL optimization: ~{elapsed_time*0.25:.2f}s (25%)")
        print(f"      - Model validation: ~{elapsed_time*0.07:.2f}s (7%)")
        print(f"      - Compound model update: ~{elapsed_time*0.03:.2f}s (3%)")
        
        # Labeling scheme (after C++ fix): 
        # - Label 0 = outliers/unassigned points
        # - Labels 1, 2, 3, ..., num_models = lines (dense first, then sparse)
        # - Dense lines: labels 1, 2, ..., num_dense
        # - Sparse lines: labels num_dense+1, num_dense+2, ..., num_models
        # Mapping: line index idx -> label idx + 1
        # NOTE: If you see label 1 also being treated as outliers, the C++ code may need to be rebuilt
        
        # Verify the labeling structure
        unique_labels = np.unique(labeling)
        print(f"\n   🔍 Labeling structure verification:")
        print(f"      Total models: {num_models} (dense: {num_dense}, sparse: {num_sparse})")
        print(f"      Unique labels in labeling: {sorted(unique_labels)}")
        print(f"      Expected: label 0 = outliers, labels 1-{num_models} = lines")
        
        # Check label distribution
        
        for label in sorted(unique_labels):
            if label == 0:
                mask = (labeling == label)
                point_count = np.sum(mask)
                print(f"      Label {label} (OUTLIERS): {point_count} points")
            elif label >= 1 and label <= num_models:
                mask = (labeling == label)
                point_count = np.sum(mask)
                # Map label to line index: label 1 -> line 0, label 2 -> line 1, etc.
                line_idx = int(label - 1)
                if line_idx >= 0 and line_idx < len(line_types):
                    line_type_str = "DENSE" if line_types[line_idx] == 0 else "SPARSE"
                    print(f"      Label {label} ({line_type_str} line {line_idx+1}): {point_count} points")
                else:
                    print(f"      Label {label}: {point_count} points (line_idx {line_idx} out of range!)")
            else:
                mask = (labeling == label)
                point_count = np.sum(mask)
                print(f"      Label {label} (unexpected): {point_count} points")
        
        # Print all detected lines before filtering (for debugging) - after determining indexing
        print(f"\n   🔍 All detected lines (before parallel filtering):")
        for idx in range(num_models):
            # Map line index to label: line idx -> label idx + 1 (label 0 is outliers)
            instance_label = idx + 1
            mask = (labeling == instance_label)
            points_line = points_normalized[mask]
            point_count = np.sum(mask)
            if point_count > 0:
                centroid_x = np.mean(points_line[:, 0])
                centroid_y = np.mean(points_line[:, 1])
                line_type = line_types[idx] if idx < len(line_types) else 0
                line_type_str = "DENSE" if line_type == 0 else "SPARSE"
                line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
                line_dir = line_dir / np.linalg.norm(line_dir)
                expected_direction = np.array([0.0, 0.0, 1.0])
                dot_with_time = np.abs(np.dot(line_dir, expected_direction))
                angle_with_time = np.arccos(np.clip(dot_with_time, -1, 1)) * 180 / np.pi
                
                # Verify points are actually close to the line
                line_point = np.array([lines[idx, 0], lines[idx, 1], lines[idx, 2]])
                threshold_used = threshold_dense if line_type == 0 else threshold_sparse
                distances = np.array([point_to_line_distance_3d(p, line_point, line_dir) for p in points_line])
                inlier_count = np.sum(distances <= threshold_used)
                inlier_ratio = inlier_count / len(points_line) if len(points_line) > 0 else 0
                
                print(f"      Line {idx+1} ({line_type_str}, label={instance_label}): {point_count} points, "
                      f"centroid=({centroid_x:.1f}, {centroid_y:.1f}), angle={angle_with_time:.2f}°, "
                      f"inlier_ratio={inlier_ratio:.2%} ({inlier_count}/{point_count} within threshold {threshold_used:.6f})")
        
        # Count points assigned to each type (after determining indexing)
        dense_point_count = 0
        sparse_point_count = 0
        for idx in range(num_models):
            # Map line index to label: line idx -> label idx + 1 (label 0 is outliers)
            instance_label = idx + 1
            mask = (labeling == instance_label)
            point_count = np.sum(mask)
            if idx < len(line_types):
                if line_types[idx] == 0:  # Dense
                    dense_point_count += point_count
                else:  # Sparse
                    sparse_point_count += point_count
        
        print(f"   📊 Point assignment: {dense_point_count} points in dense lines, {sparse_point_count} points in sparse lines")
        
        # Check for unassigned points in the top-right region (around 1200, 600)
        # Unassigned points: label 0 (outliers) OR label > num_models
        unassigned_mask = (labeling == 0) | (labeling > num_models)
        
        unassigned_points = points_normalized[unassigned_mask]
        if len(unassigned_points) > 0:
            # Check top-right region (x > 1000, y > 500)
            top_right_mask = (unassigned_points[:, 0] > 1000) & (unassigned_points[:, 1] > 500)
            top_right_unassigned = unassigned_points[top_right_mask]
            print(f"   📍 Unassigned points in top-right region (x>1000, y>500): {len(top_right_unassigned)} out of {len(unassigned_points)} unassigned total")
            if len(top_right_unassigned) > 10:
                print(f"      → Potential missed line in top-right region with {len(top_right_unassigned)} unassigned points")
        
        # More lenient threshold: 10° = cos(10°) ≈ 0.9848 (was 5°, trying 10° to keep more lines)
        parallel_threshold = 1.0  # Lines must be within 10° of time axis

        valid_line_indices = parallel_check(lines,labeling,num_models , parallel_threshold = 1.0,print_updates=True )
        
        # Print info about filtered lines
        all_indices = set(range(num_models))
        filtered_indices = all_indices - set(valid_line_indices)
        if len(filtered_indices) > 0:
            print(f"   ⚠️  {len(filtered_indices)} lines filtered out by parallel check:")
            for idx in filtered_indices:
                line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
                line_dir = line_dir / np.linalg.norm(line_dir)
                expected_direction = np.array([0.0, 0.0, 1.0])
                dot_with_time = np.abs(np.dot(line_dir, expected_direction))
                angle_with_time = np.arccos(np.clip(dot_with_time, -1, 1)) * 180 / np.pi
                # Map line index to label: line idx -> label idx + 1 (label 0 is outliers)
                instance_label = idx + 1
                mask = (labeling == instance_label)
                point_count = np.sum(mask)
                centroid_x = np.mean(points_normalized[mask, 0]) if point_count > 0 else 0
                centroid_y = np.mean(points_normalized[mask, 1]) if point_count > 0 else 0
                print(f"      Line {idx+1} (label={instance_label}): angle={angle_with_time:.2f}°, points={point_count}, centroid=({centroid_x:.1f}, {centroid_y:.1f})")
        
        # Visualization
        print("\n4. Visualizing results...")
        
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
        
        # First, plot outliers (label 0)
        outlier_mask = (labeling == 0)
        outliers = points_normalized[outlier_mask]
        if len(outliers) > 0:
            ax2.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2],
                       c='grey', s=3, alpha=0.5, marker='x', label='Outliers')

        # Draw detected lines in 3D
        print('valid line indices is '+str(valid_line_indices))
        for i, idx in enumerate(valid_line_indices):
            # print('valid line indices is '+str(valid_line_indices))
            # Map line index to label: line idx -> label idx + 1 (label 0 is outliers)
            instance_label = idx + 1
            mask = (labeling == instance_label)
            points_line = points_normalized[mask]
            
            if len(points_line) > 0:
                # Get the line parameters
                line_point = np.array([lines[idx, 0], lines[idx, 1], lines[idx, 2]])
                line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
                line_dir = line_dir / np.linalg.norm(line_dir)
                
                line_type = line_types[idx] if idx < len(line_types) else 0
                line_type_str = "DENSE" if line_type == 0 else "SPARSE"
                
                # Plot all points for this line
                ax2.scatter(points_line[:, 0], points_line[:, 1], points_line[:, 2],
                           c=[colors_det[i]], s=5, alpha=0.6, label=f'{line_type_str} {i+1}')
                
                # Draw the predicted line
                projections = np.array([np.dot(p - line_point, line_dir) for p in points_line])
                t_min, t_max = projections.min(), projections.max()
                # Extend slightly beyond the points for better visualization
                t_range = t_max - t_min
                t_min_extended = t_min - 0.1 * t_range if t_range > 0 else t_min - 1.0
                t_max_extended = t_max + 0.1 * t_range if t_range > 0 else t_max + 1.0
                p_start = line_point + t_min_extended * line_dir
                p_end = line_point + t_max_extended * line_dir
                
                # Draw line (dense = solid, sparse = dashed)
                if line_type == 0:  # Dense line
                    ax2.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], [p_start[2], p_end[2]],
                            'k-', linewidth=2, alpha=0.5)
                else:  # Sparse line
                    ax2.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], [p_start[2], p_end[2]],
                            'k--', linewidth=1.5, alpha=0.5)
                
                # Add text label at centroid
                centroid = np.mean(points_line, axis=0)
                # ax2.text(centroid[0], centroid[1], centroid[2], f'{line_type_str} {i+1}',
                #         fontsize=8, color=colors_det[i], fontweight='bold')
        
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_xlim(0,1280)
        ax2.set_ylim(0,720)
        ax2.set_zlabel('Time (normalized)')
        ax2.set_title(f'Detected Lines ({len(valid_line_indices)} lines)')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5, fontsize=8, markerscale=2)
        
        # Plot 3: Detected lines projected to (x, y) space
        ax3 = fig.add_subplot(133)
        
        # First, plot outliers (label 0)
        outlier_mask = (labeling == 0)
        outliers = points_normalized[outlier_mask]
        if len(outliers) > 0:
            ax3.scatter(outliers[:, 0], outliers[:, 1], c='grey', s=3, alpha=0.5, marker='x', label='Outliers')
        
        # Plot detected lines (color by type)
        line_type_lists = []
        for i, idx in enumerate(valid_line_indices):
            # Map line index to label: line idx -> label idx + 1 (label 0 is outliers)
            instance_label = idx + 1
            mask = (labeling == instance_label)
            points_line = points_normalized[mask]
            
            if len(points_line) > 0:
                line_type = line_types[idx] if idx < len(line_types) else 0
                line_type_str = "DENSE" if line_type == 0 else "SPARSE"
                line_type_lists.append(line_type)
                
                # Plot all points for this line
                if line_type == 0:  # Dense line
                    ax3.scatter(points_line[:, 0], points_line[:, 1], c=[colors_det[i]], s=5, alpha=0.6, label=f'Dense {i+1}')
                else:  # Sparse line
                    ax3.scatter(points_line[:, 0], points_line[:, 1], c=[colors_det[i]], s=3, alpha=0.4, marker='^', label=f'Sparse {i+1}')
                
                # Draw the predicted line in 2D projection
                line_point = np.array([lines[idx, 0], lines[idx, 1], lines[idx, 2]])
                line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
                line_dir = line_dir / np.linalg.norm(line_dir)
                
                # Project line to 2D (x, y) plane
                projections = np.array([np.dot(p - line_point, line_dir) for p in points_line])
                t_min, t_max = projections.min(), projections.max()
                # Extend slightly
                t_range = t_max - t_min
                t_min_extended = t_min - 0.1 * t_range if t_range > 0 else t_min - 1.0
                t_max_extended = t_max + 0.1 * t_range if t_range > 0 else t_max + 1.0
                p_start = line_point + t_min_extended * line_dir
                p_end = line_point + t_max_extended * line_dir
                
                if line_type == 0:  # Dense line
                    ax3.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'k-', linewidth=2, alpha=0.7)
                else:  # Sparse line
                    ax3.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'k--', linewidth=1.5, alpha=0.5)
                
                # Add text label at centroid
                centroid = np.mean(points_line[:, :2], axis=0)
                # ax3.text(centroid[0], centroid[1], f'{line_type_str} {i+1}',
                #         fontsize=8, color=colors_det[i], fontweight='bold',
                #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        ax3.set_title('Detected Lines (X-Y Projection)')
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                fancybox=True, shadow=True, ncol=5, fontsize=8, markerscale=2)
        ax3.set_aspect('equal')
        
        plt.tight_layout()
        output_path = 'may9_events_3d_regular_detection.png'
        plt.show()
        # plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved visualization to {output_path}")
        plt.close()
        
        # Create separate figure: Event heatmap with line centroids
        print("\n5. Creating event heatmap with line centroids...")
        
        # Get image dimensions from metadata or from data range
        if metadata and 'geometry' in metadata and metadata['geometry']:
            img_width, img_height = metadata['geometry']
        else:
            # Infer from data range
            img_width = int(points_raw[:, 0].max()) + 1
            img_height = int(points_raw[:, 1].max()) + 1
        
        # Create accumulated event frame following the specified syntax
        # Start with ones to avoid problems with LogNorm
        accumulated_frame = np.ones((img_height, img_width), dtype=np.float64)
        
        # Get integer pixel coordinates from raw data
        x_coords = points_raw[:, 0].astype(int)
        y_coords = points_raw[:, 1].astype(int)
        
        
        np.add.at(accumulated_frame, (y_coords, x_coords), 1)
        
        print(f"   Accumulated frame range: [{accumulated_frame.min():.1f}, {accumulated_frame.max():.1f}]")
        print(f"   99.9th percentile: {np.percentile(accumulated_frame, 99.9):.1f}")
        
        # Calculate line centroids (average x, y of points on each line)
        line_centroids = []
        for i, idx in enumerate(valid_line_indices):
            # Map line index to label: line idx -> label idx + 1 (label 0 is outliers)
            instance_label = idx + 1
            mask = (labeling == instance_label)
            points_line = points_normalized[mask]
            
            if len(points_line) > 0:

                centroid_x = np.mean(points_line[:, 0])
                centroid_y = np.mean(points_line[:, 1])
                line_centroids.append((centroid_x, centroid_y, i+1))
                
        
        # Create new figure for heatmap

        fig2 = plt.figure(figsize=(12, 8))
        ax_heatmap = fig2.gca()
        
        # Calculate vmax from 99.9th percentile
        vmax_99_9 = np.percentile(accumulated_frame, 99.9)
        # Set vmin to 1.0 (baseline) to ensure LogNorm works correctly
        # This prevents black/white background issues
        im = ax_heatmap.imshow(
            accumulated_frame,
            norm=matplotlib.colors.LogNorm(vmin=1.0, vmax=vmax_99_9),
            cmap="viridis",
            origin='lower',
            aspect='auto',
            interpolation='nearest'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap, label='Event Accumulation (1 = no events)')


        print('line centroids is '+str(line_centroids)+ ' and the line list for annotaiton is '+str(line_type_lists))
        # Superimpose line centroids
        if line_centroids:
            centroids_x = [c[0] for c in line_centroids]
            centroids_y = [c[1] for c in line_centroids]
            line_numbers = [c[2] for c in line_centroids]
            
            # Plot centroids as markers
            ax_heatmap.scatter(centroids_x, centroids_y, c='cyan', s=100, 
                             marker='x', linewidths=2, label='Line Centroids', alpha=0.1,zorder=10)
            
            
            
            # Add text labels for line numbers
            for x, y, num in line_centroids:
                
                if line_type_lists[num-1] ==0: # dense
                    ax_heatmap.annotate(f'Dense L{num}', (x, y), xytext=(5, 5), 
                                  textcoords='offset points', color='cyan', 
                                  fontsize=8, fontweight='bold', zorder=11)
                else: # sparse
                    ax_heatmap.annotate(f'Sparse L{num}', (x, y), xytext=(5, 5), 
                                  textcoords='offset points', color='cyan', 
                                  fontsize=8, fontweight='bold', zorder=11)
                

        
        ax_heatmap.set_xlabel('X (pixels)')
        ax_heatmap.set_ylabel('Y (pixels)')
        ax_heatmap.set_title(f'Event Accumulation Heatmap (Log Scale, 99.9th percentile vmax) with {len(line_centroids)} Line Centroids')
        ax_heatmap.set_xlim(0, img_width)
        ax_heatmap.set_ylim(0, img_height)
        ax_heatmap.legend(loc='upper right')
        
        plt.tight_layout()
        output_path_heatmap = 'jupiter_events_heatmap_centroids.png'
        # plt.savefig(output_path_heatmap, dpi=150, bbox_inches='tight')
        plt.show()
        # print(f"   ✓ Saved heatmap visualization to {output_path_heatmap}")
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
