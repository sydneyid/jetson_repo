#!/usr/bin/env python3
"""
Test Progressive-X 3D Line Detection on HDF5 Event Data
Reads events from HDF5 file and treats (x, y, t) as a 3D point cloud to detect lines

IMPORTANT: Due to a gflags conflict in conda environments, you MUST run this script with:
    python test_hdf5_events_3d.py 2>/dev/null
    
The gflags error occurs during C++ module loading and causes the process to exit.
Redirecting stderr at the shell level is the only reliable way to suppress it.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import h5py
from time import time
import warnings

# Suppress gflags error by redirecting stderr at file descriptor level
# The error occurs during C++ shared library loading (dlopen), so we need
# to redirect stderr at the OS level, not just Python's sys.stderr
# Note: gflags calls exit() on duplicate flags, which terminates the process.
# This redirection suppresses the error message but the process may still exit.
import sys
import os

# Add parent directory to path to import pyprogressivex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Redirect stderr at file descriptor level (works for C++ code too)
_stderr_fd = sys.stderr.fileno()
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
_stderr_dup = os.dup(_stderr_fd)  # Save original stderr

try:
    # Redirect stderr to devnull
    os.dup2(_devnull_fd, _stderr_fd)
    # Now import (gflags error message will be suppressed)
    # Note: If gflags calls exit(), the process will still terminate
    import pyprogressivex
    # If we get here, import succeeded!
except SystemExit:
    # gflags error causes SystemExit - we can't recover from this
    # The process will exit, but at least the error message is suppressed
    raise
except Exception as e:
    # Other import errors
    os.dup2(_stderr_dup, _stderr_fd)  # Restore stderr to show error
    raise
finally:
    # Restore original stderr if we're still running
    try:
        os.dup2(_stderr_dup, _stderr_fd)
        os.close(_devnull_fd)
        os.close(_stderr_dup)
    except:
        pass  # If process is exiting, this may fail

def read_hdf5_events(hdf5_path, time_window=0.25, start_time=None):
    """
    Read events from HDF5 file and create 3D point cloud from (x, y, t)
    
    Args:
        hdf5_path: Path to HDF5 event file
        time_window: Time window in seconds to extract (default: 0.25)
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
        # According to the documentation: x, y are coordinates, p is polarity, t is timestamp (in us)
        x = events['x'][:]
        y = events['y'][:]
        t = events['t'][:]  # Timestamp in microseconds
        p = events['p'][:]  # Polarity (not used, but available)
        
        print(f"  Time range: {t.min()/1e6:.3f}s to {t.max()/1e6:.3f}s (duration: {(t.max()-t.min())/1e6:.3f}s)")
        print(f"  Spatial range: X=[{x.min()}, {x.max()}], Y=[{y.min()}, {y.max()}]")
        
        # Determine time window
        if start_time is None:
            start_time = t.min()
        else:
            # Ensure start_time is in microseconds
            if start_time < 1e6:  # Assume seconds if < 1e6
                start_time = start_time * 1e6
        
        end_time = start_time + time_window * 1e6  # Convert seconds to microseconds
        
        # Filter events within time window
        time_mask = (t >= start_time) & (t < end_time)
        x_filtered = x[time_mask]
        y_filtered = y[time_mask]
        t_filtered = t[time_mask]
        p_filtered = p[time_mask]
        
        print(f"  Events in time window [{start_time/1e6:.3f}s, {end_time/1e6:.3f}s]: {len(x_filtered)}")
        
        # Create 3D point cloud: (x, y, t)
        # Normalize time to similar scale as spatial coordinates for better detection
        # We'll keep time in microseconds but may need to scale it
        points = np.column_stack([x_filtered.astype(np.float64),
                                  y_filtered.astype(np.float64),
                                  t_filtered.astype(np.float64)])
        
        # Read metadata
        metadata = {}
        for attr in f.attrs:
            metadata[attr] = f.attrs[attr]
        
        # Try to get geometry if available
        if 'geometry' in f.attrs:
            geometry = f.attrs['geometry']
            if isinstance(geometry, bytes):
                geometry = geometry.decode('utf-8')
            metadata['geometry'] = geometry
            print(f"  Geometry: {geometry}")
        
        return points, metadata

def normalize_time_coordinate(points, time_scale_factor=1e-3):
    """
    Normalize time coordinate to similar scale as spatial coordinates
    
    Args:
        points: N×3 array [x, y, t]
        time_scale_factor: Factor to scale time (default: 1e-3 means 1ms = 1 unit)
    
    Returns:
        points_normalized: N×3 array with normalized time
        time_offset: Offset that was subtracted
        time_scale: Scale factor applied
    """
    # Subtract minimum time to start from 0
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
    print("Progressive-X 3D Line Detection on HDF5 Event Data")
    print("="*70)
    
    # Read HDF5 file
    hdf5_path = 'img/events.hdf5'
    
    if not os.path.exists(hdf5_path):
        print(f"✗ Error: File not found: {hdf5_path}")
        return 1
    
    print("\n1. Reading HDF5 event file...")
    try:
        points_raw, metadata = read_hdf5_events(hdf5_path, time_window=0.25, start_time=None)
        print(f"   ✓ Loaded {len(points_raw)} events")
    except Exception as e:
        print(f"   ✗ Error reading HDF5 file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Normalize time coordinate for better detection
    # Time is in microseconds, so we scale it down to similar range as x, y
    print("\n2. Normalizing coordinates...")
    
    # Determine appropriate time scale factor
    # For event data, we want time to be on a similar scale to spatial coordinates
    # If x, y are in pixels (e.g., 0-640), we should scale time appropriately
    spatial_range = max(points_raw[:, 0].max() - points_raw[:, 0].min(),
                       points_raw[:, 1].max() - points_raw[:, 1].min())
    time_range = points_raw[:, 2].max() - points_raw[:, 2].min()
    
    # For 0.25 seconds of data, time_range should be ~250,000 microseconds
    # Scale time so that 1 unit in time dimension ≈ 1 pixel in spatial dimension
    # This makes the 3D space more isotropic for line detection
    if time_range > 0:
        # Scale so that time dimension has similar magnitude to spatial dimensions
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
    
    # Compute optimal parameters
    print("\n3. Computing optimal parameters...")
    
    from scipy.spatial.distance import cdist
    n_sample = min(100, len(points_normalized))
    sample_indices = np.random.choice(len(points_normalized), n_sample, replace=False)
    sample_points = points_normalized[sample_indices]
    dists = cdist(sample_points, sample_points)
    np.fill_diagonal(dists, np.inf)
    nearest_dists = np.min(dists, axis=1)
    estimated_noise = np.median(nearest_dists)
    
    data_range = np.max(points_normalized.max(axis=0) - points_normalized.min(axis=0))
    
    # Set threshold - for event data, lines should have very little noise
    # True lines form tight clusters with points very close to the line
    # Use a VERY STRICT threshold to avoid grouping noise into lines
    # Event data accumulates over time, so true lines should be very tight
    threshold_from_range = data_range * 0.003  # 0.3% of range (very strict)
    threshold_from_noise = estimated_noise * 1.2  # 1.2× noise (very strict)
    final_threshold = max(threshold_from_range, threshold_from_noise)
    final_threshold = max(final_threshold, data_range * 0.001)  # Minimum 0.1% of range
    final_threshold = min(final_threshold, data_range * 0.015)  # Maximum 1.5% (very strict)
    
    neighbor_radius = data_range * 0.005  # 0.5% of range (smaller for tighter grouping)
    neighbor_radius = max(neighbor_radius, estimated_noise * 0.3)
    neighbor_radius = min(neighbor_radius, data_range * 0.02)
    
    # Minimum point number - for event data, lines form larger clusters over time
    # Require substantial clusters to avoid false positives from noise
    # But not too high, or we'll merge separate parallel lines
    # Balance: enough to be a real line, but not so much that we merge lines
    # For event data, each line should be a distinct cluster, not a huge merged group
    minimum_point_number = max(40, int(len(points_normalized) * 0.008))  # 0.8% of points, at least 40
    
    print(f"   ✓ Estimated noise: {estimated_noise:.6f}")
    print(f"   ✓ Threshold: {final_threshold:.6f} ({100*final_threshold/data_range:.2f}% of range)")
    print(f"   ✓ Neighborhood radius: {neighbor_radius:.6f}")
    print(f"   ✓ Minimum points per model: {minimum_point_number}")
    
    # Run Progressive-X 3D line detection with temporal constraint
    print("\n4. Running Progressive-X 3D line detection (with temporal ordering constraint)...")
    
    try:
        t = time()
        lines, labeling = pyprogressivex.findLines3DTemporal(
            np.ascontiguousarray(points_normalized, dtype=np.float64),
            np.ascontiguousarray([], dtype=np.float64),
            threshold=final_threshold,
            conf=0.995,  # Very high confidence - be very selective, only accept high-quality lines
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=neighbor_radius,
            maximum_tanimoto_similarity=0.95,  # Slight merging tolerance - helps separate very close parallel lines
            max_iters=8000,  # Many iterations to find all parallel lines
            minimum_point_number=minimum_point_number,
            maximum_model_number=100,  # Allow many parallel lines
            sampler_id=0,
            scoring_exponent=2.0,  # Strongly favor larger, cleaner clusters
            do_logging=False
        )
        elapsed_time = time() - t
        
        num_models = lines.shape[0] if lines.size > 0 else 0
        
        print(f"   ✓ Completed in {elapsed_time:.2f} seconds")
        print(f"   ✓ Detected {num_models} line models (before filtering)")
        
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
        
        # Post-process: Filter out poor quality lines and very large lines (likely merged)
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
            # A line shouldn't be more than 2.5× the median size (likely merged)
            # For event data, lines should be similar in size (they accumulate similarly)
            max_reasonable_size = max(median_line_size * 2.5, mean_line_size * 2.0)
            print(f"   Line size stats: median={median_line_size:.0f}, mean={mean_line_size:.0f}, max_reasonable={max_reasonable_size:.0f}")
        else:
            max_reasonable_size = len(points_normalized) * 0.15
        
        valid_line_indices = []
        max_points_per_line = len(points_normalized) * 0.12  # Absolute max: 12% of points (stricter)
        
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
            
            # Filter criteria: must have high inlier ratio and reasonable size
            is_valid = True
            reasons = []
            
            if inlier_ratio < 0.90:  # At least 90% of points must be inliers (stricter for event data)
                is_valid = False
                reasons.append(f"low inlier ratio ({inlier_ratio*100:.1f}%)")
            
            if mean_distance > final_threshold * 0.3:  # Mean distance should be well below threshold (stricter)
                is_valid = False
                reasons.append(f"high mean distance ({mean_distance:.2f} vs threshold {final_threshold:.2f})")
            
            # Check absolute maximum (12% of total points)
            if len(points_assigned) > max_points_per_line:
                is_valid = False
                reasons.append(f"too many points ({len(points_assigned)} = {100*len(points_assigned)/len(points_normalized):.1f}% > {100*max_points_per_line/len(points_normalized):.1f}%)")
            
            # Check relative to other lines (shouldn't be much larger than median)
            if len(line_sizes) > 1 and len(points_assigned) > max_reasonable_size:
                is_valid = False
                reasons.append(f"unusually large ({len(points_assigned)} vs median {median_line_size:.0f}, ratio={len(points_assigned)/median_line_size:.1f}×)")
            
            if len(points_assigned) < minimum_point_number:  # Too few points
                is_valid = False
                reasons.append(f"too few points ({len(points_assigned)} < {minimum_point_number})")
            
            if is_valid:
                valid_line_indices.append(idx)
            else:
                print(f"   ✗ Filtered out line {idx+1}: {', '.join(reasons)}")
                # Reassign filtered points to outliers
                labeling[mask] = (num_models if use_zero_indexed else 0)
        
        print(f"   ✓ Kept {len(valid_line_indices)} high-quality lines out of {num_models} detected")
        
        # Update num_models to reflect filtered lines
        original_num_models = num_models
        num_models = len(valid_line_indices)
        
        # Print filtered labeling statistics
        unique_labels, counts = np.unique(labeling, return_counts=True)
        print(f"\n   Filtered labeling statistics:")
        outlier_count = 0
        for label, count in zip(unique_labels, counts):
            if (use_zero_indexed and label == original_num_models) or (not use_zero_indexed and label == 0):
                outlier_count += count
            elif label > 0 if not use_zero_indexed else label < original_num_models:
                # This is a valid line label
                line_idx = label if use_zero_indexed else label - 1
                if line_idx in valid_line_indices:
                    print(f"     Line {valid_line_indices.index(line_idx)+1}: {count} points ({100*count/len(points_normalized):.1f}%)")
        print(f"     Outliers: {outlier_count} points ({100*outlier_count/len(points_normalized):.1f}%)")
        
        # Extract detected lines and check for parallelism (only valid ones)
        detected_lines = []
        line_directions = []
        
        for idx in valid_line_indices:
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            
            line_point = np.array([lines[idx, 0], lines[idx, 1], lines[idx, 2]])
            line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
            line_dir = line_dir / np.linalg.norm(line_dir)
            
            detected_lines.append({
                'point': line_point,
                'direction': line_dir,
                'label': instance_label
            })
            line_directions.append(line_dir)
            
            # Calculate statistics for this line
            mask = (labeling == instance_label)
            points_assigned = points_normalized[mask]
            
            if len(points_assigned) > 0:
                distances = [point_to_line_distance_3d(p, line_point, line_dir) for p in points_assigned]
                distances = np.array(distances)
                inlier_count = np.sum(distances <= final_threshold)
                inlier_ratio = inlier_count / len(distances)
                
                # Calculate line extent
                vecs = points_assigned - line_point
                projections = np.dot(vecs, line_dir)
                extent = projections.max() - projections.min() if len(projections) > 0 else 0
                
                print(f"   Line {idx+1}: {len(points_assigned)} points, "
                      f"{inlier_ratio*100:.1f}% within threshold, "
                      f"mean distance: {distances.mean():.4f}, "
                      f"extent: {extent:.2f}")
                print(f"      Direction: ({line_dir[0]:.4f}, {line_dir[1]:.4f}, {line_dir[2]:.4f})")
        
        # Check parallelism - all lines should have similar directions
        if len(line_directions) > 1:
            print(f"\n   Checking parallelism of detected lines...")
            # Compute pairwise dot products (should be close to 1 for parallel lines)
            for i in range(len(line_directions)):
                for j in range(i+1, len(line_directions)):
                    dot_product = np.abs(np.dot(line_directions[i], line_directions[j]))
                    angle_deg = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
                    print(f"      Lines {i+1} & {j+1}: dot={dot_product:.4f}, angle={angle_deg:.2f}° "
                          f"{'✓ parallel' if angle_deg < 10 else '✗ not parallel'}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Add diagnostic: visualize what lines look like in the data
    print("\n5. Diagnostic: Analyzing detected lines...")
    
    if num_models > 0:
        print(f"   Analyzing {num_models} detected lines...")
        
        # Check if lines are actually visible in the 3D plot
        # For each detected line, check how well it fits the data
        for idx, det_line in enumerate(detected_lines):
            mask = (labeling == det_line['label'])
            if np.any(mask):
                points_on_line = points_normalized[mask]
                
                # Calculate how "line-like" these points are
                # Project points onto the line
                vecs = points_on_line - det_line['point']
                projections = np.dot(vecs, det_line['direction'])
                
                # Calculate perpendicular distances
                distances = []
                for p in points_on_line:
                    dist = point_to_line_distance_3d(p, det_line['point'], det_line['direction'])
                    distances.append(dist)
                distances = np.array(distances)
                
                # Calculate spread along the line
                proj_range = projections.max() - projections.min()
                proj_std = projections.std()
                
                print(f"   Line {idx+1}:")
                print(f"     Points: {len(points_on_line)}")
                print(f"     Mean distance to line: {distances.mean():.4f} (threshold: {final_threshold:.4f})")
                print(f"     Distance std: {distances.std():.4f}")
                print(f"     Projection range: {proj_range:.2f}")
                print(f"     Projection std: {proj_std:.2f}")
                print(f"     Direction: ({det_line['direction'][0]:.4f}, {det_line['direction'][1]:.4f}, {det_line['direction'][2]:.4f})")
                
                # Check if this looks like a valid line
                if distances.mean() > final_threshold * 2:
                    print(f"     ⚠️  WARNING: Mean distance ({distances.mean():.4f}) is much larger than threshold!")
                if proj_range < data_range * 0.1:
                    print(f"     ⚠️  WARNING: Line extent ({proj_range:.2f}) is very short!")
        
        # Check parallelism - all lines should have similar directions
        if len(line_directions) > 1:
            print(f"\n   Checking parallelism of detected lines...")
            avg_direction = np.mean(line_directions, axis=0)
            avg_direction = avg_direction / np.linalg.norm(avg_direction)
            print(f"   Average direction: ({avg_direction[0]:.4f}, {avg_direction[1]:.4f}, {avg_direction[2]:.4f})")
            
            # Compute pairwise dot products (should be close to 1 for parallel lines)
            parallelism_scores = []
            for i in range(len(line_directions)):
                for j in range(i+1, len(line_directions)):
                    dot_product = np.abs(np.dot(line_directions[i], line_directions[j]))
                    angle_deg = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
                    parallelism_scores.append(dot_product)
                    print(f"      Lines {i+1} & {j+1}: dot={dot_product:.4f}, angle={angle_deg:.2f}° "
                          f"{'✓ parallel' if angle_deg < 10 else '✗ not parallel'}")
            
            avg_parallelism = np.mean(parallelism_scores) if parallelism_scores else 0
            print(f"   Average parallelism score: {avg_parallelism:.4f} (1.0 = perfectly parallel)")
            
            if avg_parallelism < 0.9:
                print(f"   ⚠️  WARNING: Lines are not very parallel! Expected parallel lines.")
    
    # Visualize results
    print("\n6. Visualizing results...")
    
    # Define colors for detected lines (needed for both figures)
    colors_det = plt.cm.tab10(np.linspace(0, 1, num_models)) if num_models > 0 else []
    
    # Create separate figure for event density heatmap
    fig_heatmap = plt.figure(figsize=(12, 10))
    ax_heatmap = fig_heatmap.add_subplot(111)
    
    # Create 2D histogram of events in x-y space
    x_min, x_max = points_raw[:, 0].min(), points_raw[:, 0].max()
    y_min, y_max = points_raw[:, 1].min(), points_raw[:, 1].max()
    
    # Use a reasonable number of bins based on data resolution
    # For event data, use finer bins for better visualization
    n_bins_x = int((x_max - x_min) / 2)  # ~2 pixel bins
    n_bins_y = int((y_max - y_min) / 2)
    n_bins_x = max(50, min(200, n_bins_x))  # Between 50 and 200 bins
    n_bins_y = max(50, min(200, n_bins_y))
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        points_raw[:, 0], points_raw[:, 1],
        bins=[n_bins_x, n_bins_y],
        range=[[x_min, x_max], [y_min, y_max]]
    )
    
    # Transpose to match imshow convention (y, x)
    hist = hist.T
    
    # Apply log scaling with 99.5 percentile clipping
    hist_flat = hist.flatten()
    hist_flat_positive = hist_flat[hist_flat > 0]
    if len(hist_flat_positive) > 0:
        percentile_99_5 = np.percentile(hist_flat_positive, 99.5)
        # Clip to 99.5 percentile and apply log
        hist_clipped = np.clip(hist, 0, percentile_99_5)
        hist_log = np.log1p(hist_clipped)  # log1p = log(1+x) to handle zeros
    else:
        hist_log = hist
    
    # Plot with viridis colormap on black background
    ax_heatmap.set_facecolor('black')
    im = ax_heatmap.imshow(hist_log, 
                    extent=[x_min, x_max, y_min, y_max],
                    origin='lower',
                    cmap='viridis',
                    aspect='auto',
                    interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, label='Log Event Count (99.5% clipped)')
    cbar.ax.set_facecolor('black')
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Log Event Count (99.5% clipped)', color='white')
    
    # Overlay detected lines on the heatmap
    for idx, det_line in enumerate(detected_lines):
        mask = (labeling == det_line['label'])
        if np.any(mask):
            points_on_line = points_raw[mask]  # Use raw points for x-y projection
            
            # Project 3D line to x-y plane
            dir_3d = det_line['direction']
            dir_xy = dir_3d[:2]  # x, y components
            dir_xy_norm = np.linalg.norm(dir_xy)
            
            if dir_xy_norm > 1e-6:
                dir_xy = dir_xy / dir_xy_norm
                
                # Find the actual extent of points in the direction perpendicular to line
                center_xy = points_on_line[:, :2].mean(axis=0)
                vecs_to_center = points_on_line[:, :2] - center_xy
                projs = np.dot(vecs_to_center, dir_xy)
                
                # Draw line through the extent of points
                proj_min, proj_max = projs.min(), projs.max()
                # Extend slightly
                proj_min -= 0.1 * (proj_max - proj_min)
                proj_max += 0.1 * (proj_max - proj_min)
                
                line_x = np.array([center_xy[0] + proj_min * dir_xy[0], 
                                  center_xy[0] + proj_max * dir_xy[0]])
                line_y = np.array([center_xy[1] + proj_min * dir_xy[1], 
                                  center_xy[1] + proj_max * dir_xy[1]])
                ax_heatmap.plot(line_x, line_y, color=colors_det[idx], linewidth=4.5, 
                         linestyle='-', alpha=0.95, label=f'Line {idx+1}', zorder=10)
            else:
                # Line is mostly vertical in x-y (motion is primarily temporal)
                center_xy = points_on_line[:, :2].mean(axis=0)
                ax_heatmap.scatter([center_xy[0]], [center_xy[1]], 
                           color=colors_det[idx], s=150, marker='o', 
                           edgecolors='white', linewidths=2, 
                           label=f'Line {idx+1} (temporal)', zorder=10)
    
    ax_heatmap.set_xlabel('X (pixels)', color='white', fontsize=12)
    ax_heatmap.set_ylabel('Y (pixels)', color='white', fontsize=12)
    ax_heatmap.set_title(f'Event Density Heatmap: {num_models} lines detected', 
                 fontsize=14, fontweight='bold', color='white')
    ax_heatmap.set_aspect('equal')
    # ax_heatmap.legend(loc='lower', fontsize=9, facecolor='black', edgecolor='white', labelcolor='white')
    ax_heatmap.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    
    ax_heatmap.tick_params(colors='white')
    ax_heatmap.grid(True, alpha=0.2, color='white')
    
    # Save heatmap figure
    output_file_heatmap = 'hdf5_events_density_heatmap.png'
    fig_heatmap.savefig(output_file_heatmap, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"   ✓ Saved event density heatmap to {output_file_heatmap}")
    plt.close(fig_heatmap)
    
    # Original multi-plot figure
    fig = plt.figure(figsize=(24, 8))
    
    # Plot 1: Original event data (x, y) colored by time
    ax1 = fig.add_subplot(131)
    
    # Color by normalized time
    scatter1 = ax1.scatter(points_raw[:, 0], points_raw[:, 1], 
                          c=points_normalized[:, 2], s=1, alpha=0.5, cmap='viridis')
    plt.colorbar(scatter1, ax=ax1, label='Normalized Time')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title(f'Event Data: {len(points_raw)} events\n(0.25s window)', 
                 fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 3D view (x, y, t) - show only points assigned to lines
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Plot points colored by line assignment (not all points, to reduce clutter)
    unique_labels = np.unique(labeling)
    
    # First, plot outliers (if any) in gray
    outlier_mask = (labeling == (num_models if use_zero_indexed else 0))
    if np.any(outlier_mask):
        ax2.scatter(points_normalized[outlier_mask, 0], 
                   points_normalized[outlier_mask, 1], 
                   points_normalized[outlier_mask, 2],
                   c='gray', s=1, alpha=0.1, marker='x', label='Outliers')
    
    # Plot detected lines with their assigned points
    for idx, det_line in enumerate(detected_lines):
        mask = (labeling == det_line['label'])
        if np.any(mask):
            points_on_line = points_normalized[mask]
            # Plot points assigned to this line
            ax2.scatter(points_on_line[:, 0], points_on_line[:, 1], points_on_line[:, 2],
                       c=[colors_det[idx]], s=2, alpha=0.6, label=f'Line {idx+1} points')
            
            # Project points onto line to find extent
            vecs = points_on_line - det_line['point']
            projections = np.dot(vecs, det_line['direction'])
            t_min, t_max = projections.min(), projections.max()
            # Extend slightly beyond for visualization
            t_range = np.linspace(t_min - 0.2*(t_max-t_min), t_max + 0.2*(t_max-t_min), 200)
            line_points = det_line['point'][None, :] + t_range[:, None] * det_line['direction'][None, :]
            ax2.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                    color=colors_det[idx], linewidth=3, linestyle='-', alpha=0.9,
                    label=f'Line {idx+1}')
    
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_zlabel('Time (normalized)')
    ax2.set_title(f'3D Point Cloud: {num_models} lines detected', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=7, ncol=1)
        #
    
    plt.suptitle(f"HDF5 Event Data: {len(points_raw)} events, {num_models} lines detected",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = 'hdf5_events_3d_detection.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to {output_file}")
    plt.show()
    
    print("\n" + "="*70)
    print("✓ Test completed successfully!")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
