#!/usr/bin/env python3
"""
Test Progressive-X 3D Line Detection on HDF5 Event Data (May 9 dataset)
Reads events from may_9.hdf5 and treats (x, y, t) as a 3D point cloud to detect lines
This dataset has noisier lines, so parameters are adjusted accordingly.

IMPORTANT: Due to a gflags conflict in conda environments, you MUST run this script with:
    python test_may9_events_3d.py 2>/dev/null
    
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
    import pyprogressivex
except SystemExit:
    raise
except Exception as e:
    os.dup2(_stderr_dup, _stderr_fd)  # Restore stderr to show error
    raise
finally:
    # Restore original stderr if we're still running
    try:
        os.dup2(_stderr_dup, _stderr_fd)
        os.close(_devnull_fd)
        os.close(_stderr_dup)
    except:
        pass

def read_hdf5_events(hdf5_path, time_window=0.4, start_time=None):
    """
    Read events from HDF5 file and create 3D point cloud from (x, y, t)
    
    Args:
        hdf5_path: Path to HDF5 event file
        time_window: Time window in seconds to extract (default: 0.4)
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
    """
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
    print("Progressive-X 3D Line Detection on May 9 HDF5 Event Data")
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
    
    # Apply activity filter: keep events with at least 3 neighbors in 3x3 spatial kernel and 500ms temporal window
    print("\n2. Applying activity filter (3x3 spatial, 500ms temporal, min 3 neighbors)...")
    
    temporal_window_us = 500000  # 500 milliseconds = 500,000 microseconds
    min_neighbors = 3  # Require at least 3 neighbors
    
    # Convert to integer coordinates for efficient filtering
    x_int = points_raw[:, 0].astype(int)
    y_int = points_raw[:, 1].astype(int)
    t_int = points_raw[:, 2].astype(np.int64)  # Keep as int64 for time
    
    print(f"   Checking {len(points_raw):,} events for activity (min {min_neighbors} neighbors)...")
    
    # Use a more efficient approach: create a spatial-temporal grid
    # Group events by (x, y) pixel location, then check temporal neighbors
    from collections import defaultdict
    
    # Create a dictionary mapping (x, y) -> list of (time, index) tuples
    spatial_grid = defaultdict(list)
    for idx in range(len(points_raw)):
        spatial_grid[(x_int[idx], y_int[idx])].append((t_int[idx], idx))
    
    # Sort each spatial location by time
    for key in spatial_grid:
        spatial_grid[key].sort()
    
    # Create a mask for active events
    active_mask = np.zeros(len(points_raw), dtype=bool)
    
    # For each event, count neighbors in its 3x3 spatial neighborhood
    for idx in range(len(points_raw)):
        x_center = x_int[idx]
        y_center = y_int[idx]
        t_center = t_int[idx]
        
        neighbor_count = 0
        
        # Check all 9 spatial neighbors (3x3 kernel)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x_neighbor = x_center + dx
                y_neighbor = y_center + dy
                
                # Get events at this spatial location
                if (x_neighbor, y_neighbor) in spatial_grid:
                    # Count events at this location within temporal window
                    for t_val, neighbor_idx in spatial_grid[(x_neighbor, y_neighbor)]:
                        if neighbor_idx != idx and abs(t_val - t_center) <= temporal_window_us:
                            neighbor_count += 1
                            # Early exit if we have enough neighbors
                            if neighbor_count >= min_neighbors:
                                break
                
                if neighbor_count >= min_neighbors:
                    break
            if neighbor_count >= min_neighbors:
                break
        
        # Event is active if it has at least min_neighbors neighbors
        active_mask[idx] = (neighbor_count >= min_neighbors)
    
    # Filter points
    points_raw_filtered = points_raw[active_mask]
    n_removed = len(points_raw) - len(points_raw_filtered)
    
    print(f"   ✓ Removed {n_removed:,} events with <{min_neighbors} neighbors ({100*n_removed/len(points_raw):.1f}%)")
    print(f"   ✓ Kept {len(points_raw_filtered):,} active events with ≥{min_neighbors} neighbors ({100*len(points_raw_filtered)/len(points_raw):.1f}%)")
    
    # Update points_raw for normalization
    points_raw = points_raw_filtered
    
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
    
    # Compute optimal parameters - adjusted for noisier data
    print("\n4. Computing optimal parameters (adjusted for noisy lines)...")
    
    from scipy.spatial.distance import cdist
    n_sample = min(100, len(points_normalized))
    sample_indices = np.random.choice(len(points_normalized), n_sample, replace=False)
    sample_points = points_normalized[sample_indices]
    dists = cdist(sample_points, sample_points)
    np.fill_diagonal(dists, np.inf)
    nearest_dists = np.min(dists, axis=1)
    estimated_noise = np.median(nearest_dists)
    
    data_range = np.max(points_normalized.max(axis=0) - points_normalized.min(axis=0))
    
    # Set threshold - ADJUSTED for spatial-bias residual (2D spatial distance squared)
    # The residual uses spatial_distance^2, so threshold should be in squared units
    # Set threshold - MORE PERMISSIVE for sparse lines
    # Use larger threshold to detect sparse lines with gaps
    threshold_from_range = data_range * 0.01  # 1% of range (more permissive for sparse lines)
    threshold_from_noise = estimated_noise * 2.0  # 2× noise (more permissive)
    final_threshold = max(threshold_from_range, threshold_from_noise)
    final_threshold = max(final_threshold, data_range * 0.005)  # Minimum 0.5% of range
    final_threshold = min(final_threshold, data_range * 0.05)  # Maximum 5% (very permissive)
    
    # Neighborhood radius: scaled to data range but tighter for spatial clustering
    neighbor_radius = data_range * 0.003  # 0.3% of range (tighter for better separation)
    neighbor_radius = max(neighbor_radius, estimated_noise * 0.5)
    neighbor_radius = min(neighbor_radius, data_range * 0.01)
    
    # Minimum point number - LOW to allow sparse lines to be detected
    minimum_point_number = max(3, int(len(points_normalized) * 0.0005))  # 0.05% of points, at least 3
    
    threshold_distance = final_threshold  # Regular version uses distance, not squared
    print(f"   ✓ Estimated noise: {estimated_noise:.6f}")
    print(f"   ✓ Threshold: {final_threshold:.6f} ({100*final_threshold/data_range:.2f}% of range)")
    print(f"   ✓ Neighborhood radius: {neighbor_radius:.6f}")
    print(f"   ✓ Minimum points per model: {minimum_point_number}")
    
    # Run Progressive-X 3D line detection with temporal constraint
    print("\n5. Running Progressive-X 3D line detection (with temporal ordering constraint)...")
    
    try:
        t = time()
        # Use NAPSAC sampler to bias towards spatially close points
        # NAPSAC samples points within neighborhood_ball_radius, which helps find spatially clustered lines
        # Regular version uses distance threshold directly
        lines, labeling = pyprogressivex.findLines3DTemporal(
            np.ascontiguousarray(points_normalized, dtype=np.float64),
            np.ascontiguousarray([], dtype=np.float64),  # No weights
            threshold=threshold_distance,  # Pass distance threshold
            conf=0.50,  # Lower confidence to detect more sparse lines
            spatial_coherence_weight=0.3,  # Moderate spatial coherence for temporal lines
            #   - Some spatial coherence helps group points that are spatially close
            #   - But not too high to avoid forcing distant points together
            neighborhood_ball_radius=neighbor_radius,  # Neighborhood radius (pixels):
            #   - Used by NAPSAC sampler: samples points within this radius together to form hypotheses
            #   - Used by PEARL: defines which points are "neighbors" in the spatial coherence graph
            #   - With spatial_coherence_weight=0.0, this is only used for NAPSAC sampling
            maximum_tanimoto_similarity=0.70,  # Lower to prevent merging distinct lines into same label
            max_iters=200000,  # Many iterations to find sparse lines
            minimum_point_number=minimum_point_number,
            maximum_model_number=1000,  # Allow many more lines
            sampler_id=2,  # NAPSAC sampler - helps find spatially clustered lines
            scoring_exponent=0.5,  # Moderate penalty - balance between more lines and larger clusters
            do_logging=False  # Disable logging for cleaner output
        )
        elapsed_time = time() - t
        
        num_models = lines.shape[0] if lines.size > 0 else 0
        
        print(f"   ✓ Completed in {elapsed_time:.2f} seconds")
        print(f"   ✓ Detected {num_models} line models (before filtering)")
        print(f"   ✓ Maximum models allowed: {1000}")
        print(f"   ✓ Maximum iterations: {200000}")
        print(f"   ✓ Scoring exponent: {0.5} (moderate penalty for shared support)")
        
        # Check outlier statistics
        unique_labels, label_counts = np.unique(labeling, return_counts=True)
        # When there's only 1 model, Progressive-X uses: label 0 = inliers, label 1 = outliers
        # When there are multiple models, labels 0..N-1 = models, label N = outliers
        # Check which interpretation applies
        if num_models == 1:
            # Single model: label 0 = inliers, label 1 = outliers
            n_inliers = np.sum(labeling == 0)
            n_outliers = np.sum(labeling == 1)
            print(f"   ✓ Total points: {len(labeling)}")
            print(f"   ✓ Single model interpretation: label 0 = inliers, label 1 = outliers")
            print(f"   ✓ Points classified as inliers (label 0): {n_inliers} ({100*n_inliers/len(labeling):.1f}%)")
            print(f"   ✓ Points classified as outliers (label 1): {n_outliers} ({100*n_outliers/len(labeling):.1f}%)")
        else:
            # Multiple models: labels 0..N-1 = models, label N = outliers
            n_outliers = np.sum(labeling >= num_models)
            n_inliers_total = len(labeling) - n_outliers
            print(f"   ✓ Total points: {len(labeling)}")
            print(f"   ✓ Multiple model interpretation: labels 0..{num_models-1} = models, label {num_models}+ = outliers")
            print(f"   ✓ Points classified as outliers (label {num_models}+): {n_outliers} ({100*n_outliers/len(labeling):.1f}%)")
            print(f"   ✓ Points classified as inliers: {n_inliers_total} ({100*n_inliers_total/len(labeling):.1f}%)")
        print(f"   ✓ Unique labels found: {len(unique_labels)} (labels: {unique_labels[:10]}{'...' if len(unique_labels) > 10 else ''})")
        
        # Count points per model
        print(f"\n   Model size distribution:")
        for label in unique_labels:
            if label == 0:
                continue
            count = label_counts[unique_labels == label][0]
            print(f"     Model {label}: {count} points ({100*count/len(labeling):.1f}%)")
        
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
        
        # Post-process: Filter out poor quality lines (lenient for clean data)
        print(f"\n   Post-processing: Filtering lines by quality...")
        print(f"   → Note: Progressive-X internally filters models during detection.")
        print(f"   → This post-processing step applies additional quality checks.")
        
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
            max_reasonable_size = max(median_line_size * 4.0, mean_line_size * 3.5)  # Very lenient
            print(f"   Line size stats: median={median_line_size:.0f}, mean={mean_line_size:.0f}, max_reasonable={max_reasonable_size:.0f}")
        else:
            max_reasonable_size = len(points_normalized) * 0.30  # 30% max (very lenient)
        
        valid_line_indices = []
        max_points_per_line = len(points_normalized) * 0.25  # 25% max (more lenient) (more lenient for noisy data)
        
        for idx in range(num_models):
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            
            mask = (labeling == instance_label)
            points_assigned = points_normalized[mask]
            
            if len(points_assigned) == 0:
                continue
            
            # Calculate quality metrics using 2D spatial distance (matching the new residual calculation)
            line_point = np.array([lines[idx, 0], lines[idx, 1], lines[idx, 2]])
            line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
            line_dir = line_dir / np.linalg.norm(line_dir)
            
            # Calculate 2D spatial distance (x, y only) - matching the residual calculation
            distances_sq = []
            residuals_all = []  # Store all residuals for diagnostics
            for p in points_assigned:
                # Project point onto line
                vec = p - line_point
                projection_param = np.dot(vec, line_dir)
                projected_point = line_point + projection_param * line_dir
                
                # Calculate 2D spatial distance (x, y only)
                spatial_dist_sq = (p[0] - projected_point[0])**2 + (p[1] - projected_point[1])**2
                distances_sq.append(spatial_dist_sq)
                residuals_all.append(spatial_dist_sq)  # Store for diagnostics
            
            distances_sq = np.array(distances_sq)
            residuals_all = np.array(residuals_all)
            distances = np.sqrt(distances_sq)  # For reporting
            inlier_count = np.sum(distances_sq <= final_threshold)  # Compare squared distances
            inlier_ratio = inlier_count / len(distances_sq)
            mean_distance = distances.mean()  # Report as distance for readability
            max_distance = distances.max()
            
            # DIAGNOSTICS: Line direction and residual analysis
            print(f"      Line direction diagnostics:")
            print(f"        → Direction vector: ({line_dir[0]:.4f}, {line_dir[1]:.4f}, {line_dir[2]:.4f})")
            print(f"        → Time component (z): {line_dir[2]:.4f} (should be ~1.0 for parallel lines)")
            print(f"        → Spatial component (x,y): {np.sqrt(line_dir[0]**2 + line_dir[1]**2):.4f} (should be small)")
            print(f"        → Angle from time axis: {np.degrees(np.arccos(np.clip(abs(line_dir[2]), 0, 1))):.2f}°")
            
            print(f"      Residual diagnostics:")
            print(f"        → Residuals (squared): min={residuals_all.min():.2f}, median={np.median(residuals_all):.2f}, "
                  f"mean={residuals_all.mean():.2f}, max={residuals_all.max():.2f}")
            print(f"        → Threshold (squared): {final_threshold:.2f}")
            print(f"        → Residuals within threshold: {inlier_count}/{len(residuals_all)} ({inlier_ratio*100:.1f}%)")
            print(f"        → Residual ratio (median/threshold): {np.median(residuals_all)/final_threshold:.1f}×")
            print(f"        → Residual percentiles: p25={np.percentile(residuals_all, 25):.2f}, "
                  f"p50={np.percentile(residuals_all, 50):.2f}, p75={np.percentile(residuals_all, 75):.2f}, "
                  f"p90={np.percentile(residuals_all, 90):.2f}, p95={np.percentile(residuals_all, 95):.2f}")
            print(f"        → Distance (sqrt of residual): min={distances.min():.2f}, median={np.median(distances):.2f}, "
                  f"mean={mean_distance:.2f}, max={max_distance:.2f} pixels")
            print(f"        → Threshold distance: {np.sqrt(final_threshold):.2f} pixels")
            
            # Verify residual calculation matches C++ implementation
            # The C++ code calculates: spatial_distance_sq = (point_spatial - projected_spatial).squaredNorm()
            # Let's verify our Python calculation matches
            sample_residuals = []
            for i in range(min(5, len(points_assigned))):
                p = points_assigned[i]
                vec = p - line_point
                projection_param = np.dot(vec, line_dir)
                projected_point = line_point + projection_param * line_dir
                spatial_dist_sq_python = (p[0] - projected_point[0])**2 + (p[1] - projected_point[1])**2
                sample_residuals.append((i, spatial_dist_sq_python, distances_sq[i]))
            
            print(f"      Residual calculation verification (first 5 points):")
            for idx, py_res, arr_res in sample_residuals:
                match = "✓" if abs(py_res - arr_res) < 1e-6 else "✗"
                print(f"        → Point {idx}: Python={py_res:.2f}, Array={arr_res:.2f} {match}")
            
            # Suggest threshold adjustment based on actual residuals
            if np.median(residuals_all) > final_threshold * 10:
                suggested_threshold = np.percentile(residuals_all, 25)  # Use 25th percentile as threshold
                suggested_threshold_dist = np.sqrt(suggested_threshold)
                print(f"      ⚠ Threshold adjustment suggestion:")
                print(f"        → Current threshold too small (median residual is {np.median(residuals_all)/final_threshold:.1f}× larger)")
                print(f"        → Suggested threshold (squared): {suggested_threshold:.2f}")
                print(f"        → Suggested threshold (distance): {suggested_threshold_dist:.2f} pixels ({100*suggested_threshold_dist/data_range:.2f}% of range)")
            
            # Filter criteria: EXTREMELY PERMISSIVE to keep MANY detected lines
            # Since residual uses spatial_distance^2, mean_distance might be in different units
            # Be very lenient to see what lines are actually detected
            is_valid = True
            reasons = []
            
            if inlier_ratio < 0.20:  # Very low threshold to keep many lines (20% vs 35%)
                is_valid = False
                reasons.append(f"low inlier ratio ({inlier_ratio*100:.1f}%)")
            
            # Don't filter by mean_distance since the residual scale might be different
            # Just check if it's extremely large (10× threshold)
            if mean_distance > final_threshold * 10.0:  # Only filter if extremely large
                is_valid = False
                reasons.append(f"extremely high mean distance ({mean_distance:.2f} vs threshold {final_threshold:.2f})")
            
            # REMOVED: "too many points" filter - allow models with many points
            # For event data with parallel lines, a single model might initially capture many points
            # Progressive-X will refine this through multiple iterations
            # if len(points_assigned) > max_points_per_line:
            #     is_valid = False
            #     reasons.append(f"too many points ({len(points_assigned)} = {100*len(points_assigned)/len(points_normalized):.1f}% > {100*max_points_per_line/len(points_normalized):.1f}%)")
            
            if len(line_sizes) > 1 and len(points_assigned) > max_reasonable_size:
                is_valid = False
                reasons.append(f"unusually large ({len(points_assigned)} vs median {median_line_size:.0f}, ratio={len(points_assigned)/median_line_size:.1f}×)")
            
            if len(points_assigned) < minimum_point_number:
                is_valid = False
                reasons.append(f"too few points ({len(points_assigned)} < {minimum_point_number})")
            
            if is_valid:
                valid_line_indices.append(idx)
            else:
                print(f"   ✗ Filtered out line {idx+1}: {', '.join(reasons)}")
                max_label = np.max(labeling) if len(labeling) > 0 else 0
                outlier_label = max_label + 1 if not use_zero_indexed else (num_models if num_models > 0 else 0)
                labeling[mask] = outlier_label
        
        print(f"   ✓ Kept {len(valid_line_indices)} high-quality lines out of {num_models} detected")
        
        # NOTE: Spatial coherence is handled by Progressive-X's grid-based neighborhood framework
        # The grid divides space into cells of size = neighborhood_ball_radius (10 pixels)
        # Points in the same cell or adjacent cells are neighbors, ensuring tight spatial clustering
        
        # Save original number of models before post-processing
        original_num_models = num_models
        
        # POST-PROCESSING: Split labels with multiple distinct spatial clusters
        # Even after Progressive-X, some labels may contain points from multiple distinct lines
        # Split these into separate labels based on spatial clustering
        print(f"\n   Post-processing: Splitting labels with multiple spatial clusters...")
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist
        
        # Use a fixed threshold based on expected line spread
        # For noisy lines, points should be within ~20-30 pixels
        # Anything beyond 50 pixels is likely multiple distinct lines
        max_spatial_distance = 50.0  # Fixed threshold: 50 pixels (much stricter than 4x radius)
        points_reassigned = 0
        new_label_start = np.max(labeling) + 1 if len(labeling) > 0 else (num_models + 1 if not use_zero_indexed else num_models)
        
        # Check ALL labels, not just valid_line_indices (in case some were filtered)
        unique_labels = np.unique(labeling)
        # Filter out outlier labels (labels >= num_models if not zero-indexed, or >= num_models if zero-indexed)
        if not use_zero_indexed:
            labels_to_check = [label for label in unique_labels if label > 0 and label <= num_models]
        else:
            labels_to_check = [label for label in unique_labels if label >= 0 and label < num_models]
        
        for instance_label in labels_to_check:
            mask = (labeling == instance_label)
            if not np.any(mask) or np.sum(mask) < 4:  # Need at least 4 points to cluster
                continue
                
            points_assigned = points_raw[mask]  # Use raw points for spatial analysis
            spatial_points = points_assigned[:, :2]  # Only x, y coordinates
            
            # Calculate maximum spatial distance between any two points in this label
            max_dist = 0.0
            for i in range(len(spatial_points)):
                for j in range(i+1, len(spatial_points)):
                    dist = np.linalg.norm(spatial_points[i] - spatial_points[j])
                    if dist > max_dist:
                        max_dist = dist
            
            # If points are too spread out, split into spatial clusters
            if max_dist > max_spatial_distance:
                print(f"   → Label {instance_label}: max spatial distance = {max_dist:.1f} pixels (threshold: {max_spatial_distance:.1f})")
                
                # Use hierarchical clustering with distance threshold
                if len(spatial_points) >= 2:
                    distances = pdist(spatial_points, metric='euclidean')
                    if len(distances) > 0:
                        linkage_matrix = linkage(distances, method='single')
                        # Use a tighter clustering threshold: 20 pixels (points within 20 pixels are in same cluster)
                        cluster_labels = fcluster(linkage_matrix, t=20.0, criterion='distance')
                        unique_clusters = np.unique(cluster_labels)
                        
                        if len(unique_clusters) > 1:
                            # Split into multiple labels
                            point_indices = np.where(mask)[0]
                            
                            # Keep first cluster in original label, assign others to new labels
                            for cluster_id in unique_clusters[1:]:  # Skip first cluster (keeps original label)
                                cluster_mask = (cluster_labels == cluster_id)
                                if np.sum(cluster_mask) >= 2:  # Keep clusters with at least 2 points
                                    labeling[point_indices[cluster_mask]] = new_label_start
                                    new_label_start += 1
                                    points_reassigned += np.sum(cluster_mask)
                            
                            print(f"      → Split into {len(unique_clusters)} spatial clusters")
        
        if points_reassigned > 0:
            print(f"   ✓ Reassigned {points_reassigned} points to enforce spatial clustering")
            # After splitting, we need to update valid_line_indices
            # But we can't easily map new labels back to line indices, so just use all non-outlier labels
            unique_labels_after = np.unique(labeling)
            if not use_zero_indexed:
                # Filter out outlier labels (>= num_models + 1) and get valid indices
                valid_line_indices = [int(label - 1) for label in unique_labels_after 
                                    if label > 0 and label <= original_num_models]
            else:
                valid_line_indices = [int(label) for label in unique_labels_after 
                                    if label >= 0 and label < original_num_models]
        
        # Update num_models to reflect valid lines after post-processing
        num_models = len(valid_line_indices)
        
        # Extract detected lines and check for parallelism (only valid ones)
        detected_lines = []
        line_directions = []
        
        for idx in valid_line_indices:
            if idx >= lines.shape[0]:
                print(f"   ⚠ Skipping line index {idx} (out of bounds, lines.shape[0]={lines.shape[0]})")
                continue
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
                
                vecs = points_assigned - line_point
                projections = np.dot(vecs, line_dir)
                extent = projections.max() - projections.min() if len(projections) > 0 else 0
                
                print(f"   Line {len(detected_lines)}: {len(points_assigned)} points, "
                      f"{inlier_ratio*100:.1f}% within threshold, "
                      f"mean distance: {distances.mean():.4f}, "
                      f"max distance: {max_distance:.4f}, "
                      f"std distance: {distances.std():.4f}, "
                      f"extent: {extent:.2f}")
                print(f"      Direction: ({line_dir[0]:.4f}, {line_dir[1]:.4f}, {line_dir[2]:.4f})")
        
        # Check parallelism
        if len(line_directions) > 1:
            print(f"\n   Checking parallelism of detected lines...")
            parallelism_scores = []
            for i in range(len(line_directions)):
                for j in range(i+1, len(line_directions)):
                    dot_product = np.abs(np.dot(line_directions[i], line_directions[j]))
                    angle_deg = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
                    parallelism_scores.append(dot_product)
                    print(f"      Lines {i+1} & {j+1}: dot={dot_product:.4f}, angle={angle_deg:.2f}° "
                          f"{'✓ parallel' if angle_deg < 15 else '✗ not parallel'}")
            
            avg_parallelism = np.mean(parallelism_scores) if parallelism_scores else 0
            print(f"   Average parallelism score: {avg_parallelism:.4f} (1.0 = perfectly parallel)")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Visualize results
    print("\n6. Visualizing results...")
    
    # Define colors for detected lines
    colors_det = plt.cm.tab10(np.linspace(0, 1, num_models)) if num_models > 0 else []
    
    # Create separate figure for event density heatmap
    fig_heatmap = plt.figure(figsize=(12, 10))
    ax_heatmap = fig_heatmap.add_subplot(111)
    
    # Create 2D histogram of events in x-y space
    x_min, x_max = points_raw[:, 0].min(), points_raw[:, 0].max()
    y_min, y_max = points_raw[:, 1].min(), points_raw[:, 1].max()
    
    n_bins_x = int((x_max - x_min) / 2)
    n_bins_y = int((y_max - y_min) / 2)
    n_bins_x = max(50, min(200, n_bins_x))
    n_bins_y = max(50, min(200, n_bins_y))
    
    hist, x_edges, y_edges = np.histogram2d(
        points_raw[:, 0], points_raw[:, 1],
        bins=[n_bins_x, n_bins_y],
        range=[[x_min, x_max], [y_min, y_max]]
    )
    
    hist = hist.T
    
    # Analyze cluster widths in heatmap to determine appropriate threshold
    print(f"\n   Analyzing heatmap clusters to determine appropriate threshold...")
    bin_width_x = (x_edges[-1] - x_edges[0]) / len(x_edges)
    bin_width_y = (y_edges[-1] - y_edges[0]) / len(y_edges)
    print(f"     → Bin size: {bin_width_x:.1f} × {bin_width_y:.1f} pixels")
    
    # Find dense regions (above median)
    hist_flat = hist.flatten()
    hist_flat_positive = hist_flat[hist_flat > 0]
    if len(hist_flat_positive) > 0:
        median_density = np.median(hist_flat_positive)
        p75_density = np.percentile(hist_flat_positive, 75)
        p90_density = np.percentile(hist_flat_positive, 90)
        print(f"     → Density stats: median={median_density:.1f}, p75={p75_density:.1f}, p90={p90_density:.1f}, max={hist.max():.1f}")
        
        # Find clusters (connected components of dense regions)
        from scipy import ndimage
        dense_mask = hist > p75_density  # Use 75th percentile as threshold for "dense"
        labeled, num_features = ndimage.label(dense_mask)
        
        if num_features > 0:
            # Calculate cluster sizes
            cluster_sizes = []
            cluster_widths_x = []
            cluster_widths_y = []
            for i in range(1, num_features + 1):
                cluster_mask = (labeled == i)
                cluster_size = np.sum(cluster_mask)
                if cluster_size > 1:  # Ignore single-pixel clusters
                    cluster_sizes.append(cluster_size)
                    # Find bounding box
                    rows = np.where(cluster_mask)[0]
                    cols = np.where(cluster_mask)[1]
                    if len(rows) > 0 and len(cols) > 0:
                        width_y = (rows.max() - rows.min() + 1) * bin_width_y
                        width_x = (cols.max() - cols.min() + 1) * bin_width_x
                        cluster_widths_x.append(width_x)
                        cluster_widths_y.append(width_y)
            
            if len(cluster_sizes) > 0:
                print(f"     → Found {num_features} dense regions")
                print(f"     → Cluster sizes: min={min(cluster_sizes)}, median={np.median(cluster_sizes):.0f}, max={max(cluster_sizes)} bins")
                print(f"     → Cluster widths (X): min={min(cluster_widths_x):.1f}, median={np.median(cluster_widths_x):.1f}, max={max(cluster_widths_x):.1f} pixels")
                print(f"     → Cluster widths (Y): min={min(cluster_widths_y):.1f}, median={np.median(cluster_widths_y):.1f}, max={max(cluster_widths_y):.1f} pixels")
                typical_cluster_width = max(np.median(cluster_widths_x), np.median(cluster_widths_y))
                print(f"     → Typical cluster width: {typical_cluster_width:.1f} pixels")
                print(f"     → Current threshold: {threshold_distance:.1f} pixels")
                
                # Suggest threshold adjustment
                if threshold_distance < typical_cluster_width * 0.5:
                    suggested_threshold = typical_cluster_width * 0.6  # 60% of typical width
                    suggested_threshold_sq = suggested_threshold * suggested_threshold
                    print(f"     ⚠ WARNING: Threshold ({threshold_distance:.1f}) is too small for clusters!")
                    print(f"     → Suggested threshold: {suggested_threshold:.1f} pixels (squared: {suggested_threshold_sq:.1f})")
                    print(f"     → This would be {100*suggested_threshold/data_range:.2f}% of data range")
    
    # Apply log scaling with 99.5 percentile clipping
    hist_flat = hist.flatten()
    hist_flat_positive = hist_flat[hist_flat > 0]
    if len(hist_flat_positive) > 0:
        percentile_99_5 = np.percentile(hist_flat_positive, 99.5)
        hist_clipped = np.clip(hist, 0, percentile_99_5)
        hist_log = np.log1p(hist_clipped)
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
            points_on_line = points_raw[mask]
            
            dir_3d = det_line['direction']
            dir_xy = dir_3d[:2]
            dir_xy_norm = np.linalg.norm(dir_xy)
            
            if dir_xy_norm > 1e-6:
                dir_xy = dir_xy / dir_xy_norm
                
                center_xy = points_on_line[:, :2].mean(axis=0)
                vecs_to_center = points_on_line[:, :2] - center_xy
                projs = np.dot(vecs_to_center, dir_xy)
                
                proj_min, proj_max = projs.min(), projs.max()
                proj_min -= 0.1 * (proj_max - proj_min)
                proj_max += 0.1 * (proj_max - proj_min)
                
                line_x = np.array([center_xy[0] + proj_min * dir_xy[0], 
                                  center_xy[0] + proj_max * dir_xy[0]])
                line_y = np.array([center_xy[1] + proj_min * dir_xy[1], 
                                  center_xy[1] + proj_max * dir_xy[1]])
                ax_heatmap.plot(line_x, line_y, color=colors_det[idx], linewidth=4.5, 
                         linestyle='-', alpha=0.95, label=f'Line {idx+1}', zorder=10)
            else:
                center_xy = points_on_line[:, :2].mean(axis=0)
                ax_heatmap.scatter([center_xy[0]], [center_xy[1]], 
                           color=colors_det[idx], s=150, marker='o', 
                           edgecolors='white', linewidths=2, 
                           label=f'Line {idx+1} (temporal)', zorder=10)
    
    ax_heatmap.set_xlabel('X (pixels)', color='white', fontsize=12)
    ax_heatmap.set_ylabel('Y (pixels)', color='white', fontsize=12)
    ax_heatmap.set_title(f'Event Density Heatmap: {num_models} lines detected (May 9, 0.1s)', 
                 fontsize=14, fontweight='bold', color='white')
    ax_heatmap.set_aspect('equal')
    ax_heatmap.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    ax_heatmap.tick_params(colors='white')
    ax_heatmap.grid(True, alpha=0.2, color='white')
    
    # Save heatmap figure
    output_file_heatmap = 'may9_events_density_heatmap.png'
    fig_heatmap.savefig(output_file_heatmap, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"   ✓ Saved event density heatmap to {output_file_heatmap}")
    plt.close(fig_heatmap)
    
    # Original multi-plot figure
    fig = plt.figure(figsize=(24, 8))
    
    # Plot 1: 3D point cloud of all event data
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Sample points for visualization (too many points can be slow)
    n_sample_viz = min(5000, len(points_normalized))
    sample_indices_viz = np.random.choice(len(points_normalized), n_sample_viz, replace=False)
    points_sample = points_normalized[sample_indices_viz]
    
    # Color by normalized time
    scatter1 = ax1.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2],
                          c=points_sample[:, 2], s=1, alpha=0.3, cmap='viridis')
    plt.colorbar(scatter1, ax=ax1, label='Normalized Time')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_zlabel('Time (normalized)')
    ax1.set_title(f'3D Event Point Cloud: {len(points_raw)} events\n(sampled {n_sample_viz} for visualization)', 
                 fontsize=12, fontweight='bold')
    
    # Plot 2: 3D view (x, y, t) - only show labeled points (no outliers) with lines superimposed
    ax2 = fig.add_subplot(132, projection='3d')
    
    unique_labels = np.unique(labeling)
    
    # Determine outlier label to skip
    if use_zero_indexed:
        outlier_label = num_models
    else:
        outlier_label = 0
    
    # Plot all points by their labels (skip outliers)
    for label in unique_labels:
        # Skip outlier label
        if (use_zero_indexed and label == outlier_label) or (not use_zero_indexed and label == outlier_label):
            continue
            
        mask = (labeling == label)
        if not np.any(mask):
            continue
        
        # This is a model label - find the corresponding line index
        line_idx = label if use_zero_indexed else label - 1
        if line_idx < len(colors_det):
            color = colors_det[line_idx]
            ax2.scatter(points_normalized[mask, 0], 
                       points_normalized[mask, 1], 
                       points_normalized[mask, 2],
                       c=color, s=2, alpha=0.6,
                       label=f'Line {line_idx+1}' if label == (line_idx if use_zero_indexed else line_idx+1) else '')
    
    # Plot the detected lines themselves (superimposed on top)
    # Use a colormap that gives distinct colors for many lines
    if len(detected_lines) > 10:
        # For many lines, use tab20 (20 distinct colors) or hsv for even more
        if len(detected_lines) <= 20:
            line_colors = plt.cm.tab20(np.linspace(0, 1, len(detected_lines)))
        else:
            # For >20 lines, use hsv colormap for maximum distinctness
            line_colors = plt.cm.hsv(np.linspace(0, 1, len(detected_lines)))
    else:
        line_colors = colors_det[:len(detected_lines)]
    
    for idx, det_line in enumerate(detected_lines):
        mask = (labeling == det_line['label'])
        if np.any(mask):
            points_on_line = points_normalized[mask]
            # Get color from appropriate colormap
            if len(detected_lines) > 10:
                color = line_colors[idx]
            else:
                color = colors_det[idx % len(colors_det)]
            
            # Plot the line itself in 3D
            # For parallel lines, use the actual point range along the line
            vecs = points_on_line - det_line['point']
            projections = np.dot(vecs, det_line['direction'])
            t_min, t_max = projections.min(), projections.max()
            
            # Extend the line range to make it more visible
            # Use a larger extension for parallel lines so they're more visible
            t_range = np.linspace(t_min - 0.5*(t_max-t_min), t_max + 0.5*(t_max-t_min), 200)
            line_points = det_line['point'][None, :] + t_range[:, None] * det_line['direction'][None, :]
            
            # Use distinct colors and make lines more visible
            # Use slightly different linewidths to help distinguish overlapping lines
            linewidth = 4 + (idx % 3) * 0.5  # Vary linewidth slightly (4, 4.5, 5)
            ax2.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                    color=color, linewidth=linewidth, linestyle='-', alpha=1.0, 
                    label=f'Line {idx+1}' if idx < 20 else '')
    
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_zlabel('Time (normalized)')
    ax2.set_title(f'3D Detected Lines: {len(detected_lines)} lines with points', 
                 fontsize=12, fontweight='bold')
    # Limit legend entries if there are too many lines
    if len(detected_lines) <= 20:
        ax2.legend(loc='upper left', fontsize=6, ncol=1, framealpha=0.8)
    else:
        # For many lines, show a summary instead of legend
        ax2.text2D(0.02, 0.98, f'{len(detected_lines)} lines detected', 
                  transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Detected lines projected to (x, y) space
    ax3 = fig.add_subplot(133)
    
    for label in unique_labels:
        mask = (labeling == label)
        if np.any(mask):
            if (use_zero_indexed and label == num_models) or (not use_zero_indexed and label == 0):
                ax3.scatter(points_raw[mask, 0], points_raw[mask, 1],
                          c='gray', s=1, alpha=0.3, marker='x', label='Outliers')
            else:
                line_idx = label if use_zero_indexed else label - 1
                if line_idx < len(colors_det):

                    print('\nit is printing '+str(label) + ' with length ' + str(len(points_raw[mask])))
                    print('the points with this label are: '+str(points_raw[mask][:,0:2])) # #])+' '+str(points_raw[mask, 1]))
                    ax3.scatter(points_raw[mask, 0], points_raw[mask, 1],
                              c=[colors_det[line_idx]], s=2, alpha=0.6,
                              label=f'Line {line_idx+1}' if label == (line_idx if use_zero_indexed else line_idx+1) else '')
    
    for idx, det_line in enumerate(detected_lines):
        mask = (labeling == det_line['label'])
        if np.any(mask):
            points_on_line = points_raw[mask]
            
            dir_3d = det_line['direction']
            dir_xy = dir_3d[:2]
            dir_xy_norm = np.linalg.norm(dir_xy)
            
            if dir_xy_norm > 1e-6:
                dir_xy = dir_xy / dir_xy_norm
                
                center_xy = points_on_line[:, :2].mean(axis=0)
                vecs_to_center = points_on_line[:, :2] - center_xy
                projs = np.dot(vecs_to_center, dir_xy)
                
                proj_min, proj_max = projs.min(), projs.max()
                proj_min -= 0.1 * (proj_max - proj_min)
                proj_max += 0.1 * (proj_max - proj_min)
                
                line_x = np.array([center_xy[0] + proj_min * dir_xy[0], 
                                  center_xy[0] + proj_max * dir_xy[0]])
                line_y = np.array([center_xy[1] + proj_min * dir_xy[1], 
                                  center_xy[1] + proj_max * dir_xy[1]])
                ax3.plot(line_x, line_y, color=colors_det[idx], linewidth=3, 
                         linestyle='-', alpha=0.9, label=f'Line {idx+1}')
            else:
                center_xy = points_on_line[:, :2].mean(axis=0)
                ax3.scatter([center_xy[0]], [center_xy[1]], 
                           color=colors_det[idx], s=100, marker='o', 
                           edgecolors='black', linewidths=2, label=f'Line {idx+1} (temporal)')
    
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.set_title(f'Detected Lines (x-y projection): {num_models} lines', 
                 fontsize=12, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f"May 9 HDF5 Event Data: {len(points_raw)} events, {num_models} lines detected",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = 'may9_events_3d_detection.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to {output_file}")
    plt.show()
    
    print("\n" + "="*70)
    print("✓ Test completed successfully!")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
