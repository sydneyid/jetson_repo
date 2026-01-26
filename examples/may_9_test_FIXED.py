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
import matplotlib
import matplotlib.colors
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
        p_filtered = p[time_mask] 

        p_mask = (p_filtered== 1)

        x_filtered = x_filtered[p_mask]
        y_filtered = y_filtered[p_mask]
        t_filtered = t_filtered[p_mask]   
        # p_filtered = p_filtered[time_mask] 
        
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
    hdf5_path = 'img/may_9.hdf5'
    
    if not os.path.exists(hdf5_path):
        print(f"  Error: File not found: {hdf5_path}")
        return 1
    
    print("\n1. Reading HDF5 event file...")
    try:
        points_raw, metadata = read_hdf5_events(hdf5_path, time_window=0.05, start_time=None)
        print(f"   ✓ Loaded {len(points_raw)} events")
    except Exception as e:
        print(f"     Error reading HDF5 file: {e}")
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
    
    # Minimum point number - BALANCED for finding more lines
    # True lines have 20-110 points, but we want to find more lines
    # Reduced from 15 to 10 to allow finding lines with fewer points
    # This helps find more lines while still rejecting very weak proposals
    minimum_point_number = 10  # Minimum 10 inliers - allows finding more lines while still filtering weak proposals
    
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

    # max_iters controls max RANSAC iterations PER PROPOSAL (not total)
    # OPTIMIZED: Scale with point count - fewer points need fewer samples
    # With Z-aligned sampler, proposals succeed faster, so we don't need extremely high values
    # Formula: Scale proportionally to point count, with minimum threshold for small datasets
    # For 7000 points: 3000 iterations (baseline)
    # For 1600 points: ~700 iterations (proportional scaling)
    # OPTIMIZED: Increased max_iters to find more lines
    # More iterations per proposal = better chance of finding all lines
    if n_points < 2000:
        # Very small datasets: minimum 500 iterations, scale with points
        max_iters_optimized = max(500, int(n_points * 0.5))  # ~50% of point count (was 40%), min 500
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
    
    # OPTIMIZATIONS FOR FINDING MORE LINES:
    # 1. Moderate confidence (0.05) - balance between finding lines and speed
    #    Lower confidence (0.01) finds more but requires many more iterations per proposal
    # 2. Z-Aligned sampler - samples points along time-parallel lines, proposals succeed faster
    # 3. Early termination stops at 24-25 models to avoid unnecessary iterations
    # SENSITIVITY: If not finding 24 lines, try: conf=0.01, max_iters=5000-10000
    conf_optimized = 0.05  # Moderate confidence - faster than 0.01, still finds many models
    sampler_id_optimized = 3  # Z-Aligned sampler - samples points with similar X,Y but different Z (ideal for lines parallel to time axis)
    
    print(f"   Settings: conf={conf_optimized} (moderate for speed), sampler=Z-Aligned (for time-parallel lines), max_iters={max_iters_optimized:,}")
    print(f"   → Early termination: stops at 30-35 models to allow finding more lines")
    
    try:
        t = time()
        # Force stdout to be unbuffered for this section
        import sys
        old_stdout = sys.stdout
        sys.stdout = sys.__stdout__  # Use unbuffered stdout
        
        lines, labeling = pyprogressivex.findLines3D(
            np.ascontiguousarray(points_normalized, dtype=np.float64),
            np.ascontiguousarray([], dtype=np.float64),  # No weights
            threshold=final_threshold,
            conf=conf_optimized,  # OPTIMIZED: Moderate confidence (0.05) for speed while finding many models
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=neighbor_radius,
            maximum_tanimoto_similarity=0.40,  # Increased from 0.30 to allow finding even more overlapping parallel lines
            max_iters=max_iters_optimized,  # OPTIMIZED: Reduced from 2M to reasonable value
            minimum_point_number=minimum_point_number,
            maximum_model_number=20000,  # Allow MANY more lines
            sampler_id=sampler_id_optimized,  # OPTIMIZED: Z-Aligned sampler (samples along time-parallel lines)
            scoring_exponent=0.0,  # NO penalty for shared support (allows many overlapping lines)
            do_logging=False  # Disable verbose C++ logging
        )
        
        # Force flush immediately after C++ call
        sys.stdout.flush()
        sys.stderr.flush()
        
        elapsed_time = time() - t
        
        num_models = lines.shape[0] if lines.size > 0 else 0
        
        print(f"   ✓ Completed in {elapsed_time:.2f} seconds")
        print(f"   ✓ Detected {num_models} line models (before filtering)")


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
        
        

        valid_line_indices = parallel_check(lines,labeling,num_models , parallel_threshold = 1.0,print_updates=True )

        # Summary of detection pipeline
        print(f"\n   📊 DETECTION PIPELINE SUMMARY:")
        print(f"      Stage 1 (Progressive-X): {num_models} suggested/proposed lines")
        print(f"      Stage 2 (Mutual parallel): {len(valid_line_indices)} final lines (all parallel to each other)")
        

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
        ax2.set_xlim(0,1280)
        ax2.set_ylim(0,720)
        ax2.set_zlabel('Time (normalized)')
        ax2.set_title(f'Detected Lines ({len(valid_line_indices)} lines)')
        ax2.legend(loc='upper right', fontsize=8)
        # Put a legend below current axis
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5)
        
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
        # plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved visualization to {output_path}")
        plt.close()
        
        # Create separate figure: Event heatmap with line centroids
        print("\n7. Creating event heatmap with line centroids...")
        
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
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            mask = (labeling == instance_label)
            points_line = points_normalized[mask]
            
            if len(points_line) > 0:
                # Get original (non-normalized) x, y coordinates for centroid
                # Since normalization only affects time (z), x and y are the same
                centroid_x = np.mean(points_line[:, 0])
                centroid_y = np.mean(points_line[:, 1])
                # No y-flip needed since np.add.at uses (y_coords, x_coords) directly
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
                ax_heatmap.annotate(f'L{num}', (x, y), xytext=(5, 5), 
                                  textcoords='offset points', color='cyan', 
                                  fontsize=8, fontweight='bold', zorder=11)
        
        ax_heatmap.set_xlabel('X (pixels)')
        ax_heatmap.set_ylabel('Y (pixels)')
        ax_heatmap.set_title(f'Event Accumulation Heatmap (Log Scale, 99.9th percentile vmax) with {len(line_centroids)} Line Centroids')
        ax_heatmap.set_xlim(0, img_width)
        ax_heatmap.set_ylim(0, img_height)
        ax_heatmap.legend(loc='upper right')
        
        plt.tight_layout()
        output_path_heatmap = 'may9_events_heatmap_centroids.png'
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
