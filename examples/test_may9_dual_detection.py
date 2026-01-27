#!/usr/bin/env python3
"""
Test Progressive-X Dual 3D Line Detection on May 9 HDF5 Event Data
Uses findLines3DDual to detect both dense and sparse lines simultaneously
Reads events from Jupiter_In_Focus.hdf5 and treats (x, y, t) as a 3D point cloud
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
        events_constraint: Maximum number of events to use (None = use all)
        start_time: Start time in seconds (if None, starts from beginning)
    
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
        else:
            start_time = int(start_time * 1e6)  # Convert to microseconds
        
        end_time = start_time + time_window * 1e6  # Convert to microseconds
        time_mask = (t >= start_time) & (t < end_time)
        
        x_filtered = x[time_mask]
        y_filtered = y[time_mask]
        t_filtered = t[time_mask]   
        p_filtered = p[time_mask]

        # Filter by polarity (only positive polarity events)
        p_mask = (p_filtered == 1)
        x_filtered = x_filtered[p_mask]
        y_filtered = y_filtered[p_mask]
        t_filtered = t_filtered[p_mask]   
        p_filtered = p_filtered[p_mask]

        # Apply events constraint if specified
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

def parallel_check(lines, labeling, num_models, parallel_threshold=1.0, print_updates=False):
    """
    Check which lines are parallel to the time axis
    
    Args:
        lines: Array of line parameters
        labeling: Point labels
        num_models: Number of models
        parallel_threshold: Angle threshold in degrees (default: 1.0)
        print_updates: Whether to print filtering updates
    
    Returns:
        List of valid line indices
    """
    expected_direction = np.array([0.0, 0.0, 1.0])  # Time axis direction
    valid_line_indices = []
    parallel_threshold_cos = np.cos(np.deg2rad(parallel_threshold))
    
    # Iterate over all detected lines
    for idx in range(num_models):
        line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
        line_dir = line_dir / np.linalg.norm(line_dir)
        
        # Check if line is parallel to time axis
        dot_with_time = np.abs(np.dot(line_dir, expected_direction))
        angle_with_time = np.arccos(np.clip(dot_with_time, -1, 1)) * 180 / np.pi
        
        if dot_with_time >= parallel_threshold_cos:
            valid_line_indices.append(idx)
        else:
            if print_updates:
                print(f"   Removed line {idx+1}: not parallel to time axis (angle={angle_with_time:.2f}°, threshold={parallel_threshold}°)")
    
    if print_updates:
        print(f"\n     DETECTION PIPELINE SUMMARY:")
        print(f"      Stage 1 (Progressive-X): {num_models} suggested/proposed lines")
        print(f"      Stage 2 (Time-axis parallel): {len(valid_line_indices)} lines parallel to time axis")
    
    return valid_line_indices


def main():
    print("="*70)
    print("Progressive-X Dual 3D Line Detection on May 9 HDF5 Event Data")
    print("Using findLines3DDual for dense + sparse line detection")
    print("="*70)

    # Same parameters on Jetson and desktop: fix randomness used for noise/threshold estimation
    np.random.seed(42)

    # Read HDF5 file (path is relative to cwd – run from examples/: cd examples && python test_may9_dual_detection.py)
    hdf5_path = 'img/may_9.hdf5'

    if not os.path.exists(hdf5_path):
        print(f"  Error: File not found: {hdf5_path}")
        print(f"  Run from examples/ dir:  cd examples  &&  python test_may9_dual_detection.py")
        return 1
    
    print("\n1. Reading HDF5 event file...")
    try:
        # Use all events in time window (no constraint, no start_time filtering)
        # This matches may_9_test_FIXED.py approach
        points_raw, metadata = read_hdf5_events(hdf5_path, time_window=0.05, 
                                               events_constraint=None, start_time=None)
        print(f"     Loaded {len(points_raw)} events")
    except Exception as e:
        print(f"     Error reading HDF5 file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # OPTIMIZATION: Optional downsampling for very large datasets
    MAX_POINTS_FOR_SPEED = 500000  # Downsample if more than this
    if len(points_raw) > MAX_POINTS_FOR_SPEED:
        downsample_factor = len(points_raw) / MAX_POINTS_FOR_SPEED
        original_size = len(points_raw)
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(points_raw), size=MAX_POINTS_FOR_SPEED, replace=False)
        indices = np.sort(indices)  # Keep temporal order
        points_raw = points_raw[indices]
        print(f"    OPTIMIZATION: Downsampled from {original_size:,} to {len(points_raw):,} points "
              f"(factor: {downsample_factor:.1f}x) for speed")
    
    # Normalize time coordinate for better detection
    print("\n2. Normalizing coordinates...")
    
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
    

    print(f"     Normalized coordinate ranges:")
    print(f"     X: [{points_normalized[:, 0].min():.1f}, {points_normalized[:, 0].max():.1f}]")
    print(f"     Y: [{points_normalized[:, 1].min():.1f}, {points_normalized[:, 1].max():.1f}]")
    print(f"     T: [{points_normalized[:, 2].min():.1f}, {points_normalized[:, 2].max():.1f}]")
    
    # Compute optimal parameters - tuned for parallel lines
    print("\n3. Computing optimal parameters (tuned for parallel lines)...")
    
    n_sample = min(100, len(points_normalized))
    sample_indices = np.random.choice(len(points_normalized), n_sample, replace=False)
    sample_points = points_normalized[sample_indices]
    dists = cdist(sample_points, sample_points)
    np.fill_diagonal(dists, np.inf)
    nearest_dists = np.min(dists, axis=1)
    estimated_noise = np.median(nearest_dists)
    
    data_range = np.max(points_normalized.max(axis=0) - points_normalized.min(axis=0))
    
    # Calculate base threshold - balanced for both dense and sparse detection
    threshold_from_range = data_range * 0.001
    threshold_from_noise = estimated_noise * 1.0
    base_threshold = max(threshold_from_range, threshold_from_noise)
    base_threshold = max(base_threshold, data_range * 0.0003)  # Minimum 0.03% of range
    base_threshold = min(base_threshold, data_range * 0.012)  # Maximum cap
    
    # Neighborhood radius: scaled to data range
    neighbor_radius = data_range * 0.002  # 0.2% of range
    neighbor_radius = max(neighbor_radius, estimated_noise * 0.15)
    neighbor_radius = min(neighbor_radius, data_range * 0.010)
    
    minimum_point_number = 10  # Minimum 10 inliers
    
    print(f"     Estimated noise: {estimated_noise:.6f}")
    print(f"     Base threshold: {base_threshold:.6f} ({100*base_threshold/data_range:.2f}% of range)")
    print(f"     Neighborhood radius: {neighbor_radius:.6f}")
    print(f"     Minimum points per model: {minimum_point_number}")

    # Compare Jetson vs desktop: if these match but line counts differ, the difference is in the C++ build/env
    print(f"     [DIAG] n_points={len(points_normalized)} estimated_noise={estimated_noise:.6f} base_threshold={base_threshold:.6f} neighbor_radius={neighbor_radius:.6f}")

    print("\n4. Running Progressive-X Dual 3D line detection...")
    
    n_points = len(points_normalized)
    
    # Scale max_iters based on point count
    if n_points < 2000:
        max_iters_optimized = max(1000, int(n_points * 1.0))
    elif n_points < 5000:
        max_iters_optimized = int(n_points * 4000 / 7000)
    elif n_points < 10000:
        max_iters_optimized = 4000
    elif n_points < 100000:
        max_iters_optimized = 12000
    else:
        max_iters_optimized = 30000
    
    print(f"   Using max_iters={max_iters_optimized:,} (scaled from {n_points:,} points)")
    
    # Use higher confidence for dense line detection
    conf_dense = 0.05  # Higher confidence for dense lines
    sampler_id_optimized = 3  # Z-Aligned sampler
    

    try:
        t = time()
        import sys
        old_stdout = sys.stdout
        sys.stdout = sys.__stdout__
        
        # DUAL DETECTION: Detect both dense and sparse lines with mutual exclusivity
        threshold_dense = base_threshold * 0.9  # Slightly relaxed for dense lines
        threshold_sparse = base_threshold * 2.5  # More permissive for sparse lines
        minimum_point_number_sparse = max(3, minimum_point_number // 2)  # Lower minimum for sparse
        
        print(f"   Dense line detection: threshold={threshold_dense:.6f} ({100*threshold_dense/data_range:.2f}% of range), min_points={minimum_point_number}")
        print(f"   Sparse line detection: threshold={threshold_sparse:.6f} ({100*threshold_sparse/data_range:.2f}% of range), min_points={minimum_point_number_sparse}")
        
        lines, labeling, line_types = pyprogressivex.findLines3DDual(
            np.ascontiguousarray(points_normalized, dtype=np.float64),
            np.ascontiguousarray([], dtype=np.float64),  # No weights
            threshold_dense=threshold_dense,
            threshold_sparse=threshold_sparse,
            conf=conf_dense,
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=neighbor_radius,
            maximum_tanimoto_similarity=0.35,
            max_iters=max_iters_optimized,
            minimum_point_number_dense=minimum_point_number,
            minimum_point_number_sparse=minimum_point_number_sparse,
            maximum_model_number=20000,
            sampler_id=sampler_id_optimized,
            scoring_exponent=0.0,
            do_logging=False
        )
        
        sys.stdout.flush()
        sys.stderr.flush()
        
        elapsed_time = time() - t
        
        num_models = lines.shape[0] if lines.size > 0 else 0
        
        # Separate dense and sparse lines
        dense_mask = (line_types == 0)
        sparse_mask = (line_types == 1)
        num_dense = np.sum(dense_mask)
        num_sparse = np.sum(sparse_mask)
        
        print(f"     Completed in {elapsed_time:.2f} seconds")
        print(f"     Detected {num_models} line models total:")
        print(f"      - {num_dense} DENSE lines")
        print(f"      - {num_sparse} SPARSE lines")

        if num_models == 0:
            print("     [Jetson vs desktop] If the other machine finds lines: compare the [DIAG] line above.")
            print("     Same [DIAG] values + 0 lines here => C++/build/env differs. Different => run from examples/ and use same data.")

        # Labeling scheme: label 0 = outliers, labels 1+ = lines
        unique_labels = np.unique(labeling)


        # Filter by parallel check
        print("\n5. Filtering lines by time-axis parallelism...")
        valid_line_indices = parallel_check(lines, labeling, num_models, 
                                           parallel_threshold=1.0, print_updates=False)
    

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
        
        # First, plot outliers (label 0)
        outlier_mask = (labeling == 0)
        outliers = points_normalized[outlier_mask]
        if len(outliers) > 0:
            ax2.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2],
                       c='grey', s=3, alpha=0.5, marker='x', label='Outliers')
        
        # Draw detected lines in 3D
        for i, idx in enumerate(valid_line_indices):
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
        
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_xlim(0, 1280)
        ax2.set_ylim(0, 720)
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


        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        ax3.set_title('Detected Lines (X-Y Projection)')
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                fancybox=True, shadow=True, ncol=5, fontsize=8, markerscale=2)
        ax3.set_aspect('equal')
        
        plt.tight_layout()
        output_path = 'may9_dual_detection.png'
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
            img_width = int(points_raw[:, 0].max()) + 1
            img_height = int(points_raw[:, 1].max()) + 1
        
        # Create accumulated event frame
        accumulated_frame = np.ones((img_height, img_width), dtype=np.float64)
        x_coords = points_raw[:, 0].astype(int)
        y_coords = points_raw[:, 1].astype(int)
        np.add.at(accumulated_frame, (y_coords, x_coords), 1)
        
        print(f"   Accumulated frame range: [{accumulated_frame.min():.1f}, {accumulated_frame.max():.1f}]")
        print(f"   99.9th percentile: {np.percentile(accumulated_frame, 99.9):.1f}")
        
        # Calculate line centroids
        line_centroids = []
        for i, idx in enumerate(valid_line_indices):
            # Map line index to label: line idx -> label idx + 1 (label 0 is outliers)
            instance_label = idx + 1
            mask = (labeling == instance_label)
            points_line = points_normalized[mask]
            
            if len(points_line) > 0:
                centroid_x = np.mean(points_line[:, 0])
                centroid_y = np.mean(points_line[:, 1])
                line_type = line_types[idx] if idx < len(line_types) else 0
                line_centroids.append((centroid_x, centroid_y, i+1, line_type))
        
        # Create new figure for heatmap
        fig2 = plt.figure(figsize=(12, 8))
        ax_heatmap = fig2.gca()
        
        vmax_99_9 = np.percentile(accumulated_frame, 99.9)
        im = ax_heatmap.imshow(
            accumulated_frame,
            norm=matplotlib.colors.LogNorm(vmin=1.0, vmax=vmax_99_9),
            cmap="viridis",
            origin='lower',
            aspect='auto',
            interpolation='nearest'
        )
        
        cbar = plt.colorbar(im, ax=ax_heatmap, label='Event Accumulation (1 = no events)')
        
        # Superimpose line centroids
        if line_centroids:
            centroids_x = [c[0] for c in line_centroids]
            centroids_y = [c[1] for c in line_centroids]
            line_numbers = [c[2] for c in line_centroids]
            line_types_list = [c[3] for c in line_centroids]
            
            # Plot centroids as markers
            ax_heatmap.scatter(centroids_x, centroids_y, c='cyan', s=100, 
                             marker='x', linewidths=2, label='Line Centroids', alpha=0.1, zorder=10)
            
            # Add text labels for line numbers
            for x, y, num, line_type in line_centroids:
                line_type_str = "Dense" if line_type == 0 else "Sparse"
                ax_heatmap.annotate(f'{line_type_str} L{num}', (x, y), xytext=(5, 5), 
                                  textcoords='offset points', color='cyan', 
                                  fontsize=8, fontweight='bold', zorder=11)
        
        ax_heatmap.set_xlabel('X (pixels)')
        ax_heatmap.set_ylabel('Y (pixels)')
        ax_heatmap.set_title(f'Event Accumulation Heatmap with {len(line_centroids)} Line Centroids')
        ax_heatmap.set_xlim(0, img_width)
        ax_heatmap.set_ylim(0, img_height)
        ax_heatmap.legend(loc='upper right')
        
        plt.tight_layout()
        output_path_heatmap = 'may9_dual_detection_heatmap.png'
        # plt.savefig(output_path_heatmap, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"   ✓ Saved heatmap visualization to {output_path_heatmap}")
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
