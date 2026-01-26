#!/usr/bin/env python3
"""
Compare Progressive-X and DBSCAN on May 9 HDF5 Event Data
Uses the same time slice and events as may_9_test_FIXED.py
Compares algorithmic output: number of clusters/lines, visualizations, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import h5py
from time import time
import warnings
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

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

def read_hdf5_events(hdf5_path, time_window=0.05, start_time=None):
    """
    Read events from HDF5 file and create 3D point cloud from (x, y, t)
    Same function as may_9_test_FIXED.py
    
    Args:
        hdf5_path: Path to HDF5 event file
        time_window: Time window in seconds to extract (default: 0.05)
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
        p = events['p'][:]  # Polarity
        
        # Filter by time window (same as may_9_test_FIXED.py)
        if start_time is None:
            start_time = t.min()
        
        end_time = start_time + time_window * 1e6  # Convert to microseconds
        time_mask = (t >= start_time) & (t < end_time)
        
        x_filtered = x[time_mask]
        y_filtered = y[time_mask]
        t_filtered = t[time_mask]   
        p_filtered = p[time_mask] 

        # Same polarity filter as may_9_test_FIXED.py
        p_mask = (p_filtered == 1)

        x_filtered = x_filtered[p_mask]
        y_filtered = y_filtered[p_mask]
        t_filtered = t_filtered[p_mask]
        
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
    Same function as may_9_test_FIXED.py
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

def run_progressivex(points_normalized, threshold, neighbor_radius, minimum_point_number):
    """
    Run Progressive-X with same parameters as may_9_test_FIXED.py
    """
    print("\n" + "="*70)
    print("Running Progressive-X")
    print("="*70)
    
    n_points = len(points_normalized)
    
    # Same max_iters calculation as may_9_test_FIXED.py
    if n_points < 2000:
        max_iters_optimized = max(500, int(n_points * 0.5))
    elif n_points < 5000:
        max_iters_optimized = int(n_points * 4000 / 7000)
    elif n_points < 10000:
        max_iters_optimized = 4000
    elif n_points < 100000:
        max_iters_optimized = 12000
    else:
        max_iters_optimized = 30000
    
    conf_optimized = 0.05
    sampler_id_optimized = 3  # Z-Aligned sampler
    
    print(f"   Settings: conf={conf_optimized}, sampler=Z-Aligned, max_iters={max_iters_optimized:,}")
    print(f"   Threshold: {threshold:.6f}, min_points: {minimum_point_number}")
    
    t_start = time()
    lines, labeling = pyprogressivex.findLines3D(
        np.ascontiguousarray(points_normalized, dtype=np.float64),
        np.ascontiguousarray([], dtype=np.float64),  # No weights
        threshold=threshold,
        conf=conf_optimized,
        spatial_coherence_weight=0.0,
        neighborhood_ball_radius=neighbor_radius,
        maximum_tanimoto_similarity=0.40,
        max_iters=max_iters_optimized,
        minimum_point_number=minimum_point_number,
        maximum_model_number=20000,
        sampler_id=sampler_id_optimized,
        scoring_exponent=0.0,
        do_logging=False
    )
    t_end = time()
    
    num_models = len(lines)
    print(f"\n   ✓ Progressive-X found {num_models} lines in {t_end - t_start:.2f} seconds")
    
    # Count points per line
    unique_labels = np.unique(labeling)
    points_per_line = {}
    for label in unique_labels:
        if label >= 0:  # Skip outliers (label -1)
            count = np.sum(labeling == label)
            points_per_line[label] = count
    
    print(f"   Points per line: min={min(points_per_line.values()) if points_per_line else 0}, "
          f"max={max(points_per_line.values()) if points_per_line else 0}, "
          f"mean={np.mean(list(points_per_line.values())) if points_per_line else 0:.1f}")
    
    return lines, labeling, t_end - t_start

def run_dbscan(points_normalized, eps, min_samples):
    """
    Run DBSCAN clustering on normalized 3D points
    """
    print("\n" + "="*70)
    print("Running DBSCAN")
    print("="*70)
    
    print(f"   Settings: eps={eps:.6f}, min_samples={min_samples}")
    
    t_start = time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    dbscan_labels = dbscan.fit_predict(points_normalized)
    t_end = time()
    
    num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    num_noise = np.sum(dbscan_labels == -1)
    
    print(f"\n   ✓ DBSCAN found {num_clusters} clusters in {t_end - t_start:.2f} seconds")
    print(f"   Noise points: {num_noise} ({100*num_noise/len(points_normalized):.1f}%)")
    
    # Count points per cluster
    unique_labels = np.unique(dbscan_labels)
    points_per_cluster = {}
    for label in unique_labels:
        if label >= 0:  # Skip noise (label -1)
            count = np.sum(dbscan_labels == label)
            points_per_cluster[label] = count
    
    if points_per_cluster:
        print(f"   Points per cluster: min={min(points_per_cluster.values())}, "
              f"max={max(points_per_cluster.values())}, "
              f"mean={np.mean(list(points_per_cluster.values())):.1f}")
    
    return dbscan_labels, t_end - t_start

def compute_dbscan_eps(points_normalized, k=4):
    """
    Compute optimal eps for DBSCAN using k-nearest neighbors heuristic
    """
    print(f"\n   Computing DBSCAN eps using {k}-nearest neighbors...")
    
    # Sample points for faster computation
    n_sample = min(1000, len(points_normalized))
    sample_indices = np.random.choice(len(points_normalized), n_sample, replace=False)
    sample_points = points_normalized[sample_indices]
    
    # Compute k-nearest neighbors distances
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(sample_points)
    distances, indices = nbrs.kneighbors(sample_points)
    
    # Get k-th nearest neighbor distances (skip self)
    k_distances = distances[:, k]
    k_distances_sorted = np.sort(k_distances)
    
    # Use median as eps estimate
    eps_estimate = np.median(k_distances)
    
    # Alternative: use elbow method (knee point)
    # Find point of maximum curvature
    n = len(k_distances_sorted)
    if n > 10:
        # Use percentile as alternative estimate
        eps_percentile = np.percentile(k_distances, 75)
        eps_estimate = (eps_estimate + eps_percentile) / 2
    
    print(f"   Estimated eps: {eps_estimate:.6f}")
    return eps_estimate

def main():
    print("="*70)
    print("Progressive-X vs DBSCAN Comparison on May 9 HDF5 Event Data")
    print("="*70)
    
    # Read HDF5 file (same as may_9_test_FIXED.py)
    hdf5_path = 'img/may_9.hdf5'
    
    if not os.path.exists(hdf5_path):
        print(f"✗ Error: File not found: {hdf5_path}")
        return 1
    
    print("\n1. Reading HDF5 event file...")
    try:
        # Same parameters as may_9_test_FIXED.py
        points_raw, metadata = read_hdf5_events(hdf5_path, time_window=0.05, start_time=None)
        print(f"   ✓ Loaded {len(points_raw)} events")
    except Exception as e:
        print(f"   ✗ Error reading HDF5 file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Normalize time coordinate (same as may_9_test_FIXED.py)
    print("\n2. Normalizing coordinates...")
    spatial_range = max(points_raw[:, 0].max() - points_raw[:, 0].min(),
                       points_raw[:, 1].max() - points_raw[:, 1].min())
    time_range = points_raw[:, 2].max() - points_raw[:, 2].min()
    
    if time_range > 0:
        time_scale_factor = spatial_range / time_range
    else:
        time_scale_factor = 1e-6
    
    points_normalized, time_offset, time_scale = normalize_time_coordinate(
        points_raw, time_scale_factor=time_scale_factor
    )
    
    print(f"   ✓ Normalized {len(points_normalized)} points")
    print(f"   Spatial range: {spatial_range:.1f} pixels")
    print(f"   Time range: {time_range/1e6:.3f} seconds")
    print(f"   Time scale factor: {time_scale_factor:.6f}")
    
    # Compute parameters (same as may_9_test_FIXED.py)
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
    
    # Progressive-X parameters (same as may_9_test_FIXED.py)
    threshold_from_range = data_range * 0.0005
    threshold_from_noise = estimated_noise * 0.8
    final_threshold = max(threshold_from_range, threshold_from_noise)
    final_threshold = max(final_threshold, data_range * 0.0003)
    final_threshold = min(final_threshold, data_range * 0.008)
    
    neighbor_radius = data_range * 0.002
    neighbor_radius = max(neighbor_radius, estimated_noise * 0.15)
    neighbor_radius = min(neighbor_radius, data_range * 0.010)
    
    minimum_point_number = 10
    
    print(f"   Progressive-X threshold: {final_threshold:.6f}")
    print(f"   Neighborhood radius: {neighbor_radius:.6f}")
    print(f"   Minimum points: {minimum_point_number}")
    
    # DBSCAN parameters (use similar scale to Progressive-X threshold)
    # DBSCAN eps should be similar to threshold for fair comparison
    dbscan_eps = compute_dbscan_eps(points_normalized, k=4)
    # Use threshold as alternative eps if it's reasonable
    dbscan_eps = (dbscan_eps + final_threshold) / 2  # Average of both estimates
    dbscan_min_samples = minimum_point_number  # Same as Progressive-X minimum
    
    print(f"   DBSCAN eps: {dbscan_eps:.6f}")
    print(f"   DBSCAN min_samples: {dbscan_min_samples}")
    
    # Run Progressive-X
    progx_lines, progx_labeling, progx_time = run_progressivex(
        points_normalized, final_threshold, neighbor_radius, minimum_point_number
    )
    
    # Run DBSCAN
    dbscan_labels, dbscan_time = run_dbscan(points_normalized, dbscan_eps, dbscan_min_samples)
    
    # Comparison summary
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"Progressive-X:")
    print(f"  - Lines found: {len(progx_lines)}")
    print(f"  - Runtime: {progx_time:.2f} seconds")
    print(f"  - Points assigned: {np.sum(progx_labeling >= 0)} ({100*np.sum(progx_labeling >= 0)/len(points_normalized):.1f}%)")
    print(f"  - Outliers: {np.sum(progx_labeling < 0)} ({100*np.sum(progx_labeling < 0)/len(points_normalized):.1f}%)")
    
    num_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"\nDBSCAN:")
    print(f"  - Clusters found: {num_dbscan_clusters}")
    print(f"  - Runtime: {dbscan_time:.2f} seconds")
    print(f"  - Points assigned: {np.sum(dbscan_labels >= 0)} ({100*np.sum(dbscan_labels >= 0)/len(points_normalized):.1f}%)")
    print(f"  - Noise: {np.sum(dbscan_labels == -1)} ({100*np.sum(dbscan_labels == -1)/len(points_normalized):.1f}%)")
    
    print(f"\nSpeed comparison:")
    print(f"  - Progressive-X: {progx_time:.2f}s ({progx_time/dbscan_time:.1f}x slower)")
    print(f"  - DBSCAN: {dbscan_time:.2f}s")
    
    # Visualization
    print("\n4. Creating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 3D scatter plots
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    # Original data (no coloring)
    ax1.scatter(points_normalized[:, 0], points_normalized[:, 1], points_normalized[:, 2],
                c='gray', s=0.5, alpha=0.3, label='All points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('T (normalized)')
    ax1.set_title('Original Event Data')
    ax1.legend()
    
    # Progressive-X results
    unique_progx_labels = np.unique(progx_labeling)
    colors_progx = plt.cm.tab20(np.linspace(0, 1, len(unique_progx_labels)))
    for i, label in enumerate(unique_progx_labels):
        if label >= 0 and label < len(progx_lines):
            mask = (progx_labeling == label)
            ax2.scatter(points_normalized[mask, 0], points_normalized[mask, 1], points_normalized[mask, 2],
                       c=[colors_progx[i]], s=1, alpha=0.6, label=f'Line {label}')

            # Draw detected line - use label as index (assuming 0-indexed labels)
            line_point = np.array([progx_lines[label, 0], progx_lines[label, 1], progx_lines[label, 2]])
            line_dir = np.array([progx_lines[label, 3], progx_lines[label, 4], progx_lines[label, 5]])
            line_dir = line_dir / np.linalg.norm(line_dir)

            points_line = points_normalized[mask]
            if len(points_line) > 0:
                projections = np.array([np.dot(p - line_point, line_dir) for p in points_line])
                t_min, t_max = projections.min(), projections.max()
                p_start = line_point + t_min * line_dir
                p_end = line_point + t_max * line_dir
                ax2.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], [p_start[2], p_end[2]],
                        'k-', linewidth=2, alpha=0.5)

        else:
            mask = (progx_labeling == label)
            ax2.scatter(points_normalized[mask, 0], points_normalized[mask, 1], points_normalized[mask, 2],
                       c='black', s=0.5, alpha=0.2, label='Outliers')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('T (normalized)')
    ax2.set_title(f'Progressive-X: {len(progx_lines)} lines')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    
    # DBSCAN results
    unique_dbscan_labels = np.unique(dbscan_labels)
    colors_dbscan = plt.cm.tab20(np.linspace(0, 1, len(unique_dbscan_labels)))
    for i, label in enumerate(unique_dbscan_labels):
        if label >= 0:
            mask = (dbscan_labels == label)
            ax3.scatter(points_normalized[mask, 0], points_normalized[mask, 1], points_normalized[mask, 2],
                       c=[colors_dbscan[i]], s=1, alpha=0.6, label=f'Cluster {label}')
        else:
            mask = (dbscan_labels == label)
            ax3.scatter(points_normalized[mask, 0], points_normalized[mask, 1], points_normalized[mask, 2],
                       c='black', s=0.5, alpha=0.2, label='Noise')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('T (normalized)')
    ax3.set_title(f'DBSCAN: {num_dbscan_clusters} clusters')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    
    # 2D projections (X-Y plane)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Original data (X-Y projection)
    ax4.scatter(points_normalized[:, 0], points_normalized[:, 1], c='gray', s=0.5, alpha=0.3)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Original Event Data (X-Y projection)')
    ax4.set_aspect('equal')
    
    # Progressive-X (X-Y projection)
    for i, label in enumerate(unique_progx_labels):
        if label >= 0:
            mask = (progx_labeling == label)
            ax5.scatter(points_normalized[mask, 0], points_normalized[mask, 1],
                       c=[colors_progx[i]], s=4, alpha=0.6, label=f'Line {label}')
        else:
            mask = (progx_labeling == label)
            ax5.scatter(points_normalized[mask, 0], points_normalized[mask, 1],
                       c='black', s=0.5, alpha=0.2)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title(f'Progressive-X: {len(progx_lines)} lines (X-Y projection)')
    ax5.set_aspect('equal')
    
    # DBSCAN (X-Y projection)
    for i, label in enumerate(unique_dbscan_labels):
        if label >= 0:
            mask = (dbscan_labels == label)
            ax6.scatter(points_normalized[mask, 0], points_normalized[mask, 1],
                       c=[colors_dbscan[i]], s=4, alpha=0.9, label=f'Cluster {label}')
        else:
            mask = (dbscan_labels == label)
            ax6.scatter(points_normalized[mask, 0], points_normalized[mask, 1],
                       c='black', s=0.5, alpha=0.2)
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title(f'DBSCAN: {num_dbscan_clusters} clusters (X-Y projection)')
    ax6.set_aspect('equal')
    
    plt.tight_layout()
    
    output_path = 'progressivex_vs_dbscan_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved comparison visualization to {output_path}")
    
    plt.show()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
