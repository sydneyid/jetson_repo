#!/usr/bin/env python3
"""
Stability Test for Progressive-X Jupiter Line Detection
Runs test_jupiter_in_focus multiple times and analyzes variance in results
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main function from test_jupiter_in_focus
from test_jupiter_in_focus import (
    read_hdf5_events, normalize_time_coordinate, 
    point_to_line_distance_3d, parallel_check
)

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


def run_single_detection(hdf5_path, time_window=0.05, events_constraint=500, start_time=14.06, 
                        threshold_dense=None, threshold_sparse=None, conf_dense=0.05,
                        max_iters=1000, minimum_point_number=10, minimum_point_number_sparse=5,
                        neighbor_radius=None, maximum_tanimoto_similarity=0.35, sampler_id=3):
    """
    Run a single detection and return results
    
    Returns:
        dict with keys: num_models, num_dense, num_sparse, lines, line_types, labeling,
                       valid_line_indices, points_normalized, metadata
    """
    # Read and normalize points
    points_raw, metadata = read_hdf5_events(hdf5_path, time_window=time_window, 
                                           events_constraint=events_constraint, 
                                           start_time=start_time)
    
    # Normalize time coordinate
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
    
    # Compute parameters if not provided
    n_sample = min(100, len(points_normalized))
    sample_indices = np.random.choice(len(points_normalized), n_sample, replace=False)
    sample_points = points_normalized[sample_indices]
    from scipy.spatial.distance import cdist
    dists = cdist(sample_points, sample_points)
    np.fill_diagonal(dists, np.inf)
    nearest_dists = np.min(dists, axis=1)
    estimated_noise = np.median(nearest_dists)
    
    data_range = np.max(points_normalized.max(axis=0) - points_normalized.min(axis=0))
    
    if threshold_dense is None:
        threshold_from_range = data_range * 0.001
        threshold_from_noise = estimated_noise * 1.0
        base_threshold = max(threshold_from_range, threshold_from_noise)
        base_threshold = max(base_threshold, data_range * 0.0003)
        base_threshold = min(base_threshold, data_range * 0.012)
        threshold_dense = base_threshold * 0.9
        threshold_sparse = base_threshold * 2.5
    
    if neighbor_radius is None:
        neighbor_radius = data_range * 0.002
        neighbor_radius = max(neighbor_radius, estimated_noise * 0.15)
        neighbor_radius = min(neighbor_radius, data_range * 0.010)
    
    # Run detection
    lines, labeling, line_types = pyprogressivex.findLines3DDual(
        np.ascontiguousarray(points_normalized, dtype=np.float64),
        np.ascontiguousarray([], dtype=np.float64),
        threshold_dense=threshold_dense,
        threshold_sparse=threshold_sparse,
        conf=conf_dense,
        spatial_coherence_weight=0.0,
        neighborhood_ball_radius=neighbor_radius,
        maximum_tanimoto_similarity=maximum_tanimoto_similarity,
        max_iters=max_iters,
        minimum_point_number_dense=minimum_point_number,
        minimum_point_number_sparse=minimum_point_number_sparse,
        maximum_model_number=20000,
        sampler_id=sampler_id,
        scoring_exponent=0.0,
        do_logging=False
    )
    
    num_models = lines.shape[0] if lines.size > 0 else 0
    dense_mask = (line_types == 0)
    sparse_mask = (line_types == 1)
    num_dense = np.sum(dense_mask)
    num_sparse = np.sum(sparse_mask)
    
    # Filter by parallel check
    valid_line_indices = parallel_check(lines, labeling, num_models, 
                                       parallel_threshold=1.0, print_updates=False)
    
    return {
        'num_models': num_models,
        'num_dense': num_dense,
        'num_sparse': num_sparse,
        'num_valid': len(valid_line_indices),
        'lines': lines,
        'line_types': line_types,
        'labeling': labeling,
        'valid_line_indices': valid_line_indices,
        'points_normalized': points_normalized,
        'metadata': metadata,
        'threshold_dense': threshold_dense,
        'threshold_sparse': threshold_sparse
    }


def extract_line_metrics(result):
    """
    Extract metrics for each detected line
    
    Returns:
        list of dicts, one per valid line
    """
    metrics = []
    lines = result['lines']
    line_types = result['line_types']
    labeling = result['labeling']
    points_normalized = result['points_normalized']
    valid_line_indices = result['valid_line_indices']
    threshold_dense = result['threshold_dense']
    threshold_sparse = result['threshold_sparse']
    
    for idx in valid_line_indices:
        # Map line index to label: line idx -> label idx + 1 (label 0 is outliers)
        instance_label = idx + 1
        mask = (labeling == instance_label)
        points_line = points_normalized[mask]
        
        if len(points_line) > 0:
            line_point = np.array([lines[idx, 0], lines[idx, 1], lines[idx, 2]])
            line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
            line_dir = line_dir / np.linalg.norm(line_dir)
            
            line_type = line_types[idx] if idx < len(line_types) else 0
            threshold_used = threshold_dense if line_type == 0 else threshold_sparse
            
            # Calculate metrics
            centroid = np.mean(points_line, axis=0)
            distances = np.array([point_to_line_distance_3d(p, line_point, line_dir) 
                                 for p in points_line])
            inliers = np.sum(distances <= threshold_used)
            inlier_ratio = inliers / len(points_line) if len(points_line) > 0 else 0
            
            metrics.append({
                'line_idx': idx,
                'line_type': 'DENSE' if line_type == 0 else 'SPARSE',
                'num_points': len(points_line),
                'centroid_x': centroid[0],
                'centroid_y': centroid[1],
                'centroid_z': centroid[2],
                'inlier_ratio': inlier_ratio,
                'mean_distance': np.mean(distances),
                'max_distance': np.max(distances),
                'line_point': line_point,
                'line_dir': line_dir
            })
    
    return metrics


def run_stability_test(n_runs=20, hdf5_path=None, **kwargs):
    """
    Run detection multiple times and collect statistics
    
    Args:
        n_runs: Number of runs to perform
        hdf5_path: Path to HDF5 file (if None, uses default)
        **kwargs: Additional parameters to pass to run_single_detection
    """
    if hdf5_path is None:
        hdf5_path = '/Users/sydneydolan/Documents/may9_data/Jupiter_In_Focus.hdf5'
    
    print("="*70)
    print(f"Progressive-X Jupiter Line Detection Stability Test")
    print(f"Running {n_runs} iterations...")
    print("="*70)
    
    all_results = []
    all_metrics = []
    
    for run_idx in range(n_runs):
        print(f"\nRun {run_idx + 1}/{n_runs}...", end=' ', flush=True)
        try:
            result = run_single_detection(hdf5_path, **kwargs)
            metrics = extract_line_metrics(result)
            all_results.append(result)
            all_metrics.append(metrics)
            print(f"✓ ({result['num_valid']} valid lines)")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Stability Analysis Results")
    print(f"{'='*70}\n")
    
    # Overall statistics
    num_models_list = [r['num_models'] for r in all_results]
    num_dense_list = [r['num_dense'] for r in all_results]
    num_sparse_list = [r['num_sparse'] for r in all_results]
    num_valid_list = [r['num_valid'] for r in all_results]
    
    print(f"Overall Detection Statistics:")
    print(f"  Total models detected:")
    print(f"    Mean: {np.mean(num_models_list):.2f}, Std: {np.std(num_models_list):.2f}, "
          f"Range: [{np.min(num_models_list)}, {np.max(num_models_list)}]")
    print(f"  Dense lines:")
    print(f"    Mean: {np.mean(num_dense_list):.2f}, Std: {np.std(num_dense_list):.2f}, "
          f"Range: [{np.min(num_dense_list)}, {np.max(num_dense_list)}]")
    print(f"  Sparse lines:")
    print(f"    Mean: {np.mean(num_sparse_list):.2f}, Std: {np.std(num_sparse_list):.2f}, "
          f"Range: [{np.min(num_sparse_list)}, {np.max(num_sparse_list)}]")
    print(f"  Valid lines (after parallel filter):")
    print(f"    Mean: {np.mean(num_valid_list):.2f}, Std: {np.std(num_valid_list):.2f}, "
          f"Range: [{np.min(num_valid_list)}, {np.max(num_valid_list)}]")
    
    # Analyze line positions (centroids)
    # Group lines by approximate position to track same line across runs
    print(f"\n{'='*70}")
    print(f"Line Position Stability (centroids):")
    print(f"{'='*70}")
    
    # Collect all line centroids
    all_centroids = []
    for run_idx, metrics in enumerate(all_metrics):
        for m in metrics:
            all_centroids.append({
                'run': run_idx,
                'line_idx': m['line_idx'],
                'type': m['line_type'],
                'x': m['centroid_x'],
                'y': m['centroid_y'],
                'num_points': m['num_points']
            })
    
    if len(all_centroids) > 0:
        # Cluster centroids by position to find same lines across runs
        centroids_xy = np.array([[c['x'], c['y']] for c in all_centroids])
        
        # Estimate number of distinct lines (use mode of num_valid)
        n_clusters = int(np.round(np.mean(num_valid_list)))
        if n_clusters > 0 and len(centroids_xy) >= n_clusters:
            try:
                try:
                    from sklearn.cluster import KMeans
                except ImportError:
                    print("  Note: sklearn not available, using simple distance-based clustering")
                    KMeans = None
                
                if KMeans is not None:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(centroids_xy)
                else:
                    # Simple distance-based clustering
                    from scipy.spatial.distance import cdist
                    # Use first run's centroids as cluster centers
                    first_run_centroids = [c for c in all_centroids if c['run'] == 0]
                    if len(first_run_centroids) > 0:
                        centers = np.array([[c['x'], c['y']] for c in first_run_centroids])
                        distances = cdist(centroids_xy, centers)
                        cluster_labels = np.argmin(distances, axis=1)
                    else:
                        cluster_labels = np.zeros(len(centroids_xy), dtype=int)
                
                # Group by cluster
                clusters = defaultdict(list)
                for i, c in enumerate(all_centroids):
                    clusters[cluster_labels[i]].append(c)
                
                print(f"\nFound {len(clusters)} distinct line positions across runs:")
                for cluster_id, cluster_points in sorted(clusters.items()):
                    if len(cluster_points) > 0:
                        x_mean = np.mean([p['x'] for p in cluster_points])
                        y_mean = np.mean([p['y'] for p in cluster_points])
                        x_std = np.std([p['x'] for p in cluster_points])
                        y_std = np.std([p['y'] for p in cluster_points])
                        detection_rate = len(cluster_points) / n_runs
                        types = [p['type'] for p in cluster_points]
                        type_mode = max(set(types), key=types.count)
                        
                        print(f"  Line {cluster_id + 1}:")
                        print(f"    Position: ({x_mean:.1f} ± {x_std:.1f}, {y_mean:.1f} ± {y_std:.1f})")
                        print(f"    Detection rate: {detection_rate:.1%} ({len(cluster_points)}/{n_runs} runs)")
                        print(f"    Type: {type_mode}")
            except Exception as e:
                print(f"  Could not cluster lines: {e}")
    
    # Create visualization
    print(f"\n{'='*70}")
    print(f"Generating visualization...")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of lines detected per run
    ax1 = axes[0, 0]
    ax1.plot(range(1, n_runs + 1), num_models_list, 'o-', label='Total models', alpha=0.7)
    ax1.plot(range(1, n_runs + 1), num_dense_list, 's-', label='Dense lines', alpha=0.7)
    ax1.plot(range(1, n_runs + 1), num_sparse_list, '^-', label='Sparse lines', alpha=0.7)
    ax1.plot(range(1, n_runs + 1), num_valid_list, 'd-', label='Valid lines', alpha=0.7)
    ax1.axhline(np.mean(num_valid_list), color='r', linestyle='--', alpha=0.5, label=f'Mean ({np.mean(num_valid_list):.1f})')
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('Number of Lines')
    ax1.set_title('Line Count Stability Across Runs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of line counts
    ax2 = axes[0, 1]
    ax2.hist(num_valid_list, bins=range(int(np.min(num_valid_list)), int(np.max(num_valid_list)) + 2), 
            alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(num_valid_list), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(num_valid_list):.2f}')
    ax2.axvline(np.median(num_valid_list), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(num_valid_list):.2f}')
    ax2.set_xlabel('Number of Valid Lines')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Valid Line Counts')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Centroid positions (if we have clustering)
    ax3 = axes[1, 0]
    if len(all_centroids) > 0:
        for cluster_id, cluster_points in sorted(clusters.items()):
            if len(cluster_points) > 0:
                x_vals = [p['x'] for p in cluster_points]
                y_vals = [p['y'] for p in cluster_points]
                x_mean = np.mean(x_vals)
                y_mean = np.mean(y_vals)
                x_std = np.std(x_vals)
                y_std = np.std(y_vals)
                
                # Plot points
                ax3.scatter(x_vals, y_vals, alpha=0.3, s=20)
                # Plot mean with error bars
                ax3.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, 
                           fmt='o', capsize=5, capthick=2, markersize=8, 
                           label=f'Line {cluster_id + 1}')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        ax3.set_title('Line Centroid Positions (with std dev)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
    
    # Plot 4: Coefficient of variation
    ax4 = axes[1, 1]
    metrics_to_plot = {
        'Total Models': num_models_list,
        'Dense Lines': num_dense_list,
        'Sparse Lines': num_sparse_list,
        'Valid Lines': num_valid_list
    }
    cv_values = []
    metric_names = []
    for name, values in metrics_to_plot.items():
        if np.mean(values) > 0:
            cv = np.std(values) / np.mean(values)
            cv_values.append(cv)
            metric_names.append(name)
    
    if len(cv_values) > 0:
        bars = ax4.bar(metric_names, cv_values, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Coefficient of Variation (σ/μ)')
        ax4.set_title('Stability Metrics (Lower = More Stable)')
        ax4.grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for bar, cv in zip(bars, cv_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cv:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = 'jupiter_stability_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {output_path}")
    plt.close()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"{'='*70}")
    print(f"  Total runs: {n_runs}")
    print(f"  Successful runs: {len(all_results)}")
    print(f"  Average valid lines per run: {np.mean(num_valid_list):.2f} ± {np.std(num_valid_list):.2f}")
    if np.mean(num_valid_list) > 0:
        cv_valid = np.std(num_valid_list) / np.mean(num_valid_list)
        print(f"  Coefficient of variation (valid lines): {cv_valid:.3f}")
        if cv_valid < 0.1:
            print(f"  → Excellent stability (CV < 0.1)")
        elif cv_valid < 0.2:
            print(f"  → Good stability (CV < 0.2)")
        elif cv_valid < 0.3:
            print(f"  → Moderate stability (CV < 0.3)")
        else:
            print(f"  → Low stability (CV >= 0.3)")
    
    return all_results, all_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test stability of Progressive-X line detection')
    parser.add_argument('--runs', type=int, default=20, help='Number of runs (default: 20)')
    parser.add_argument('--hdf5', type=str, default=None, help='Path to HDF5 file')
    args = parser.parse_args()
    
    results, metrics = run_stability_test(n_runs=args.runs, hdf5_path=args.hdf5)
    
    print(f"\n{'='*70}")
    print(f"Stability test completed!")
    print(f"{'='*70}")
