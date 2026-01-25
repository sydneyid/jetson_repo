#!/usr/bin/env python3
"""
Comprehensive Robustness Test for Progressive-X 3D Line Detection
Tests multiple scenarios: different numbers of lines, orientations, noise levels, and outlier fractions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from time import time

# Add parent directory to path to import pyprogressivex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyprogressivex

def generate_3d_lines_advanced(num_lines=3, num_points_per_line=100, noise_sigma=0.01, 
                                outlier_fraction=0.5, seed=42, line_config='random'):
    """
    Generate 3D synthetic data with lines in various configurations
    
    Args:
        num_lines: Number of lines to generate
        num_points_per_line: Number of points per line
        noise_sigma: Standard deviation of Gaussian noise
        outlier_fraction: Fraction of outliers
        seed: Random seed
        line_config: Configuration type ('axis_aligned', 'random', 'intersecting', 'parallel')
    
    Returns:
        points: N×3 array of 3D points
        labels: N array of labels (0=outlier, 1,2,...=line indices)
        true_lines: List of true line parameters
    """
    np.random.seed(seed)
    
    true_lines = []
    
    if line_config == 'axis_aligned':
        # Lines along x, y, z axes
        directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        points_on_lines = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0])
        ]
        for i in range(num_lines):
            if i < len(directions):
                true_lines.append({
                    'point': points_on_lines[i],
                    'direction': directions[i]
                })
            else:
                # Additional lines along other axes
                axis = i % 3
                direction = np.zeros(3)
                direction[axis] = 1.0
                point = np.random.uniform(-0.5, 0.5, 3)
                true_lines.append({'point': point, 'direction': direction})
    
    elif line_config == 'random':
        # Randomly oriented lines
        for i in range(num_lines):
            # Random direction (unit vector)
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # Random point on line
            point = np.random.uniform(-1, 1, 3)
            
            true_lines.append({
                'point': point,
                'direction': direction
            })
    
    elif line_config == 'intersecting':
        # Lines that intersect at a common point
        intersection_point = np.array([0.0, 0.0, 0.0])
        for i in range(num_lines):
            # Random direction
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            true_lines.append({
                'point': intersection_point,
                'direction': direction
            })
    
    elif line_config == 'parallel':
        # Parallel lines (same direction, different positions)
        base_direction = np.random.randn(3)
        base_direction = base_direction / np.linalg.norm(base_direction)
        
        for i in range(num_lines):
            # Offset perpendicular to the direction
            offset = np.random.randn(3)
            offset = offset - np.dot(offset, base_direction) * base_direction
            offset = offset / (np.linalg.norm(offset) + 1e-10) * (i * 0.3)
            
            point = np.array([0.0, 0.0, 0.0]) + offset
            
            true_lines.append({
                'point': point,
                'direction': base_direction
            })
    
    else:
        raise ValueError(f"Unknown line_config: {line_config}")
    
    # Generate points on each line
    all_points = []
    all_labels = []
    
    for line_idx, line in enumerate(true_lines):
        line_label = line_idx + 1
        
        # Generate points along the line
        t_values = np.linspace(-1, 1, num_points_per_line)
        
        for t in t_values:
            # Point on line: p = point + t * direction
            point = line['point'] + t * line['direction']
            
            # Add Gaussian noise perpendicular to the line
            dir_norm = line['direction'] / np.linalg.norm(line['direction'])
            
            # Find a perpendicular vector
            if abs(dir_norm[0]) < 0.9:
                perp1 = np.array([1, 0, 0])
            else:
                perp1 = np.array([0, 1, 0])
            perp1 = perp1 - np.dot(perp1, dir_norm) * dir_norm
            perp1_norm = np.linalg.norm(perp1)
            if perp1_norm > 1e-10:
                perp1 = perp1 / perp1_norm
            else:
                perp1 = np.array([0, 0, 1]) - np.dot(np.array([0, 0, 1]), dir_norm) * dir_norm
                perp1 = perp1 / np.linalg.norm(perp1)
            
            # Second perpendicular vector
            perp2 = np.cross(dir_norm, perp1)
            perp2_norm = np.linalg.norm(perp2)
            if perp2_norm > 1e-10:
                perp2 = perp2 / perp2_norm
            else:
                perp2 = np.array([0, 1, 0]) - np.dot(np.array([0, 1, 0]), dir_norm) * dir_norm
                perp2 = perp2 / np.linalg.norm(perp2)
            
            # Add noise in perpendicular plane
            noise = noise_sigma * np.random.randn() * perp1 + noise_sigma * np.random.randn() * perp2
            noisy_point = point + noise
            
            all_points.append(noisy_point)
            all_labels.append(line_label)
    
    # Generate outliers
    num_inliers = len(all_points)
    num_outliers = int(num_inliers * outlier_fraction / (1 - outlier_fraction))
    
    # Bounding box
    inlier_points = np.array(all_points)
    bbox_min = inlier_points.min(axis=0) - 0.5
    bbox_max = inlier_points.max(axis=0) + 0.5
    
    for _ in range(num_outliers):
        outlier = np.random.uniform(bbox_min, bbox_max)
        all_points.append(outlier)
        all_labels.append(0)
    
    # Shuffle points
    indices = np.random.permutation(len(all_points))
    all_points = np.array(all_points)[indices]
    all_labels = np.array(all_labels)[indices]
    
    return all_points, all_labels, true_lines

def point_to_line_distance_3d(point, line_point, line_dir):
    """Calculate perpendicular distance from a 3D point to a 3D line"""
    vec = point - line_point
    cross = np.cross(vec, line_dir)
    return np.linalg.norm(cross)

def run_test_case(case_name, num_lines, line_config, noise_sigma, outlier_fraction, 
                  num_points_per_line=100, seed=None):
    """Run a single test case and return results"""
    if seed is None:
        seed = hash(case_name) % 10000
    
    print(f"\n{'='*70}")
    print(f"Test Case: {case_name}")
    print(f"{'='*70}")
    print(f"  Configuration: {line_config}")
    print(f"  Number of lines: {num_lines}")
    print(f"  Points per line: {num_points_per_line}")
    print(f"  Noise sigma: {noise_sigma}")
    print(f"  Outlier fraction: {outlier_fraction*100:.1f}%")
    
    # Generate data
    points, true_labels, true_lines = generate_3d_lines_advanced(
        num_lines=num_lines,
        num_points_per_line=num_points_per_line,
        noise_sigma=noise_sigma,
        outlier_fraction=outlier_fraction,
        seed=seed,
        line_config=line_config
    )
    
    print(f"  Total points: {len(points)} ({num_lines * num_points_per_line} inliers, {np.sum(true_labels == 0)} outliers)")
    
    # Compute parameters
    from scipy.spatial.distance import cdist
    n_sample = min(100, len(points))
    sample_indices = np.random.choice(len(points), n_sample, replace=False)
    sample_points = points[sample_indices]
    dists = cdist(sample_points, sample_points)
    np.fill_diagonal(dists, np.inf)
    nearest_dists = np.min(dists, axis=1)
    estimated_noise = np.median(nearest_dists)
    
    data_range = np.max(points.max(axis=0) - points.min(axis=0))
    
    threshold_from_range = data_range * 0.02
    threshold_from_noise = estimated_noise * 3.0
    final_threshold = max(threshold_from_range, threshold_from_noise)
    final_threshold = max(final_threshold, data_range * 0.005)
    final_threshold = min(final_threshold, data_range * 0.05)
    
    neighbor_radius = data_range * 0.005
    neighbor_radius = max(neighbor_radius, estimated_noise * 0.5)
    neighbor_radius = min(neighbor_radius, data_range * 0.02)
    
    minimum_point_number = max(10, int(len(points) * 0.02))
    
    # Run detection
    try:
        t = time()
        lines, labeling = pyprogressivex.findLines3D(
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray([], dtype=np.float64),
            threshold=final_threshold,
            conf=0.99,
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=neighbor_radius,
            maximum_tanimoto_similarity=1.0,
            max_iters=1000,
            minimum_point_number=minimum_point_number,
            maximum_model_number=num_lines + 5,  # Allow some extra
            sampler_id=0,
            scoring_exponent=1.0,
            do_logging=False
        )
        elapsed_time = time() - t
        
        num_models = lines.shape[0] if lines.size > 0 else 0
        
        # Evaluate results
        results = {
            'case_name': case_name,
            'num_lines_expected': num_lines,
            'num_lines_detected': num_models,
            'elapsed_time': elapsed_time,
            'points': points,
            'true_labels': true_labels,
            'detected_labels': labeling,
            'detected_lines': lines,
            'true_lines': true_lines,
            'threshold': final_threshold
        }
        
        # Calculate accuracy metrics
        unique_labels, counts = np.unique(labeling, return_counts=True)
        n_outliers_detected = np.sum(labeling == 0)
        n_outliers_true = np.sum(true_labels == 0)
        
        # Check if labels are 0-indexed
        use_zero_indexed = False
        if num_models > 0 and n_outliers_detected > minimum_point_number:
            label0_mask = (labeling == 0)
            points_label0 = points[label0_mask]
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
        
        results['use_zero_indexed'] = use_zero_indexed
        results['n_outliers_detected'] = n_outliers_detected
        results['n_outliers_true'] = n_outliers_true
        
        print(f"  ✓ Detection completed in {elapsed_time:.2f}s")
        print(f"  ✓ Detected {num_models} lines (expected {num_lines})")
        print(f"  ✓ Outliers: detected {n_outliers_detected}, true {n_outliers_true}")
        
        if num_models == num_lines:
            print(f"  ✓✓✓ CORRECT number of lines detected!")
        else:
            print(f"  ⚠️  Mismatch: expected {num_lines}, got {num_models}")
        
        return results
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_test_case(results, save_path=None):
    """Visualize a single test case"""
    if results is None:
        return
    
    fig = plt.figure(figsize=(18, 6))
    
    points = results['points']
    true_labels = results['true_labels']
    detected_labels = results['detected_labels']
    true_lines = results['true_lines']
    detected_lines = results['detected_lines']
    num_models = results['num_lines_detected']
    use_zero_indexed = results['use_zero_indexed']
    
    # Plot 1: True lines
    ax1 = fig.add_subplot(131, projection='3d')
    colors_true = plt.cm.tab20(np.linspace(0, 1, len(true_lines)))
    for idx, line in enumerate(true_lines):
        t_range = np.linspace(-1.2, 1.2, 100)
        line_points = line['point'][None, :] + t_range[:, None] * line['direction'][None, :]
        ax1.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                color=colors_true[idx], linewidth=3, alpha=0.7, label=f'True Line {idx+1}')
    
    for label in range(len(true_lines) + 1):
        mask = (true_labels == label)
        if np.any(mask):
            if label == 0:
                ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                          c='gray', s=10, alpha=0.3, marker='x', label='True Outliers')
            else:
                ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                          c=[colors_true[label-1]], s=20, alpha=0.6)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'True: {len(true_lines)} lines', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    
    # Plot 2: Detected lines
    ax2 = fig.add_subplot(132, projection='3d')
    colors_det = plt.cm.tab10(np.linspace(0, 1, num_models)) if num_models > 0 else []
    
    detected_line_params = []
    for idx in range(num_models):
        if use_zero_indexed:
            instance_label = idx
        else:
            instance_label = idx + 1
        
        line_point = np.array([detected_lines[idx, 0], detected_lines[idx, 1], detected_lines[idx, 2]])
        line_dir = np.array([detected_lines[idx, 3], detected_lines[idx, 4], detected_lines[idx, 5]])
        line_dir = line_dir / np.linalg.norm(line_dir)
        detected_line_params.append({'point': line_point, 'direction': line_dir, 'label': instance_label})
        
        t_range = np.linspace(-1.5, 1.5, 100)
        line_points = line_point[None, :] + t_range[:, None] * line_dir[None, :]
        ax2.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                color=colors_det[idx], linewidth=2, linestyle='--', alpha=0.8, label=f'Detected {idx+1}')
    
    unique_labels = np.unique(detected_labels)
    for label in unique_labels:
        mask = (detected_labels == label)
        if np.any(mask):
            if (use_zero_indexed and label == num_models) or (not use_zero_indexed and label == 0):
                ax2.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                          c='red', s=10, alpha=0.5, marker='x', label='Detected Outliers')
            else:
                line_idx = label if use_zero_indexed else label - 1
                if line_idx < len(colors_det):
                    ax2.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                              c=[colors_det[line_idx]], s=30, alpha=0.7)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Detected: {num_models} lines', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    
    # Plot 3: Comparison
    ax3 = fig.add_subplot(133, projection='3d')
    
    # True lines (thin)
    for idx, line in enumerate(true_lines):
        t_range = np.linspace(-1.2, 1.2, 100)
        line_points = line['point'][None, :] + t_range[:, None] * line['direction'][None, :]
        ax3.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                color=colors_true[idx], linewidth=1, alpha=0.3, linestyle='-')
    
    # Detected lines (thick)
    for det_line in detected_line_params:
        t_range = np.linspace(-1.5, 1.5, 100)
        line_points = det_line['point'][None, :] + t_range[:, None] * det_line['direction'][None, :]
        line_idx = det_line['label'] if use_zero_indexed else det_line['label'] - 1
        if line_idx < len(colors_det):
            ax3.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                    color=colors_det[line_idx], linewidth=2, linestyle='--', alpha=0.8)
    
    # Outliers only
    true_outlier_mask = (true_labels == 0)
    pred_outlier_mask = (detected_labels == (num_models if use_zero_indexed else 0))
    
    ax3.scatter(points[true_outlier_mask, 0], points[true_outlier_mask, 1], points[true_outlier_mask, 2],
              c='gray', s=10, alpha=0.3, marker='x', label='True Outliers')
    ax3.scatter(points[pred_outlier_mask, 0], points[pred_outlier_mask, 1], points[pred_outlier_mask, 2],
              c='red', s=15, alpha=0.5, marker='+', label='Detected Outliers')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Comparison', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    
    plt.suptitle(f"{results['case_name']}: {len(true_lines)} expected, {num_models} detected",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {save_path}")
    
    plt.close()

def main():
    print("="*70)
    print("Progressive-X 3D Line Detection: Comprehensive Robustness Test")
    print("="*70)
    
    # Define test cases
    test_cases = [
        # Basic cases
        {'name': 'Case 1: 3 lines, axis-aligned', 'num_lines': 3, 'config': 'axis_aligned', 
         'noise': 0.01, 'outliers': 0.3, 'points_per_line': 100},
        {'name': 'Case 2: 5 lines, random', 'num_lines': 5, 'config': 'random',
         'noise': 0.01, 'outliers': 0.4, 'points_per_line': 80},
        {'name': 'Case 3: 7 lines, intersecting', 'num_lines': 7, 'config': 'intersecting',
         'noise': 0.015, 'outliers': 0.5, 'points_per_line': 70},
        {'name': 'Case 4: 4 lines, parallel', 'num_lines': 4, 'config': 'parallel',
         'noise': 0.01, 'outliers': 0.3, 'points_per_line': 90},
        
        # More challenging cases
        {'name': 'Case 5: 10 lines, random', 'num_lines': 10, 'config': 'random',
         'noise': 0.02, 'outliers': 0.5, 'points_per_line': 60},
        {'name': 'Case 6: 6 lines, high noise', 'num_lines': 6, 'config': 'random',
         'noise': 0.03, 'outliers': 0.4, 'points_per_line': 100},
        {'name': 'Case 7: 8 lines, many outliers', 'num_lines': 8, 'config': 'random',
         'noise': 0.01, 'outliers': 0.6, 'points_per_line': 70},
        {'name': 'Case 8: 5 lines, intersecting, high noise', 'num_lines': 5, 'config': 'intersecting',
         'noise': 0.025, 'outliers': 0.4, 'points_per_line': 80},
    ]
    
    all_results = []
    
    # Run all test cases
    for i, test_case in enumerate(test_cases):
        results = run_test_case(
            case_name=test_case['name'],
            num_lines=test_case['num_lines'],
            line_config=test_case['config'],
            noise_sigma=test_case['noise'],
            outlier_fraction=test_case['outliers'],
            num_points_per_line=test_case['points_per_line'],
            seed=42 + i
        )
        
        if results:
            all_results.append(results)
            # Visualize
            save_path = f"3d_test_case_{i+1}_{test_case['name'].replace(' ', '_').replace(':', '')}.png"
            visualize_test_case(results, save_path=save_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    correct_detections = 0
    total_cases = len(all_results)
    
    for results in all_results:
        case_name = results['case_name']
        expected = results['num_lines_expected']
        detected = results['num_lines_detected']
        status = "✓" if expected == detected else "✗"
        print(f"{status} {case_name}: Expected {expected}, Detected {detected}")
        if expected == detected:
            correct_detections += 1
    
    print(f"\nAccuracy: {correct_detections}/{total_cases} cases ({100*correct_detections/total_cases:.1f}%)")
    print(f"{'='*70}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
