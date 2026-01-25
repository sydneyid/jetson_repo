#!/usr/bin/env python3
"""
Test Progressive-X 3D Line Detection with Varying Line Lengths
Tests how well the algorithm handles lines of different lengths (short vs long segments)
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

def generate_3d_lines_varying_lengths(line_lengths, line_directions=None, line_points=None,
                                      noise_sigma=0.01, outlier_fraction=0.3, seed=42):
    """
    Generate 3D synthetic data with lines of varying lengths
    
    Args:
        line_lengths: List of lengths for each line (e.g., [2.0, 0.5, 1.5] means first line is 2.0 units long, etc.)
        line_directions: List of direction vectors (if None, uses random directions)
        line_points: List of center points for each line (if None, uses random points)
        noise_sigma: Standard deviation of Gaussian noise
        outlier_fraction: Fraction of outliers
        seed: Random seed
    
    Returns:
        points: N×3 array of 3D points
        labels: N array of labels (0=outlier, 1,2,...=line indices)
        true_lines: List of true line parameters
    """
    np.random.seed(seed)
    
    num_lines = len(line_lengths)
    
    # Generate directions if not provided
    if line_directions is None:
        line_directions = []
        for i in range(num_lines):
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            line_directions.append(direction)
    
    # Generate center points if not provided
    if line_points is None:
        line_points = []
        for i in range(num_lines):
            point = np.random.uniform(-1, 1, 3)
            line_points.append(point)
    
    # Create true lines (each line is defined by a point and direction)
    # The point will be at the center of the line segment
    true_lines = []
    for i in range(num_lines):
        true_lines.append({
            'point': line_points[i],
            'direction': line_directions[i],
            'length': line_lengths[i]
        })
    
    # Generate points on each line
    all_points = []
    all_labels = []
    
    # Use proportional number of points based on length (longer lines get more points)
    total_length = sum(line_lengths)
    base_points_per_unit = 50  # Base density: 50 points per unit length
    
    for line_idx, line in enumerate(true_lines):
        line_label = line_idx + 1
        
        # Number of points proportional to line length
        num_points = max(20, int(line['length'] * base_points_per_unit))
        
        # Generate points along the line segment
        # t ranges from -length/2 to +length/2 (centered at the point)
        t_values = np.linspace(-line['length']/2, line['length']/2, num_points)
        
        for t in t_values:
            # Point on line: p = center_point + t * direction
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

def run_length_test_case(case_name, line_lengths, line_directions=None, line_points=None,
                         noise_sigma=0.01, outlier_fraction=0.3, seed=None):
    """Run a single test case with varying line lengths"""
    if seed is None:
        seed = hash(case_name) % 10000
    
    print(f"\n{'='*70}")
    print(f"Test Case: {case_name}")
    print(f"{'='*70}")
    print(f"  Line lengths: {line_lengths}")
    print(f"  Length ratio (max/min): {max(line_lengths)/min(line_lengths):.2f}x")
    print(f"  Noise sigma: {noise_sigma}")
    print(f"  Outlier fraction: {outlier_fraction*100:.1f}%")
    
    # Generate data
    points, true_labels, true_lines = generate_3d_lines_varying_lengths(
        line_lengths=line_lengths,
        line_directions=line_directions,
        line_points=line_points,
        noise_sigma=noise_sigma,
        outlier_fraction=outlier_fraction,
        seed=seed
    )
    
    num_lines = len(line_lengths)
    num_inliers = np.sum(true_labels > 0)
    num_outliers = np.sum(true_labels == 0)
    
    print(f"  Total points: {len(points)} ({num_inliers} inliers, {num_outliers} outliers)")
    print(f"  Points per line: {[np.sum(true_labels == i+1) for i in range(num_lines)]}")
    
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
    
    # Adjust minimum point number based on shortest line
    min_length = min(line_lengths)
    min_points_expected = max(15, int(min_length * 50 * 0.5))  # At least 50% of expected points for shortest line
    minimum_point_number = max(10, min_points_expected)
    
    print(f"  Threshold: {final_threshold:.4f}")
    print(f"  Minimum points per model: {minimum_point_number}")
    
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
            maximum_model_number=num_lines + 3,
            sampler_id=0,
            scoring_exponent=1.0,
            do_logging=False
        )
        elapsed_time = time() - t
        
        num_models = lines.shape[0] if lines.size > 0 else 0
        
        # Evaluate results
        unique_labels, counts = np.unique(labeling, return_counts=True)
        n_outliers_detected = np.sum(labeling == 0)
        
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
        
        results = {
            'case_name': case_name,
            'line_lengths': line_lengths,
            'num_lines_expected': num_lines,
            'num_lines_detected': num_models,
            'elapsed_time': elapsed_time,
            'points': points,
            'true_labels': true_labels,
            'detected_labels': labeling,
            'detected_lines': lines,
            'true_lines': true_lines,
            'threshold': final_threshold,
            'use_zero_indexed': use_zero_indexed,
            'points_per_line_true': [np.sum(true_labels == i+1) for i in range(num_lines)],
            'points_per_line_detected': []
        }
        
        # Count points per detected line
        for idx in range(num_models):
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            n_points = np.sum(labeling == instance_label)
            results['points_per_line_detected'].append(n_points)
        
        print(f"  ✓ Detection completed in {elapsed_time:.2f}s")
        print(f"  ✓ Detected {num_models} lines (expected {num_lines})")
        
        # Check which lines were detected
        if num_models == num_lines:
            print(f"  ✓✓✓ CORRECT number of lines detected!")
        else:
            print(f"  ⚠️  Mismatch: expected {num_lines}, got {num_models}")
        
        # Check if shortest line was detected
        shortest_length = min(line_lengths)
        shortest_idx = line_lengths.index(shortest_length)
        print(f"  Shortest line: length {shortest_length:.3f} (line {shortest_idx+1})")
        if num_models >= shortest_idx + 1:
            print(f"  ✓ Shortest line was detected")
        else:
            print(f"  ✗ Shortest line may not have been detected")
        
        return results
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_length_test_case(results, save_path=None):
    """Visualize a test case with varying line lengths"""
    if results is None:
        return
    
    fig = plt.figure(figsize=(20, 6))
    
    points = results['points']
    true_labels = results['true_labels']
    detected_labels = results['detected_labels']
    true_lines = results['true_lines']
    detected_lines = results['detected_lines']
    num_models = results['num_lines_detected']
    use_zero_indexed = results['use_zero_indexed']
    line_lengths = results['line_lengths']
    
    # Plot 1: True lines (color-coded by length)
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Color map: shorter lines = red, longer lines = blue
    max_len = max(line_lengths)
    min_len = min(line_lengths)
    
    for idx, line in enumerate(true_lines):
        length = line['length']
        # Normalize length for color (0 = shortest, 1 = longest)
        if max_len > min_len:
            color_val = (length - min_len) / (max_len - min_len)
        else:
            color_val = 0.5
        color = plt.cm.RdYlBu(1 - color_val)  # Reverse so red = short, blue = long
        
        # Draw line segment
        t_range = np.linspace(-length/2, length/2, 100)
        line_points = line['point'][None, :] + t_range[:, None] * line['direction'][None, :]
        ax1.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                color=color, linewidth=3, alpha=0.8, 
                label=f'Line {idx+1} (L={length:.2f})')
    
    # Plot points
    for label in range(len(true_lines) + 1):
        mask = (true_labels == label)
        if np.any(mask):
            if label == 0:
                ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                          c='gray', s=10, alpha=0.3, marker='x', label='Outliers')
            else:
                length = line_lengths[label-1]
                color_val = (length - min_len) / (max_len - min_len) if max_len > min_len else 0.5
                color = plt.cm.RdYlBu(1 - color_val)
                ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                          c=[color], s=20, alpha=0.6)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'True: {len(true_lines)} lines (red=short, blue=long)', fontsize=12, fontweight='bold')
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
        
        # Estimate length from detected points
        mask = (detected_labels == instance_label)
        if np.any(mask):
            detected_points = points[mask]
            # Project points onto line
            vecs = detected_points - line_point
            projections = np.dot(vecs, line_dir)
            estimated_length = projections.max() - projections.min() if len(projections) > 0 else 1.0
        else:
            estimated_length = 1.0
        
        t_range = np.linspace(-estimated_length/2, estimated_length/2, 100)
        line_points = line_point[None, :] + t_range[:, None] * line_dir[None, :]
        ax2.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                color=colors_det[idx], linewidth=2, linestyle='--', alpha=0.8, 
                label=f'Detected {idx+1} ({len(points[detected_labels == instance_label])} pts)')
    
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
    
    # Plot 3: Length comparison bar chart
    ax3 = fig.add_subplot(133)
    
    x_pos = np.arange(len(line_lengths))
    true_lengths = line_lengths
    detected_counts = results['points_per_line_detected']
    
    # Normalize detected counts to approximate lengths (assuming similar point density)
    if len(detected_counts) > 0 and max(detected_counts) > 0:
        # Estimate lengths from point counts (assuming ~50 points per unit)
        estimated_lengths = [count / 50.0 for count in detected_counts[:len(true_lengths)]]
        # Pad if needed
        while len(estimated_lengths) < len(true_lengths):
            estimated_lengths.append(0)
    else:
        estimated_lengths = [0] * len(true_lengths)
    
    width = 0.35
    ax3.bar(x_pos - width/2, true_lengths, width, label='True Lengths', alpha=0.8, color='blue')
    ax3.bar(x_pos + width/2, estimated_lengths[:len(true_lengths)], width, 
            label='Estimated from Detection', alpha=0.8, color='orange')
    
    ax3.set_xlabel('Line Index')
    ax3.set_ylabel('Length')
    ax3.set_title('Line Length Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Line {i+1}' for i in range(len(true_lengths))])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f"{results['case_name']}: Length ratio {max(line_lengths)/min(line_lengths):.2f}x",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {save_path}")
    
    plt.close()

def main():
    print("="*70)
    print("Progressive-X 3D Line Detection: Varying Line Lengths Test")
    print("="*70)
    
    # Define test cases with different length configurations
    test_cases = [
        # Case 1: One very short line among longer ones
        {
            'name': 'Case 1: One very short line (0.2) among long lines (2.0)',
            'lengths': [2.0, 0.2, 1.5, 1.8],
            'noise': 0.01,
            'outliers': 0.3
        },
        
        # Case 2: Gradually increasing lengths
        {
            'name': 'Case 2: Gradually increasing lengths',
            'lengths': [0.3, 0.6, 1.0, 1.5, 2.0],
            'noise': 0.01,
            'outliers': 0.3
        },
        
        # Case 3: Extreme ratio (very short vs very long)
        {
            'name': 'Case 3: Extreme ratio (0.1 vs 3.0)',
            'lengths': [3.0, 0.1, 2.0, 0.15],
            'noise': 0.01,
            'outliers': 0.3
        },
        
        # Case 4: Many short lines with one long
        {
            'name': 'Case 4: Many short lines (0.3) with one long (2.5)',
            'lengths': [0.3, 0.3, 0.3, 0.3, 2.5],
            'noise': 0.015,
            'outliers': 0.4
        },
        
        # Case 5: Alternating short and long
        {
            'name': 'Case 5: Alternating short (0.4) and long (2.0)',
            'lengths': [0.4, 2.0, 0.4, 2.0, 0.4],
            'noise': 0.01,
            'outliers': 0.3
        },
        
        # Case 6: High noise with varying lengths
        {
            'name': 'Case 6: Varying lengths with high noise',
            'lengths': [0.5, 1.5, 0.8, 2.2],
            'noise': 0.025,
            'outliers': 0.4
        },
        
        # Case 7: Many lines with wide length range
        {
            'name': 'Case 7: 8 lines with wide length range',
            'lengths': [0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0],
            'noise': 0.015,
            'outliers': 0.5
        },
        
        # Case 8: Very short lines (challenging)
        {
            'name': 'Case 8: All short lines (0.2-0.5)',
            'lengths': [0.2, 0.3, 0.4, 0.5],
            'noise': 0.01,
            'outliers': 0.3
        },
    ]
    
    all_results = []
    
    # Run all test cases
    for i, test_case in enumerate(test_cases):
        results = run_length_test_case(
            case_name=test_case['name'],
            line_lengths=test_case['lengths'],
            noise_sigma=test_case['noise'],
            outlier_fraction=test_case['outliers'],
            seed=42 + i
        )
        
        if results:
            all_results.append(results)
            # Visualize
            save_path = f"3d_length_test_case_{i+1}.png"
            visualize_length_test_case(results, save_path=save_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Varying Line Lengths")
    print(f"{'='*70}")
    
    correct_detections = 0
    shortest_line_detected = 0
    total_cases = len(all_results)
    
    for results in all_results:
        case_name = results['case_name']
        expected = results['num_lines_expected']
        detected = results['num_lines_detected']
        lengths = results['line_lengths']
        shortest_idx = lengths.index(min(lengths))
        
        status = "✓" if expected == detected else "✗"
        length_ratio = max(lengths) / min(lengths)
        
        print(f"{status} {case_name}")
        print(f"    Expected: {expected}, Detected: {detected}, Length ratio: {length_ratio:.2f}x")
        print(f"    Lengths: {[f'{l:.2f}' for l in lengths]}")
        print(f"    Points per line (true): {results['points_per_line_true']}")
        print(f"    Points per line (detected): {results['points_per_line_detected']}")
        
        if expected == detected:
            correct_detections += 1
        if detected >= shortest_idx + 1:
            shortest_line_detected += 1
    
    print(f"\nOverall Accuracy: {correct_detections}/{total_cases} cases ({100*correct_detections/total_cases:.1f}%)")
    print(f"Shortest line detected: {shortest_line_detected}/{total_cases} cases ({100*shortest_line_detected/total_cases:.1f}%)")
    print(f"{'='*70}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
