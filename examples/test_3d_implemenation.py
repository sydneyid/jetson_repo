#!/usr/bin/env python3
"""
Test Progressive-X 3D Line Detection
Creates a synthetic 3D scene with 3 lines and 50% outliers, then detects them
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

def generate_3d_lines(num_points_per_line=100, noise_sigma=0.01, outlier_fraction=0.5, seed=42):
    """
    Generate 3D synthetic data with 3 lines and outliers
    
    Args:
        num_points_per_line: Number of points per line
        noise_sigma: Standard deviation of Gaussian noise added to inlier points
        outlier_fraction: Fraction of outliers (0.5 = 50% outliers)
        seed: Random seed
    
    Returns:
        points: N×3 array of 3D points
        labels: N array of labels (0=outlier, 1,2,3=line indices)
        true_lines: List of true line parameters [point, direction] for each line
    """
    np.random.seed(seed)
    
    # Define 3 lines in 3D space
    # Each line is defined by a point and a direction vector
    true_lines = [
        {
            'point': np.array([0.0, 0.0, 0.0]),
            'direction': np.array([1.0, 0.0, 0.0])  # Line along x-axis
        },
        {
            'point': np.array([0.0, 0.0, 0.0]),
            'direction': np.array([0.0, 1.0, 0.0])  # Line along y-axis
        },
        {
            'point': np.array([0.0, 0.0, 0.0]),
            'direction': np.array([0.0, 0.0, 1.0])  # Line along z-axis
        }
    ]
    
    # Generate points on each line
    all_points = []
    all_labels = []
    
    for line_idx, line in enumerate(true_lines):
        line_label = line_idx + 1  # Labels: 1, 2, 3
        
        # Generate points along the line
        # Parameter t ranges from -1 to 1
        t_values = np.linspace(-1, 1, num_points_per_line)
        
        for t in t_values:
            # Point on line: p = point + t * direction
            point = line['point'] + t * line['direction']
            
            # Add Gaussian noise perpendicular to the line
            # Generate two perpendicular vectors to the line direction
            dir_norm = line['direction'] / np.linalg.norm(line['direction'])
            
            # Find a perpendicular vector
            if abs(dir_norm[0]) < 0.9:
                perp1 = np.array([1, 0, 0])
            else:
                perp1 = np.array([0, 1, 0])
            perp1 = perp1 - np.dot(perp1, dir_norm) * dir_norm
            perp1 = perp1 / np.linalg.norm(perp1)
            
            # Second perpendicular vector (cross product)
            perp2 = np.cross(dir_norm, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # Add noise in perpendicular plane
            noise = noise_sigma * np.random.randn() * perp1 + noise_sigma * np.random.randn() * perp2
            noisy_point = point + noise
            
            all_points.append(noisy_point)
            all_labels.append(line_label)
    
    # Generate outliers (50% of inlier points)
    num_inliers = len(all_points)
    num_outliers = int(num_inliers * outlier_fraction / (1 - outlier_fraction))
    
    # Outliers are random points in a bounding box
    # Find bounding box of inlier points
    inlier_points = np.array(all_points)
    bbox_min = inlier_points.min(axis=0) - 0.5
    bbox_max = inlier_points.max(axis=0) + 0.5
    
    for _ in range(num_outliers):
        outlier = np.random.uniform(bbox_min, bbox_max)
        all_points.append(outlier)
        all_labels.append(0)  # 0 = outlier
    
    # Shuffle points
    indices = np.random.permutation(len(all_points))
    all_points = np.array(all_points)[indices]
    all_labels = np.array(all_labels)[indices]
    
    return all_points, all_labels, true_lines

def point_to_line_distance_3d(point, line_point, line_dir):
    """
    Calculate perpendicular distance from a 3D point to a 3D line
    
    Args:
        point: 3D point [x, y, z]
        line_point: Point on the line [x, y, z]
        line_dir: Direction vector of the line [dx, dy, dz] (should be normalized)
    
    Returns:
        Distance from point to line
    """
    vec = point - line_point
    cross = np.cross(vec, line_dir)
    return np.linalg.norm(cross)

def main():
    print("="*70)
    print("Progressive-X: 3D Line Detection Test")
    print("="*70)
    
    # Generate synthetic 3D data
    print("\n1. Generating synthetic 3D data...")
    num_points_per_line = 100
    noise_sigma = 0.01
    outlier_fraction = 0.5  # 50% outliers
    
    points, true_labels, true_lines = generate_3d_lines(
        num_points_per_line=num_points_per_line,
        noise_sigma=noise_sigma,
        outlier_fraction=outlier_fraction,
        seed=42
    )
    
    print(f"   ✓ Generated {len(points)} points")
    print(f"   ✓ {num_points_per_line * 3} inlier points (3 lines)")
    print(f"   ✓ {np.sum(true_labels == 0)} outlier points ({100*np.sum(true_labels == 0)/len(points):.1f}%)")
    print(f"   ✓ Point range: X=[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
          f"Y=[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
          f"Z=[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # Compute optimal parameters
    print("\n2. Computing optimal parameters...")
    
    # Estimate noise from nearest neighbor distances
    from scipy.spatial.distance import cdist
    n_sample = min(100, len(points))
    sample_indices = np.random.choice(len(points), n_sample, replace=False)
    sample_points = points[sample_indices]
    dists = cdist(sample_points, sample_points)
    np.fill_diagonal(dists, np.inf)
    nearest_dists = np.min(dists, axis=1)
    estimated_noise = np.median(nearest_dists)
    
    # Compute data range
    data_range = np.max(points.max(axis=0) - points.min(axis=0))
    
    # Set threshold as a percentage of data range
    threshold_from_range = data_range * 0.02  # 2% of range
    threshold_from_noise = estimated_noise * 3.0  # 3× noise estimate
    final_threshold = max(threshold_from_range, threshold_from_noise)
    final_threshold = max(final_threshold, data_range * 0.005)  # At least 0.5% of range
    final_threshold = min(final_threshold, data_range * 0.05)   # At most 5% of range
    
    # Neighborhood radius: scale with data
    neighbor_radius = data_range * 0.005  # 0.5% of range
    neighbor_radius = max(neighbor_radius, estimated_noise * 0.5)
    neighbor_radius = min(neighbor_radius, data_range * 0.02)
    
    # Minimum point number
    minimum_point_number = max(10, int(len(points) * 0.02))  # 2% of points, at least 10
    
    print(f"   ✓ Estimated noise: {estimated_noise:.6f}")
    print(f"   ✓ Threshold: {final_threshold:.6f} ({100*final_threshold/data_range:.2f}% of range)")
    print(f"   ✓ Neighborhood radius: {neighbor_radius:.6f} ({100*neighbor_radius/data_range:.2f}% of range)")
    print(f"   ✓ Minimum points per model: {minimum_point_number} ({100*minimum_point_number/len(points):.1f}% of total)")
    
    # Run Progressive-X 3D line detection
    print("\n3. Running Progressive-X 3D line detection...")
    
    try:
        t = time()
        lines, labeling = pyprogressivex.findLines3D(
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray([], dtype=np.float64),  # No weights
            threshold=final_threshold,
            conf=0.99,
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=neighbor_radius,
            maximum_tanimoto_similarity=1.0,  # No merging
            max_iters=1000,
            minimum_point_number=minimum_point_number,
            maximum_model_number=5,  # Allow up to 5 models (we expect 3)
            sampler_id=0,  # Uniform sampler
            scoring_exponent=1.0,
            do_logging=False
        )
        elapsed_time = time() - t
        
        num_models = lines.shape[0] if lines.size > 0 else 0
        print(f"   ✓ Completed in {elapsed_time:.2f} seconds")
        print(f"   ✓ Found {num_models} line models")
        
        # Check labeling statistics
        unique_labels, counts = np.unique(labeling, return_counts=True)
        print(f"   Labeling statistics:")
        for label, count in zip(unique_labels, counts):
            if label == 0:
                print(f"     Label 0: {count} points ({100*count/len(points):.1f}%) [outliers]")
            else:
                print(f"     Label {label}: {count} points ({100*count/len(points):.1f}%) [line {label}]")
        
        # Evaluate detection quality
        print(f"\n4. Evaluating detection quality...")
        
        # Check if labels are 0-indexed or 1-indexed
        # If label 0 has many points and forms a good line, it might be the first model
        use_zero_indexed = False
        if num_models > 0:
            label0_mask = (labeling == 0)
            if np.sum(label0_mask) > minimum_point_number:
                # Check if label 0 forms a good line
                points_label0 = points[label0_mask]
                if len(points_label0) >= 2:
                    # Fit line to label 0 points
                    centroid = points_label0.mean(axis=0)
                    centered = points_label0 - centroid
                    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
                    line_dir = Vt[0]
                    line_dir = line_dir / np.linalg.norm(line_dir)
                    
                    # Check distances
                    distances_label0 = [point_to_line_distance_3d(p, centroid, line_dir) for p in points_label0]
                    inlier_ratio_label0 = np.sum(np.array(distances_label0) <= final_threshold) / len(distances_label0)
                    if inlier_ratio_label0 > 0.6:
                        use_zero_indexed = True
                        print(f"   ✓ Using 0-indexed interpretation: label 0 = first model")
        
        # Calculate accuracy metrics
        # Match detected lines to true lines
        detected_lines = []
        for idx in range(num_models):
            if use_zero_indexed:
                instance_label = idx
            else:
                instance_label = idx + 1
            
            # Extract line parameters: [p₀x, p₀y, p₀z, dx, dy, dz]
            line_point = np.array([lines[idx, 0], lines[idx, 1], lines[idx, 2]])
            line_dir = np.array([lines[idx, 3], lines[idx, 4], lines[idx, 5]])
            line_dir = line_dir / np.linalg.norm(line_dir)  # Normalize
            
            detected_lines.append({
                'point': line_point,
                'direction': line_dir,
                'label': instance_label
            })
            
            # Calculate distances for points assigned to this line
            mask = (labeling == instance_label)
            points_assigned = points[mask]
            
            if len(points_assigned) > 0:
                distances = [point_to_line_distance_3d(p, line_point, line_dir) for p in points_assigned]
                distances = np.array(distances)
                inlier_count = np.sum(distances <= final_threshold)
                inlier_ratio = inlier_count / len(distances)
                
                print(f"   Line {idx+1}: {len(points_assigned)} points, "
                      f"{inlier_ratio*100:.1f}% within threshold, "
                      f"mean distance: {distances.mean():.4f}")
        
        # Calculate classification accuracy
        # Determine which detected line corresponds to which true line
        # For simplicity, match by closest direction alignment
        if num_models > 0:
            # Create predicted labels based on detected lines
            predicted_labels = np.zeros(len(points), dtype=int)
            for det_line in detected_lines:
                mask = (labeling == det_line['label'])
                predicted_labels[mask] = det_line['label']
            
            # For evaluation, we'll compare outlier detection
            true_outlier_mask = (true_labels == 0)
            if use_zero_indexed:
                pred_outlier_mask = (labeling == num_models)  # Highest label = outliers
            else:
                pred_outlier_mask = (labeling == 0)  # Label 0 = outliers
            
            n_true_outliers = np.sum(true_outlier_mask)
            n_pred_outliers = np.sum(pred_outlier_mask)
            n_correct_outliers = np.sum(true_outlier_mask & pred_outlier_mask)
            
            print(f"\n   Outlier Detection:")
            print(f"     True outliers: {n_true_outliers}")
            print(f"     Predicted outliers: {n_pred_outliers}")
            print(f"     Correctly identified: {n_correct_outliers} ({100*n_correct_outliers/n_true_outliers:.1f}%)")
            
            # Overall accuracy
            correct = np.sum(predicted_labels > 0) == np.sum(true_labels > 0)  # Same number of inliers
            print(f"     Overall: {'✓' if correct else '✗'} Detected {num_models} lines (expected 3)")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Visualize results
    print("\n5. Visualizing results...")
    
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: True lines and points
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot true lines
    colors_true = ['red', 'green', 'blue']
    for idx, line in enumerate(true_lines):
        # Draw line segment
        t_range = np.linspace(-1.2, 1.2, 100)
        line_points = line['point'][None, :] + t_range[:, None] * line['direction'][None, :]
        ax1.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
                color=colors_true[idx], linewidth=3, alpha=0.7, label=f'True Line {idx+1}')
    
    # Plot points colored by true labels
    for label in [0, 1, 2, 3]:
        mask = (true_labels == label)
        if np.any(mask):
            if label == 0:
                ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                          c='gray', s=10, alpha=0.3, marker='x', label='True Outliers')
            else:
                ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                          c=colors_true[label-1], s=20, alpha=0.6, label=f'True Line {label}')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('True Lines and Points', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    
    # Plot 2: Detected lines
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Plot detected lines
    colors_det = plt.cm.tab10(np.linspace(0, 1, num_models)) if num_models > 0 else []
    for idx, det_line in enumerate(detected_lines):
        # Draw line segment (extend in both directions)
        t_range = np.linspace(-1.5, 1.5, 100)
        line_points = det_line['point'][None, :] + t_range[:, None] * det_line['direction'][None, :]
        ax2.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                color=colors_det[idx], linewidth=2, linestyle='--', alpha=0.8, label=f'Detected Line {idx+1}')
    
    # Plot points colored by detected labels
    for label in unique_labels:
        mask = (labeling == label)
        if np.any(mask):
            if (use_zero_indexed and label == num_models) or (not use_zero_indexed and label == 0):
                ax2.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                          c='red', s=10, alpha=0.5, marker='x', label='Detected Outliers')
            else:
                line_idx = label if use_zero_indexed else label - 1
                if line_idx < len(colors_det):
                    ax2.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                              c=[colors_det[line_idx]], s=30, alpha=0.7, 
                              label=f'Line {line_idx+1}' if label == (line_idx if use_zero_indexed else line_idx+1) else '')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Detected Lines ({num_models} lines)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    
    # Plot 3: Comparison (detected vs true)
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot true lines (thin, light)
    for idx, line in enumerate(true_lines):
        t_range = np.linspace(-1.2, 1.2, 100)
        line_points = line['point'][None, :] + t_range[:, None] * line['direction'][None, :]
        ax3.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                color=colors_true[idx], linewidth=1, alpha=0.3, linestyle='-')
    
    # Plot detected lines (thick, dark)
    for idx, det_line in enumerate(detected_lines):
        t_range = np.linspace(-1.5, 1.5, 100)
        line_points = det_line['point'][None, :] + t_range[:, None] * det_line['direction'][None, :]
        ax3.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                color=colors_det[idx], linewidth=2, linestyle='--', alpha=0.8)
    
    # Plot only outliers for clarity
    true_outlier_mask = (true_labels == 0)
    pred_outlier_mask = (labeling == (num_models if use_zero_indexed else 0))
    
    ax3.scatter(points[true_outlier_mask, 0], points[true_outlier_mask, 1], points[true_outlier_mask, 2],
              c='gray', s=10, alpha=0.3, marker='x', label='True Outliers')
    ax3.scatter(points[pred_outlier_mask, 0], points[pred_outlier_mask, 1], points[pred_outlier_mask, 2],
              c='red', s=15, alpha=0.5, marker='+', label='Detected Outliers')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Comparison: True (thin) vs Detected (thick)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    
    plt.suptitle(f"3D Line Detection: {len(points)} points, {num_models} lines detected, "
                f"{100*np.sum(true_labels == 0)/len(points):.0f}% outliers",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = '3d_line_detection_result.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.show()
    
    print("\n" + "="*70)
    print("✓ Test completed successfully!")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
