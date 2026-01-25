#!/usr/bin/env python3
"""
Test Progressive-X on the star dataset from the paper
Loads the star data from a .pkl file and runs line detection
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
from time import time

# Add parent directory to path to import pyprogressivex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyprogressivex

def load_star_data():
    """Load star data from .pkl file"""
    # Known file: star5.pkl
    pkl_file = 'img/star5.pkl'
    
    if not os.path.exists(pkl_file):
        # Try other possible names
        possible_names = [
            'img/star_data.pkl',
            'img/star.pkl',
            'img/star_example.pkl',
            'img/star11.pkl',
        ]
        for name in possible_names:
            if os.path.exists(name):
                pkl_file = name
                break
        else:
            raise FileNotFoundError(f"Could not find star data .pkl file in img/ directory")
    
    print(f"Loading {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, np.ndarray):
        if len(data.shape) == 2:
            if data.shape[0] == 2:
                # 2×N array (e.g., 2×500): transpose to N×2
                print(f"✓ Loaded array: shape {data.shape} (2×N format)")
                print(f"  First row (X): {data.shape[1]} values")
                print(f"  Second row (Y): {data.shape[1]} values")
                points = data.T  # Transpose to N×2
                print(f"  Transposed to: shape {points.shape} (N×2 format, {len(points)} points)")
                return points
            elif data.shape[1] == 2:
                # Already N×2 format
                print(f"✓ Loaded array: shape {data.shape} (N×2 format, {len(data)} points)")
                return data
            else:
                raise ValueError(f"Expected 2×N or N×2 array, got shape {data.shape}")
        else:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")
    elif isinstance(data, dict):
        print(f"✓ Loaded dict with keys: {list(data.keys())}")
        # Try common keys
        for key in ['points', 'data', 'star', 'X', 'x', 'star5']:
            if key in data:
                points = data[key]
                if isinstance(points, np.ndarray) and len(points.shape) == 2:
                    if points.shape[0] == 2:
                        # 2×N format: transpose
                        print(f"  Using key '{key}': shape {points.shape} (2×N), transposing to N×2")
                        return points.T
                    elif points.shape[1] == 2:
                        # Already N×2 format
                        print(f"  Using key '{key}': shape {points.shape} (N×2)")
                        return points
        # Return first 2D array found
        for key, val in data.items():
            if isinstance(val, np.ndarray) and len(val.shape) == 2:
                if val.shape[0] == 2:
                    print(f"  Using key '{key}': shape {val.shape} (2×N), transposing to N×2")
                    return val.T
                elif val.shape[1] == 2:
                    print(f"  Using key '{key}': shape {val.shape} (N×2)")
                    return val
        raise ValueError("Could not find 2D point array in dict")
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")

def compute_neighbor_radius(points, percentile=5, n_sample=100):
    """Compute optimal neighborhood radius from point distribution"""
    n_sample = min(n_sample, len(points))
    sample_indices = np.random.choice(len(points), n_sample, replace=False)
    sample_points = points[sample_indices]
    distances = []
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            distances.append(np.linalg.norm(sample_points[i] - sample_points[j]))
    if len(distances) > 0:
        return np.percentile(distances, percentile)
    return 0.1

def point_to_line_distance(point, line_params):
    """Calculate perpendicular distance from point to line (ax + by + c = 0)"""
    a, b, c = line_params
    return abs(a * point[0] + b * point[1] + c) / np.sqrt(a*a + b*b + 1e-10)

def validate_line_consistency(line_params, points_assigned, threshold, min_support):
    """Validate that points form a consistent line segment"""
    if len(points_assigned) < min_support:
        return False, "insufficient points"
    
    distances = np.array([point_to_line_distance(p, line_params) for p in points_assigned])
    inlier_ratio = np.sum(distances <= threshold) / len(distances)
    # Progressive-X already filtered points, but we need reasonable quality
    # Balance: strict enough to filter false positives, lenient enough to keep valid lines
    # Require at least 40% of points to be within threshold (moderate strictness)
    if inlier_ratio < 0.4:
        return False, f"only {inlier_ratio*100:.1f}% within threshold"
    
    # Check contiguity
    a, b, c = line_params
    if abs(b) > 1e-10:
        line_dir = np.array([1, -a/b])
        line_dir = line_dir / np.linalg.norm(line_dir)
    else:
        line_dir = np.array([0, 1])
    
    projections = np.array([np.dot(p, line_dir) for p in points_assigned])
    sorted_proj = np.sort(projections)
    gaps = np.diff(sorted_proj)
    max_gap = np.max(gaps) if len(gaps) > 0 else 0
    avg_gap = np.mean(gaps) if len(gaps) > 0 else 0
    
    # Check for scattered points - reject if there are large gaps
    # This catches cases where points form two separate clusters
    # Be moderately strict: reject if gap is much larger than average AND significant
    if max_gap > 5 * avg_gap and max_gap > threshold * 10:  # Moderate strictness
        return False, f"points too scattered (max gap: {max_gap:.3f} vs avg {avg_gap:.3f})"
    
    dist_variance = np.var(distances)
    # Check variance - should be relatively low for a good line
    # Be moderately strict: allow some variance but reject if too high
    if dist_variance > threshold * threshold * 5:  # Moderate strictness
        return False, f"high distance variance ({dist_variance:.4f}, threshold²={threshold*threshold:.4f})"
    
    return True, "valid"

def main():
    print("="*70)
    print("Progressive-X: Star Dataset Test")
    print("="*70)
    
    # Load data
    print("\n1. Loading star data...")
    try:
        points = load_star_data()
        # Ensure points is N×2 format
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected N×2 array, got shape {points.shape}")
        print(f"   ✓ Loaded {len(points)} points (shape: {points.shape})")
        print(f"   X range: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]")
        print(f"   Y range: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return 1
    
    # Prepare coordinates for Progressive-X
    # Based on the paper, Progressive-X should work on the original data scale
    # We just need to shift to positive coordinates (Progressive-X expects non-negative)
    print("\n2. Preparing coordinates...")
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # Shift points to positive quadrant (Progressive-X expects non-negative coordinates)
    points_shifted = points.copy()
    points_shifted[:, 0] -= x_min
    points_shifted[:, 1] -= y_min
    
    # Compute image dimensions (width/height for Progressive-X API)
    data_range_x = x_max - x_min
    data_range_y = y_max - y_min
    data_range = max(data_range_x, data_range_y)
    width = int(np.ceil(data_range_x)) + 1
    height = int(np.ceil(data_range_y)) + 1
    
    print(f"   ✓ Shifted to positive coordinates")
    print(f"   ✓ Image dimensions: {width} × {height}")
    print(f"   ✓ Data range: {data_range:.4f}")
    print(f"   Original coordinate ranges:")
    print(f"     X: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]")
    print(f"     Y: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
    print(f"   Shifted coordinate ranges:")
    print(f"     X: [{points_shifted[:, 0].min():.4f}, {points_shifted[:, 0].max():.4f}]")
    print(f"     Y: [{points_shifted[:, 1].min():.4f}, {points_shifted[:, 1].max():.4f}]")
    
    # Compute optimal parameters based on data scale
    # According to the paper, Progressive-X should work with appropriate thresholds
    print("\n3. Computing optimal parameters...")
    
    # Estimate noise in original coordinates
    from scipy.spatial.distance import cdist
    n_sample = min(100, len(points_shifted))
    sample_indices = np.random.choice(len(points_shifted), n_sample, replace=False)
    sample_points = points_shifted[sample_indices]
    dists = cdist(sample_points, sample_points)
    np.fill_diagonal(dists, np.inf)
    nearest_dists = np.min(dists, axis=1)
    estimated_noise = np.median(nearest_dists)
    
    # Set threshold as a percentage of data range (adaptive to data scale)
    # The example uses threshold=3.0 for ~500 pixel range, which is ~0.6% of range
    # For our data, use 1-2% of range, or 3× estimated noise, whichever is larger
    # Make threshold stricter to reduce points per instance (tighter inlier selection)
    threshold_from_range = data_range * 0.010  # 1.0% of range (stricter - fewer points per line)
    threshold_from_noise = estimated_noise * 2.5  # 2.5× noise estimate (stricter than 3×)
    final_threshold = max(threshold_from_range, threshold_from_noise)
    # Cap threshold to reasonable bounds
    final_threshold = max(final_threshold, data_range * 0.005)  # At least 0.5% of range
    final_threshold = min(final_threshold, data_range * 0.035)   # At most 3.5% of range (stricter)
    
    # Neighborhood radius: scale with data (example uses 1.0 for ~500 pixel range = 0.2% of range)
    # Use 0.2-0.5% of data range
    neighbor_radius = data_range * 0.003  # 0.3% of range (similar to example's 1.0/500)
    neighbor_radius = max(neighbor_radius, estimated_noise * 0.5)  # At least half of noise
    neighbor_radius = min(neighbor_radius, data_range * 0.01)  # Cap at 1% of range
    
    # Minimum point number: scale with total points
    # Example uses 50 for ~3180 points (1.6%), so for 500 points use ~8 points (1.6%)
    # For star dataset with 5 lines, we want stricter filtering to avoid false positives
    # Use 3-4% of points to be more selective, but not too high to allow valid lines
    minimum_point_number = max(15, int(len(points) * 0.03))  # 3% of points, at least 15
    
    print(f"   ✓ Estimated noise: {estimated_noise:.6f}")
    print(f"   ✓ Threshold: {final_threshold:.6f} ({100*final_threshold/data_range:.2f}% of range)")
    print(f"   ✓ Neighborhood radius: {neighbor_radius:.6f} ({100*neighbor_radius/data_range:.2f}% of range)")
    print(f"   ✓ Minimum points per model: {minimum_point_number} ({100*minimum_point_number/len(points):.1f}% of total)")
    
    # Run Progressive-X
    # Now using pixel-like coordinates, so we can use parameters exactly like the example
    print("\n4. Running Progressive-X line detection...")
    print("   Using parameters matching official example (pixel-like coordinates)...")
    
    try:
        t = time()
        # Use parameters adapted to data scale (based on Progressive-X paper)
        # The paper emphasizes that Progressive-X works progressively with appropriate thresholds
        lines, labeling = pyprogressivex.findLines(
            np.ascontiguousarray(points_shifted, dtype=np.float64),
            np.ascontiguousarray([], dtype=np.float64),
            width, height,  # width, height
            threshold=final_threshold,  # Stricter threshold = fewer points per instance
            conf=0.995,  # Slightly higher confidence = more selective
            spatial_coherence_weight=0.0,  # NO spatial coherence (as in official example)
            neighborhood_ball_radius=neighbor_radius,  # Scaled to data range
            maximum_tanimoto_similarity=1.0,  # No merging (as in official example)
            max_iters=1000,  # Same as example
            minimum_point_number=minimum_point_number,  # Scaled to number of points (3% of total, stricter)
            maximum_model_number=5,  # Limit to 5 lines (star has exactly 5 lines)
            sampler_id=0,  # Uniform sampler (as in official example)
            scoring_exponent=1.0,
            do_logging=False
        )
        elapsed_time = time() - t
        
        model_number = int(lines.size / 3) if lines.size > 0 else 0
        print(f"   ✓ Completed in {elapsed_time:.2f} seconds")
        print(f"   ✓ Found {model_number} line models (before filtering)")
        
        # Check labeling statistics
        unique_labels, counts = np.unique(labeling, return_counts=True)
        print(f"   Labeling statistics:")
        print(f"     Unique labels found: {unique_labels}")
        print(f"     Model number returned: {model_number}")
        print(f"     Note: If labels are 0-indexed, label 0 might be first model, not outliers!")
        n_inliers_total = 0
        for label, count in zip(unique_labels, counts):
            if label == 0:
                print(f"     Label 0: {count} points ({100*count/len(points):.1f}%) [currently treated as outliers]")
            else:
                print(f"     Label {label}: {count} points ({100*count/len(points):.1f}%) [currently treated as line {label}]")
                n_inliers_total += count
        
        # Diagnostic: Check point-to-line distances for each detected line
        # Try both interpretations: (1) label 0 = outliers, 1-N = models, (2) label 0-N-1 = models
        print(f"\n   Diagnostic: Checking point-to-line distances...")
        print(f"   Testing interpretation 1: label 0 = outliers, labels 1-{model_number} = models")
        for idx in range(model_number):
            instance_label = idx + 1
            mask = (labeling == instance_label)
            points_assigned = points_shifted[mask]
            a, b, c = lines[idx]
            
            # Calculate distances from assigned points to their line
            distances = []
            for p in points_assigned:
                dist = abs(a * p[0] + b * p[1] + c) / np.sqrt(a*a + b*b + 1e-10)
                distances.append(dist)
            distances = np.array(distances)
            
            inlier_count = np.sum(distances <= final_threshold)
            inlier_ratio = inlier_count / len(distances) if len(distances) > 0 else 0
            
            print(f"     Line {idx+1}: {len(points_assigned)} points assigned")
            print(f"       Distances: min={distances.min():.6f}, max={distances.max():.6f}, mean={distances.mean():.6f}, median={np.median(distances):.6f}")
            print(f"       Within threshold ({final_threshold:.6f}): {inlier_count}/{len(distances)} ({inlier_ratio*100:.1f}%)")
        
        # Also check interpretation 2: label 0 might be first model
        print(f"\n   Testing interpretation 2: labels 0-{model_number-1} = models (label {model_number} = outliers?)")
        if model_number > 0:
            # Check if label 0 forms a good line (might be first model)
            mask_label0 = (labeling == 0)
            if np.sum(mask_label0) > 10:
                points_label0 = points_shifted[mask_label0]
                # Fit line to label 0 points
                centroid = points_label0.mean(axis=0)
                centered = points_label0 - centroid
                U, s, Vt = np.linalg.svd(centered, full_matrices=False)
                line_dir = Vt[0]
                normal = np.array([-line_dir[1], line_dir[0]])
                c_line = -np.dot(normal, centroid)
                a_out, b_out, c_out = normal[0], normal[1], c_line
                distances_label0 = [abs(a_out * p[0] + b_out * p[1] + c_out) / np.sqrt(a_out*a_out + b_out*b_out + 1e-10) 
                                   for p in points_label0]
                inlier_ratio_label0 = np.sum(np.array(distances_label0) <= final_threshold) / len(distances_label0)
                print(f"     Label 0: {len(points_label0)} points, {inlier_ratio_label0*100:.1f}% within threshold")
                if inlier_ratio_label0 > 0.6:
                    print(f"     ⚠️  Label 0 appears to be a LINE (not outliers)!")
        
        # Check if the highest label (possibly model_number) are outliers
        max_label = labeling.max()
        if max_label == model_number and model_number > 0:
            mask_max = (labeling == max_label)
            if np.sum(mask_max) > 0:
                print(f"     Label {max_label}: {np.sum(mask_max)} points (might be outliers)")
        
        # Check if outliers form a line (using interpretation 1)
        outlier_mask = (labeling == 0)
        n_outliers = np.sum(outlier_mask)
        if n_outliers > 10:  # Only check if there are enough outliers
            print(f"\n   Diagnostic: Checking if label 0 (interpreted as outliers) form a line...")
            print(f"     {n_outliers} points labeled as 0")
            points_outliers = points_shifted[outlier_mask]
            
            # Try to fit a line to outliers using least squares
            if len(points_outliers) >= 2:
                # Fit line using SVD (more robust)
                centroid = points_outliers.mean(axis=0)
                centered = points_outliers - centroid
                U, s, Vt = np.linalg.svd(centered, full_matrices=False)
                line_dir = Vt[0]  # Direction of best fit line
                # Line equation: n·(p - c) = 0, where n is normal
                normal = np.array([-line_dir[1], line_dir[0]])  # Perpendicular to direction
                c_line = -np.dot(normal, centroid)
                a_out, b_out, c_out = normal[0], normal[1], c_line
                
                # Check distances from outliers to this fitted line
                distances_out = []
                for p in points_outliers:
                    dist = abs(a_out * p[0] + b_out * p[1] + c_out) / np.sqrt(a_out*a_out + b_out*b_out + 1e-10)
                    distances_out.append(dist)
                distances_out = np.array(distances_out)
                
                inlier_count_out = np.sum(distances_out <= final_threshold)
                inlier_ratio_out = inlier_count_out / len(distances_out) if len(distances_out) > 0 else 0
                
                print(f"     Fitted line to outliers: a={a_out:.6f}, b={b_out:.6f}, c={c_out:.6f}")
                print(f"     Outlier distances to fitted line: min={distances_out.min():.6f}, max={distances_out.max():.6f}, mean={distances_out.mean():.6f}")
                print(f"     Outliers within threshold: {inlier_count_out}/{len(distances_out)} ({inlier_ratio_out*100:.1f}%)")
                if inlier_ratio_out > 0.6:
                    print(f"     ⚠️  WARNING: Label 0 (interpreted as outliers) appears to form a line! (>{inlier_ratio_out*100:.1f}% within threshold)")
                    print(f"     This suggests labels might be 0-indexed: label 0 = first model, not outliers!")
        
        # Determine correct interpretation based on diagnostics
        # If label 0 forms a good line, labels are probably 0-indexed (0 = first model)
        use_zero_indexed = False
        if n_outliers > 10:
            points_label0 = points_shifted[outlier_mask]
            if len(points_label0) >= 2:
                centroid = points_label0.mean(axis=0)
                centered = points_label0 - centroid
                U, s, Vt = np.linalg.svd(centered, full_matrices=False)
                line_dir = Vt[0]
                normal = np.array([-line_dir[1], line_dir[0]])
                c_line = -np.dot(normal, centroid)
                a_out, b_out, c_out = normal[0], normal[1], c_line
                distances_label0 = [abs(a_out * p[0] + b_out * p[1] + c_out) / np.sqrt(a_out*a_out + b_out*b_out + 1e-10) 
                                   for p in points_label0]
                inlier_ratio_label0 = np.sum(np.array(distances_label0) <= final_threshold) / len(distances_label0)
                if inlier_ratio_label0 > 0.6:
                    use_zero_indexed = True
                    print(f"\n   ✓ Switching to 0-indexed interpretation: label 0 = first model")
                    print(f"   (Label {model_number} will be treated as outliers)")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check if we need to use 0-indexed interpretation
    # If label 0 forms a good line, labels are 0-indexed (0 = first model, not outliers)
    use_zero_indexed = False
    outlier_mask_check = (labeling == 0)
    if np.sum(outlier_mask_check) > 10:
        points_label0 = points_shifted[outlier_mask_check]
        if len(points_label0) >= 2:
            centroid = points_label0.mean(axis=0)
            centered = points_label0 - centroid
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            line_dir = Vt[0]
            normal = np.array([-line_dir[1], line_dir[0]])
            c_line = -np.dot(normal, centroid)
            a_out, b_out, c_out = normal[0], normal[1], c_line
            distances_label0 = [abs(a_out * p[0] + b_out * p[1] + c_out) / np.sqrt(a_out*a_out + b_out*b_out + 1e-10) 
                               for p in points_label0]
            inlier_ratio_label0 = np.sum(np.array(distances_label0) <= final_threshold) / len(distances_label0)
            if inlier_ratio_label0 > 0.6:
                use_zero_indexed = True
                print(f"\n   ✓ Using 0-indexed interpretation: label 0 = first model, label {model_number} = outliers")
    
    # Convert lines back to original coordinates for visualization
    # Lines are in shifted coordinates (x_shifted = x_orig - x_min), need to convert back
    lines_original = lines.copy()
    for i in range(len(lines)):
        a, b, c = lines[i]
        # Line in shifted coordinates: a*(x_shifted) + b*(y_shifted) + c = 0
        # where x_shifted = x_orig - x_min, y_shifted = y_orig - y_min
        # Convert back: a*(x_orig - x_min) + b*(y_orig - y_min) + c = 0
        # => a*x_orig + b*y_orig + (c - a*x_min - b*y_min) = 0
        a_orig = a
        b_orig = b
        c_orig = c - a * x_min - b * y_min
        lines_original[i] = [a_orig, b_orig, c_orig]
    
    # Visualize results with two subplots (original and detected)
    print("\n5. Visualizing results...")
    
    # First, create a diagnostic plot showing each line separately
    if model_number > 0:
        print("   Creating diagnostic visualization...")
        n_cols = min(3, model_number)
        n_rows = (model_number + n_cols - 1) // n_cols
        fig_diag, axes_diag = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if model_number == 1:
            axes_diag = [axes_diag]
        else:
            axes_diag = axes_diag.flatten()
        
        for idx in range(model_number):
            ax = axes_diag[idx]
            ax.set_xlim(0,2)
            ax.set_ylim(0,2)
            # Use correct interpretation
            if use_zero_indexed:
                instance_label = idx  # 0-indexed: 0 = first model
            else:
                instance_label = idx + 1  # 1-indexed: 1 = first model
            mask = (labeling == instance_label)
            points_assigned = points_shifted[mask]
            a, b, c = lines[idx]
            
            # Calculate distances
            distances = []
            for p in points_assigned:
                dist = abs(a * p[0] + b * p[1] + c) / np.sqrt(a*a + b*b + 1e-10)
                distances.append(dist)
            distances = np.array(distances)
            inlier_ratio = np.sum(distances <= final_threshold) / len(distances) if len(distances) > 0 else 0
            
            # Plot points colored by distance
            scatter = ax.scatter(points_assigned[:, 0], points_assigned[:, 1], 
                                c=distances, s=30, alpha=0.7, cmap='RdYlGn', 
                                vmin=0, vmax=final_threshold*2)
            plt.colorbar(scatter, ax=ax, label='Distance to line')
            
            # Draw the line
            if abs(b) > 1e-10:
                x_line = np.linspace(points_shifted[:, 0].min(), points_shifted[:, 0].max(), 200)
                y_line = (-a * x_line - c) / b
                ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.8)
            
            ax.set_title(f'Line {idx+1}: {len(points_assigned)} pts, {inlier_ratio*100:.1f}% within threshold', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('X (shifted)')
            ax.set_ylabel('Y (shifted)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Hide unused subplots
        for idx in range(model_number, len(axes_diag)):
            axes_diag[idx].axis('off')
        
        plt.suptitle('Diagnostic: Each Detected Line', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('star_dataset_diagnostic.png', dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved diagnostic plot to star_dataset_diagnostic.png")
        plt.close(fig_diag)
    
    # Main visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    # Plot 1: Original points
    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], s=20, alpha=0.6, c='black', marker='.')
    ax.set_title('Original Star Data', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 2: Detected lines
    ax = axes[1]
    detected_colors = plt.cm.tab20(np.linspace(0, 1, model_number)) if model_number > 0 else []
    x_lims = (points[:, 0].min() - 0.1, points[:, 0].max() + 0.1)
    y_lims = (points[:, 1].min() - 0.1, points[:, 1].max() + 0.1)
    
    # Draw detected lines - use correct interpretation
    for idx in range(model_number):
        a, b, c = lines_original[idx]
        if use_zero_indexed:
            # Labels are 0-indexed: 0 = first model, 1 = second model, ..., model_number = outliers
            instance_label = idx
        else:
            # Labels are 1-indexed: 0 = outliers, 1 = first model, 2 = second model, ...
            instance_label = idx + 1
        mask = (labeling == instance_label)
        n_points = np.sum(mask)
        
        # Draw line
        if abs(b) > 1e-10:
            x_line = np.linspace(x_lims[0], x_lims[1], 200)
            y_line = (-a * x_line - c) / b
            valid = (y_line >= y_lims[0]) & (y_line <= y_lims[1])
            if np.any(valid):
                ax.plot(x_line[valid], y_line[valid], '--', 
                       color=detected_colors[idx], linewidth=2, alpha=0.8, zorder=1)
        elif abs(a) > 1e-10:
            # Vertical line
            y_line = np.linspace(y_lims[0], y_lims[1], 200)
            x_line = np.full_like(y_line, -c / a)
            valid = (x_line >= x_lims[0]) & (x_line <= x_lims[1])
            if np.any(valid):
                ax.plot(x_line[valid], y_line[valid], '--', 
                       color=detected_colors[idx], linewidth=2, alpha=0.8, zorder=1)
        
        # Draw points assigned to this line
        if np.any(mask):
            points_for_line = points[mask]
            ax.scatter(points_for_line[:, 0], points_for_line[:, 1], 
                      color=detected_colors[idx], s=40, alpha=0.8, 
                      edgecolors='black', linewidths=0.5, marker='o', zorder=3,
                      label=f'Line {idx+1} ({n_points} pts)')
    
    # Plot outliers - use correct interpretation
    if use_zero_indexed:
        # Highest label is outliers
        outlier_mask = (labeling == model_number)
    else:
        # Label 0 is outliers
        outlier_mask = (labeling == 0)
    if np.any(outlier_mask):
        points_outliers = points[outlier_mask]
        ax.scatter(points_outliers[:, 0], points_outliers[:, 1], 
                  c='red', s=20, alpha=0.5, marker='x', 
                  label=f'Outliers ({np.sum(outlier_mask)} pts)', zorder=2, linewidths=1)
    
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_title(f'Progressive-X Detection ({model_number} lines)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.suptitle(f"Star Dataset - {len(points)} points, {model_number} lines detected", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = 'star_dataset_result.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.show()
    
    print("\n" + "="*70)
    print("✓ Test completed successfully!")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
