#!/usr/bin/env python3
"""Test Progressive-X on pentagram line detection problem"""

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

def generate_pentagram(points_per_edge=25, radius=1.0, noise_sigma=0.0075, jitter=0.1, 
                       outlier_fraction=0.10, seed=0):
    """
    Generate pentagram with Gaussian noise and outliers.
    
    Args:
        points_per_edge: Number of points per edge
        radius: Radius of pentagram
        noise_sigma: Gaussian noise standard deviation (σn)
        jitter: Jitter for stratified sampling
        outlier_fraction: Fraction of points that are outliers (0.10 = 10%)
        seed: Random seed
    """
    np.random.seed(seed)
    angles = 2 * np.pi * np.arange(5) / 5
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    edges = [(0,2), (2,4), (4,1), (1,3), (3,0)]
    points = []
    labels = []
    
    # Generate points for each edge
    for i, (a, b) in enumerate(edges):
        v0, v1 = vertices[a], vertices[b]
        bins = np.linspace(0.0, 1.0, points_per_edge + 1)
        t = bins[:-1] + np.random.rand(points_per_edge) * (bins[1] - bins[0]) * jitter
        segment = (1 - t)[:, None] * v0 + t[:, None] * v1
        # Add Gaussian noise with σn = 0.0075
        segment += np.random.normal(scale=noise_sigma, size=segment.shape)
        points.append(segment)
        labels += [i] * points_per_edge
    
    # Stack all inlier points
    inlier_points = np.vstack(points)
    inlier_labels = np.array(labels)
    
    # Calculate number of outliers (10% of total points)
    n_inliers = len(inlier_points)
    n_outliers = int(n_inliers * outlier_fraction / (1 - outlier_fraction))
    
    # Generate outliers uniformly in a bounding box around the pentagram
    x_range = (inlier_points[:, 0].min() - 0.5, inlier_points[:, 0].max() + 0.5)
    y_range = (inlier_points[:, 1].min() - 0.5, inlier_points[:, 1].max() + 0.5)
    
    outlier_x = np.random.uniform(x_range[0], x_range[1], n_outliers)
    outlier_y = np.random.uniform(y_range[0], y_range[1], n_outliers)
    outlier_points = np.stack([outlier_x, outlier_y], axis=1)
    outlier_labels = np.zeros(n_outliers, dtype=int)  # Label 0 for outliers
    
    # Combine inliers and outliers
    all_points = np.vstack([inlier_points, outlier_points])
    all_labels = np.hstack([inlier_labels + 1, outlier_labels])  # Shift inlier labels to 1-5, keep outliers as 0
    
    # Shuffle to mix inliers and outliers
    indices = np.random.permutation(len(all_points))
    all_points = all_points[indices]
    all_labels = all_labels[indices]
    
    return all_points, all_labels

print("="*70)
print("Pentagram Line Detection Test - Progressive-X")
print("="*70)

# Generate pentagram with Gaussian noise (σn = 0.0075) and 10% outliers
print("\n1. Generating pentagram data...")
print("   Parameters: σn = 0.0075, 10% outliers")
points, ground_truth_labels = generate_pentagram(
    points_per_edge=25,
    radius=1.0,
    noise_sigma=0.0075,  # Gaussian noise σn = 0.0075
    jitter=0.1,
    outlier_fraction=0.1,  # 10% outliers
    seed=0
)

n_inliers = np.sum(ground_truth_labels > 0)
n_outliers = np.sum(ground_truth_labels == 0)
print(f"   ✓ Generated {len(points)} total points")
print(f"     - Inliers: {n_inliers} points (5 lines)")
print(f"     - Outliers: {n_outliers} points ({100*n_outliers/len(points):.1f}%)")

# Import and setup
print("\n2. Setting up Progressive-X...")
try:
    from test_2 import MultiX, LineOrSegmentEstimator, UniformSampler
    
    estimator = LineOrSegmentEstimator(use_segments=False)
    sampler = UniformSampler()
    
    # Simple neighbor radius computation
    distances = []
    for i in range(min(50, len(points))):
        for j in range(i+1, min(50, len(points))):
            distances.append(np.linalg.norm(points[i] - points[j]))
    neighbor_radius = np.percentile(distances, 5) if len(distances) > 0 else 0.1
    
    multi_x = MultiX(points, k=8, neighbor_radius=neighbor_radius)
    print(f"   ✓ Neighbor radius: {neighbor_radius:.4f}")
    
    # Run Progressive-X
    print("\n3. Running Progressive-X...")
    start_time = time.time()
    
    try:
        multi_x.run(
            estimator, sampler,
            n_hyp=500,
            lam=0.01,  # Spatial coherence weight
            iters=15,
            inlier_threshold=0.08,  # Inlier threshold
            label_cost=None,  # Automatic
            hmax=None  # No limit on number of models
        )
    except KeyboardInterrupt:
        print("\n   ⚠️  Interrupted by user")
        raise
    except Exception as e:
        print(f"\n   ✗ Error during run: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    total_time = time.time() - start_time
    print(f"   ✓ Completed in {total_time:.2f}s")
    print(f"   ✓ Found {len(multi_x.instances)} instances (before filtering)")
    
    # Post-processing: Filter instances with low support
    print("\n4. Post-processing: Filtering low-support instances...")
    if hasattr(multi_x, 'labels') and len(multi_x.labels) > 0:
        min_support = max(2, n_inliers // 20)  # Lenient: allow 2+ points
        print(f"   Minimum support: {min_support} points per instance")
        
        filtered_instances = []
        filtered_labels = multi_x.labels.copy()
        
        for idx, inst in enumerate(multi_x.instances):
            instance_label = idx + 1
            support = np.sum(multi_x.labels == instance_label)
            
            desc = inst.get('descriptor', None)
            if desc is not None:
                m, b = desc[0], desc[1]
                # Filter out unreasonable lines (extreme slopes/intercepts)
                is_reasonable = abs(m) < 1000 and abs(b) < 1000
                
                if support >= min_support and is_reasonable:
                    filtered_instances.append(inst)
                else:
                    # Reassign to outliers
                    filtered_labels[multi_x.labels == instance_label] = 0
                    reason = []
                    if support < min_support:
                        reason.append(f"only {support} points")
                    if not is_reasonable:
                        reason.append(f"unreasonable line (m={m:.1f}, b={b:.1f})")
                    print(f"   - Removed instance {idx+1}: {', '.join(reason)}")
            else:
                # No descriptor - remove
                filtered_labels[multi_x.labels == instance_label] = 0
                print(f"   - Removed instance {idx+1} (no descriptor)")
        
        multi_x.instances = filtered_instances
        multi_x.labels = filtered_labels
        print(f"   ✓ After filtering: {len(multi_x.instances)} instances")
    
    # Metrics
    if hasattr(multi_x, 'labels') and len(multi_x.labels) > 0:
        print("\n5. Classification metrics...")
        metrics = multi_x.compute_classification_accuracy(ground_truth_labels)
        print(f"   Accuracy: {metrics.get('overall_accuracy', 0):.3f}")
        print(f"   Precision: {metrics.get('precision', 0):.3f}")
        print(f"   Recall: {metrics.get('recall', 0):.3f}")
        print(f"   F1 Score: {metrics.get('f1_score', 0):.3f}")
    
    # Plot
    print("\n6. Plotting results...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    # Plot 1: Raw points
    ax = axes[0]
    inlier_mask = ground_truth_labels > 0
    outlier_mask = ground_truth_labels == 0
    ax.scatter(points[inlier_mask, 0], points[inlier_mask, 1], 
              c='black', s=15, alpha=0.5, label='Inliers')
    ax.scatter(points[outlier_mask, 0], points[outlier_mask, 1], 
              c='red', s=15, alpha=0.5, marker='x', label='Outliers')
    ax.set_title(f'Raw Points ({len(points)} total, {np.sum(outlier_mask)} outliers)', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 2: Ground truth
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for i in range(1, 6):  # Labels are 1-5 for lines, 0 for outliers
        mask = ground_truth_labels == i
        if np.any(mask):
            line_idx = i - 1
            ax.scatter(points[mask, 0], points[mask, 1], c=[colors[line_idx]], 
                      s=15, alpha=0.7, label=f'Line {i} ({np.sum(mask)} pts)')
    outlier_mask = ground_truth_labels == 0
    if np.any(outlier_mask):
        ax.scatter(points[outlier_mask, 0], points[outlier_mask, 1], 
                  c='red', s=15, alpha=0.5, marker='x', 
                  label=f'Outliers ({np.sum(outlier_mask)} pts)')
    ax.set_title('Ground Truth (5 lines + outliers)', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 3: Progressive-X Detection
    ax = axes[2]
    x_lims = axes[0].get_xlim()
    y_lims = axes[0].get_ylim()
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    
    has_instances = len(multi_x.instances) > 0
    has_labels = hasattr(multi_x, 'labels') and multi_x.labels is not None and len(multi_x.labels) > 0
    
    if has_instances and has_labels:
        detected_colors = plt.cm.tab10(np.linspace(0, 1, len(multi_x.instances)))
        x_min, x_max = x_lims
        y_min, y_max = y_lims
        
        # Draw detected lines first
        for idx, inst in enumerate(multi_x.instances):
            desc = inst.get('descriptor', None)
            if desc is not None:
                m, b = desc[0], desc[1]
                instance_label = idx + 1
                mask = (multi_x.labels == instance_label)
                n_points = np.sum(mask)
                
                # Generate line across plot
                x_line = np.linspace(x_min, x_max, 200)
                y_line = m * x_line + b
                
                ax.plot(x_line, y_line, '--', 
                       color=detected_colors[idx], linewidth=3, alpha=0.9,
                       label=f'Line {idx+1}: y={m:.3f}x+{b:.3f} ({n_points} pts)',
                       zorder=1)
        
        # Plot points assigned to each instance
        for idx, inst in enumerate(multi_x.instances):
            instance_label = idx + 1
            mask = (multi_x.labels == instance_label)
            
            if np.any(mask):
                ax.scatter(points[mask, 0], points[mask, 1], 
                          color=detected_colors[idx], s=60, alpha=0.95, 
                          edgecolors='black', linewidths=1.5,
                          marker='o', zorder=3)
        
        # Plot outliers
        outlier_mask = (multi_x.labels == 0)
        if np.any(outlier_mask):
            ax.scatter(points[outlier_mask, 0], points[outlier_mask, 1], 
                      c='black', s=15, alpha=0.6, marker='x', 
                      label=f'Outliers ({np.sum(outlier_mask)} pts)',
                      zorder=2, linewidths=1.5)
    else:
        ax.scatter(points[:, 0], points[:, 1], c='gray', s=15, alpha=0.6, label='All points', zorder=3)
        if not has_instances:
            ax.text(0.5, 0.5, f'No instances detected\n(Expected: 5 lines)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        elif not has_labels:
            ax.text(0.5, 0.5, f'No labels assigned\n({len(multi_x.instances)} instances found)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    ax.set_title(f'Progressive-X Detection ({len(multi_x.instances)} instances)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if has_instances and has_labels:
        handles, labels_legend = ax.get_legend_handles_labels()
        seen = set()
        filtered_handles = []
        filtered_labels_legend = []
        for h, l in zip(handles, labels_legend):
            if l not in seen:
                filtered_handles.append(h)
                filtered_labels_legend.append(l)
                seen.add(l)
        if len(filtered_handles) > 0:
            ax.legend(filtered_handles, filtered_labels_legend, loc='best', fontsize=8, ncol=1, framealpha=0.9)
    
    plt.suptitle(f"Pentagram Line Detection - Expected: 5 lines, Found: {len(multi_x.instances)} instances", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = "pentagram_progressivex_results.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Expected: 5 lines + {n_outliers} outliers")
    print(f"Detected: {len(multi_x.instances)} instances")
    if hasattr(multi_x, 'labels') and len(multi_x.labels) > 0:
        metrics = multi_x.compute_classification_accuracy(ground_truth_labels)
        print(f"Accuracy: {metrics.get('overall_accuracy', 0):.3f}")
    
    if len(multi_x.instances) == 5:
        print("✓ SUCCESS: Correctly identified 5 lines!")
    elif len(multi_x.instances) >= 4:
        print("⚠️  PARTIAL: Found most lines")
    else:
        print("✗ Found fewer than 4 lines")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Test completed!")
