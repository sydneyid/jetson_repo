#!/usr/bin/env python3
"""
Example: Multiple Line Detection using Progressive-X
Converted from Jupyter notebook example_multi_lines.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

# Add parent directory to path to import pyprogressivex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pyprogressivex
    print("✓ Successfully imported pyprogressivex")
except ImportError as e:
    print(f"✗ Failed to import pyprogressivex: {e}")
    print("Trying alternative import paths...")
    # Try importing from build directory
    build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build')
    if os.path.exists(build_path):
        sys.path.insert(0, build_path)
    try:
        import pyprogressivex
        print("✓ Successfully imported pyprogressivex from build directory")
    except ImportError:
        print("✗ Could not import pyprogressivex. Please check installation.")
        sys.exit(1)

import random
from random import randint
from time import time

def random_color(label):
    """Generate a random color for visualization"""
    if label == 0:
        return (255, 0, 0)
    elif label == 1:
        return (0, 255, 0)
    elif label == 2:
        return (0, 0, 255)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_edges_and_line(img, edge_points, line, color):
    """Draw edge points and line on image"""
    # Draw the edge points on the image
    for point in edge_points:
        cv2.circle(img, tuple(point), radius=4, color=color, thickness=-1)

    # Convert the implicit line equation to two endpoints of a line segment
    a, b, c = line
    y1 = 0
    x1 = int(-c/a) if a != 0 else 0
    y2 = img.shape[0] - 1
    x2 = int(-(b*y2 + c)/a) if a != 0 else 0

    # Draw the line on the image
    cv2.line(img, (x1, y1), (x2, y2), color, thickness=2)

def verify_pyprogressivex(img, points, weights, threshold=2.0):
    """Run Progressive-X line detection"""
    if weights is None:
        weights = []

    lines, labeling = pyprogressivex.findLines(
        np.ascontiguousarray(points), 
        np.ascontiguousarray(weights), 
        img.shape[1], img.shape[0], 
        threshold=threshold,
        conf=0.99,
        spatial_coherence_weight=0.0,
        neighborhood_ball_radius=1.0,
        maximum_tanimoto_similarity=1.0,
        max_iters=1000,
        minimum_point_number=50,
        maximum_model_number=-1,
        sampler_id=0,
        scoring_exponent=1.0,
        do_logging=False)    
    return lines, labeling

def main():
    print("="*70)
    print("Progressive-X: Multiple Line Detection Example")
    print("="*70)
    
    # Load the images
    print("\n1. Loading image...")
    img_path = '../graph-cut-ransac/build/data/adam/adam1.png'
    if not os.path.exists(img_path):
        print(f"✗ Image not found: {img_path}")
        print("Please ensure the graph-cut-ransac data directory exists.")
        return
    
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    print(f"   ✓ Loaded image: {img.shape}")
    
    # Convert the image to grayscale
    print("\n2. Processing image...")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale image
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(img_blur, 50, 150)
    print(f"   ✓ Edge detection complete")
    
    # Get the indices of edge points
    print("\n3. Extracting edge points...")
    edge_points_original = np.argwhere(edges == 255)
    edge_points = edge_points_original[:, ::-1]  # Swap columns
    
    print(f"   ✓ Found {len(edge_points)} edge points")
    
    # Run Progressive-X
    print("\n4. Running Progressive-X line detection...")
    print("   (This may take a moment...)")
    
    try:
        t = time()
        lines, labeling = verify_pyprogressivex(img, edge_points, None, threshold=3.0)
        elapsed_time = time() - t
        
        model_number = int(lines.size / 3)
        
        print(f"   ✓ Completed in {elapsed_time:.2f} seconds")
        print(f"   ✓ Found {model_number} line models")
        
        # Visualize results
        print("\n5. Visualizing results...")
        line_img = img.copy()
        
        # Match original notebook exactly
        for idx in range(model_number):    
            mask = np.zeros(len(labeling))  # Original creates this but doesn't use it
            indices = [i for i, e in enumerate(labeling) if e == idx]
            color = random_color(idx)
            draw_edges_and_line(line_img, edge_points[indices], lines[idx], color)
        
        # Display like original notebook (no title, no subplot)
        plt.imshow(line_img)
        plt.figure()
        
        # Save the figure
        output_file = 'example_multi_lines_result.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved result to {output_file}")
        plt.show()
        
        print("\n" + "="*70)
        print("✓ Example completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error during line detection: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
