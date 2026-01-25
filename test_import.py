#!/usr/bin/env python3
"""Simple test to verify pyprogressivex import"""

import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Current directory:", os.getcwd())
print("\nTrying to import pyprogressivex...")

# Try different import methods
try:
    import pyprogressivex
    print("✓ SUCCESS: Imported pyprogressivex directly")
    print("Available functions:", [x for x in dir(pyprogressivex) if not x.startswith('_')])
except ImportError as e:
    print(f"✗ Direct import failed: {e}")
    
    # Try from src
    try:
        sys.path.insert(0, 'src')
        import pyprogressivex
        print("✓ SUCCESS: Imported from src directory")
    except ImportError as e2:
        print(f"✗ Import from src failed: {e2}")
        
        # Try from build
        try:
            sys.path.insert(0, 'build')
            import pyprogressivex
            print("✓ SUCCESS: Imported from build directory")
        except ImportError as e3:
            print(f"✗ All import methods failed")
            print(f"  Direct: {e}")
            print(f"  From src: {e2}")
            print(f"  From build: {e3}")
            sys.exit(1)

print("\n✓ Import successful! Testing findLines function...")
try:
    # Test with minimal data
    import numpy as np
    points = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
    weights = np.array([])
    
    print("Calling findLines with test data...")
    lines, labeling = pyprogressivex.findLines(
        points, weights, 100, 100,
        threshold=1.0,
        conf=0.99,
        spatial_coherence_weight=0.0,
        neighborhood_ball_radius=1.0,
        maximum_tanimoto_similarity=1.0,
        max_iters=100,
        minimum_point_number=2,
        maximum_model_number=-1,
        sampler_id=0,
        scoring_exponent=1.0,
        do_logging=False
    )
    print(f"✓ findLines works! Found {len(lines)} lines")
    print("✓ All tests passed!")
except Exception as e:
    print(f"✗ Error calling findLines: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
