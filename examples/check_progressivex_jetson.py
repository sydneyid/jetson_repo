#!/usr/bin/env python3
"""
Run this on the Jetson (from the examples/ dir) to verify Progressive-X is actually
running. Does NOT redirect stderr, so you see any C++/GLOG messages.

  cd examples && python3 check_progressivex_jetson.py

If this prints "Progressive-X is working" and finds 1 line, the library is running.
If import fails or it finds 0 lines on synthetic data, the build or runtime is broken.
"""

import sys
import os

# Allow importing pyprogressivex from repo root (no stderr redirect here)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("1. Importing pyprogressivex (stderr visible for any C++ warnings)...")
    try:
        import pyprogressivex
        print("   OK: import succeeded")
    except Exception as e:
        print("   FAIL: import error:", e)
        import traceback
        traceback.print_exc()
        return 1

    if not hasattr(pyprogressivex, 'findLines3DDual'):
        print("   FAIL: pyprogressivex has no findLines3DDual")
        return 1
    print("   OK: findLines3DDual exists")

    print("\n2. Building synthetic 3D line (200 points on a line + a few outliers)...")
    import numpy as np
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    line_pts = np.column_stack([t, 0.5 * t, 2 * t])  # clear 3D line
    noise = 0.02 * np.random.randn(*line_pts.shape)
    points = np.ascontiguousarray(line_pts + noise, dtype=np.float64)
    # optional outliers
    points[0] += np.array([1.0, 1.0, 1.0])
    points[1] += np.array([-0.5, 0.5, 0.5])
    print("   Points shape:", points.shape)

    print("\n3. Calling findLines3DDual (do_logging=True so you see C++ output)...")
    try:
        lines, labeling, line_types = pyprogressivex.findLines3DDual(
            points,
            np.ascontiguousarray([], dtype=np.float64),
            threshold_dense=0.1,
            threshold_sparse=0.3,
            conf=0.95,
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=0.5,
            maximum_tanimoto_similarity=0.35,
            max_iters=2000,
            minimum_point_number_dense=5,
            minimum_point_number_sparse=3,
            maximum_model_number=100,
            sampler_id=0,  # uniform sampler – no FLANN/geometry needed
            scoring_exponent=0.0,
            do_logging=True,
        )
        num = lines.shape[0] if lines.size > 0 else 0
        print("\n   Returned: num_lines={}, labeling len={}".format(num, len(labeling)))
    except Exception as e:
        print("   FAIL: findLines3DDual raised:", e)
        import traceback
        traceback.print_exc()
        return 1

    if num >= 1:
        print("\n*** Progressive-X is working. (Found {} line(s) on synthetic data.) ***".format(num))
        return 0
    else:
        print("\n*** Progressive-X ran but found 0 lines on easy synthetic data. ***")
        print("   Check: OpenCV/Eigen linkage, threshold/params, or try sampler_id=0.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
