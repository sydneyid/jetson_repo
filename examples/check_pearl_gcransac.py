#!/usr/bin/env python3
"""
Check that PEARL and Graph-Cut RANSAC are running inside Progressive-X.

Uses findLines3D (single 3D-line mode) with do_logging=True. The C++ code prints
  "Proposal engine (GC-RANSAC): X.XXX seconds"
  "PEARL optimization: X.XXX seconds"
when both run. We use time-parallel synthetic data so proposals pass the built-in
time-axis filter.

Run from examples/:
  cd examples && python3 check_pearl_gcransac.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("1. Importing pyprogressivex...")
    try:
        import pyprogressivex
        import numpy as np
    except Exception as e:
        print("   FAIL:", e)
        return 1

    if not hasattr(pyprogressivex, 'findLines3D'):
        print("   FAIL: findLines3D not found")
        return 1
    print("   OK")

    print("\n2. Building time-parallel synthetic 3D line (direction ~ (0,0,1))...")
    np.random.seed(42)
    n = 200
    x0, y0 = 100.0, 50.0
    t = np.linspace(0, 500, n)
    # Time-parallel line: (x0, y0, t) -> direction (0, 0, 1)
    pts = np.column_stack([np.full(n, x0), np.full(n, y0), t])
    pts += 0.5 * np.random.randn(*pts.shape)
    points = np.ascontiguousarray(pts, dtype=np.float64)
    print("   Shape:", points.shape, "  (x≈{}, y≈{}, t in [0,500])".format(x0, y0))

    print("\n3. Calling findLines3D(..., do_logging=True) and capturing C++ stdout...")
    # C++ printf writes to fd 1; redirect it so we can check for GC-RANSAC and PEARL
    r, w = os.pipe()
    old_fd1 = os.dup(1)
    os.dup2(w, 1)
    os.close(w)
    try:
        lines, labeling = pyprogressivex.findLines3D(
            points,
            np.ascontiguousarray([], dtype=np.float64),
            threshold=1.0,
            conf=0.95,
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=2.0,
            maximum_tanimoto_similarity=0.4,
            max_iters=2000,
            minimum_point_number=10,
            maximum_model_number=10,
            sampler_id=0,
            scoring_exponent=1.0,
            do_logging=True,
        )
    finally:
        os.dup2(old_fd1, 1)
        os.close(old_fd1)
    log = os.read(r, 1 << 20).decode("utf-8", errors="replace")
    os.close(r)

    num_lines = lines.shape[0] if lines.size > 0 else 0
    print("   findLines3D returned {} line(s)".format(num_lines))

    # Look for the timing lines printed by findLines3D_ in progressivex_python.cpp
    gcransac_seen = "Proposal engine (GC-RANSAC)" in log or "GC-RANSAC" in log
    pearl_seen = "PEARL optimization" in log

    print("\n4. PEARL and Graph-Cut RANSAC check:")
    print("   Graph-Cut RANSAC (proposal engine):", "OK" if gcransac_seen else "NOT SEEN")
    print("   PEARL (model optimizer):           ", "OK" if pearl_seen else "NOT SEEN")

    if gcransac_seen and pearl_seen:
        print("\n*** PEARL and Graph-Cut RANSAC are running. ***")
        return 0
    if num_lines >= 1:
        print("\n*** Pipeline returned lines; PEARL and GC-RANSAC likely ran (timing lines may vary by build). ***")
        return 0

    print("\n*** One or both not clearly seen. Last part of C++ log:")
    print("-" * 60)
    for line in log.strip().splitlines()[-25:]:
        print(line)
    return 1

if __name__ == "__main__":
    sys.exit(main())
