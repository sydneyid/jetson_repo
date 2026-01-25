"""
Wrapper for Progressive-X API to match the expected interface in test scripts.
This adapts the function-based pyprogressivex API to a class-based interface.
"""

import numpy as np
import pyprogressivex


class LineOrSegmentEstimator:
    """Dummy estimator class - not used directly with progressive-x API"""
    def __init__(self, use_segments=False):
        self.use_segments = use_segments


class UniformSampler:
    """Dummy sampler class - sampler is selected via sampler_id parameter"""
    pass


class MultiX:
    """
    Wrapper class for Progressive-X line detection.
    Adapts the function-based pyprogressivex.findLines() to a class-based interface.
    """
    
    def __init__(self, points, k=8, neighbor_radius=None):
        """
        Initialize Multi-X with points.
        
        Args:
            points: numpy array of shape [n, 2] with 2D points
            k: number of neighbors (not directly used, kept for compatibility)
            neighbor_radius: radius for neighborhood graph (auto-computed if None)
        """
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.shape[1] != 2:
            raise ValueError(f"Points must be [n, 2], got shape {self.points.shape}")
        
        self.k = k
        self.neighbor_radius = neighbor_radius
        
        # Compute image bounds from points (needed for progressive-x API)
        self.x_min, self.x_max = self.points[:, 0].min(), self.points[:, 0].max()
        self.y_min, self.y_max = self.points[:, 1].min(), self.points[:, 1].max()
        self.width = int(np.ceil(self.x_max - self.x_min)) + 1
        self.height = int(np.ceil(self.y_max - self.y_min)) + 1
        
        # Auto-compute neighbor_radius if not provided
        if self.neighbor_radius is None:
            # Compute 5th percentile of pairwise distances
            n_sample = min(100, len(self.points))
            indices = np.random.choice(len(self.points), n_sample, replace=False)
            sample_points = self.points[indices]
            distances = []
            for i in range(n_sample):
                for j in range(i+1, n_sample):
                    distances.append(np.linalg.norm(sample_points[i] - sample_points[j]))
            if len(distances) > 0:
                self.neighbor_radius = np.percentile(distances, 5)
            else:
                self.neighbor_radius = 0.1
        
        # Results will be stored here after run()
        self.instances = []
        self.labels = None
    
    def run(self, estimator, sampler, n_hyp=500, lam=0.01, iters=15, 
            inlier_threshold=0.08, label_cost=None, hmax=None):
        """
        Run Progressive-X line detection.
        
        Args:
            estimator: LineOrSegmentEstimator (not used directly)
            sampler: UniformSampler (not used directly, sampler_id=0 for uniform)
            n_hyp: maximum number of hypotheses (maps to max_iters)
            lam: spatial coherence weight (lambda)
            iters: number of iterations (maps to max_iters)
            inlier_threshold: threshold for inlier detection
            label_cost: not used (progressive-x handles this automatically)
            hmax: maximum number of models (None = unlimited)
        """
        # Convert points to image coordinates (shift to positive quadrant)
        # Progressive-X expects points in image coordinate space
        points_shifted = self.points.copy()
        points_shifted[:, 0] -= self.x_min
        points_shifted[:, 1] -= self.y_min
        
        # Map parameters to progressive-x API
        # n_hyp and iters both contribute to max_iters
        max_iters = max(n_hyp, iters * 10)  # Progressive-X uses iterations differently
        
        # Call progressive-x findLines
        # Returns: (lines_array, labeling_array)
        # lines_array: [num_models, 3] where each line is [a, b, c] in ax + by + c = 0
        # labeling_array: [n] where 0 = outlier, 1,2,... = model indices
        lines_array, labeling = pyprogressivex.findLines(
            points_shifted,
            np.array([]),  # weights (empty)
            self.width,
            self.height,
            threshold=inlier_threshold,
            conf=0.99,  # confidence
            spatial_coherence_weight=lam,
            neighborhood_ball_radius=self.neighbor_radius,
            maximum_tanimoto_similarity=1.0,  # Allow all models
            max_iters=max_iters,
            minimum_point_number=2,  # Minimum points per model
            maximum_model_number=-1 if hmax is None else hmax,
            sampler_id=0,  # 0 = uniform sampler
            scoring_exponent=1.0,
            do_logging=False
        )
        
        # Convert lines from ax + by + c = 0 to y = mx + b format
        # For line ax + by + c = 0:
        #   If b != 0: y = (-a/b)x + (-c/b)  => m = -a/b, b = -c/b
        #   If b == 0: vertical line x = -c/a (handle separately)
        self.instances = []
        self.labels = labeling.copy()
        
        for i in range(len(lines_array)):
            a, b, c = lines_array[i]
            
            # Convert from shifted coordinates back to original coordinates
            # The line equation is in shifted space: a*(x-x_min) + b*(y-y_min) + c = 0
            # Expand: a*x - a*x_min + b*y - b*y_min + c = 0
            # => a*x + b*y + (c - a*x_min - b*y_min) = 0
            c_original = c - a * self.x_min - b * self.y_min
            
            if abs(b) > 1e-10:  # Non-vertical line
                m = -a / b
                b_intercept = -c_original / b
                descriptor = np.array([m, b_intercept])
            else:  # Vertical line (x = constant)
                # Represent as very large slope
                m = 1e10 if a > 0 else -1e10
                b_intercept = -c_original / a if abs(a) > 1e-10 else 0
                descriptor = np.array([m, b_intercept])
            
            self.instances.append({
                'descriptor': descriptor,
                'line_params': np.array([a, b, c_original])  # Store original params too
            })
        
        # Progressive-X uses 0-indexed labels where 0 = outlier
        # The script expects 1-indexed labels where 0 = outlier, 1,2,... = models
        # So we need to shift: 0 stays 0, 1->1, 2->2, etc.
        # Actually, progressive-x already uses 0 for outliers and 1,2,... for models
        # So the labeling should already be correct!
    
    def median_shift(self, instances, bandwidth=None, k_neighbors=5):
        """
        Simple fallback: return all instances as separate modes.
        This is a placeholder - the actual implementation would do mode-seeking.
        """
        return instances, list(range(len(instances)))
    
    def compute_classification_accuracy(self, ground_truth_labels):
        """
        Compute classification accuracy metrics.
        
        Args:
            ground_truth_labels: array of ground truth labels (0=outlier, 1-5=lines)
        
        Returns:
            dict with accuracy, precision, recall, f1_score
        """
        if self.labels is None or len(self.labels) == 0:
            return {
                'overall_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        # Ground truth: 0 = outlier, 1-5 = line labels
        # Predicted: 0 = outlier, 1,2,... = detected models
        
        # Convert to binary: 0 = outlier, 1 = inlier
        gt_binary = (ground_truth_labels > 0).astype(int)
        pred_binary = (self.labels > 0).astype(int)
        
        # Overall accuracy
        overall_accuracy = np.mean(gt_binary == pred_binary)
        
        # Precision: of predicted inliers, how many are actually inliers?
        if np.sum(pred_binary) > 0:
            precision = np.sum((gt_binary == 1) & (pred_binary == 1)) / np.sum(pred_binary)
        else:
            precision = 0.0
        
        # Recall: of actual inliers, how many did we detect?
        if np.sum(gt_binary) > 0:
            recall = np.sum((gt_binary == 1) & (pred_binary == 1)) / np.sum(gt_binary)
        else:
            recall = 0.0
        
        # F1 score
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        return {
            'overall_accuracy': float(overall_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score)
        }
