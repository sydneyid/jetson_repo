#pragma once

#include "scoring_function.h"
#include "progx_model.h"

namespace progx
{
	template<class _ModelEstimator>
	class MSACScoringFunctionWithCompoundModel : public gcransac::ScoringFunction<_ModelEstimator>
	{
	protected:
		// Squared truncated threshold
		double squared_truncated_threshold;

		// Number of points
		size_t point_number; 

		// Minimum inliers required (for early rejection heuristic)
		size_t min_inliers_required;

		// The exponent of the score shared with the compound model instance and
		// substracted from the final score.  
		int exponent_of_shared_score;

		// The pointer of the compound model instance
		const std::vector<Model<_ModelEstimator>> *compound_model;

		// The pointer of the preference vector of the compound model instance
		const Eigen::VectorXd *compound_preference_vector;

	public:
		MSACScoringFunctionWithCompoundModel() : 
			exponent_of_shared_score(2.5),
			min_inliers_required(15)  // Default minimum (will be set via initialize)
		{

		}

		~MSACScoringFunctionWithCompoundModel()
		{

		}

		void setExponent(const int exponent_of_shared_score_)
		{
			exponent_of_shared_score = exponent_of_shared_score_;
		}

		void setCompoundModel(
			const std::vector<Model<_ModelEstimator>> *compound_model_, // The pointer of the compound model instance
			const Eigen::VectorXd *compound_preference_vector_) // The pointer of the preference vector of the compound model instance
		{
			compound_preference_vector = compound_preference_vector_;
			compound_model = compound_model_;
		}

		void initialize(
			const double squared_truncated_threshold_, // Squared truncated threshold
			const size_t point_number_) // Number of points
		{
			squared_truncated_threshold = squared_truncated_threshold_;
			point_number = point_number_;
		}

		// Set minimum inliers required (for early rejection heuristic)
		void setMinInliersRequired(const size_t min_inliers_)
		{
			min_inliers_required = min_inliers_;
		}

		// Return the score of a model w.r.t. the data points and the threshold
		inline gcransac::Score getScore(
			const cv::Mat &points_, // The input data points
			gcransac::Model &model_, // The current model parameters
			const _ModelEstimator &estimator_, // The model estimator
			const double threshold_, // The inlier-outlier threshold
			std::vector<size_t> &inliers_, // The selected inliers
			const gcransac::Score &best_score_ = gcransac::Score(), // The score of the current so-far-the-best model
			const bool store_inliers_ = true, // A flag to decide if the inliers should be stored
			const std::vector<const std::vector<size_t>*> *index_sets = nullptr) const // Index sets to be verified
		{
			gcransac::Score score; // The current score
			if (store_inliers_) // If the inlier should be stored, clear the variables
				inliers_.clear();
			double squared_residual, score_value; // The point-to-model residual
			Eigen::MatrixXd preference_vector = Eigen::MatrixXd::Zero(point_number, 1); // Initializing the preference vector

			// SMART HEURISTIC: Early rejection for clearly terrible models
			// Only reject models with near-zero inliers in a quick sample check
			// This avoids expensive full inlier counting for obviously bad models
			// Very conservative: only reject if sample shows < 1% inliers (clearly terrible)
			if (point_number > 1000 && best_score_.inlier_number > min_inliers_required) {
				// Only do early rejection if we already have a good model (not the first few)
				const size_t quick_check_points = 30;  // Check 30 points quickly
				size_t quick_inliers = 0;
				const size_t step = point_number / quick_check_points;
				
				// Quick check: sample 30 evenly spaced points
				for (size_t i = 0; i < point_number; i += step) {
					squared_residual = estimator_.squaredResidual(points_.row(i), model_.descriptor);
					if (squared_residual < squared_truncated_threshold) {
						++quick_inliers;
					}
				}
				
				// Very conservative: Only reject if we see < 1% inliers (clearly terrible model)
				// This means < 1 inlier in 30 point sample - model is almost certainly bad
				if (quick_inliers == 0) {
					// Zero inliers in sample - model is clearly terrible, reject immediately
					return gcransac::Score();
				}
			}

			// Iterate through all points, calculate the squared_residuals and store the points as inliers if needed.
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
			{
				// Calculate the point-to-model residual
				squared_residual = estimator_.squaredResidual(points_.row(point_idx), model_.descriptor);

				// If the residual is smaller than the threshold, store it as an inlier and
				// increase the score.
				if (squared_residual < squared_truncated_threshold)
				{
					if (store_inliers_) // Store the point as an inlier if needed.
						inliers_.emplace_back(point_idx);

					// Increase the inlier number
					++(score.inlier_number);

					// Calculate the score (coming from the truncated quadratic loss) implied by the current point
					score_value = MAX(0.0, 1.0 - squared_residual / squared_truncated_threshold);

					// Increase the score
					score.value += score_value; 

					// The preference value. It is proportional to the likelihood. 
					preference_vector(point_idx) = score_value;
						
				}

				// Interrupt if there is no chance of being better than the best model
				if (point_number - point_idx + score.inlier_number < best_score_.inlier_number)
					return gcransac::Score();
			}

			// Calculating the support shared with the compound model
			if (compound_model->size() > 0)
			{
				double shared_support = 0; // The shared support
				double unexplored_support = 0; // STRUCTURAL: Support in unexplored regions
				double total_model_support = 0; // Total support of the new model

				// Iterate through all points and calculate the shared support
				for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
				{
					double compound_pref = (*compound_preference_vector)(point_idx);
					double model_pref = preference_vector(point_idx);
					shared_support += MIN(compound_pref, model_pref);
					total_model_support += model_pref;
					
					// STRUCTURAL: Boost score for points in unexplored regions (low compound preference)
					// This helps find models in dense but unexplored regions
					if (compound_pref < 0.2 && model_pref > 0.3) {
						unexplored_support += model_pref * 2.0; // Strong boost for unexplored regions
					}
				}

				// STRUCTURAL FIX: Make scoring much more lenient to find more models
				// The issue is that after 2 models, compound_preference_vector covers most points
				// So we need to be very lenient with shared support penalty
				if (exponent_of_shared_score == 0) {
					// No penalty - allows finding many overlapping lines
					// Strong boost for unexplored regions to encourage exploration
					score.value += unexplored_support;
				} else {
					// STRUCTURAL: Drastically reduce penalty to allow finding more models
					// Even with exponent > 0, we want to find many overlapping parallel lines
					// So we use a very small penalty (1% of original) plus boost for unexplored
					double penalty = std::pow(shared_support, exponent_of_shared_score);
					score.value -= penalty * 0.01; // Only 1% of original penalty (was 10%)
					// Strong boost for unexplored regions
					score.value += unexplored_support;
					
					// STRUCTURAL: Additional boost if model has significant unexplored support
					// This helps models that cover new regions even if they overlap with existing models
					if (unexplored_support > total_model_support * 0.3) {
						score.value += total_model_support * 0.2; // Extra boost for models with >30% unexplored
					}
				}
			}

			// Return the final score
			return score;
		}
	};
}