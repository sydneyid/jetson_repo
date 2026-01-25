#pragma once

#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include "PEARL.h"
#include "GCRANSAC.h"
#include "types.h"
#include "scoring_function_with_compound_model.h"
#include "progress_visualizer.h"

#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/progressive_napsac_sampler.h"
#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "estimators/essential_estimator.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"

#include "progx_model.h"

#include <glog/logging.h>

namespace progx
{
	struct MultiModelSettings
	{
		// The settings of the proposal engine
		gcransac::utils::Settings proposal_engine_settings;
		std::vector<double> point_weights;

		size_t minimum_number_of_inliers,
			max_proposal_number_without_change,
			cell_number_in_neighborhood_graph,
			maximum_model_number;

		double maximum_tanimoto_similarity,
			confidence, // Required confidence in the result
			one_minus_confidence, // 1 - confidence
			inlier_outlier_threshold, // The inlier-outlier threshold
			spatial_coherence_weight; // The weight of the spatial coherence term

		void setConfidence(const double &confidence_)
		{
			confidence = confidence_;
			one_minus_confidence = 1.0 - confidence;
		}

		MultiModelSettings() :
			maximum_tanimoto_similarity(0.5),
			minimum_number_of_inliers(20),
			cell_number_in_neighborhood_graph(8),
			max_proposal_number_without_change(10),
			spatial_coherence_weight(0.14),
			inlier_outlier_threshold(2.0),
			confidence(0.95),
			one_minus_confidence(0.05),
			maximum_model_number(std::numeric_limits<size_t>::max())
		{
			proposal_engine_settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph;
			proposal_engine_settings.max_iteration_number = 5000;
			proposal_engine_settings.max_local_optimization_number = 50;
			proposal_engine_settings.threshold = inlier_outlier_threshold;
			proposal_engine_settings.confidence = confidence;
			proposal_engine_settings.spatial_coherence_weight = 0.975;
		}
	};

	struct IterationStatistics
	{
		double time_of_proposal_engine,
			time_of_model_validation,
			time_of_optimization,
			time_of_compound_model_update;
		size_t number_of_instances;
	};

	struct MultiModelStatistics
	{
		double processing_time,
			total_time_of_proposal_engine,
			total_time_of_model_validation,
			total_time_of_optimization,
			total_time_of_compound_model_calculation;
		std::vector<std::vector<size_t>> inliers_of_each_model;
		std::vector<size_t> labeling;
		std::vector<IterationStatistics> iteration_statistics;
		size_t total_proposals_tested = 0;  // Total candidate proposals tested
		size_t total_proposals_accepted = 0;  // Total proposals accepted

		void addIterationStatistics(IterationStatistics iteration_statistics_)
		{
			iteration_statistics.emplace_back(iteration_statistics_);

			total_time_of_proposal_engine += iteration_statistics_.time_of_proposal_engine;
			total_time_of_model_validation += iteration_statistics_.time_of_model_validation;
			total_time_of_optimization += iteration_statistics_.time_of_optimization;
			total_time_of_compound_model_calculation += iteration_statistics_.time_of_compound_model_update;
		}
	};

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
		class ProgressiveX
	{
	protected:
		// The proposal engine (i.e. Graph-Cut RANSAC) estimating a putative model in the beginning of each iteration
		std::unique_ptr<gcransac::GCRANSAC<
			// The model estimator used for estimating the instance parameters from a set of points
			_ModelEstimator,
			// The type of the used neighborhood graph
			_NeighborhoodGraph,
			// The scoring class which consideres the compound instance when calculating the score of a model instance
			MSACScoringFunctionWithCompoundModel<_ModelEstimator>>> 
			proposal_engine;

		// The model optimizer optimizing the compound model parameters in each iteration
		std::unique_ptr<pearl::PEARL<
			// The type of the used neighborhood graph which is needed for determining the spatial coherence cost inside PEARL
			_NeighborhoodGraph,
			// The model estimator used for estimating the instance parameters from a set of points
			_ModelEstimator>> model_optimizer;

		// The model estimator which estimates the model parameters from a set of points
		_ModelEstimator model_estimator;

		// The statistics of Progressive-X containing everything which the user might be curious about,
		// e.g., processing time, results, etc.
		MultiModelStatistics statistics;

		// The set of models (i.e., the compound instance) maintained throughout the multi-model fitting. 
		std::vector<progx::Model<_ModelEstimator>> models;

		// The preference vector of the compound model instance
		Eigen::VectorXd compound_preference_vector;

		// The truncated squared inlier-outlier threshold
		double truncated_squared_threshold,
			scoring_exponent; 

		size_t point_number; // The number of points

		// The visualizer which demonstrates the procedure by showing the labeling in each intermediate step
		ProgressVisualizer * const visualizer;

		// The settings of the algorithm
		MultiModelSettings settings;

		// Flag determining if logging is required
		bool do_logging;

		// Setting all the initial parameters
		void initialize(const cv::Mat &data_);

		// Check if the putative model instance should be included in the optimization procedure
		inline bool isPutativeModelValid(
			const cv::Mat &data_, // All data points
			progx::Model<_ModelEstimator> &model_, // The model instance to check
			const gcransac::utils::RANSACStatistics &statistics_); // The RANSAC statistics of the putative model

		// Update the compound model's preference vector
		void updateCompoundModel(const cv::Mat &data_);
		
		// Predicts the number of inliers in the data which are unseen yet, 
		// i.e. not covered by the compound instance.
		size_t getPredictedUnseenInliers(
			const double confidence_, // The RANSAC confidence
			const size_t sample_size_, // The size of a minimal sample
			const size_t iteration_number_, // The current number of RANSAC iterations
			const size_t inlier_number_of_compound_model_); // The number of inliers of the compound model instance

	public:

		ProgressiveX(ProgressVisualizer * const visualizer_ = nullptr) :
			visualizer(visualizer_),
			do_logging(false),
			scoring_exponent(2)
		{
		}

		void log(bool log_)
		{
			do_logging = log_;
		}

		// The function applying Progressive-X to a set of data points
		void run(const cv::Mat &data_, // All data points
			const _NeighborhoodGraph &neighborhood_graph_, // The neighborhood graph
			_MainSampler &main_sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
			_LocalOptimizerSampler &local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
		
		void setScoringExponent(const double &scoring_exponent_)
		{
			scoring_exponent = scoring_exponent_;
		}

		// Returns a constant reference to the settings of the multi-model fitting
		const MultiModelSettings &getSettings() const
		{
			return settings;
		}

		// Returns a reference to the settings of the multi-model fitting
		MultiModelSettings &getMutableSettings()
		{
			return settings;
		}


		// Returns a constant reference to the statistics of the multi-model fitting
		const MultiModelStatistics &getStatistics() const
		{
			return statistics;
		}

		// Returns a reference to the statistics of the multi-model fitting
		MultiModelStatistics &getMutableStatistics()
		{
			return statistics;
		}

		// Returns a constant reference to the estimated model instances
		const std::vector<Model<_ModelEstimator>> &getModels() const
		{
			return models;
		}

		// Returns a reference to the statistics of the multi-model fitting
		std::vector<Model<_ModelEstimator>> &getMutableModels()
		{
			return models;
		}

		// Returns the number of model instances estimated
		size_t getModelNumber() const
		{
			return models.size();
		}
	};

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
	void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::run(
		const cv::Mat &data_, // All data points
		const _NeighborhoodGraph &neighborhood_graph_, // The neighborhood graph
		_MainSampler &main_sampler_, // The sampler used in the main RANSAC loop of GC-RANSAC
		_LocalOptimizerSampler &local_optimization_sampler_) // The sampler used in the local optimization of GC-RANSAC
	{
		if (do_logging)
			std::cout << "Progressive-X is started...\n";

		// Initializing the procedure
		initialize(data_);

		size_t number_of_ransac_iterations = 0, // The total number of RANSAC iterations
			unaccepted_putative_instances = 0, // The number of consecutive putative instances not accepted to be optimized
			unseen_inliers = point_number; // Predicted number of unseen inliers in the data		
		std::chrono::time_point<std::chrono::system_clock> start, end, // Variables for time measurement
			main_start = std::chrono::system_clock::now(), main_end;
		std::chrono::duration<double> elapsed_seconds; // The elapsed time in seconds

		if (do_logging)
			std::cout << "The main iteration is started...\n";
		// OPTIMIZED: Increased to 2000 iterations to find more lines
		// Early termination will still stop when we have enough models, but allows more exploration
		statistics.total_proposals_tested = 0;
		statistics.total_proposals_accepted = 0;
		size_t consecutive_empty_proposals = 0;  // Track empty proposals
		for (size_t current_iteration = 0; current_iteration < 2000; ++current_iteration)
		{
			if (do_logging)
			{
				std::cout << "-------------------------------------------\n";
				std::cout << "Iteration " << current_iteration + 1 << ".\n";
			}

			// Statistics of the current iteration
			IterationStatistics iteration_statistics;

			/***********************************
			*** Model instance proposal step ***
			***********************************/	
			// The putative model proposed by the proposal engine
			progx::Model<_ModelEstimator> putative_model;

			// Reset the samplers
			main_sampler_.reset();
			local_optimization_sampler_.reset();

			// Applying the proposal engine to get a new putative model
			proposal_engine->run(data_, // The data points
				model_estimator, // The model estimator to be used
				&main_sampler_, // The sampler used for the main RANSAC loop
				&local_optimization_sampler_, // The sampler used for the local optimization
				&neighborhood_graph_, // The neighborhood graph
				putative_model);
			
			++statistics.total_proposals_tested; // Count all proposals tested
			
			if (putative_model.descriptor.rows() == 0 ||
				putative_model.descriptor.cols() == 0)
			{
				++consecutive_empty_proposals;
				// STRUCTURAL: If proposal engine returns empty model, it might be because
				// all remaining points are already covered. Reset compound_preference_vector
				// periodically to allow finding more models in overlapping regions.
				// More aggressive: decay more frequently to find more models
				if (models.size() >= 1 && (unaccepted_putative_instances > 10 || consecutive_empty_proposals > 5)) {
					// After fewer failed attempts, decay compound_preference_vector to allow
					// finding models in regions that were previously covered
					if (do_logging)
						std::cout << "Decaying compound preference vector to allow finding more models.\n";
					compound_preference_vector *= 0.3; // Decay by 70% (more aggressive) to allow re-exploration
					consecutive_empty_proposals = 0;  // Reset counter
					// Update scoring function with new preference vector
					MSACScoringFunctionWithCompoundModel<_ModelEstimator> &scoring =
						proposal_engine->getMutableScoringFunction();
					scoring.setCompoundModel(&models, &compound_preference_vector);
				}
				continue;
			}
			consecutive_empty_proposals = 0;  // Reset on successful proposal

			// Set a reference to the model estimator in the putative model instance
			putative_model.setEstimator(&model_estimator);
						
			// Get the RANSAC statistics to know the inliers of the proposal
			const gcransac::utils::RANSACStatistics &proposal_engine_statistics = 
				proposal_engine->getRansacStatistics();

			// Update the current iteration's statistics
			iteration_statistics.time_of_proposal_engine = 
				proposal_engine_statistics.processing_time;

			// Increase the total number of RANSAC iterations
			number_of_ransac_iterations += 
				proposal_engine_statistics.iteration_number;

			if (do_logging)
				std::cout << "A model proposed with " <<
					proposal_engine_statistics.inliers.size() << " inliers\nin " <<
					iteration_statistics.time_of_proposal_engine << " seconds (" << 
					proposal_engine_statistics.iteration_number << " iterations).\n";

			/*************************************
			*** Model instance validation step ***
			*************************************/
			if (do_logging)
				std::cout << "Check if the model should be added to the compound instance.\n";

			// The starting time of the model validation
			start = std::chrono::system_clock::now();
			if (!isPutativeModelValid(data_,
				putative_model,
				proposal_engine_statistics))
			{
				++unaccepted_putative_instances;
				// Report rejection only if logging is enabled
				if (do_logging) {
					printf("Rejected proposal %zu: %zu inliers, %zu consecutive rejections (max: %zu)\n",
						statistics.total_proposals_tested, proposal_engine_statistics.inliers.size(),
						unaccepted_putative_instances, settings.max_proposal_number_without_change);
					fflush(stdout);
				}
				// OPTIMIZED: Stop earlier if we have enough models (24+) and many rejections
				// If we have 24+ models, stop after fewer rejections since we've found enough
				if (models.size() >= 24 && unaccepted_putative_instances >= (settings.max_proposal_number_without_change / 2)) {
					if (do_logging)
						printf("Stopping: %zu models found (sufficient), %zu consecutive rejections\n", models.size(), unaccepted_putative_instances);
					break;
				}
				// If we have fewer than 24 models, reset rejection counter and keep going
				if (unaccepted_putative_instances >= settings.max_proposal_number_without_change && models.size() < 24) {
					unaccepted_putative_instances = 0;
					compound_preference_vector *= 0.1;  // Very aggressively decay (90%) to explore new regions
					MSACScoringFunctionWithCompoundModel<_ModelEstimator> &scoring =
						proposal_engine->getMutableScoringFunction();
					scoring.setCompoundModel(&models, &compound_preference_vector);
					if (do_logging)
						printf("Reset rejection counter - continuing search for more models (currently have %zu, target: 24)\n", models.size());
					continue;
				}
				// Stop if we've exhausted rejections and don't have enough models yet
				if (unaccepted_putative_instances == settings.max_proposal_number_without_change && models.size() < 24)
					break;
				continue;
			}
			
			// The end time of the model validation
			end = std::chrono::system_clock::now();

			// The elapsed time in seconds
			elapsed_seconds = end - start;

			// Update the current iteration's statistics
			iteration_statistics.time_of_model_validation =
				elapsed_seconds.count();

			++statistics.total_proposals_accepted; // Count accepted proposals
			unaccepted_putative_instances = 0;  // Reset rejection counter on acceptance
			
			// Report acceptance only if logging is enabled
			if (do_logging) {
				printf("✓ Accepted model %zu: %zu inliers, %zu total models, %zu/%zu proposals accepted (%.1f%%)\n",
					models.size() + 1, proposal_engine_statistics.inliers.size(),
					models.size() + 1, statistics.total_proposals_accepted, statistics.total_proposals_tested,
					statistics.total_proposals_tested > 0 ? 100.0 * statistics.total_proposals_accepted / statistics.total_proposals_tested : 0.0);
				fflush(stdout);
			}
			if (do_logging) {
				std::cout << "The model has been accepted in " <<
					iteration_statistics.time_of_model_validation << " seconds.\n";
			}

			/******************************************
			*** Compound instance optimization step ***
			******************************************/
			// The starting time of the model optimization
			start = std::chrono::system_clock::now();

			// Add the putative instance to the compound one
			// Preference vector was already calculated during validation
			models.emplace_back(putative_model);

			// If only a single model instance is known, use the inliers of GC-RANSAC
			// to initialize the labeling.
			if (do_logging)
				std::cout << "Model optimization started...\n";
			if (models.size() == 1)
			{
				// Store the inliers of the current model to the statistics object
				statistics.inliers_of_each_model.emplace_back(
					proposal_engine->getRansacStatistics().inliers);

				// Set the labeling so that the outliers will have label 1 and the inliers label 0.
				std::fill(statistics.labeling.begin(), statistics.labeling.end(), 1);
				for (const size_t &point_idx : statistics.inliers_of_each_model.back())
					statistics.labeling[point_idx] = 0;
			}
			// Otherwise, apply an optimizer the determine the labeling
			else
			{
				// OPTIMIZED: Skip PEARL optimization more aggressively to reduce runtime
				// PEARL is expensive (25% of runtime) and not needed until we have many models
				// Only run PEARL every 5 models or when we have >= 8 models (was every 3)
				// This reduces PEARL time from 25% to ~10-15% of total runtime
				bool should_run_pearl = (models.size() >= 8) && (models.size() % 5 == 0 || models.size() <= 10);
				
				if (should_run_pearl) {
					// Apply the model optimizer
					model_optimizer->run(data_,
						&neighborhood_graph_,
						&model_estimator,
						&models);

					size_t model_number = 0;
					model_optimizer->getLabeling(statistics.labeling, model_number);

					if (model_number != models.size() && do_logging)
					{
						size_t removed_count = models.size() - model_number;
						printf("⚠ PEARL optimization removed %zu model(s) (had 0 inliers after optimization)\n", removed_count);
						printf("   Models before optimization: %zu, after: %zu\n", models.size(), model_number);
						fflush(stdout);
						std::cout << "Models have been removed during the optimization.\n\n";
					}
				} else {
					// STRUCTURAL OPTIMIZATION: For models 2, 4, 6, etc., use simple labeling
					// Assign points to nearest model without full PEARL optimization
					// This is much faster and sufficient for initial models
					for (size_t point_idx = 0; point_idx < point_number; ++point_idx) {
						double min_residual = std::numeric_limits<double>::max();
						size_t best_label = models.size(); // Outlier label
						
						for (size_t model_idx = 0; model_idx < models.size(); ++model_idx) {
							double residual = model_estimator.squaredResidual(data_.row(point_idx), models[model_idx]);
							if (residual < truncated_squared_threshold && residual < min_residual) {
								min_residual = residual;
								best_label = model_idx;
							}
						}
						statistics.labeling[point_idx] = best_label;
					}
				}
			}

			// The end time of the model optimization
			end = std::chrono::system_clock::now();

			// The elapsed time in seconds
			elapsed_seconds = end - start;

			// Update the current iteration's statistics
			iteration_statistics.time_of_optimization =
				elapsed_seconds.count();

			if (do_logging)
				std::cout << "Model optimization finished in " <<
					iteration_statistics.time_of_optimization << " seconds.\n";

			// The starting time of the model validation
			start = std::chrono::system_clock::now();

			// STRUCTURAL OPTIMIZATION: Always update compound model incrementally
			// This is faster than full recomputation and ensures accuracy for scoring
			// Incremental update: only add the new model's preference vector
			if (models.size() > 0) {
				const auto &new_model = models.back();
				// Update compound preference vector with new model's preferences
				// This is O(n) instead of O(n*m) for full recomputation
				for (size_t point_idx = 0; point_idx < point_number; ++point_idx) {
					double new_pref = new_model.preference_vector(point_idx);
					compound_preference_vector(point_idx) = 
						MAX(compound_preference_vector(point_idx), new_pref);
				}
			}

			// The end time of the model optimization
			end = std::chrono::system_clock::now();

			// The elapsed time in seconds
			elapsed_seconds = end - start;

			// Update the current iteration's statistics
			iteration_statistics.time_of_compound_model_update =
				elapsed_seconds.count();

			if (do_logging)
				std::cout << "Compound instance (containing " << models.size() << " models) is updated in " <<
					iteration_statistics.time_of_compound_model_update << " seconds.\n";

			// Store the instance number in the iteration's statistics
			iteration_statistics.number_of_instances = models.size();

			/************************************
			*** Updating the iteration number ***
			************************************/
			// If there is a only a single model instance, PEARL was not applied. 
			// Thus, the inliers are in the RANSAC statistics of the instance.
			if (models.size() == 1)
				unseen_inliers = getPredictedUnseenInliers(settings.one_minus_confidence, // 1.0 - confidence
					_ModelEstimator::sampleSize(), // The size of a minimal sample
					number_of_ransac_iterations, // The total number of RANSAC iterations applied
					statistics.inliers_of_each_model.size()); // The inlier number of the compound model instance

			else
				unseen_inliers = getPredictedUnseenInliers(settings.one_minus_confidence, // 1.0 - confidence
					_ModelEstimator::sampleSize(), // The size of a minimal sample
					number_of_ransac_iterations, // The total number of RANSAC iterations applied
					point_number - model_optimizer->getOutlierNumber()); // The inlier number of the compound model instance

			// Add the current iteration's statistics to the statistics object
			statistics.addIterationStatistics(iteration_statistics);

			if (do_logging)
				std::cout << "The predicted number of inliers (with confidence " << settings.confidence <<
					")\nnot covered by the compound instance is " << unseen_inliers << ".\n";

			// If it is likely, that there are fewer inliers in the data than the minimum number,
			// terminate.
			// OPTIMIZED: Early termination when we have enough models (24-25) and few unseen inliers
			// This stops the algorithm once we've found the target number of lines
			if (models.size() >= 24 && unseen_inliers < settings.minimum_number_of_inliers) {
				if (do_logging)
					printf("Early termination: %zu models found, %zu unseen inliers predicted\n", models.size(), unseen_inliers);
				break;
			}
			// Also stop if we have 25+ models regardless of unseen_inliers (we have enough)
			if (models.size() >= 25) {
				if (do_logging)
					printf("Early termination: %zu models found (sufficient), stopping\n", models.size());
				break;
			}
			// If we only have 1 model, continue to find at least one more
			// Only stop if we've tried many iterations without finding more models

			// If we have enough models, terminate.
			if (getModelNumber() >= settings.maximum_model_number)
				break;

			// Visualize the labeling results if needed
			if (visualizer != nullptr)
			{
				visualizer->setLabelNumber(models.size() + 1);
				visualizer->visualize(0, "Labeling");
			}
		}

		// Print summary only if logging is enabled
		if (do_logging) {
			printf("\n===========================================\n");
			printf("Progressive-X Algorithm Summary:\n");
			printf("  Total iterations: %zu\n", statistics.iteration_statistics.size());
			printf("  Total candidate proposals tested: %zu\n", statistics.total_proposals_tested);
			printf("  Total proposals accepted: %zu\n", statistics.total_proposals_accepted);
			if (statistics.total_proposals_tested > 0) {
				printf("  Acceptance rate: %.2f%%\n", 100.0 * statistics.total_proposals_accepted / statistics.total_proposals_tested);
			}
			printf("  Final number of models found: %zu\n", models.size());
			printf("  NOTE: Some models may have been removed during PEARL optimization\n");
			printf("  (models with 0 inliers after optimization are automatically removed)\n");
			printf("  Stopped after: %zu iterations\n", statistics.iteration_statistics.size());
			printf("===========================================\n\n");
			fflush(stdout);
		}

		main_end = std::chrono::system_clock::now();

		// The elapsed time in seconds
		elapsed_seconds = main_end - main_start;

		statistics.processing_time = elapsed_seconds.count();
	}

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
	size_t ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::getPredictedUnseenInliers(
		const double one_minus_confidence_,
		const size_t sample_size_,
		const size_t iteration_number_,
		const size_t inlier_number_of_compound_model_)
	{
		// Number of points in the data which have not yet been assigned to any model
		const size_t unseen_point_number = point_number - inlier_number_of_compound_model_;

		const double one_over_iteration_number = 1.0 / iteration_number_;
		const double one_over_sample_size = 1.0 / sample_size_;

		// Calculate the ratio of the maximum inlier number from the sample size, current iteration number and confidence
		const double inlier_ratio = 
			pow(1.0 - pow(one_minus_confidence_, one_over_iteration_number), one_over_sample_size);

		// Return the number of unseen inliers
		return static_cast<size_t>(std::round(unseen_point_number * inlier_ratio));
	}

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
	void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::initialize(const cv::Mat &data_)
	{
		point_number = data_.rows; // The number of data points
		statistics.labeling.resize(point_number, 0); // The labeling which assigns each point to a model instance. Initially, all points are considered outliers.
		truncated_squared_threshold = 9.0 / 4.0 * settings.inlier_outlier_threshold *  settings.inlier_outlier_threshold;
		compound_preference_vector = Eigen::VectorXd::Zero(data_.rows);

		// Initializing the model optimizer, i.e., PEARL
		// Reduced to 30 iterations for faster execution (~40s target)
		// PEARL iterations are expensive, so reducing helps speed up execution significantly
		model_optimizer = std::make_unique<pearl::PEARL<_NeighborhoodGraph,
			_ModelEstimator>>(
				settings.inlier_outlier_threshold,
				settings.spatial_coherence_weight,
				settings.minimum_number_of_inliers,
				settings.point_weights,
				5,  // OPTIMIZED: Reduced from 15 to 5 for faster execution - PEARL converges very quickly
				do_logging);

		// Initializing the proposal engine, i.e., Graph-Cut RANSAC
		proposal_engine = std::make_unique < gcransac::GCRANSAC <_ModelEstimator,
			_NeighborhoodGraph,
			MSACScoringFunctionWithCompoundModel<_ModelEstimator>>>();

		gcransac::utils::Settings &proposal_engine_settings = proposal_engine->settings;
		proposal_engine_settings = settings.proposal_engine_settings;
		proposal_engine_settings.confidence = settings.confidence;
		proposal_engine_settings.threshold = settings.inlier_outlier_threshold;
		proposal_engine_settings.spatial_coherence_weight = settings.spatial_coherence_weight;

		MSACScoringFunctionWithCompoundModel<_ModelEstimator> &scoring =
			proposal_engine->getMutableScoringFunction();
		scoring.setCompoundModel(&models, 
			&compound_preference_vector);
		// STRUCTURAL FIX: Cast to int, but scoring function now handles exponent=0 correctly
		scoring.setExponent(static_cast<int>(scoring_exponent));
		// OPTIMIZATION: Set minimum inliers for early rejection heuristic
		// This allows quick rejection of bad models before expensive full inlier counting
		scoring.setMinInliersRequired(settings.minimum_number_of_inliers);

		// Initialize the visualizer if needed
		if (visualizer != nullptr)
		{
			visualizer->setLabeling(&statistics.labeling, // Set the labeling pointer 
				1); // Initially, only the outlier model instance exists
		}
	}

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
	inline bool ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::isPutativeModelValid(
		const cv::Mat &data_,
		progx::Model<_ModelEstimator> &model_,
		const gcransac::utils::RANSACStatistics &statistics_)
	{
		// Number of inliers without considering that there are more model instances in the scene
		const size_t inlier_number = statistics_.inliers.size();

		// If the putative model has fewer inliers than the minimum, it is considered invalid.
		// OPTIMIZED: Require minimum_number_of_inliers (15) to reject weak proposals early
		// This speeds up GC-RANSAC significantly by rejecting proposals before expensive validation
		// True lines have 20-110 points, so requiring 15 is safe and speeds up the algorithm
		if (inlier_number < settings.minimum_number_of_inliers)
			return false;

		// STRUCTURAL OPTIMIZATION: Always calculate preference vector (needed for compound model scoring)
		// But skip Tanimoto check for first 50 models to allow finding many parallel lines
		model_.setPreferenceVector(data_, // All data points
			truncated_squared_threshold); // The truncated squared threshold

		// For 3D lines: Check if line is parallel to time axis (Z direction)
		// Lines should be parallel to (0,0,1) for event data
		// Only check if this is a 3D line estimator (descriptor has 6 elements)
		if (model_.descriptor.rows() == 6 && model_.descriptor.cols() == 1) {
			// Extract direction vector (last 3 elements: dx, dy, dz)
			Eigen::Vector3d direction(
				model_.descriptor(3, 0),
				model_.descriptor(4, 0),
				model_.descriptor(5, 0)
			);
			direction.normalize();
			
			// Check if parallel to time axis (Z): direction should be close to (0, 0, 1) or (0, 0, -1)
			Eigen::Vector3d time_axis(0.0, 0.0, 1.0);
			double dot_with_time = std::abs(direction.dot(time_axis));
			
			// Require lines to be within 10° of time axis (cos(10°) ≈ 0.9848)
			double min_parallel_dot = 0.9848; // cos(10°)
				if (dot_with_time < min_parallel_dot) {
				double angle_deg = std::acos(std::max(-1.0, std::min(1.0, dot_with_time))) * 180.0 / 3.14159265358979323846;
				if (do_logging)
					printf("Rejected non-time-parallel line: angle=%.1f° (threshold=10°), direction=(%.3f,%.3f,%.3f)\n",
						angle_deg, direction(0), direction(1), direction(2));
				return false;
			}
		}
		
		// For 3D lines: Also check spatial separation to reject duplicates
		// Parallel lines can have high Tanimoto similarity, so check if they're too close spatially
		if (model_.descriptor.rows() == 6 && model_.descriptor.cols() == 1 && models.size() > 0) {
			// Extract new line parameters: point (p₀x, p₀y, p₀z) and direction (dx, dy, dz)
			Eigen::Vector3d new_point(model_.descriptor(0, 0), model_.descriptor(1, 0), model_.descriptor(2, 0));
			Eigen::Vector3d new_dir(model_.descriptor(3, 0), model_.descriptor(4, 0), model_.descriptor(5, 0));
			new_dir.normalize();
			
			// Check distance to existing lines
			double min_separation = std::numeric_limits<double>::max();
			for (const auto &existing_model : models) {
				if (existing_model.descriptor.rows() == 6 && existing_model.descriptor.cols() == 1) {
					Eigen::Vector3d existing_point(existing_model.descriptor(0, 0), existing_model.descriptor(1, 0), existing_model.descriptor(2, 0));
					Eigen::Vector3d existing_dir(existing_model.descriptor(3, 0), existing_model.descriptor(4, 0), existing_model.descriptor(5, 0));
					existing_dir.normalize();
					
					// Check if directions are parallel (for parallel lines, check spatial separation)
					double dir_dot = std::abs(new_dir.dot(existing_dir));
					if (dir_dot > 0.95) {  // Lines are nearly parallel
						// Calculate perpendicular distance between parallel lines
						Eigen::Vector3d vec_between = new_point - existing_point;
						Eigen::Vector3d perp = vec_between - vec_between.dot(new_dir) * new_dir;
						double separation = perp.norm();
						min_separation = std::min(min_separation, separation);
					}
				}
			}
			
			// Reject if too close to an existing parallel line (within 3× threshold for parallel lines)
			// For parallel lines, we need more separation to be considered distinct
			double min_separation_threshold = settings.inlier_outlier_threshold * 3.0;
			if (min_separation < min_separation_threshold && min_separation < std::numeric_limits<double>::max()) {
				if (do_logging)
					printf("Rejected duplicate: too close to existing parallel line (separation=%.2f < threshold=%.2f, model_count=%zu)\n",
						min_separation, min_separation_threshold, models.size());
				return false;
			}
		}
		
		// STRUCTURAL OPTIMIZATION: Apply Tanimoto similarity check to reject duplicates
		// Check similarity starting from model 2 to catch duplicates very early
		size_t current_model_count = getModelNumber();
		if (current_model_count >= 2) {
			// Calculate the Tanimoto-distance of the preference vectors of the current and the 
			// compound model instance.
			const double dot_product = model_.preference_vector.dot(compound_preference_vector);
			double norm_sum = model_.preference_vector.squaredNorm() + compound_preference_vector.squaredNorm();
			if (norm_sum > 1e-10) {  // Avoid division by zero
				double tanimoto_similarity = dot_product / (norm_sum - dot_product);

				// Use the actual setting (0.40 from Python) to reject duplicates
				// This will reject models that are too similar to existing ones
				double effective_max_similarity = settings.maximum_tanimoto_similarity;
				
				// Debug: Report Tanimoto similarity only if logging is enabled
				if (do_logging && current_model_count <= 10 && (statistics.total_proposals_tested % 5 == 0)) {
					printf("Tanimoto check: similarity=%.3f, threshold=%.3f, model_count=%zu\n",
						tanimoto_similarity, effective_max_similarity, current_model_count);
				}
				
				if (effective_max_similarity < tanimoto_similarity) {
					if (do_logging)
						printf("Rejected duplicate: Tanimoto similarity %.3f > threshold %.3f (model %zu, tested %zu)\n", 
							tanimoto_similarity, effective_max_similarity, current_model_count, statistics.total_proposals_tested);
					return false;
				}
			} else {
				// Debug: Report if norm_sum is too small (only if logging)
				if (do_logging && current_model_count <= 5)
					printf("Tanimoto check skipped: norm_sum too small (%.2e)\n", norm_sum);
			}
		}
		// For first 2 models: skip Tanimoto similarity check to allow initial exploration

		return true;
	}

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
	void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::updateCompoundModel(const cv::Mat &data_)
	{
		// Do not do anything if there are no models in the compound instance
		if (models.size() == 0)
			return;

		// Reset the preference vector of the compound instance
		compound_preference_vector.setConstant(0);

		// Iterate through all instances in the compound one and 
		// update the preference values
		for (auto &model : models)
		{
			// Iterate through all points and estimate the preference values
			double squared_residual;
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
			{
				// The point-to-model residual
				squared_residual = model_estimator.squaredResidual(data_.row(point_idx), model);
				
				// Update the preference vector of the compound model. Since the point-to-<compound model>
				// residual is defined through the union of distance fields of the contained models,
				// the implied preference is the highest amongst the stored model instances. 
				compound_preference_vector(point_idx) =
					MAX(compound_preference_vector(point_idx), model.preference_vector(point_idx));
			}
		}
	}
}