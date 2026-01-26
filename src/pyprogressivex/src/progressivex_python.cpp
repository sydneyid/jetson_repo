#include "progressivex_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#include "progx_utils.h"
#include "utils.h"
#include "GCoptimization.h"
#include "neighborhood/grid_neighborhood_graph.h"
#include "neighborhood/flann_neighborhood_graph.h"

#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/napsac_sampler.h"
#include "samplers/progressive_napsac_sampler.h"
#include "samplers/z_aligned_sampler.h"

#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "estimators/essential_estimator.h"
#include "estimators/estimator.h"
#include "estimators/solver_engine.h"
#include "vanishing_point_estimator.h"
#include "solver_vanishing_point_two_lines.h"

#include "progressive_x.h"

#include <ctime>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	#include <direct.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>
#include <glog/logging.h>
#include <algorithm>
#include <numeric>

// Global initialization function for Google Logging
// This ensures InitGoogleLogging is only called once, even if there are
// gflags conflicts (which will cause the process to exit, but at least
// we only try once)
// 
// IMPORTANT: If there are gflags conflicts (duplicate flag definitions),
// InitGoogleLogging will call exit() and terminate the process. This cannot
// be caught or prevented from Python. The only solution is to fix the
// environment or suppress stderr at the shell level: python script.py 2>/dev/null
namespace {
	static std::once_flag g_logging_init_flag;
	static bool g_logging_initialized = false;
	
	inline void initGoogleLoggingOnce() {
		std::call_once(g_logging_init_flag, []() {
			if (!g_logging_initialized) {
				// Try to initialize logging
				// If this fails due to gflags conflicts, the process will exit
				// There's no way to catch this error - it's a fatal exit()
				try {
					google::InitGoogleLogging("pyprogessivex");
					g_logging_initialized = true;
				} catch (...) {
					// InitGoogleLogging doesn't throw - it calls exit() on error
					// This catch will never execute, but it's here for clarity
					g_logging_initialized = true; // Mark as attempted
				}
			}
		});
	}
}

int find6DPoses_(
	const std::vector<double>& imagePoints,
	const std::vector<double>& worldPoints,
	const std::vector<double>& intrinsicParams,
	std::vector<size_t>& labeling,
	std::vector<double>& poses,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number)
{
	// Initialize Google's logging library (once globally)
	initGoogleLoggingOnce();
	
	// Calculate the inverse of the intrinsic camera parameters
	Eigen::Matrix3d K;
	K << intrinsicParams[0], intrinsicParams[1], intrinsicParams[2],
		intrinsicParams[3], intrinsicParams[4], intrinsicParams[5],
		intrinsicParams[6], intrinsicParams[7], intrinsicParams[8];
	const Eigen::Matrix3d Kinv =
		K.inverse();
	
	Eigen::Vector3d vec;
	vec(2) = 1;
	size_t num_tents = imagePoints.size() / 2;
	cv::Mat points(num_tents, 5, CV_64F);
	cv::Mat normalized_points(num_tents, 5, CV_64F);
	size_t iterations = 0;
	for (size_t i = 0; i < num_tents; ++i) {
		vec(0) = imagePoints[2 * i];
		vec(1) = imagePoints[2 * i + 1];
		
		points.at<double>(i, 0) = imagePoints[2 * i];
		points.at<double>(i, 1) = imagePoints[2 * i + 1];
		points.at<double>(i, 2) = worldPoints[3 * i];
		points.at<double>(i, 3) = worldPoints[3 * i + 1];
		points.at<double>(i, 4) = worldPoints[3 * i + 2];
		
		normalized_points.at<double>(i, 0) = Kinv.row(0) * vec;
		normalized_points.at<double>(i, 1) = Kinv.row(1) * vec;
		normalized_points.at<double>(i, 2) = worldPoints[3 * i];
		normalized_points.at<double>(i, 3) = worldPoints[3 * i + 1];
		normalized_points.at<double>(i, 4) = worldPoints[3 * i + 2];
	}
	
	// Normalize the threshold
	const double f = 0.5 * (K(0,0) + K(1,1));
	const double normalized_threshold =
		threshold / f;
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds

	printf("Neighborhood calculation time = %f secs.\n", elapsed_seconds.count());

	// The main sampler is used inside the local optimization
	gcransac::sampler::UniformSampler main_sampler(&points);

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultPnPEstimator, // The type of the used model estimator
		gcransac::sampler::UniformSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = normalized_threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;

	progressive_x.run(normalized_points, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;
	poses.reserve(12 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		poses.emplace_back(model.descriptor(0, 0));
		poses.emplace_back(model.descriptor(0, 1));
		poses.emplace_back(model.descriptor(0, 2));
		poses.emplace_back(model.descriptor(0, 3));
		poses.emplace_back(model.descriptor(1, 0));
		poses.emplace_back(model.descriptor(1, 1));
		poses.emplace_back(model.descriptor(1, 2));
		poses.emplace_back(model.descriptor(1, 3));
		poses.emplace_back(model.descriptor(2, 0));
		poses.emplace_back(model.descriptor(2, 1));
		poses.emplace_back(model.descriptor(2, 2));
		poses.emplace_back(model.descriptor(2, 3));
	}
	
	return progressive_x.getModelNumber();
}

int findHomographies_(
	std::vector<double>& correspondences,
	std::vector<size_t>& labeling,
	std::vector<double>& homographies,
	const size_t &source_image_width,
	const size_t &source_image_height,
	const size_t &destination_image_width,
	const size_t &destination_image_height,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library (once globally)
	initGoogleLoggingOnce();
	
	const size_t num_tents = correspondences.size() / 4;
		
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = gcransac::utils::DefaultHomographyEstimator::sampleSize();
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the correspondences to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: Progressive NAPSAC sampler requires the correspondences to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			kSampleSize, // The size of a minimal sample
			{
                static_cast<double>(source_image_width),
                static_cast<double>(source_image_height),
                static_cast<double>(destination_image_width),
                static_cast<double>(destination_image_height)
            }, // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	}
	else if (sampler_id == 3) // Initializing a NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>(&points, &neighborhood));
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultHomographyEstimator, // The type of the used model estimator
		AbstractSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;
	// INCREASE max_proposal_number_without_change to allow more attempts before stopping
	// Default is 10, which causes early stopping when models are rejected
	settings.max_proposal_number_without_change = 100;  // Allow 100 consecutive rejections before stopping
	// Setting the scoring exponent
	progressive_x.setScoringExponent(scoring_exponent);

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		*main_sampler.get(), // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;

	homographies.reserve(9 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		homographies.emplace_back(model.descriptor(0, 0));
		homographies.emplace_back(model.descriptor(0, 1));
		homographies.emplace_back(model.descriptor(0, 2));
		homographies.emplace_back(model.descriptor(1, 0));
		homographies.emplace_back(model.descriptor(1, 1));
		homographies.emplace_back(model.descriptor(1, 2));
		homographies.emplace_back(model.descriptor(2, 0));
		homographies.emplace_back(model.descriptor(2, 1));
		homographies.emplace_back(model.descriptor(2, 2));
	}
	
	return progressive_x.getModelNumber();
}

int findVanishingPoints_(
	std::vector<double>& lines,
	std::vector<double>& weights,
	std::vector<size_t>& labeling,
	std::vector<double>& vanishing_points,
	const size_t &image_width,
	const size_t &image_height,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library (once globally)
	initGoogleLoggingOnce();
	
	const size_t num_lines = lines.size() / 4;
		
	cv::Mat points(num_lines, 4, CV_64F, &lines[0]);
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.

	// The default estimator for homography fitting
	typedef gcransac::estimator::VanishingPointEstimator<
		gcransac::estimator::solver::VanishingPointTwoLineSolver, // The solver used for fitting a model to a minimal sample
		gcransac::estimator::solver::VanishingPointTwoLineSolver> // The solver used for fitting a model to a non-minimal sample
		DefaultVanishingPointEstimator;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = DefaultVanishingPointEstimator::sampleSize();
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the correspondences to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		DefaultVanishingPointEstimator, // The type of the used model estimator
		AbstractSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// Weights of the lines used in LSQ fitting
	settings.point_weights = weights;
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// SPEED OPTIMIZATION: Reduce local optimization iterations (default 50 -> 10)
	// Local optimization is expensive and often doesn't need 50 iterations
	settings.proposal_engine_settings.max_local_optimization_number = 10;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;
	// Setting the scoring exponent
	progressive_x.setScoringExponent(scoring_exponent);
	// Set the logging parameter
	progressive_x.log(do_logging);

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		*main_sampler.get(), // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;

	vanishing_points.reserve(3 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		vanishing_points.emplace_back(model.descriptor(0));
		vanishing_points.emplace_back(model.descriptor(1));
		vanishing_points.emplace_back(model.descriptor(2));
	}
	
	return progressive_x.getModelNumber();
}

int findLines_(
	std::vector<double>& input_points,
	std::vector<double>& weights,
	std::vector<size_t>& labeling,
	std::vector<double>& lines,
	const size_t &image_width,
	const size_t &image_height,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library (once globally)
	initGoogleLoggingOnce();
	
	const size_t num_tents = input_points.size() / 2;
		
	cv::Mat points(num_tents, 2, CV_64F, &input_points[0]);
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = gcransac::utils::DefaultHomographyEstimator::sampleSize();
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the points to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else if (sampler_id == 2) // Initializing a NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>(&points, &neighborhood));
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::Default2DLineEstimator, // The type of the used model estimator
		AbstractSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;
	// INCREASE max_proposal_number_without_change to allow more attempts before stopping
	// Default is 10, which causes early stopping when models are rejected
	settings.max_proposal_number_without_change = 100;  // Allow 100 consecutive rejections before stopping
	// Setting the scoring exponent
	progressive_x.setScoringExponent(scoring_exponent);

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		*main_sampler.get(), // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;

	lines.reserve(3 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		lines.emplace_back(model.descriptor(0));
		lines.emplace_back(model.descriptor(1));
		lines.emplace_back(model.descriptor(2));
	}
	
	return progressive_x.getModelNumber();
}

int findTwoViewMotions_(
	std::vector<double>& correspondences,
	std::vector<size_t>& labeling,
	std::vector<double>& motions,
	const size_t &source_image_width,
	const size_t &source_image_height,
	const size_t &destination_image_width,
	const size_t &destination_image_height,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library (once globally)
	initGoogleLoggingOnce();
	
	const size_t num_tents = correspondences.size() / 4;
		
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = gcransac::utils::DefaultFundamentalMatrixEstimator::sampleSize();
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the correspondences to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: Progressive NAPSAC sampler requires the correspondences to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			kSampleSize, // The size of a minimal sample
			{
                static_cast<double>(source_image_width),
                static_cast<double>(source_image_height),
                static_cast<double>(destination_image_width),
                static_cast<double>(destination_image_height)
            }, // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	}
	else if (sampler_id == 3) // Initializing a NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>(&points, &neighborhood));
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultFundamentalMatrixEstimator, // The type of the used model estimator
		AbstractSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		*main_sampler.get(), // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;
	
	motions.reserve(9 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		motions.emplace_back(model.descriptor(0, 0));
		motions.emplace_back(model.descriptor(0, 1));
		motions.emplace_back(model.descriptor(0, 2));
		motions.emplace_back(model.descriptor(1, 0));
		motions.emplace_back(model.descriptor(1, 1));
		motions.emplace_back(model.descriptor(1, 2));
		motions.emplace_back(model.descriptor(2, 0));
		motions.emplace_back(model.descriptor(2, 1));
		motions.emplace_back(model.descriptor(2, 2));
	}
	
	return progressive_x.getModelNumber();
}

// ============================================================================
// 3D Line Estimator Implementation
// ============================================================================

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// 3D Line Solver: Estimates a 3D line from 3D points
			// A 3D line is represented as: point + direction vector
			// Model descriptor: [p₀x, p₀y, p₀z, dx, dy, dz] (6 parameters)
			class Line3DSolver : public SolverEngine
			{
			public:
				Line3DSolver() {}
				~Line3DSolver() {}

				static constexpr bool returnMultipleModels() { return false; }
				static constexpr size_t maximumSolutions() { return 1; }
				static constexpr size_t sampleSize() { return 2; } // 2 points needed for a 3D line
				static constexpr bool needsGravity() { return false; }

				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_,
					const size_t *sample_,
					size_t sample_number_,
					std::vector<Model> &models_,
					const double *weights_ = nullptr) const
				{
					const double* dataPtr = reinterpret_cast<const double*>(data_.data);
					const int kColumns = data_.cols;

					if (sample_number_ == 2)
					{
						// Minimal case: 2 points define a line
						const double* p1 = dataPtr + kColumns * sample_[0];
						const double* p2 = dataPtr + kColumns * sample_[1];

						// Point on line: use first point
						Eigen::Vector3d point(p1[0], p1[1], p1[2]);
						
						// Direction vector: from p1 to p2, normalized
						Eigen::Vector3d direction(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
						double dir_norm = direction.norm();
						if (dir_norm < 1e-10) // Points are too close
							return false;
						direction.normalize();

						models_.resize(models_.size() + 1);
						models_.back().descriptor.resize(6, 1);
						models_.back().descriptor << point, direction;
						return true;
					}
					else
					{
						// Non-minimal case: fit line to multiple points using SVD
						// Find the line that minimizes perpendicular distances
						Eigen::MatrixXd points(sample_number_, 3);
						for (size_t i = 0; i < sample_number_; ++i)
						{
							const double* p = dataPtr + kColumns * sample_[i];
							points(i, 0) = p[0];
							points(i, 1) = p[1];
							points(i, 2) = p[2];
						}

						// Center the points
						Eigen::Vector3d centroid = points.colwise().mean();
						Eigen::MatrixXd centered = points.rowwise() - centroid.transpose();

						// SVD to find the direction (principal component)
						Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinV);
						Eigen::Vector3d direction = svd.matrixV().col(0);
						direction.normalize();

						// Use centroid as point on line
						Eigen::Vector3d point = centroid;

						models_.resize(models_.size() + 1);
						models_.back().descriptor.resize(6, 1);
						models_.back().descriptor << point, direction;
						return true;
					}
				}
			};
		}

		// 3D Line Estimator
		template<class _MinimalSolverEngine, class _NonMinimalSolverEngine>
		class Line3DEstimator : public Estimator<cv::Mat, Model>
		{
		protected:
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

		public:
			Line3DEstimator() :
				minimal_solver(std::make_shared<_MinimalSolverEngine>()),
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>())
			{}

			~Line3DEstimator() {}

			static constexpr size_t sampleSize() { return _MinimalSolverEngine::sampleSize(); }
			static constexpr size_t maximumMinimalSolutions() { return _MinimalSolverEngine::maximumSolutions(); }
			static constexpr bool isWeightingApplicable() { return true; }

			inline size_t inlierLimit() const { return 7 * sampleSize(); }

			inline bool estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				std::vector<Model>* models_) const
			{
				return minimal_solver->estimateModel(data_, sample_, sampleSize(), *models_);
			}

			inline bool estimateModelNonminimal(
				const cv::Mat& data_,
				const size_t *sample_,
				const size_t &sample_number,
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const
			{
				return non_minimal_solver->estimateModel(data_, sample_, sample_number, *models_, weights_);
			}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// Calculate point-to-line distance in 3D
			inline double residual(const cv::Mat& data_, const Model& model) const
			{
				return std::sqrt(squaredResidual(data_, model));
			}

			inline double squaredResidual(const cv::Mat& data_, const Model& model) const
			{
				// data_ is a single point (a row): shape [1, 3]
				const double* pointPtr = reinterpret_cast<const double*>(data_.data);
				
				// Extract line parameters: [p₀x, p₀y, p₀z, dx, dy, dz]
				Eigen::Vector3d line_point(model.descriptor(0), model.descriptor(1), model.descriptor(2));
				Eigen::Vector3d line_dir(model.descriptor(3), model.descriptor(4), model.descriptor(5));
				
				// Point coordinates (row-major storage)
				Eigen::Vector3d point(pointPtr[0], pointPtr[1], pointPtr[2]);
				
				// Vector from line point to data point
				Eigen::Vector3d vec = point - line_point;
				
				// Perpendicular distance: ||(point - line_point) × direction||
				Eigen::Vector3d cross = vec.cross(line_dir);
				return cross.squaredNorm();
			}
		};

		// Default 3D Line Estimator type
		typedef Line3DEstimator<solver::Line3DSolver, solver::Line3DSolver> Default3DLineEstimator;

		// 3D Line Solver for Sparse/Noisy Lines
		// Uses more robust fitting for sparse or noisy data
		namespace solver
		{
			class Line3DSparseSolver : public SolverEngine
			{
			public:
				Line3DSparseSolver() {}
				~Line3DSparseSolver() {}

				static constexpr bool returnMultipleModels() { return false; }
				static constexpr size_t maximumSolutions() { return 1; }
				static constexpr size_t sampleSize() { return 2; } // 2 points needed for a 3D line
				static constexpr bool needsGravity() { return false; }

				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_,
					const size_t *sample_,
					size_t sample_number_,
					std::vector<Model> &models_,
					const double *weights_ = nullptr) const
				{
					const double* dataPtr = reinterpret_cast<const double*>(data_.data);
					const int kColumns = data_.cols;

					if (sample_number_ == 2)
					{
						// Minimal case: 2 points define a line
						const double* p1 = dataPtr + kColumns * sample_[0];
						const double* p2 = dataPtr + kColumns * sample_[1];

						// Point on line: use first point
						Eigen::Vector3d point(p1[0], p1[1], p1[2]);
						
						// Direction vector: from p1 to p2, normalized
						Eigen::Vector3d direction(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
						double dir_norm = direction.norm();
						if (dir_norm < 1e-10) // Points are too close
							return false;
						direction.normalize();

						models_.resize(models_.size() + 1);
						models_.back().descriptor.resize(6, 1);
						models_.back().descriptor << point, direction;
						return true;
					}
					else
					{
						// Non-minimal case: fit line to multiple points using robust SVD
						// For sparse lines, we use the same SVD approach but the threshold
						// will be larger when this estimator is used
						Eigen::MatrixXd points(sample_number_, 3);
						for (size_t i = 0; i < sample_number_; ++i)
						{
							const double* p = dataPtr + kColumns * sample_[i];
							points(i, 0) = p[0];
							points(i, 1) = p[1];
							points(i, 2) = p[2];
						}

						// Center the points
						Eigen::Vector3d centroid = points.colwise().mean();
						Eigen::MatrixXd centered = points.rowwise() - centroid.transpose();

						// SVD to find the direction (principal component)
						Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinV);
						Eigen::Vector3d direction = svd.matrixV().col(0);
						direction.normalize();

						// Use centroid as point on line
						Eigen::Vector3d point = centroid;

						models_.resize(models_.size() + 1);
						models_.back().descriptor.resize(6, 1);
						models_.back().descriptor << point, direction;
						return true;
					}
				}
			};
		}

		// 3D Line Estimator for Sparse/Noisy Lines
		// Uses the same residual calculation but will be called with a larger threshold
		template<class _MinimalSolverEngine, class _NonMinimalSolverEngine>
		class Line3DSparseEstimator : public Estimator<cv::Mat, Model>
		{
		protected:
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

		public:
			Line3DSparseEstimator() :
				minimal_solver(std::make_shared<_MinimalSolverEngine>()),
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>())
			{}

			~Line3DSparseEstimator() {}

			static constexpr size_t sampleSize() { return _MinimalSolverEngine::sampleSize(); }
			static constexpr size_t maximumMinimalSolutions() { return _MinimalSolverEngine::maximumSolutions(); }
			static constexpr bool isWeightingApplicable() { return true; }

			inline size_t inlierLimit() const { return 7 * sampleSize(); }

			inline bool estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				std::vector<Model>* models_) const
			{
				return minimal_solver->estimateModel(data_, sample_, sampleSize(), *models_);
			}

			inline bool estimateModelNonminimal(
				const cv::Mat& data_,
				const size_t *sample_,
				const size_t &sample_number,
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const
			{
				return non_minimal_solver->estimateModel(data_, sample_, sample_number, *models_, weights_);
			}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// Calculate point-to-line distance in 3D (same as regular estimator)
			inline double residual(const cv::Mat& data_, const Model& model) const
			{
				return std::sqrt(squaredResidual(data_, model));
			}

			inline double squaredResidual(const cv::Mat& data_, const Model& model) const
			{
				// Same residual calculation as regular 3D line estimator
				const double* pointPtr = reinterpret_cast<const double*>(data_.data);
				
				// Extract line parameters: [p₀x, p₀y, p₀z, dx, dy, dz]
				Eigen::Vector3d line_point(model.descriptor(0), model.descriptor(1), model.descriptor(2));
				Eigen::Vector3d line_dir(model.descriptor(3), model.descriptor(4), model.descriptor(5));
				
				// Point coordinates (row-major storage)
				Eigen::Vector3d point(pointPtr[0], pointPtr[1], pointPtr[2]);
				
				// Vector from line point to data point
				Eigen::Vector3d vec = point - line_point;
				
				// Perpendicular distance: ||(point - line_point) × direction||
				Eigen::Vector3d cross = vec.cross(line_dir);
				return cross.squaredNorm();
			}
		};

		// Default 3D Line Sparse Estimator type
		typedef Line3DSparseEstimator<solver::Line3DSparseSolver, solver::Line3DSparseSolver> Default3DLineSparseEstimator;

		// 3D Line Solver with Temporal Constraint (for event data)
		// Ensures that the line direction has positive time component (dt > 0)
		// and that points along the line have monotonically increasing time
		namespace solver
		{
			class Line3DTemporalSolver : public SolverEngine
			{
			public:
				Line3DTemporalSolver() {}
				~Line3DTemporalSolver() {}

				static constexpr size_t sampleSize() { return 2; }
				static constexpr size_t maximumSolutions() { return 1; }

				inline bool estimateModel(
					const cv::Mat& data_,
					const size_t *sample_,
					size_t sample_number_,
					std::vector<Model> &models_,
					const double *weights_ = nullptr) const
				{
					const double* dataPtr = reinterpret_cast<const double*>(data_.data);
					const int kColumns = data_.cols;

					if (sample_number_ == 2)
					{
						// Minimal case: 2 points define a line
						// For event data: lines are parallel to Z (time) axis
						// Direction is ALWAYS (0, 0, 1) - parallel to time axis
						const double* p1 = dataPtr + kColumns * sample_[0];
						const double* p2 = dataPtr + kColumns * sample_[1];

						// Ensure p1 has earlier time than p2 (or swap if needed)
						// Time is the 3rd coordinate (index 2)
						const double* p_early = p1;
						const double* p_late = p2;
						if (p1[2] > p2[2]) // If p1 has later time, swap
						{
							p_early = p2;
							p_late = p1;
						}

						// For parallel lines: use spatial center (x, y) and earliest time
						// For 2 points, use midpoint (same as mean, but clearer intent)
						// Direction is ALWAYS (0, 0, 1) - parallel to time axis
						Eigen::Vector3d point;
						point(0) = (p_early[0] + p_late[0]) / 2.0;  // Spatial x center (midpoint)
						point(1) = (p_early[1] + p_late[1]) / 2.0;  // Spatial y center (midpoint)
						point(2) = p_early[2];  // Earliest time
						
						// Direction is ALWAYS parallel to time axis
						Eigen::Vector3d direction(0.0, 0.0, 1.0);

						models_.resize(models_.size() + 1);
						models_.back().descriptor.resize(6, 1);
						models_.back().descriptor << point, direction;
						return true;
					}
					else
					{
						// Non-minimal case: fit line to multiple points
						// For event data: lines are parallel to Z (time) axis
						// Direction is ALWAYS (0, 0, 1) - parallel to time axis
						Eigen::MatrixXd points(sample_number_, 3);
						for (size_t i = 0; i < sample_number_; ++i)
						{
							const double* p = dataPtr + kColumns * sample_[i];
							points(i, 0) = p[0];
							points(i, 1) = p[1];
							points(i, 2) = p[2];
						}

						// Sort points by time (ascending) to find earliest time
						std::vector<size_t> indices(sample_number_);
						std::iota(indices.begin(), indices.end(), 0);
						std::sort(indices.begin(), indices.end(), 
							[&points](size_t i, size_t j) { return points(i, 2) < points(j, 2); });

						// For parallel lines: use spatial center (x, y) and earliest time
						// Use MEDIAN for robust center calculation (better for noisy data)
						// Sort x and y coordinates separately to find medians
						std::vector<double> x_coords(sample_number_), y_coords(sample_number_);
						for (size_t i = 0; i < sample_number_; ++i)
						{
							x_coords[i] = points(i, 0);
							y_coords[i] = points(i, 1);
						}
						std::sort(x_coords.begin(), x_coords.end());
						std::sort(y_coords.begin(), y_coords.end());
						double x_center = x_coords[sample_number_ / 2];  // Median
						double y_center = y_coords[sample_number_ / 2];  // Median
						
						// Point on line: spatial center with earliest time
						Eigen::Vector3d point;
						point(0) = x_center;
						point(1) = y_center;
						point(2) = points(indices[0], 2);  // Earliest time
						
						// Direction is ALWAYS parallel to time axis
						Eigen::Vector3d direction(0.0, 0.0, 1.0);

						models_.resize(models_.size() + 1);
						models_.back().descriptor.resize(6, 1);
						models_.back().descriptor << point, direction;
						return true;
					}
				}
			};
		}

		// 3D Line Estimator with Temporal Constraint
		// Penalizes points that violate temporal ordering (going backwards in time)
		template<class _MinimalSolverEngine, class _NonMinimalSolverEngine>
		class Line3DTemporalEstimator : public Estimator<cv::Mat, Model>
		{
		protected:
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

		public:
			Line3DTemporalEstimator() :
				minimal_solver(std::make_shared<_MinimalSolverEngine>()),
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>())
			{}

			~Line3DTemporalEstimator() {}

			static constexpr size_t sampleSize() { return _MinimalSolverEngine::sampleSize(); }
			static constexpr size_t maximumMinimalSolutions() { return _MinimalSolverEngine::maximumSolutions(); }
			static constexpr bool isWeightingApplicable() { return true; }

			inline size_t inlierLimit() const { return 7 * sampleSize(); }

			inline bool estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				std::vector<Model>* models_) const
			{
				return minimal_solver->estimateModel(data_, sample_, sampleSize(), *models_);
			}

			inline bool estimateModelNonminimal(
				const cv::Mat& data_,
				const size_t *sample_,
				const size_t &sample_number,
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const
			{
				return non_minimal_solver->estimateModel(data_, sample_, sample_number, *models_, weights_);
			}

			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// Calculate point-to-line distance in 3D with temporal constraint
			inline double residual(const cv::Mat& data_, const Model& model) const
			{
				return std::sqrt(squaredResidual(data_, model));
			}

			inline double squaredResidual(const cv::Mat& data_, const Model& model) const
			{
				// data_ is a single point (a row): shape [1, 3] = [x, y, t]
				const double* pointPtr = reinterpret_cast<const double*>(data_.data);
				
				// Extract line parameters: [p₀x, p₀y, p₀z, dx, dy, dz]
				Eigen::Vector3d line_point(model.descriptor(0), model.descriptor(1), model.descriptor(2));
				Eigen::Vector3d line_dir(model.descriptor(3), model.descriptor(4), model.descriptor(5));
				
				// Point coordinates (row-major storage)
				Eigen::Vector3d point(pointPtr[0], pointPtr[1], pointPtr[2]);
				
				// FOR PARALLEL LINES (direction = (0, 0, 1)):
				// The line point is (x0, y0, t0) where (x0, y0) is the spatial center
				// The direction is always (0, 0, 1) - parallel to time axis
				// For a point (x, y, t), the spatial distance is simply sqrt((x-x0)^2 + (y-y0)^2)
				
				// Calculate spatial distance from point to line's spatial center
				double dx = point(0) - line_point(0);
				double dy = point(1) - line_point(1);
				double spatial_distance = std::sqrt(dx*dx + dy*dy);
				double spatial_distance_sq = spatial_distance * spatial_distance;
				
				// SPATIAL CONSTRAINT: Tight clustering for sparse lines
				// With smaller neighborhood radius (5 pixels), use tighter thresholds
				// This prevents distant points from being grouped into the same line
				const double soft_threshold = 10.0;  // Soft threshold: points within this have low penalty (2x neighborhood radius)
				const double hard_threshold = 20.0;  // Hard threshold: points beyond this are rejected (4x neighborhood radius)
				
				// Reject very distant points completely
				if (spatial_distance > hard_threshold)
				{
					return 1e10;
				}
				
				// PRIMARY: Spatial alignment in (x, y) - events should be at same spatial location
				// For lines parallel to time axis, spatial distance is the only metric
				// Use soft penalty: low penalty within soft_threshold, increasing penalty beyond
				double spatial_residual;
				if (spatial_distance <= soft_threshold)
				{
					// Within soft threshold: use squared distance (normal behavior)
					spatial_residual = spatial_distance_sq;
				}
				else
				{
					// Beyond soft threshold: add increasing penalty
					// This allows some spread for noisy lines but penalizes distant points
					double excess = spatial_distance - soft_threshold;
					spatial_residual = soft_threshold * soft_threshold + 10.0 * excess * excess;
				}
				
				// TEMPORAL CONSTRAINT: Time must increase along line direction
				// For parallel lines, time should be >= line_point(2) (earliest time)
				double temporal_penalty = 0.0;
				if (point(2) < line_point(2) - 0.1)  // Going backwards in time (with small tolerance)
				{
					// Heavy penalty for going backwards in time
					double time_diff = line_point(2) - point(2);
					temporal_penalty = 10.0 * time_diff * spatial_distance;
				}
				// For forward time (point(2) >= line_point(2)), no penalty regardless of time difference
				// This allows points that are 5000s apart to still be in the same line
				// as long as time is increasing
				
				// Combined residual: spatial alignment + temporal ordering
				// The hard constraint (max_spatial_distance) prevents distant points from being assigned to the same line
				return spatial_residual + temporal_penalty;
			}
		};

		// Default 3D Line Temporal Estimator type
		typedef Line3DTemporalEstimator<solver::Line3DTemporalSolver, solver::Line3DTemporalSolver> Default3DLineTemporalEstimator;
	}
}

int findLines3D_(
	std::vector<double>& input_points,
	std::vector<double>& weights,
	std::vector<size_t>& labeling,
	std::vector<double>& lines,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library (once globally)
	initGoogleLoggingOnce();
	
	const size_t num_tents = input_points.size() / 3; // 3D points: x, y, z
	cv::Mat points(num_tents, 3, CV_64F, &input_points[0]);
	
	// Initialize the neighborhood used in Graph-cut RANSAC
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points,
		neighborhood_ball_radius);

	// Initialize the samplers
	constexpr size_t kSampleSize = 2; // 2 points for a 3D line
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Uniform sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // PROSAC sampler
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the points to be ordered by quality.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else if (sampler_id == 2) // NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>(&points, &neighborhood));
	else if (sampler_id == 3) // Z-Aligned sampler (for lines parallel to Z-axis/time)
	{
		// Sample points that are close in X,Y but far apart in Z (time)
		// OPTIMIZED: Use smaller spatial threshold and limit search for speed
		double spatial_thresh = threshold * 0.5;  // Half the inlier threshold for spatial proximity
		double min_z_sep = threshold * 2.0;      // Minimum Z separation (time difference)
		// Note: Z-Aligned sampler is being used (no logging needed)
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::ZAlignedSampler(&points, spatial_thresh, min_z_sep));
	}
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform), 1 (PROSAC), 2 (NAPSAC), 3 (Z-Aligned)\n",
			sampler_id);
		return 0;
	}

	// The local optimization sampler
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X with 3D line estimator
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph,
		gcransac::estimator::Default3DLineEstimator,
		AbstractSampler,
		gcransac::sampler::UniformSampler>
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	settings.minimum_number_of_inliers = minimum_point_number;
	settings.inlier_outlier_threshold = threshold;
	settings.setConfidence(confidence);
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	settings.spatial_coherence_weight = spatial_coherence_weight;
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// STRUCTURAL OPTIMIZATIONS FOR PROPOSAL ENGINE (GC-RANSAC):
	// These changes make the pipeline more efficient by reducing expensive operations
	// 1. Single local optimization (default 50 -> 1) - single LO is often sufficient
	settings.proposal_engine_settings.max_local_optimization_number = 1;
	// 2. Disable final iterated least squares (saves time, minimal quality impact)
	settings.proposal_engine_settings.do_final_iterated_least_squares = false;
	// 3. Single graph-cut iteration (default 10 -> 1) - single iteration often sufficient
	settings.proposal_engine_settings.max_graph_cut_number = 1;
	// 4. Allow LO earlier (default varies -> 5) - start optimizing sooner
	settings.proposal_engine_settings.min_iteration_number_before_lo = 5;
	// 5. Enable inlier limit to speed up local optimization
	settings.proposal_engine_settings.use_inlier_limit = true;
	// 6. Single least squares iteration (default 10 -> 1) - single iteration often sufficient
	settings.proposal_engine_settings.max_least_squares_iterations = 1;
	// 7. Lower min iterations (default 20 -> 5) - converge faster
	settings.proposal_engine_settings.min_iteration_number = 5;
	// Set max_proposal_number_without_change - balance between finding models and stopping early
	// OPTIMIZED: Reduced to 300 - early termination at 24-25 models will stop before this limit
	settings.max_proposal_number_without_change = 300;  // Reduced from 500 - early termination handles stopping
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;
	progressive_x.setScoringExponent(scoring_exponent);
	// Enable logging to see candidate proposals
	progressive_x.log(do_logging);

	progressive_x.run(points,
		neighborhood,
		*main_sampler.get(),
		local_optimization_sampler);
	
	// Get statistics for timing breakdown
	const auto &stats = progressive_x.getStatistics();
	
	// Always print proposal summary to stdout (stderr is redirected in Python script)
		// Print proposal summary only if logging is enabled
		if (do_logging) {
			fprintf(stdout, "\n=== Progressive-X Candidate Proposals ===\n");
			fprintf(stdout, "Total candidate proposals tested: %zu\n", stats.total_proposals_tested);
			fprintf(stdout, "Total proposals accepted: %zu\n", stats.total_proposals_accepted);
			fprintf(stdout, "Acceptance rate: %.1f%%\n", stats.total_proposals_tested > 0 ? 100.0 * stats.total_proposals_accepted / stats.total_proposals_tested : 0.0);
			fprintf(stdout, "Final number of models: %zu\n", stats.iteration_statistics.size());
			fprintf(stdout, "==========================================\n\n");
			fflush(stdout);
		}
	
	// Print timing breakdown if logging is enabled
	if (do_logging) {
		printf("\n=== Progressive-X Timing Breakdown ===\n");
		printf("Total processing time: %.3f seconds\n", stats.processing_time);
		if (stats.processing_time > 0) {
			printf("  Proposal engine (GC-RANSAC): %.3f seconds (%.1f%%)\n", 
				stats.total_time_of_proposal_engine,
				100.0 * stats.total_time_of_proposal_engine / stats.processing_time);
			printf("  PEARL optimization: %.3f seconds (%.1f%%)\n",
				stats.total_time_of_optimization,
				100.0 * stats.total_time_of_optimization / stats.processing_time);
			printf("  Model validation: %.3f seconds (%.1f%%)\n",
				stats.total_time_of_model_validation,
				100.0 * stats.total_time_of_model_validation / stats.processing_time);
			printf("  Compound model update: %.3f seconds (%.1f%%)\n",
				stats.total_time_of_compound_model_calculation,
				100.0 * stats.total_time_of_compound_model_calculation / stats.processing_time);
		}
		printf("Number of iterations: %zu\n", stats.iteration_statistics.size());
		printf("========================================\n\n");
	}
	
	// The obtained labeling
	labeling = stats.labeling;
	
	lines.reserve(6 * progressive_x.getModelNumber()); // 6 params per 3D line
	
	// Saving the 3D line parameters: [p₀x, p₀y, p₀z, dx, dy, dz] for each line
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		lines.emplace_back(model.descriptor(0)); // p₀x
		lines.emplace_back(model.descriptor(1)); // p₀y
		lines.emplace_back(model.descriptor(2)); // p₀z
		lines.emplace_back(model.descriptor(3)); // dx
		lines.emplace_back(model.descriptor(4)); // dy
		lines.emplace_back(model.descriptor(5)); // dz
	}
	
	return progressive_x.getModelNumber();
}

int findLines3DTemporal_(
	std::vector<double>& input_points,
	std::vector<double>& weights,
	std::vector<size_t>& labeling,
	std::vector<double>& lines,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library (once globally)
	initGoogleLoggingOnce();
	
	const size_t num_tents = input_points.size() / 3; // 3D points: x, y, t
	cv::Mat points(num_tents, 3, CV_64F, &input_points[0]);
	
	// For neighborhood graph and NAPSAC sampler: use only spatial dimensions (x, y), ignore time
	// This ensures that points at the same spatial location but different times are considered neighbors
	cv::Mat points_spatial(num_tents, 2, CV_64F);
	double x_min = std::numeric_limits<double>::max(), x_max = std::numeric_limits<double>::lowest();
	double y_min = std::numeric_limits<double>::max(), y_max = std::numeric_limits<double>::lowest();
	
	for (size_t i = 0; i < num_tents; ++i)
	{
		double x = points.at<double>(i, 0);
		double y = points.at<double>(i, 1);
		points_spatial.at<double>(i, 0) = x; // x
		points_spatial.at<double>(i, 1) = y; // y
		// time dimension (z) is ignored for neighborhood calculation
		
		if (x < x_min) x_min = x;
		if (x > x_max) x_max = x;
		if (y < y_min) y_min = y;
		if (y > y_max) y_max = y;
	}
	
	// Calculate spatial range for grid-based neighborhood
	double x_range = x_max - x_min;
	double y_range = y_max - y_min;
	
	// Use grid-based neighborhood: divides space into cells
	// Points in the same cell (and adjacent cells) are neighbors
	// This is ideal for clustering spatially close points regardless of time
	// Use a finer grid (smaller cells) to ensure only very close points are neighbors
	// Cell size = half of neighborhood radius for tighter spatial clustering
	// This prevents distant points from being neighbors
	double cell_size = neighborhood_ball_radius * 0.5; // 2.5 pixels for 5-pixel neighborhood radius
	
	// Calculate number of cells needed to cover the spatial range with the desired cell size
	// Add 1 to ensure we cover the full range
	size_t grid_cells_x = static_cast<size_t>(std::ceil(x_range / cell_size)) + 1;
	size_t grid_cells_y = static_cast<size_t>(std::ceil(y_range / cell_size)) + 1;
	
	// Use the larger of the two to create a square grid (ensures coverage)
	size_t grid_cells = std::max(grid_cells_x, grid_cells_y);
	
	// Use the desired cell size directly - don't recalculate to evenly divide
	// This ensures cells are small enough for tight spatial clustering
	double cell_size_x = cell_size;
	double cell_size_y = cell_size;
	
	// Initialize grid-based neighborhood graph (better for tight spatial clusters)
	// GridNeighborhoodGraph<2> means 2D spatial grid (x, y only)
	gcransac::neighborhood::GridNeighborhoodGraph<2> neighborhood(&points_spatial,
		{ cell_size_x, cell_size_y }, // Cell sizes for x and y
		grid_cells); // Number of cells per dimension

	// Initialize the samplers
	constexpr size_t kSampleSize = 2; // 2 points for a 3D line
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Uniform sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // PROSAC sampler
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the points to be ordered by quality.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else if (sampler_id == 2) // NAPSAC sampler
		// NAPSAC uses the neighborhood graph to sample spatially close points
		// With grid-based neighborhood, it samples points from the same or adjacent cells
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::NapsacSampler<gcransac::neighborhood::GridNeighborhoodGraph<2>>(&points_spatial, &neighborhood));
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %zu. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	// The local optimization sampler
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X with 3D line temporal estimator
	// Use GridNeighborhoodGraph<2> for spatial-only (x,y) neighborhood, ignoring time
	progx::ProgressiveX<gcransac::neighborhood::GridNeighborhoodGraph<2>,
		gcransac::estimator::Default3DLineTemporalEstimator,
		AbstractSampler,
		gcransac::sampler::UniformSampler>
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	settings.minimum_number_of_inliers = minimum_point_number;
	settings.inlier_outlier_threshold = threshold;
	settings.setConfidence(confidence);
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	settings.spatial_coherence_weight = spatial_coherence_weight;
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// SPEED OPTIMIZATION: Reduce local optimization iterations for 3D temporal lines
	// Local optimization is expensive - reduce from default 50 to 10 for faster execution
	settings.proposal_engine_settings.max_local_optimization_number = 10;
	// Set max_proposal_number_without_change - use 200 to allow finding more models
	// This allows the algorithm to try more proposals before giving up
	settings.max_proposal_number_without_change = 200;
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;
	progressive_x.setScoringExponent(scoring_exponent);

	progressive_x.run(points,
		neighborhood,
		*main_sampler.get(),
		local_optimization_sampler);
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;

	lines.reserve(6 * progressive_x.getModelNumber()); // 6 params per 3D line
	
	// Saving the 3D line parameters: [p₀x, p₀y, p₀z, dx, dy, dz] for each line
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		lines.emplace_back(model.descriptor(0)); // p₀x
		lines.emplace_back(model.descriptor(1)); // p₀y
		lines.emplace_back(model.descriptor(2)); // p₀z
		lines.emplace_back(model.descriptor(3)); // dx
		lines.emplace_back(model.descriptor(4)); // dy
		lines.emplace_back(model.descriptor(5)); // dz
	}
	
	return progressive_x.getModelNumber();
}

// Dual 3D Line Detection: Detects both dense and sparse lines with mutual exclusivity
int findLines3DDual_(
	std::vector<double>& input_points,
	std::vector<double>& weights,
	std::vector<size_t>& labeling,
	std::vector<double>& lines,
	std::vector<int>& line_types,  // 0 = dense, 1 = sparse
	const double &spatial_coherence_weight,
	const double &threshold_dense,  // Threshold for dense lines
	const double &threshold_sparse, // Threshold for sparse lines (typically larger)
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number_dense,  // Min points for dense lines
	const size_t &minimum_point_number_sparse, // Min points for sparse lines
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library (once globally)
	initGoogleLoggingOnce();
	
	const size_t num_tents = input_points.size() / 3; // 3D points: x, y, z
	cv::Mat points(num_tents, 3, CV_64F, &input_points[0]);
	
	// Initialize the neighborhood used in Graph-cut RANSAC
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points,
		neighborhood_ball_radius);

	// Initialize the samplers
	constexpr size_t kSampleSize = 2; // 2 points for a 3D line
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Uniform sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // PROSAC sampler
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the points to be ordered by quality.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else if (sampler_id == 2) // NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>(&points, &neighborhood));
	else if (sampler_id == 3) // Z-Aligned sampler
	{
		double spatial_thresh = threshold_dense * 0.5;
		double min_z_sep = threshold_dense * 2.0;
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::ZAlignedSampler(&points, spatial_thresh, min_z_sep));
	}
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d\n", sampler_id);
		return 0;
	}

	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Step 1: Detect DENSE lines using regular 3D line estimator
	if (do_logging)
		printf("Detecting DENSE lines (threshold=%.6f, min_points=%zu)...\n", threshold_dense, minimum_point_number_dense);

	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph,
		gcransac::estimator::Default3DLineEstimator,
		AbstractSampler,
		gcransac::sampler::UniformSampler>
		progressive_x_dense(nullptr);

	progx::MultiModelSettings &settings_dense = progressive_x_dense.getMutableSettings();
	settings_dense.minimum_number_of_inliers = minimum_point_number_dense;
	settings_dense.inlier_outlier_threshold = threshold_dense;
	settings_dense.setConfidence(confidence);
	settings_dense.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	settings_dense.spatial_coherence_weight = spatial_coherence_weight;
	settings_dense.proposal_engine_settings.max_iteration_number = max_iters;
	settings_dense.proposal_engine_settings.max_local_optimization_number = 1;
	settings_dense.proposal_engine_settings.do_final_iterated_least_squares = false;
	settings_dense.proposal_engine_settings.max_graph_cut_number = 1;
	settings_dense.proposal_engine_settings.min_iteration_number_before_lo = 5;
	settings_dense.proposal_engine_settings.use_inlier_limit = true;
	settings_dense.proposal_engine_settings.max_least_squares_iterations = 1;
	settings_dense.proposal_engine_settings.min_iteration_number = 5;
	settings_dense.max_proposal_number_without_change = 300;
	if (maximum_model_number > 0)
		settings_dense.maximum_model_number = maximum_model_number;
	progressive_x_dense.setScoringExponent(scoring_exponent);
	progressive_x_dense.log(do_logging);

	progressive_x_dense.run(points, neighborhood, *main_sampler, local_optimization_sampler);

	// Get dense line results
	const auto &stats_dense = progressive_x_dense.getStatistics();
	std::vector<size_t> labeling_dense = stats_dense.labeling;
	size_t num_dense_lines = progressive_x_dense.getModelNumber();

	if (do_logging)
		printf("Found %zu DENSE lines\n", num_dense_lines);

	// Step 2: Detect SPARSE lines using sparse 3D line estimator
	// Only consider points NOT assigned to dense lines
	// Note: Progressive-X labeling: 0 = first model (or outlier if 0-indexed), 1, 2, 3... = other models
	// We need to check: if label >= num_dense_lines, it's an outlier (or if 0-indexed, label < num_dense_lines means assigned)
	// Actually, labels are 0-indexed: 0, 1, 2, ..., num_dense_lines-1 are the dense line labels
	// So any point with label >= num_dense_lines is an outlier (not assigned to any dense line)
	cv::Mat points_sparse;
	std::vector<size_t> sparse_point_indices;
	for (size_t i = 0; i < num_tents; ++i)
	{
		// Check if point is NOT assigned to any dense line
		// If labeling is 0-indexed: labels 0..(num_dense_lines-1) are dense lines
		// If num_dense_lines == 0, all points are outliers
		// If num_dense_lines > 0, check if label >= num_dense_lines (outlier) OR if we're using 1-indexed, check differently
		bool is_outlier = true;
		if (num_dense_lines > 0)
		{
			// Labels are 0-indexed: 0, 1, 2, ..., num_dense_lines-1 are dense line labels
			// So if label < num_dense_lines, it's assigned to a dense line
			if (labeling_dense[i] < num_dense_lines)
			{
				is_outlier = false;
			}
		}
		
		if (is_outlier)
		{
			sparse_point_indices.push_back(i);
		}
	}
	
	if (do_logging)
		printf("Points available for sparse detection: %zu out of %zu (%.1f%%)\n",
			sparse_point_indices.size(), num_tents, 100.0 * sparse_point_indices.size() / num_tents);

	if (sparse_point_indices.empty())
	{
		if (do_logging)
			printf("No points available for SPARSE line detection (all assigned to dense lines)\n");
		// Return only dense lines
		labeling = labeling_dense;
		lines.reserve(6 * num_dense_lines);
		line_types.reserve(num_dense_lines);
		for (size_t model_idx = 0; model_idx < num_dense_lines; ++model_idx)
		{
			const auto &model = progressive_x_dense.getModels()[model_idx];
			lines.emplace_back(model.descriptor(0));
			lines.emplace_back(model.descriptor(1));
			lines.emplace_back(model.descriptor(2));
			lines.emplace_back(model.descriptor(3));
			lines.emplace_back(model.descriptor(4));
			lines.emplace_back(model.descriptor(5));
			line_types.push_back(0); // Dense line
		}
		return num_dense_lines;
	}

	// Create sparse points matrix
	points_sparse = cv::Mat(static_cast<int>(sparse_point_indices.size()), 3, CV_64F);
	for (size_t i = 0; i < sparse_point_indices.size(); ++i)
	{
		const double* src = points.ptr<double>(sparse_point_indices[i]);
		double* dst = points_sparse.ptr<double>(i);
		dst[0] = src[0];
		dst[1] = src[1];
		dst[2] = src[2];
	}

	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood_sparse(&points_sparse,
		neighborhood_ball_radius);

	// Create sampler for sparse points
	std::unique_ptr<AbstractSampler> main_sampler_sparse;
	if (sampler_id == 0)
		main_sampler_sparse = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points_sparse));
	else if (sampler_id == 1)
		main_sampler_sparse = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points_sparse, kSampleSize));
	else if (sampler_id == 2)
		main_sampler_sparse = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>(&points_sparse, &neighborhood_sparse));
	else if (sampler_id == 3)
	{
		double spatial_thresh = threshold_sparse * 0.5;
		double min_z_sep = threshold_sparse * 2.0;
		main_sampler_sparse = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::ZAlignedSampler(&points_sparse, spatial_thresh, min_z_sep));
	}
	else
		main_sampler_sparse = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points_sparse));

	gcransac::sampler::UniformSampler local_optimization_sampler_sparse(&points_sparse);

	if (do_logging)
		printf("Detecting SPARSE lines from %zu unassigned points (threshold=%.6f, min_points=%zu, conf=%.3f)...\n",
			sparse_point_indices.size(), threshold_sparse, minimum_point_number_sparse, confidence);

	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph,
		gcransac::estimator::Default3DLineSparseEstimator,
		AbstractSampler,
		gcransac::sampler::UniformSampler>
		progressive_x_sparse(nullptr);

	progx::MultiModelSettings &settings_sparse = progressive_x_sparse.getMutableSettings();
	settings_sparse.minimum_number_of_inliers = minimum_point_number_sparse;
	settings_sparse.inlier_outlier_threshold = threshold_sparse;
	// Use lower confidence for sparse lines to be more thorough
	settings_sparse.setConfidence(confidence * 0.5); // More thorough search for sparse lines
	settings_sparse.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	settings_sparse.spatial_coherence_weight = spatial_coherence_weight;
	// More iterations for sparse lines since they're harder to find
	settings_sparse.proposal_engine_settings.max_iteration_number = max_iters * 2; // Double iterations for sparse
	settings_sparse.proposal_engine_settings.max_local_optimization_number = 1;
	settings_sparse.proposal_engine_settings.do_final_iterated_least_squares = false;
	settings_sparse.proposal_engine_settings.max_graph_cut_number = 1;
	settings_sparse.proposal_engine_settings.min_iteration_number_before_lo = 5;
	settings_sparse.proposal_engine_settings.use_inlier_limit = true;
	settings_sparse.proposal_engine_settings.max_least_squares_iterations = 1;
	settings_sparse.proposal_engine_settings.min_iteration_number = 5;
	settings_sparse.max_proposal_number_without_change = 300;
	if (maximum_model_number > 0)
		settings_sparse.maximum_model_number = maximum_model_number;
	progressive_x_sparse.setScoringExponent(scoring_exponent);
	progressive_x_sparse.log(do_logging);

	progressive_x_sparse.run(points_sparse, neighborhood_sparse, *main_sampler_sparse, local_optimization_sampler_sparse);

	// Get sparse line results (relative to sparse_point_indices)
	const auto &stats_sparse = progressive_x_sparse.getStatistics();
	std::vector<size_t> labeling_sparse_relative = stats_sparse.labeling;
	size_t num_sparse_lines = progressive_x_sparse.getModelNumber();

	if (do_logging)
		printf("Found %zu SPARSE lines from %zu unassigned points\n", num_sparse_lines, sparse_point_indices.size());

	// Step 3: Merge results with mutual exclusivity
	// Labeling scheme: 0 = outliers, 1, 2, 3, ... = lines
	// Initialize all as outliers (label 0)
	labeling.resize(num_tents, 0);
	
	// Assign dense line labels (offset by 1: lines are 1, 2, 3, ..., num_dense_lines)
	// Progressive-X dense detection uses 0-indexed: 0, 1, 2, ..., num_dense_lines-1
	// We map these to: 1, 2, 3, ..., num_dense_lines
	for (size_t i = 0; i < num_tents; ++i)
	{
		// If point is assigned to a dense line (label < num_dense_lines), map to 1-indexed
		if (labeling_dense[i] < num_dense_lines)
		{
			labeling[i] = labeling_dense[i] + 1; // Map 0-indexed dense label to 1-indexed (1, 2, 3, ...)
		}
	}

	// Assign sparse line labels (offset by num_dense_lines + 1)
	// Sparse labels in relative space: 0, 1, 2, ... (0-indexed from sparse detection)
	// Map to global: num_dense_lines + 1, num_dense_lines + 2, num_dense_lines + 3, ...
	for (size_t sparse_idx = 0; sparse_idx < sparse_point_indices.size(); ++sparse_idx)
	{
		// If point is assigned to a sparse line (label < num_sparse_lines in relative space)
		if (labeling_sparse_relative[sparse_idx] < num_sparse_lines)
		{
			size_t original_idx = sparse_point_indices[sparse_idx];
			// Map sparse label (0, 1, 2...) to global label (num_dense_lines + 1, num_dense_lines + 2, ...)
			// Note: +1 because dense lines already use 1, 2, ..., num_dense_lines
			labeling[original_idx] = num_dense_lines + labeling_sparse_relative[sparse_idx] + 1;
		}
	}

	// Combine line parameters
	size_t total_lines = num_dense_lines + num_sparse_lines;
	lines.reserve(6 * total_lines);
	line_types.reserve(total_lines);

	// Add dense lines
	for (size_t model_idx = 0; model_idx < num_dense_lines; ++model_idx)
	{
		const auto &model = progressive_x_dense.getModels()[model_idx];
		lines.emplace_back(model.descriptor(0));
		lines.emplace_back(model.descriptor(1));
		lines.emplace_back(model.descriptor(2));
		lines.emplace_back(model.descriptor(3));
		lines.emplace_back(model.descriptor(4));
		lines.emplace_back(model.descriptor(5));
		line_types.push_back(0); // Dense line
	}

	// Add sparse lines
	for (size_t model_idx = 0; model_idx < num_sparse_lines; ++model_idx)
	{
		const auto &model = progressive_x_sparse.getModels()[model_idx];
		lines.emplace_back(model.descriptor(0));
		lines.emplace_back(model.descriptor(1));
		lines.emplace_back(model.descriptor(2));
		lines.emplace_back(model.descriptor(3));
		lines.emplace_back(model.descriptor(4));
		lines.emplace_back(model.descriptor(5));
		line_types.push_back(1); // Sparse line
	}

	if (do_logging)
		printf("Total: %zu lines (%zu dense, %zu sparse)\n", total_lines, num_dense_lines, num_sparse_lines);

	return total_lines;
}