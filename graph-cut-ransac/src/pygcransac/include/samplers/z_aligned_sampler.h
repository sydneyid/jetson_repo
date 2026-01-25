// Custom sampler for 3D lines parallel to Z-axis (time axis)
// Samples points that are spatially close (similar X,Y) but have different Z values
#pragma once

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "../uniform_random_generator.h"
#include "sampler.h"

namespace gcransac
{
	namespace sampler
	{
		class ZAlignedSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			std::unique_ptr<utils::UniformRandomGenerator<size_t>> random_generator;
			double spatial_threshold;  // Maximum spatial distance (X,Y) for sampling
			double min_z_separation;   // Minimum Z separation to ensure different time points

		public:
			explicit ZAlignedSampler(const cv::Mat * const container_, 
				double spatial_threshold_ = 5.0,  // Default: 5 pixels spatial tolerance
				double min_z_separation_ = 10.0)  // Default: 10 time units minimum separation
				: Sampler(container_),
				spatial_threshold(spatial_threshold_),
				min_z_separation(min_z_separation_)
			{
				initialized = initialize(container_);
			}

			~ZAlignedSampler() 
			{
				utils::UniformRandomGenerator<size_t> *generator_ptr = random_generator.release();
				delete generator_ptr;
			}

			const std::string getName() const { return "Z-Aligned Sampler"; }

			bool initialize(const cv::Mat * const container_)
			{
				random_generator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
				random_generator->resetGenerator(0,
					static_cast<size_t>(container_->rows));
				return true;
			}
			
			void update(
				const size_t* const subset_,
				const size_t& sample_size_,
				const size_t& iteration_number_,
				const double& inlier_ratio_) 
			{
				// No update needed for this sampler
			}

			void reset()
			{
				random_generator->resetGenerator(0,
					static_cast<size_t>(container->rows));
			}

			// Samples points that are spatially close but have different Z values
			// OPTIMIZED: Early exit and limit search to reduce overhead
			OLGA_INLINE bool sample(const std::vector<size_t> &pool_,
				size_t * const subset_,
				size_t sample_size_)
			{
				// For 3D lines, we need 2 points
				if (sample_size_ != 2 || pool_.size() < 2)
					return false;

				const cv::Mat &points = *container;
				const size_t max_search = std::min(pool_.size(), static_cast<size_t>(500)); // Limit search to first 500 points for speed

				// Sample first point randomly
				random_generator->resetGenerator(0, static_cast<size_t>(pool_.size() - 1));
				size_t first_idx = static_cast<size_t>(random_generator->getRandomNumber());
				subset_[0] = pool_[first_idx];
				
				const double *first_point = points.ptr<double>(subset_[0]);
				double first_x = first_point[0];
				double first_y = first_point[1];
				double first_z = first_point[2];

				// Find candidate second points that are:
				// 1. Spatially close (similar X,Y) - within spatial_threshold
				// 2. Have different Z (time) - at least min_z_separation apart
				// OPTIMIZED: Limit search and use squared distance to avoid sqrt
				std::vector<size_t> candidates;
				candidates.reserve(10); // Pre-allocate for common case
				const double spatial_thresh_sq = spatial_threshold * spatial_threshold;
				
				for (size_t i = 0; i < max_search; ++i)
				{
					if (i == first_idx) continue;
					
					const double *point = points.ptr<double>(pool_[i]);
					double dx = point[0] - first_x;
					double dy = point[1] - first_y;
					double dz = point[2] - first_z;
					
					// Use squared distance to avoid sqrt (faster)
					double spatial_dist_sq = dx * dx + dy * dy;
					double z_separation = std::abs(dz);
					
					// Accept if spatially close AND has sufficient Z separation
					if (spatial_dist_sq <= spatial_thresh_sq && z_separation >= min_z_separation)
					{
						candidates.push_back(pool_[i]);
						// Early exit if we found enough candidates (don't need to check all)
						if (candidates.size() >= 20)
							break;
					}
				}

				// If we found candidates, sample randomly from them
				if (!candidates.empty())
				{
					random_generator->resetGenerator(0, static_cast<size_t>(candidates.size() - 1));
					size_t candidate_idx = static_cast<size_t>(random_generator->getRandomNumber());
					subset_[1] = candidates[candidate_idx];
					return true;
				}

				// Fallback: if no candidates found, use uniform sampling
				// This ensures we don't fail completely
				size_t second_idx;
				do {
					random_generator->resetGenerator(0, static_cast<size_t>(pool_.size() - 1));
					second_idx = static_cast<size_t>(random_generator->getRandomNumber());
				} while (second_idx == first_idx);
				
				subset_[1] = pool_[second_idx];
				return true;
			}
		};
	}
}
