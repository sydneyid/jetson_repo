#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "progressivex_python.h"

namespace py = pybind11;

py::tuple find6DPoses(
	py::array_t<double>  x1y1_,
	py::array_t<double>  x2y2z2_,
	py::array_t<double>  K_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int maximum_model_number) 
{
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=3");
	}
	if (NUM_TENTS < 3) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=3");
	}
	
	py::buffer_info buf1a = x2y2z2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 3) {
		throw std::invalid_argument("x2y2z2 should be an array with dims [n,3], n>=3");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and x2y2z2 should be the same size");
	}
	
	py::buffer_info buf1K = K_.request();
	size_t DIMK1 = buf1K.shape[0];
	size_t DIMK2 = buf1K.shape[1];

	if (DIMK1 != 3 || DIMK2 != 3) {
		throw std::invalid_argument("K should be an array with dims [3,3]");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

	double *ptr1a = (double *)buf1a.ptr;
	std::vector<double> x2y2z2;
	x2y2z2.assign(ptr1a, ptr1a + buf1a.size);

	double *ptr1K = (double *)buf1K.ptr;
	std::vector<double> K;
	K.assign(ptr1K, ptr1K + buf1K.size);
	
	std::vector<double> poses;
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = find6DPoses_(
		x1y1,
		x2y2z2,
		K,
		labeling,
		poses,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number);

	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> poses_ = py::array_t<double>(
		{ static_cast<py::ssize_t>(num_models) * 3, static_cast<py::ssize_t>(4) }
	);

	py::buffer_info buf2 = poses_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 12 * num_models; i++)
		ptr2[i] = poses[i];
	return py::make_tuple(poses_, labeling_);
}

py::tuple findHomographies(
	py::array_t<double>  corrs_,
	size_t w1, size_t h1,
	size_t w2, size_t h2,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int maximum_model_number,
	int sampler_id,
	double scoring_exponent,
	bool do_logging) {
		
	py::buffer_info buf1 = corrs_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("corrs should be an array with dims [n,4], n>=4");
	}
	if (NUM_TENTS < 4) {
		throw std::invalid_argument("corrs should be an array with dims [n,4], n>=4");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> corrs;
	corrs.assign(ptr1, ptr1 + buf1.size);

	std::vector<double> homographies;
	
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = findHomographies_(
		corrs,
		labeling,
		homographies,
		w1, h1,
		w2, h2,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number,
		sampler_id,
		scoring_exponent,
		do_logging);
		
	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> homographies_ = py::array_t<double>(
		{ static_cast<py::ssize_t>(num_models) * 3, static_cast<py::ssize_t>(3) }
	);
	py::buffer_info buf2 = homographies_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 9 * num_models; i++)
		ptr2[i] = homographies[i];
	return py::make_tuple(homographies_, labeling_);
}

py::tuple findVanishingPoints(
	py::array_t<double>  lines_,
	py::array_t<double>  weights_,
	size_t w, size_t h,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int maximum_model_number,
	int sampler_id,
	double scoring_exponent,
	bool do_logging) {
		
	py::buffer_info buf1 = lines_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("lines should be an array with dims [n,4], n>=2");
	}
	if (NUM_TENTS < 2) {
		throw std::invalid_argument("lines should be an array with dims [n,4], n>=2");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> corrs;
	corrs.assign(ptr1, ptr1 + buf1.size);

	// Parsing the weights
	py::buffer_info bufW = weights_.request();
	DIM = bufW.ndim;

	std::vector<double> weights;
	if (DIM > 0)
	{
		double *ptrW = (double *)bufW.ptr;
		weights.assign(ptrW, ptrW + bufW.size);
	}

	std::vector<double> vanishingPoints;	
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = findVanishingPoints_(
		corrs,
		weights,
		labeling,
		vanishingPoints,
		w, h,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number,
		sampler_id,
		scoring_exponent,
		do_logging);
		
	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> vanishingPoints_ = py::array_t<double>(
		{ static_cast<py::ssize_t>(num_models), static_cast<py::ssize_t>(3) }
	);
	py::buffer_info buf2 = vanishingPoints_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 3 * num_models; i++)
		ptr2[i] = vanishingPoints[i];
	return py::make_tuple(vanishingPoints_, labeling_);
}

py::tuple findLines(
	py::array_t<double> points_,
	py::array_t<double> weights_,
	size_t w, size_t h,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int maximum_model_number,
	int sampler_id,
	double scoring_exponent,
	bool do_logging) 
{
	py::buffer_info buf1 = points_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) 
		throw std::invalid_argument("Points should be an array with dims [n,3], n>=2");
	if (NUM_TENTS < 2) 
		throw std::invalid_argument("Points should be an array with dims [n,3], n>=2");

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> corrs;
	corrs.assign(ptr1, ptr1 + buf1.size);

	// Parsing the weights
	py::buffer_info bufW = weights_.request();
	DIM = bufW.ndim;

	std::vector<double> weights;
	if (DIM > 0)
	{
		double *ptrW = (double *)bufW.ptr;
		weights.assign(ptrW, ptrW + bufW.size);
	}

	std::vector<double> lines;	
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = findLines_(
		corrs,
		weights,
		labeling,
		lines,
		w, h,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number,
		sampler_id,
		scoring_exponent,
		do_logging);
		
	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> lines_ = py::array_t<double>(
		{ static_cast<py::ssize_t>(num_models), static_cast<py::ssize_t>(3) }
	);
	py::buffer_info buf2 = lines_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 3 * num_models; i++)
		ptr2[i] = lines[i];
	return py::make_tuple(lines_, labeling_);
}

py::tuple findLines3D(
	py::array_t<double> points_,
	py::array_t<double> weights_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int maximum_model_number,
	int sampler_id,
	double scoring_exponent,
	bool do_logging) 
{
	py::buffer_info buf1 = points_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 3) 
		throw std::invalid_argument("Points should be an array with dims [n,3], n>=2");
	if (NUM_TENTS < 2) 
		throw std::invalid_argument("Points should be an array with dims [n,3], n>=2");

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> corrs;
	corrs.assign(ptr1, ptr1 + buf1.size);

	// Parsing the weights
	py::buffer_info bufW = weights_.request();
	DIM = bufW.ndim;

	std::vector<double> weights;
	if (DIM > 0)
	{
		double *ptrW = (double *)bufW.ptr;
		weights.assign(ptrW, ptrW + bufW.size);
	}

	std::vector<double> lines;	
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = findLines3D_(
		corrs,
		weights,
		labeling,
		lines,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number,
		sampler_id,
		scoring_exponent,
		do_logging);
		
	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> lines_ = py::array_t<double>(
		{ static_cast<py::ssize_t>(num_models), static_cast<py::ssize_t>(6) }
	);
	py::buffer_info buf2 = lines_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 6 * num_models; i++)
		ptr2[i] = lines[i];
	return py::make_tuple(lines_, labeling_);
}

py::tuple findLines3DTemporal(
	py::array_t<double> points_,
	py::array_t<double> weights_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int maximum_model_number,
	int sampler_id,
	double scoring_exponent,
	bool do_logging) 
{
	py::buffer_info buf1 = points_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 3) 
		throw std::invalid_argument("Points should be an array with dims [n,3], n>=2");
	if (NUM_TENTS < 2) 
		throw std::invalid_argument("Points should be an array with dims [n,3], n>=2");

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> corrs;
	corrs.assign(ptr1, ptr1 + buf1.size);

	py::buffer_info bufW = weights_.request();
	DIM = bufW.ndim;

	std::vector<double> weights;
	if (DIM > 0)
	{
		double *ptrW = (double *)bufW.ptr;
		weights.assign(ptrW, ptrW + bufW.size);
	}

	std::vector<double> lines;	
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = findLines3DTemporal_(
		corrs,
		weights,
		labeling,
		lines,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number,
		sampler_id,
		scoring_exponent,
		do_logging);

	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);

	py::array_t<double> lines_ = py::array_t<double>(
		{ static_cast<py::ssize_t>(num_models), static_cast<py::ssize_t>(6) }
	);
	py::buffer_info buf2 = lines_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 6 * num_models; i++)
		ptr2[i] = lines[i];
	return py::make_tuple(lines_, labeling_);
}

py::tuple findLines3DDual(
	py::array_t<double> points_,
	py::array_t<double> weights_,
	double threshold_dense,
	double threshold_sparse,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number_dense,
	int minimum_point_number_sparse,
	int maximum_model_number,
	int sampler_id,
	double scoring_exponent,
	bool do_logging) 
{
	py::buffer_info buf1 = points_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 3) 
		throw std::invalid_argument("Points should be an array with dims [n,3], n>=2");
	if (NUM_TENTS < 2) 
		throw std::invalid_argument("Points should be an array with dims [n,3], n>=2");

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> corrs;
	corrs.assign(ptr1, ptr1 + buf1.size);

	// Parsing the weights
	py::buffer_info bufW = weights_.request();
	DIM = bufW.ndim;

	std::vector<double> weights;
	if (DIM > 0)
	{
		double *ptrW = (double *)bufW.ptr;
		weights.assign(ptrW, ptrW + bufW.size);
	}

	std::vector<double> lines;	
	std::vector<size_t> labeling(NUM_TENTS);
	std::vector<int> line_types;

	int num_models = findLines3DDual_(
		corrs,
		weights,
		labeling,
		lines,
		line_types,
		spatial_coherence_weight,
		threshold_dense,
		threshold_sparse,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number_dense,
		minimum_point_number_sparse,
		maximum_model_number,
		sampler_id,
		scoring_exponent,
		do_logging);
		
	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> lines_ = py::array_t<double>(
		{ static_cast<py::ssize_t>(num_models), static_cast<py::ssize_t>(6) }
	);
	py::buffer_info buf2 = lines_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 6 * num_models; i++)
		ptr2[i] = lines[i];

	py::array_t<int> line_types_ = py::array_t<int>(num_models);
	py::buffer_info buf4 = line_types_.request();
	int *ptr4 = (int *)buf4.ptr;
	for (size_t i = 0; i < num_models; i++)
		ptr4[i] = line_types[i];

	return py::make_tuple(lines_, labeling_, line_types_);
}

py::tuple findTwoViewMotions(
	py::array_t<double>  corrs_,
	size_t w1, size_t h1,
	size_t w2, size_t h2,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int maximum_model_number,
	int sampler_id,
	double scoring_exponent,
	bool do_logging)
{		
	py::buffer_info buf1 = corrs_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("corrs should be an array with dims [n,4], n>=7");
	}
	if (NUM_TENTS < 7) {
		throw std::invalid_argument("corrs should be an array with dims [n,4], n>=7");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> corrs;
	corrs.assign(ptr1, ptr1 + buf1.size);

	std::vector<double> motions;	
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = findTwoViewMotions_(
		corrs,
		labeling,
		motions,
		w1, h1,
		w2, h2,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number,
		sampler_id,
		scoring_exponent,
		do_logging);
		
	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> motions_ = py::array(
		py::dtype::of<double>(),
		py::array::ShapeContainer{ static_cast<py::ssize_t>(num_models) * 3,
								static_cast<py::ssize_t>(3) }
	).cast<py::array_t<double>>();
	py::buffer_info buf2 = motions_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 9 * num_models; i++)
		ptr2[i] = motions[i];
	return py::make_tuple(motions_, labeling_);
}

PYBIND11_PLUGIN(pyprogressivex) {
                                                                             
    py::module m("pyprogressivex", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pyprogressivex
        .. autosummary::
           :toctree: _generate
           
           find6DPoses,
           findHomographies,
           findTwoViewMotions,
		   findPlanes,
		   findVanishingPoints,
    )doc");

	m.def("findHomographies", &findHomographies, R"doc(some doc)doc",
		py::arg("corrs"),
		py::arg("w1"),
		py::arg("h1") ,
		py::arg("w2"),
		py::arg("h2"),
		py::arg("threshold") = 4.0,
		py::arg("conf") = 0.5,
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("neighborhood_ball_radius") = 200.0,
		py::arg("maximum_tanimoto_similarity") = 0.4,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 10,
		py::arg("maximum_model_number") = -1,
		py::arg("sampler_id") = 3,
		py::arg("scoring_exponent") = 2,
		py::arg("do_logging") = false);

	m.def("findVanishingPoints", &findVanishingPoints, R"doc(some doc)doc",
		py::arg("lines"),
		py::arg("weights"),
		py::arg("w"),
		py::arg("h"),
		py::arg("threshold") = 4.0,
		py::arg("conf") = 0.5,
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("neighborhood_ball_radius") = 200.0,
		py::arg("maximum_tanimoto_similarity") = 0.4,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 10,
		py::arg("maximum_model_number") = -1,
		py::arg("sampler_id") = 3,
		py::arg("scoring_exponent") = 2,
		py::arg("do_logging") = false);

	m.def("findTwoViewMotions", &findTwoViewMotions, R"doc(some doc)doc",
		py::arg("corrs"),
		py::arg("w1"),
		py::arg("h1") ,
		py::arg("w2"),
		py::arg("h2"),
		py::arg("threshold") = 4.0,
		py::arg("conf") = 0.5,
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("neighborhood_ball_radius") = 200.0,
		py::arg("maximum_tanimoto_similarity") = 0.4,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 10,
		py::arg("maximum_model_number") = -1,
		py::arg("sampler_id") = 3,
		py::arg("scoring_exponent") = 3,		
		py::arg("do_logging") = false);
		
	m.def("find6DPoses", &find6DPoses, R"doc(some doc)doc",
		py::arg("x1y1"),
		py::arg("x2y2z2"),
		py::arg("K"),
		py::arg("threshold") = 4.0,
		py::arg("conf") = 0.90,
		py::arg("spatial_coherence_weight") = 0.1,
		py::arg("neighborhood_ball_radius") = 20.0,
		py::arg("maximum_tanimoto_similarity") = 0.9,
		py::arg("max_iters") = 400,
		py::arg("minimum_point_number") = 2 * 3,
		py::arg("maximum_model_number") = -1);

	m.def("findLines", &findLines, R"doc(some doc)doc",
		py::arg("points"),
		py::arg("weights"),
		py::arg("w"),
		py::arg("h"),
		py::arg("threshold") = 2.0,
		py::arg("conf") = 0.5,
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("neighborhood_ball_radius") = 200.0,
		py::arg("maximum_tanimoto_similarity") = 0.4,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 10,
		py::arg("maximum_model_number") = -1,
		py::arg("sampler_id") = 3,
		py::arg("scoring_exponent") = 2,
		py::arg("do_logging") = false);

	m.def("findLines3D", &findLines3D, R"doc(
		Find multiple 3D lines in a point cloud using Progressive-X.
		
		Args:
			points: numpy array of shape [n, 3] with 3D points (x, y, z)
			weights: numpy array of weights (can be empty)
			threshold: distance threshold for inlier detection (in 3D space)
			conf: confidence level (default: 0.99)
			spatial_coherence_weight: weight for spatial coherence term (default: 0.0)
			neighborhood_ball_radius: radius for neighborhood graph (default: adaptive)
			maximum_tanimoto_similarity: maximum similarity for model merging (default: 1.0)
			max_iters: maximum number of iterations (default: 1000)
			minimum_point_number: minimum points per line (default: 10)
			maximum_model_number: maximum number of lines (-1 for no limit, default: -1)
			sampler_id: sampler type (0=uniform, 1=PROSAC, 2=NAPSAC, default: 0)
			scoring_exponent: scoring exponent (default: 1.0)
			do_logging: enable logging (default: False)
		
		Returns:
			tuple: (lines, labeling)
			- lines: numpy array of shape [num_lines, 6] where each line is [p₀x, p₀y, p₀z, dx, dy, dz]
			  (point on line + direction vector)
			- labeling: numpy array of shape [n] where 0=outlier, 1,2,...=line indices
	)doc",
		py::arg("points"),
		py::arg("weights"),
		py::arg("threshold") = 0.1,
		py::arg("conf") = 0.99,
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("neighborhood_ball_radius") = 0.1,
		py::arg("maximum_tanimoto_similarity") = 1.0,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 10,
		py::arg("maximum_model_number") = -1,
		py::arg("sampler_id") = 0,
		py::arg("scoring_exponent") = 1.0,
		py::arg("do_logging") = false);

	m.def("findLines3DTemporal", &findLines3DTemporal, R"doc(
		Find multiple 3D lines in event data (space-time) using Progressive-X with temporal ordering constraint.
		This version enforces that points on a line must have monotonically increasing time.
		Designed for event-based vision data where lines represent moving edges/features.
		
		Args:
			points: numpy array of shape [n, 3] with 3D points (x, y, t) where t is time
			weights: numpy array of weights (can be empty)
			threshold: distance threshold for inlier detection (in 3D space)
			conf: confidence level (default: 0.99)
			spatial_coherence_weight: weight for spatial coherence term (default: 0.0)
			neighborhood_ball_radius: radius for neighborhood graph (default: adaptive)
			maximum_tanimoto_similarity: maximum similarity for model merging (default: 1.0)
			max_iters: maximum number of iterations (default: 1000)
			minimum_point_number: minimum points per line (default: 10)
			maximum_model_number: maximum number of lines (-1 for no limit, default: -1)
			sampler_id: sampler type (0=uniform, 1=PROSAC, 2=NAPSAC, 3=Z-Aligned, default: 3)
			scoring_exponent: scoring exponent (default: 0.0)
			do_logging: enable verbose logging (default: False)
		
		Returns:
			tuple: (lines, labeling) where lines is [num_lines, 6] and labeling is [n]
	)doc",
		py::arg("points"),
		py::arg("weights"),
		py::arg("threshold"),
		py::arg("conf") = 0.99,
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("neighborhood_ball_radius") = 200.0,
		py::arg("maximum_tanimoto_similarity") = 1.0,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 10,
		py::arg("maximum_model_number") = -1,
		py::arg("sampler_id") = 3,
		py::arg("scoring_exponent") = 0.0,
		py::arg("do_logging") = false);

	m.def("findLines3DDual", &findLines3DDual, R"doc(
		Find both dense and sparse 3D lines using Progressive-X with mutual exclusivity.
		First detects dense lines, then detects sparse lines from remaining unassigned points.
		Each point can only be assigned to one line type (dense or sparse).
		
		Args:
			points: numpy array of shape [n, 3] with 3D points (x, y, z)
			weights: numpy array of weights (can be empty)
			threshold_dense: distance threshold for dense line detection
			threshold_sparse: distance threshold for sparse line detection (typically larger)
			conf: confidence level (default: 0.05)
			spatial_coherence_weight: weight for spatial coherence term (default: 0.0)
			neighborhood_ball_radius: radius for neighborhood graph (default: adaptive)
			maximum_tanimoto_similarity: maximum similarity for model merging (default: 0.40)
			max_iters: maximum number of iterations (default: 4000)
			minimum_point_number_dense: minimum points per dense line (default: 10)
			minimum_point_number_sparse: minimum points per sparse line (default: 5)
			maximum_model_number: maximum number of lines (-1 for no limit, default: 20000)
			sampler_id: sampling type (0=uniform, 1=PROSAC, 2=NAPSAC, 3=Z-Aligned, default: 3)
			scoring_exponent: scoring exponent (default: 0.0)
			do_logging: enable verbose logging (default: False)
		
		Returns:
			tuple: (lines, labeling, line_types) where:
				- lines: [num_lines, 6] array of line parameters [p0x, p0y, p0z, dx, dy, dz]
				- labeling: [n] array of point labels (0-indexed, SIZE_MAX for outliers)
				- line_types: [num_lines] array of line types (0=dense, 1=sparse)
	)doc",
		py::arg("points"),
		py::arg("weights"),
		py::arg("threshold_dense"),
		py::arg("threshold_sparse"),
		py::arg("conf") = 0.05,
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("neighborhood_ball_radius") = 200.0,
		py::arg("maximum_tanimoto_similarity") = 0.40,
		py::arg("max_iters") = 4000,
		py::arg("minimum_point_number_dense") = 10,
		py::arg("minimum_point_number_sparse") = 5,
		py::arg("maximum_model_number") = 20000,
		py::arg("sampler_id") = 3,
		py::arg("scoring_exponent") = 0.0,
		py::arg("do_logging") = false);

  return m.ptr();
}
