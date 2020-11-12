#pragma once


#include <cassert>
#include "cutil_math.h"
#include "cuda_runtime.h"

#include "Camera.h"
#include "Light.h"

#include "ExportSemantics.h"

class EXPORT Renderer
{
public:
	struct EXPORT Parameters
	{
		Parameters() :
			weights(nullptr),
			biases(nullptr),
			width(640),
			height(480),
			image(nullptr),
			device_image(nullptr),
			eps(1e-5),
			slope(0.1),
			min_dist(1.8e-3) {}

		Camera cam;
		Light light;
		unsigned char* image;
		unsigned char* device_image;

		unsigned int num_layers, layer_size;
		float slope;
		float* weights, * biases;

		unsigned int width, height;

		unsigned int samples;
		float eps;
		float min_dist;
	};


	Renderer()
	{
		params_.device_image = makeImage(params_.width, params_.height);
	}

	~Renderer()
	{
		if (params_.device_image) gpuDelete(params_.device_image);
		if (params_.weights) gpuDelete(params_.weights);
		if (params_.biases) gpuDelete(params_.biases);
	}

	inline void
	resolution(unsigned int width, unsigned int height)
	{
		assert(width > 0 && height > 0);
		params_.width = width;
		params_.height = height;
		if (params_.device_image) gpuDelete(params_.device_image);
		params_.device_image = makeImage(params_.width, params_.height);
	}
	inline std::pair<unsigned int, unsigned int>
	resolution() const
	{
		return std::make_pair(params_.width, params_.height);
	}

	inline float
	minDist() const
	{
		return params_.min_dist;
	}
	inline void
	minDist(float min_dist)
	{
		params_.min_dist = min_dist;
	}

	inline float
	eps() const
	{
		return params_.eps;
	}
	inline void
	eps(float eps)
	{
		assert(eps > 0);
		params_.eps = eps;
	}

	inline const Light&
	light() const
	{
		return params_.light;
	}
	inline void
	light(const Light& light)
	{
		params_.light = light;
	}

	inline const Camera&
	camera() const
	{
		return params_.cam;
	}
	inline void
	camera(const Camera& cam)
	{
		params_.cam = cam;
	}


protected:


	void
	render();

	unsigned char*
	makeImage(unsigned int width, unsigned int height);

	void
	gpuDelete(unsigned char * image);

	void
	gpuDelete(float * data);

	void
	copyDataToGPU(float ** dst, const float * src, size_t numel);

	Parameters params_;


};
