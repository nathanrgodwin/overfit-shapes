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
			H(0),
			N(0),
			background_color(make_float3(0,0,0)),
			min_dist(1.8e-3) {}

		Camera cam;
		Light light;
		unsigned char* image;
		unsigned char* device_image;

		unsigned int H, N;
		float slope;
		float* weights, * biases;

		unsigned int width, height;

		unsigned int samples;
		float eps;
		float min_dist;
		float3 background_color;
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
	setResolution(unsigned int width, unsigned int height)
	{
		assert(width > 0 && height > 0);
		params_.width = width;
		params_.height = height;
		if (params_.device_image) gpuDelete(params_.device_image);
		params_.device_image = makeImage(params_.width, params_.height);
	}
	inline std::pair<unsigned int, unsigned int>
	getResolution() const
	{
		return std::make_pair(params_.width, params_.height);
	}

	inline float
	getMinDist() const
	{
		return params_.min_dist;
	}
	inline void
	setMinDist(float min_dist)
	{
		params_.min_dist = min_dist;
	}

	inline float
	getEps() const
	{
		return params_.eps;
	}
	inline void
	setEps(float eps)
	{
		assert(eps > 0);
		params_.eps = eps;
	}

	inline const Light&
	getLight() const
	{
		return params_.light;
	}
	inline void
	setLight(const Light& light)
	{
		params_.light = light;
	}

	inline const Camera&
	getCamera() const
	{
		return params_.cam;
	}
	inline void
	setCamera(const Camera& cam)
	{
		params_.cam = cam;
	}

	inline std::tuple<float,float,float>
	getBackgroundColor()
	{
		return std::make_tuple(params_.background_color.x,
			params_.background_color.y,
			params_.background_color.z);
	}

	inline void
	setBackgroundColor(float r, float g, float b)
	{
		params_.background_color = make_float3(r, g, b);
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
