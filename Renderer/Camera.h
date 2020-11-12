#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cassert>
#include <cmath>
#include <tuple>
#include "cuda_runtime.h"

class Camera
{
private:
	float3 pos_;
	float3 dir_;
	float fov_scale_;
	float fov_;
	float max_dist_ = 10.0f;
	float3 up_;
	float3 side_;

public:

	inline std::tuple<float, float, float>
	position() const
	{
		return std::make_tuple(pos_.x, pos_.y, pos_.z);
	}
	inline void
	position(float x, float y, float z)
	{
		pos_ = make_float3(x, y, z);
	}
	inline __device__ float3
	positionf() const
	{
		return pos_;
	}

	inline std::tuple<float, float, float>
	direction() const
	{
		return std::make_tuple(dir_.x, dir_.y, dir_.z);
	}
	inline void
	direction(float x, float y, float z)
	{
		dir_ = make_float3(x, y, z);
		side_ = normalize(cross(dir_, make_float3(side_.x, side_.y, side_.z)));
		up_ = normalize(cross(side_, dir_));
	}
	inline __device__ float3
	directionf() const
	{
		return dir_;
	}

	inline std::tuple<float, float, float>
	up() const
	{
		return std::make_tuple(up_.x, up_.y, up_.z);
	}
	inline __device__ float3
	upf() const
	{
		return up_;
	}

	inline std::tuple<float, float, float>
	side() const
	{
		return std::make_tuple(side_.x, side_.y, side_.z);
	}
	inline void
	side(float x, float y, float z)
	{
		side_ = normalize(cross(dir_, make_float3(x, y, z)));
		up_ = normalize(cross(side_, dir_));
	}
	inline __device__ float3
	sidef() const
	{
		return side_;
	}

	inline float
	fov() const
	{
		return fov_;
	}
	inline __device__ float
	fovScale() const
	{
		return fov_scale_;
	}
	inline void
	fov(float fov)
	{
		assert(fov > 0);
		fov_ = fov;
		fov_scale_ = 1.0f / std::tan((fov_ / 180.0f * float(M_PI)) / 2.0f);
	}

	__device__ inline float
	maxDist() const
	{
		return max_dist_;
	}
	inline void
	maxDist(float max_dist)
	{
		assert(max_dist > 0);
		max_dist_ = max_dist;
	}

	Camera() :
		pos_(make_float3(2.0f, 0.0f, 0.5f)),
		fov_(128.0f),
		max_dist_(10.0f)
	{
		dir_ = normalize(-pos_);
		side(0, 1, 0);
		fov_scale_ = 1.0f / std::tan((fov_ / 180.0f * float(M_PI)) / 2.0f);
	}

};
