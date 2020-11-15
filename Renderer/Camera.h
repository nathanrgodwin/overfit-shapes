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
	float3 side_internal_;

public:

	Camera() :
		pos_(make_float3(1,1,1)),
		fov_(120.0f),
		max_dist_(10.0f)
	{
		dir_ = normalize(-pos_);
		setSide({ 0, 1, 0 });
		fov_scale_ = 1.0f / std::tan((fov_ / 180.0f * float(M_PI)) / 2.0f);
	}

	inline std::tuple<float, float, float>
	getPosition() const
	{
		return std::make_tuple(pos_.x, pos_.y, pos_.z);
	}
	inline void
	setPosition(std::tuple<float,float,float> pos)
	{
		pos_ = make_float3(std::get<0>(pos), std::get<1>(pos), std::get<2>(pos));
	}
	inline __device__ float3
	positionf() const
	{
		return pos_;
	}

	inline std::tuple<float, float, float>
	getDirection() const
	{
		return std::make_tuple(dir_.x, dir_.y, dir_.z);
	}
	inline void
	setDirection(std::tuple<float,float,float> dir)
	{
		dir_ = make_float3(std::get<0>(dir), std::get<1>(dir), std::get<2>(dir));
		side_ = normalize(cross(dir_, side_internal_));
		up_ = normalize(cross(side_, dir_));
	}
	inline __device__ float3
	directionf() const
	{
		return dir_;
	}

	inline std::tuple<float, float, float>
	getUp() const
	{
		return std::make_tuple(up_.x, up_.y, up_.z);
	}
	inline __device__ float3
	upf() const
	{
		return up_;
	}

	inline std::tuple<float, float, float>
	getSide() const
	{
		return std::make_tuple(side_.x, side_.y, side_.z);
	}
	inline void
	setSide(std::tuple<float,float,float> side)
	{
		side_internal_ = make_float3(std::get<0>(side), std::get<1>(side), std::get<2>(side));
		side_ = normalize(cross(dir_, side_internal_));
		up_ = normalize(cross(side_, dir_));
	}
	inline __device__ float3
	sidef() const
	{
		return side_;
	}

	inline float
	getFOV() const
	{
		return fov_;
	}
	inline __device__ float
	fovScale() const
	{
		return fov_scale_;
	}
	inline void
	setFOV(float fov)
	{
		assert(fov > 0);
		fov_ = fov;
		fov_scale_ = 1.0f / std::tan((fov_ / 180.0f * float(M_PI)) / 2.0f);
	}

	__device__ inline float
	getMaxDist() const
	{
		return max_dist_;
	}
	inline void
	setMaxDist(float max_dist)
	{
		assert(max_dist > 0);
		max_dist_ = max_dist;
	}

};
